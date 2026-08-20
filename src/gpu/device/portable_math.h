//! \file portable_math.h
//! Deterministic fp32 math kernels for the transport engine.
//!
//! Vendor math libraries differ from each other at the ulp level (Metal's
//! sqrt/log/exp/sin/cos disagree with host libm in 26-82% of calls). Those
//! flips are individually harmless but resolve asymmetrically across a
//! Monte Carlo ensemble: paired-input experiments on Godiva showed the
//! device ensemble leaking 0.13% less than the identical host-compiled
//! engine (11 sigma). These kernels use only +,-,*,/ and bit operations —
//! no FMA, no vendor intrinsics — so every target (Metal, CUDA, host)
//! computes identical bits. Accuracy ~1-2 ulp, plenty below Monte Carlo
//! statistical resolution; determinism is the point.
//!
//! Domains are the transport engine's: p_logf on (0,1]; p_expf on
//! [-90, 90]; p_sincosf on [0, 2*pi]; p_sqrtf on [0, inf).

#pragma once

DEVICE_FN float p_bits_to_float(uint32_gpu b)
{
#if defined(GPU_TARGET_METAL)
  return as_type<float>(b);
#elif defined(GPU_TARGET_CUDA)
  return __uint_as_float(b);
#else
  union {
    uint32_gpu u;
    float f;
  } c;
  c.u = b;
  return c.f;
#endif
}

DEVICE_FN uint32_gpu p_float_to_bits(float f)
{
#if defined(GPU_TARGET_METAL)
  return as_type<uint32_gpu>(f);
#elif defined(GPU_TARGET_CUDA)
  return __float_as_uint(f);
#else
  union {
    uint32_gpu u;
    float f;
  } c;
  c.f = f;
  return c.u;
#endif
}

//! Natural log, ~1 ulp, deterministic. x > 0 finite.
DEVICE_FN float p_logf(float x)
{
  uint32_gpu ix = p_float_to_bits(x);
  int32_gpu e = (int32_gpu)((ix >> 23) & 0xff) - 126;
  uint32_gpu mb = (ix & 0x007fffffu) | 0x3f000000u; // mantissa in [0.5,1)
  float m = p_bits_to_float(mb);
  if (m < 0.70710678f) {
    m = m + m;
    e -= 1;
  }
  float f = m - 1.0f;
  float z = f * f;
  // Cephes-style minimax for log(1+f) on [sqrt(1/2)-1, sqrt(2)-1]
  float py = 7.0376836292e-2f;
  py = py * f + -1.1514610310e-1f;
  py = py * f + 1.1676998740e-1f;
  py = py * f + -1.2420140846e-1f;
  py = py * f + 1.4249322787e-1f;
  py = py * f + -1.6668057665e-1f;
  py = py * f + 2.0000714765e-1f;
  py = py * f + -2.4999993993e-1f;
  py = py * f + 3.3333331174e-1f;
  py = py * f * z;
  float ef = (float)e;
  py = py + ef * -2.12194440e-4f;
  py = py - 0.5f * z;
  float r = f + py;
  r = r + ef * 0.693359375f;
  return r;
}

//! exp, ~1 ulp, deterministic. |x| < 88.
DEVICE_FN float p_expf(float x)
{
  // k = round(x/ln2)
  float kf = x * 1.44269504088896341f;
  kf = floorf(kf + 0.5f);
  float r = x - kf * 0.693359375f;
  r = r - kf * -2.12194440e-4f;
  float z = r * r;
  float py = 1.9875691500e-4f;
  py = py * r + 1.3981999507e-3f;
  py = py * r + 8.3334519073e-3f;
  py = py * r + 4.1665795894e-2f;
  py = py * r + 1.6666665459e-1f;
  py = py * r + 5.0000001201e-1f;
  py = py * z + r + 1.0f;
  int32_gpu k = (int32_gpu)kf;
  uint32_gpu scale = (uint32_gpu)(k + 127) << 23;
  return py * p_bits_to_float(scale);
}

//! sin and cos for x in [0, 2*pi] (the azimuth-sampling domain),
//! deterministic. Quadrant reduction + Cephes polynomials.
DEVICE_FN void p_sincosf(float x, THREAD float* s, THREAD float* c)
{
  // reduce x = y + q2*(pi/2) with y in [-pi/4, pi/4], q2 in {0,1,2,3}
  float j = floorf(x * 1.27323954473516f); // x / (pi/4)
  int32_gpu q = (int32_gpu)j;
  if (q & 1) {
    q += 1;
    j += 1.0f;
  }
  // extended-precision pi/4 splits (Cephes DP1/DP2/DP3)
  float y = x - j * 0.78515625f;
  y = y - j * 2.4187564849853515625e-4f;
  y = y - j * 3.77489497744594108e-8f;
  int32_gpu q2 = (q >> 1) & 3;
  float z = y * y;
  // sin(y) polynomial
  float ps = -1.9515295891e-4f;
  ps = ps * z + 8.3321608736e-3f;
  ps = ps * z + -1.6666654611e-1f;
  ps = ps * z * y + y;
  // cos(y) polynomial
  float pc = 2.443315711809948e-5f;
  pc = pc * z + -1.388731625493765e-3f;
  pc = pc * z + 4.166664568298827e-2f;
  pc = pc * z * z - 0.5f * z + 1.0f;
  // sin(x) = [sin, cos, -sin, -cos][q2], cos(x) = [cos, -sin, -cos, sin][q2]
  float sv, cv;
  switch (q2) {
  case 0:
    sv = ps;
    cv = pc;
    break;
  case 1:
    sv = pc;
    cv = -ps;
    break;
  case 2:
    sv = -ps;
    cv = -pc;
    break;
  default:
    sv = -pc;
    cv = ps;
    break;
  }
  *s = sv;
  *c = cv;
}

//! sqrt via integer seed + Newton, deterministic, ~1-2 ulp. x >= 0.
DEVICE_FN float p_sqrtf(float x)
{
  if (x <= 0.0f)
    return 0.0f;
  uint32_gpu i = p_float_to_bits(x);
  i = 0x5f3759dfu - (i >> 1); // rsqrt seed
  float y = p_bits_to_float(i);
  // Newton for rsqrt (3 iterations)
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  y = y * (1.5f - 0.5f * x * y * y);
  float r = x * y;
  // one Heron step in sqrt space for the last bit
  r = 0.5f * (r + x / r);
  return r;
}

DEVICE_FN float p_sinf(float x)
{
  float s, c;
  p_sincosf(x, &s, &c);
  return s;
}
DEVICE_FN float p_cosf(float x)
{
  float s, c;
  p_sincosf(x, &s, &c);
  return c;
}

// From here on, device code computes these functions portably.
#ifdef sqrtf
#undef sqrtf
#endif
#ifdef logf
#undef logf
#endif
#ifdef expf
#undef expf
#endif
#ifdef sinf
#undef sinf
#endif
#ifdef cosf
#undef cosf
#endif
#define sqrtf(x) p_sqrtf(x)
#define logf(x) p_logf(x)
#define expf(x) p_expf(x)
#define sinf(x) p_sinf(x)
#define cosf(x) p_cosf(x)
