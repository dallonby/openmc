//! \file f1d.h
//! Flattened Function1D evaluation (endf.cpp semantics) and the shared
//! sampled-outcome POD. Kept in its own header so both the S(a,b) and
//! continuous-energy modules can use it.

#pragma once

// Function1D blob (i32 arena):
//   [0] type: 0 polynomial, 1 tabulated
//   [1] n (coefficients or pairs)
//   [2] f32 offset (poly: coef[n]; tab: x[n] then y[n])
//   [3] n_regions (tabulated)
//   [4] region-pair offset in i32 arena: (nbt[i], interp[i]) pairs
#define GPU_F1D_POLY 0
#define GPU_F1D_TAB 1

// interpolation codes (match openmc::Interpolation)
#define GPU_INTERP_HISTOGRAM 1
#define GPU_INTERP_LINLIN 2
#define GPU_INTERP_LINLOG 3
#define GPU_INTERP_LOGLIN 4
#define GPU_INTERP_LOGLOG 5

struct GpuSampleEA {
  float E_out;
  float mu;
};

DEVICE_FN float gpu_f1d_view(GLOBAL const int32_gpu* i32_arena,
  GLOBAL const float* f32_arena, int32_gpu blob, float x)
{
  GLOBAL const int32_gpu* h = i32_arena + blob;
  int32_gpu type = h[0];
  int32_gpu n = h[1];
  GLOBAL const float* d = f32_arena + h[2];
  if (type == GPU_F1D_POLY) {
    float y = 0.0f;
    for (int32_gpu i = n - 1; i >= 0; --i)
      y = y * x + d[i];
    return y;
  }
  // tabulated
  GLOBAL const float* xv = d;
  GLOBAL const float* yv = d + n;
  if (x <= xv[0])
    return yv[0];
  if (x >= xv[n - 1])
    return yv[n - 1];
  // binary search: largest i with xv[i] <= x
  int32_gpu lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    int32_gpu mid = (lo + hi) / 2;
    if (x >= xv[mid])
      lo = mid;
    else
      hi = mid;
  }
  int32_gpu interp = GPU_INTERP_LINLIN;
  int32_gpu n_regions = h[3];
  if (n_regions > 0) {
    GLOBAL const int32_gpu* reg = i32_arena + h[4];
    for (int32_gpu j = 0; j < n_regions; ++j) {
      if (lo < reg[2 * j]) {
        interp = reg[2 * j + 1];
        break;
      }
    }
  }
  float x0 = xv[lo], x1 = xv[lo + 1];
  float y0 = yv[lo], y1 = yv[lo + 1];
  if (interp == GPU_INTERP_HISTOGRAM)
    return y0;
  if (x1 == x0)
    return y0;
  switch (interp) {
  case GPU_INTERP_LINLIN:
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0);
  case GPU_INTERP_LINLOG:
    return y0 + logf(x / x0) / logf(x1 / x0) * (y1 - y0);
  case GPU_INTERP_LOGLIN:
    return y0 * expf((x - x0) / (x1 - x0) * logf(y1 / y0));
  default: // log-log
    return y0 * expf(logf(x / x0) / logf(x1 / x0) * logf(y1 / y0));
  }
}
