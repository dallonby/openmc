//! \file mg.h
//! Device port of OpenMC multigroup physics (mgxs.cpp, scattdata.cpp,
//! physics_mg.cpp @ develop 86ceaad3c). v1 scope: isotropic angle
//! representation, single temperature (selected at flatten), macroscopic
//! data per material, scattering angle laws: isotropic / tabular /
//! histogram (Legendre data is converted to tabular upstream at load).

#pragma once

// Per-material MG header. Vector block at xs_off in the f32 arena:
//   [0]   total[G]
//   [1G]  absorption[G]
//   [2G]  nu_fission[G]
//   [3G]  prompt_nu_fission[G]
//   [4G]  fission[G]
//   [5G]  scatt_xs[G]
//   [6G]  inverse_velocity[G]
#define GPU_MGV_TOTAL 0
#define GPU_MGV_ABSORPTION 1
#define GPU_MGV_NU_FISSION 2
#define GPU_MGV_PROMPT_NU_FISSION 3
#define GPU_MGV_FISSION 4
#define GPU_MGV_SCATT_XS 5
#define GPU_MGV_INV_VELOCITY 6
#define GPU_MGV_COUNT 7

#define GPU_MG_ANGLE_ISOTROPIC 0
#define GPU_MG_ANGLE_TABULAR 1
#define GPU_MG_ANGLE_HISTOGRAM 2

struct GpuMgMat {
  uint32_gpu xs_off;        // f32 arena: GPU_MGV_COUNT vectors of length G
  uint32_gpu chi_p_off;     // f32: chi_prompt [G][G] probabilities (row = g_in)
  uint32_gpu dnf_off;       // f32: delayed_nu_fission [ndg][G]
  uint32_gpu chi_d_off;     // f32: chi_delayed [ndg][G][G]
  uint32_gpu decay_off;     // f32: decay_rate [ndg]
  uint32_gpu sc_bounds_off; // i32: gmin[G], gmax[G]
  uint32_gpu sc_rowptr_off; // i32: row_ptr[G+1] into per-pair arrays
  uint32_gpu sc_prob_off;   // f32: P0 transfer probabilities (rows concat)
  uint32_gpu sc_mult_off;   // f32: multiplicity per pair
  uint32_gpu sc_ang_off;    // i32: 3 ints per pair {type, n_mu, f32_off};
                            // f32_off -> fmu[n_mu] then cdf[n_mu]
  uint32_gpu fissionable;
  uint32_gpu n_delayed;
};

struct GpuMgView {
  GLOBAL const GpuMgMat* mats;
  GLOBAL const int32_gpu* i32;
  GLOBAL const float* f32;
  uint32_gpu n_groups;
};

DEVICE_FN float gpu_mg_vec(GpuMgView mg, GpuMgMat m, int which, int32_gpu g)
{
  return mg.f32[m.xs_off + (uint32_gpu)which * mg.n_groups + (uint32_gpu)g];
}

// macro XS for the current group (Mgxs::calculate_xs equivalent)
struct GpuMacroXS {
  float total;
  float absorption;
  float fission;
  float nu_fission;
};

DEVICE_FN GpuMacroXS gpu_mg_calculate_xs(
  GpuMgView mg, int32_gpu i_mat, int32_gpu g, float density_mult)
{
  GpuMgMat m = mg.mats[i_mat];
  GpuMacroXS xs;
  xs.total = gpu_mg_vec(mg, m, GPU_MGV_TOTAL, g) * density_mult;
  xs.absorption = gpu_mg_vec(mg, m, GPU_MGV_ABSORPTION, g) * density_mult;
  xs.fission = gpu_mg_vec(mg, m, GPU_MGV_FISSION, g) * density_mult;
  xs.nu_fission = m.fissionable
                    ? gpu_mg_vec(mg, m, GPU_MGV_NU_FISSION, g) * density_mult
                    : 0.0f;
  return xs;
}

// rotate_angle (math_functions.cpp:772): rotate direction u by polar cosine
// mu with azimuth sampled uniformly. The pole branch uses the CPU's exact
// expansion about the v component (a different-but-valid frame there would
// be a constant azimuth phase — statistically identical but it splits
// paired CPU/GPU trajectories for z-aligned directions). The pole test
// matches CPU's 1e-10: in fp32 the smallest nonzero b is ~3.4e-4, so any
// smaller threshold selects exactly the b == 0 case.
DEVICE_FN GpuVec3 gpu_rotate_angle(GpuVec3 u, float mu, THREAD uint64_gpu* seed)
{
  float phi = 6.283185307179586f * gpu_prn(seed);
  float a = sqrtf(fmaxf(0.0f, 1.0f - mu * mu));
  float cos_phi = cosf(phi);
  float sin_phi = sinf(phi);
  float u0 = u.x, v0 = u.y, w0 = u.z;
  float b = sqrtf(fmaxf(0.0f, 1.0f - w0 * w0));
  GpuVec3 out;
  if (b > 1.0e-10f) {
    out.x = mu * u0 + a * (u0 * w0 * cos_phi - v0 * sin_phi) / b;
    out.y = mu * v0 + a * (v0 * w0 * cos_phi + u0 * sin_phi) / b;
    out.z = mu * w0 - a * b * cos_phi;
  } else {
    b = sqrtf(fmaxf(0.0f, 1.0f - v0 * v0));
    out.x = mu * u0 + a * (-u0 * v0 * sin_phi + w0 * cos_phi) / b;
    out.y = mu * v0 + a * b * sin_phi;
    out.z = mu * w0 - a * (v0 * w0 * sin_phi + u0 * cos_phi) / b;
  }
  // renormalize to guard fp32 drift (CPU fp64 skips this)
  float n = sqrtf(out.x * out.x + out.y * out.y + out.z * out.z);
  out.x /= n;
  out.y /= n;
  out.z /= n;
  return out;
}

DEVICE_FN GpuVec3 gpu_isotropic_direction(THREAD uint64_gpu* seed)
{
  float mu = 2.0f * gpu_prn(seed) - 1.0f;
  float phi = 6.283185307179586f * gpu_prn(seed);
  float a = sqrtf(fmaxf(0.0f, 1.0f - mu * mu));
  return gpu_v3(mu, a * cosf(phi), a * sinf(phi));
}

//! ScattData::sample_energy + angle sampling + multiplicity.
//! Returns new group; updates *mu and *wgt.
DEVICE_FN int32_gpu gpu_mg_sample_scatter(GpuMgView mg, GpuMgMat m,
  int32_gpu gin, THREAD float* mu, THREAD float* wgt, THREAD uint64_gpu* seed)
{
  int32_gpu G = (int32_gpu)mg.n_groups;
  int32_gpu gmin = mg.i32[m.sc_bounds_off + gin];
  int32_gpu gmax = mg.i32[m.sc_bounds_off + G + gin];
  int32_gpu row = mg.i32[m.sc_rowptr_off + gin];

  // outgoing group: linear scan of transfer probabilities (scattdata.cpp:160)
  float xi = gpu_prn(seed);
  float prob = 0.0f;
  int32_gpu i_gout = 0;
  int32_gpu gout = gmin;
  for (; gout < gmax; ++gout) {
    prob += mg.f32[m.sc_prob_off + row + i_gout];
    if (xi < prob)
      break;
    ++i_gout;
  }

  // angle
  int32_gpu pair = row + i_gout;
  int32_gpu atype = mg.i32[m.sc_ang_off + 3 * pair];
  int32_gpu n_mu = mg.i32[m.sc_ang_off + 3 * pair + 1];
  int32_gpu aoff = mg.i32[m.sc_ang_off + 3 * pair + 2];
  float mu_out;
  if (atype == GPU_MG_ANGLE_ISOTROPIC) {
    mu_out = 2.0f * gpu_prn(seed) - 1.0f;
  } else if (atype == GPU_MG_ANGLE_TABULAR) {
    // mu grid = linspace(-1,1,n); f32 block: fmu[n], cdf[n]
    float dmu = 2.0f / (float)(n_mu - 1);
    float c = gpu_prn(seed);
    GLOBAL const float* fmu = mg.f32 + aoff;
    GLOBAL const float* cdf = mg.f32 + aoff + n_mu;
    int32_gpu k = 0;
    for (; k < n_mu - 2; ++k) {
      if (c < cdf[k + 1])
        break;
    }
    float mu0 = -1.0f + (float)k * dmu;
    float p0 = fmu[k];
    float p1 = fmu[k + 1];
    // stable lin-lin inversion (see gpu_sample_tabular)
    float frac = (p1 - p0) / dmu;
    float disc = fmaxf(0.0f, p0 * p0 + 2.0f * frac * (c - cdf[k]));
    float denom = p0 + sqrtf(disc);
    mu_out = (denom > 0.0f) ? mu0 + 2.0f * (c - cdf[k]) / denom : mu0;
  } else { // histogram
    float dmu = 2.0f / (float)n_mu;
    float c = gpu_prn(seed);
    GLOBAL const float* cdf = mg.f32 + aoff + n_mu; // running integral
    int32_gpu k = 0;
    for (; k < n_mu - 1; ++k) {
      if (c < cdf[k])
        break;
    }
    mu_out = -1.0f + (float)k * dmu + gpu_prn(seed) * dmu;
  }
  if (mu_out > 1.0f)
    mu_out = 1.0f;
  if (mu_out < -1.0f)
    mu_out = -1.0f;
  *mu = mu_out;
  *wgt *= mg.f32[m.sc_mult_off + pair];
  return gout;
}

//! Mgxs::sample_fission_energy — exactly 2 RNs, both drawn before branching
struct GpuMgFission {
  int32_gpu gout;
  int32_gpu dg; // -1 = prompt
};

DEVICE_FN GpuMgFission gpu_mg_sample_fission_energy(
  GpuMgView mg, GpuMgMat m, int32_gpu gin, THREAD uint64_gpu* seed)
{
  int32_gpu G = (int32_gpu)mg.n_groups;
  float nu_f = gpu_mg_vec(mg, m, GPU_MGV_NU_FISSION, gin);
  float prob_prompt = gpu_mg_vec(mg, m, GPU_MGV_PROMPT_NU_FISSION, gin);
  float xi_pd = gpu_prn(seed) * nu_f;
  float xi_gout = gpu_prn(seed);
  GpuMgFission out;
  out.dg = -1;
  out.gout = 0;
  if (xi_pd <= prob_prompt) {
    float pg = 0.0f;
    GLOBAL const float* chi = mg.f32 + m.chi_p_off + gin * G;
    int32_gpu gout = 0;
    for (; gout < G; ++gout) {
      pg += chi[gout];
      if (xi_gout < pg)
        break;
    }
    out.gout = (gout < G) ? gout : G - 1;
  } else {
    int32_gpu ndg = (int32_gpu)m.n_delayed;
    int32_gpu dg = 0;
    for (; dg < ndg; ++dg) {
      prob_prompt += mg.f32[m.dnf_off + dg * G + gin];
      if (xi_pd < prob_prompt)
        break;
    }
    if (dg > ndg - 1)
      dg = ndg - 1;
    out.dg = dg;
    float pg = 0.0f;
    GLOBAL const float* chi = mg.f32 + m.chi_d_off + (dg * G + gin) * G;
    int32_gpu gout = 0;
    for (; gout < G; ++gout) {
      pg += chi[gout];
      if (xi_gout < pg)
        break;
    }
    out.gout = (gout < G) ? gout : G - 1;
  }
  return out;
}
