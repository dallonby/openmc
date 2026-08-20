//! \file sab.h
//! Device port of S(a,b) thermal scattering (thermal.cpp,
//! secondary_thermal.cpp @ develop 86ceaad3c): bound elastic (coherent
//! Bragg, incoherent Debye-Waller, discrete-cosine) and inelastic
//! (continuous and discrete/equiprobable) laws, with the micro
//! cross-section blending of Nuclide::calculate_sab_xs. Single temperature
//! per table (selected at flatten).

#pragma once

// elastic xs representations
#define GPU_SABXS_NONE 0
#define GPU_SABXS_COHERENT 1   // blob: [n, edges_off, factors_off]
#define GPU_SABXS_INCOHERENT 2 // blob: f32 off (bound_xs, debye_waller)
#define GPU_SABXS_TAB 3        // blob: f1d

// thermal angle-energy law codes
#define GPU_TDIST_COH_EL 20
#define GPU_TDIST_INCOH_EL 21
#define GPU_TDIST_INCOH_EL_DISC 22
#define GPU_TDIST_INCOH_INEL_DISC 23
#define GPU_TDIST_INCOH_INEL_CONT 24
#define GPU_TDIST_MIXED_EL 25

struct GpuSabTable {
  float awr;
  float kT;
  float energy_max;
  int32_gpu elastic_xs_type; // GPU_SABXS_*
  int32_gpu elastic_xs_blob;
  int32_gpu elastic_dist; // thermal law blob or -1
  int32_gpu inelastic_xs_f1d;
  int32_gpu inelastic_dist;
};

struct GpuSabView {
  GLOBAL const GpuSabTable* tables;
  GLOBAL const int32_gpu* i32;
  GLOBAL const float* f32;
};

//! elastic + inelastic bound cross sections at E (ThermalData::calculate_xs)
DEVICE_FN void gpu_sab_xs(
  GpuSabView sv, GpuSabTable t, float E, THREAD float* el, THREAD float* inel)
{
  *inel = gpu_f1d_view(sv.i32, sv.f32, t.inelastic_xs_f1d, E);
  switch (t.elastic_xs_type) {
  case GPU_SABXS_COHERENT: {
    GLOBAL const int32_gpu* h = sv.i32 + t.elastic_xs_blob;
    int32_gpu n = h[0];
    GLOBAL const float* edges = sv.f32 + h[1];
    GLOBAL const float* factors = sv.f32 + h[2];
    if (E < edges[0]) {
      *el = 0.0f;
    } else {
      int32_gpu lo = 0, hi = n - 1;
      if (E >= edges[n - 1]) {
        lo = n - 1;
      } else {
        while (hi - lo > 1) {
          int32_gpu mid = (lo + hi) / 2;
          if (E >= edges[mid])
            lo = mid;
          else
            hi = mid;
        }
      }
      *el = factors[lo] / E;
    }
    break;
  }
  case GPU_SABXS_INCOHERENT: {
    GLOBAL const float* d = sv.f32 + t.elastic_xs_blob;
    float W = d[1];
    float c = 2.0f * E * W;
    // sigma_b/2 * (1 - exp(-2c)) / c   (ENDF-102 Eq. 7.5 integrated)
    *el = (c > 1.0e-6f) ? d[0] * 0.5f * (1.0f - expf(-2.0f * c)) / c
                        : d[0] * (1.0f - 0.5f * c);
    break;
  }
  case GPU_SABXS_TAB:
    *el = gpu_f1d_view(sv.i32, sv.f32, t.elastic_xs_blob, E);
    break;
  default:
    *el = 0.0f;
    break;
  }
}

//! get_energy_index equivalent on an f32 grid
DEVICE_FN void gpu_sab_energy_index(GLOBAL const float* eg, int32_gpu n,
  float E, THREAD int32_gpu* i, THREAD float* f)
{
  *i = 0;
  *f = 0.0f;
  if (E >= eg[0]) {
    int32_gpu lo = 0, hi = n - 1;
    if (E >= eg[n - 1]) {
      lo = n - 1;
    } else {
      while (hi - lo > 1) {
        int32_gpu mid = (lo + hi) / 2;
        if (E >= eg[mid])
          lo = mid;
        else
          hi = mid;
      }
    }
    *i = lo;
    if (lo + 1 < n)
      *f = (E - eg[lo]) / (eg[lo + 1] - eg[lo]);
  }
}

//! Sample one thermal law blob. Mirrors secondary_thermal.cpp exactly,
//! including RN order and the discrete-cosine smearing.
DEVICE_FN GpuSampleEA gpu_sab_sample_dist(
  GpuSabView sv, int32_gpu blob, float E_in, THREAD uint64_gpu* seed)
{
  GpuSampleEA out;
  out.E_out = E_in;
  out.mu = 1.0f;
  // MIXED_EL redirects to a sub-law; loop instead of recursing (MSL
  // forbids recursion)
  for (int redirect = 0; redirect < 2; ++redirect) {
    GLOBAL const int32_gpu* h = sv.i32 + blob;
    switch (h[0]) {
  case GPU_TDIST_COH_EL: {
    // [1] n, [2] edges off, [3] factors off
    int32_gpu n = h[1];
    GLOBAL const float* edges = sv.f32 + h[2];
    GLOBAL const float* factors = sv.f32 + h[3];
    // lower_bound_index(edges, E_in)
    int32_gpu i = 0;
    {
      int32_gpu lo = 0, hi = n - 1;
      if (E_in >= edges[n - 1]) {
        lo = n - 1;
      } else {
        while (hi - lo > 1) {
          int32_gpu mid = (lo + hi) / 2;
          if (E_in >= edges[mid])
            lo = mid;
          else
            hi = mid;
        }
      }
      i = lo;
    }
    float prob = gpu_prn(seed) * factors[i];
    // k = std::lower_bound(factors, factors + i, prob)
    int32_gpu k = 0;
    while (k < i && factors[k] < prob)
      ++k;
    out.mu = 1.0f - 2.0f * edges[k] / E_in;
    break;
  }
  case GPU_TDIST_INCOH_EL: {
    // [1] f32 off (bound_xs, debye_waller)
    float W = sv.f32[h[1] + 1];
    float c = 2.0f * E_in * W;
    if (c > 1.0e-5f) {
      out.mu =
        logf(1.0f + gpu_prn(seed) * (expf(2.0f * c) - 1.0f)) / c - 1.0f;
    } else {
      out.mu = 2.0f * gpu_prn(seed) - 1.0f;
    }
    break;
  }
  case GPU_TDIST_INCOH_EL_DISC: {
    // [1] n_E, [2] Egrid off, [3] n_mu, [4] mu_out off [n_E][n_mu]
    int32_gpu n_e = h[1];
    GLOBAL const float* eg = sv.f32 + h[2];
    int32_gpu n_mu = h[3];
    GLOBAL const float* mo = sv.f32 + h[4];
    int32_gpu i;
    float f;
    gpu_sab_energy_index(eg, n_e, E_in, &i, &f);
    int32_gpu i1 = (i + 1 < n_e) ? i + 1 : i;
    int32_gpu k = (int32_gpu)(gpu_prn(seed) * (float)n_mu);
    if (k > n_mu - 1)
      k = n_mu - 1;
    float mu = mo[i * n_mu + k] + f * (mo[i1 * n_mu + k] - mo[i * n_mu + k]);
    float mu_left =
      (k == 0) ? -1.0f - (mu + 1.0f)
               : mo[i * n_mu + k - 1] +
                   f * (mo[i1 * n_mu + k - 1] - mo[i * n_mu + k - 1]);
    float mu_right =
      (k == n_mu - 1) ? 1.0f + (1.0f - mu)
                      : mo[i * n_mu + k + 1] +
                          f * (mo[i1 * n_mu + k + 1] - mo[i * n_mu + k + 1]);
    mu += fminf(mu - mu_left, mu_right - mu) * (gpu_prn(seed) - 0.5f);
    out.mu = mu;
    break;
  }
  case GPU_TDIST_INCOH_INEL_DISC: {
    // [1] n_E, [2] Egrid, [3] n_out, [4] n_mu, [5] skewed,
    // [6] e_out off [n_E][n_out], [7] mu off [n_E][n_out][n_mu]
    int32_gpu n_e = h[1];
    GLOBAL const float* eg = sv.f32 + h[2];
    int32_gpu n_out = h[3];
    int32_gpu n_mu = h[4];
    int32_gpu skewed = h[5];
    GLOBAL const float* eo = sv.f32 + h[6];
    GLOBAL const float* mo = sv.f32 + h[7];
    int32_gpu i;
    float f;
    gpu_sab_energy_index(eg, n_e, E_in, &i, &f);
    int32_gpu i1 = (i + 1 < n_e) ? i + 1 : i;
    int32_gpu j;
    if (!skewed) {
      j = (int32_gpu)(gpu_prn(seed) * (float)n_out);
      if (j > n_out - 1)
        j = n_out - 1;
    } else {
      float r = gpu_prn(seed) * (float)(n_out - 3);
      if (r > 1.0f)
        j = (int32_gpu)(r + 1.0f);
      else if (r > 0.6f)
        j = n_out - 2;
      else if (r > 0.5f)
        j = n_out - 1;
      else if (r > 0.1f)
        j = 1;
      else
        j = 0;
    }
    out.E_out =
      (1.0f - f) * eo[i * n_out + j] + f * eo[i1 * n_out + j];
    int32_gpu k = (int32_gpu)(gpu_prn(seed) * (float)n_mu);
    if (k > n_mu - 1)
      k = n_mu - 1;
    int32_gpu base_i = (i * n_out + j) * n_mu + k;
    int32_gpu base_i1 = (i1 * n_out + j) * n_mu + k;
    out.mu = (1.0f - f) * mo[base_i] + f * mo[base_i1];
    break;
  }
  case GPU_TDIST_INCOH_INEL_CONT: {
    // [1] n_E, [2] Egrid, [3..] per-E blob offsets
    // per-E: [0] n_out [1] n_mu [2] f32 off (e_out, pdf, cdf, mu rows)
    int32_gpu n_e = h[1];
    GLOBAL const float* eg = sv.f32 + h[2];
    int32_gpu i;
    float f0;
    gpu_sab_energy_index(eg, n_e, E_in, &i, &f0);
    int32_gpu l = (f0 > 0.5f) ? i + 1 : i;
    if (l > n_e - 1)
      l = n_e - 1;
    GLOBAL const int32_gpu* tl = sv.i32 + h[3 + l];
    int32_gpu n = tl[0];
    int32_gpu n_mu = tl[1];
    GLOBAL const float* eo = sv.f32 + tl[2];
    GLOBAL const float* pdf = eo + n;
    GLOBAL const float* cdf = eo + 2 * n;
    GLOBAL const float* mu_rows = eo + 3 * n;
    float r1 = gpu_prn(seed);
    float c_j = cdf[0];
    float c_j1 = c_j;
    int32_gpu j = 0;
    for (; j < n - 1; ++j) {
      c_j1 = cdf[j + 1];
      if (r1 < c_j1)
        break;
      c_j = c_j1;
    }
    if (j > n - 2) {
      j = n - 2;
      c_j = cdf[j];
      c_j1 = cdf[j + 1];
    }
    float E_l_j = eo[j];
    float p_l_j = pdf[j];
    float E_l_j1 = eo[j + 1];
    float p_l_j1 = pdf[j + 1];
    float frac = (p_l_j1 - p_l_j) / (E_l_j1 - E_l_j);
    float E_out;
    // stable lin-lin inversion (see gpu_sample_tabular)
    {
      float disc = fmaxf(0.0f, p_l_j * p_l_j + 2.0f * frac * (r1 - c_j));
      float denom = p_l_j + sqrtf(disc);
      E_out = (denom > 0.0f) ? E_l_j + 2.0f * (r1 - c_j) / denom : E_l_j;
    }
    float E_l = eg[l];
    if (E_out < 0.5f * E_l)
      E_out *= 2.0f * E_in / E_l - 1.0f;
    else
      E_out += E_in - E_l;
    out.E_out = E_out;
    float f = (c_j1 > c_j) ? (r1 - c_j) / (c_j1 - c_j) : 0.0f;
    int32_gpu k = (int32_gpu)(gpu_prn(seed) * (float)n_mu);
    if (k > n_mu - 1)
      k = n_mu - 1;
    int32_gpu j1 = (j + 1 < n) ? j + 1 : j;
    float mu = mu_rows[j * n_mu + k] +
               f * (mu_rows[j1 * n_mu + k] - mu_rows[j * n_mu + k]);
    float mu_left =
      (k == 0) ? -1.0f - (mu + 1.0f)
               : mu_rows[j * n_mu + k - 1] +
                   f * (mu_rows[j1 * n_mu + k - 1] - mu_rows[j * n_mu + k - 1]);
    float mu_right =
      (k == n_mu - 1)
        ? 1.0f + (1.0f - mu)
        : mu_rows[j * n_mu + k + 1] +
            f * (mu_rows[j1 * n_mu + k + 1] - mu_rows[j * n_mu + k + 1]);
    mu += fminf(mu - mu_left, mu_right - mu) * (gpu_prn(seed) - 0.5f);
    out.mu = mu;
    break;
  }
  case GPU_TDIST_MIXED_EL: {
    // [1] coherent blob, [2] incoherent blob, [3] table index (for xs)
    GpuSabTable t = sv.tables[h[3]];
    // evaluate both elastic components: coherent from its blob, incoherent
    // from the second law's xs stored as f1d at [4]
    float xs_coh = 0.0f;
    {
      GLOBAL const int32_gpu* ch = sv.i32 + h[1];
      int32_gpu n = ch[1];
      GLOBAL const float* edges = sv.f32 + ch[2];
      GLOBAL const float* factors = sv.f32 + ch[3];
      if (E_in >= edges[0]) {
        int32_gpu lo = 0, hi = n - 1;
        if (E_in >= edges[n - 1]) {
          lo = n - 1;
        } else {
          while (hi - lo > 1) {
            int32_gpu mid = (lo + hi) / 2;
            if (E_in >= edges[mid])
              lo = mid;
            else
              hi = mid;
          }
        }
        xs_coh = factors[lo] / E_in;
      }
    }
    float xs_incoh = gpu_f1d_view(sv.i32, sv.f32, h[4], E_in);
    (void)t;
    blob = (gpu_prn(seed) * (xs_coh + xs_incoh) < xs_coh) ? h[1] : h[2];
    continue;
  }
    }
    break;
  }
  return out;
}
