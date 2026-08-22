//! \file ce.h
//! Device port of OpenMC continuous-energy neutron physics: cross-section
//! lookup (nuclide.cpp), URR probability tables (urr.cpp/nuclide.cpp),
//! free-gas elastic scattering, secondary distributions
//! (distribution_angle/energy, secondary_{uncorrelated,kalbach,correlated,
//! nbody}) and the collision flow (physics.cpp) @ develop 86ceaad3c.
//!
//! v1 envelope: single temperature per nuclide (chosen at flatten), S(a,b)
//! thermal scattering via sab.h, no DBRC/RVS resonance upscattering
//! (flattener rejects when enabled), no photon production, analog capture.
//! Random-number consumption order mirrors the CPU exactly where
//! trajectories are shared.

#pragma once

#ifdef GPU_HOST_DEBUG
void gpu_host_debug_tab(float r1, int32_gpu k, float ck, float xk, float pk,
  float x1, float p1, int32_gpu interp, int32_gpu nd);
void gpu_host_debug_angle(int32_gpu i, float r1, int32_gpu k, float ck,
  int32_gpu n_mu, int32_gpu interp, float xk, float pk, float m);
#endif

// ---------------------------------------------------------------------------
// PODs
// ---------------------------------------------------------------------------

struct GpuNuclide {
  float awr;
  float kT;            // selected temperature (eV)
  // f32: interleaved per-grid-point records, ascending in energy:
  //   [E, total, absorption]              (xs_stride = 3, non-fissionable)
  //   [E, total, absorption, fis, nu_fis] (xs_stride = 5, fissionable)
  // Energy and cross sections share a cache line because the lookup is
  // bound by random cache-line fetches, not by the search: a log-grid-bins
  // sweep from 1e3 to 2e5 moved the W-slab kernel by <1%, while separate
  // grid/xs arrays cost 2-3 lines per nuclide per collision.
  uint32_gpu grid_off;
  uint32_gpu n_grid;
  uint32_gpu xs_stride;
  uint32_gpu loggrid_off; // i32: n_log_bins+1 hash -> grid index
  uint32_gpu elastic_off; // f32: n_grid elastic xs
  uint32_gpu fissionable;
  int32_gpu total_nu_f1d;  // i32 fn blob or -1
  uint32_gpu n_delayed;    // precursor group count (first fission rx)
  uint32_gpu n_fission_rx; // 1, or several for partial fission
  uint32_gpu fis_rx_off;   // i32: n_fission_rx offsets; each record is
                           // [rx_hdr_off, n_products,
                           //  (yield_f1d, dist_off, decay_f32_off) * n]
  uint32_gpu n_inelastic;
  uint32_gpu inelastic_off; // i32: n_inelastic reaction-header offsets
  int32_gpu elastic_angle;  // angle-distribution blob or -1 (isotropic)
  int32_gpu urr_off;        // i32 urr blob or -1
  uint32_gpu index;         // global nuclide index (URR stream correlation)
};

// reaction header in i32 arena:
//   [0] mt
//   [1] scatter_in_cm
//   [2] threshold index
//   [3] n_xs
//   [4] xs_off (f32)
//   [5] q_off (f32; single value)
//   [6] yield f1d blob (i32) or -1 (treated as 1)
//   [7] product distribution descriptor (i32)
#define GPU_RX_MT 0
#define GPU_RX_CM 1
#define GPU_RX_THRESHOLD 2
#define GPU_RX_NXS 3
#define GPU_RX_XSOFF 4
#define GPU_RX_QOFF 5
#define GPU_RX_YIELD 6
#define GPU_RX_DIST 7
#define GPU_RX_WORDS 8

// AngleEnergy distribution descriptor types
#define GPU_DIST_LEVEL 0
#define GPU_DIST_CONT_TAB 1
#define GPU_DIST_KALBACH 2
#define GPU_DIST_UNCORR 3
#define GPU_DIST_MAXWELL 4
#define GPU_DIST_EVAPORATION 5
#define GPU_DIST_WATT 6
#define GPU_DIST_NBODY 7
#define GPU_DIST_CORRELATED 8
#define GPU_DIST_MULTI 9 // applicability-weighted set of distributions

struct GpuCeView {
  GLOBAL const GpuNuclide* nuclides;
  GLOBAL const int32_gpu* i32;
  GLOBAL const float* f32;
  uint32_gpu n_nuclides;
  uint32_gpu n_log_bins;
  float log_spacing;
  float energy_min;
  float energy_max;
  uint32_gpu urr_on;
};

// per-nuclide micro cross sections at the current energy
struct GpuMicroXS {
  float total;
  float absorption;
  float fission;
  float nu_fission;
  float elastic; // < 0 -> not yet computed
  float thermal; // bound S(a,b) elastic+inelastic (scaled by sab_frac)
  float thermal_elastic;
  float sab_frac;
  int32_gpu i_sab; // thermal table index or -1
  int32_gpu i_grid;
  float interp;
  uint32_gpu use_ptable;
};

// ---------------------------------------------------------------------------
// Function1D evaluation (endf.cpp semantics)
// ---------------------------------------------------------------------------

DEVICE_FN float gpu_f1d(GpuCeView ce, int32_gpu blob, float x)
{
  return gpu_f1d_view(ce.i32, ce.f32, blob, x);
}


// ---------------------------------------------------------------------------
// XS lookup
// ---------------------------------------------------------------------------

//! Nuclide::calculate_xs equivalent (single temperature), including the
//! S(a,b) blending of calculate_sab_xs when i_sab >= 0.
//! First URR band with cdf[i] > r, clamped to n-1: the index a forward linear
//! scan would stop on, found in log2(n) uniform steps instead. The tables are
//! ~20 bands wide and neighbouring histories draw unrelated variates, so the
//! scan made every SIMD group wait on its unluckiest lane. Worth 3.2% of the
//! W-slab kernel; the CDF is monotone, so the index is unchanged.
DEVICE_FN int32_gpu gpu_urr_band(GLOBAL const float* cdf, int32_gpu n, float r)
{
  int32_gpu lo = 0;
  int32_gpu hi = n - 1;
  while (lo < hi) {
    int32_gpu mid = (lo + hi) >> 1;
    if (cdf[mid] <= r)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

DEVICE_FN GpuMicroXS gpu_ce_micro_xs(GpuCeView ce, GpuNuclide nuc, float E,
  int32_gpu i_log, int32_gpu i_sab, float sab_frac, GpuSabView sab,
  THREAD uint64_gpu* seeds)
{
  GpuMicroXS m;
  m.elastic = -1.0f;
  m.thermal = 0.0f;
  m.thermal_elastic = 0.0f;
  m.sab_frac = 0.0f;
  m.i_sab = -1;
  m.use_ptable = 0;

  GLOBAL const float* rec = ce.f32 + nuc.grid_off;
  int32_gpu st = (int32_gpu)nuc.xs_stride;
  int32_gpu ng = (int32_gpu)nuc.n_grid;
  int32_gpu i_grid;
  if (E <= rec[0]) {
    i_grid = 0;
  } else if (E >= rec[st * (ng - 1)]) {
    i_grid = ng - 2;
  } else {
    GLOBAL const int32_gpu* lg = ce.i32 + nuc.loggrid_off;
    int32_gpu lo = lg[i_log];
    int32_gpu hi = lg[i_log + 1] + 1;
    if (hi > ng - 1)
      hi = ng - 1;
    // binary search in [lo, hi] for largest i with grid[i] <= E
    while (hi - lo > 1) {
      int32_gpu mid = (lo + hi) / 2;
      if (E >= rec[st * mid])
        lo = mid;
      else
        hi = mid;
    }
    i_grid = lo;
  }
  // fp32 casts can collapse a RUN of adjacent fp64 knots to one value:
  // walk to the end of the run (never past ng-2, so no OOB read), and use
  // f = 0 if the grid ends inside a collapsed run (row i_grid, no interp)
  while (i_grid + 2 < ng && rec[st * i_grid] == rec[st * (i_grid + 1)])
    ++i_grid;
  GLOBAL const float* r0 = rec + st * i_grid; // this point and the next share
  GLOBAL const float* r1 = r0 + st;           // a cache line for stride 3
  float dgrid = r1[0] - r0[0];
  float f = (dgrid > 0.0f) ? (E - r0[0]) / dgrid : 0.0f;
  m.i_grid = i_grid;
  m.interp = f;

  m.total = (1.0f - f) * r0[1] + f * r1[1];
  m.absorption = (1.0f - f) * r0[2] + f * r1[2];
  if (nuc.fissionable) {
    m.fission = (1.0f - f) * r0[3] + f * r1[3];
    m.nu_fission = (1.0f - f) * r0[4] + f * r1[4];
  } else {
    m.fission = 0.0f;
    m.nu_fission = 0.0f;
  }

  // S(a,b) blending (nuclide.cpp:835 calculate_sab_xs); the caller gates
  // i_sab on E < table.energy_max
  if (i_sab >= 0) {
    GpuSabTable t = sab.tables[i_sab];
    float el_b, inel_b;
    gpu_sab_xs(sab, t, E, &el_b, &inel_b);
    m.thermal = sab_frac * (el_b + inel_b);
    m.thermal_elastic = sab_frac * el_b;
    GLOBAL const float* exs = ce.f32 + nuc.elastic_off;
    float el_free = (1.0f - f) * exs[i_grid] + f * exs[i_grid + 1];
    m.total = m.total + m.thermal - sab_frac * el_free;
    m.elastic = m.thermal + (1.0f - sab_frac) * el_free;
    m.i_sab = i_sab;
    m.sab_frac = sab_frac;
  }

  // URR probability tables
  if (ce.urr_on && nuc.urr_off >= 0) {
    GLOBAL const int32_gpu* uh = ce.i32 + nuc.urr_off;
    // [0] n_energy [1] n_cdf [2] interp [3] inelastic rx header or -1
    // [4] multiply_smooth [5] Egrid f32 off [6] cdf f32 off [7] xs f32 off
    int32_gpu n_e = uh[0];
    GLOBAL const float* ue = ce.f32 + uh[5];
    if (E > ue[0] && E < ue[n_e - 1]) {
      m.use_ptable = 1;
      int32_gpu n_cdf = uh[1];
      // energy index
      int32_gpu lo = 0, hi = n_e - 1;
      while (hi - lo > 1) {
        int32_gpu mid = (lo + hi) / 2;
        if (E >= ue[mid])
          lo = mid;
        else
          hi = mid;
      }
      int32_gpu i_e = lo;
      // nuclide-correlated variate from the URR stream (not consumed)
      uint64_gpu urr_seed = seeds[GPU_STREAM_URR_PTABLE];
      uint64_gpu fut = gpu_future_seed((uint64_gpu)nuc.index, urr_seed);
      uint64_gpu w = gpu_prn_word(&fut);
      float r = ((float)(w >> 40u) + 0.5f) * 0x1p-24f;
      // band search on both bracketing tables
      GLOBAL const float* cdf = ce.f32 + uh[6];
      int32_gpu i_low = gpu_urr_band(cdf + i_e * n_cdf, n_cdf, r);
      int32_gpu i_up = gpu_urr_band(cdf + (i_e + 1) * n_cdf, n_cdf, r);
      GLOBAL const float* uxs = ce.f32 + uh[7]; // triplets (el, fis, ngam)
      float el, fis, cap;
      if (uh[2] == GPU_INTERP_LINLIN) {
        float fe = (E - ue[i_e]) / (ue[i_e + 1] - ue[i_e]);
        el = (1.0f - fe) * uxs[3 * (i_e * n_cdf + i_low)] +
             fe * uxs[3 * ((i_e + 1) * n_cdf + i_up)];
        fis = (1.0f - fe) * uxs[3 * (i_e * n_cdf + i_low) + 1] +
              fe * uxs[3 * ((i_e + 1) * n_cdf + i_up) + 1];
        cap = (1.0f - fe) * uxs[3 * (i_e * n_cdf + i_low) + 2] +
              fe * uxs[3 * ((i_e + 1) * n_cdf + i_up) + 2];
      } else { // log-log
        float fe = logf(E / ue[i_e]) / logf(ue[i_e + 1] / ue[i_e]);
        float e0 = uxs[3 * (i_e * n_cdf + i_low)];
        float e1 = uxs[3 * ((i_e + 1) * n_cdf + i_up)];
        el = (e0 > 0.0f && e1 > 0.0f)
               ? expf((1.0f - fe) * logf(e0) + fe * logf(e1))
               : 0.0f;
        float f0 = uxs[3 * (i_e * n_cdf + i_low) + 1];
        float f1 = uxs[3 * ((i_e + 1) * n_cdf + i_up) + 1];
        fis = (f0 > 0.0f && f1 > 0.0f)
                ? expf((1.0f - fe) * logf(f0) + fe * logf(f1))
                : 0.0f;
        float c0 = uxs[3 * (i_e * n_cdf + i_low) + 2];
        float c1 = uxs[3 * ((i_e + 1) * n_cdf + i_up) + 2];
        cap = (c0 > 0.0f && c1 > 0.0f)
                ? expf((1.0f - fe) * logf(c0) + fe * logf(c1))
                : 0.0f;
      }
      // inelastic competition
      float inelastic = 0.0f;
      if (uh[3] >= 0) {
        GLOBAL const int32_gpu* rx = ce.i32 + uh[3];
        int32_gpu thr = rx[GPU_RX_THRESHOLD];
        if (i_grid >= thr) {
          GLOBAL const float* rxs = ce.f32 + rx[GPU_RX_XSOFF];
          int32_gpu k = i_grid - thr;
          inelastic = (1.0f - f) * rxs[k] + f * rxs[k + 1];
        }
      }
      if (uh[4]) { // multiply_smooth
        GLOBAL const float* exs = ce.f32 + nuc.elastic_off;
        float sm_el = (1.0f - f) * exs[i_grid] + f * exs[i_grid + 1];
        el *= sm_el;
        cap *= (m.absorption - m.fission);
        fis *= m.fission;
      }
      if (el < 0.0f)
        el = 0.0f;
      if (fis < 0.0f)
        fis = 0.0f;
      if (cap < 0.0f)
        cap = 0.0f;
      m.elastic = el;
      m.absorption = cap + fis;
      m.fission = fis;
      m.total = el + inelastic + cap + fis;
      if (nuc.fissionable && fis > 0.0f) {
        float nu =
          (nuc.total_nu_f1d >= 0) ? gpu_f1d(ce, nuc.total_nu_f1d, E) : 0.0f;
        // total nu falls back to prompt in the flattener when absent
        m.nu_fission = nu * fis;
      } else {
        m.nu_fission = 0.0f;
      }
    }
  }
  return m;
}

DEVICE_FN float gpu_ce_elastic_xs(
  GpuCeView ce, GpuNuclide nuc, THREAD GpuMicroXS* m)
{
  if (m->elastic < 0.0f) {
    GLOBAL const float* exs = ce.f32 + nuc.elastic_off;
    m->elastic =
      (1.0f - m->interp) * exs[m->i_grid] + m->interp * exs[m->i_grid + 1];
  }
  return m->elastic;
}

// ---------------------------------------------------------------------------
// Tabular sampling helpers
// ---------------------------------------------------------------------------

//! Sample a (histogram | lin-lin) tabular density given arrays x, p, c of
//! length n with tabulated CDF c (normalized). Mirrors the within-bin logic
//! of ContinuousTabular::sample / Tabular::sample.
DEVICE_FN float gpu_sample_tabular(GLOBAL const float* x, GLOBAL const float* p,
  GLOBAL const float* c, int32_gpu n, int32_gpu n_discrete, int32_gpu interp,
  float r1, THREAD int32_gpu* k_out, THREAD float* c_k_out,
  THREAD float* c_k1_out)
{
  // discrete lines first. c_k / c_k1 keep the CPU walk's exact lifecycle:
  // c_k stays stale (c[k-1]) on the first continuous bin after discrete
  // lines, and c_k1 is only assigned inside the continuous loop (INFTY when
  // that loop never runs; stale == c_k when it exhausts the last bin) —
  // the correlated law's nearest-CDF angle pick depends on both quirks.
  int32_gpu k = 0;
  float c_k = c[0];
  float c_k1 = GPU_INFTY;
  int32_gpu end = n - 2;
  bool discrete_hit = false;
  for (int32_gpu j = 0; j < n_discrete; ++j) {
    k = j;
    c_k = c[k];
    if (r1 < c_k) {
      end = j;
      discrete_hit = true;
      break;
    }
  }
  if (!discrete_hit) {
    for (int32_gpu j = n_discrete; j < end; ++j) {
      k = j;
      c_k1 = c[k + 1];
      if (r1 < c_k1)
        break;
      k = j + 1;
      c_k = c_k1;
    }
  }
  *k_out = k;
  *c_k_out = c_k;
  *c_k1_out = c_k1;

  float xk = x[k];
#ifdef GPU_HOST_DEBUG
  gpu_host_debug_tab(r1, k, c_k, xk, p[k], (k + 1 < n) ? x[k + 1] : 99.0f,
    (k + 1 < n) ? p[k + 1] : 99.0f, interp, n_discrete);
#endif
  if (interp == GPU_INTERP_HISTOGRAM || k < n_discrete) {
    if (k >= n_discrete && p[k] > 0.0f)
      return xk + (r1 - c_k) / p[k];
    return xk;
  }
  // lin-lin: numerically stable inversion. The textbook form
  // (sqrt(p^2 + 2m(r-c)) - p)/m collapses to the bin edge in fp32 when the
  // pdf is nearly flat (m ~ ulp): p^2 + 2m(r-c) rounds back to p^2. The
  // algebraically identical form 2(r-c)/(p + sqrt(p^2 + 2m(r-c))) is
  // stable for m -> 0 and needs no flat-pdf special case.
  float x1 = x[k + 1];
  if (x1 == xk) {
    return (p[k] > 0.0f) ? xk + (r1 - c_k) / p[k] : xk;
  }
  float frac = (p[k + 1] - p[k]) / (x1 - xk);
  float disc = fmaxf(0.0f, p[k] * p[k] + 2.0f * frac * (r1 - c_k));
  float denom = p[k] + sqrtf(disc);
  if (denom <= 0.0f)
    return xk;
  return xk + 2.0f * (r1 - c_k) / denom;
}

//! AngleDistribution::sample — tabulated mu tables on an incident-energy
//! grid, stochastic bin pick, then tabular inverse-CDF.
DEVICE_FN float gpu_sample_angle_dist(
  GpuCeView ce, int32_gpu blob, float E, THREAD uint64_gpu* seed)
{
  GLOBAL const int32_gpu* h = ce.i32 + blob;
  int32_gpu n_e = h[0];
  GLOBAL const float* eg = ce.f32 + h[1];
  // get_energy_index + stochastic pick
  int32_gpu i = 0;
  float r = 0.0f;
  if (E >= eg[0]) {
    int32_gpu lo = 0, hi = n_e - 1;
    if (E >= eg[n_e - 1]) {
      lo = n_e - 1;
    } else {
      while (hi - lo > 1) {
        int32_gpu mid = (lo + hi) / 2;
        if (E >= eg[mid])
          lo = mid;
        else
          hi = mid;
      }
    }
    i = lo;
    if (i + 1 < n_e) {
      // fp32 casts can collapse adjacent fp64 knots; r -> 0 matches the
      // CPU limit (E == both knots) instead of an Inf/NaN pick
      float de = eg[i + 1] - eg[i];
      r = (de > 0.0f) ? (E - eg[i]) / de : 0.0f;
    }
  }
  if (r > gpu_prn(seed))
    ++i;
  GLOBAL const int32_gpu* tb = ce.i32 + h[2 + i]; // per-E table header
  int32_gpu n_mu = tb[0];
  int32_gpu interp = tb[1];
  GLOBAL const float* mu = ce.f32 + tb[2];
  GLOBAL const float* p = mu + n_mu;
  GLOBAL const float* c = mu + 2 * n_mu;
  float r1 = gpu_prn(seed);
  int32_gpu k;
  float ck, ck1;
  float m = gpu_sample_tabular(mu, p, c, n_mu, 0, interp, r1, &k, &ck, &ck1);
#ifdef GPU_HOST_DEBUG
  gpu_host_debug_angle(i, r1, k, ck, n_mu, interp, mu[k], p[k], m);
#endif
  if (m < -1.0f)
    m = -1.0f;
  if (m > 1.0f)
    m = 1.0f;
  return m;
}

// ---------------------------------------------------------------------------
// Energy / angle-energy distributions
// ---------------------------------------------------------------------------

DEVICE_FN float gpu_maxwell_spectrum(float T, THREAD uint64_gpu* seed)
{
  float r1 = gpu_prn(seed);
  float r2 = gpu_prn(seed);
  float r3 = gpu_prn(seed);
  float c = cosf(1.5707963267948966f * r3);
  return -T * (logf(r1) + logf(r2) * c * c);
}

DEVICE_FN float gpu_watt_spectrum(float a, float b, THREAD uint64_gpu* seed)
{
  float w = gpu_maxwell_spectrum(a, seed);
  float u = 2.0f * gpu_prn(seed) - 1.0f;
  return w + 0.25f * a * a * b + u * sqrtf(a * a * b * w);
}

//! Sample one terminal AngleEnergy law. MULTI and UNCORR redirects are
//! resolved iteratively in gpu_sample_dist (MSL forbids recursion).
//! Mirrors the per-law algorithms including RN order.
DEVICE_FN GpuSampleEA gpu_sample_dist_terminal(
  GpuCeView ce, int32_gpu blob, float E_in, THREAD uint64_gpu* seed)
{
  GpuSampleEA out;
  out.E_out = 0.0f;
  out.mu = 1.0f;
  GLOBAL const int32_gpu* h = ce.i32 + blob;
  int32_gpu type = h[0];

  switch (type) {
  case GPU_DIST_LEVEL: {
    GLOBAL const float* d = ce.f32 + h[1]; // threshold, mass_ratio
    out.E_out = d[1] * (E_in - d[0]);
    // pure energy law: no RN; the UNCORR wrapper supplies the angle
    return out;
  }
  case GPU_DIST_MAXWELL: {
    GLOBAL const float* d = ce.f32 + h[2]; // u
    float theta = gpu_f1d(ce, h[1], E_in);
    float u_r = d[0];
    for (int i = 0; i < 100000; ++i) {
      out.E_out = gpu_maxwell_spectrum(theta, seed);
      if (out.E_out <= E_in - u_r)
        break;
    }
    return out;
  }
  case GPU_DIST_EVAPORATION: {
    GLOBAL const float* d = ce.f32 + h[2];
    float theta = gpu_f1d(ce, h[1], E_in);
    float u_r = d[0];
    float y = (E_in - u_r) / theta;
    float v = 1.0f - expf(-y);
    float x;
    do {
      x = -logf((1.0f - v * gpu_prn(seed)) * (1.0f - v * gpu_prn(seed)));
    } while (x > y);
    out.E_out = x * theta;
    return out;
  }
  case GPU_DIST_WATT: {
    GLOBAL const float* d = ce.f32 + h[3];
    float a = gpu_f1d(ce, h[1], E_in);
    float b = gpu_f1d(ce, h[2], E_in);
    float u_r = d[0];
    for (int i = 0; i < 100000; ++i) {
      out.E_out = gpu_watt_spectrum(a, b, seed);
      if (out.E_out <= E_in - u_r)
        break;
    }
    return out;
  }
  case GPU_DIST_NBODY: {
    // [1] n_bodies, [2] f32 off (Ap, A, Q)
    int32_gpu n_bodies = h[1];
    GLOBAL const float* d = ce.f32 + h[2];
    float Ap = d[0], A = d[1], Q = d[2];
    float E_max = (Ap - 1.0f) / Ap * (A / (A + 1.0f) * E_in + Q);
    out.mu = 2.0f * gpu_prn(seed) - 1.0f; // mu BEFORE energy (CPU order)
    float x = gpu_maxwell_spectrum(1.0f, seed);
    float y;
    if (n_bodies == 3) {
      y = gpu_maxwell_spectrum(1.0f, seed);
    } else if (n_bodies == 4) {
      y = -logf(gpu_prn(seed) * gpu_prn(seed) * gpu_prn(seed));
    } else {
      float r1 = gpu_prn(seed), r2 = gpu_prn(seed), r3 = gpu_prn(seed),
            r4 = gpu_prn(seed), r5 = gpu_prn(seed), r6 = gpu_prn(seed);
      float cc = cosf(1.5707963267948966f * r6);
      y = -logf(r1 * r2 * r3 * r4) - logf(r5) * cc * cc;
    }
    out.E_out = E_max * x / (x + y);
    return out;
  }
  case GPU_DIST_CONT_TAB:
  case GPU_DIST_KALBACH: {
    // header: [1] histogram_interp (cont-tab only), [2] n_E,
    //         [3] Egrid f32 off, [4..] per-E table blob offsets
    bool km = (type == GPU_DIST_KALBACH);
    int32_gpu hist_interp = h[1];
    int32_gpu n_e = h[2];
    GLOBAL const float* eg = ce.f32 + h[3];

    int32_gpu i;
    float r;
    if (km) {
      // Kalbach uses get_energy_index (clamped, r=0 below grid)
      i = 0;
      r = 0.0f;
      if (E_in >= eg[0]) {
        int32_gpu lo = 0, hi = n_e - 1;
        if (E_in >= eg[n_e - 1]) {
          lo = n_e - 1;
        } else {
          while (hi - lo > 1) {
            int32_gpu mid = (lo + hi) / 2;
            if (E_in >= eg[mid])
              lo = mid;
            else
              hi = mid;
          }
        }
        i = lo;
        if (i + 1 < n_e) {
          float de = eg[i + 1] - eg[i];
          r = (de > 0.0f) ? (E_in - eg[i]) / de : 0.0f;
        }
      }
    } else {
      if (E_in < eg[0]) {
        i = 0;
        r = 0.0f;
      } else if (E_in > eg[n_e - 1]) {
        i = n_e - 2;
        r = 1.0f;
      } else {
        int32_gpu lo = 0, hi = n_e - 1;
        while (hi - lo > 1) {
          int32_gpu mid = (lo + hi) / 2;
          if (E_in >= eg[mid])
            lo = mid;
          else
            hi = mid;
        }
        i = lo;
        float de = eg[i + 1] - eg[i];
        r = (de > 0.0f) ? (E_in - eg[i]) / de : 0.0f;
      }
    }
    int32_gpu l = i;
    if (!(hist_interp && !km)) {
      if (r > gpu_prn(seed))
        l = i + 1;
    }

    // per-E table: [0] n_discrete [1] interp [2] n_out [3] f32 off
    // (e_out[n], p[n], c[n], and for KM r[n], a[n])
    GLOBAL const int32_gpu* tl = ce.i32 + h[4 + l];
    int32_gpu n_disc = tl[0];
    int32_gpu interp = tl[1];
    int32_gpu n_out = tl[2];
    GLOBAL const float* xe = ce.f32 + tl[3];
    GLOBAL const float* pe = xe + n_out;
    GLOBAL const float* cc = xe + 2 * n_out;

    // interpolation bounds for unit-base scaling (computed pre-sample)
    GLOBAL const int32_gpu* ti = ce.i32 + h[4 + i];
    GLOBAL const int32_gpu* ti1 = ce.i32 + h[4 + ((i + 1 < n_e) ? i + 1 : i)];
    int32_gpu nd_i = ti[0];
    int32_gpu nd_i1 = ti1[0];
    GLOBAL const float* xi = ce.f32 + ti[3];
    GLOBAL const float* xi1 = ce.f32 + ti1[3];
    float E_i_1 = xi[nd_i];
    float E_i_K = xi[ti[2] - 1];
    float E_i1_1 = xi1[nd_i1];
    float E_i1_K = xi1[ti1[2] - 1];
    float E_1 = E_i_1 + r * (E_i1_1 - E_i_1);
    float E_K = E_i_K + r * (E_i1_K - E_i_K);

    float r1 = gpu_prn(seed);
    int32_gpu k;
    float c_k, c_k1;
    float E_out = gpu_sample_tabular(
      xe, pe, cc, n_out, n_disc, interp, r1, &k, &c_k, &c_k1);

    float km_r = 0.0f, km_a = 0.0f;
    if (km) {
      GLOBAL const float* rr = xe + 3 * n_out;
      GLOBAL const float* aa = xe + 4 * n_out;
      if (interp == GPU_INTERP_HISTOGRAM || k < n_disc || xe[k + 1] == xe[k]) {
        km_r = rr[k];
        km_a = aa[k];
      } else {
        float fr = (E_out - xe[k]) / (xe[k + 1] - xe[k]);
        km_r = rr[k] + fr * (rr[k + 1] - rr[k]);
        km_a = aa[k] + fr * (aa[k + 1] - aa[k]);
      }
    }

    // unit-base scaling
    bool scale =
      km ? (k >= n_disc) : (!hist_interp && n_out > 1 && k >= n_disc);
    if (scale) {
      GLOBAL const float* xl = ce.f32 + tl[3];
      float El_1 = xl[tl[0]];
      float El_K = xl[tl[2] - 1];
      if (El_K != El_1)
        E_out = E_1 + (E_out - El_1) * (E_K - E_1) / (El_K - El_1);
    }
    out.E_out = E_out;

    if (km) {
      // Kalbach-Mann angle systematics
      if (gpu_prn(seed) > km_r) {
        float T =
          (2.0f * gpu_prn(seed) - 1.0f) * (expf(km_a) - expf(-km_a)) * 0.5f;
        out.mu = logf(T + sqrtf(T * T + 1.0f)) / km_a;
      } else {
        float rr1 = gpu_prn(seed);
        out.mu = logf(rr1 * expf(km_a) + (1.0f - rr1) * expf(-km_a)) / km_a;
      }
      if (out.mu > 1.0f)
        out.mu = 1.0f;
      if (out.mu < -1.0f)
        out.mu = -1.0f;
    }
    // non-KM continuous tabular is a pure energy law: angle comes from the
    // UNCORR wrapper
    return out;
  }
  case GPU_DIST_CORRELATED: {
    // header: [1] n_E, [2] Egrid off, [3..] per-E table blobs
    // per-E: [0] n_disc [1] interp [2] n_out [3] f32(e,p,c) [4..] mu blobs
    int32_gpu n_e = h[1];
    GLOBAL const float* eg = ce.f32 + h[2];
    int32_gpu i = 0;
    float r = 0.0f;
    if (E_in >= eg[0]) {
      int32_gpu lo = 0, hi = n_e - 1;
      if (E_in >= eg[n_e - 1]) {
        lo = n_e - 1;
      } else {
        while (hi - lo > 1) {
          int32_gpu mid = (lo + hi) / 2;
          if (E_in >= eg[mid])
            lo = mid;
          else
            hi = mid;
        }
      }
      i = lo;
      if (i + 1 < n_e) {
        float de = eg[i + 1] - eg[i];
        r = (de > 0.0f) ? (E_in - eg[i]) / de : 0.0f;
      }
    }
    int32_gpu l = i;
    if (r > gpu_prn(seed))
      l = i + 1;

    GLOBAL const int32_gpu* tl = ce.i32 + h[3 + l];
    int32_gpu n_disc = tl[0];
    int32_gpu interp = tl[1];
    int32_gpu n_out = tl[2];
    GLOBAL const float* xe = ce.f32 + tl[3];
    GLOBAL const float* pe = xe + n_out;
    GLOBAL const float* cc = xe + 2 * n_out;

    GLOBAL const int32_gpu* ti = ce.i32 + h[3 + i];
    GLOBAL const int32_gpu* ti1 = ce.i32 + h[3 + ((i + 1 < n_e) ? i + 1 : i)];
    float E_i_1 = (ce.f32 + ti[3])[ti[0]];
    float E_i_K = (ce.f32 + ti[3])[ti[2] - 1];
    float E_i1_1 = (ce.f32 + ti1[3])[ti1[0]];
    float E_i1_K = (ce.f32 + ti1[3])[ti1[2] - 1];
    float E_1 = E_i_1 + r * (E_i1_1 - E_i_1);
    float E_K = E_i_K + r * (E_i1_K - E_i_K);

    float r1 = gpu_prn(seed);
    int32_gpu k;
    float c_k, c_k1;
    float E_out = gpu_sample_tabular(
      xe, pe, cc, n_out, n_disc, interp, r1, &k, &c_k, &c_k1);
    if (k >= n_disc && n_out > 1) {
      float El_1 = xe[n_disc];
      float El_K = xe[n_out - 1];
      if (El_K != El_1)
        E_out = E_1 + (E_out - El_1) * (E_K - E_1) / (El_K - El_1);
    }
    out.E_out = E_out;

    // nearest-CDF-neighbour angle table (histogram: always k). c_k1 is the
    // CPU walk's value: INFTY on a discrete hit, stale (== c_k) when the
    // walk exhausts the last bin — which then always picks table k+1.
    int32_gpu kk = k;
    if (!(r1 - c_k < c_k1 - r1 || interp == GPU_INTERP_HISTOGRAM))
      kk = k + 1;
    GLOBAL const int32_gpu* mb = ce.i32 + tl[4 + kk];
    int32_gpu n_mu = mb[0];
    int32_gpu mu_interp = mb[1];
    GLOBAL const float* mv = ce.f32 + mb[2];
    GLOBAL const float* mp = mv + n_mu;
    GLOBAL const float* mc = mv + 2 * n_mu;
    float r2 = gpu_prn(seed);
    int32_gpu km_;
    float ck_, ck1_;
    float mu = gpu_sample_tabular(
      mv, mp, mc, n_mu, 0, mu_interp, r2, &km_, &ck_, &ck1_);
    if (mu > 1.0f)
      mu = 1.0f;
    if (mu < -1.0f)
      mu = -1.0f;
    out.mu = mu;
    return out;
  }
  }
  return out;
}

//! Sample one AngleEnergy distribution (descriptor at `blob`). MULTI
//! (applicability pick) and UNCORR (angle wrapper) redirect into a nested
//! law; MSL forbids recursion, so they are resolved iteratively with the
//! CPU's RN order (MULTI's pick RN, then UNCORR's angle, then the nested
//! law's draws). An UNCORR angle overrides the terminal law's mu, exactly
//! as the CPU wrapper does. Real trees are at most MULTI -> UNCORR ->
//! energy law; the bound is a safety net.
DEVICE_FN GpuSampleEA gpu_sample_dist(
  GpuCeView ce, int32_gpu blob, float E_in, THREAD uint64_gpu* seed)
{
  float mu_uncorr = 1.0f;
  bool has_uncorr_mu = false;
  for (int redirect = 0; redirect < 8; ++redirect) {
    GLOBAL const int32_gpu* h = ce.i32 + blob;
    int32_gpu type = h[0];
    if (type == GPU_DIST_MULTI) {
      // [1] n, then n pairs (applicability_f1d, dist_off)
      int32_gpu n = h[1];
      int32_gpu pick = n - 1;
      if (n > 1) {
        float prob = 0.0f;
        float c = gpu_prn(seed);
        for (int32_gpu i = 0; i < n; ++i) {
          prob += gpu_f1d(ce, h[2 + 2 * i], E_in);
          if (c <= prob) {
            pick = i;
            break;
          }
        }
      }
      blob = h[2 + 2 * pick + 1];
    } else if (type == GPU_DIST_UNCORR) {
      // [1] angle blob or -1, [2] energy dist blob; CPU order: angle first
      if (h[1] >= 0)
        mu_uncorr = gpu_sample_angle_dist(ce, h[1], E_in, seed);
      else
        mu_uncorr = 2.0f * gpu_prn(seed) - 1.0f;
      has_uncorr_mu = true;
      blob = h[2];
    } else {
      break;
    }
  }
  GpuSampleEA out = gpu_sample_dist_terminal(ce, blob, E_in, seed);
  if (has_uncorr_mu)
    out.mu = mu_uncorr;
  return out;
}

// ---------------------------------------------------------------------------
// Free-gas target velocity (constant-XS approximation)
// ---------------------------------------------------------------------------

//! sample_cxs_target_velocity (physics.cpp:1024) — returns target velocity
//! in sqrt(eV) units.
DEVICE_FN GpuVec3 gpu_sample_cxs_target_velocity(
  float awr, float E, GpuVec3 u, float kT, THREAD uint64_gpu* seed)
{
  float beta_vn = sqrtf(awr * E / kT);
  float alpha = 1.0f / (1.0f + 1.7724538509055159f * beta_vn * 0.5f);
  float beta_vt_sq;
  float mu;
  for (;;) {
    float r1 = gpu_prn(seed);
    float r2 = gpu_prn(seed);
    if (gpu_prn(seed) < alpha) {
      beta_vt_sq = -logf(r1 * r2);
    } else {
      float c = cosf(1.5707963267948966f * gpu_prn(seed));
      beta_vt_sq = -logf(r1) - logf(r2) * c * c;
    }
    float beta_vt = sqrtf(beta_vt_sq);
    mu = 2.0f * gpu_prn(seed) - 1.0f;
    float accept = sqrtf(fmaxf(0.0f, beta_vn * beta_vn + beta_vt_sq -
                                       2.0f * beta_vn * beta_vt * mu)) /
                   (beta_vn + beta_vt);
    if (gpu_prn(seed) < accept)
      break;
  }
  float vt = sqrtf(beta_vt_sq * kT / awr);
  return gpu_scale(gpu_rotate_angle(u, mu, seed), vt);
}
