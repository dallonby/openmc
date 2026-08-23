//! \file transport.h
//! The GPU transport loop: a faithful port of
//! transport_history_based_single_particle (simulation.cpp:958) and the
//! Particle::event_* chain (particle.cpp) for multigroup and
//! continuous-energy neutrons (eigenvalue, analog physics).
//! One thread per source particle; fission sites are appended to a device
//! bank with (parent_id, progeny_id) so host-side sort_bank /
//! synchronize_bank reproduce CPU ordering semantics. (n,xn) clones use an
//! in-thread LIFO stack, matching the CPU's local secondary bank.

#pragma once

#ifdef GPU_HOST_DEBUG
struct GpuGeomState;
void gpu_host_debug_lost(THREAD const GpuGeomState* gs, int32_gpu tok,
  GpuVec3 r0, GpuVec3 u0, GpuVec3 u_new);
void gpu_host_debug_lost_where(int where, THREAD const GpuGeomState* gs,
  float E);
void gpu_host_debug_exit(
  THREAD const GpuGeomState* gs, float distance, float d_coll);
void gpu_host_stat_collision(float E, float r);
void gpu_host_stat_elastic(float mu_lab, float eratio);
void gpu_host_stat_inelastic(int32_gpu mt);
void gpu_host_stat_absorb();
void gpu_host_stat_flight(float d);
void gpu_host_trace_fly(int64_gpu id, float d, float dc, float db, float E);
void gpu_host_trace_collide(int64_gpu id, float E, GpuVec3 r, int32_gpu nuc);
void gpu_host_trace_elastic(int64_gpu id, float E);
void gpu_host_trace_inelastic(int64_gpu id, int32_gpu mt);
void gpu_host_trace_postfis(
  int64_gpu id, uint32_gpu seedlo, int32_gpu nprog, uint32_gpu urrlo);
void gpu_host_leak(int64_gpu id);
void gpu_host_stat_fsite(float r);
#endif

// Per-thread array bounds. Defaults serve the host replay build; the runtime
// MSL compile prepends model-specialized values (max nuclides per material,
// n_coord_levels+1, max filters per tally) so per-thread stack/register use
// — and therefore occupancy — tracks the model instead of the worst case.
#ifndef GPU_MAX_MAT_NUCLIDES
#define GPU_MAX_MAT_NUCLIDES 32
#endif
#ifndef GPU_MAX_SECONDARY_STACK
#define GPU_MAX_SECONDARY_STACK 8
#endif

// ------------------------------------------------------------- tally view --
struct GpuTallyView {
  GLOBAL const GpuTallyDesc* tallies;
  GLOBAL const GpuFilterDesc* filters;
  GLOBAL const GpuMesh* meshes;
  GLOBAL const int32_gpu* i32;
  GLOBAL const float* f32;
  GLOBAL gpu_atomic_f32* accum;
  uint32_gpu n_tallies;
};

// ---- Structured-mesh helpers (StructuredMesh semantics: 1-based ijk) ----
DEVICE_FN int32_gpu gpu_mesh_index_dir(
  float r, float ll, float ur, float w, int32_gpu shape)
{
  if (r <= ll)
    return (r == ll) ? 1 : 0;
  if (r >= ur)
    return (r == ur) ? shape : shape + 1;
  return (int32_gpu)ceilf((r - ll) / w);
}

//! lower_bound_index(grid,val)+1 on an explicit ascending grid of `npts`
//! points (StructuredMesh::get_index_in_direction for non-uniform grids):
//! 1-based cell index, 0 below the grid, npts above it.
DEVICE_FN int32_gpu gpu_grid_index(
  GLOBAL const float* g, int32_gpu npts, float v)
{
  // CPU: lower_bound_index(begin,end,v)+1, i.e.
  //   v == g[0]            -> bin 1
  //   g[k-1] < v <= g[k]   -> bin k      (exact interior edge = LOWER bin)
  //   v == g[npts-1]       -> bin npts-1 (last valid bin, not outside)
  if (v < g[0])
    return 0;
  if (v == g[0])
    return 1;
  if (v > g[npts - 1])
    return npts;
  // first index with g[idx] >= v  (std::lower_bound)
  int32_gpu lo = 0, hi = npts - 1;
  while (hi - lo > 1) {
    int32_gpu mid = (lo + hi) / 2;
    if (g[mid] >= v)
      hi = mid;
    else
      lo = mid;
  }
  int32_gpu idx = (g[lo] >= v) ? lo : hi;
  return idx; // (idx - 1) + 1
}

//! cylindrical (r,phi,z) indices in the mesh-local frame
DEVICE_FN bool gpu_mesh_indices_cyl(
  GpuMesh m, GpuVec3 r, GLOBAL const float* f32, THREAD int32_gpu* ijk)
{
  float lx = r.x - m.ox, ly = r.y - m.oy, lz = r.z - m.oz;
  float rr = sqrtf(lx * lx + ly * ly);
  bool in = true;
  ijk[0] = gpu_grid_index(f32 + m.rgrid_off, m.nx + 1, rr);
  if (ijk[0] < 1 || ijk[0] > m.nx)
    in = false;
  ijk[1] = 1;
  if (!(m.full_phi && m.ny == 1)) {
    float phi = (rr < 1.0e-8f) ? 0.0f : atan2f(ly, lx);
    if (phi < 0.0f)
      phi += 6.283185307179586f;
    ijk[1] = gpu_grid_index(f32 + m.phigrid_off, m.ny + 1, phi);
    if (m.full_phi) {
      // only a full 2*pi grid wraps (CPU sanitize_phi)
      if (ijk[1] < 1)
        ijk[1] = m.ny;
      if (ijk[1] > m.ny)
        ijk[1] = 1;
    } else if (ijk[1] < 1 || ijk[1] > m.ny) {
      in = false; // a wedge mesh genuinely ends here
    }
  }
  ijk[2] = gpu_grid_index(f32 + m.zgrid_off, m.nz + 1, lz);
  if (ijk[2] < 1 || ijk[2] > m.nz)
    in = false;
  return in;
}

//! get_indices: 1-based indices; returns in_mesh
DEVICE_FN bool gpu_mesh_indices(
  GpuMesh m, GpuVec3 r, GLOBAL const float* f32, THREAD int32_gpu* ijk)
{
  if (m.kind == GPU_MESH_CYLINDRICAL)
    return gpu_mesh_indices_cyl(m, r, f32, ijk);
  bool in = true;
  ijk[0] = gpu_mesh_index_dir(r.x, m.llx, m.urx, m.wx, m.nx);
  if (ijk[0] < 1 || ijk[0] > m.nx)
    in = false;
  ijk[1] = 1;
  ijk[2] = 1;
  if (m.n_dim >= 2) {
    ijk[1] = gpu_mesh_index_dir(r.y, m.lly, m.ury, m.wy, m.ny);
    if (ijk[1] < 1 || ijk[1] > m.ny)
      in = false;
  }
  if (m.n_dim >= 3) {
    ijk[2] = gpu_mesh_index_dir(r.z, m.llz, m.urz, m.wz, m.nz);
    if (ijk[2] < 1 || ijk[2] > m.nz)
      in = false;
  }
  return in;
}

//! find_r_crossing: distance to radial shell `shell` (local frame), > l,
//! or GPU_INFTY. Mirrors CylindricalMesh::find_r_crossing.
DEVICE_FN float gpu_cyl_r_cross(GpuMesh m, GLOBAL const float* f32, GpuVec3 lr,
  GpuVec3 u, float l, int32_gpu shell)
{
  if (shell < 0 || shell > m.nx)
    return GPU_INFTY;
  float r0 = (f32 + m.rgrid_off)[shell];
  if (r0 == 0.0f)
    return GPU_INFTY;
  float denom = u.x * u.x + u.y * u.y;
  if (denom < 1.0e-12f)
    return GPU_INFTY;
  float inv = 1.0f / denom;
  float p = (u.x * lr.x + u.y * lr.y) * inv;
  float R = sqrtf(lr.x * lr.x + lr.y * lr.y);
  float D = p * p - (R - r0) * (R + r0) * inv;
  if (D < 0.0f)
    return GPU_INFTY;
  D = sqrtf(D);
  if (fabsf(R - r0) <= 1.0e-6f * (1.0f + fabsf(r0)))
    return GPU_INFTY;
  if (-p - D > l)
    return -p - D;
  if (-p + D > l)
    return -p + D;
  return GPU_INFTY;
}

//! find_phi_crossing (local frame). Mirrors CylindricalMesh::find_phi_crossing.
DEVICE_FN float gpu_cyl_phi_cross(GpuMesh m, GLOBAL const float* f32,
  GpuVec3 lr, GpuVec3 u, float l, int32_gpu shell)
{
  if (m.full_phi && m.ny == 1)
    return GPU_INFTY;
  // phi grid has ny+1 points [0..ny]; CPU indexes grid_[1][shell] after
  // sanitize (shell wraps into [0,ny])
  int32_gpu idx = shell;
  if (idx < 0)
    idx = m.ny;
  if (idx > m.ny)
    idx = 0;
  float p0 = (f32 + m.phigrid_off)[idx];
  float c0 = cosf(p0), s0 = sinf(p0);
  float denom = (u.x * s0 - u.y * c0);
  if (fabsf(denom) > 1.0e-8f) {
    float sdist = -(lr.x * s0 - lr.y * c0) / denom;
    if ((sdist > l) &&
        ((c0 * (lr.x + sdist * u.x) + s0 * (lr.y + sdist * u.y)) > 0.0f))
      return sdist;
  }
  return GPU_INFTY;
}

//! cylindrical distance to next grid boundary in axis k (from local start),
//! writing the next index. l = distance already traveled along the track.
DEVICE_FN float gpu_mesh_dist_cyl(GpuMesh m, GLOBAL const float* f32,
  THREAD const int32_gpu* ijk, int k, GpuVec3 r0, GpuVec3 u, float l,
  THREAD int32_gpu* next)
{
  GpuVec3 lr = gpu_v3(r0.x - m.ox, r0.y - m.oy, r0.z - m.oz);
  if (k == 0) {
    float d_out = gpu_cyl_r_cross(m, f32, lr, u, l, ijk[0]);
    float d_in = gpu_cyl_r_cross(m, f32, lr, u, l, ijk[0] - 1);
    if (d_in < d_out) {
      *next = ijk[0] - 1;
      return d_in;
    }
    *next = ijk[0] + 1;
    return d_out;
  } else if (k == 1) {
    float d_up = gpu_cyl_phi_cross(m, f32, lr, u, l, ijk[1]);
    float d_dn = gpu_cyl_phi_cross(m, f32, lr, u, l, ijk[1] - 1);
    if (d_dn < d_up) {
      int32_gpu nx = ijk[1] - 1;
      *next = (nx < 1 && m.full_phi) ? m.ny : nx;
      return d_dn;
    }
    int32_gpu nx = ijk[1] + 1;
    *next = (nx > m.ny && m.full_phi) ? 1 : nx;
    return d_up;
  } else {
    *next = ijk[2];
    if (fabsf(u.z) < 1.0e-8f)
      return GPU_INFTY;
    GLOBAL const float* zg = f32 + m.zgrid_off;
    if (u.z > 0.0f && ijk[2] <= m.nz) {
      *next = ijk[2] + 1;
      return (zg[ijk[2]] - lr.z) / u.z;
    } else if (u.z < 0.0f && ijk[2] > 0) {
      *next = ijk[2] - 1;
      return (zg[ijk[2] - 1] - lr.z) / u.z;
    }
    return GPU_INFTY;
  }
}

DEVICE_FN int32_gpu gpu_mesh_bin(GpuMesh m, THREAD const int32_gpu* ijk)
{
  if (m.n_dim == 1)
    return ijk[0] - 1;
  if (m.n_dim == 2)
    return (ijk[1] - 1) * m.nx + ijk[0] - 1;
  return ((ijk[2] - 1) * m.ny + (ijk[1] - 1)) * m.nx + ijk[0] - 1;
}

DEVICE_FN float gpu_mesh_ll(GpuMesh m, int k)
{
  return k == 0 ? m.llx : (k == 1 ? m.lly : m.llz);
}
DEVICE_FN float gpu_mesh_w(GpuMesh m, int k)
{
  return k == 0 ? m.wx : (k == 1 ? m.wy : m.wz);
}
DEVICE_FN int32_gpu gpu_mesh_shape(GpuMesh m, int k)
{
  return k == 0 ? m.nx : (k == 1 ? m.ny : m.nz);
}
DEVICE_FN float gpu_vec_comp(GpuVec3 v, int k)
{
  return k == 0 ? v.x : (k == 1 ? v.y : v.z);
}

//! distance_to_grid_boundary (distance measured from the track start r0, as
//! on the CPU); returns distance, writes next index. l = already-traveled.
DEVICE_FN float gpu_mesh_dist(GpuMesh m, GLOBAL const float* f32,
  THREAD const int32_gpu* ijk, int k, GpuVec3 r0, GpuVec3 u, float l,
  THREAD int32_gpu* next)
{
  if (m.kind == GPU_MESH_CYLINDRICAL)
    return gpu_mesh_dist_cyl(m, f32, ijk, k, r0, u, l, next);
  *next = ijk[k];
  float uk = gpu_vec_comp(u, k);
  if (uk == 0.0f)
    return GPU_INFTY;
  float ll = gpu_mesh_ll(m, k), w = gpu_mesh_w(m, k);
  int32_gpu sh = gpu_mesh_shape(m, k);
  if (uk > 0.0f) {
    if (ijk[k] <= sh) {
      *next = ijk[k] + 1;
      return (ll + (float)ijk[k] * w - gpu_vec_comp(r0, k)) / uk;
    }
  } else if (ijk[k] >= 1) {
    *next = ijk[k] - 1;
    return (ll + (float)(ijk[k] - 1) * w - gpu_vec_comp(r0, k)) / uk;
  }
  return GPU_INFTY;
}

//! Match one filter. Writes every matching bin (cell/universe filters can
//! match at several coordinate levels — CPU get_all_bins pushes them all
//! and FilterBinIter scores the cartesian product) and returns the count;
//! 0 means the filter missed. `bins` must hold GPU_MAX_COORD entries.
DEVICE_FN int32_gpu gpu_filter_match(GpuTallyView tv, GpuFilterDesc f,
  THREAD const GpuGeomState* gs, float E, THREAD int32_gpu* bins)
{
  switch (f.type) {
  case GPU_FILTER_CELL: {
    int32_gpu cnt = 0;
    for (int32_gpu j = 0; j < gs->n_coord; ++j) {
      int32_gpu b = tv.i32[f.map_off + gs->coord[j].cell];
      if (b >= 0)
        bins[cnt++] = b;
    }
    return cnt;
  }
  case GPU_FILTER_MATERIAL: {
    if (gs->material < 0)
      return 0;
    int32_gpu b = tv.i32[f.map_off + gs->material];
    if (b < 0)
      return 0;
    bins[0] = b;
    return 1;
  }
  case GPU_FILTER_UNIVERSE: {
    int32_gpu cnt = 0;
    for (int32_gpu j = 0; j < gs->n_coord; ++j) {
      int32_gpu b = tv.i32[f.map_off + gs->coord[j].universe];
      if (b >= 0)
        bins[cnt++] = b;
    }
    return cnt;
  }
  case GPU_FILTER_ENERGY: {
    GLOBAL const float* edges = tv.f32 + f.map_off;
    uint32_gpu n = f.n_bins;
    if (E < edges[0] || E > edges[n])
      return 0;
    uint32_gpu lo = 0, hi = n;
    while (hi - lo > 1) {
      uint32_gpu mid = (lo + hi) / 2;
      if (E >= edges[mid])
        lo = mid;
      else
        hi = mid;
    }
    bins[0] = (int32_gpu)lo;
    return 1;
  }
  case GPU_FILTER_MESH: {
    // point sample (collision estimator; MeshFilter::get_all_bins ->
    // get_bin). Tracklength mesh tallies are scored by track splitting in
    // gpu_score_tallies instead.
    GpuMesh mesh = tv.meshes[f.mesh];
    int32_gpu ijk[3];
    if (!gpu_mesh_indices(mesh, gs->coord[0].r, tv.f32, ijk))
      return 0;
    bins[0] = gpu_mesh_bin(mesh, ijk);
    return 1;
  }
  }
  return 0;
}

//! Score one (coeff, filter-combination set) into a tally: every
//! combination of the per-filter match lists (CPU FilterBinIter).
DEVICE_FN void gpu_score_combos(GpuTallyView tv, GpuTallyDesc td,
  THREAD const int32_gpu* fbins, THREAD const int32_gpu* fcnt, float coeff,
  GpuMacroXS xs, float scat, float elastic)
{
  int32_gpu idx[GPU_MAX_TALLY_FILTERS];
  for (uint32_gpu fi = 0; fi < td.n_filters; ++fi)
    idx[fi] = 0;
  for (;;) {
    int32_gpu flat = 0;
    for (uint32_gpu fi = 0; fi < td.n_filters; ++fi) {
      GpuFilterDesc f = tv.filters[td.filter_off + fi];
      flat += fbins[fi * GPU_MAX_COORD + idx[fi]] * (int32_gpu)f.stride;
    }
    for (uint32_gpu s = 0; s < td.n_scores; ++s) {
      int32_gpu code = tv.i32[td.score_off + s];
      float val;
      switch (code) {
      case GPU_SCORE_FLUX:
        val = coeff;
        break;
      case GPU_SCORE_TOTAL:
        val = coeff * xs.total;
        break;
      case GPU_SCORE_ABSORPTION:
        val = coeff * xs.absorption;
        break;
      case GPU_SCORE_FISSION:
        val = coeff * xs.fission;
        break;
      case GPU_SCORE_NU_FISSION:
        val = coeff * xs.nu_fission;
        break;
      case GPU_SCORE_SCATTER:
        val = coeff * scat;
        break;
      case GPU_SCORE_ELASTIC:
        val = coeff * elastic;
        break;
      default:
        val = 0.0f;
        break;
      }
      if (val != 0.0f) {
        gpu_atomic_add_f(
          tv.accum + td.accum_off + (uint32_gpu)flat * td.n_scores + s, val);
      }
    }
    // odometer over the per-filter match lists
    int32_gpu fi = (int32_gpu)td.n_filters - 1;
    for (; fi >= 0; --fi) {
      if (++idx[fi] < fcnt[fi])
        break;
      idx[fi] = 0;
    }
    if (fi < 0)
      break;
  }
}

//! Score all tallies of the given estimator for one event.
//! coeff: tracklength -> wgt * distance; collision -> wgt / Sigma_t.
//! r0/u/dist describe the flight (tracklength) for mesh track splitting.
//! Macroscopic score values are supplied by the caller (mode-specific).
DEVICE_FN void gpu_score_tallies(GpuTallyView tv, THREAD const GpuGeomState* gs,
  uint32_gpu estimator, float coeff, float E, GpuMacroXS xs, float scat,
  float elastic, GpuVec3 r0, GpuVec3 u, float dist)
{
  for (uint32_gpu t = 0; t < tv.n_tallies; ++t) {
    GpuTallyDesc td = tv.tallies[t];
    if (td.estimator != estimator)
      continue;
    int32_gpu fbins[GPU_MAX_TALLY_FILTERS * GPU_MAX_COORD];
    int32_gpu fcnt[GPU_MAX_TALLY_FILTERS];
    int32_gpu mesh_fi = -1;
    bool miss = false;
    for (uint32_gpu fi = 0; fi < td.n_filters; ++fi) {
      GpuFilterDesc f = tv.filters[td.filter_off + fi];
      if (f.type == GPU_FILTER_MESH && estimator == GPU_ESTIMATOR_TRACKLENGTH) {
        mesh_fi = (int32_gpu)fi; // filled per track segment below
        fcnt[fi] = 1;
        continue;
      }
      fcnt[fi] = gpu_filter_match(tv, f, gs, E, fbins + fi * GPU_MAX_COORD);
      if (fcnt[fi] == 0) {
        miss = true;
        break;
      }
    }
    if (miss)
      continue;
    if (mesh_fi < 0) {
      gpu_score_combos(tv, td, fbins, fcnt, coeff, xs, scat, elastic);
      continue;
    }

    // ---- tracklength mesh filter: StructuredMesh::raytrace_mesh port ----
    GpuMesh m = tv.meshes[tv.filters[td.filter_off + mesh_fi].mesh];
    float total = dist;
    if (total <= 0.0f)
      continue;
    int32_gpu ijk[3];
    bool in_mesh = gpu_mesh_indices(
      m, gpu_add(r0, gpu_scale(u, GPU_TINY_BIT)), tv.f32, ijk);
    if (total < 2.0f * GPU_TINY_BIT) {
      if (in_mesh) {
        fbins[mesh_fi * GPU_MAX_COORD] = gpu_mesh_bin(m, ijk);
        gpu_score_combos(tv, td, fbins, fcnt, coeff, xs, scat, elastic);
      }
      continue;
    }
    float dk[3];
    int32_gpu nk[3];
    int nd = m.n_dim;
    for (int k = 0; k < nd; ++k)
      dk[k] = gpu_mesh_dist(m, tv.f32, ijk, k, r0, u, 0.0f, &nk[k]);
    float traveled = 0.0f;
    int32_gpu guard = 4 * (m.nx + m.ny + m.nz) + 16;
    while (guard-- > 0) {
      if (in_mesh) {
        int kmin = 0;
        for (int k = 1; k < nd; ++k)
          if (dk[k] < dk[kmin])
            kmin = k;
        float seg_end = fminf(dk[kmin], total);
        float frac = (seg_end - traveled) / total;
        if (frac > 0.0f) {
          fbins[mesh_fi * GPU_MAX_COORD] = gpu_mesh_bin(m, ijk);
          gpu_score_combos(
            tv, td, fbins, fcnt, coeff * frac, xs, scat, elastic);
        }
        traveled = dk[kmin];
        if (traveled >= total)
          break;
        ijk[kmin] = nk[kmin];
        dk[kmin] = gpu_mesh_dist(m, tv.f32, ijk, kmin, r0, u, traveled, &nk[kmin]);
        in_mesh = (ijk[kmin] >= 1 && ijk[kmin] <= gpu_mesh_shape(m, kmin));
      } else {
        int kmax = -1;
        for (int k = 0; k < nd; ++k) {
          if ((ijk[k] < 1 || ijk[k] > gpu_mesh_shape(m, k)) &&
              dk[k] > traveled) {
            traveled = dk[k];
            kmax = k;
          }
        }
        if (kmax == -1)
          traveled += GPU_TINY_BIT;
        if (traveled >= total)
          break;
        in_mesh = gpu_mesh_indices(
          m, gpu_add(r0, gpu_scale(u, traveled + GPU_TINY_BIT)), tv.f32, ijk);
        for (int k = 0; k < nd; ++k)
          dk[k] = gpu_mesh_dist(m, tv.f32, ijk, k, r0, u, traveled, &nk[k]);
      }
    }
  }
}

//! MG tally-score cross sections. CPU scores SCORE_TOTAL / SCORE_ABSORPTION
//! from the density_mult-scaled macro cache but SCORE_FISSION /
//! SCORE_NU_FISSION / SCORE_SCATTER through Mgxs::get_xs, which does NOT
//! apply density_mult — an upstream asymmetry, mirrored here for parity
//! (see PORT_NOTES.md).
DEVICE_FN GpuMacroXS gpu_mg_score_xs(
  GpuMgView mg, int32_gpu i_mat, int32_gpu g, GpuMacroXS xs)
{
  GpuMgMat m = mg.mats[i_mat];
  GpuMacroXS s = xs;
  s.fission = gpu_mg_vec(mg, m, GPU_MGV_FISSION, g);
  s.nu_fission =
    m.fissionable ? gpu_mg_vec(mg, m, GPU_MGV_NU_FISSION, g) : 0.0f;
  return s;
}

// -------------------------------------------------------------- transport --

struct GpuBanks {
  GLOBAL const GpuSourceSite* source;
  //! eigenvalue: fission bank. fixed-source: the spill bank that holds
  //! secondaries (weight-window splits, (n,xn) clones) which did not fit in
  //! the per-thread stack; the host re-dispatches them until it drains.
  GLOBAL GpuSourceSite* fission;
  GLOBAL gpu_atomic_u32* counters;
  GLOBAL uint32_gpu* progeny;
  GLOBAL float* red_slots; // [n_particles][GPU_RED_WIDTH], no atomics
  GLOBAL GpuTraceRec* trace;        // debug event trace (id-gated)
};

#define GPU_TRACE_MAX 4096

// in-thread record for (n,xn) clones (CPU: local_secondary_bank, LIFO)
struct GpuSecondary {
  float r[3];
  float u[3];
  float E;
  float wgt;
  float time;
};

//! neutron speed in cm/s (Particle::speed, relativistic). Uses the CPU's
//! cancellation-free form C*sqrt(E(E+2m))/(E+m): the 1-(m/(E+m))^2 form
//! collapses to exactly 0 below ~32 eV in fp32 (ulp(m) is 64 eV), which
//! made every thermal flight time infinite.
DEVICE_FN float gpu_neutron_speed(float E)
{
  float m = 939.56542052e6f;
  return 2.99792458e10f * sqrtf(E * (E + 2.0f * m)) / (E + m);
}

//! fp32 relocation rescue: escalating nudges along the current direction
//! (1x, 8x, 64x TINY_BIT — at most ~7e-4 cm total) with a root re-search
//! after each. A crossing at a corner can sit inside the fp32 sign band of
//! a second, coincident surface, where a single TINY_BIT does not clear
//! the band and the containment test keeps flipping.
DEVICE_FN bool gpu_rescue_find_cell(
  GpuGeomData geom, THREAD GpuGeomState* gs, int32_gpu root, int32_gpu levels)
{
  float step = GPU_TINY_BIT;
  for (int s = 0; s < 3; ++s) {
    gpu_move_distance(gs, step);
    if (gpu_exhaustive_find_cell(geom, gs, root, levels))
      return true;
    step *= 8.0f;
  }
  return false;
}

//! total delayed nu for a CE nuclide (sum of per-group yields of the
//! first fission reaction, mirroring Nuclide::nu)
DEVICE_FN float gpu_ce_nu_delayed(GpuCeView ce, GpuNuclide nuc, float E)
{
  int32_gpu rec0 = ce.i32[nuc.fis_rx_off];
  float nu = 0.0f;
  for (uint32_gpu g = 1; g <= nuc.n_delayed; ++g) {
    int32_gpu y = ce.i32[rec0 + 2 + 3 * g];
    nu += gpu_f1d(ce, y, E);
  }
  return nu;
}

// ---------------------------------------------------------------------
// Variance reduction (weight windows + survival biasing). Fixed-source,
// non-multiplying models only — flatten rejects anything else, so there is
// no interaction with fission-bank progeny bookkeeping.
// ---------------------------------------------------------------------

#define GPU_WW_REL_TOL 1.0e-11f

struct GpuVrState {
  float wgt_born;
  float wgt_ww_born;
  float ww_factor;
  int32_gpu n_split;
};

//! WeightWindows::get_weight_window — returns false when the particle is
//! outside the mesh / energy range or the window is invalid (lower <= 0).
DEVICE_FN bool gpu_ww_lookup(GCONST GpuControl& ctl, GpuTallyView tv, float E,
  GpuVec3 r, THREAD float* lower, THREAD float* upper)
{
  if (!ctl.ww_on)
    return false;
  int32_gpu e_bin = 0;
  if (ctl.ww_n_energy > 1) {
    GLOBAL const float* eb = tv.f32 + ctl.ww_ebounds_off;
    if (E < eb[0] || E > eb[ctl.ww_n_energy])
      return false;
    uint32_gpu lo = 0, hi = ctl.ww_n_energy;
    while (hi - lo > 1) {
      uint32_gpu mid = (lo + hi) / 2;
      if (E >= eb[mid])
        lo = mid;
      else
        hi = mid;
    }
    e_bin = (int32_gpu)lo;
  }
  GpuMesh m = tv.meshes[ctl.ww_mesh];
  int32_gpu ijk[3];
  if (!gpu_mesh_indices(m, r, tv.f32, ijk))
    return false;
  int32_gpu bin = gpu_mesh_bin(m, ijk);
  if (bin < 0 || bin >= (int32_gpu)ctl.ww_n_mesh_bins)
    return false;
  uint32_gpu idx = (uint32_gpu)e_bin * ctl.ww_n_mesh_bins + (uint32_gpu)bin;
  float lw = tv.f32[ctl.ww_lower_off + idx];
  if (!(lw > 0.0f)) // invalid / game-off cell
    return false;
  *lower = lw;
  *upper = tv.f32[ctl.ww_upper_off + idx];
  return true;
}

//! Push one secondary: per-thread stack first, then the global spill bank.
//! Returns false when neither had room (the caller must then conserve
//! weight itself — see gpu_apply_weight_window).
DEVICE_FN bool gpu_push_secondary(GCONST GpuControl& ctl, GpuBanks banks,
  THREAD GpuSecondary* sec_stack, THREAD int32_gpu* n_stack,
  THREAD const GpuVrState* vr, GpuVec3 r, GpuVec3 u, float E, float wgt,
  float time, THREAD const uint64_gpu* seeds)
{
  if (*n_stack < GPU_MAX_SECONDARY_STACK) {
    THREAD GpuSecondary* sc = &sec_stack[(*n_stack)++];
    sc->r[0] = r.x;
    sc->r[1] = r.y;
    sc->r[2] = r.z;
    sc->u[0] = u.x;
    sc->u[1] = u.y;
    sc->u[2] = u.z;
    sc->E = E;
    sc->wgt = wgt;
    sc->time = time;
    return true;
  }
  uint32_gpu idx = gpu_atomic_add_u32(banks.counters + GPU_CTR_SPILL, 1u);
  if (idx >= ctl.spill_cap) {
    gpu_atomic_add_u32(banks.counters + GPU_CTR_SPILL_DROP, 1u);
    return false;
  }
  GpuSourceSite site;
  site.r[0] = r.x;
  site.r[1] = r.y;
  site.r[2] = r.z;
  site.u[0] = u.x;
  site.u[1] = u.y;
  site.u[2] = u.z;
  site.E = E;
  site.time = time;
  site.wgt = wgt;
  site.delayed_group = 0;
  site.parent_id = 0;
  site.progeny_id = 0;
  site.wgt_born = vr->wgt_born;
  site.wgt_ww_born = vr->wgt_ww_born;
  site.ww_factor = vr->ww_factor;
  site.n_split = vr->n_split;
  // carry the parent's stream so the child continues it rather than
  // replaying some primary particle's sequence
  // unique id from a monotonic counter, in an id space disjoint from the
  // primaries', so siblings get independent streams (copying the parent's
  // state would make them identical particles)
  uint64_gpu serial =
    (uint64_gpu)gpu_atomic_add_u32(banks.counters + GPU_CTR_SPILL_SERIAL, 1u);
  uint64_gpu base = ((uint64_gpu)ctl.spill_uid_hi << 32) |
                    (uint64_gpu)ctl.spill_uid_lo;
  uint64_gpu uid = base + serial + 1;
  site.uid_lo = (uint32_gpu)(uid & 0xffffffffu);
  site.uid_hi = (uint32_gpu)(uid >> 32);
  banks.fission[idx] = site;
  return true;
}

//! russian_roulette (physics_common.cpp)
DEVICE_FN void gpu_russian_roulette(
  THREAD float* wgt, float weight_survive, THREAD uint64_gpu* seed)
{
  if (weight_survive * gpu_prn(seed) < *wgt)
    *wgt = weight_survive;
  else
    *wgt = 0.0f;
}

//! apply_weight_window (weight_windows.cpp): split above the window,
//! roulette below it. Splits go to the stack/spill bank.
DEVICE_FN void gpu_apply_weight_window(GCONST GpuControl& ctl, GpuTallyView tv,
  GpuBanks banks, THREAD GpuSecondary* sec_stack, THREAD int32_gpu* n_stack,
  THREAD GpuVrState* vr, GpuVec3 r, GpuVec3 u, float E, float time,
  THREAD float* wgt, THREAD uint64_gpu* seeds)
{
  if (*wgt <= 0.0f || E <= 0.0f)
    return;
  float lower, upper;
  if (!gpu_ww_lookup(ctl, tv, E, r, &lower, &upper)) {
    if (vr->wgt_ww_born == -1.0f)
      vr->wgt_ww_born = 1.0f;
    return;
  }
  if (vr->wgt_ww_born == -1.0f)
    vr->wgt_ww_born = 0.5f * (lower + upper);
  // normalize by the window the history was born in
  float scale = vr->wgt_born / vr->wgt_ww_born;
  lower *= scale;
  upper *= scale;
  float survival = lower * ctl.ww_survival_ratio;
  float weight = *wgt;

  if (weight < ctl.ww_weight_cutoff) {
    *wgt = 0.0f;
    return;
  }
  if (vr->ww_factor == 0.0f && ctl.ww_max_lb_ratio > 1.0f &&
      weight > lower * ctl.ww_max_lb_ratio)
    vr->ww_factor = weight / (lower * ctl.ww_max_lb_ratio);
  if (vr->ww_factor > 1.0f) {
    lower *= vr->ww_factor;
    upper *= vr->ww_factor;
    survival *= vr->ww_factor;
  }

  if (weight > upper * (1.0f + GPU_WW_REL_TOL)) {
    if (vr->n_split >= ctl.ww_max_history_splits)
      return;
    float n_split = ceilf(weight / ((1.0f + GPU_WW_REL_TOL) * upper));
    if (n_split < 2.0f)
      n_split = 2.0f;
    if (n_split > (float)ctl.ww_max_split)
      n_split = (float)ctl.ww_max_split;
    vr->n_split += (int32_gpu)n_split;
    float w_each = weight / n_split;
    int32_gpu i_split = (int32_gpu)(n_split + 0.5f);
    // Splitting is optional: if the stack and spill bank are both full we
    // simply make FEWER copies and the parent keeps the weight the missing
    // copies would have carried. Total weight is conserved exactly, so the
    // estimator stays unbiased — only the variance reduction is weaker.
    // (Dropping an already-divided copy, by contrast, destroys weight and
    // biases the tally low; that is what this avoids.)
    int32_gpu made = 0;
    for (int32_gpu l = 0; l < i_split - 1; ++l) {
      if (gpu_push_secondary(
            ctl, banks, sec_stack, n_stack, vr, r, u, E, w_each, time, seeds))
        ++made;
    }
    *wgt = weight - (float)made * w_each;
  } else if (weight < lower * (1.0f - GPU_WW_REL_TOL)) {
    float ws = weight * (float)ctl.ww_max_split;
    if (survival < ws)
      ws = survival;
    gpu_russian_roulette(wgt, ws, &seeds[GPU_STREAM_TRACKING]);
  }
}

//! Full history for one source particle. Mirrors the CPU event loop.
DEVICE_FN void gpu_run_particle(uint32_gpu tid, GCONST GpuControl& ctl,
  GpuGeomData geom, GpuMgView mg, GpuCeView ce, GpuSabView sab,
  GpuTallyView tv, GpuBanks banks)
{
  // ---- initialize_history ----
  GpuSourceSite src = banks.source[ctl.source_offset + tid];
  int64_gpu index_source = (int64_gpu)(ctl.source_offset + tid) + 1;

  uint64_gpu seeds[GPU_N_STREAMS];
  if (ctl.source_is_spill) {
    // spilled secondary: seed from its own unique id (see GpuSourceSite)
    gpu_init_particle_seeds(
      (int64_gpu)(((uint64_gpu)src.uid_hi << 32) | (uint64_gpu)src.uid_lo),
      ctl.master_seed, ctl.prn_stride, seeds);
  } else {
    gpu_init_particle_seeds((int64_gpu)ctl.seed_base + index_source,
      ctl.master_seed, ctl.prn_stride, seeds);
  }
  int stream = GPU_STREAM_TRACKING;

#if GPU_DIAG_BALLAST > 0
  // diagnostic: adds a known quantity of live private state, held across the
  // whole event loop, to locate this kernel on the occupancy-vs-private-state
  // curve. A synthetic Metal benchmark shows gather throughput falling 3.3x
  // between ~128 B and 2 KB of live array; this says where we already sit.
  float ballast[GPU_DIAG_BALLAST];
  for (int i = 0; i < GPU_DIAG_BALLAST; ++i)
    ballast[i] = (float)(tid + (uint32_gpu)i);
#endif

  GpuGeomState gs;
  gs.n_coord = 1;
  gs.surface = GPU_SURFACE_NONE;
  gs.material = GPU_MATERIAL_VOID;
  gs.sqrtkT = 0.0f;
  gs.density_mult = 1.0f;
  gs.lost = 0;
  for (int i = 0; i < GPU_MAX_COORD; ++i)
    gpu_coord_reset(&gs.coord[i]);
  gs.coord[0].r = gpu_v3(src.r[0], src.r[1], src.r[2]);
  gs.coord[0].u = gpu_v3(src.u[0], src.u[1], src.u[2]);
  gs.coord[0].universe = ctl.root_universe;

  bool is_ce = (ctl.energy_mode == GPU_MODE_CE);
  float wgt = src.wgt;
  float time = src.time;
  int32_gpu g = 0;
  float E;
  if (is_ce) {
    E = src.E;
  } else {
    g = (int32_gpu)src.E;
    E = geom.f32[ctl.mg_bin_avg_off + g];
  }

  // variance-reduction state (inherited across a spill boundary)
  GpuVrState vr;
  vr.wgt_born = (src.wgt_born > 0.0f) ? src.wgt_born : wgt;
  vr.wgt_ww_born = src.wgt_ww_born;
  vr.ww_factor = src.ww_factor;
  vr.n_split = src.n_split;

  float k_tl = 0.0f, k_col = 0.0f, k_abs = 0.0f, k_leak = 0.0f;
  int32_gpu n_progeny = 0;
  uint32_gpu n_events = 0;
  uint32_gpu leaked = 0;

  GpuSecondary sec_stack[GPU_MAX_SECONDARY_STACK];
  int32_gpu n_stack = 0;

  // per-material micro cache (CE)
  GpuMicroXS micros[GPU_MAX_MAT_NUCLIDES];
  // macro/micro XS cache: CPU skips recalculation when material, energy and
  // density multiplier are unchanged (a surface crossing changes none of
  // them), so a flight that ends at a boundary costs no XS lookups
  GpuMacroXS xs;
  xs.total = 0.0f;
  xs.absorption = 0.0f;
  xs.fission = 0.0f;
  xs.nu_fission = 0.0f;
  GpuMaterial mat;
  mat.n_nuclides = 0;
  int32_gpu xs_key_mat = -2;
  float xs_key_E = -1.0f;
  float xs_key_dm = 0.0f;

  // birth weight-window checkpoint (CPU applies apply_weight_windows in
  // initialize_particle_track, which is what fixes wgt_ww_born for the whole
  // history; doing it later normalizes against the wrong window)
  if (ctl.ww_on && wgt > 0.0f)
    gpu_apply_weight_window(ctl, tv, banks, sec_stack, &n_stack, &vr,
      gs.coord[0].r, gs.coord[0].u, E, time, &wgt, seeds);

  bool found =
    gpu_exhaustive_find_cell(geom, &gs, ctl.root_universe, ctl.n_coord_levels);
  if (!found) {
    // fp32 source positions can sit epsilon-on/outside a surface (fission
    // sites are banked at collision points, which can lie exactly on one):
    // escalate nudges along the flight direction, then against it (sites
    // born at a wall with outward-pointing directions need the latter)
    GpuVec3 r_born = gs.coord[0].r;
    found =
      gpu_rescue_find_cell(geom, &gs, ctl.root_universe, ctl.n_coord_levels);
    if (!found) {
      gs.coord[0].r = r_born;
      float step = -GPU_TINY_BIT;
      for (int s = 0; s < 3 && !found; ++s) {
        gpu_move_distance(&gs, step);
        found = gpu_exhaustive_find_cell(
          geom, &gs, ctl.root_universe, ctl.n_coord_levels);
        step *= 8.0f;
      }
    }
  }
  if (!found) {
#ifdef GPU_HOST_DEBUG
    gpu_host_debug_lost_where(1, &gs, E);
#endif
    gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
    gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST_INIT, 1u);
    wgt = 0.0f;
  }

  for (;;) {
    // ---- event loop ----
    while (wgt > 0.0f) {
      // event_calculate_xs (cached, see above)
      float mac_scat = 0.0f;
      float mac_elastic = 0.0f;
      if (gs.material != xs_key_mat || E != xs_key_E ||
          gs.density_mult != xs_key_dm) {
        xs_key_mat = gs.material;
        xs_key_E = E;
        xs_key_dm = gs.density_mult;
        xs.total = 0.0f;
        xs.absorption = 0.0f;
        xs.fission = 0.0f;
        xs.nu_fission = 0.0f;
        mat.n_nuclides = 0;
      if (gs.material != GPU_MATERIAL_VOID) {
        if (is_ce) {
          mat = geom.materials[gs.material];
          int32_gpu i_log =
            (int32_gpu)(logf(E / ce.energy_min) / ce.log_spacing);
          if (i_log < 0)
            i_log = 0;
          if (i_log >= (int32_gpu)ce.n_log_bins)
            i_log = (int32_gpu)ce.n_log_bins - 1;
          for (uint32_gpu i = 0; i < mat.n_nuclides; ++i) {
            int32_gpu in = geom.i32[mat.nuclide_off + i];
            int32_gpu i_sab =
              (mat.sab_off >= 0) ? geom.i32[mat.sab_off + i] : -1;
            float sfrac = 0.0f;
            if (i_sab >= 0) {
              // material.cpp:854: the table only applies below its cutoff
              if (E > sab.tables[i_sab].energy_max)
                i_sab = -1;
              else
                sfrac = geom.f32[mat.sab_frac_off + i];
            }
            micros[i] = gpu_ce_micro_xs(
              ce, ce.nuclides[in], E, i_log, i_sab, sfrac, sab, seeds);
            float dens = geom.f32[mat.density_off + i] * gs.density_mult;
            xs.total += dens * micros[i].total;
            xs.absorption += dens * micros[i].absorption;
            xs.fission += dens * micros[i].fission;
            xs.nu_fission += dens * micros[i].nu_fission;
          }
        } else {
          xs = gpu_mg_calculate_xs(mg, gs.material, g, gs.density_mult);
        }
      }
      }

      // event_advance
      gs.boundary = gpu_distance_to_boundary(geom, &gs);
      if (gs.lost) {
#ifdef GPU_HOST_DEBUG
        gpu_host_debug_lost_where(2, &gs, E);
#endif
        gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
        gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST_ADVANCE, 1u);
        wgt = 0.0f;
        break;
      }
      float d_coll = (xs.total <= 0.0f)
                       ? GPU_INFTY
                       : -logf(gpu_prn(&seeds[stream])) / xs.total;
      float distance = fminf(gs.boundary.d, d_coll);
      if (distance >= GPU_INFTY) {
        // grazing tangency can hide every surface in fp32: nudge and
        // re-resolve (bounded by the event cap)
        gs.surface = GPU_SURFACE_NONE;
        if (!gpu_rescue_find_cell(
              geom, &gs, ctl.root_universe, ctl.n_coord_levels)) {
#ifdef GPU_HOST_DEBUG
          gpu_host_debug_lost_where(3, &gs, E);
#endif
          gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
          gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST_ADVANCE, 1u);
          wgt = 0.0f;
          break;
        }
        ++n_events;
        if (n_events >= ctl.max_events) {
          gpu_atomic_add_u32(banks.counters + GPU_CTR_MAX_EVENT_HIT, 1u);
          wgt = 0.0f;
          break;
        }
        continue;
      }
#ifdef GPU_HOST_DEBUG
      gpu_host_stat_flight(distance);
      gpu_host_trace_fly(index_source, distance,
        (float)(seeds[stream] & 0xffffffu), gs.boundary.d, E);
#endif
      if ((int32_gpu)index_source == ctl.trace_id) {
        uint32_gpu ti = gpu_atomic_add_u32(banks.counters + GPU_CTR_TRACE, 1u);
        if (ti < GPU_TRACE_MAX) {
          banks.trace[ti].code = 0.0f;
          banks.trace[ti].a = distance;
          banks.trace[ti].b = (float)(seeds[stream] & 0xffffffu);
          banks.trace[ti].c = E;
        }
      }
      GpuVec3 r_flight0 = gs.coord[0].r; // track start (mesh tallies)
      gpu_move_distance(&gs, distance);
      if (is_ce) {
        time += distance / gpu_neutron_speed(E);
      } else {
        // Particle::speed(): the default inverse velocity covers void and
        // any material entry that is not positive (mgxs.cpp get_xs)
        float iv = geom.f32[ctl.mg_default_iv_off + g];
        if (gs.material != GPU_MATERIAL_VOID) {
          GpuMgMat m = mg.mats[gs.material];
          float miv = gpu_mg_vec(mg, m, GPU_MGV_INV_VELOCITY, g);
          if (miv > 0.0f)
            iv = miv;
        }
        time += distance * iv;
      }
      // tracklength keff estimator
      k_tl += wgt * distance * xs.nu_fission;
      // tracklength tallies
      if (tv.n_tallies > 0) {
        if (gs.material != GPU_MATERIAL_VOID) {
          if (is_ce) {
            if (ctl.need_scatter)
              mac_scat = xs.total - xs.absorption;
            // the elastic macro XS needs a per-nuclide pass over the whole
            // material on every flight; skip it entirely unless a tally
            // actually scores elastic
            if (ctl.need_elastic) {
              for (uint32_gpu i = 0; i < mat.n_nuclides; ++i) {
                int32_gpu in = geom.i32[mat.nuclide_off + i];
                GpuNuclide nuc = ce.nuclides[in];
                float dens = geom.f32[mat.density_off + i] * gs.density_mult;
                mac_elastic += dens * gpu_ce_elastic_xs(ce, nuc, &micros[i]);
              }
            }
          } else {
            GpuMgMat m = mg.mats[gs.material];
            // CPU MgxsType::SCATTER via get_xs: no density_mult (the
            // flattened vector is already multiplicity-corrected)
            mac_scat = gpu_mg_vec(mg, m, GPU_MGV_SCATT_XS, g);
          }
        }
        GpuMacroXS xs_sc = (is_ce || gs.material == GPU_MATERIAL_VOID)
                             ? xs
                             : gpu_mg_score_xs(mg, gs.material, g, xs);
        gpu_score_tallies(tv, &gs, GPU_ESTIMATOR_TRACKLENGTH, wgt * distance, E,
          xs_sc, mac_scat, mac_elastic, r_flight0, gs.coord[0].u, distance);
      }
      if (distance > GPU_TINY_BIT)
        gs.surface = GPU_SURFACE_NONE;

      if (d_coll > gs.boundary.d) {
        // ---- event_cross_surface ----
        gs.surface = gs.boundary.surface;
        gs.n_coord = gs.boundary.coord_level;
        bool has_lat = gs.boundary.lat_trans[0] != 0 ||
                       gs.boundary.lat_trans[1] != 0 ||
                       gs.boundary.lat_trans[2] != 0;
        if (has_lat) {
          if (!gpu_cross_lattice(
                geom, &gs, ctl.root_universe, ctl.n_coord_levels)) {
            gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
            gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST_LATTICE, 1u);
            wgt = 0.0f;
            break;
          }
        } else {
          int32_gpu tok = gs.surface;
          int32_gpu i_surf = (tok > 0 ? tok : -tok) - 1;
          GpuSurface surf = geom.surfaces[i_surf];
          if (surf.bc == GPU_BC_VACUUM) {
            k_leak += wgt;
            wgt = 0.0f;
            leaked = 1u;
#ifdef GPU_HOST_DEBUG
            gpu_host_leak(index_source);
#endif
            break;
          } else if (surf.bc == GPU_BC_REFLECTIVE || surf.bc == GPU_BC_WHITE) {
            // CPU cross_reflective_bc: BCs on lower-universe surfaces are
            // fatal (the surface frame there is not the root frame)
            if (gs.n_coord != 1) {
              gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
              gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST_REFLECT, 1u);
              wgt = 0.0f;
              break;
            }
            GpuVec3 r0 = gs.coord[0].r;
            GpuVec3 u0 = gs.coord[0].u;
            GpuVec3 n = gpu_surf_normal(geom, i_surf, r0);
            GpuVec3 u_new;
            if (surf.bc == GPU_BC_REFLECTIVE) {
              u_new = gpu_reflect_dir(u0, n);
            } else {
              // Surface::diffuse_reflect: mu about the unflipped normalized
              // normal, with '>=' so grazing (u.n == 0) reflects inward
              float nn = gpu_norm(n);
              GpuVec3 n_hat = gpu_scale(n, 1.0f / nn);
              float proj = gpu_dot(u0, n_hat);
              float mu_r = (proj >= 0.0f) ? -sqrtf(gpu_prn(&seeds[stream]))
                                          : sqrtf(gpu_prn(&seeds[stream]));
              u_new = gpu_rotate_angle(n_hat, mu_r, &seeds[stream]);
            }
            float un = gpu_norm(u_new);
            u_new = gpu_scale(u_new, 1.0f / un);
            // CPU applies the surface checkpoint after reflection too
            if (ctl.ww_on && ctl.ww_checkpoint_surface && wgt > 0.0f) {
              gpu_apply_weight_window(ctl, tv, banks, sec_stack, &n_stack, &vr,
                gs.coord[0].r, u_new, E, time, &wgt, seeds);
              if (wgt <= 0.0f)
                break;
            }
            // CPU cross_reflective_bc PINS the root cell (coord(0).cell =
            // cell_last(0)) and re-finds only the lower universes: the
            // reflected particle is still in the same root-level cell, and
            // never re-deciding root containment is what stops corner
            // reflections from coin-flipping out on the second surface's
            // fp32 sign (previously the dominant pincell loss class). The
            // descent itself is still needed to rebuild lattice/universe
            // levels below the root.
            gs.surface = -tok;
            gs.coord[0].r = r0;
            gs.coord[0].u = u_new;
            gs.n_coord = 1;
            for (int lv = 1; lv < GPU_MAX_COORD; ++lv)
              gpu_coord_reset(&gs.coord[lv]);
            if (!gpu_find_cell_inner(geom, &gs, ctl.n_coord_levels)) {
              gs.surface = GPU_SURFACE_NONE;
              if (!gpu_rescue_find_cell(
                    geom, &gs, ctl.root_universe, ctl.n_coord_levels)) {
#ifdef GPU_HOST_DEBUG
                gpu_host_debug_lost(&gs, tok, r0, u0, u_new);
#endif
                gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
                gpu_atomic_add_u32(
                  banks.counters + GPU_CTR_LOST_REFLECT, 1u);
                wgt = 0.0f;
                break;
              }
            }
          } else {
            // weight-window surface checkpoint (particle.cpp:400)
            if (ctl.ww_on && ctl.ww_checkpoint_surface && wgt > 0.0f) {
              gpu_apply_weight_window(ctl, tv, banks, sec_stack, &n_stack, &vr,
                gs.coord[0].r, gs.coord[0].u, E, time, &wgt, seeds);
              if (wgt <= 0.0f)
                break;
            }
            // transmission: search from the crossing level down, trying
            // the crossed surface's adjacent cells first
            if (!gpu_local_find_cell_adj(geom, &gs, ctl.n_coord_levels)) {
              if (!gpu_exhaustive_find_cell(
                    geom, &gs, ctl.root_universe, ctl.n_coord_levels)) {
                gs.surface = GPU_SURFACE_NONE;
                if (!gpu_rescue_find_cell(
                      geom, &gs, ctl.root_universe, ctl.n_coord_levels)) {
                  gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
                  gpu_atomic_add_u32(
                    banks.counters + GPU_CTR_LOST_ADVANCE, 1u);
                  wgt = 0.0f;
                  break;
                }
              }
            }
          }
        }
      } else {
        // ---- event_collide ----
        k_col += wgt * xs.nu_fission / xs.total;
        // preserved for post-collision cell reconciliation (particle.cpp)
        int32_gpu presurf = gs.surface;
        bool near_surface = (presurf != GPU_SURFACE_NONE);
        gs.surface = GPU_SURFACE_NONE;
        float E_pre = E;
#ifdef GPU_HOST_DEBUG
        if (!gpu_cell_contains(
              geom, gs.coord[0].cell, gs.coord[0].r, gs.coord[0].u, presurf))
          gpu_host_debug_lost_where(7, &gs, E);
#endif
#ifdef GPU_HOST_DEBUG
        gpu_host_stat_collision(E, gpu_norm(gs.coord[0].r));
#endif

        // collision-estimator tallies (pre-collision weight)
        if (tv.n_tallies > 0) {
          GpuMacroXS xs_sc =
            is_ce ? xs : gpu_mg_score_xs(mg, gs.material, g, xs);
          gpu_score_tallies(tv, &gs, GPU_ESTIMATOR_COLLISION, wgt / xs.total, E,
            xs_sc, mac_scat, mac_elastic, gs.coord[0].r, gs.coord[0].u, 0.0f);
        }

        if (is_ce) {
          // ---- sample_nuclide (physics.cpp:528) ----
          float cutoff = gpu_prn(&seeds[stream]) * xs.total;
          float prob = 0.0f;
          uint32_gpu i_sel = 0;
          for (uint32_gpu i = 0; i < mat.n_nuclides; ++i) {
            i_sel = i;
            prob +=
              geom.f32[mat.density_off + i] * gs.density_mult * micros[i].total;
            if (prob >= cutoff)
              break;
          }
          int32_gpu i_nuc = geom.i32[mat.nuclide_off + i_sel];
          GpuNuclide nuc = ce.nuclides[i_nuc];
          THREAD GpuMicroXS* mic = &micros[i_sel];
#ifdef GPU_HOST_DEBUG
          gpu_host_trace_collide(index_source, E, gs.coord[0].r, i_nuc);
#endif
          if ((int32_gpu)index_source == ctl.trace_id) {
            uint32_gpu ti = gpu_atomic_add_u32(banks.counters + GPU_CTR_TRACE, 1u);
            if (ti < GPU_TRACE_MAX) {
              banks.trace[ti].code = 1.0f;
              banks.trace[ti].a = E;
              banks.trace[ti].b = (float)i_nuc;
              banks.trace[ti].c = gpu_norm(gs.coord[0].r);
            }
          }

          // ---- fission bank (physics.cpp:174) ----
          if (nuc.fissionable && mic->fission > 0.0f &&
              ctl.run_mode == GPU_RUN_EIGENVALUE) {
            // sample_fission (physics.cpp:629): ONE partial-fission pick
            // per collision, before the site count — every site from this
            // collision uses the same reaction, and the RN is consumed
            // even when the sampled site count is 0 (CPU RN order).
            int32_gpu rec0 = ce.i32[nuc.fis_rx_off];
            int32_gpu rec = rec0;
            if (nuc.n_fission_rx > 1 && !mic->use_ptable) {
              float fcut = gpu_prn(&seeds[stream]) * mic->fission;
              float fpr = 0.0f;
              for (uint32_gpu fr = 0; fr < nuc.n_fission_rx; ++fr) {
                rec = ce.i32[nuc.fis_rx_off + fr];
                GLOBAL const int32_gpu* rh = ce.i32 + ce.i32[rec];
                int32_gpu thr = rh[GPU_RX_THRESHOLD];
                if (mic->i_grid >= thr &&
                    mic->i_grid - thr + 1 < rh[GPU_RX_NXS]) {
                  GLOBAL const float* rxs = ce.f32 + rh[GPU_RX_XSOFF];
                  int32_gpu kk = mic->i_grid - thr;
                  fpr += (1.0f - mic->interp) * rxs[kk] +
                         mic->interp * rxs[kk + 1];
                }
                if (fpr > fcut)
                  break;
              }
            }
            int32_gpu rec_n_prod = ce.i32[rec + 1];
            float nu_t = wgt / ctl.keff * mic->nu_fission / mic->total;
            int32_gpu nu = (int32_gpu)nu_t;
            if (gpu_prn(&seeds[stream]) <= (nu_t - (float)nu))
              ++nu;
            for (int32_gpu n = 0; n < nu; ++n) {
              GpuSourceSite site;
              site.r[0] = gs.coord[0].r.x;
              site.r[1] = gs.coord[0].r.y;
              site.r[2] = gs.coord[0].r.z;
              site.time = time;
              site.wgt = 1.0f;
              site.delayed_group = 0;
              int32_gpu prec = rec; // reaction supplying the products
              int32_gpu n_prod = rec_n_prod;
              // sample_fission_neutron (physics.cpp:1077)
              float nu_tot = gpu_f1d(ce, nuc.total_nu_f1d, E);
              float nu_d =
                (nuc.n_delayed > 0) ? gpu_ce_nu_delayed(ce, nuc, E) : 0.0f;
              float beta = (nu_tot > 0.0f) ? nu_d / nu_tot : 0.0f;
              int32_gpu dg = 0; // product index (0 = prompt)
              // CPU draws the prompt/delayed RN unconditionally, even when
              // beta == 0 (sample_fission_neutron) — keep the stream paired
              float beta_xi = gpu_prn(&seeds[stream]);
              if (nuc.n_delayed > 0 && beta_xi < beta) {
                // CPU walks the SELECTED reaction's delayed-product yields
                // (rx.products_[group]); fall back to the first fission
                // reaction only when the partial carries no delayed
                // products (upstream would index out of bounds there).
                if (n_prod <= 1) {
                  prec = rec0;
                  n_prod = ce.i32[prec + 1];
                }
                float xi = gpu_prn(&seeds[stream]) * nu_d;
                float pr = 0.0f;
                uint32_gpu gg = 1;
                for (; gg < nuc.n_delayed; ++gg) {
                  pr += gpu_f1d(ce, ce.i32[prec + 2 + 3 * gg], E);
                  if (xi < pr)
                    break;
                }
                if (gg > nuc.n_delayed)
                  gg = nuc.n_delayed;
                dg = (int32_gpu)gg;
                if (dg >= n_prod)
                  dg = n_prod - 1;
                site.delayed_group = dg;
                float lambda = ce.f32[ce.i32[prec + 2 + 3 * dg + 2]];
                site.time -= logf(gpu_prn(&seeds[stream])) / lambda;
              }
              // energy from the product distribution (reject above E_max)
              float mu_f = 1.0f;
              float E_f = 0.0f;
              int32_gpu dist = ce.i32[prec + 2 + 3 * dg + 1];
              for (int it = 0; it < 100; ++it) {
                GpuSampleEA sf = gpu_sample_dist(ce, dist, E, &seeds[stream]);
                E_f = sf.E_out;
                mu_f = sf.mu;
                if (E_f < ce.energy_max)
                  break;
              }
              site.E = E_f;
              GpuVec3 uf =
                gpu_rotate_angle(gs.coord[0].u, mu_f, &seeds[stream]);
              site.u[0] = uf.x;
              site.u[1] = uf.y;
              site.u[2] = uf.z;
              site.parent_id = (int32_gpu)(index_source - 1);
              site.progeny_id = n_progeny;
              uint32_gpu idx =
                gpu_atomic_add_u32(banks.counters + GPU_CTR_FISSION_BANK, 1u);
              if (idx >= ctl.fission_bank_cap)
                break;
              ++n_progeny;
#ifdef GPU_HOST_DEBUG
              gpu_host_stat_fsite(gpu_norm(gs.coord[0].r));
#endif
              banks.fission[idx] = site;
            }
          }

#ifdef GPU_HOST_DEBUG
          gpu_host_trace_postfis(index_source,
            (uint32_gpu)(seeds[stream] & 0xffffffu), n_progeny,
            (uint32_gpu)(seeds[GPU_STREAM_URR_PTABLE] & 0xffffffu));
#endif
          if ((int32_gpu)index_source == ctl.trace_id) {
            uint32_gpu ti = gpu_atomic_add_u32(banks.counters + GPU_CTR_TRACE, 1u);
            if (ti < GPU_TRACE_MAX) {
              banks.trace[ti].code = 4.0f;
              banks.trace[ti].a = (float)(seeds[stream] & 0xffffffu);
              banks.trace[ti].b = (float)n_progeny;
              banks.trace[ti].c =
                (float)(seeds[GPU_STREAM_URR_PTABLE] & 0xffffffu);
            }
          }
          // ---- absorption (physics.cpp:672) ----
          if (mic->absorption > 0.0f) {
            if (ctl.survival_biasing) {
              // implicit capture: remove the absorbed fraction of the weight
              float wgt_absorb = wgt * mic->absorption / mic->total;
              wgt -= wgt_absorb;
              if (ctl.run_mode == GPU_RUN_EIGENVALUE)
                k_abs += wgt_absorb * mic->nu_fission / mic->absorption;
            } else if (mic->absorption >
                       gpu_prn(&seeds[stream]) * mic->total) {
              k_abs += wgt * mic->nu_fission / mic->absorption;
              wgt = 0.0f;
#ifdef GPU_HOST_DEBUG
              gpu_host_stat_absorb();
#endif
            }
          }
          if (wgt > 0.0f) {
            // ---- scatter (physics.cpp:708) ----
            float cut =
              gpu_prn(&seeds[stream]) * (mic->total - mic->absorption);
            float el = gpu_ce_elastic_xs(ce, nuc, mic);
            if (mic->i_sab >= 0 && el - mic->thermal <= cut && el > cut) {
              // S(a,b) scatter (physics.cpp sab_scatter +
              // ThermalData::sample_dist)
              GpuSabTable t = sab.tables[mic->i_sab];
              int32_gpu tblob =
                (gpu_prn(&seeds[stream]) <
                  mic->thermal_elastic / mic->thermal)
                  ? t.elastic_dist
                  : t.inelastic_dist;
              GpuSampleEA r =
                gpu_sab_sample_dist(sab, tblob, E, &seeds[stream]);
              float mu = r.mu;
              if (mu > 1.0f)
                mu = 1.0f;
              if (mu < -1.0f)
                mu = -1.0f;
              E = r.E_out;
              gs.coord[0].u =
                gpu_rotate_angle(gs.coord[0].u, mu, &seeds[stream]);
            } else if (el - mic->thermal > cut) {
              // elastic with free-gas target motion (physics.cpp:788)
              float vel = sqrtf(E);
              GpuVec3 u0 = gs.coord[0].u;
              GpuVec3 v_n = gpu_scale(u0, vel);
              GpuVec3 v_t = gpu_v3(0.0f, 0.0f, 0.0f);
              if (!mic->use_ptable &&
                  (E < ctl.free_gas_threshold * nuc.kT || nuc.awr <= 1.0f)) {
                v_t = gpu_sample_cxs_target_velocity(
                  nuc.awr, E, u0, nuc.kT, &seeds[stream]);
              }
              GpuVec3 v_cm = gpu_scale(
                gpu_add(v_n, gpu_scale(v_t, nuc.awr)), 1.0f / (nuc.awr + 1.0f));
              v_n = gpu_sub(v_n, v_cm);
              vel = gpu_norm(v_n);
              float mu_cm;
              if (ctl.debug_iso_mu || nuc.elastic_angle < 0)
                mu_cm = 2.0f * gpu_prn(&seeds[stream]) - 1.0f;
              else
                mu_cm = gpu_sample_angle_dist(
                  ce, nuc.elastic_angle, E, &seeds[stream]);
              GpuVec3 u_cm = gpu_scale(v_n, 1.0f / vel);
              v_n =
                gpu_scale(gpu_rotate_angle(u_cm, mu_cm, &seeds[stream]), vel);
              v_n = gpu_add(v_n, v_cm);
              E = gpu_dot(v_n, v_n);
              vel = sqrtf(E);
              GpuVec3 u_new = gpu_scale(v_n, 1.0f / vel);
              float un = gpu_norm(u_new);
#ifdef GPU_HOST_DEBUG
              gpu_host_stat_elastic(gpu_dot(u0, u_new) / un, E / E_pre);
              gpu_host_trace_elastic(index_source, E);
#endif
              if ((int32_gpu)index_source == ctl.trace_id) {
                uint32_gpu ti = gpu_atomic_add_u32(banks.counters + GPU_CTR_TRACE, 1u);
                if (ti < GPU_TRACE_MAX) {
                  banks.trace[ti].code = 2.0f;
                  banks.trace[ti].a = E;
                  banks.trace[ti].b = mu_cm;
                  banks.trace[ti].c = 0.0f;
                }
              }
              gs.coord[0].u = gpu_scale(u_new, 1.0f / un);
            } else {
              // inelastic: scan reactions (physics.cpp:764)
              float pr = el;
              int32_gpu rx_off = ce.i32[nuc.inelastic_off];
              for (uint32_gpu j = 0; j < nuc.n_inelastic && pr <= cut; ++j) {
                rx_off = ce.i32[nuc.inelastic_off + j];
                GLOBAL const int32_gpu* rxh = ce.i32 + rx_off;
                int32_gpu thr = rxh[GPU_RX_THRESHOLD];
                if (mic->i_grid >= thr &&
                    mic->i_grid - thr + 1 < rxh[GPU_RX_NXS]) {
                  GLOBAL const float* rxs = ce.f32 + rxh[GPU_RX_XSOFF];
                  int32_gpu k = mic->i_grid - thr;
                  pr +=
                    (1.0f - mic->interp) * rxs[k] + mic->interp * rxs[k + 1];
                }
              }
              GLOBAL const int32_gpu* rx = ce.i32 + rx_off;
#ifdef GPU_HOST_DEBUG
              gpu_host_stat_inelastic(rx[GPU_RX_MT]);
              gpu_host_trace_inelastic(index_source, rx[GPU_RX_MT]);
#endif
              if ((int32_gpu)index_source == ctl.trace_id) {
                uint32_gpu ti = gpu_atomic_add_u32(banks.counters + GPU_CTR_TRACE, 1u);
                if (ti < GPU_TRACE_MAX) {
                  banks.trace[ti].code = 3.0f;
                  banks.trace[ti].a = (float)rx[GPU_RX_MT];
                  banks.trace[ti].b = E;
                  banks.trace[ti].c = 0.0f;
                }
              }
              float E_in = E;
              GpuSampleEA se =
                gpu_sample_dist(ce, rx[GPU_RX_DIST], E_in, &seeds[stream]);
              float E_out = se.E_out;
              float mu = se.mu;
              if (rx[GPU_RX_CM]) {
                // CM -> lab (physics.cpp:1153)
                float A = nuc.awr;
                float E_cm = E_out;
                E_out =
                  E_cm + (E_in + 2.0f * mu * (A + 1.0f) * sqrtf(E_in * E_cm)) /
                           ((A + 1.0f) * (A + 1.0f));
                mu = mu * sqrtf(E_cm / E_out) +
                     1.0f / (A + 1.0f) * sqrtf(E_in / E_out);
              }
              if (mu > 1.0f)
                mu = 1.0f;
              if (mu < -1.0f)
                mu = -1.0f;
              E = E_out;
              gs.coord[0].u =
                gpu_rotate_angle(gs.coord[0].u, mu, &seeds[stream]);
              // yield: integral -> clones, else weight multiplication
              int32_gpu yblob = rx[GPU_RX_YIELD];
              if (yblob >= 0) {
                float y = gpu_f1d(ce, yblob, E_in);
                // integral yield -> clones; Particle::create_secondary
                // rejects secondaries below the energy cutoff at creation
                if (floorf(y) == y && y > 0.0f && E >= ctl.energy_cutoff) {
                  int32_gpu extra = (int32_gpu)(y + 0.5f) - 1;
                  int32_gpu missed = 0;
                  for (int32_gpu q = 0; q < extra; ++q) {
                    if (ctl.run_mode == GPU_RUN_FIXED_SOURCE) {
                      // if the banks are full, fall back to implicit
                      // multiplication: the parent carries the weight of the
                      // clones that could not be created (the same treatment
                      // OpenMC uses for non-integer yields), so weight is
                      // conserved instead of a real neutron being lost
                      if (!gpu_push_secondary(ctl, banks, sec_stack, &n_stack,
                            &vr, gs.coord[0].r, gs.coord[0].u, E, wgt, time,
                            seeds))
                        ++missed;
                    } else if (n_stack < GPU_MAX_SECONDARY_STACK) {
                      THREAD GpuSecondary* sc = &sec_stack[n_stack++];
                      sc->r[0] = gs.coord[0].r.x;
                      sc->r[1] = gs.coord[0].r.y;
                      sc->r[2] = gs.coord[0].r.z;
                      sc->u[0] = gs.coord[0].u.x;
                      sc->u[1] = gs.coord[0].u.y;
                      sc->u[2] = gs.coord[0].u.z;
                      sc->E = E;
                      sc->wgt = wgt;
                      sc->time = time;
                    } else {
                      // eigenvalue mode has no spill bank: conserve the
                      // neutron's weight by implicit multiplication rather
                      // than dropping it
                      ++missed;
                      gpu_atomic_add_u32(
                        banks.counters + GPU_CTR_SECONDARY_BANK, 1u);
                    }
                  }
                  if (missed > 0)
                    wgt *= (float)(missed + 1);
                } else {
                  wgt *= y;
                }
              }
            }
            // keep energies on the tabulated grid floor
            if (E < 1.0e-11f)
              E = 1.0e-11f;
          }
          // URR stream discipline (physics.cpp:164)
          if (E != E_pre) {
            gpu_advance_prn_seed(
              (int64_gpu)ce.n_nuclides, &seeds[GPU_STREAM_URR_PTABLE]);
          }
        } else {
          // ---- multigroup collision (physics_mg.cpp) ----
          GpuMgMat m = mg.mats[gs.material];
          if (m.fissionable && ctl.run_mode == GPU_RUN_EIGENVALUE) {
            float nu_t = wgt / ctl.keff * xs.nu_fission / xs.total;
            int32_gpu nu = (int32_gpu)nu_t;
            if (gpu_prn(&seeds[stream]) <= (nu_t - (float)nu))
              ++nu;
            for (int32_gpu n = 0; n < nu; ++n) {
              // CPU RN order (physics_mg.cpp:144): mu, phi, then energy
              float mu_f = 2.0f * gpu_prn(&seeds[stream]) - 1.0f;
              float phi = 6.283185307179586f * gpu_prn(&seeds[stream]);
              GpuMgFission fe =
                gpu_mg_sample_fission_energy(mg, m, g, &seeds[stream]);
              GpuSourceSite site;
              site.r[0] = gs.coord[0].r.x;
              site.r[1] = gs.coord[0].r.y;
              site.r[2] = gs.coord[0].r.z;
              float a = sqrtf(fmaxf(0.0f, 1.0f - mu_f * mu_f));
              site.u[0] = mu_f;
              site.u[1] = a * cosf(phi);
              site.u[2] = a * sinf(phi);
              site.E = (float)fe.gout;
              site.time = time;
              site.wgt = 1.0f;
              site.delayed_group = fe.dg + 1;
              if (fe.dg >= 0) {
                float lambda = mg.f32[m.decay_off + fe.dg];
                site.time -= logf(gpu_prn(&seeds[stream])) / lambda;
              }
              site.parent_id = (int32_gpu)(index_source - 1);
              site.progeny_id = n_progeny;
              uint32_gpu idx =
                gpu_atomic_add_u32(banks.counters + GPU_CTR_FISSION_BANK, 1u);
              if (idx >= ctl.fission_bank_cap)
                break;
              ++n_progeny;
              banks.fission[idx] = site;
            }
          }
          bool absorbed = false;
          if (xs.absorption > 0.0f) {
            if (ctl.survival_biasing) {
              float wgt_absorb = wgt * xs.absorption / xs.total;
              wgt -= wgt_absorb;
              if (ctl.run_mode == GPU_RUN_EIGENVALUE)
                k_abs += wgt_absorb * xs.nu_fission / xs.absorption;
            } else if (xs.absorption > gpu_prn(&seeds[stream]) * xs.total) {
              k_abs += wgt * xs.nu_fission / xs.absorption;
              wgt = 0.0f;
              absorbed = true;
            }
          }
          if (!absorbed && wgt > 0.0f) {
            float mu;
            int32_gpu gout =
              gpu_mg_sample_scatter(mg, m, g, &mu, &wgt, &seeds[stream]);
            gs.coord[0].u = gpu_rotate_angle(gs.coord[0].u, mu, &seeds[stream]);
            g = gout;
            E = geom.f32[ctl.mg_bin_avg_off + g];
          }
        }

        if (wgt > 0.0f) {
          // propagate direction down coordinate levels (rotation-aware)
          for (int32_gpu j = 1; j < gs.n_coord; ++j) {
            THREAD GpuCoord* parent = &gs.coord[j - 1];
            GpuCell pc = geom.cells[parent->cell];
            if (pc.rot_off >= 0) {
              gs.coord[j].u = gpu_rotate(parent->u, geom.f32 + pc.rot_off);
            } else {
              gs.coord[j].u = parent->u;
            }
          }
        }

        // ---- post-collision variance reduction (collision(),
        // physics.cpp:99-116): weight windows at the collision checkpoint,
        // else russian roulette when survival biasing is on ----
        if (wgt > 0.0f) {
          if (ctl.ww_on && ctl.ww_checkpoint_collision) {
            gpu_apply_weight_window(ctl, tv, banks, sec_stack, &n_stack, &vr,
              gs.coord[0].r, gs.coord[0].u, E, time, &wgt, seeds);
          } else if (ctl.survival_biasing && wgt < ctl.weight_cutoff) {
            gpu_russian_roulette(&wgt, ctl.weight_survive, &seeds[stream]);
          }
          // energy cutoff (physics.cpp:112) comes AFTER the window, as on CPU
          if (is_ce && wgt > 0.0f && E < ctl.energy_cutoff)
            wgt = 0.0f;
        }

        // reconcile_cell_after_collision (geometry.cpp), run on EVERY
        // collision: CPU gates this on a near-surface token, but an fp32
        // flight can overshoot a surface by more than TINY_BIT (token
        // already cleared) and collide epsilon-outside its cell — a stale
        // chain then transports through walls (the infinite wall planes
        // keep reflecting the escapee outside the box forever). The repair
        // runs in place (a trial copy of the coordinate state costs enough
        // stack to collapse occupancy); an unrepairable state is lost,
        // as on the CPU.
        if (wgt > 0.0f) {
          int32_gpu invalid_level = -1;
          for (int32_gpu lev = 0; lev < gs.n_coord; ++lev) {
            if (gs.coord[lev].cell == GPU_C_NONE ||
                !gpu_cell_contains(geom, gs.coord[lev].cell, gs.coord[lev].r,
                  gs.coord[lev].u, GPU_SURFACE_NONE)) {
              invalid_level = lev;
              break;
            }
          }
          if (invalid_level >= 0) {
            bool fixed = false;
            if (gs.coord[invalid_level].cell != GPU_C_NONE) {
              gs.n_coord = invalid_level + 1;
              fixed = gpu_local_find_cell(geom, &gs, ctl.n_coord_levels);
            }
            if (!fixed)
              fixed = gpu_exhaustive_find_cell(
                geom, &gs, ctl.root_universe, ctl.n_coord_levels);
            if (!fixed) {
              // overshoot rescue: escalating nudges against, then along,
              // the outgoing direction (the overshoot was along the OLD
              // flight direction, so either sign may point back inside)
              GpuVec3 r_c = gs.coord[0].r;
              float step = -GPU_TINY_BIT;
              for (int s = 0; s < 3 && !fixed; ++s) {
                gpu_move_distance(&gs, step);
                fixed = gpu_exhaustive_find_cell(
                  geom, &gs, ctl.root_universe, ctl.n_coord_levels);
                step *= 8.0f;
              }
              if (!fixed) {
                gs.coord[0].r = r_c;
                fixed = gpu_rescue_find_cell(
                  geom, &gs, ctl.root_universe, ctl.n_coord_levels);
              }
            }
            if (!fixed) {
              gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
              gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST_RECONCILE, 1u);
              wgt = 0.0f;
            }
          }
        }
      }

      ++n_events;
#if GPU_DIAG_BALLAST > 0
      ballast[n_events % GPU_DIAG_BALLAST] += E;
#endif
      if (n_events >= ctl.max_events) {
        gpu_atomic_add_u32(banks.counters + GPU_CTR_MAX_EVENT_HIT, 1u);
        wgt = 0.0f;
      }
    }

    // ---- revive from the local secondary stack (LIFO, CPU-compatible) ----
    if (n_stack == 0)
      break;
    GpuSecondary sc = sec_stack[--n_stack];
    gs.n_coord = 1;
    gs.surface = GPU_SURFACE_NONE;
    gs.lost = 0;
    for (int i = 0; i < GPU_MAX_COORD; ++i)
      gpu_coord_reset(&gs.coord[i]);
    gs.coord[0].r = gpu_v3(sc.r[0], sc.r[1], sc.r[2]);
    gs.coord[0].u = gpu_v3(sc.u[0], sc.u[1], sc.u[2]);
    gs.coord[0].universe = ctl.root_universe;
    E = sc.E;
    wgt = sc.wgt;
    time = sc.time;
    n_events = 0;
    if (!gpu_exhaustive_find_cell(
          geom, &gs, ctl.root_universe, ctl.n_coord_levels)) {
      gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
      wgt = 0.0f;
    }
  }

  // ---- event_death ----
  // keff estimators are written to this particle's OWN slot (no atomics) and
  // summed on the host in fp64 in index order. Atomic accumulation here was
  // order-dependent, and because simulation::keff feeds back into the next
  // generation's fission-site count (nu_t = wgt/keff * nu_f/total), a 1-ulp
  // difference eventually flips one particle's nu and the whole run diverges:
  // measured as a bimodal MG k (1.34172 / 1.34121) with the two runs
  // bit-identical for 82 batches before splitting. Per-particle slots make
  // the reduction deterministic (and remove 4 atomics per history).
  uint32_gpu slot = tid * GPU_RED_WIDTH;
#if GPU_DIAG_BALLAST > 0
  {
    float bsum = 0.0f;
    for (int i = 0; i < GPU_DIAG_BALLAST; ++i)
      bsum += ballast[i];
    // never true, but the compiler cannot prove it: keeps the array live
    if (bsum == -1.0e30f)
      gpu_atomic_add_u32(banks.counters + GPU_CTR_TRACE, 1u);
  }
#endif
  banks.red_slots[slot + GPU_RED_K_TRACKLENGTH] = k_tl;
  banks.red_slots[slot + GPU_RED_K_COLLISION] = k_col;
  banks.red_slots[slot + GPU_RED_K_ABSORPTION] = k_abs;
  banks.red_slots[slot + GPU_RED_LEAKAGE] = k_leak;
  banks.red_slots[slot + GPU_RED_EVENTS] = (float)n_events;
  // progeny count with the leak flag in the top bit (debug diagnostics)
  banks.progeny[ctl.source_offset + tid] =
    (uint32_gpu)n_progeny | (leaked << 31);
}
