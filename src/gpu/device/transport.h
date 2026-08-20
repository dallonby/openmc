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

#define GPU_MAX_MAT_NUCLIDES 32
#define GPU_MAX_SECONDARY_STACK 8

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

//! Match one filter; returns bin index or -1.
DEVICE_FN int32_gpu gpu_filter_match(
  GpuTallyView tv, GpuFilterDesc f, THREAD const GpuGeomState* gs, float E)
{
  switch (f.type) {
  case GPU_FILTER_CELL: {
    for (int32_gpu j = 0; j < gs->n_coord; ++j) {
      int32_gpu b = tv.i32[f.map_off + gs->coord[j].cell];
      if (b >= 0)
        return b;
    }
    return -1;
  }
  case GPU_FILTER_MATERIAL: {
    if (gs->material < 0)
      return -1;
    return tv.i32[f.map_off + gs->material];
  }
  case GPU_FILTER_UNIVERSE: {
    for (int32_gpu j = 0; j < gs->n_coord; ++j) {
      int32_gpu b = tv.i32[f.map_off + gs->coord[j].universe];
      if (b >= 0)
        return b;
    }
    return -1;
  }
  case GPU_FILTER_ENERGY: {
    GLOBAL const float* edges = tv.f32 + f.map_off;
    uint32_gpu n = f.n_bins;
    if (E < edges[0] || E > edges[n])
      return -1;
    uint32_gpu lo = 0, hi = n;
    while (hi - lo > 1) {
      uint32_gpu mid = (lo + hi) / 2;
      if (E >= edges[mid])
        lo = mid;
      else
        hi = mid;
    }
    return (int32_gpu)lo;
  }
  case GPU_FILTER_MESH: {
    GpuMesh mesh = tv.meshes[f.mesh];
    GpuVec3 r = gs->coord[0].r;
    int32_gpu i = (int32_gpu)floorf((r.x - mesh.llx) / mesh.wx);
    int32_gpu j = (int32_gpu)floorf((r.y - mesh.lly) / mesh.wy);
    int32_gpu k = (int32_gpu)floorf((r.z - mesh.llz) / mesh.wz);
    if (i < 0 || i >= mesh.nx || j < 0 || j >= mesh.ny || k < 0 || k >= mesh.nz)
      return -1;
    return mesh.nx * mesh.ny * k + mesh.nx * j + i;
  }
  }
  return -1;
}

//! Score all tallies of the given estimator for one event.
//! coeff: tracklength -> wgt * distance; collision -> wgt / Sigma_t.
//! Macroscopic score values are supplied by the caller (mode-specific).
DEVICE_FN void gpu_score_tallies(GpuTallyView tv, THREAD const GpuGeomState* gs,
  uint32_gpu estimator, float coeff, float E, GpuMacroXS xs, float scat,
  float elastic)
{
  for (uint32_gpu t = 0; t < tv.n_tallies; ++t) {
    GpuTallyDesc td = tv.tallies[t];
    if (td.estimator != estimator)
      continue;
    int32_gpu flat = 0;
    bool miss = false;
    for (uint32_gpu fi = 0; fi < td.n_filters; ++fi) {
      GpuFilterDesc f = tv.filters[td.filter_off + fi];
      int32_gpu b = gpu_filter_match(tv, f, gs, E);
      if (b < 0) {
        miss = true;
        break;
      }
      flat += b * (int32_gpu)f.stride;
    }
    if (miss)
      continue;
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
  }
}

// -------------------------------------------------------------- transport --

struct GpuBanks {
  GLOBAL const GpuSourceSite* source;
  GLOBAL GpuSourceSite* fission;
  GLOBAL gpu_atomic_u32* counters;
  GLOBAL uint32_gpu* progeny;
  GLOBAL gpu_atomic_f32* red_slots; // [ceil(n/256)][GPU_RED_WIDTH]
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

//! neutron speed in cm/s (Particle::speed, relativistic)
DEVICE_FN float gpu_neutron_speed(float E)
{
  float inv = 939.56542052e6f / (E + 939.56542052e6f);
  return 2.99792458e10f * sqrtf(fmaxf(0.0f, 1.0f - inv * inv));
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

//! Full history for one source particle. Mirrors the CPU event loop.
DEVICE_FN void gpu_run_particle(uint32_gpu tid, GCONST GpuControl& ctl,
  GpuGeomData geom, GpuMgView mg, GpuCeView ce, GpuTallyView tv, GpuBanks banks)
{
  // ---- initialize_history ----
  GpuSourceSite src = banks.source[ctl.source_offset + tid];
  int64_gpu index_source = (int64_gpu)(ctl.source_offset + tid) + 1;

  uint64_gpu seeds[GPU_N_STREAMS];
  gpu_init_particle_seeds((int64_gpu)ctl.seed_base + index_source,
    ctl.master_seed, ctl.prn_stride, seeds);
  int stream = GPU_STREAM_TRACKING;

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

  float k_tl = 0.0f, k_col = 0.0f, k_abs = 0.0f, k_leak = 0.0f;
  int32_gpu n_progeny = 0;
  uint32_gpu n_events = 0;
  uint32_gpu leaked = 0;

  GpuSecondary sec_stack[GPU_MAX_SECONDARY_STACK];
  int32_gpu n_stack = 0;

  // per-material micro cache (CE)
  GpuMicroXS micros[GPU_MAX_MAT_NUCLIDES];

  bool found =
    gpu_exhaustive_find_cell(geom, &gs, ctl.root_universe, ctl.n_coord_levels);
  if (!found) {
    gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
    gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST_INIT, 1u);
    wgt = 0.0f;
  }

  for (;;) {
    // ---- event loop ----
    while (wgt > 0.0f) {
      // event_calculate_xs
      GpuMacroXS xs;
      xs.total = 0.0f;
      xs.absorption = 0.0f;
      xs.fission = 0.0f;
      xs.nu_fission = 0.0f;
      float mac_scat = 0.0f;
      float mac_elastic = 0.0f;
      GpuMaterial mat;
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
            micros[i] = gpu_ce_micro_xs(ce, ce.nuclides[in], E, i_log, seeds);
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

      // event_advance
      gs.boundary = gpu_distance_to_boundary(geom, &gs);
      if (gs.lost) {
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
        gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
        gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST_ADVANCE, 1u);
        wgt = 0.0f;
        break;
      }
#ifdef GPU_HOST_DEBUG
      gpu_host_stat_flight(distance);
      gpu_host_trace_fly(index_source, distance,
        (float)(seeds[stream] & 0xffffffu), gs.boundary.d, E);
#endif
      if ((int32_gpu)index_source == ctl.trace_id) {
        uint32_gpu ti = gpu_atomic_add_u32(banks.counters + 7, 1u);
        if (ti < GPU_TRACE_MAX) {
          banks.trace[ti].code = 0.0f;
          banks.trace[ti].a = distance;
          banks.trace[ti].b = (float)(seeds[stream] & 0xffffffu);
          banks.trace[ti].c = E;
        }
      }
      gpu_move_distance(&gs, distance);
      if (is_ce) {
        time += distance / gpu_neutron_speed(E);
      } else if (gs.material != GPU_MATERIAL_VOID) {
        GpuMgMat m = mg.mats[gs.material];
        time += distance * gpu_mg_vec(mg, m, GPU_MGV_INV_VELOCITY, g);
      }
      // tracklength keff estimator
      k_tl += wgt * distance * xs.nu_fission;
      // tracklength tallies
      if (tv.n_tallies > 0) {
        if (gs.material != GPU_MATERIAL_VOID) {
          if (is_ce) {
            mac_scat = xs.total - xs.absorption;
            for (uint32_gpu i = 0; i < mat.n_nuclides; ++i) {
              int32_gpu in = geom.i32[mat.nuclide_off + i];
              GpuNuclide nuc = ce.nuclides[in];
              float dens = geom.f32[mat.density_off + i] * gs.density_mult;
              mac_elastic += dens * gpu_ce_elastic_xs(ce, nuc, &micros[i]);
            }
          } else {
            GpuMgMat m = mg.mats[gs.material];
            mac_scat = gpu_mg_vec(mg, m, GPU_MGV_SCATT_XS, g) * gs.density_mult;
          }
        }
        gpu_score_tallies(tv, &gs, GPU_ESTIMATOR_TRACKLENGTH, wgt * distance, E,
          xs, mac_scat, mac_elastic);
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
            GpuVec3 r0 = gs.coord[0].r;
            GpuVec3 u0 = gs.coord[0].u;
            GpuVec3 n = gpu_surf_normal(geom, i_surf, r0);
            GpuVec3 u_new;
            if (surf.bc == GPU_BC_REFLECTIVE) {
              u_new = gpu_reflect_dir(u0, n);
            } else {
              float nn = gpu_norm(n);
              GpuVec3 n_hat = gpu_scale(n, 1.0f / nn);
              if (gpu_dot(u0, n_hat) > 0.0f)
                n_hat = gpu_scale(n_hat, -1.0f);
              float mu_r = sqrtf(gpu_prn(&seeds[stream]));
              u_new = gpu_rotate_angle(n_hat, mu_r, &seeds[stream]);
            }
            float un = gpu_norm(u_new);
            u_new = gpu_scale(u_new, 1.0f / un);
            gs.surface = -tok;
            gs.coord[0].r = r0;
            gs.coord[0].u = u_new;
            gs.n_coord = 1;
            if (!gpu_exhaustive_find_cell(
                  geom, &gs, ctl.root_universe, ctl.n_coord_levels)) {
#ifdef GPU_HOST_DEBUG
              gpu_host_debug_lost(&gs, tok, r0, u0, u_new);
#endif
              gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
              gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST_REFLECT, 1u);
              wgt = 0.0f;
              break;
            }
          } else {
            // transmission: search from the crossing level down
            if (!gpu_local_find_cell(geom, &gs, ctl.n_coord_levels)) {
              if (!gpu_exhaustive_find_cell(
                    geom, &gs, ctl.root_universe, ctl.n_coord_levels)) {
                gs.surface = GPU_SURFACE_NONE;
                gpu_move_distance(&gs, GPU_TINY_BIT);
                if (!gpu_exhaustive_find_cell(
                      geom, &gs, ctl.root_universe, ctl.n_coord_levels)) {
                  gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
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
        gs.surface = GPU_SURFACE_NONE;
        float E_pre = E;
#ifdef GPU_HOST_DEBUG
        gpu_host_stat_collision(E, gpu_norm(gs.coord[0].r));
#endif

        // collision-estimator tallies (pre-collision weight)
        if (tv.n_tallies > 0) {
          gpu_score_tallies(tv, &gs, GPU_ESTIMATOR_COLLISION, wgt / xs.total, E,
            xs, mac_scat, mac_elastic);
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
            uint32_gpu ti = gpu_atomic_add_u32(banks.counters + 7, 1u);
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
              // sample_fission (physics.cpp:586): partial-fission
              // selection consumes one RN unless in the URR
              int32_gpu rec = ce.i32[nuc.fis_rx_off];
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
              int32_gpu n_prod = ce.i32[rec + 1];
              // sample_fission_neutron (physics.cpp:1077)
              float nu_tot = gpu_f1d(ce, nuc.total_nu_f1d, E);
              float nu_d =
                (nuc.n_delayed > 0) ? gpu_ce_nu_delayed(ce, nuc, E) : 0.0f;
              float beta = (nu_tot > 0.0f) ? nu_d / nu_tot : 0.0f;
              int32_gpu dg = 0; // product index (0 = prompt)
              if (nuc.n_delayed > 0 && gpu_prn(&seeds[stream]) < beta) {
                float xi = gpu_prn(&seeds[stream]) * nu_d;
                int32_gpu rec0 = ce.i32[nuc.fis_rx_off];
                float pr = 0.0f;
                uint32_gpu gg = 1;
                for (; gg < nuc.n_delayed; ++gg) {
                  pr += gpu_f1d(ce, ce.i32[rec0 + 2 + 3 * gg], E);
                  if (xi < pr)
                    break;
                }
                if (gg > nuc.n_delayed)
                  gg = nuc.n_delayed;
                dg = (int32_gpu)gg;
                // delayed products come from the selected reaction when it
                // has them, else from the first fission reaction
                if (dg >= n_prod) {
                  rec = rec0;
                  n_prod = ce.i32[rec + 1];
                  if (dg >= n_prod)
                    dg = n_prod - 1;
                }
                site.delayed_group = dg;
                float lambda = ce.f32[ce.i32[rec + 2 + 3 * dg + 2]];
                site.time -= logf(gpu_prn(&seeds[stream])) / lambda;
              }
              // energy from the product distribution (reject above E_max)
              float mu_f = 1.0f;
              float E_f = 0.0f;
              int32_gpu dist = ce.i32[rec + 2 + 3 * dg + 1];
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
            uint32_gpu ti = gpu_atomic_add_u32(banks.counters + 7, 1u);
            if (ti < GPU_TRACE_MAX) {
              banks.trace[ti].code = 4.0f;
              banks.trace[ti].a = (float)(seeds[stream] & 0xffffffu);
              banks.trace[ti].b = (float)n_progeny;
              banks.trace[ti].c =
                (float)(seeds[GPU_STREAM_URR_PTABLE] & 0xffffffu);
            }
          }
          // ---- absorption (analog, physics.cpp:672) ----
          if (mic->absorption > 0.0f) {
            if (mic->absorption > gpu_prn(&seeds[stream]) * mic->total) {
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
            if (el > cut) {
              // elastic with free-gas target motion (physics.cpp:788)
              float vel = sqrtf(E);
              GpuVec3 u0 = gs.coord[0].u;
              GpuVec3 v_n = gpu_scale(u0, vel);
              GpuVec3 v_t = gpu_v3(0.0f, 0.0f, 0.0f);
              if (!mic->use_ptable &&
                  (E < 400.0f * nuc.kT || nuc.awr <= 1.0f)) {
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
                uint32_gpu ti = gpu_atomic_add_u32(banks.counters + 7, 1u);
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
                uint32_gpu ti = gpu_atomic_add_u32(banks.counters + 7, 1u);
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
                if (floorf(y) == y && y > 0.0f) {
                  int32_gpu extra = (int32_gpu)(y + 0.5f) - 1;
                  for (int32_gpu q = 0; q < extra; ++q) {
                    if (n_stack < GPU_MAX_SECONDARY_STACK) {
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
                      gpu_atomic_add_u32(
                        banks.counters + GPU_CTR_SECONDARY_BANK, 1u);
                    }
                  }
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
              GpuMgFission fe =
                gpu_mg_sample_fission_energy(mg, m, g, &seeds[stream]);
              float mu_f = 2.0f * gpu_prn(&seeds[stream]) - 1.0f;
              float phi = 6.283185307179586f * gpu_prn(&seeds[stream]);
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
            if (xs.absorption > gpu_prn(&seeds[stream]) * xs.total) {
              k_abs += wgt * xs.nu_fission / xs.absorption;
              wgt = 0.0f;
              absorbed = true;
            }
          }
          if (!absorbed) {
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
      }

      ++n_events;
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
  uint32_gpu slot = (tid >> 8) * GPU_RED_WIDTH;
  if (k_tl != 0.0f)
    gpu_atomic_add_f(banks.red_slots + slot + GPU_RED_K_TRACKLENGTH, k_tl);
  if (k_col != 0.0f)
    gpu_atomic_add_f(banks.red_slots + slot + GPU_RED_K_COLLISION, k_col);
  if (k_abs != 0.0f)
    gpu_atomic_add_f(banks.red_slots + slot + GPU_RED_K_ABSORPTION, k_abs);
  if (k_leak != 0.0f)
    gpu_atomic_add_f(banks.red_slots + slot + GPU_RED_LEAKAGE, k_leak);
  // progeny count with the leak flag in the top bit (debug diagnostics)
  banks.progeny[ctl.source_offset + tid] =
    (uint32_gpu)n_progeny | (leaked << 31);
}
