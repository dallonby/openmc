//! \file transport.h
//! The GPU transport loop: a faithful port of
//! transport_history_based_single_particle (simulation.cpp:958) and the
//! Particle::event_* chain (particle.cpp) for the v1 feature set
//! (multigroup neutrons, eigenvalue/fixed-source, analog physics).
//! One thread per source particle; fission sites appended to a device bank
//! with (parent_id, progeny_id) so the host-side sort_bank/synchronize_bank
//! reproduce CPU ordering semantics.

#pragma once

#ifdef GPU_HOST_DEBUG
struct GpuGeomState;
void gpu_host_debug_lost(THREAD const GpuGeomState* gs, int32_gpu tok,
  GpuVec3 r0, GpuVec3 u0, GpuVec3 u_new);
void gpu_host_debug_exit(
  THREAD const GpuGeomState* gs, float distance, float d_coll);
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

//! Match one filter; returns bin index or -1.
DEVICE_FN int32_gpu gpu_filter_match(GpuTallyView tv, GpuFilterDesc f,
  THREAD const GpuGeomState* gs, float E)
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
    // binary search for bin with edges[b] <= E < edges[b+1]
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
    if (i < 0 || i >= mesh.nx || j < 0 || j >= mesh.ny || k < 0 ||
        k >= mesh.nz)
      return -1;
    return mesh.nx * mesh.ny * k + mesh.nx * j + i;
  }
  }
  return -1;
}

//! Score all tallies of the given estimator for one event.
//! For tracklength events, coeff = wgt * distance; for collision events,
//! coeff = wgt / Sigma_t. Score value = coeff * Sigma_score (flux: coeff).
DEVICE_FN void gpu_score_tallies(GpuTallyView tv, GpuMgView mg,
  THREAD const GpuGeomState* gs, uint32_gpu estimator, float coeff, float E,
  int32_gpu g)
{
  for (uint32_gpu t = 0; t < tv.n_tallies; ++t) {
    GpuTallyDesc td = tv.tallies[t];
    if (td.estimator != estimator)
      continue;
    // filter evaluation
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
    // material macro data for score values
    GpuMacroXS xs;
    xs.total = 0.0f;
    xs.absorption = 0.0f;
    xs.fission = 0.0f;
    xs.nu_fission = 0.0f;
    float scat = 0.0f;
    if (gs->material >= 0) {
      xs = gpu_mg_calculate_xs(mg, gs->material, g, gs->density_mult);
      GpuMgMat m = mg.mats[gs->material];
      scat = gpu_mg_vec(mg, m, GPU_MGV_SCATT_XS, g) * gs->density_mult;
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
};

//! Full history for one source particle. Mirrors the CPU event loop.
DEVICE_FN void gpu_run_particle(uint32_gpu tid, GCONST GpuControl& ctl,
  GpuGeomData geom, GpuMgView mg, GpuTallyView tv, GpuBanks banks)
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

  float wgt = src.wgt;
  float time = src.time;
  int32_gpu g = 0;
  float E;
  if (ctl.energy_mode == GPU_MODE_MG) {
    g = (int32_gpu)src.E;
    E = geom.f32[ctl.mg_bin_avg_off + g];
  } else {
    E = src.E;
  }

  float k_tl = 0.0f, k_col = 0.0f, k_abs = 0.0f, k_leak = 0.0f;
  int32_gpu n_progeny = 0;
  uint32_gpu n_events = 0;

  bool found =
    gpu_exhaustive_find_cell(geom, &gs, ctl.root_universe, ctl.n_coord_levels);
  if (!found) {
    gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
    gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST_INIT, 1u);
    wgt = 0.0f;
  }

  // ---- event loop ----
  while (wgt > 0.0f) {
    // event_calculate_xs
    GpuMacroXS xs;
    xs.total = 0.0f;
    xs.absorption = 0.0f;
    xs.fission = 0.0f;
    xs.nu_fission = 0.0f;
    if (gs.material != GPU_MATERIAL_VOID) {
      xs = gpu_mg_calculate_xs(mg, gs.material, g, gs.density_mult);
    }

    // event_advance
    gs.boundary = gpu_distance_to_boundary(geom, &gs);
    if (gs.lost) {
      gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
      gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST_ADVANCE, 1u);
      break;
    }
    float d_coll = (xs.total <= 0.0f)
                     ? GPU_INFTY
                     : -logf(gpu_prn(&seeds[stream])) / xs.total;
    float distance = fminf(gs.boundary.d, d_coll);
    if (distance >= GPU_INFTY) {
      // nowhere to go: kill (mirrors lost handling)
      gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
      gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST_ADVANCE, 1u);
      break;
    }
    gpu_move_distance(&gs, distance);
#ifdef GPU_HOST_DEBUG
    if ((fabsf(gs.coord[0].r.x) > 1.89005f ||
          fabsf(gs.coord[0].r.y) > 1.89005f)) {
      gpu_host_debug_exit(&gs, distance, d_coll);
    }
#endif
    if (ctl.energy_mode == GPU_MODE_MG && gs.material != GPU_MATERIAL_VOID) {
      GpuMgMat m = mg.mats[gs.material];
      float inv_v = gpu_mg_vec(mg, m, GPU_MGV_INV_VELOCITY, g);
      time += distance * inv_v;
    }
    // tracklength keff estimator
    k_tl += wgt * distance * xs.nu_fission;
    // tracklength tallies
    if (tv.n_tallies > 0) {
      gpu_score_tallies(
        tv, mg, &gs, GPU_ESTIMATOR_TRACKLENGTH, wgt * distance, E, g);
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
          break;
        }
      } else {
        int32_gpu tok = gs.surface;
        int32_gpu i_surf = (tok > 0 ? tok : -tok) - 1;
        GpuSurface surf = geom.surfaces[i_surf];
        if (surf.bc == GPU_BC_VACUUM) {
          k_leak += wgt;
          wgt = 0.0f;
          break;
        } else if (surf.bc == GPU_BC_REFLECTIVE ||
                   surf.bc == GPU_BC_WHITE) {
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
            break;
          }
        } else {
          // transmission: search from the crossing level down
          if (!gpu_local_find_cell(geom, &gs, ctl.n_coord_levels)) {
            // fall back to a full search, then a nudged retry
            if (!gpu_exhaustive_find_cell(
                  geom, &gs, ctl.root_universe, ctl.n_coord_levels)) {
              gs.surface = GPU_SURFACE_NONE;
              gpu_move_distance(&gs, GPU_TINY_BIT);
              if (!gpu_exhaustive_find_cell(
                    geom, &gs, ctl.root_universe, ctl.n_coord_levels)) {
                gpu_atomic_add_u32(banks.counters + GPU_CTR_LOST, 1u);
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

      GpuMgMat m = mg.mats[gs.material];

      // fission bank (eigenvalue): physics_mg.cpp create_fission_sites
      if (m.fissionable && ctl.run_mode == GPU_RUN_EIGENVALUE) {
        float nu_t = wgt / ctl.keff * xs.nu_fission / xs.total;
        int32_gpu nu = (int32_gpu)nu_t;
        if (gpu_prn(&seeds[stream]) <= (nu_t - (float)nu))
          ++nu;
        for (int32_gpu n = 0; n < nu; ++n) {
          GpuMgFission fe =
            gpu_mg_sample_fission_energy(mg, m, g, &seeds[stream]);
          // isotropic direction, sampled inline (physics_mg.cpp:146)
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
          if (idx >= ctl.fission_bank_cap) {
            // bank full: mirror SharedArray clamp semantics
            break;
          }
          ++n_progeny;
          banks.fission[idx] = site;
        }
      }

      // collision-estimator tallies (scored with pre-collision weight)
      if (tv.n_tallies > 0) {
        gpu_score_tallies(
          tv, mg, &gs, GPU_ESTIMATOR_COLLISION, wgt / xs.total, E, g);
      }

      // absorption (analog): physics_mg.cpp:244 — the absorption estimator
      // scores only when the collision actually absorbs
      if (xs.absorption > 0.0f) {
        if (xs.absorption > gpu_prn(&seeds[stream]) * xs.total) {
          k_abs += wgt * xs.nu_fission / xs.absorption;
          wgt = 0.0f;
          break;
        }
      }

      // scatter
      float mu;
      int32_gpu gout =
        gpu_mg_sample_scatter(mg, m, g, &mu, &wgt, &seeds[stream]);
      GpuVec3 u_new = gpu_rotate_angle(gs.coord[0].u, mu, &seeds[stream]);
      // propagate direction down coordinate levels (rotation-aware)
      gs.coord[0].u = u_new;
      for (int32_gpu j = 1; j < gs.n_coord; ++j) {
        THREAD GpuCoord* parent = &gs.coord[j - 1];
        GpuCell pc = geom.cells[parent->cell];
        if (pc.rot_off >= 0) {
          gs.coord[j].u = gpu_rotate(parent->u, geom.f32 + pc.rot_off);
        } else {
          gs.coord[j].u = parent->u;
        }
      }
      g = gout;
      E = geom.f32[ctl.mg_bin_avg_off + g];
    }

    ++n_events;
    if (n_events >= ctl.max_events) {
      gpu_atomic_add_u32(banks.counters + GPU_CTR_MAX_EVENT_HIT, 1u);
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
  banks.progeny[ctl.source_offset + tid] = (uint32_gpu)n_progeny;
}
