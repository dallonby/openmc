//! \file metal_main.h
//! Metal kernel entry points. Buffer indices must match OmgSlot in
//! src/gpu/backend.h. This file is only meaningful in the concatenated MSL
//! source; CUDA and host builds provide their own wrappers.

#if defined(GPU_TARGET_METAL)

kernel void openmc_transport(constant GpuControl& ctl [[buffer(0)]],
  const device GpuSurface* surfaces [[buffer(1)]],
  const device GpuCell* cells [[buffer(2)]],
  const device GpuUniverse* universes [[buffer(3)]],
  const device GpuLattice* lattices [[buffer(4)]],
  const device GpuMaterial* materials [[buffer(5)]],
  const device GpuMgMat* mgmats [[buffer(6)]],
  const device int32_gpu* i32_arena [[buffer(7)]],
  const device float* f32_arena [[buffer(8)]],
  const device GpuSourceSite* source [[buffer(9)]],
  device GpuSourceSite* fission [[buffer(10)]],
  device gpu_atomic_u32* counters [[buffer(11)]],
  device uint32_gpu* progeny [[buffer(12)]],
  device float* red_slots [[buffer(13)]],
  const device GpuTallyDesc* tally_descs [[buffer(14)]],
  const device GpuFilterDesc* filter_descs [[buffer(15)]],
  const device GpuMesh* meshes [[buffer(16)]],
  device gpu_atomic_f32* tally_accum [[buffer(17)]],
  const device GpuNuclide* nuclides [[buffer(18)]],
  device GpuTraceRec* trace_buf [[buffer(19)]],
  device const GpuSabTable* sab_tables [[buffer(20)]],
  uint tid [[thread_position_in_grid]])
{
  // Persistent threads: when ctl.n_work_threads is set the grid is a fixed
  // pool and each thread pulls particle indices from a shared cursor. When it
  // is 0 the grid is one thread per particle (original behaviour).
  if (ctl.n_work_threads == 0 && tid >= ctl.n_particles)
    return;

  GpuGeomData geom;
  geom.surfaces = surfaces;
  geom.cells = cells;
  geom.universes = universes;
  geom.lattices = lattices;
  geom.materials = materials;
  geom.i32 = i32_arena;
  geom.surf_adj_off = ctl.surf_adj_off;
  geom.f32 = f32_arena;

  GpuCeView ce;
  ce.nuclides = nuclides;
  ce.i32 = i32_arena;
  ce.f32 = f32_arena;
  ce.n_nuclides = ctl.n_nuclides;
  ce.n_log_bins = ctl.ce_n_log_bins;
  ce.log_spacing = ctl.ce_log_spacing;
  ce.energy_min = ctl.energy_min;
  ce.energy_max = ctl.energy_max;
  ce.urr_on = ctl.urr_on;

  GpuMgView mg;
  mg.mats = mgmats;
  mg.i32 = i32_arena;
  mg.f32 = f32_arena;
  mg.n_groups = ctl.n_groups;

  GpuSabView sab;
  sab.tables = sab_tables;
  sab.i32 = i32_arena;
  sab.f32 = f32_arena;

  GpuTallyView tv;
  tv.tallies = tally_descs;
  tv.filters = filter_descs;
  tv.meshes = meshes;
  tv.i32 = i32_arena;
  tv.f32 = f32_arena;
  // replicated accumulator banks: keeps per-bank fp32 sums small enough
  // that individual track contributions never round away (see GpuControl)
  tv.accum =
    tally_accum + (tid % ctl.tally_replicas) * ctl.tally_accum_stride;
  tv.n_tallies = ctl.n_tallies;

  GpuBanks banks;
  banks.source = source;
  banks.fission = fission;
  banks.counters = counters;
  banks.progeny = progeny;
  banks.red_slots = red_slots;
  banks.trace = trace_buf;

  float ev = 0.0f;
  if (ctl.n_work_threads == 0) {
    gpu_run_particle(tid, ctl, geom, mg, ce, sab, tv, banks);
    ev = red_slots[tid * GPU_RED_WIDTH + GPU_RED_EVENTS];
  } else {
    for (;;) {
      uint32_gpu i = gpu_atomic_add_u32(counters + GPU_CTR_WORK, 1u);
      if (i >= ctl.n_particles)
        break;
      gpu_run_particle(i, ctl, geom, mg, ce, sab, tv, banks);
      ev += red_slots[i * GPU_RED_WIDTH + GPU_RED_EVENTS];
    }
  }

  // Lane utilisation over the whole dispatch: the group cannot retire until
  // its busiest lane is done, so this is the fraction of lane-slots that did
  // real work. Accumulated through counters because with a work queue a
  // thread no longer maps to one particle slot.
  float mx = simd_max(ev);
  float sm = simd_sum(ev);
  float width = (float)simd_sum(1.0f);
  float eff = (mx > 0.0f && width > 0.0f) ? sm / (width * mx) : 0.0f;
  gpu_atomic_add_u32(
    counters + GPU_CTR_SIMDEFF_ACC, (uint32_gpu)(eff * 100.0f));
  gpu_atomic_add_u32(counters + GPU_CTR_SIMDEFF_N, 1u);
}

// Math-function probe: y = f(x) for the host to compare against libm
kernel void openmc_math_probe(const device float* x [[buffer(0)]],
  device float* y [[buffer(1)]], constant uint& which [[buffer(2)]],
  uint tid [[thread_position_in_grid]])
{
  float v = x[tid];
  switch (which) {
  case 0:
    y[tid] = p_logf(v);
    break;
  case 6:
    y[tid] = p_expf(v);
    break;
  case 7:
    y[tid] = p_sinf(v);
    break;
  case 8:
    y[tid] = p_cosf(v);
    break;
  case 9:
    y[tid] = p_sqrtf(v);
    break;
  case 1:
    y[tid] = expf(v);
    break;
  case 2:
    y[tid] = sinf(v);
    break;
  case 3:
    y[tid] = cosf(v);
    break;
  case 4:
    y[tid] = sqrtf(v);
    break;
  default:
    y[tid] = 1.0f / v;
    break;
  }
}

// Bit-exactness self-test: advances OpenMC PRN streams on-device so the host
// can verify integer-identical RNG behaviour at initialization.
kernel void openmc_rng_selftest(
  device uint64_gpu* io [[buffer(0)]], uint tid [[thread_position_in_grid]])
{
  // io layout per thread: [seed_in, n_steps, word_out, seed_out]
  device uint64_gpu* rec = io + 4 * tid;
  uint64_gpu seed = rec[0];
  uint64_gpu n = rec[1];
  uint64_gpu w = 0;
  for (uint64_gpu i = 0; i < n; ++i) {
    THREAD uint64_gpu s = seed;
    w = gpu_prn_word(&s);
    seed = s;
  }
  rec[2] = w;
  rec[3] = seed;
}

#endif // GPU_TARGET_METAL
