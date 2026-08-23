// Host replay/audit instrumentation hooks in the device headers activate
// under this define; it must precede every device-header inclusion.
#define GPU_HOST_DEBUG

//! \file runner.cpp
//! Orchestrates GPU transport generations: flatten → upload → dispatch →
//! feed results back into OpenMC's host-side (fp64) statistics pipeline.
//! Everything outside the particle flight — bank synchronization, keff
//! statistics, tally accumulation across batches, statepoints — is untouched
//! upstream code, which is what keeps GPU runs drop-in compatible.

#include "openmc/gpu_interface.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fmt/format.h>

#include "openmc/bank.h"
#include "openmc/capi.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/material.h"
#include "openmc/math_functions.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/particle_data.h"
#include "openmc/random_lcg.h"
#include "openmc/reaction.h"
#include "openmc/reaction_product.h"
#include "openmc/secondary_uncorrelated.h"

namespace openmc {
// file-scope globals in random_lcg.cpp with external linkage
extern int64_t master_seed;
extern uint64_t prn_stride;
} // namespace openmc
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/source.h"
#include "openmc/tallies/tally.h"

#include "backend.h"
#include "flatten.h"

// Host replay of the device transport loop for lost-particle diagnosis:
// the same single-source kernels compile for the host, and RNG streams are
// identical, so lost particles reproduce deterministically (modulo fastmath
// contraction differences).
#include "device/transport.h"

void gpu_host_debug_escape(const GpuGeomState* gs)
{
  std::fprintf(stderr,
    "[gpu-debug] escape-after-lattice-cross r=(%.7f %.7f %.7f) u=(%.6f %.6f "
    "%.6f) surface=%d li=(%d %d %d) cell0=%d\n",
    gs->coord[0].r.x, gs->coord[0].r.y, gs->coord[0].r.z, gs->coord[0].u.x,
    gs->coord[0].u.y, gs->coord[0].u.z, gs->surface,
    gs->coord[gs->n_coord - 1].li[0], gs->coord[gs->n_coord - 1].li[1],
    gs->coord[gs->n_coord - 1].li[2], gs->coord[0].cell);
}

namespace {
struct ReplayStats {
  double n_coll = 0, n_el = 0, n_lvl = 0, n_cont = 0, n_xn = 0, n_abs = 0,
         n_fs = 0;
  double sE = 0, smu = 0, ser = 0, sr = 0, srf = 0;
  double n_fl = 0, sd = 0;
} rstats;
} // namespace

void gpu_host_stat_collision(float E, float r)
{
  rstats.n_coll += 1;
  rstats.sE += E;
  rstats.sr += r;
}
void gpu_host_stat_elastic(float mu_lab, float eratio)
{
  rstats.n_el += 1;
  rstats.smu += mu_lab;
  rstats.ser += eratio;
}
void gpu_host_stat_inelastic(int32_gpu mt)
{
  if (mt >= 51 && mt <= 90)
    rstats.n_lvl += 1;
  else if (mt == 91)
    rstats.n_cont += 1;
  else
    rstats.n_xn += 1;
}
namespace {
int64_t gpu_trace_id()
{
  static int64_t v = std::getenv("OPENMC_TRACE_ID")
                       ? std::atoll(std::getenv("OPENMC_TRACE_ID"))
                       : -1;
  return v;
}
} // namespace

namespace {
std::vector<uint8_t> gpu_leak_map;
} // namespace

namespace openmc {
namespace gpu {
//! set by finalize() so the next simulation in this process does not reuse
//! the previous model's prefetched source sites
bool gpu_prefetch_reset = true;
} // namespace gpu
} // namespace openmc

void gpu_host_leak(int64_gpu id)
{
  if (!gpu_leak_map.empty() && (size_t)(id - 1) < gpu_leak_map.size())
    gpu_leak_map[id - 1] = 1;
}

void gpu_host_trace_fly(int64_gpu id, float d, float dc, float db, float E)
{
  if (id == gpu_trace_id())
    std::fprintf(
      stderr, "[T-gpu] fly d=%a seedlo=%.0f bdry=%a E=%a\n", d, dc, db, E);
}
void gpu_host_trace_collide(int64_gpu id, float E, GpuVec3 r, int32_gpu nuc)
{
  if (id == gpu_trace_id())
    std::fprintf(stderr, "[T-gpu] collide E=%a rx=%a nuc=%d\n", E, r.x, nuc);
}
void gpu_host_trace_postfis(
  int64_gpu id, uint32_gpu seedlo, int32_gpu nprog, uint32_gpu urrlo)
{
  if (id == gpu_trace_id())
    std::fprintf(stderr, "[T-gpu] postfis seedlo=%u nprog=%d urrlo=%u\n",
      seedlo, nprog, urrlo);
}

void gpu_host_trace_elastic(int64_gpu id, float E)
{
  if (id == gpu_trace_id())
    std::fprintf(stderr, "[T-gpu]   elastic E'=%a\n", E);
}
void gpu_host_trace_inelastic(int64_gpu id, int32_gpu mt)
{
  if (id == gpu_trace_id())
    std::fprintf(stderr, "[T-gpu]   inelastic MT=%d\n", mt);
}

void gpu_host_stat_flight(float d)
{
  rstats.n_fl += 1;
  rstats.sd += d;
}
void gpu_host_stat_absorb()
{
  rstats.n_abs += 1;
}
void gpu_host_stat_fsite(float r)
{
  rstats.n_fs += 1;
  rstats.srf += r;
}

void gpu_host_debug_tab(float r1, int32_gpu k, float ck, float xk, float pk,
  float x1, float p1, int32_gpu interp, int32_gpu nd)
{
  if (!std::getenv("OPENMC_GPU_AUDIT3"))
    return;
  std::fprintf(stderr,
    "[tab] r1=%.6f k=%d ck=%.6f xk=%+.4f pk=%.4f x1=%+.4f p1=%.4f "
    "interp=%d nd=%d\n",
    r1, k, ck, xk, pk, x1, p1, interp, nd);
}

void gpu_host_debug_angle(int32_gpu i, float r1, int32_gpu k, float ck,
  int32_gpu n_mu, int32_gpu interp, float xk, float pk, float m)
{
  if (!std::getenv("OPENMC_GPU_AUDIT3"))
    return;
  std::fprintf(stderr,
    "[angle] i=%d r1=%.6f k=%d ck=%.6f n_mu=%d interp=%d xk=%+.4f "
    "pk=%.4f -> m=%+.6f\n",
    i, r1, k, ck, n_mu, interp, xk, pk, m);
}

void gpu_host_debug_exit(const GpuGeomState* gs, float distance, float d_coll)
{
  static int count = 0;
  if (count++ > 5)
    return;
  std::fprintf(stderr,
    "[gpu-debug] EXIT r=(%.7f %.7f %.7f) u=(%.6f %.6f %.6f) d=%.7f "
    "d_coll=%.6g bdry{d=%.7f surf=%d lvl=%d trans=(%d %d %d)} on_surf=%d "
    "ncoord=%d\n",
    gs->coord[0].r.x, gs->coord[0].r.y, gs->coord[0].r.z, gs->coord[0].u.x,
    gs->coord[0].u.y, gs->coord[0].u.z, distance, d_coll, gs->boundary.d,
    gs->boundary.surface, gs->boundary.coord_level, gs->boundary.lat_trans[0],
    gs->boundary.lat_trans[1], gs->boundary.lat_trans[2], gs->surface,
    gs->n_coord);
}

void gpu_host_debug_lost_where(int where, const GpuGeomState* gs, float E)
{
  std::fprintf(stderr, "[gpu-debug] lost(site=%d) E=%g surf=%d ncoord=%d mat=%d\n",
    where, E, gs->surface, gs->n_coord, gs->material);
  for (int j = 0; j < gs->n_coord; ++j)
    std::fprintf(stderr,
      "  L%d cell=%d univ=%d lat=%d li=(%d %d %d) r=(%.7f %.7f %.7f) "
      "u=(%.6f %.6f %.6f)\n",
      j, gs->coord[j].cell, gs->coord[j].universe, gs->coord[j].lattice,
      gs->coord[j].li[0], gs->coord[j].li[1], gs->coord[j].li[2],
      gs->coord[j].r.x, gs->coord[j].r.y, gs->coord[j].r.z, gs->coord[j].u.x,
      gs->coord[j].u.y, gs->coord[j].u.z);
}

void gpu_host_debug_lost(
  const GpuGeomState* gs, int32_gpu tok, GpuVec3 r0, GpuVec3 u0, GpuVec3 u_new)
{
  std::fprintf(stderr,
    "[gpu-debug] reflect-lost tok=%d r=(%.7f %.7f %.7f) u_in=(%.6f %.6f "
    "%.6f) u_out=(%.6f %.6f %.6f) ncoord=%d cell0=%d\n",
    tok, r0.x, r0.y, r0.z, u0.x, u0.y, u0.z, u_new.x, u_new.y, u_new.z,
    gs->n_coord, gs->coord[0].cell);
}

extern "C" {
// generated by cmake/EmbedMsl.cmake from src/gpu/device/*.h
extern const unsigned char openmc_gpu_msl[];
extern const unsigned long openmc_gpu_msl_len;
}

namespace openmc {
namespace gpu {

namespace {

struct Engine {
  double t_source = 0.0, t_dispatch = 0.0, t_post = 0.0; // host phase timers
  void* ctx = nullptr;
  FlatModel flat;
  bool active = false;
  std::string reason;
  double gpu_seconds = 0.0;
  int64_t lost_total = 0;
  uint32_t n_red_slots = 0;
  uint32_t tally_replicas = 1;
  uint64_t spill_uid_cursor = 0; // monotonic spill-id allocator
  double total_events = 0.0;     // summed device events (perf diagnostics)
  double total_xseval = 0.0;     // summed cross-section evaluations
  double simdeff_sum = 0.0;      // summed per-thread SIMD-group efficiency
  double simdeff_n = 0.0;
};

Engine eng;

int overall_generation_()
{
  return settings::gen_per_batch * (simulation::current_batch - 1) +
         simulation::current_gen;
}

bool fail(std::string why)
{
  eng.reason = std::move(why);
  if (eng.ctx) {
    omg_metal_destroy(eng.ctx);
    eng.ctx = nullptr;
  }
  eng.active = false;
  warning(fmt::format(
    "GPU transport disabled — {}. Running on the CPU instead.", eng.reason));
  return false;
}

template<typename T>
bool upload(int slot, const std::vector<T>& v)
{
  size_t bytes = v.size() * sizeof(T);
  if (omg_metal_buffer(eng.ctx, slot, bytes))
    return false;
  if (!v.empty())
    std::memcpy(omg_metal_contents(eng.ctx, slot), v.data(), bytes);
  return true;
}

//! Verify the device PRN generator is integer-identical to the host's.
bool rng_selftest()
{
  const int N = 64;
  std::vector<uint64_t> io(4 * N);
  for (int i = 0; i < N; ++i) {
    io[4 * i] = 0x9e3779b97f4a7c15ull * (uint64_t)(i + 1) + 12345ull;
    io[4 * i + 1] = 1000 + 17 * (uint64_t)i;
  }
  if (omg_metal_buffer(eng.ctx, 0, io.size() * 8))
    return false;
  std::memcpy(omg_metal_contents(eng.ctx, 0), io.data(), io.size() * 8);
  char err[512];
  if (omg_metal_dispatch(eng.ctx, "openmc_rng_selftest", N, err, sizeof(err)))
    return false;
  auto* out = static_cast<uint64_t*>(omg_metal_contents(eng.ctx, 0));
  for (int i = 0; i < N; ++i) {
    uint64_t seed = io[4 * i];
    uint64_t n = io[4 * i + 1];
    uint64_t word = 0;
    uint64_t host_seed = seed;
    double last = 0.0;
    for (uint64_t k = 0; k < n; ++k)
      last = prn(&host_seed);
    // recompute the output word with the host-compiled device generator
    uint64_t s2 = seed;
    for (uint64_t k = 0; k < n; ++k) {
      uint64_t tmp = s2;
      word = gpu_prn_word(&tmp);
      s2 = tmp;
    }
    if (out[4 * i + 3] != host_seed || out[4 * i + 3] != s2 ||
        out[4 * i + 2] != word ||
        std::abs(std::ldexp((double)word, -64) - last) > 1e-15) {
      return false;
    }
  }
  return true;
}

} // namespace

const std::string& inactive_reason()
{
  return eng.reason;
}

bool active()
{
  return eng.active;
}

void try_initialize()
{
  if (!settings::gpu)
    return;
  if (eng.active)
    return;

#ifdef OPENMC_USE_MPI
  if (mpi::n_procs > 1) {
    fail("MPI runs are not in the GPU v1 envelope");
    return;
  }
#endif
  if (settings::run_mode != RunMode::EIGENVALUE &&
      settings::run_mode != RunMode::FIXED_SOURCE) {
    fail("only eigenvalue and fixed-source runs are in the GPU envelope");
    return;
  }
  if (simulation::work_per_rank > 0x7fffffffLL) {
    fail("more than 2^31 particles per generation");
    return;
  }
  if (!omg_metal_available()) {
    fail("no Metal device available");
    return;
  }

  if (!flatten_model(eng.flat)) {
    fail(eng.flat.reject_reason);
    return;
  }

  eng.ctx = omg_metal_create();
  if (!eng.ctx) {
    fail("could not create Metal context");
    return;
  }

  // compile device source, specialized to this model: the per-thread array
  // bounds (nuclides per material, coordinate depth, filters per tally,
  // secondary stack) size the kernel's stack/register footprint and hence
  // occupancy — the worst-case defaults cost up to ~2x on simple models
  uint32_t max_nuc = 1;
  for (const auto& m : eng.flat.materials)
    max_nuc = std::max(max_nuc, m.n_nuclides);
  uint32_t max_filt = 1;
  for (const auto& t : eng.flat.tallies)
    max_filt = std::max(max_filt, t.n_filters);
  // diagnostic override: force a larger per-thread nuclide array without
  // changing the physics, to separate "thread-local footprint limits
  // parallelism" from "per-nuclide work"
  if (const char* e = std::getenv("OPENMC_GPU_FORCE_MAXNUC")) {
    uint32_t v = (uint32_t)atoi(e);
    if (v > max_nuc)
      max_nuc = v;
  }
  // diagnostic: shrink the per-thread secondary stack to test whether live
  // private state, not instruction count, bounds this kernel
  uint32_t sec_stack = settings::run_CE ? 8 : 1;
  if (const char* e = std::getenv("OPENMC_GPU_SEC_STACK"))
    sec_stack = (uint32_t)std::max(1, atoi(e));
  int ballast = 0;
  if (const char* e = std::getenv("OPENMC_GPU_BALLAST"))
    ballast = std::max(0, atoi(e));
  std::string preamble = fmt::format(
    "#define GPU_MAX_MAT_NUCLIDES {}\n#define GPU_MAX_COORD {}\n"
    "#define GPU_MAX_TALLY_FILTERS {}\n#define GPU_MAX_SECONDARY_STACK {}\n"
    "#define GPU_DIAG_BALLAST {}\n",
    max_nuc, model::n_coord_levels + 1, max_filt, sec_stack, ballast);
  std::string src = preamble +
    std::string(reinterpret_cast<const char*>(openmc_gpu_msl),
      (size_t)openmc_gpu_msl_len);
  char err[4096];
  if (omg_metal_compile(eng.ctx, src.c_str(), err, sizeof(err))) {
    fail(fmt::format("device kernel compilation failed: {}", err));
    return;
  }

  if (!rng_selftest()) {
    fail("device RNG failed the bit-exactness self-test");
    return;
  }

  // static data upload
  if (!upload(OMG_SLOT_SURFACES, eng.flat.surfaces) ||
      !upload(OMG_SLOT_CELLS, eng.flat.cells) ||
      !upload(OMG_SLOT_UNIVERSES, eng.flat.universes) ||
      !upload(OMG_SLOT_LATTICES, eng.flat.lattices) ||
      !upload(OMG_SLOT_MATERIALS, eng.flat.materials) ||
      !upload(OMG_SLOT_MGMATS, eng.flat.mgmats) ||
      !upload(OMG_SLOT_I32, eng.flat.i32) ||
      !upload(OMG_SLOT_F32, eng.flat.f32) ||
      !upload(OMG_SLOT_NUCLIDES, eng.flat.nuclides) ||
      !upload(OMG_SLOT_SAB, eng.flat.sab_tables) ||
      !upload(OMG_SLOT_TALLIES, eng.flat.tallies) ||
      !upload(OMG_SLOT_FILTERS, eng.flat.filters) ||
      !upload(OMG_SLOT_MESHES, eng.flat.meshes)) {
    fail("device buffer allocation failed");
    return;
  }

  // dynamic buffers
  int64_t n = simulation::work_per_rank;
  eng.n_red_slots = (uint32_t)n; // one slot group per particle
  // replicated tally banks: enough that per-bank per-batch sums stay far
  // from the fp32 integer boundary (2^24), capped at 128 MB of banks
  eng.tally_replicas = 64;
  while (eng.tally_replicas > 1 &&
         (size_t)eng.flat.tally_accum_size * eng.tally_replicas * 4 >
           (size_t)128 << 20)
    eng.tally_replicas >>= 1;
  if (eng.tally_replicas < 8 && eng.flat.tally_accum_size > 0) {
    warning(fmt::format(
      "GPU tally accumulators fit only {} replica bank(s) ({} bins); very "
      "large batches may lose small track contributions to fp32 rounding in "
      "hot bins. Reduce particles per batch if tallies look low.",
      eng.tally_replicas, eng.flat.tally_accum_size));
  }
  int64_t bank_cap = 3 * n;
  if (omg_metal_buffer(eng.ctx, OMG_SLOT_CONTROL, sizeof(GpuControl)) ||
      omg_metal_buffer(
        eng.ctx, OMG_SLOT_SOURCE, (size_t)n * sizeof(GpuSourceSite)) ||
      omg_metal_buffer(
        eng.ctx, OMG_SLOT_FISSION, (size_t)bank_cap * sizeof(GpuSourceSite)) ||
      omg_metal_buffer(
        eng.ctx, OMG_SLOT_COUNTERS, GPU_CTR_COUNT * sizeof(uint32_t)) ||
      omg_metal_buffer(eng.ctx, OMG_SLOT_PROGENY, (size_t)n * 4) ||
      omg_metal_buffer(eng.ctx, OMG_SLOT_REDSLOTS,
        (size_t)eng.n_red_slots * GPU_RED_WIDTH * 4) ||
      omg_metal_buffer(eng.ctx, OMG_SLOT_TACCUM,
        std::max<size_t>(1, eng.flat.tally_accum_size) *
          (size_t)eng.tally_replicas * 4) ||
      omg_metal_buffer(
        eng.ctx, OMG_SLOT_TRACE, GPU_TRACE_MAX * sizeof(GpuTraceRec))) {
    fail("device bank allocation failed");
    return;
  }

  eng.active = true;
  eng.reason.clear();
  write_message(
    fmt::format("GPU transport engine active on {} ({} surfaces, {} cells, "
                "{} materials, {} tallies flattened)",
      omg_metal_device_name(eng.ctx), eng.flat.surfaces.size(),
      eng.flat.cells.size(), eng.flat.mgmats.size(), eng.flat.tallies.size()),
    5);
}

void transport_generation()
{
  auto t_gen0 = std::chrono::steady_clock::now();
  int64_t n = simulation::work_per_rank;

  // ---- control block ----
  auto* ctl =
    static_cast<GpuControl*>(omg_metal_contents(eng.ctx, OMG_SLOT_CONTROL));
  std::memset(ctl, 0, sizeof(GpuControl));
  ctl->n_particles = (uint32_t)n;
  ctl->source_offset = 0;
  ctl->run_mode = (settings::run_mode == RunMode::FIXED_SOURCE)
                    ? GPU_RUN_FIXED_SOURCE
                    : GPU_RUN_EIGENVALUE;
  ctl->energy_mode = settings::run_CE ? GPU_MODE_CE : GPU_MODE_MG;
  ctl->keff = (float)simulation::keff;
  if (!settings::run_CE)
    ctl->n_groups = (uint32_t)data::mg.num_energy_groups_;
  ctl->n_nuclides = (uint32_t)eng.flat.nuclides.size();
  ctl->ce_n_log_bins = (uint32_t)settings::n_log_bins;
  ctl->ce_log_spacing = (float)simulation::log_spacing;
  ctl->energy_min = (float)data::energy_min[0];
  ctl->energy_max = (float)data::energy_max[0];
  ctl->urr_on = settings::urr_ptables_on ? 1u : 0u;
  ctl->debug_iso_mu = std::getenv("OPENMC_ISO_MU") ? 1u : 0u;
  ctl->trace_id = (int32_t)gpu_trace_id();
  ctl->max_events =
    (uint32_t)std::min<int64_t>(settings::max_particle_events, 0x7fffffff);
  ctl->seed_base =
    (uint64_t)((simulation::total_gen + (int64_t)overall_generation_() - 1) *
               settings::n_particles);
  ctl->master_seed = (uint64_t)openmc::master_seed;
  ctl->prn_stride = openmc::prn_stride;
  ctl->fission_bank_cap = (uint32_t)(3 * n);
  ctl->root_universe = model::root_universe;
  ctl->n_coord_levels = model::n_coord_levels;
  ctl->n_cells = (uint32_t)eng.flat.cells.size();
  ctl->n_surfaces = (uint32_t)eng.flat.surfaces.size();
  ctl->mg_bin_avg_off = eng.flat.mg_bin_avg_off;
  ctl->mg_default_iv_off = eng.flat.mg_default_iv_off;
  ctl->energy_cutoff = (float)settings::energy_cutoff[0];
  ctl->free_gas_threshold = (float)settings::free_gas_threshold;
  ctl->tally_accum_stride = eng.flat.tally_accum_size;
  ctl->tally_replicas = eng.tally_replicas;
  ctl->surf_adj_off = eng.flat.surf_adj_off;
  // variance reduction (fixed-source only; flatten enforces that)
  ctl->survival_biasing = settings::survival_biasing ? 1u : 0u;
  ctl->weight_cutoff = (float)settings::weight_cutoff;
  ctl->weight_survive = (float)settings::weight_survive;
  ctl->ww_on = (eng.flat.ww_mesh >= 0) ? 1u : 0u;
  ctl->ww_mesh = eng.flat.ww_mesh;
  ctl->ww_n_energy = eng.flat.ww_n_energy;
  ctl->ww_ebounds_off = eng.flat.ww_ebounds_off;
  ctl->ww_lower_off = eng.flat.ww_lower_off;
  ctl->ww_upper_off = eng.flat.ww_upper_off;
  ctl->ww_n_mesh_bins = eng.flat.ww_n_mesh_bins;
  ctl->ww_survival_ratio = (float)eng.flat.ww_survival_ratio;
  ctl->ww_max_lb_ratio = (float)eng.flat.ww_max_lb_ratio;
  ctl->ww_weight_cutoff = (float)eng.flat.ww_weight_cutoff;
  ctl->ww_max_split = eng.flat.ww_max_split;
  ctl->ww_max_history_splits = (int32_t)settings::max_history_splits;
  ctl->ww_checkpoint_collision =
    settings::weight_window_checkpoint_collision ? 1u : 0u;
  ctl->ww_checkpoint_surface =
    settings::weight_window_checkpoint_surface ? 1u : 0u;
  ctl->spill_cap = ctl->fission_bank_cap;
  ctl->source_is_spill = 0u;
  // Spilled secondaries draw particle ids from a space disjoint from the
  // primaries' (id = batch*gen*n_particles stays far below 2^50), advanced
  // by every spill already produced so ids never repeat across generations.
  {
    const uint64_t base = (1ull << 50) + eng.spill_uid_cursor;
    ctl->spill_uid_lo = (uint32_t)(base & 0xffffffffull);
    ctl->spill_uid_hi = (uint32_t)(base >> 32);
  }

  // active tallies only (device scores every desc it is told about)
  bool tallies_active = false;
  for (const auto& t : model::tallies)
    if (t->active_) {
      tallies_active = true;
      break;
    }
  ctl->n_tallies = tallies_active ? (uint32_t)eng.flat.tallies.size() : 0;
  // do the active tallies score elastic / scatter at all?
  ctl->need_elastic = 0u;
  ctl->need_scatter = 0u;
  if (ctl->n_tallies > 0) {
    for (size_t ti = 0; ti < eng.flat.tallies.size(); ++ti) {
      const GpuTallyDesc& td = eng.flat.tallies[ti];
      int32_t host_idx = eng.flat.tally_host_index[ti];
      if (!model::tallies[host_idx]->active_)
        continue;
      for (uint32_t k = 0; k < td.n_scores; ++k) {
        int32_t code = eng.flat.i32[td.score_off + k];
        if (code == GPU_SCORE_ELASTIC)
          ctl->need_elastic = 1u;
        if (code == GPU_SCORE_SCATTER)
          ctl->need_scatter = 1u;
      }
    }
  }

  // ---- source upload (fp64 -> fp32) ----
  auto* src =
    static_cast<GpuSourceSite*>(omg_metal_contents(eng.ctx, OMG_SLOT_SOURCE));
  double src_weight = 0.0;
  auto fill_site = [](GpuSourceSite& g, const SourceSite& s) {
    g.r[0] = (float)s.r.x;
    g.r[1] = (float)s.r.y;
    g.r[2] = (float)s.r.z;
    g.u[0] = (float)s.u.x;
    g.u[1] = (float)s.u.y;
    g.u[2] = (float)s.u.z;
    g.E = (float)s.E;
    g.time = (float)s.time;
    g.wgt = (float)s.wgt;
    g.delayed_group = s.delayed_group;
    g.parent_id = 0;
    g.progeny_id = 0;
    g.uid_lo = 0;
    g.uid_hi = 0;
    g.wgt_born = (float)s.wgt;
    g.wgt_ww_born = -1.0f; // unset: the first window lookup fixes it
    g.ww_factor = 0.0f;
    g.n_split = 0;
  };
  // Fixed-source sites do not depend on transport results, so the NEXT
  // generation's sites are sampled while this generation's kernel runs
  // (measured: host sampling was ~22% of wall time). The prefetch uses the
  // next generation's seed base (+n_particles: only overall_generation
  // advances within a run) and is bit-identical to on-demand sampling;
  // OPENMC_GPU_NO_PREFETCH=1 disables it for that check.
  static std::vector<GpuSourceSite> prefetched;
  static double prefetched_weight = 0.0;
  static bool have_prefetch = false;
  if (gpu_prefetch_reset) { // cleared by finalize(); see gpu_prefetch_reset
    prefetched.clear();
    prefetched_weight = 0.0;
    have_prefetch = false;
    gpu_prefetch_reset = false;
  }
  // Prefetch assumes the next generation's ids are simply +n_particles.
  // With the shared secondary bank (which weight windows enable by default)
  // compute_particle_id() also folds in simulation_tracks_completed, so that
  // assumption breaks and the same source sites would be resampled.
  const bool prefetch_ok = settings::run_mode == RunMode::FIXED_SOURCE &&
                           !settings::use_shared_secondary_bank &&
                           !std::getenv("OPENMC_GPU_NO_PREFETCH");
  if (std::getenv("OPENMC_GPU_PREFETCH_DEBUG"))
    std::fprintf(stderr,
      "[prefetch-debug] batch %d gen %d overall %d total_gen %lld id(1)=%lld "
      "have_prefetch=%d\n",
      simulation::current_batch, simulation::current_gen, overall_generation_(),
      (long long)simulation::total_gen,
      (long long)compute_transport_seed(compute_particle_id(1)),
      (int)have_prefetch);
  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    if (have_prefetch) {
      std::memcpy(src, prefetched.data(), (size_t)n * sizeof(GpuSourceSite));
      src_weight = prefetched_weight;
      have_prefetch = false;
    } else {
      // CPU fixed-source samples the external source on the fly per history
      // (initialize_history): host-sample here with the identical fp64
      // upstream code and per-particle STREAM_SOURCE seed discipline
#pragma omp parallel for reduction(+ : src_weight)
      for (int64_t i = 0; i < n; ++i) {
        int64_t id = compute_transport_seed(compute_particle_id(i + 1));
        uint64_t seed = init_seed(id, STREAM_SOURCE);
        SourceSite s = sample_external_source(&seed);
        fill_site(src[i], s);
        src_weight += s.wgt;
      }
    }
  } else {
    for (int64_t i = 0; i < n; ++i) {
      const SourceSite& s = simulation::source_bank[i];
      fill_site(src[i], s);
      src_weight += s.wgt;
    }
  }
  simulation::total_weight += src_weight;
  if (std::getenv("OPENMC_GPU_PREFETCH_DEBUG")) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(src);
    uint64_t h = 1469598103934665603ull;
    for (size_t k = 0; k < (size_t)n * sizeof(GpuSourceSite) / 4; ++k)
      h = (h ^ w[k]) * 1099511628211ull;
    std::fprintf(stderr, "[prefetch-debug] source hash %016llx\n",
      (unsigned long long)h);
  }
  auto t_gen1 = std::chrono::steady_clock::now();

  // ---- zero per-generation accumulators ----
  std::memset(omg_metal_contents(eng.ctx, OMG_SLOT_COUNTERS), 0,
    GPU_CTR_COUNT * sizeof(uint32_t));
  std::memset(omg_metal_contents(eng.ctx, OMG_SLOT_PROGENY), 0, (size_t)n * 4);
  std::memset(omg_metal_contents(eng.ctx, OMG_SLOT_REDSLOTS), 0,
    (size_t)eng.n_red_slots * GPU_RED_WIDTH * 4);
  if (eng.flat.tally_accum_size > 0)
    std::memset(omg_metal_contents(eng.ctx, OMG_SLOT_TACCUM), 0,
      (size_t)eng.flat.tally_accum_size * eng.tally_replicas * 4);

  // ---- device math-function bias probe ----
  if (std::getenv("OPENMC_GPU_MATHPROBE")) {
    const int N = 1000000;
    omg_metal_buffer(eng.ctx, 0, N * 4);
    omg_metal_buffer(eng.ctx, 1, N * 4);
    omg_metal_buffer(eng.ctx, 2, 4);
    float* x = (float*)omg_metal_contents(eng.ctx, 0);
    float* y = (float*)omg_metal_contents(eng.ctx, 1);
    uint32_t* wsel = (uint32_t*)omg_metal_contents(eng.ctx, 2);
    const char* names[10] = {"p_log", "exp", "sin", "cos", "sqrt", "recip",
      "p_exp", "p_sin", "p_cos", "p_sqrt"};
    uint64_t seed = 777;
    for (int w : {0, 6, 7, 8, 9}) {
      for (int i = 0; i < N; ++i) {
        double u = prn(&seed);
        switch (w) {
        case 0:
          x[i] = (float)u; // log over (0,1): the -log(xi) flight case
          break;
        case 6:
          x[i] = (float)(-20.0 + 40.0 * u);
          break;
        case 7:
        case 8:
          x[i] = (float)(6.283185307179586 * u);
          break;
        default:
          x[i] = (float)(u * 100.0);
          break;
        }
      }
      *wsel = (uint32_t)w;
      char err[256];
      omg_metal_dispatch(eng.ctx, "openmc_math_probe", N, err, sizeof(err));
      double sum_rel = 0, max_rel = 0, n_diff = 0;
      for (int i = 0; i < N; ++i) {
        double ref;
        switch (w) {
        case 0:
          ref = std::log((double)x[i]);
          break;
        case 1:
          ref = std::exp((double)x[i]);
          break;
        case 2:
          ref = std::sin((double)x[i]);
          break;
        case 3:
          ref = std::cos((double)x[i]);
          break;
        case 4:
          ref = std::sqrt((double)x[i]);
          break;
        default:
          ref = 1.0 / (double)x[i];
          break;
        }
        double rel = (ref != 0.0) ? ((double)y[i] - ref) / ref : 0.0;
        sum_rel += rel;
        if (std::abs(rel) > std::abs(max_rel))
          max_rel = rel;
        float host_f;
        switch (w) {
        case 0:
          host_f = p_logf(x[i]);
          break;
        case 6:
          host_f = p_expf(x[i]);
          break;
        case 7:
          host_f = p_sinf(x[i]);
          break;
        case 8:
          host_f = p_cosf(x[i]);
          break;
        default:
          host_f = p_sqrtf(x[i]);
          break;
        }
        if (host_f != y[i])
          n_diff += 1;
      }
      std::fprintf(stderr,
        "[mathprobe] %-6s mean_rel=%+.3e max_rel=%+.3e host_mismatch=%.4f\n",
        names[w], sum_rel / N, max_rel, n_diff / N);
    }
    std::exit(0);
  }

  // ---- XS lookup chain audit: GPU flattened vs CPU calculate_xs ----
  if (std::getenv("OPENMC_GPU_AUDIT4")) {
    GpuCeView cev;
    cev.nuclides = eng.flat.nuclides.data();
    cev.i32 = eng.flat.i32.data();
    cev.f32 = eng.flat.f32.data();
    cev.n_nuclides = (uint32_t)eng.flat.nuclides.size();
    cev.n_log_bins = (uint32_t)settings::n_log_bins;
    cev.log_spacing = (float)simulation::log_spacing;
    cev.energy_min = (float)data::energy_min[0];
    cev.energy_max = (float)data::energy_max[0];
    cev.urr_on = 0; // compare smooth XS (ptables differ by design)
    Particle p;
    p.material() = 0;
    p.sqrtkT() = (float)std::sqrt(eng.flat.nuclides[0].kT);
    const GpuMaterial& gm = eng.flat.materials[0];
    double sum_rel[4] = {}, max_rel[4] = {};
    const int NE = 200000;
    uint64_t seed = 555;
    bool saved_urr = settings::urr_ptables_on;
    settings::urr_ptables_on = false;
    for (int it = 0; it < NE; ++it) {
      double u = prn(&seed);
      double E = 1e3 * std::pow(1e4, u); // 1 keV .. 10 MeV
      p.E() = E;
      p.invalidate_neutron_xs();
      p.material_last() = C_NONE;
      p.sqrtkT_last() = -1;
      model::materials[0]->calculate_xs(p);
      double c[4] = {p.macro_xs().total, p.macro_xs().absorption,
        p.macro_xs().fission, p.macro_xs().nu_fission};
      // gpu side
      float Ef = (float)E;
      int32_t i_log = (int32_t)(logf(Ef / cev.energy_min) / cev.log_spacing);
      if (i_log < 0)
        i_log = 0;
      if (i_log >= (int32_t)cev.n_log_bins)
        i_log = (int32_t)cev.n_log_bins - 1;
      uint64_t dummy_seeds[4] = {1, 2, 3, 4};
      double g[4] = {};
      for (uint32_t i = 0; i < gm.n_nuclides; ++i) {
        int32_t in = eng.flat.i32[gm.nuclide_off + i];
        GpuSabView no_sab {};
        GpuMicroXS mi = gpu_ce_micro_xs(cev, eng.flat.nuclides[in], Ef,
          i_log, -1, 0.0f, no_sab, dummy_seeds);
        float dens = eng.flat.f32[gm.density_off + i];
        g[0] += dens * mi.total;
        g[1] += dens * mi.absorption;
        g[2] += dens * mi.fission;
        g[3] += dens * mi.nu_fission;
      }
      for (int q = 0; q < 4; ++q) {
        if (c[q] > 1e-10) {
          double rel = (g[q] - c[q]) / c[q];
          sum_rel[q] += rel;
          if (std::abs(rel) > std::abs(max_rel[q]))
            max_rel[q] = rel;
        }
      }
    }
    settings::urr_ptables_on = saved_urr;
    const char* names[4] = {"total", "absorption", "fission", "nu_fission"};
    for (int q = 0; q < 4; ++q)
      std::fprintf(stderr, "[audit4] %-10s mean_rel=%+.3e max_rel=%+.3e\n",
        names[q], sum_rel[q] / NE, max_rel[q]);
    std::exit(0);
  }

  // ---- paired-sample dump for one distribution (debug) ----
  if (std::getenv("OPENMC_GPU_AUDIT3")) {
    GpuCeView cev;
    cev.nuclides = eng.flat.nuclides.data();
    cev.i32 = eng.flat.i32.data();
    cev.f32 = eng.flat.f32.data();
    cev.n_nuclides = (uint32_t)eng.flat.nuclides.size();
    // U235 assumed at index 0 in the Godiva model; MT52 is inelastic j=1
    const auto& nuc = *data::nuclides[0];
    GpuNuclide gn = eng.flat.nuclides[0];
    for (uint32_t j = 0; j < gn.n_inelastic; ++j) {
      int32_t rx_off = eng.flat.i32[gn.inelastic_off + j];
      const int32_t* rxh = eng.flat.i32.data() + rx_off;
      if (rxh[0] != 52)
        continue;
      const Reaction& rx = *nuc.reactions_[nuc.index_inelastic_scatter_[j]];
      auto* un = dynamic_cast<UncorrelatedAngleEnergy*>(
        rx.products_[0].distribution_[0].get());
      double Ed = 4.0e6;
      {
        const int32_t* dh = eng.flat.i32.data() + rxh[GPU_RX_DIST];
        std::fprintf(stderr, "[audit3] dist type=%d n=%d\n", dh[0], dh[1]);
      }
      // dump the flattened table nearest 4 MeV
      {
        const int32_t* ah =
          eng.flat.i32.data() + eng.flat.i32[rxh[GPU_RX_DIST] + 1];
        int n_e = ah[0];
        const float* eg = eng.flat.f32.data() + ah[1];
        int i = 0;
        while (i + 1 < n_e && eg[i + 1] <= Ed)
          ++i;
        for (int tt = i; tt <= i + 1 && tt < n_e; ++tt) {
          const int32_t* tb = eng.flat.i32.data() + ah[2 + tt];
          int n_mu = tb[0];
          const float* mu = eng.flat.f32.data() + tb[2];
          std::fprintf(stderr, "[audit3] table i=%d/%d n_mu=%d interp=%d\n", tt,
            n_e, n_mu, tb[1]);
          for (int q = 0; q < n_mu && q < 12; ++q)
            std::fprintf(stderr, "[audit3]   mu=%+.6f p=%.6f c=%.6f\n", mu[q],
              mu[n_mu + q], mu[2 * n_mu + q]);
        }
      }
      uint64_t s1 = 42, s2 = 42;
      for (int it = 0; it < 16; ++it) {
        double eo_c, mu_c;
        rx.products_[0].sample(Ed, eo_c, mu_c, &s1);
        GpuSampleEA r = gpu_sample_dist(cev, rxh[GPU_RX_DIST], (float)Ed, &s2);
        std::fprintf(stderr,
          "[audit3] cpu(E'=%.6g mu=%+.6f)  gpu(E'=%.6g mu=%+.6f)%s\n", eo_c,
          mu_c, r.E_out, r.mu,
          (std::abs(mu_c - r.mu) > 1e-3 ||
            std::abs(eo_c - r.E_out) > 1e-3 * eo_c)
            ? "   <-- DIVERGES"
            : "");
      }
      std::exit(0);
    }
    std::exit(0);
  }

  // ---- A/B audit of secondary distributions vs the CPU samplers ----
  if (std::getenv("OPENMC_GPU_AUDIT2")) {
    GpuCeView cev;
    cev.nuclides = eng.flat.nuclides.data();
    cev.i32 = eng.flat.i32.data();
    cev.f32 = eng.flat.f32.data();
    cev.n_nuclides = (uint32_t)eng.flat.nuclides.size();
    const int NS = 500000;
    double Es[6] = {5.0e5, 1.5e6, 4.0e6, 6.5e6, 1.0e7, 1.4e7};
    for (uint32_t in = 0; in < cev.n_nuclides; ++in) {
      const auto& nuc = *data::nuclides[in];
      GpuNuclide gn = eng.flat.nuclides[in];
      // inelastic reactions
      for (uint32_t j = 0; j < gn.n_inelastic; ++j) {
        int32_t rx_off = eng.flat.i32[gn.inelastic_off + j];
        const int32_t* rxh = eng.flat.i32.data() + rx_off;
        int mt = rxh[0];
        const Reaction& rx = *nuc.reactions_[nuc.index_inelastic_scatter_[j]];
        for (double Ed : Es) {
          if (Ed < nuc.reactions_[nuc.index_inelastic_scatter_[j]]
                       ->xs_[0]
                       .threshold *
                     0)
            continue;
          double c_sum_e = 0, c_sum_mu = 0;
          uint64_t seed = 7777 + in * 131 + j;
          for (int it = 0; it < NS; ++it) {
            double eo, mu;
            rx.products_[0].sample(Ed, eo, mu, &seed);
            c_sum_e += eo;
            c_sum_mu += mu;
          }
          double g_sum_e = 0, g_sum_mu = 0;
          uint64_t gseed = 7777 + in * 131 + j;
          for (int it = 0; it < NS; ++it) {
            GpuSampleEA r =
              gpu_sample_dist(cev, rxh[GPU_RX_DIST], (float)Ed, &gseed);
            g_sum_e += r.E_out;
            g_sum_mu += r.mu;
          }
          double ce_ = c_sum_e / NS, ge = g_sum_e / NS;
          if (ce_ > 0 && (std::abs(ge - ce_) / ce_ > 3e-3 ||
                           std::abs(g_sum_mu - c_sum_mu) / NS > 3e-3)) {
            std::fprintf(stderr,
              "[audit2] nuc=%u MT=%d E=%.3g  CPU<E'>=%.5g GPU<E'>=%.5g  "
              "CPU<mu>=%.4f GPU<mu>=%.4f\n",
              in, mt, Ed, ce_, ge, c_sum_mu / NS, g_sum_mu / NS);
          }
        }
      }
      // fission prompt spectrum
      if (gn.fissionable) {
        int32_t rec = eng.flat.i32[gn.fis_rx_off];
        int32_t dist = eng.flat.i32[rec + 2 + 1];
        for (double Ed : Es) {
          double c_sum_e = 0;
          uint64_t seed = 991 + in;
          for (int it = 0; it < NS; ++it) {
            double eo, mu;
            nuc.fission_rx_[0]->products_[0].sample(Ed, eo, mu, &seed);
            c_sum_e += eo;
          }
          double g_sum_e = 0;
          uint64_t gseed = 991 + in;
          for (int it = 0; it < NS; ++it) {
            GpuSampleEA r = gpu_sample_dist(cev, dist, (float)Ed, &gseed);
            g_sum_e += r.E_out;
          }
          double ce_ = c_sum_e / NS, ge = g_sum_e / NS;
          std::fprintf(stderr,
            "[audit2] nuc=%u FISSION E=%.3g CPU<E'>=%.6g GPU<E'>=%.6g "
            "rel=%+.5f\n",
            in, Ed, ce_, ge, (ge - ce_) / ce_);
        }
      }
    }
    std::exit(0);
  }

  // ---- optional single-collision microphysics audit ----
  if (const char* aud = std::getenv("OPENMC_GPU_AUDIT")) {
    GpuCeView cev;
    cev.nuclides = eng.flat.nuclides.data();
    cev.i32 = eng.flat.i32.data();
    cev.f32 = eng.flat.f32.data();
    cev.n_nuclides = (uint32_t)eng.flat.nuclides.size();
    for (uint32_t in = 0; in < cev.n_nuclides; ++in) {
      GpuNuclide nuc = eng.flat.nuclides[in];
      const auto& cnuc = *data::nuclides[in];
      auto* un = dynamic_cast<UncorrelatedAngleEnergy*>(
        cnuc.reactions_[0]->products_[0].distribution_[0].get());
      for (int ie = 0; ie < 28; ++ie) {
        double Ed = 1.0e4 * std::pow(1.5e7 / 1.0e4, ie / 27.0);
        float E = (float)Ed;
        uint64_t seed = 0xabcdef12345ull + in;
        double sum_mu = 0.0;
        const int NS = 400000;
        for (int it = 0; it < NS; ++it) {
          float mu = (nuc.elastic_angle >= 0)
                       ? gpu_sample_angle_dist(cev, nuc.elastic_angle, E, &seed)
                       : 2.0f * gpu_prn(&seed) - 1.0f;
          sum_mu += mu;
        }
        // CPU sampler reference (paired seed: decisions align, so any
        // residual difference is fp32-vs-fp64, not noise)
        uint64_t cseed = 0xabcdef12345ull + in;
        double c_sum = 0.0;
        for (int it = 0; it < NS; ++it)
          c_sum += un->angle().sample(Ed, &cseed);
        double gm = sum_mu / NS, cm = c_sum / NS;
        if (std::abs(gm - cm) > 2.5e-3) {
          std::fprintf(stderr,
            "[audit] nuc=%u E=%.4g GPU<mu>=%.5f CPU<mu>=%.5f diff=%+.5f\n", in,
            Ed, gm, cm, gm - cm);
        }
        // full elastic event moments: fp32 gpu path vs fp64 replica of the
        // CPU algorithm (at-rest branch), paired seeds
        {
          const double A = cnuc.awr_;
          uint64_t s32 = 0x5152535455ull + in;
          uint64_t s64 = 0x5152535455ull + in;
          double g_mu_lab = 0, g_de = 0, c_mu_lab = 0, c_de = 0;
          const int NF = 300000;
          for (int it = 0; it < NF; ++it) {
            // fp32 (device code, host-compiled)
            {
              GpuVec3 u0 = gpu_v3(0.26726124f, 0.53452248f, 0.80178373f);
              float vel = sqrtf(E);
              GpuVec3 v_n = gpu_scale(u0, vel);
              GpuVec3 v_cm = gpu_scale(v_n, 1.0f / ((float)A + 1.0f));
              v_n = gpu_sub(v_n, v_cm);
              vel = gpu_norm(v_n);
              float mu_cm =
                (nuc.elastic_angle >= 0)
                  ? gpu_sample_angle_dist(cev, nuc.elastic_angle, E, &s32)
                  : 2.0f * gpu_prn(&s32) - 1.0f;
              GpuVec3 u_cm = gpu_scale(v_n, 1.0f / vel);
              v_n = gpu_scale(gpu_rotate_angle(u_cm, mu_cm, &s32), vel);
              v_n = gpu_add(v_n, v_cm);
              float E2 = gpu_dot(v_n, v_n);
              float v2 = sqrtf(E2);
              g_mu_lab += gpu_dot(u0, v_n) / v2;
              g_de += E2 / E;
            }
            // fp64 replica of physics.cpp elastic_scatter (at rest)
            {
              double u0[3] = {0.26726124, 0.53452248, 0.80178373};
              double vel = std::sqrt(Ed);
              double vn[3] = {u0[0] * vel, u0[1] * vel, u0[2] * vel};
              double vcm[3] = {
                vn[0] / (A + 1), vn[1] / (A + 1), vn[2] / (A + 1)};
              for (int q = 0; q < 3; ++q)
                vn[q] -= vcm[q];
              vel = std::sqrt(vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]);
              double mu_cm = un->angle().empty() ? 2 * prn(&s64) - 1
                                                 : un->angle().sample(Ed, &s64);
              double ucm[3] = {vn[0] / vel, vn[1] / vel, vn[2] / vel};
              Direction dd =
                rotate_angle({ucm[0], ucm[1], ucm[2]}, mu_cm, nullptr, &s64);
              vn[0] = dd.x * vel;
              vn[1] = dd.y * vel;
              vn[2] = dd.z * vel;
              for (int q = 0; q < 3; ++q)
                vn[q] += vcm[q];
              double E2 = vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2];
              double v2 = std::sqrt(E2);
              c_mu_lab += (u0[0] * vn[0] + u0[1] * vn[1] + u0[2] * vn[2]) / v2;
              c_de += E2 / Ed;
            }
          }
          if (std::abs(g_mu_lab - c_mu_lab) / NF > 1.5e-3 ||
              std::abs(g_de - c_de) / NF > 1e-3) {
            std::fprintf(stderr,
              "[audit-el] nuc=%u E=%.4g mu_lab %.5f/%.5f  E'/E "
              "%.5f/%.5f\n",
              in, Ed, g_mu_lab / NF, c_mu_lab / NF, g_de / NF, c_de / NF);
          }
        }
      }
    }
    (void)aud;
    std::exit(0);
  }

  // ---- optional host replay for lost-particle diagnosis ----
  if (std::getenv("OPENMC_GPU_REPLAY")) {
    if (std::getenv("OPENMC_LEAK_MAP") && gpu_leak_map.empty())
      gpu_leak_map.assign((size_t)settings::n_particles, 0);
    GpuGeomData geom;
    geom.surfaces = eng.flat.surfaces.data();
    geom.cells = eng.flat.cells.data();
    geom.universes = eng.flat.universes.data();
    geom.lattices = eng.flat.lattices.data();
    geom.i32 = eng.flat.i32.data();
    geom.f32 = eng.flat.f32.data();
    geom.surf_adj_off = eng.flat.surf_adj_off;
    GpuMgView mgv;
    mgv.mats = eng.flat.mgmats.data();
    mgv.i32 = eng.flat.i32.data();
    mgv.f32 = eng.flat.f32.data();
    mgv.n_groups = ctl->n_groups;
    geom.materials = eng.flat.materials.data();
    GpuCeView cev;
    cev.nuclides = eng.flat.nuclides.data();
    cev.i32 = eng.flat.i32.data();
    cev.f32 = eng.flat.f32.data();
    cev.n_nuclides = ctl->n_nuclides;
    cev.n_log_bins = ctl->ce_n_log_bins;
    cev.log_spacing = ctl->ce_log_spacing;
    cev.energy_min = ctl->energy_min;
    cev.energy_max = ctl->energy_max;
    cev.urr_on = ctl->urr_on;
    GpuSabView sabv;
    sabv.tables = eng.flat.sab_tables.data();
    sabv.i32 = eng.flat.i32.data();
    sabv.f32 = eng.flat.f32.data();
    GpuTallyView tvv {};
    tvv.n_tallies = 0;
    tvv.i32 = eng.flat.i32.data();
    tvv.f32 = eng.flat.f32.data();
    std::vector<GpuSourceSite> host_fission(3 * (size_t)n);
    std::vector<uint32_t> host_prog(n);
    std::vector<float> host_red(eng.n_red_slots * GPU_RED_WIDTH, 0.f);
    std::vector<GpuTraceRec> host_trace(GPU_TRACE_MAX);
    uint32_t host_ctr[GPU_CTR_COUNT] = {};
    GpuBanks hb;
    hb.source = src;
    hb.fission = host_fission.data();
    hb.counters = host_ctr;
    hb.progeny = host_prog.data();
    hb.red_slots = host_red.data();
    hb.trace = host_trace.data();
    for (int64_t i = 0; i < n; ++i)
      gpu_run_particle((uint32_gpu)i, *ctl, geom, mgv, cev, sabv, tvv, hb);
    std::fprintf(stderr,
      "[gpu-debug] host replay: lost=%u (init %u advance %u lattice %u "
      "reflect %u)\n",
      host_ctr[GPU_CTR_LOST], host_ctr[GPU_CTR_LOST_INIT],
      host_ctr[GPU_CTR_LOST_ADVANCE], host_ctr[GPU_CTR_LOST_LATTICE],
      host_ctr[GPU_CTR_LOST_REFLECT]);
    if (rstats.n_coll > 0) {
      std::fprintf(stderr,
        "[stats-replay] coll=%.0f elastic=%.0f level=%.0f cont=%.0f "
        "xn=%.0f absorb=%.0f fsites=%.0f <E_coll>=%.1f <mu_el>=%.6f "
        "<E'/E_el>=%.6f\n",
        rstats.n_coll, rstats.n_el, rstats.n_lvl, rstats.n_cont, rstats.n_xn,
        rstats.n_abs, rstats.n_fs, rstats.sE / rstats.n_coll,
        rstats.smu / rstats.n_el, rstats.ser / rstats.n_el);
      std::fprintf(stderr,
        "[stats-replay] <r_coll>=%.6f <r_fsite>=%.6f flights=%.0f "
        "<d>=%.6f\n",
        rstats.sr / rstats.n_coll, rstats.srf / rstats.n_fs, rstats.n_fl,
        rstats.sd / rstats.n_fl);
      // cumulative across generations: no reset
    }
  }

  char err[1024];
  // Deterministic fp64 reduction of the per-particle keff slots: fixed chunk
  // partition, each chunk summed in index order, chunks combined in index
  // order — identical for any thread count and run to run. Runs once per
  // dispatch because the slots are indexed by thread id.
  auto reduce_keff = [&](int64_t n_slots) {
    auto* red =
      static_cast<float*>(omg_metal_contents(eng.ctx, OMG_SLOT_REDSLOTS));
    double sums[GPU_RED_WIDTH] = {};
    const int64_t chunk = 8192;
    const int64_t n_chunks = (n_slots + chunk - 1) / chunk;
    std::vector<double> part((size_t)n_chunks * GPU_RED_WIDTH, 0.0);
#pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < n_chunks; ++c) {
      double acc[GPU_RED_WIDTH] = {};
      const int64_t lo = c * chunk;
      const int64_t hi = std::min(lo + chunk, n_slots);
      for (int64_t i = lo; i < hi; ++i)
        for (int k = 0; k < GPU_RED_WIDTH; ++k)
          acc[k] += red[i * GPU_RED_WIDTH + k];
      for (int k = 0; k < GPU_RED_WIDTH; ++k)
        part[(size_t)c * GPU_RED_WIDTH + k] = acc[k];
    }
    for (int64_t c = 0; c < n_chunks; ++c)
      for (int k = 0; k < GPU_RED_WIDTH; ++k)
        sums[k] += part[(size_t)c * GPU_RED_WIDTH + k];
    global_tally_tracklength += sums[GPU_RED_K_TRACKLENGTH];
    global_tally_collision += sums[GPU_RED_K_COLLISION];
    global_tally_absorption += sums[GPU_RED_K_ABSORPTION];
    global_tally_leakage += sums[GPU_RED_LEAKAGE];
    eng.total_events += sums[GPU_RED_EVENTS];
    eng.total_xseval += sums[GPU_RED_XSEVAL];
    eng.simdeff_sum += sums[GPU_RED_SIMDEFF];
    eng.simdeff_n += (double)n_slots;
  };

  // ---- dispatch in chunks, with interactivity-watchdog recovery ----
  // macOS aborts a compute command buffer that runs long enough to hurt
  // desktop responsiveness (kIOGPUCommandBufferCallbackErrorImpacting-
  // Interactivity). A whole batch in one buffer is seconds of GPU work, so
  // anything else the user does could kill a run that is hours old. Submit
  // the batch in chunks short enough to stay under the watchdog, and if one
  // is killed anyway, rewind and replay it smaller.
  static uint32_t s_chunk = 0;
  if (s_chunk == 0) {
    s_chunk = 1u << 20;
    if (const char* e = std::getenv("OPENMC_GPU_CHUNK"))
      s_chunk = (uint32_t)std::max(4096, atoi(e));
  }
  const bool last_generation =
    simulation::current_batch >= settings::n_batches &&
    simulation::current_gen >= settings::gen_per_batch;
  auto do_prefetch = [&]() {
    if (!prefetch_ok || last_generation)
      return;
    prefetched.resize((size_t)n);
    double w = 0.0;
#pragma omp parallel for reduction(+ : w)
    for (int64_t i = 0; i < n; ++i) {
      int64_t id = compute_transport_seed(compute_particle_id(i + 1)) +
                   settings::n_particles;
      uint64_t seed = init_seed(id, STREAM_SOURCE);
      SourceSite s = sample_external_source(&seed);
      fill_site(prefetched[i], s);
      w += s.wgt;
    }
    prefetched_weight = w;
    have_prefetch = true;
  };

  // State that accumulates across chunks. A killed command buffer may still
  // have let some threads finish, so their tally scores and bank appends are
  // already in memory; replaying without rewinding would double-count them.
  // The fission and spill banks are addressed by their counters, so rewinding
  // the counters logically rewinds the banks too.
  const size_t taccum_bytes =
    (size_t)eng.flat.tally_accum_size * eng.tally_replicas * 4;
  std::vector<uint8_t> taccum_save(taccum_bytes);
  uint32_t ctr_save[GPU_CTR_COUNT];
  auto snapshot_accum = [&]() {
    if (taccum_bytes)
      std::memcpy(taccum_save.data(),
        omg_metal_contents(eng.ctx, OMG_SLOT_TACCUM), taccum_bytes);
    std::memcpy(ctr_save, omg_metal_contents(eng.ctx, OMG_SLOT_COUNTERS),
      sizeof(ctr_save));
  };
  auto rewind_accum = [&]() {
    if (taccum_bytes)
      std::memcpy(omg_metal_contents(eng.ctx, OMG_SLOT_TACCUM),
        taccum_save.data(), taccum_bytes);
    std::memcpy(omg_metal_contents(eng.ctx, OMG_SLOT_COUNTERS), ctr_save,
      sizeof(ctr_save));
  };

  // Runs one chunk; returns how many particles it actually consumed, which
  // is smaller than requested if the watchdog forced a reduction.
  auto run_chunk = [&](uint32_t offset, uint32_t count, bool prefetch) {
    for (;;) {
      ctl->n_particles = count;
      ctl->source_offset = offset;
      std::memset(omg_metal_contents(eng.ctx, OMG_SLOT_REDSLOTS), 0,
        (size_t)count * GPU_RED_WIDTH * 4);
      snapshot_accum();
      bool bad = omg_metal_dispatch_async(
        eng.ctx, "openmc_transport", count, err, sizeof(err));
      if (!bad) {
        if (prefetch)
          do_prefetch();
        bad = omg_metal_wait(eng.ctx, err, sizeof(err));
      }
      if (!bad) {
        eng.gpu_seconds += omg_metal_last_time(eng.ctx);
        reduce_keff((int64_t)count);
        return count;
      }
      if (std::strstr(err, "Interactivity") == nullptr || count <= 4096)
        fatal_error(fmt::format("GPU transport dispatch failed: {}", err));
      rewind_accum();
      count /= 2;
      s_chunk = count;
      warning(fmt::format("GPU dispatch interrupted by the system for "
                          "impacting interactivity; retrying with {} "
                          "particles per dispatch",
        count));
    }
  };

  for (uint32_t off = 0, first = 1; off < (uint32_t)n; first = 0)
    off += run_chunk(off, std::min<uint32_t>(s_chunk, (uint32_t)n - off),
      first != 0);
  ctl->n_particles = (uint32_t)n;
  ctl->source_offset = 0;


  // ---- spill drain: weight-window splits and (n,xn) clones that did not
  // fit in a thread's local stack were banked globally; re-dispatch them
  // until the bank drains. Tally accumulators and keff totals accumulate
  // across passes, so the result is the same as if every secondary had been
  // transported in the first pass.
  {
    auto* ctr0 =
      static_cast<uint32_t*>(omg_metal_contents(eng.ctx, OMG_SLOT_COUNTERS));
    const int max_passes = 256;
    for (int pass = 0; pass < max_passes; ++pass) {
      uint32_t spill = std::min(ctr0[GPU_CTR_SPILL], ctl->spill_cap);
      if (spill == 0)
        break;
      // The source buffer holds n sites while the spill bank holds up to
      // spill_cap (> n), so drain in source-sized chunks and keep the
      // remainder at the front of the bank for the next pass. The device
      // appends new spills from the retained count upward, so nothing is
      // overwritten.
      const uint32_t src_cap = (uint32_t)n;
      uint32_t take = std::min(std::min(spill, src_cap), s_chunk);
      auto* spill_bank = static_cast<GpuSourceSite*>(
        omg_metal_contents(eng.ctx, OMG_SLOT_FISSION));
      auto* src_buf = static_cast<GpuSourceSite*>(
        omg_metal_contents(eng.ctx, OMG_SLOT_SOURCE));
      std::memcpy(src_buf, spill_bank, (size_t)take * sizeof(GpuSourceSite));
      uint32_t remain = spill - take;
      if (remain > 0)
        std::memmove(spill_bank, spill_bank + take,
          (size_t)remain * sizeof(GpuSourceSite));
      ctr0[GPU_CTR_SPILL] = remain;
      ctl->n_particles = take;
      ctl->source_offset = 0;
      ctl->source_is_spill = 1u;
      std::memset(omg_metal_contents(eng.ctx, OMG_SLOT_REDSLOTS), 0,
        (size_t)take * GPU_RED_WIDTH * 4);
      if (omg_metal_dispatch_async(
            eng.ctx, "openmc_transport", take, err, sizeof(err)) ||
          omg_metal_wait(eng.ctx, err, sizeof(err))) {
        fatal_error(fmt::format("GPU spill dispatch failed: {}", err));
      }
      eng.gpu_seconds += omg_metal_last_time(eng.ctx);
      reduce_keff((int64_t)take);
      if (pass == max_passes - 1 && ctr0[GPU_CTR_SPILL] > 0) {
        // those sites carry real weight; abandoning them would bias the
        // tally low, so stop rather than report a wrong answer
        fatal_error(fmt::format(
          "GPU variance reduction did not converge: {} secondaries still "
          "banked after {} spill passes. Soften the weight windows or lower "
          "max_split.",
          ctr0[GPU_CTR_SPILL], max_passes));
      }
    }
    eng.spill_uid_cursor += ctr0[GPU_CTR_SPILL_SERIAL];
  }
  auto t_gen2 = std::chrono::steady_clock::now();

  // ---- counters ----
  auto* ctr =
    static_cast<uint32_t*>(omg_metal_contents(eng.ctx, OMG_SLOT_COUNTERS));
  if (ctl->trace_id >= 0) {
    auto* tr =
      static_cast<GpuTraceRec*>(omg_metal_contents(eng.ctx, OMG_SLOT_TRACE));
    uint32_t nrec = std::min(ctr[GPU_CTR_TRACE], (uint32_t)GPU_TRACE_MAX);
    for (uint32_t i = 0; i < nrec; ++i) {
      const char* names[5] = {
        "fly", "collide", "elastic", "inelastic", "postfis"};
      int code = (int)tr[i].code;
      std::fprintf(stderr, "[T-dev] %s a=%a b=%a c=%a\n",
        names[code < 5 ? code : 0], tr[i].a, tr[i].b, tr[i].c);
    }
  }
  uint32_t n_sites = std::min(ctr[GPU_CTR_FISSION_BANK], ctl->fission_bank_cap);
  if (ctr[GPU_CTR_FISSION_BANK] > ctl->fission_bank_cap) {
    warning(fmt::format(
      "The GPU fission bank is full ({} sites sampled, capacity {}). "
      "Additional fission sites were not banked; results may be "
      "non-deterministic.",
      ctr[GPU_CTR_FISSION_BANK], ctl->fission_bank_cap));
  }
  uint32_t n_lost = ctr[GPU_CTR_LOST];
  if (n_lost > 0) {
    eng.lost_total += n_lost;
    warning(fmt::format(
      "GPU transport lost {} particles this generation "
      "({} total; init {} advance {} lattice {} reflect {} reconcile {})",
      n_lost, eng.lost_total, ctr[GPU_CTR_LOST_INIT], ctr[GPU_CTR_LOST_ADVANCE],
      ctr[GPU_CTR_LOST_LATTICE], ctr[GPU_CTR_LOST_REFLECT],
      ctr[GPU_CTR_LOST_RECONCILE]));
    // Feed the upstream lost-particle accounting and apply the same abort
    // policy as Particle::mark_as_lost (checked per generation here). The
    // thresholds are the standard settings (max_lost_particles /
    // rel_max_lost_particles), so users accepting fp32 residual losses can
    // raise them exactly as they would on the CPU.
    simulation::n_lost_particles += (int)n_lost;
    auto n_sim = simulation::current_batch * settings::gen_per_batch *
                 simulation::work_per_rank;
    if (simulation::n_lost_particles >= settings::max_lost_particles &&
        simulation::n_lost_particles >=
          settings::rel_max_lost_particles * n_sim) {
      fatal_error("Maximum number of lost particles has been reached.");
    }
  }
  if (ctr[GPU_CTR_SPILL_DROP] > 0) {
    // Not a correctness problem: when the bank is full the device makes
    // fewer split copies and the parent keeps the remaining weight, so
    // totals are conserved. It does mean the weight windows are asking for
    // more splitting than the bank can hold, i.e. weaker variance
    // reduction than requested.
    warning(fmt::format(
      "GPU weight windows requested {} more split copies than the spill bank "
      "({} sites) could hold; those splits were declined (weight conserved, "
      "results unbiased, variance reduction weaker than requested). Soften "
      "the windows or lower max_split.",
      ctr[GPU_CTR_SPILL_DROP], ctl->spill_cap));
  }
  if (ctr[GPU_CTR_SECONDARY_BANK] > 0) {
    // The CPU never drops (n,xn) clones (dynamically sized secondary
    // bank); silently losing them would lose physics, so this is fatal.
    fatal_error(fmt::format(
      "GPU dropped {} (n,xn) clones (per-thread secondary stack full). "
      "Deepen GPU_MAX_SECONDARY_STACK for this model.",
      ctr[GPU_CTR_SECONDARY_BANK]));
  }
  if (ctr[GPU_CTR_MAX_EVENT_HIT] > 0) {
    warning(fmt::format("{} particles hit the max-event cap on the GPU",
      ctr[GPU_CTR_MAX_EVENT_HIT]));
  }

  // ---- fission bank + progeny bookkeeping (feeds upstream sort/sync;
  // eigenvalue only — fixed-source models are non-multiplying here) ----
  if (settings::run_mode == RunMode::EIGENVALUE) {
    auto* fb = static_cast<GpuSourceSite*>(
      omg_metal_contents(eng.ctx, OMG_SLOT_FISSION));
    simulation::fission_bank.resize(n_sites);
    for (uint32_t i = 0; i < n_sites; ++i) {
      const GpuSourceSite& g = fb[i];
      SourceSite s;
      s.r = {g.r[0], g.r[1], g.r[2]};
      s.u = {g.u[0], g.u[1], g.u[2]};
      s.E = g.E;
      s.time = g.time;
      s.wgt = g.wgt;
      s.delayed_group = g.delayed_group;
      s.surf_id = 0;
      s.particle = ParticleType::neutron();
      s.parent_id = g.parent_id;
      s.progeny_id = g.progeny_id;
      simulation::fission_bank[i] = s;
    }
    auto* prog =
      static_cast<uint32_t*>(omg_metal_contents(eng.ctx, OMG_SLOT_PROGENY));
    for (int64_t i = 0; i < n; ++i)
      simulation::progeny_per_particle[i] = prog[i] & 0x7fffffffu;
    if (std::getenv("OPENMC_GPU_BANKCHECK")) {
      // sort_bank permutes by progeny_per_particle[parent]+progeny_id; that
      // is only a bijection if the counts match the banked sites exactly
      uint64_t sum = 0;
      for (int64_t i = 0; i < n; ++i)
        sum += simulation::progeny_per_particle[i];
      std::vector<uint8_t> hit(n_sites, 0);
      std::vector<uint64_t> scan(n, 0);
      uint64_t acc = 0;
      for (int64_t i = 0; i < n; ++i) {
        scan[i] = acc;
        acc += simulation::progeny_per_particle[i];
      }
      uint32_t dup = 0, oob = 0;
      for (uint32_t i = 0; i < n_sites; ++i) {
        const GpuSourceSite& g = fb[i];
        uint64_t idx = scan[g.parent_id] + g.progeny_id;
        if (idx >= n_sites) {
          ++oob;
          continue;
        }
        if (hit[idx]++)
          ++dup;
      }
      uint32_t unwritten = 0;
      for (uint32_t i = 0; i < n_sites; ++i)
        if (!hit[i])
          ++unwritten;
      if (sum != n_sites || dup || oob || unwritten)
        std::fprintf(stderr,
          "[bankcheck] MISMATCH sum(progeny)=%llu n_sites=%u dup=%u oob=%u "
          "unwritten=%u\n",
          (unsigned long long)sum, n_sites, dup, oob, unwritten);
    }
  }
  auto* prog =
    static_cast<uint32_t*>(omg_metal_contents(eng.ctx, OMG_SLOT_PROGENY));
  if (const char* lm = std::getenv("OPENMC_LEAK_MAP")) {
    std::string path = std::string(lm) + ".dev";
    FILE* f = std::fopen(path.c_str(), "wb");
    for (int64_t i = 0; i < n; ++i) {
      uint8_t v = (prog[i] >> 31) & 1u;
      std::fwrite(&v, 1, 1, f);
    }
    std::fclose(f);
  }


  // ---- tally results (fp32 batch bins -> fp64 running VALUE) ----
  if (ctl->n_tallies > 0) {
    auto* acc =
      static_cast<float*>(omg_metal_contents(eng.ctx, OMG_SLOT_TACCUM));
    for (size_t ti = 0; ti < eng.flat.tallies.size(); ++ti) {
      const GpuTallyDesc& td = eng.flat.tallies[ti];
      int32_t host_idx = eng.flat.tally_host_index[ti];
      if (!model::tallies[host_idx]->active_)
        continue;
      double* results;
      size_t shape[3];
      if (openmc_tally_results(host_idx, &results, shape))
        continue;
      for (uint32_t b = 0; b < td.n_filter_bins; ++b) {
        for (uint32_t s = 0; s < td.n_scores; ++s) {
          double v = 0.0;
          for (uint32_t k = 0; k < eng.tally_replicas; ++k)
            v += acc[(size_t)k * eng.flat.tally_accum_size + td.accum_off +
                     b * td.n_scores + s];
          results[(b * td.n_scores + s) * 3 + 0] += v;
        }
      }
    }
  }
  auto t_gen3 = std::chrono::steady_clock::now();
  eng.t_source += std::chrono::duration<double>(t_gen1 - t_gen0).count();
  eng.t_dispatch += std::chrono::duration<double>(t_gen2 - t_gen1).count();
  eng.t_post += std::chrono::duration<double>(t_gen3 - t_gen2).count();
}

void finalize()
{
  gpu_prefetch_reset = true;
  if (std::getenv("OPENMC_GPU_TIMING")) {
    std::fprintf(stderr,
      "[gpu-timing] host source/upload %.3f s, dispatch(wait) %.3f s "
      "(device %.3f s), copy-back %.3f s\n",
      eng.t_source, eng.t_dispatch, eng.gpu_seconds, eng.t_post);
  }
  if (const char* lm = std::getenv("OPENMC_LEAK_MAP")) {
    if (!gpu_leak_map.empty()) {
      std::string path = std::string(lm) + ".gpu";
      FILE* f = std::fopen(path.c_str(), "wb");
      std::fwrite(gpu_leak_map.data(), 1, gpu_leak_map.size(), f);
      std::fclose(f);
      gpu_leak_map.clear();
    }
  }
  if (eng.ctx) {
    if (eng.active) {
      write_message(
        fmt::format("GPU transport device time: {:.3f} s ({:.4g} events, "
                    "{:.2f} ns/event, {:.2f} xs evals/event, "
                    "SIMD eff {:.1f}%)",
          eng.gpu_seconds, eng.total_events,
          eng.total_events > 0 ? eng.gpu_seconds / eng.total_events * 1e9 : 0.0,
          eng.total_events > 0 ? eng.total_xseval / eng.total_events : 0.0,
          eng.simdeff_n > 0 ? 100.0 * eng.simdeff_sum / eng.simdeff_n : 0.0),
        6);
    }
    omg_metal_destroy(eng.ctx);
    eng.ctx = nullptr;
  }
  eng.active = false;
  eng.gpu_seconds = 0.0;
  eng.lost_total = 0;
}

} // namespace gpu
} // namespace openmc
