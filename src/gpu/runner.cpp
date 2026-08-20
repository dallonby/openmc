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
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/material.h"
#include "openmc/nuclide.h"
#include "openmc/reaction.h"
#include "openmc/reaction_product.h"
#include "openmc/secondary_uncorrelated.h"
#include "openmc/particle.h"
#include "openmc/particle_data.h"
#include "openmc/math_functions.h"
#include "openmc/random_lcg.h"

namespace openmc {
// file-scope globals in random_lcg.cpp with external linkage
extern int64_t master_seed;
extern uint64_t prn_stride;
}
#include "openmc/settings.h"
#include "openmc/simulation.h"
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
  double n_coll = 0, n_el = 0, n_lvl = 0, n_cont = 0, n_xn = 0,
         n_abs = 0, n_fs = 0;
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

void gpu_host_leak(int64_gpu id)
{
  if (!gpu_leak_map.empty() && (size_t)(id - 1) < gpu_leak_map.size())
    gpu_leak_map[id - 1] = 1;
}

void gpu_host_trace_fly(int64_gpu id, float d, float dc, float db, float E)
{
  if (id == gpu_trace_id())
    std::fprintf(stderr, "[T-gpu] fly d=%a seedlo=%.0f bdry=%a E=%a\n", d,
      dc, db, E);
}
void gpu_host_trace_collide(int64_gpu id, float E, GpuVec3 r, int32_gpu nuc)
{
  if (id == gpu_trace_id())
    std::fprintf(stderr, "[T-gpu] collide E=%a rx=%a nuc=%d\n", E, r.x,
      nuc);
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
    gs->boundary.surface, gs->boundary.coord_level,
    gs->boundary.lat_trans[0], gs->boundary.lat_trans[1],
    gs->boundary.lat_trans[2], gs->surface, gs->n_coord);
}

void gpu_host_debug_lost(const GpuGeomState* gs, int32_gpu tok, GpuVec3 r0,
  GpuVec3 u0, GpuVec3 u_new)
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
  void* ctx = nullptr;
  FlatModel flat;
  bool active = false;
  std::string reason;
  double gpu_seconds = 0.0;
  int64_t lost_total = 0;
  uint32_t n_red_slots = 0;
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
  if (settings::run_mode != RunMode::EIGENVALUE) {
    fail("only eigenvalue runs are in the GPU v1 envelope");
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

  // compile device source
  std::string src(reinterpret_cast<const char*>(openmc_gpu_msl),
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
      !upload(OMG_SLOT_TALLIES, eng.flat.tallies) ||
      !upload(OMG_SLOT_FILTERS, eng.flat.filters) ||
      !upload(OMG_SLOT_MESHES, eng.flat.meshes)) {
    fail("device buffer allocation failed");
    return;
  }

  // dynamic buffers
  int64_t n = simulation::work_per_rank;
  eng.n_red_slots = (uint32_t)((n + 255) / 256);
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
        std::max<size_t>(1, eng.flat.tally_accum_size) * 4) ||
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
      eng.flat.cells.size(), eng.flat.mgmats.size(),
      eng.flat.tallies.size()),
    5);
}

void transport_generation()
{
  int64_t n = simulation::work_per_rank;

  // ---- control block ----
  auto* ctl = static_cast<GpuControl*>(
    omg_metal_contents(eng.ctx, OMG_SLOT_CONTROL));
  std::memset(ctl, 0, sizeof(GpuControl));
  ctl->n_particles = (uint32_t)n;
  ctl->source_offset = 0;
  ctl->run_mode = GPU_RUN_EIGENVALUE;
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
  ctl->max_events = (uint32_t)std::min<int64_t>(
    settings::max_particle_events, 0x7fffffff);
  ctl->seed_base = (uint64_t)((simulation::total_gen +
                                (int64_t)overall_generation_() - 1) *
                              settings::n_particles);
  ctl->master_seed = (uint64_t)openmc::master_seed;
  ctl->prn_stride = openmc::prn_stride;
  ctl->fission_bank_cap = (uint32_t)(3 * n);
  ctl->root_universe = model::root_universe;
  ctl->n_coord_levels = model::n_coord_levels;
  ctl->n_cells = (uint32_t)eng.flat.cells.size();
  ctl->n_surfaces = (uint32_t)eng.flat.surfaces.size();
  ctl->mg_bin_avg_off = eng.flat.mg_bin_avg_off;

  // active tallies only (device scores every desc it is told about)
  bool tallies_active = false;
  for (const auto& t : model::tallies)
    if (t->active_) {
      tallies_active = true;
      break;
    }
  ctl->n_tallies = tallies_active ? (uint32_t)eng.flat.tallies.size() : 0;

  // ---- source upload (fp64 -> fp32) ----
  auto* src = static_cast<GpuSourceSite*>(
    omg_metal_contents(eng.ctx, OMG_SLOT_SOURCE));
  double src_weight = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const SourceSite& s = simulation::source_bank[i];
    GpuSourceSite& g = src[i];
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
    src_weight += s.wgt;
  }
  simulation::total_weight += src_weight;

  // ---- zero per-generation accumulators ----
  std::memset(omg_metal_contents(eng.ctx, OMG_SLOT_COUNTERS), 0,
    GPU_CTR_COUNT * sizeof(uint32_t));
  std::memset(omg_metal_contents(eng.ctx, OMG_SLOT_PROGENY), 0, (size_t)n * 4);
  std::memset(omg_metal_contents(eng.ctx, OMG_SLOT_REDSLOTS), 0,
    (size_t)eng.n_red_slots * GPU_RED_WIDTH * 4);
  if (eng.flat.tally_accum_size > 0)
    std::memset(omg_metal_contents(eng.ctx, OMG_SLOT_TACCUM), 0,
      (size_t)eng.flat.tally_accum_size * 4);

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
        GpuMicroXS mi = gpu_ce_micro_xs(
          cev, eng.flat.nuclides[in], Ef, i_log, dummy_seeds);
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
        const int32_t* ah = eng.flat.i32.data() +
                            eng.flat.i32[rxh[GPU_RX_DIST] + 1];
        int n_e = ah[0];
        const float* eg = eng.flat.f32.data() + ah[1];
        int i = 0;
        while (i + 1 < n_e && eg[i + 1] <= Ed)
          ++i;
        for (int tt = i; tt <= i + 1 && tt < n_e; ++tt) {
          const int32_t* tb = eng.flat.i32.data() + ah[2 + tt];
          int n_mu = tb[0];
          const float* mu = eng.flat.f32.data() + tb[2];
          std::fprintf(stderr,
            "[audit3] table i=%d/%d n_mu=%d interp=%d\n", tt, n_e, n_mu,
            tb[1]);
          for (int q = 0; q < n_mu && q < 12; ++q)
            std::fprintf(stderr, "[audit3]   mu=%+.6f p=%.6f c=%.6f\n",
              mu[q], mu[n_mu + q], mu[2 * n_mu + q]);
        }
      }
      uint64_t s1 = 42, s2 = 42;
      for (int it = 0; it < 16; ++it) {
        double eo_c, mu_c;
        rx.products_[0].sample(Ed, eo_c, mu_c, &s1);
        GpuSampleEA r =
          gpu_sample_dist(cev, rxh[GPU_RX_DIST], (float)Ed, &s2);
        std::fprintf(stderr,
          "[audit3] cpu(E'=%.6g mu=%+.6f)  gpu(E'=%.6g mu=%+.6f)%s\n",
          eo_c, mu_c, r.E_out, r.mu,
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
          if (ce_ > 0 &&
              (std::abs(ge - ce_) / ce_ > 3e-3 ||
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
                       ? gpu_sample_angle_dist(
                           cev, nuc.elastic_angle, E, &seed)
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
            "[audit] nuc=%u E=%.4g GPU<mu>=%.5f CPU<mu>=%.5f diff=%+.5f\n",
            in, Ed, gm, cm, gm - cm);
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
              GpuVec3 v_cm =
                gpu_scale(v_n, 1.0f / ((float)A + 1.0f));
              v_n = gpu_sub(v_n, v_cm);
              vel = gpu_norm(v_n);
              float mu_cm = (nuc.elastic_angle >= 0)
                              ? gpu_sample_angle_dist(
                                  cev, nuc.elastic_angle, E, &s32)
                              : 2.0f * gpu_prn(&s32) - 1.0f;
              GpuVec3 u_cm = gpu_scale(v_n, 1.0f / vel);
              v_n = gpu_scale(
                gpu_rotate_angle(u_cm, mu_cm, &s32), vel);
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
              vel = std::sqrt(
                vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2]);
              double mu_cm = un->angle().empty()
                               ? 2 * prn(&s64) - 1
                               : un->angle().sample(Ed, &s64);
              double ucm[3] = {vn[0] / vel, vn[1] / vel, vn[2] / vel};
              Direction dd =
                rotate_angle({ucm[0], ucm[1], ucm[2]}, mu_cm, nullptr, &s64);
              vn[0] = dd.x * vel;
              vn[1] = dd.y * vel;
              vn[2] = dd.z * vel;
              for (int q = 0; q < 3; ++q)
                vn[q] += vcm[q];
              double E2 =
                vn[0] * vn[0] + vn[1] * vn[1] + vn[2] * vn[2];
              double v2 = std::sqrt(E2);
              c_mu_lab +=
                (u0[0] * vn[0] + u0[1] * vn[1] + u0[2] * vn[2]) / v2;
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
      gpu_run_particle((uint32_gpu)i, *ctl, geom, mgv, cev, tvv, hb);
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
        rstats.n_coll, rstats.n_el, rstats.n_lvl, rstats.n_cont,
        rstats.n_xn, rstats.n_abs, rstats.n_fs, rstats.sE / rstats.n_coll,
        rstats.smu / rstats.n_el, rstats.ser / rstats.n_el);
      std::fprintf(stderr,
        "[stats-replay] <r_coll>=%.6f <r_fsite>=%.6f flights=%.0f "
        "<d>=%.6f\n",
        rstats.sr / rstats.n_coll, rstats.srf / rstats.n_fs, rstats.n_fl,
        rstats.sd / rstats.n_fl);
      // cumulative across generations: no reset
    }
  }

  // ---- dispatch ----
  char err[1024];
  if (omg_metal_dispatch(
        eng.ctx, "openmc_transport", (unsigned)n, err, sizeof(err))) {
    fatal_error(fmt::format("GPU transport dispatch failed: {}", err));
  }
  eng.gpu_seconds += omg_metal_last_time(eng.ctx);

  // ---- counters ----
  auto* ctr =
    static_cast<uint32_t*>(omg_metal_contents(eng.ctx, OMG_SLOT_COUNTERS));
  if (ctl->trace_id >= 0) {
    auto* tr = static_cast<GpuTraceRec*>(
      omg_metal_contents(eng.ctx, OMG_SLOT_TRACE));
    uint32_t nrec = std::min(ctr[7], (uint32_t)GPU_TRACE_MAX);
    for (uint32_t i = 0; i < nrec; ++i) {
      const char* names[5] = {"fly", "collide", "elastic", "inelastic",
        "postfis"};
      int code = (int)tr[i].code;
      std::fprintf(stderr, "[T-dev] %s a=%a b=%a c=%a\n",
        names[code < 5 ? code : 0], tr[i].a, tr[i].b, tr[i].c);
    }
  }
  uint32_t n_sites = std::min(ctr[GPU_CTR_FISSION_BANK], ctl->fission_bank_cap);
  uint32_t n_lost = ctr[GPU_CTR_LOST];
  if (n_lost > 0) {
    eng.lost_total += n_lost;
    warning(fmt::format("GPU transport lost {} particles this generation "
                        "({} total; init {} advance {} lattice {} reflect {})",
      n_lost, eng.lost_total, ctr[GPU_CTR_LOST_INIT],
      ctr[GPU_CTR_LOST_ADVANCE], ctr[GPU_CTR_LOST_LATTICE],
      ctr[GPU_CTR_LOST_REFLECT]));
  }
  if (ctr[GPU_CTR_SECONDARY_BANK] > 0) {
    warning(fmt::format(
      "GPU dropped {} (n,xn) clones (secondary stack full)",
      ctr[GPU_CTR_SECONDARY_BANK]));
  }
  if (ctr[GPU_CTR_MAX_EVENT_HIT] > 0) {
    warning(fmt::format("{} particles hit the max-event cap on the GPU",
      ctr[GPU_CTR_MAX_EVENT_HIT]));
  }

  // ---- fission bank + progeny bookkeeping (feeds upstream sort/sync) ----
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
  if (const char* lm = std::getenv("OPENMC_LEAK_MAP")) {
    std::string path = std::string(lm) + ".dev";
    FILE* f = std::fopen(path.c_str(), "wb");
    for (int64_t i = 0; i < n; ++i) {
      uint8_t v = (prog[i] >> 31) & 1u;
      std::fwrite(&v, 1, 1, f);
    }
    std::fclose(f);
  }

  // ---- keff estimator reduction (fp32 slots -> fp64 globals) ----
  auto* red =
    static_cast<float*>(omg_metal_contents(eng.ctx, OMG_SLOT_REDSLOTS));
  double sums[GPU_RED_WIDTH] = {};
  for (uint32_t s = 0; s < eng.n_red_slots; ++s)
    for (int k = 0; k < GPU_RED_WIDTH; ++k)
      sums[k] += red[s * GPU_RED_WIDTH + k];
  global_tally_tracklength += sums[GPU_RED_K_TRACKLENGTH];
  global_tally_collision += sums[GPU_RED_K_COLLISION];
  global_tally_absorption += sums[GPU_RED_K_ABSORPTION];
  global_tally_leakage += sums[GPU_RED_LEAKAGE];

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
          results[(b * td.n_scores + s) * 3 + 0] +=
            acc[td.accum_off + b * td.n_scores + s];
        }
      }
    }
  }
}

void finalize()
{
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
        fmt::format("GPU transport device time: {:.3f} s", eng.gpu_seconds),
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
