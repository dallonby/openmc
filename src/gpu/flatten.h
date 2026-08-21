//! \file flatten.h
//! Host-side flattened model: POD arenas built from the loaded OpenMC data
//! model, ready for upload to a GPU backend.

#ifndef OPENMC_GPU_FLATTEN_H
#define OPENMC_GPU_FLATTEN_H

#include <cstdint>
#include <string>
#include <vector>

// device POD definitions compiled for the host — order matters
// clang-format off
#include "device/dialect.h"
#include "device/types.h"
#include "device/rng.h"
#include "device/portable_math.h"
#include "device/geometry.h"
#include "device/mg.h"
#include "device/f1d.h"
#include "device/sab.h"
#include "device/ce.h"
// clang-format on

// device POD definitions, compiled for host

namespace openmc {
namespace gpu {

struct FlatModel {
  std::vector<GpuSurface> surfaces;
  std::vector<GpuCell> cells;
  std::vector<GpuUniverse> universes;
  std::vector<GpuLattice> lattices;
  std::vector<GpuMaterial> materials;
  std::vector<GpuMgMat> mgmats;
  std::vector<GpuNuclide> nuclides;
  std::vector<GpuSabTable> sab_tables;
  std::vector<int32_t> i32;
  std::vector<float> f32;

  std::vector<GpuTallyDesc> tallies;
  std::vector<GpuFilterDesc> filters;
  std::vector<GpuMesh> meshes;
  std::vector<int32_t> tally_host_index; // model::tallies index per desc
  uint32_t tally_accum_size {0};

  uint32_t mg_bin_avg_off {0};
  uint32_t mg_default_iv_off {0};
  uint32_t surf_adj_off {0};

  // weight windows (fixed-source variance reduction); ww_mesh < 0 = disabled
  int32_t ww_mesh {-1};
  uint32_t ww_n_energy {1};
  uint32_t ww_ebounds_off {0};
  uint32_t ww_lower_off {0};
  uint32_t ww_upper_off {0};
  uint32_t ww_n_mesh_bins {0};
  double ww_survival_ratio {0.5};
  double ww_max_lb_ratio {1.0};
  double ww_weight_cutoff {1.0e-38};
  int32_t ww_max_split {10};

  //! empty when the model flattened cleanly; otherwise names the first
  //! unsupported feature (the engine then stays inactive)
  std::string reject_reason;
};

//! Build the flat model. Returns false (with reject_reason set) if the model
//! uses features outside the GPU engine's v1 envelope.
bool flatten_model(FlatModel& out);

//! CE-mode data flattening (implemented in flatten_ce.cpp)
bool flatten_ce(FlatModel& out);

//! Re-flatten only the tally descriptors (tally activation changes per
//! batch); returns false with reason on unsupported active tallies.
bool flatten_tallies(FlatModel& out);

} // namespace gpu
} // namespace openmc

#endif
