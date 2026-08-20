//! \file flatten.h
//! Host-side flattened model: POD arenas built from the loaded OpenMC data
//! model, ready for upload to a GPU backend.

#ifndef OPENMC_GPU_FLATTEN_H
#define OPENMC_GPU_FLATTEN_H

#include <cstdint>
#include <string>
#include <vector>

// device POD definitions, compiled for host
#include "device/dialect.h"
#include "device/types.h"
#include "device/rng.h"
#include "device/geometry.h"
#include "device/mg.h"

namespace openmc {
namespace gpu {

struct FlatModel {
  std::vector<GpuSurface> surfaces;
  std::vector<GpuCell> cells;
  std::vector<GpuUniverse> universes;
  std::vector<GpuLattice> lattices;
  std::vector<GpuMaterial> materials;
  std::vector<GpuMgMat> mgmats;
  std::vector<int32_t> i32;
  std::vector<float> f32;

  std::vector<GpuTallyDesc> tallies;
  std::vector<GpuFilterDesc> filters;
  std::vector<GpuMesh> meshes;
  std::vector<int32_t> tally_host_index; // model::tallies index per desc
  uint32_t tally_accum_size {0};

  uint32_t mg_bin_avg_off {0};

  //! empty when the model flattened cleanly; otherwise names the first
  //! unsupported feature (the engine then stays inactive)
  std::string reject_reason;
};

//! Build the flat model. Returns false (with reject_reason set) if the model
//! uses features outside the GPU engine's v1 envelope.
bool flatten_model(FlatModel& out);

//! Re-flatten only the tally descriptors (tally activation changes per
//! batch); returns false with reason on unsupported active tallies.
bool flatten_tallies(FlatModel& out);

} // namespace gpu
} // namespace openmc

#endif
