//! \file gpu_interface.h
//! Host-side interface to the GPU transport engine (Apple Metal backend
//! today; the device code and backend ABI are structured so an NVIDIA/CUDA
//! backend can be added without touching the kernels — see README_METAL.md
//! and PORT_NOTES.md at the repository root).

#ifndef OPENMC_GPU_INTERFACE_H
#define OPENMC_GPU_INTERFACE_H

#include <string>

namespace openmc {
namespace gpu {

//! Attempt to set up the GPU engine for the current model. Called at the end
//! of openmc_simulation_init() when settings::gpu is true. On any
//! unsupported feature the engine is left inactive and a warning naming the
//! feature is emitted; the run proceeds on the CPU.
void try_initialize();

//! True when the engine initialized successfully and transport should be
//! dispatched to the device.
bool active();

//! Run one generation of particle transport on the device, filling the
//! fission bank, global keff estimators, and active tally results exactly
//! where the CPU transport loop would.
void transport_generation();

//! Release device resources.
void finalize();

//! Reason the engine is inactive (empty when active); for diagnostics.
const std::string& inactive_reason();

} // namespace gpu
} // namespace openmc

#endif // OPENMC_GPU_INTERFACE_H
