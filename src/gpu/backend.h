//! \file backend.h
//! C ABI between the portable GPU runner and a device backend.
//! Implemented today by backend_metal.mm (Apple Metal). A CUDA backend
//! implements the same functions (omg_cuda_*) and the runner selects one at
//! build time. The ABI is deliberately plain C: no std types cross it, so
//! backends can be built with a different toolchain than libopenmc.

#ifndef OPENMC_GPU_BACKEND_H
#define OPENMC_GPU_BACKEND_H

#include <cstddef>

// Logical buffer slots — must match the kernel wrapper's [[buffer(N)]]
// bindings (metal_main.h) and any future CUDA argument order.
enum OmgSlot {
  OMG_SLOT_CONTROL = 0,
  OMG_SLOT_SURFACES = 1,
  OMG_SLOT_CELLS = 2,
  OMG_SLOT_UNIVERSES = 3,
  OMG_SLOT_LATTICES = 4,
  OMG_SLOT_MATERIALS = 5,
  OMG_SLOT_MGMATS = 6,
  OMG_SLOT_I32 = 7,
  OMG_SLOT_F32 = 8,
  OMG_SLOT_SOURCE = 9,
  OMG_SLOT_FISSION = 10,
  OMG_SLOT_COUNTERS = 11,
  OMG_SLOT_PROGENY = 12,
  OMG_SLOT_REDSLOTS = 13,
  OMG_SLOT_TALLIES = 14,
  OMG_SLOT_FILTERS = 15,
  OMG_SLOT_MESHES = 16,
  OMG_SLOT_TACCUM = 17,
  OMG_SLOT_NUCLIDES = 18,
  OMG_SLOT_TRACE = 19,
  OMG_SLOT_COUNT = 20
};

extern "C" {

//! 1 if a usable device exists
int omg_metal_available(void);

//! Create/destroy an engine context
void* omg_metal_create(void);
void omg_metal_destroy(void* ctx);

//! Compile the device source (returns 0 on success; error text in err)
int omg_metal_compile(void* ctx, const char* src, char* err, int errcap);

//! Ensure buffer at slot has at least `bytes` capacity (shared storage).
//! Returns 0 on success.
int omg_metal_buffer(void* ctx, int slot, size_t bytes);

//! Host pointer to the (unified-memory) contents of a slot buffer
void* omg_metal_contents(void* ctx, int slot);

//! Dispatch kernel `fn` over n threads; blocks until complete.
int omg_metal_dispatch(
  void* ctx, const char* fn, unsigned int nthreads, char* err, int errcap);

//! GPU time of the last dispatch in seconds
double omg_metal_last_time(void* ctx);

//! Device name (UTF-8, valid until context destroyed)
const char* omg_metal_device_name(void* ctx);

} // extern "C"

#endif // OPENMC_GPU_BACKEND_H
