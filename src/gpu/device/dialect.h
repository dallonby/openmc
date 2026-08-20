//! \file dialect.h
//! Portable device-code dialect for the OpenMC GPU transport engine.
//!
//! Device code under src/gpu/device/ is written once in a restricted C++
//! subset and compiled for multiple targets:
//!   - Apple Metal (MSL): concatenated + runtime-compiled by backend_metal.mm
//!   - NVIDIA CUDA: included from a .cu translation unit (backend_cuda, future)
//!   - Host C++: included for debugging/reference execution and layout checks
//!
//! Rules for device code:
//!   * No recursion, no virtual dispatch, no dynamic allocation, no exceptions.
//!   * All global-memory pointers carry the GLOBAL qualifier; pointers to
//!     function-local state carry THREAD.
//!   * Shared host/device structs contain only fixed-width scalars (no
//!     pointers, no vector types — MSL float3 has 16-byte alignment).
//!   * Primary compute type is fp32 (Apple GPUs have no fp64). 64-bit
//!     integer math is available on all targets and used for RNG.

#pragma once

// Deterministic arithmetic across targets: no FMA contraction anywhere in
// device code (portable_math.h relies on exact operation sequences).
#pragma STDC FP_CONTRACT OFF

#if defined(__METAL_VERSION__)
// ---------------------------------------------------------------- Metal ----
#include <metal_stdlib>
using namespace metal;

// MSL contracts a*b+c into FMA across statements by default even in safe
// math mode. That changes rounding in cancellation-sensitive expressions
// (quadratic discriminants, dot products) enough to bias transport
// statistically relative to the IEEE host reference. Disable contraction:
// bitwise agreement with the host-compiled engine matters more than the
// fused-multiply-add throughput.
#pragma STDC FP_CONTRACT OFF

#define GPU_TARGET_METAL 1
#define GLOBAL device
#define GCONST constant
#define THREAD thread
#define TGROUP threadgroup
#define DEVICE_FN inline
#define KERNEL kernel

typedef uint uint32_gpu;
typedef int int32_gpu;
typedef ulong uint64_gpu;
typedef long int64_gpu;
typedef ushort uint16_gpu;
typedef uchar uint8_gpu;

// C-style math names for portability with CUDA/host code
#define sqrtf(x) metal::sqrt(x)
#define logf(x) metal::log(x)
#define log2f(x) metal::log2(x)
#define expf(x) metal::exp(x)
#define fabsf(x) metal::fabs(x)
#define fminf(x, y) metal::fmin(x, y)
#define fmaxf(x, y) metal::fmax(x, y)
#define floorf(x) metal::floor(x)
#define ceilf(x) metal::ceil(x)
#define lroundf(x) (int64_gpu) metal::rint(x)
#define copysignf(x, y) metal::copysign(x, y)
#define sinf(x) metal::sin(x)
#define cosf(x) metal::cos(x)
#define tanf(x) metal::tan(x)
#define acosf(x) metal::acos(x)
#define atan2f(x, y) metal::atan2(x, y)
#define powf(x, y) metal::pow(x, y)
#define fmodf(x, y) metal::fmod(x, y)
#define cbrtf(x) metal::powr(metal::fabs(x), 1.0f / 3.0f)
#define isnan_gpu(x) metal::isnan(x)
#define isinf_gpu(x) metal::isinf(x)

// Atomics on device memory
DEVICE_FN float gpu_atomic_add_f(GLOBAL atomic_float* addr, float v)
{
  return atomic_fetch_add_explicit(addr, v, memory_order_relaxed);
}
DEVICE_FN uint32_gpu gpu_atomic_add_u32(GLOBAL atomic_uint* addr, uint32_gpu v)
{
  return atomic_fetch_add_explicit(addr, v, memory_order_relaxed);
}
DEVICE_FN int32_gpu gpu_atomic_add_i32(GLOBAL atomic_int* addr, int32_gpu v)
{
  return atomic_fetch_add_explicit(addr, v, memory_order_relaxed);
}
typedef atomic_float gpu_atomic_f32;
typedef atomic_uint gpu_atomic_u32;
typedef atomic_int gpu_atomic_i32;

#elif defined(__CUDACC__)
// ----------------------------------------------------------------- CUDA ----
#include <cstdint>

#define GPU_TARGET_CUDA 1
#define GLOBAL
#define GCONST
#define THREAD
#define TGROUP __shared__
#define DEVICE_FN __device__ __forceinline__
#define KERNEL extern "C" __global__

typedef uint32_t uint32_gpu;
typedef int32_t int32_gpu;
typedef uint64_t uint64_gpu;
typedef int64_t int64_gpu;
typedef uint16_t uint16_gpu;
typedef uint8_t uint8_gpu;

#define isnan_gpu(x) isnan(x)
#define isinf_gpu(x) isinf(x)

typedef float gpu_atomic_f32;
typedef uint32_t gpu_atomic_u32;
typedef int32_t gpu_atomic_i32;
DEVICE_FN float gpu_atomic_add_f(gpu_atomic_f32* addr, float v)
{
  return atomicAdd(addr, v);
}
DEVICE_FN uint32_gpu gpu_atomic_add_u32(gpu_atomic_u32* addr, uint32_gpu v)
{
  return atomicAdd(addr, v);
}
DEVICE_FN int32_gpu gpu_atomic_add_i32(gpu_atomic_i32* addr, int32_gpu v)
{
  return atomicAdd(addr, v);
}

#else
// ----------------------------------------------------- Host (reference) ----
#include <atomic>
#include <cmath>
#include <cstdint>

#define GPU_TARGET_HOST 1
#define GLOBAL
#define GCONST
#define THREAD
#define TGROUP
#define DEVICE_FN inline
#define KERNEL inline

typedef uint32_t uint32_gpu;
typedef int32_t int32_gpu;
typedef uint64_t uint64_gpu;
typedef int64_t int64_gpu;
typedef uint16_t uint16_gpu;
typedef uint8_t uint8_gpu;

#define isnan_gpu(x) std::isnan(x)
#define isinf_gpu(x) std::isinf(x)

typedef float gpu_atomic_f32;
typedef uint32_t gpu_atomic_u32;
typedef int32_t gpu_atomic_i32;
inline float gpu_atomic_add_f(gpu_atomic_f32* addr, float v)
{
  float old = *addr;
  *addr += v;
  return old;
}
inline uint32_gpu gpu_atomic_add_u32(gpu_atomic_u32* addr, uint32_gpu v)
{
  uint32_gpu old = *addr;
  *addr += v;
  return old;
}
inline int32_gpu gpu_atomic_add_i32(gpu_atomic_i32* addr, int32_gpu v)
{
  int32_gpu old = *addr;
  *addr += v;
  return old;
}
#endif
