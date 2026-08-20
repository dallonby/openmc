//! \file rng.h
//! Device implementation of OpenMC's PCG-RXS-M-XS 64/64 generator
//! (src/random_lcg.cpp), bit-exact with the CPU: identical 64-bit integer
//! state sequences and identical skip-ahead (Brown's O(log2 N) algorithm).
//! The only divergence is the float conversion: the CPU returns the top 53
//! bits as an fp64 in [0,1); here the top 24 bits produce an fp32 in (0,1)
//! (half-ulp centering keeps log(prn) finite without biasing means).

#pragma once

#define GPU_N_STREAMS 4
#define GPU_STREAM_TRACKING 0
#define GPU_STREAM_SOURCE 1
#define GPU_STREAM_URR_PTABLE 2
#define GPU_STREAM_VOLUME 3

#define GPU_PRN_MULT 6364136223846793005ul
#define GPU_PRN_ADD 1442695040888963407ul

// Advance the LCG state and return the RXS-M-XS output word (bit-exact with
// the value the CPU converts via ldexp(word, -64)).
DEVICE_FN uint64_gpu gpu_prn_word(THREAD uint64_gpu* seed)
{
  *seed = GPU_PRN_MULT * (*seed) + GPU_PRN_ADD;
  uint64_gpu word =
    ((*seed >> ((*seed >> 59u) + 5u)) ^ *seed) * 12605985483714917081ul;
  return (word >> 43u) ^ word;
}

// fp32 uniform variate on (0,1)
DEVICE_FN float gpu_prn(THREAD uint64_gpu* seed)
{
  uint64_gpu w = gpu_prn_word(seed);
  return ((float)(w >> 40u) + 0.5f) * 0x1p-24f;
}

// Brown's skip-ahead: coefficients (g,c) such that seed_n = g*seed + c
struct GpuPrnCoeff {
  uint64_gpu mult;
  uint64_gpu add;
};

DEVICE_FN GpuPrnCoeff gpu_future_seed_coefficients(uint64_gpu n)
{
  uint64_gpu g = GPU_PRN_MULT;
  uint64_gpu c = GPU_PRN_ADD;
  uint64_gpu g_new = 1;
  uint64_gpu c_new = 0;
  while (n > 0) {
    if (n & 1ul) {
      g_new *= g;
      c_new = c_new * g + c;
    }
    c *= (g + 1ul);
    g *= g;
    n >>= 1;
  }
  GpuPrnCoeff out;
  out.mult = g_new;
  out.add = c_new;
  return out;
}

DEVICE_FN uint64_gpu gpu_future_seed(uint64_gpu n, uint64_gpu seed)
{
  GpuPrnCoeff k = gpu_future_seed_coefficients(n);
  return k.mult * seed + k.add;
}

// Mirror of init_particle_seeds (src/random_lcg.cpp:109): one coefficient
// computation, N_STREAMS multiply-adds.
DEVICE_FN void gpu_init_particle_seeds(
  int64_gpu id, uint64_gpu master_seed, uint64_gpu stride,
  THREAD uint64_gpu* seeds)
{
  GpuPrnCoeff k = gpu_future_seed_coefficients((uint64_gpu)id * stride);
  for (int i = 0; i < GPU_N_STREAMS; i++) {
    seeds[i] = k.mult * (master_seed + (uint64_gpu)i) + k.add;
  }
}

DEVICE_FN void gpu_advance_prn_seed(int64_gpu n, THREAD uint64_gpu* seed)
{
  *seed = gpu_future_seed((uint64_gpu)n, *seed);
}
