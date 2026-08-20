# OpenMC on Apple Silicon GPUs (Metal)

This branch adds a GPU transport engine to OpenMC for Apple Silicon,
implemented with Metal compute kernels and designed for extension to
NVIDIA/CUDA. It is drop-in compatible: the same `openmc` executable, Python
API, input files, and statepoint outputs — with particle transport running
on the GPU when the model is inside the engine's envelope, and automatic
CPU fallback (with a warning naming the unsupported feature) when it is
not.

```python
import openmc

model = openmc.examples.pwr_pin_cell()   # any normal OpenMC model
model.settings.gpu = True                # <-- the only change
model.run()
```

or, for an existing model directory:

```bash
OPENMC_GPU=1 openmc
```

## Highlights

- **FP32-first compute** ("a more Apple-native datatype than fp64"): Apple
  GPUs have no hardware double precision, so the transport kernels are pure
  fp32 + 64-bit integer math. Batch statistics, bank synchronization, k
  estimation, tallies across batches, and all file output stay in the
  upstream fp64 host pipeline, so statistical accuracy is preserved and
  outputs are byte-compatible.
- **Bit-exact random number streams**: OpenMC's PCG-RXS-M-XS 64/64
  generator, its per-particle skip-ahead seeding, and the URR
  probability-table stream discipline run with integer-identical state
  sequences on the GPU (verified by an on-device self-test at startup).
- **Deterministic cross-platform arithmetic**: transcendentals are
  bit-portable polynomial kernels (no vendor libm, no FMA contraction), so
  Metal, host, and future CUDA builds compute identical bits — vendor ulp
  differences were measured to bias Monte Carlo ensembles (see
  `PORT_NOTES.md`).
- **Fission-bank reproducibility semantics preserved**: sites carry
  `(parent_id, progeny_id)` so the upstream `sort_bank` /
  `synchronize_bank` machinery (uniform combing, entropy, statepoints)
  works unchanged.
- Single-source device code (`src/gpu/device/`) written in a portable
  dialect compiled as MSL at runtime (no offline Metal toolchain needed),
  as host C++ (a reference "replay" engine used for debugging and
  validation), and structured for a CUDA backend (`src/gpu/backend.h`).

## Building

Requirements: macOS on Apple Silicon, Xcode (or CLT) with the Metal
framework, CMake ≥ 3.16, HDF5, libomp (both via Homebrew).

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DHDF5_ROOT=/opt/homebrew/opt/hdf5 \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib ..
make -j
pip install -e ..
```

`OPENMC_USE_METAL` defaults to `ON` on Apple Silicon (`OFF` elsewhere). The
device kernels are embedded as source and compiled by the Metal runtime at
startup — builds need no `xcrun metal`.

## Enabling the GPU

Any one of:

- Python: `model.settings.gpu = True`
- settings.xml: `<gpu>true</gpu>`
- Environment: `OPENMC_GPU=1` (overrides the settings file; `OPENMC_GPU=0`
  disables)

At simulation start the engine flattens the model and data; if anything is
outside the envelope it prints
`GPU transport disabled — <reason>. Running on the CPU instead.` and the
run proceeds normally on the CPU.

## v1 envelope

| Supported | Falls back to CPU |
|---|---|
| k-eigenvalue runs | fixed-source, photon transport, MPI |
| Continuous-energy neutrons: pointwise XS, URR probability tables, free-gas elastic, level/continuum inelastic (uncorrelated, Kalbach-Mann, correlated, N-body), (n,xn), prompt + delayed fission | S(a,b) thermal scattering, windowed multipole, resonance upscattering (DBRC/RVS), multi-temperature models |
| Multigroup: macroscopic isotropic MGXS, tabular/histogram scattering laws, prompt + delayed fission | angle-dependent MGXS, Legendre sampling (use the default `tabular_legendre` conversion) |
| CSG: all quadric surface types, universes, rectangular lattices, translations/rotations, vacuum/reflective/white BCs | tori, hex lattices, periodic BCs, boundary albedo, DAGMC, distribcell/multi-instance materials |
| Tallies: cell/material/universe/energy/mesh filters; flux, total, absorption, fission, nu-fission, scatter, elastic scores; tracklength + collision estimators | analog estimators, nuclide bins, other filters/scores |
| Analog capture, Russian-roulette-free transport (upstream defaults) | survival biasing, weight windows |

## Validation (M3 Ultra, ENDF/B-VIII.0, vs CPU OpenMC from this tree)

| Case | CPU k | GPU k | Agreement |
|---|---|---|---|
| 7-group MG 3×3 pin lattice (10M active histories) | 1.34156 ± 0.00024 | 1.34193 ± 0.00025 | 1.1σ |
| CE PWR pincell, no S(a,b) (3M active) | 1.23939 ± 0.00057 | 1.23971 ± 0.00053 | 0.4σ; 58 tally bins, mean z² = 0.88–1.01 |
| CE Godiva bare HEU sphere (1M active) | 1.00125 ± 0.00059 | 0.99499 ± 0.00071 | −630 pcm on this 57%-leakage benchmark — known open item (device-arithmetic ensemble effect; see `PORT_NOTES.md`) |

## Performance

See `PORT_NOTES.md` and the benchmark table in the branch discussion;
representative numbers on an M3 Ultra (80-core GPU) vs the same tree's CPU
build on 28 threads: multigroup ~2.6M particles/s (GPU) vs ~0.3M (CPU);
thermal CE pincell ~0.7M/s vs ~21k/s (~34×). GPU throughput improves with
larger `particles` per batch (the GPU is under-occupied below ~10^5
particles in flight).

## Architecture

```
src/gpu/
  device/          single-source kernels (Metal / CUDA-ready / host replay)
    dialect.h        target shims: address spaces, types, atomics
    portable_math.h  deterministic fp32 log/exp/sincos/sqrt
    types.h          POD data model shared host/device
    rng.h            PCG-RXS-M-XS 64/64 + skip-ahead (bit-exact)
    geometry.h       CSG tracking (infix regions, lattices, BCs)
    mg.h  ce.h       multigroup / continuous-energy physics
    transport.h      the per-particle history loop
    metal_main.h     Metal kernel entry points (buffer bindings)
  backend.h        C ABI a device backend implements (slots, dispatch)
  backend_metal.mm Metal implementation (runtime MSL compile)
  flatten.{h,cpp}  geometry/MG/tally flatteners (host, fp64 -> fp32)
  flatten_ce.cpp   continuous-energy data flattener
  runner.cpp       per-generation orchestration + fp64 statistics hand-off
```

The GPU replaces only within-generation particle transport
(`transport_history_based()`); everything else is untouched upstream code.
A host-compiled build of the same kernels (the "replay" engine) plus
env-gated audits, paired-history traces, and leak bitmaps (`PORT_NOTES.md`)
provide the validation and debugging story.

## Extending to CUDA

The device code compiles under `__CUDACC__` through `dialect.h` (raw
pointers, `atomicAdd`, `__global__` wrappers). A CUDA backend needs:
`src/gpu/backend_cuda.cu` implementing the `omg_*` ABI of `backend.h` over
the same buffer slots, a kernel wrapper mirroring `metal_main.h`, and
`--fmad=false` for bit parity with the portable math layer. `real_t` is
fp32 throughout; an fp64 CUDA build is possible by widening the typedefs in
`dialect.h` but is deliberately not the default.
