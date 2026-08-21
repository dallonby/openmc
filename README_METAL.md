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
| k-eigenvalue runs; fixed-source runs in non-multiplying models (the external source is host-sampled in fp64 with the upstream per-particle seed discipline) | fixed-source with fissionable materials (subcritical multiplication), photon transport, MPI |
| Continuous-energy neutrons: pointwise XS, URR probability tables, free-gas elastic (`free_gas_threshold` honored), S(a,b) thermal scattering (coherent/incoherent elastic, continuous + discrete inelastic), level/continuum inelastic (uncorrelated, Kalbach-Mann, correlated, N-body), (n,xn), prompt + delayed fission, energy cutoff | windowed multipole, resonance upscattering (DBRC/RVS), multi-temperature models, temperature interpolation, NCrystal, isotropic-in-lab (p0) scattering, time cutoffs |
| Multigroup: macroscopic isotropic MGXS, tabular/histogram scattering laws, prompt + delayed fission | angle-dependent MGXS, Legendre sampling (use the default `tabular_legendre` conversion) |
| CSG: all quadric surface types, universes, rectangular lattices, translations/rotations, vacuum/reflective/white BCs | tori, hex lattices, periodic BCs, boundary albedo, DAGMC, distribcell/multi-instance materials |
| Tallies: cell/material/universe/energy filters (up to 4 per tally, nested cell/universe matches score every combination as on CPU); flux, total, absorption, fission, nu-fission, scatter, elastic scores; tracklength + collision estimators | mesh filters (tracklength track-splitting not ported), analog estimators, nuclide bins, differential tallies, other filters/scores |
| Standard lost-particle accounting and abort thresholds; upstream statepoint/source outputs | track output, surface-source writing, collision-track files, overlap checking |
| Analog capture, Russian-roulette-free transport (upstream defaults) | survival biasing, weight windows |

## Validation (M3 Ultra, ENDF/B-VIII.0, vs CPU OpenMC from this tree)

| Case | CPU k | GPU k | Agreement |
|---|---|---|---|
| 7-group MG 3×3 pin lattice (10M active histories) | 1.34156 ± 0.00024 | 1.34172 ± 0.00023 | 0.5σ; 70 tally bins; 0 lost particles |
| CE PWR pincell, no S(a,b) (3M active) | 1.23939 ± 0.00057 | 1.23923 ± 0.00052 | 0.2σ; 58 tally bins |
| CE PWR pincell **with S(a,b)** (3M active) | 1.23691 ± 0.00054 | 1.23691 ± 0.00053 | 0.0σ; 58 tally bins |
| CE Godiva bare HEU sphere (1M active) | 1.00125 ± 0.00059 | 1.00138 ± 0.00071 | 0.1σ; 0 lost particles |
| CE Godiva bare HEU sphere (11M active) | 1.00041 ± 0.00022 | 1.00017 ± 0.00021 | 0.8σ (Δ = −24 pcm); leakage fraction agrees to 0.9σ |
| CE Godiva bare HEU sphere (400M active) | 1.000051 ± 0.000032 | 1.000013 ± 0.000035 | **Δ = −3.8 ± 4.7 pcm** — any residual fp32-vs-fp64 bias is within [−13, +6] pcm at 95%; leakage fraction identical (0.57310) |

Lost particles are counted into the standard `n_lost_particles`
accounting and the upstream abort thresholds apply (raise
`max_lost_particles` in settings exactly as on the CPU if a model needs
it). After the fp32 geometry hardening the measured residual rates are 0
(MG lattice, Godiva) and ~2×10⁻⁶ (S(α,β) pincell).

A device-vs-host ensemble discrepancy on Godiva (−630 pcm) that shipped
in the first push was root-caused to the Metal compiler miscompiling a
(formally illegal) recursive distribution dispatcher and is fixed —
the Metal engine now reproduces the host-compiled engine's leak outcome
**bit-for-bit on 2M paired histories** (see `PORT_NOTES.md`).

## Performance

Active-batch calculation rates with tallies enabled on an M3 Ultra
(80-core GPU), solo runs. **CPU baselines use each case's best measured
thread count** — on this machine the CPU tally path scales *negatively*
above ~8 threads (atomic contention on tally bins: Godiva with tallies
runs 1.70M/s at 4 threads but 0.16M/s at 28), so all-cores numbers
flatter the GPU misleadingly. An x86_64 build of the same tree running
under Rosetta 2 is included as the migration baseline.

| Case | particles/batch | CPU best (native arm64) | CPU best (x86_64 under Rosetta) | GPU | GPU vs native | GPU vs Rosetta |
|---|---|---|---|---|---|---|
| MG 7-group pin lattice | 100k | 149k/s (8t) | 115k/s (8t) | 3.11M/s | 21× | 27× |
| MG 7-group pin lattice | 1M | ~149k/s | ~115k/s | 3.93M/s | 26× | 34× |
| CE pincell with S(a,b) | 20k | 70k/s (4t) | 60k/s (8t) | 583k/s | 8.3× | 9.7× |
| CE Godiva (fast) | 100k | 1.70M/s (4t) | 1.24M/s (8t) | 10.2M/s | 6.0× | 8.2× |
| CE Godiva (fast) | 1M | ~1.70M/s | ~1.24M/s | 10.8M/s | 6.4× | 8.7× |

Rosetta's translation penalty is cleanest single-threaded: native/Rosetta
per-core throughput is 1.84× (MG), 1.37× (S(a,b) pincell), 1.37×
(Godiva); at the multi-thread optimum the shared contention bottleneck
partially masks it (1.16–1.37×). Rosetta runs reproduced native k
bit-for-bit on MG and Godiva with matched seeds.

GPU throughput improves with larger `particles` per batch (the GPU is
under-occupied below ~10^5 particles in flight). Practical CPU tip
independent of the GPU: on many-core Apple Silicon, run tallied problems
at 4–8 OpenMP threads, not all cores.

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
