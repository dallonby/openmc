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
| Tallies: cell/material/universe/energy/regular- and cylindrical-mesh filters (up to 4 per tally; nested cell/universe matches score every combination, tracklength mesh tallies do track splitting, all as on CPU); flux, total, absorption, fission, nu-fission, scatter, elastic scores; tracklength + collision estimators | translated/rotated or non-regular mesh filters, analog estimators, nuclide bins, differential tallies, other filters/scores |
| Standard lost-particle accounting and abort thresholds; upstream statepoint/source outputs | track output, surface-source writing, collision-track files, overlap checking |
| Variance reduction in fixed-source runs: survival biasing (implicit capture + Russian roulette) and mesh weight windows (splitting via a global spill bank the host re-dispatches until it drains) | variance reduction in eigenvalue runs, multiple weight-window domains, photon weight windows |

## Validation (M3 Ultra, ENDF/B-VIII.0, vs CPU OpenMC from this tree)

| Case | CPU k | GPU k | Agreement |
|---|---|---|---|
| 7-group MG 3×3 pin lattice (10M active histories) | 1.34156 ± 0.00024 | 1.34172 ± 0.00023 | 0.5σ; 70 tally bins; 0 lost particles |
| CE PWR pincell, no S(a,b) (3M active) | 1.23939 ± 0.00057 | 1.23923 ± 0.00052 | 0.2σ; 58 tally bins |
| CE PWR pincell **with S(a,b)** (3M active) | 1.23691 ± 0.00054 | 1.23691 ± 0.00053 | 0.0σ; 58 tally bins |
| CE Godiva bare HEU sphere (1M active) | 1.00125 ± 0.00059 | 1.00138 ± 0.00071 | 0.1σ; 0 lost particles |
| CE Godiva bare HEU sphere (11M active) | 1.00041 ± 0.00022 | 1.00017 ± 0.00021 | 0.8σ (Δ = −24 pcm); leakage fraction agrees to 0.9σ |
| CE Godiva bare HEU sphere (400M active) | 1.000051 ± 0.000032 | 1.000013 ± 0.000035 | **Δ = −3.8 ± 4.7 pcm** — any residual fp32-vs-fp64 bias is within [−13, +6] pcm at 95%; leakage fraction identical (0.57310) |
| Fixed-source deep-penetration slab (0.6 m W / WC+H₂O / W₂B₅ / W₂B₅+H₂O, 14.06 MeV, fast attenuation up to 1.5×10⁵) | 60-bin depth profiles | 60-bin depth profiles | every bin within statistics (mean z² 0.6–1.5 over 480 bins); material flux ratios agree to ~0.5–1% |

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

Active-batch calculation rates on an M3 Ultra (80-core GPU), current
build, each row a **single model run on both engines** (no cross-problem
comparisons). CPU rates are the best over a thread sweep; the "fixed CPU"
column reflects the per-thread-accumulator fix (PORT_NOTES finding 3).

| Case | GPU | CPU best (fixed) | GPU speedup | note |
|---|---|---|---|---|
| MG 7-group 3×3 lattice, 70 tally bins | 3.25 M/s | 0.156 M/s (8t) | **21×** | CPU still collapses above 8t — the bottleneck here is per-bin tally-scoring atomics, which the accumulator fix does **not** address |
| CE Godiva (bare sphere, eigenvalue) | 14.9 M/s | 4.69 M/s (16t) | **3.2×** | the accumulator fix removed this CPU cliff, so this is a clean compute-vs-compute ratio |
| CE tungsten deep-penetration slab (60 cm, fixed source) | 1.71 M/s (60-cell) / 2.54 M/s (mesh tally) | 0.41 M/s (16t) | **4.2× / 6.2×** | the PROCESS-shielding workload |

**Read these honestly.** The largest multiples occur where the CPU
baseline is throttled by atomic contention that the GPU's per-thread
accumulation avoids — MG (21×) is as much "the CPU still contends on tally
bins" as "the GPU is fast". Where that contention was removed (Godiva),
the honest ratio is ~3×, which is closer to the real fp32-GPU-vs-CPU
compute advantage on these parts. Earlier revisions of this file quoted
Godiva at 6× against a *contention-crippled* 4-thread CPU baseline; those
numbers were inflated by a CPU bug since fixed, and are corrected here.

The GPU wins on every workload measured; the *size* of the win depends on
how contended the CPU comparison is. Deep-penetration shielding — the
workload this port was built for — runs 4–6× a properly-tuned CPU, with
the mesh-tally path (single shield cell + tracklength track splitting) at
the top of that range.

Both of those supposed levers have since been measured and neither is the
constraint. History-length divergence is real (SIMD efficiency 10.9% on the
W slab) but not binding: a persistent-thread work queue raises lane
occupancy to 40% and buys only 9%. Tally atomics have ample headroom — a
microbenchmark puts Apple's native float atomics at 27 G/s against the
tally path's 14% share. An Xcode GPU capture shows the kernel is bandwidth
bound at the last-level cache, with geometry the dominant consumer of
global accesses, so the live lever is delta tracking, which removes
surface-distance work rather than rearranging it. See PORT_NOTES.

## Variance reduction (fixed source)

Deep-penetration shielding is dominated by the rare deep survivors, so
raw speed matters less than variance reduction. Both mechanisms are
ported and gated on the analog result:

| Depth in a 0.6 m tungsten slab | survival biasing | weight windows |
|---|---|---|
| 30 cm | 0.6× | 0.2× |
| 45 cm | 1.7× | 1.6× |
| 55 cm | 0.8× | **5.8×** |
| 60 cm | 0.2× | **5.5×** |

(Figure of merit `1/(relative error² × runtime)` relative to an analog run
of the same model, hand-tuned exponential windows `lower ∝ e^{-x/8cm}`.
Read these as single-realization estimates: the deepest bins have the
fewest counts, so their FOM is itself noisy — an earlier single run of
this same configuration gave 19.9× at 60 cm, which did not reproduce, and
the repeatable figure is ~5–6×. Gains depend strongly on how well the
windows are tuned; a generated (MAGIC/FW-CADIS) set should do better than
this hand-fit.) The pattern that *is* robust: weight windows cost time in
the shallow bins and buy it back at depth, which is the whole point.

**Unbiasedness** is the property that matters more than the speedup, and
it is tested rather than assumed: against a 40M-history analog reference,
two weight-window runs with independent seeds land at mean z = +1.55 and
−1.34 (deep-half mean ratios 1.0051 and 0.9956) — the deviation flips
sign with the seed, which is fluctuation, not bias. An analog run at the
same statistics shows the same spread against the reference. The test
bounds any bias at roughly the ±0.5% level on the deep bins.

Splitting uses the per-thread secondary stack first and spills to a global
bank when it is full; the host drains that bank in source-sized chunks,
re-dispatching until it empties, with tallies and keff accumulating across
passes. Spilled particles carry their parent's RNG state so they resume
that stream rather than replaying a primary particle's sequence.

**Saturation degrades gracefully rather than biasing.** When the banks are
full the device makes *fewer* split copies and the parent keeps the weight
the missing copies would have carried; an (n,xn) clone that cannot be
placed falls back to implicit multiplication (the parent's weight is
multiplied), which is the treatment OpenMC already uses for non-integer
yields. Weight is conserved exactly in both cases, so the estimator stays
unbiased and only the variance reduction weakens — the run warns that it
delivered less splitting than requested. A stress case (`max_split` 40,
e-folding 4 cm, banks saturating with ~35M requested copies per
generation) returns a deep-half flux ratio of 1.0071 against the
40M-history analog reference; before this treatment the same case came
back 74% low.

## Architecture## Architecture

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
