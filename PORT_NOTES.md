# Metal GPU Port — Engineering Notes

Working notes for the `metal` branch: findings about upstream OpenMC made
during the port, fp32/GPU-specific design decisions with their evidence, and
open items. Validation numbers are from an Apple M3 Ultra (96 GB), macOS 26,
ENDF/B-VIII.0, against CPU OpenMC built from this same tree.

## Upstream findings

1. **`Mgxs::get_xs` CHI_DELAYED summed the wrong tensor** (fixed on this
   branch in `src/mgxs.cpp`). The `gout == nullptr` branches indexed the
   rank-3 `delayed_nu_fission` tensor `[angle][dg][gin]` with four indices
   and iterated `shape(3)` of a rank-3 tensor; the sums are over outgoing
   groups so they must read `chi_delayed` `[angle][dg][gin][gout]`.
   Failure mode: `tensor::Tensor::operator()` has no rank check
   (`include/openmc/tensor.h:568`), so the four-index read walks past the
   stride vector, and `delayed_nu_fission.shape(3)` returns 0, which makes
   the dg==nullptr branch silently return 0. The branch is unreachable
   from current callers (the only external CHI_* caller,
   `src/random_ray/flat_source_domain.cpp:1208`, passes a non-null
   `gout`), so it never produced wrong physics upstream — but any new
   caller would have hit it. The rank-4-vs-comment mismatch in
   `include/openmc/xsdata.h` (which documented chi_delayed as
   `[angle][in][out][dg]`) is fixed on this branch too. Candidate for an
   upstream PR; the missing rank check in `tensor::Tensor::operator()` is
   a second hardening candidate.

2. **`Nuclide::reaction_index_` narrows `SIZE_MAX` to `int`**
   (`nuclide.cpp` fills the `array<size_t, 902>` with `C_NONE` (=-1, i.e.
   `SIZE_MAX`) and reads entries into `int`, testing `< 0`). Correct only
   through the narrowing wraparound; worth an `int32_t` cleanup upstream.

3. **CPU tally scoring collapses above ~8 threads on Apple Silicon**
   (observation, not a fork change): with ~50 tally bins, Godiva runs
   1.70M/s at 4 threads, 1.17M at 8, 0.31M at 16, 0.15M at 28 — the
   `#pragma omp atomic` tally accumulation cache-line ping-pongs, likely
   compounded by libomp scheduling across P+E cores. Without tallies,
   28 threads still only reaches 0.98M/s vs 0.80M single-threaded.
   Reported CPU baselines in README_METAL.md therefore use each case's
   best thread count. Possibly worth per-thread tally buffers upstream on
   many-core targets.

4. **The event-based queue sort is dead code** (`src/event.cpp:81` —
   commented out pending TBB); `EventQueueItem::operator<` exists and works.
   Noted for the future event-based GPU mode, where the same sort is the
   standard divergence-reduction step.

## fp32 design decisions (and their evidence)

### Geometry tolerances

CPU tolerances (`FP_COINCIDENT` 1e-12, `FP_PRECISION` 1e-14, `TINY_BIT`
1e-8) are below fp32 resolution and were re-derived (see
`src/gpu/device/types.h`). Two changes go beyond re-scaling:

* **Plane surfaces have no coincidence band on the GPU.** With a band, a
  particle that collides within the band of a boundary-condition plane and
  scatters outward sees the wall as "already on it" (distance = inf) while
  a coincident lattice edge still fires — it escapes through the corner of
  the geometry. At fp64 the band (1e-12) is statistically unreachable; at
  fp32 a 1e-5..1e-6 band leaked ~25 particles per 10^4 histories in the MG
  pin lattice. The on-surface token plus the sign of the distance handle
  every plane case exactly; a d≈0 crossing correctly fires the BC and the
  token flip prevents re-crossing loops.

* **Boundary selection has an absolute significance floor**
  (`GPU_FP_ABS_TIEBREAK`): a lattice edge may steal the boundary from a
  coincident real surface only when it is closer by more than 1e-6 cm.
  With near-zero distances the CPU's purely relative guard
  (`FP_REL_PRECISION`) amplifies fp32 noise and hands the crossing to the
  lattice, skipping the boundary condition.

### Numerically stable CDF inversion

The textbook lin-lin tabular inversion
`x_k + (sqrt(p^2 + 2m(r-c)) - p)/m` collapses to the bin's left edge in
fp32 whenever a pdf is nearly flat: with `m ~ ulp(p)`, `p^2 + 2m(r-c)`
rounds back to `p^2`. ENDF angular tables store fp64 pdfs that cast to
"almost equal" fp32 values, so this fired constantly (U-235 MT=52 at 4 MeV
sampled mu = {-1, 0, +1} bin edges; ⟨mu⟩ biased -0.088). The algebraically
identical form `x_k + 2(r-c)/(p + sqrt(p^2 + 2m(r-c)))` is stable as
m → 0 and needs no flat-pdf special case. Applied in `gpu_sample_tabular`
(CE) and the MG tabular angle sampler.

### Deterministic portable math (`src/gpu/device/portable_math.h`)

Vendor fp32 math libraries disagree at the ulp level: measured against host
libm on 10^6 samples, Metal (safe math mode) differs on 26% (sqrt), 43-47%
(sin/cos/log), 82% (exp) of calls, each with |mean signed relative error|
≤ 4e-8. Individually harmless — but a paired-input experiment (identical
frozen 2M-particle source, identical RNG streams) showed the device
ensemble leaking 0.13% absolute less than the host-compiled build of the
*same engine source* (replay), an 11.7 sigma asymmetry among outcome-
flipped histories, while replay matched fp64 CPU OpenMC to 3e-5.
log/exp/sin/cos/sqrt are therefore implemented as bit-portable polynomial
kernels (Cephes-derived, no FMA, FP contraction off) that produce identical
bits on Metal, host, and any IEEE fp32 target: the device math probe
verifies 0 bit-mismatches device-vs-host for all five functions at engine
startup builds.

### RNG

OpenMC's PCG-RXS-M-XS 64/64 runs bit-exact on Apple GPUs (64-bit integer
ops), including Brown's O(log N) skip-ahead used for per-particle seeding
and the URR `STREAM_URR_PTABLE` correlation discipline. An on-device
self-test verifies integer-identical streams at every engine
initialization. The only divergence from CPU is the uniform conversion:
top-24-bit `(bits + 0.5) * 2^-24` in (0,1) instead of 53-bit fp64 —
midpoint offset keeps `log(xi)` finite without measurable bias (1.8e-8 on
the mean flight length).

## Validation status (2026-08-20)

| Case | CPU | GPU | Agreement |
|---|---|---|---|
| 7-group MG 3x3 pin lattice, 10M active | k=1.34156(24) | k=1.34193(25) | 1.1 sigma |
| CE PWR pincell (no S(a,b)), 3M active | k=1.23939(57) | k=1.23971(53) | 0.4 sigma; 58 tally bins mean z^2 = 0.88-1.01 |
| CE PWR pincell with S(a,b), 3M active | k=1.23691(54) | k=1.23672(47) | 0.3 sigma; 58 tally bins mean z^2 = 0.23-1.25 |
| CE Godiva (57% leakage), 1M active | k=1.00125(59) | k=0.99499(71) | **-630 pcm — open item, see below** |

Lost particles: ~1e-4 of histories (pincell), 0 (Godiva), ~7e-6 (MG
lattice); counted and warned per generation.

## Open items

1. **Device-arithmetic ensemble bias on leakage-dominated fast systems**
   (the Godiva -0.5%). Forensic state: per-event physics verified unbiased
   (sampler A/B audits vs the CPU objects at 6 energies; XS chain to 2e-8
   mean; full elastic kinematics swept 1e4-1.5e7 eV paired-seed); host-
   compiled engine (replay) matches fp64 CPU to 3e-5 leak on frozen
   sources; the Metal-compiled binary of the same source diverges from
   replay on 2.66% of paired histories with an 11.7 sigma non-leak
   asymmetry — *stable under safe math mode, FP contraction off, and the
   portable math layer* (bit-identical math verified on device). A traced
   divergent history is bitwise identical through five events including
   RNG cursor fingerprints, then samples a different level-inelastic mu
   from an apparently identical cursor. Next steps: trace the angle-table
   index and r1 bits inside the sampler on-device; suspect remaining
   candidates are a Metal compiler transform in the branchy tabular-scan
   loops or an unnoticed address-space aliasing effect. Moderated systems
   are unaffected (pincell 0.4 sigma).
2. Fixed-source mode, MPI, photon transport, DAGMC, hex lattices, tori,
   periodic BCs, survival biasing, weight windows, multi-temperature
   models, distribcell — all detected and fall back to CPU with a warning.
3. Event-based device pipeline (history-based v1 leaves SIMD occupancy on
   the table for CE); unionized/material-major energy grids.
4. CUDA backend: the dialect and backend ABI are in place
   (`src/gpu/device/dialect.h`, `src/gpu/backend.h`); needs
   `backend_cuda.cu` implementing the same slots and a kernel wrapper, plus
   `-ffp-contract=off`/`--fmad=false` for bit parity with the portable
   math layer.

## Debug tooling (env-gated, zero cost when unset)

| Variable | Effect |
|---|---|
| `OPENMC_GPU` | enable the GPU engine (equivalent to `settings.gpu`) |
| `OPENMC_GPU_REPLAY` | run each generation on the host-compiled engine too (diagnostics) |
| `OPENMC_GPU_AUDIT` / `AUDIT2` / `AUDIT3` / `AUDIT4` | sampler and XS audits vs the CPU implementations |
| `OPENMC_GPU_MATHPROBE` | device math bit-comparison vs host |
| `OPENMC_STATS` | per-event transport statistics (CPU side) |
| `OPENMC_TRACE_ID=n` | paired event trace of particle n (CPU / replay / device) |
| `OPENMC_LEAK_MAP=path` | per-particle leak bitmaps (CPU / replay / device) |
| `OPENMC_ISO_MU` | ablation: force isotropic CM elastic in both engines |
