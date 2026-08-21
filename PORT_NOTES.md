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
   `[angle][in][out][dg]`) is fixed on this branch too. Submitted upstream as
   [PR #4067](https://github.com/openmc-dev/openmc/pull/4067); the missing
   rank check in `tensor::Tensor::operator()` is a second hardening
   candidate.

2. **`Nuclide::reaction_index_` narrows `SIZE_MAX` to `int`**
   (`nuclide.cpp` fills the `array<size_t, 902>` with `C_NONE` (=-1, i.e.
   `SIZE_MAX`) and reads entries into `int`, testing `< 0`). Correct only
   through the narrowing wraparound; worth an `int32_t` cleanup upstream.

3. **CPU throughput collapsed above ~8 threads on Apple Silicon —
   root-caused and fixed on this branch.** The cause was NOT the tally
   bins but the per-particle `#pragma omp atomic` updates of shared
   globals: `simulation::total_weight` at the start of every history
   (`initialize_particle_track`) and the four `global_tally_*` keff
   accumulators at the end (`Particle::event_death`). On the two-die M3
   Ultra those serialize every history across all cores; a 28-thread
   `sample` profile showed exactly those frames dominating. The fix
   (`src/particle.cpp`) accumulates into cache-line-padded (`alignas(64)`)
   per-thread slots and flushes them into the globals once per generation
   (`flush_thread_accumulators`). Godiva with tallies went from 1.74M/s
   at 16 threads / 0.81M/s at 28 to 4.69M/s / 3.92M/s, results
   bit-identical (k and every tally bin unchanged to the last digit).
   This is a strong upstream candidate for many-core hosts. A smaller
   residual (16 -> 28 threads still droops slightly) is the remaining
   per-score tally atomics; threadgroup/thread-local tally tiles are the
   next step.

4. **The event-based queue sort is dead code** (`src/event.cpp:81` —
   commented out pending TBB); `EventQueueItem::operator<` exists and works.
   Noted for the future event-based GPU mode, where the same sort is the
   standard divergence-reduction step.

5. **MG non-analog tally scores apply `density_mult` inconsistently**
   (observation, mirrored on this branch for parity): `Mgxs::calculate_xs`
   multiplies the cached macro total/absorption/nu-fission by
   `p.density_mult()`, and `SCORE_TOTAL` / `SCORE_ABSORPTION` score from
   that cache — but non-analog `SCORE_FISSION` / `SCORE_NU_FISSION` /
   `SCORE_SCATTER` go through `Mgxs::get_xs`, which never applies
   `density_mult` (`tally_scoring.cpp`). For any cell with a density
   multiplier ≠ 1 in MG mode, total/absorption and fission/scatter tallies
   use different densities. The GPU scores reproduce the CPU behavior
   exactly (`gpu_mg_score_xs`). Reported upstream as
   [issue #4070](https://github.com/openmc-dev/openmc/issues/4070).

6. **`ContinuousTabular::sample` applies the lin-lin inversion to discrete
   lines** (`distribution_energy.cpp`): after a discrete hit (`r1 < c[k]`,
   `k < n_discrete`), the histogram branch correctly returns the line
   energy, but a lin-lin table falls into the continuous inversion with a
   *negative* `r1 - c_k`, shifting the sampled energy off the discrete
   line. The GPU sampler deliberately deviates here and returns the exact
   line energy for any discrete hit. A census of ENDF/B-VIII.0 found the
   combination in 19,913 secondary-photon tables (discrete gammas of
   (n,2n)/(n,3n)/(n,n') continuum) and zero neutron-product tables, so
   coupled neutron-photon runs sample smeared gamma lines while
   neutron-only results are unaffected. Reported upstream as
   [issue #4068](https://github.com/openmc-dev/openmc/issues/4068) with
   fix [PR #4069](https://github.com/openmc-dev/openmc/pull/4069). Related quirk, mirrored
   rather than fixed: `CorrelatedAngleEnergy::sample_dist` leaves `c_k1`
   stale (== `c_k`) when the CDF walk exhausts the last bin, so the
   nearest-CDF angle-table pick always chooses table `k+1` there.

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

## Validation status (2026-08-21, after both review fix waves)

| Case | CPU | GPU | Agreement |
|---|---|---|---|
| 7-group MG 3x3 pin lattice, 10M active | k=1.34156(24) | k=1.34172(23) | 0.5 sigma; 70 tally bins; **0 lost particles** |
| CE PWR pincell (no S(a,b)), 3M active | k=1.23939(57) | k=1.23923(52) | 0.2 sigma; 58 tally bins |
| CE PWR pincell with S(a,b), 3M active | k=1.23691(54) | k=1.23691(53) | 0.0 sigma; 58 tally bins |
| CE Godiva (57% leakage), 1M active | k=1.00125(59) | k=1.00138(71) | 0.1 sigma; 0 lost |
| CE Godiva, 11M active | k=1.00041(22) | k=1.00017(21) | 0.8 sigma (dk = -24 pcm); leakage fraction 0.57294(15) vs 0.57314(16) |
| CE Godiva, 400M active | k=1.000051(32) | k=1.000013(35) | dk = -3.8 +/- 4.7 pcm (0.8 sigma); 95% CI on any residual fp32-vs-fp64 bias: [-13, +6] pcm. Leakage fraction 0.57310 on both engines. 16.8M histories/s on the GPU (1M/batch, no tallies) |

Lost particles after the fp32 geometry hardening (below): 0 (MG lattice,
11M histories), 0 (Godiva), 8 per 3.4M (S(a,b) pincell — reconcile-class
residual, 2.4e-6). The GPU feeds `simulation::n_lost_particles` and
enforces the upstream abort thresholds (`max_lost_particles` /
`rel_max_lost_particles`), checked per generation; models that exceed
them can raise the standard settings exactly as on the CPU.

Cross-ISA check of the CPU reference itself: an x86_64 build of this tree
run under Rosetta 2 reproduces the native arm64 build **bit-for-bit** on
Godiva (15M histories: identical k to all digits, all 57 tally bins
exactly equal) and on the MG lattice; the S(a,b) pincell differs only by
trajectory reshuffling from ulp-level libm differences under translation
(dk = +79 pcm at 1.1 sigma over 3M histories, 58 bins mean z^2
0.18-0.66). The fp64 CPU reference used for GPU validation is therefore
instruction-set-independent.

## Resolved: the Godiva device-vs-host bias was a Metal recursion miscompile

The -630 pcm Godiva discrepancy (and the device-vs-replay paired-history
divergence behind it) is **fixed and root-caused**. `gpu_sample_dist`
dispatched MULTI (applicability) and UNCORR (angle-wrapper) AngleEnergy
laws by *recursing into itself*. MSL formally forbids recursion; the
Metal compiler accepted the code and silently miscompiled it, corrupting
a fraction of sampled secondaries (level-inelastic mu prominently — the
forensic trace of "bitwise identical through five events, then a
different mu from an identical RNG cursor" was this). The host-compiled
replay engine compiled the same recursion correctly, which is why replay
matched fp64 CPU while the device did not, and why the effect survived
safe math mode, FP contraction settings, and the bit-portable math layer
— it was never a floating-point effect.

Fix: MULTI/UNCORR now resolve through a bounded iterative redirect loop
(`gpu_sample_dist` -> `gpu_sample_dist_terminal`), preserving the CPU RN
order exactly.

Evidence (Godiva, one 2M-particle generation, identical source and
seeds, device vs host-compiled replay of the same engine source):

| Build | paired leak-bit disagreements | device leak | replay leak |
|---|---|---|---|
| iterative dispatch (fixed) | **0 / 2,000,000 (0.0000%)** | 0.418545 | 0.418545 |
| recursive dispatch (ablation) | 53,182 (2.6591%) | 0.417245 | 0.418545 |

The ablation reproduces the historical 2.66% flip rate and the 0.13%
absolute leak deficit exactly; restoring the iterative dispatcher returns
the engine to bit-identical leak outcomes. k on Godiva moved from
0.99499(71) to 1.00138(71) at 1M histories (CPU: 1.00125(59)) and agrees
to -24 pcm at 11M histories.

Practical rule for this codebase (and the future CUDA backend): **no
recursion in device code, ever, even when the toolchain appears to accept
it.** The `sab.h` MIXED_EL redirect loop already followed this rule; the
distribution dispatcher now does too.

## Open items

1. Fixed-source mode is supported for non-multiplying models (the
   external source is host-sampled per particle in fp64 with the upstream
   STREAM_SOURCE seed discipline; the device fission blocks are gated by
   run mode, and the fission-bank/progeny copy-back is eigenvalue-only).
   A deep-penetration acceptance model — the PROCESS-audit CP-shield slab
   (60 cm W / WC+H2O / W2B5 / W2B5+H2O, 14.06 MeV planar source, 60 x
   1 cm depth-cell flux profiles) — lives in
   `tools/metal-validation/cp_shield.py`.
2. Fixed-source with fissionable materials, MPI, photon transport, DAGMC,
   hex lattices, tori,
   periodic BCs, survival biasing, weight windows, multi-temperature
   models, temperature interpolation, resonance upscattering (DBRC/RVS),
   NCrystal, isotropic-in-lab (p0) scattering, neutron time cutoffs,
   mesh tally filters, distribcell — all detected and fall back to CPU
   with a warning.
3. Event-based device pipeline (history-based v1 leaves SIMD occupancy on
   the table for CE); unionized/material-major energy grids.
4. CUDA backend: the dialect and backend ABI are in place
   (`src/gpu/device/dialect.h`, `src/gpu/backend.h`); needs
   `backend_cuda.cu` implementing the same slots and a kernel wrapper, plus
   `-ffp-contract=off`/`--fmad=false` for bit parity with the portable
   math layer.

## 2026-08-21 cross-review fix wave

The branch was independently reviewed (external AI review tooling reading
this tree against the CPU sources); every claim was re-verified against
both code paths before acting. Fixes landed, all CPU-faithfulness or
correctness items:

* **Watt spectrum** (`gpu_watt_spectrum`): fluctuating term used
  `sqrt(a*b*W)` instead of `sqrt(a^2 b W)` — the spectrum width collapsed
  ~1000x for eV-scale `a`. Live wherever ACE laws use Watt.
* **Recursive AngleEnergy dispatch** replaced with an iterative redirect
  loop — root cause of the Godiva bias (see above).
* **Lattice exit** (`gpu_cross_lattice`): leaving a lattice now re-searches
  from the base coords like CPU `cross_lattice`, instead of losing the
  particle (no `outer`) or searching `outer` in tile-local coordinates.
* **`reconcile_cell_after_collision` ported**: a direction change during a
  near-surface collision re-validates every coordinate level; previously
  the stale cell chain could track a history through the wrong material
  across a curved surface.
* **Cell/universe tally filters** now score every matching coordinate
  level (cartesian product over filters, as CPU `FilterBinIter`), not the
  first hit. Filters per tally bounded at `GPU_MAX_TALLY_FILTERS` (4),
  rejected above.
* **Partial fission**: the reaction is picked once per collision before
  the site-count RN (CPU `sample_fission` order), not per site; delayed
  precursor yields/decay/spectra come from the *selected* reaction.
* **Complex-region distance**: coincident-surface skipping now applies
  only when actually on a surface (`on_surface != 0`), matching
  `Region::distance_complex`; union cells no longer skip genuinely-near
  first boundaries.
* **MG SCORE_SCATTER** scores scatter (nu-scatter / mean multiplicity),
  not the nu-scatter integral; MG fission/nu-fission scores drop
  `density_mult` to mirror the CPU `get_xs` asymmetry (upstream finding 5).
* **MG fission-site RN order** (mu, phi, then energy), **histogram-flat
  angle collapse removed** (kept for tabular, where the RN count matches),
  **MG void/zero inverse-velocity time advance** via
  `default_inverse_velocity`.
* **`rotate_angle` pole branch** now uses the CPU expansion (was a
  constant azimuth phase offset — statistically identical but it split
  paired trajectories); **white-BC grazing** uses `>=` like
  `Surface::diffuse_reflect`; **BCs on nested-universe surfaces** mark the
  particle lost (CPU behavior) instead of reflecting in the root frame.
* **Energy cutoff** implemented (kill after collision, CE);
  **`free_gas_threshold`** passed through instead of hardcoded 400 kT;
  unsupported-feature rejections added for DBRC/RVS, NCrystal, p0
  lab-isotropic scattering, temperature interpolation, finite neutron
  time cutoffs.
* **`gpu_sample_tabular`** keeps the CPU walk's exact stale-`c_k` /
  `c_k1` lifecycle (discrete+continuous ACE tables, correlated-law angle
  pick); duplicate fp32-cast energy-grid knots guarded (`r -> 0`).
* **Trace counter** got its own slot (`GPU_CTR_TRACE`) instead of
  aliasing `GPU_CTR_LOST_REFLECT`; fission-bank overflow now warns like
  CPU; the CPU-side `OPENMC_ISO_MU` ablation hook caches its `getenv`.

## 2026-08-21 second review (Codex) and the fp32 geometry hardening

A second independent review pass (Codex CLI, reading the post-fix tree)
confirmed the recursion root-cause section above and surfaced a further
set of verified findings, all fixed:

* **fp32 neutron speed** used `1-(m/(E+m))^2`, which cancels to exactly 0
  below ~32 eV (ulp of the neutron mass is 64 eV) — every thermal flight
  time was infinite. Now the CPU's `C*sqrt(E(E+2m))/(E+m)` form.
* **Differential tallies** were silently scored as ordinary tallies —
  rejected at flatten. Likewise rejected: surface-source writing,
  collision-track output, overlap checking, track output, N-body laws
  with n outside 3..5.
* **Lost-particle accounting** now feeds `simulation::n_lost_particles`
  and enforces the upstream dual-threshold abort per generation;
  secondary-stack overflow ((n,xn) clones) is fatal instead of a warning;
  below-cutoff clones are not created (CPU `create_secondary` parity).
* **Flatten hardening**: a fissionable nuclide without a usable nu
  function rejects; a sticky-error catch-all rejects any nuclide whose
  nested law recorded an unsupported construct; CE grid fp32 casts that
  collapse RUNS of knots are walked safely on device (no zero-division /
  OOB at the grid tail).
* **Complex-region crossing bound** is now the geometric maximum
  (2 crossings per quadric token) instead of a fixed 64.
* **RN parity**: the prompt/delayed RN is drawn even when a nuclide has
  no delayed groups (CPU draws it unconditionally).

The loss investigation this triggered ended with the fp32 geometry
hardening — the lost-particle rate on moderated/lattice models dropped
from ~5e-5 to 0–2.4e-6 with no measurable throughput cost:

1. **Lattice-exit recovery** searches the outer universe in the
   extrapolated tile frame (the frame the descent itself uses); root
   re-search — which must re-decide an on-boundary fp32 point and
   coin-flips — only when there is no outer universe. (A root-only
   "CPU-faithful" version lost ~1e-4 of MG-lattice histories.)
2. **Tile overshoot**: a negative tile-face distance (local point epsilon
   past the face with stale indices) is an already-happened crossing —
   taken at d = 0 and re-indexed, not marked lost.
3. **Reflection pins the root cell** exactly like CPU
   `cross_reflective_bc` (`coord(0).cell = cell_last(0)`), rebuilding
   only the lower universes; re-deciding root containment at the wall
   was the dominant pincell loss class.
4. **Post-collision reconciliation runs on every collision** (CPU gates
   it on a near-surface token; an fp32 flight can overshoot a surface by
   more than TINY_BIT with the token already cleared, and a stale chain
   then streams through walls — the infinite wall planes kept reflecting
   escapees outside the box, whose banked fission sites failed placement
   a generation later). The repair runs in place — a trial copy of the
   coordinate state cost enough thread stack to collapse occupancy
   (17x on the pincell) — with escalating bidirectional nudge rescue,
   and an unrepairable state is lost, as on the CPU.
5. **Directional on-surface plane logic**: the on-surface token
   suppresses a plane only in the moving-away direction; moving back
   produces a d = 0 crossing so BCs fire. At exactly f == 0 with no
   token (fp32 corner quantization), the cell's own region token sign
   disambiguates the side — without it, -0/u = +0 fabricated
   zero-distance crossings in both directions and corner reflections
   ping-ponged until the event cap (the pre-fix reflect re-search had
   been killing those states as losses, masking the loop).

Device-vs-replay bit identity was re-verified after the geometry
changes: 0 disagreements on 2M paired Godiva histories.

Review improvement backlog (not yet implemented): CE cross-section
caching across non-collision events, per-thread micro-XS working-set
reduction (occupancy), active-tally compaction + threadgroup-local tally
reduction, optional wavefront pipeline for collision-heavy CE, compiled
metallib caching keyed on source hash, a host-side arena validator, and
full-width RNG-state traces.

## Deep-penetration acceptance (2026-08-21, fixed-source mode)

The fixed-source envelope was accepted against the regime the reactor
suite does not probe: the PROCESS-audit CP-shield slab
(`tools/metal-validation/cp_shield.py`) — a 14.06 MeV isotropic planar
source into 0.60 m of shield, 60 x 1 cm depth-cell flux profiles
(total and E > 0.1 MeV), four materials spanning fast attenuations of
6x10^3 (W) to 1.5x10^5 (W2B5), analog transport, independent seeds.

Metal-CPU vs metal-GPU, 60 bins x 2 scores x 4 materials (GPU 4e8
histories/material, CPU 1e8):

| Material | total flux | E > 0.1 MeV flux |
|---|---|---|
| W | mean z² 1.46, max\|z\| 2.7 | 1.15 / 2.7 |
| WC+13%H2O | 0.92 / 2.4 | 0.96 / 2.6 |
| W2B5 | 0.60 / 1.6 | 0.73 / 1.8 |
| W2B5+13%H2O | 1.39 / 3.2 | 1.47 / 3.7 |

Every profile is statistics-consistent to the deepest bin; the
W2B5/WC+H2O fast-flux ratios (the quantity PROCESS's CP refit consumes)
agree between engines to 0.5% at 30 cm and ~1% at 55 cm. A same-model
version bridge (the audit's archived 0.15.3 model run byte-identical on
this tree's CPU) reproduces the 0.15.3 results to ratio 1.000 — no
version skew.

**fp32 finding the acceptance test caught: tally-accumulator
saturation.** Per-batch tally bins accumulate in fp32 device atomics;
once a bin's batch sum nears 2^24, sub-ulp track contributions round
away and the tally biases LOW — measured -1.3% on the slab's front bin
at 4x10^6 particles/batch (-0.07% at 1x10^6; invisible in the
eigenvalue suite's smaller per-bin sums). Fixed structurally, not by
capping batch sizes: tally accumulation now uses 64 replicated banks
(thread tid scores into bank tid % 64; the host sums banks in fp64),
which keeps per-bank sums ~64x below the hazard and reduces tally-atomic
contention. The failing configuration reads 1.0002 +/- 0.0002 after the
fix, and the eigenvalue suite is bit-unchanged.

Reference-data note: the audit's archived 0.15.3 depth profiles were run
with photon transport on and tallies that carry NO particle filter (the
current audit script adds one; the archived models predate it), so those
profiles include the secondary-gamma population — +20% at the face and
up to +40% deep. The clean neutron-only profiles from this acceptance
run supersede them for neutron-flux ratio work.

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
