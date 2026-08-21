# GPU optimisation target: the CP-shield deep-penetration benchmark

`bench_cp_shield.py` is a performance benchmark with a built-in correctness
gate. It exists because the Metal engine's speedup on *this* workload is the
smallest we have measured — **4.3–5.4× over an 8-thread CPU** (M3 Ultra,
2026-08-21) against 6–26× on the reactor-shaped validation cases — and
because this is the workload the PROCESS shielding work actually runs.

## The workload

Fixed-source, continuous-energy, analog, neutron-only. 60 cm slab under a
14.06 MeV planar isotropic source; 60 explicit 1 cm cells; cell-filtered
E > 0.1 MeV and total flux (tracklength). Four materials spanning history
length: pure W (~10³ elastic events per slowing-down history — long, highly
divergent walks), WC+13% H₂O and W₂B₅+13% H₂O (S(α,β) thermalisation),
monolithic W₂B₅ (short — ¹⁰B capture). Attenuation ≈ 10⁷ over 55 cm, so
deep bins are populated by the rare long survivors.

## Baselines (2026-08-21, 4×10⁸ histories/material, seed 42, 1×10⁷ per batch)

| material | CPU fp64, 8 threads | GPU | speedup |
|---|---|---|---|
| W | 0.276 M/s | 1.28 M/s | 4.7× |
| WC+H₂O | 0.325 M/s | 1.61 M/s | 4.9× |
| W₂B₅ | 0.477 M/s | 2.58 M/s | 5.4× |
| W₂B₅+H₂O | 0.594 M/s | 2.53 M/s | 4.3× |

The CPU baseline is already at its best thread count (tally scoring collapses
above ~8 threads on this machine), so the speedups are honest.

## What to optimise (hypotheses, in the order I'd test them)

1. **History-length divergence.** One thread per history: lanes whose
   neutron is captured early idle until the longest walk in the group ends.
   Pure W is the worst case and the slowest material per history. The
   event-based / wavefront pipeline on the roadmap is the structural fix;
   a cheaper first step is measuring events-per-history and lane occupancy
   per material to size the prize.
2. **Boundary crossings dominate the event stream.** 1 cm cells in a
   medium whose mean free path is ~0.7–3 cm mean most "events" are surface
   crossings, not collisions. Porting tracklength mesh tallies (track
   splitting) would let the geometry collapse to 3 cells and move the
   binning into the tally — that removes most crossings outright.
3. **Cross-section lookups.** Dense W resonance grids; cache the material
   XS across non-collision events (on the backlog) and consider
   material-major / unionised energy grids.
4. **Tally atomics.** 120 bins × many threads; threadgroup-local reduction
   before the global atomics (on the backlog).

## The rules

- Run the gate at every step: `--seed 42` (the reference is seed 7), all
  four materials, default 4×10⁷ histories (~30–60 s per material now).
  PASS = per tally |mean z| < 1.0, RMS z < 1.5, max |z| < 4.5 over 60
  bins, ln-ratio slope over 10–40 cm within 5σ or below 0.03 %/cm, zero
  lost particles, zero event-cap kills, engine active (no CPU fallback).
  The cuts are calibrated on the 2026-08-21 full-statistics passes
  (adjacent bins share histories, so naive errors understate the spread
  of mean z and slope). An optimisation that fails the gate is not an
  optimisation; a gate failure on an unchanged engine means the
  reference or the cuts need revisiting, not the engine.
- Bit-identity with the CPU is **not** required (lane re-ordering changes
  accumulation order); statistical identity **is**. Keep RN-stream parity
  per history where the design allows — it is what makes paired-history
  debugging possible.
- Report `x vs GPU-base` (rate relative to the 2026-08-21 GPU numbers) and
  `x vs CPU8`; keep the `bench_<label>.json` files so the trend is visible.
- Use `--sweep 1e5,1e6,2e6,1e7` to map occupancy vs particles-per-batch
  before and after a change; the engine is under-occupied below ~10⁵ in
  flight and this workload may want more.

## Running

```bash
PATH=build/bin:$PATH OMP_NUM_THREADS=8 python tools/metal-validation/bench_cp_shield.py \
  /path/to/endfb-viii.0-hdf5/cross_sections.xml runs/bench --label <git-sha>
```

Reference profiles: `reference/cp_shield_cpu_fp64_1e8.json` (CPU fp64,
seed 7, 10⁸ histories/material, this tree at 82f42585b). Regenerate with
`--cpu --histories 1e8 --batches 40 --seed 7` if the CPU physics changes.
