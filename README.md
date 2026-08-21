# OpenMC on Apple Silicon GPUs

**The first port of the [OpenMC](https://github.com/openmc-dev/openmc)
Monte Carlo particle transport code to Apple GPUs.** Real continuous-energy
nuclear data, the real OpenMC code base, one line to enable — and
**3–21× the throughput of the same machine's best-tuned CPU run**
(honest per-workload numbers below).

```python
import openmc

model = openmc.examples.pwr_pin_cell()   # any normal OpenMC model
model.settings.gpu = True                # <-- the only change
model.run()
```

No new input format, no separate toolchain, no offline shader compiler.
The same `openmc` executable, Python API, XML inputs, and statepoint files
— transport runs on the GPU when the model is inside the engine's
envelope, and falls back to the CPU with a clear warning when it is not.

## Performance

Measured on an M3 Ultra (80-core GPU), each row a single model run on
both engines. CPU baselines use the best thread count over a sweep, on
the current build (with the many-core scaling fix described in
[PORT_NOTES.md](PORT_NOTES.md) — earlier tables here quoted CPU baselines
crippled by an atomic-contention bug and overstated the GPU; those are
corrected below):

| Case | GPU | CPU best | Speedup |
|---|---|---|---|
| 7-group MG pin lattice (70 tally bins) | **3.25M histories/s** | 0.156M/s (8t) | **21×** |
| CE Godiva (bare sphere) | **14.9M histories/s** | 4.69M/s (16t) | **3.2×** |
| CE tungsten deep-penetration slab | **1.7–2.5M histories/s** | 0.41M/s (16t) | **4–6×** |

Read these honestly: the biggest multiples land where the CPU is throttled
by atomic contention the GPU avoids (MG still collapses above 8 threads on
per-bin tally atomics — a separate, unfixed CPU bottleneck), while on
Godiva, where that contention was fixed, the honest compute-vs-compute
ratio is ~3×. The GPU wins on every workload; how much depends on how
contended the CPU comparison is. The GPU also keeps scaling with batch
size (400M Godiva histories in 25 s) and has headroom left — an
event-based pipeline for long divergent histories, and threadgroup-local
tally tiles, are the next levers.

## Accuracy

Speed is easy if you don't check the answer. This port checks the answer:

- **k-effective agrees with fp64 CPU OpenMC to −3.8 ± 4.7 pcm** on a
  400M-history Godiva run — any residual fp32-vs-fp64 bias is bounded to
  **[−13, +6] pcm at 95% confidence**, an order of magnitude below
  nuclear-data uncertainty. Both engines also land on the ICSBEP
  experimental value.
- Every validation case (MG lattice, CE pincell with and without S(α,β),
  Godiva) agrees with the CPU reference **within 1σ**, tally suites
  included.
- The Metal engine reproduces the host-compiled build of the same source
  **bit-for-bit** over 2 million paired histories — the device computes
  exactly what the code says.
- Random-number streams are **bit-exact** with upstream's PCG generator,
  including skip-ahead seeding and URR stream discipline, so fission
  banks are reproducible, not just statistically equivalent.
- Zero lost particles on the MG lattice and Godiva suites; upstream's
  standard lost-particle accounting and abort thresholds are enforced.

The engineering behind those numbers is documented in
[PORT_NOTES.md](PORT_NOTES.md) — including the war story of a −630 pcm
bias root-caused by ablation to the Metal compiler silently miscompiling
(formally illegal) device recursion, and the fp32 geometry redesign that
took lost-particle rates from ~5×10⁻⁵ to zero on the lattice suites.
Two independent AI code reviews were run against the CPU sources and
every verified finding fixed; the audit trail is in the same file.

## Built to extend: CUDA-ready by construction

The device code is a **single source** written in a small portable
dialect ([src/gpu/device/](src/gpu/device/)) that compiles three ways
today:

1. **Metal** — MSL generated and compiled at runtime; no `xcrun metal`,
   no offline toolchain, works from a stock Xcode CLT install.
2. **Host C++** — a bit-identical "replay" engine used for validation and
   forensics (the reason the compiler bug above was findable at all).
3. **CUDA-ready** — the dialect shims (`GLOBAL`/`THREAD`, atomics,
   typedefs) and the backend ABI ([src/gpu/backend.h](src/gpu/backend.h))
   are the complete integration surface: an NVIDIA port is one
   `backend_cuda.cu` implementing the same buffer slots, not a rewrite.

Determinism is part of the design: transcendentals are bit-portable
polynomial kernels (no vendor libm, no FMA contraction), so Metal, host,
and future CUDA builds compute identical bits — vendor ulp differences
were measured to bias Monte Carlo ensembles at 11σ before this layer
existed.

**Physics envelope**: full continuous-energy neutron transport
(ENDF/B-VIII.0 pointwise data, URR probability tables, S(α,β) thermal
scattering, partial fission with delayed neutrons, all secondary angle–
energy laws, free-gas scattering, (n,xn)) plus multigroup mode; CSG
geometry with lattices, reflective/white/vacuum BCs; tracklength and
collision tallies. Anything outside the envelope is detected at startup
and runs on the CPU unchanged. Details: [README_METAL.md](README_METAL.md).

## Documentation

| | |
|---|---|
| [README_METAL.md](README_METAL.md) | Build, usage, envelope, validation, benchmarks |
| [PORT_NOTES.md](PORT_NOTES.md) | Engineering notes: fp32 design decisions with evidence, upstream findings, review audit trail |
| [tools/metal-validation/](tools/metal-validation/) | The validation models and comparison tooling |
| [openmc-dev/openmc#4067](https://github.com/openmc-dev/openmc/pull/4067) | First upstream bugfix contributed from this work |

This repository tracks upstream OpenMC (`develop`); the fork branch for
upstream PRs lives at
[dallonby/openmc](https://github.com/dallonby/openmc). Everything below
is the upstream OpenMC README, unchanged.

---

# OpenMC Monte Carlo Particle Transport Code

[![License](https://img.shields.io/badge/license-MIT-green)](https://docs.openmc.org/en/latest/license.html)
[![GitHub Actions build status (Linux)](https://github.com/openmc-dev/openmc/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/openmc-dev/openmc/actions/workflows/ci.yml)
[![Code Coverage](https://coveralls.io/repos/github/openmc-dev/openmc/badge.svg?branch=develop)](https://coveralls.io/github/openmc-dev/openmc?branch=develop)
[![dockerhub-publish-develop-dagmc](https://github.com/openmc-dev/openmc/workflows/dockerhub-publish-develop-dagmc/badge.svg)](https://github.com/openmc-dev/openmc/actions?query=workflow%3Adockerhub-publish-develop-dagmc)
[![dockerhub-publish-develop](https://github.com/openmc-dev/openmc/workflows/dockerhub-publish-develop/badge.svg)](https://github.com/openmc-dev/openmc/actions?query=workflow%3Adockerhub-publish-develop)
[![conda-pacakge](https://anaconda.org/conda-forge/openmc/badges/version.svg)](https://anaconda.org/conda-forge/openmc)

The OpenMC project aims to provide a fully-featured Monte Carlo particle
transport code based on modern methods. It is a constructive solid geometry,
continuous-energy transport code that uses HDF5 format cross sections. The
project started under the Computational Reactor Physics Group at MIT.

Complete documentation on the usage of OpenMC is hosted on Read the Docs (both
for the [latest release](https://docs.openmc.org/en/stable/) and
[developmental](https://docs.openmc.org/en/latest/) version). If you are
interested in the project, or would like to help and contribute, please get in
touch on the OpenMC [discussion forum](https://openmc.discourse.group/).

## Installation

Detailed [installation
instructions](https://docs.openmc.org/en/stable/usersguide/install.html)
can be found in the User's Guide.

## Citing

If you use OpenMC in your research, please consider giving proper attribution by
citing the following publication:

- Paul K. Romano, Nicholas E. Horelik, Bryan R. Herman, Adam G. Nelson, Benoit
  Forget, and Kord Smith, "[OpenMC: A State-of-the-Art Monte Carlo Code for
  Research and Development](https://doi.org/10.1016/j.anucene.2014.07.048),"
  *Ann. Nucl. Energy*, **82**, 90--97 (2015).

## Troubleshooting

If you run into problems compiling, installing, or running OpenMC, first check
the [Troubleshooting
section](https://docs.openmc.org/en/stable/usersguide/troubleshoot.html) in the
User's Guide. If you are not able to find a solution to your problem there,
please post to the [discussion forum](https://openmc.discourse.group/).

## Reporting Bugs

OpenMC is hosted on GitHub and all bugs are reported and tracked through the
[Issues](https://github.com/openmc-dev/openmc/issues) feature on GitHub.
However, GitHub Issues should not be used for common troubleshooting purposes.
If you are having trouble installing the code or getting your model to run
properly, you should first send a message to the [discussion
forum](https://openmc.discourse.group/). If it turns out your issue really is a
bug in the code, an issue will then be created on GitHub. If you want to request
that a feature be added to the code, you may create an Issue on github.

## License

OpenMC is distributed under the MIT/X
[license](https://docs.openmc.org/en/stable/license.html).
