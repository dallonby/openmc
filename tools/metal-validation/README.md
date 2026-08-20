# Metal GPU validation cases

Reference problems and tooling used to validate the GPU transport engine
(see ../../README_METAL.md and ../../PORT_NOTES.md):

- `godiva.py` — bare HEU sphere (fast spectrum; the leakage-stress case)
- `pincell.py` — PWR pincell, continuous energy; `--no-sab` omits thermal
  scattering tables
- `mg_pincell.py` — 7-group C5G7-style 3x3 pin lattice (multigroup)
- `compare_tallies.py` — statistical statepoint comparison (per-bin z
  scores against combined uncertainties)
- `bench.sh` — the CPU-vs-GPU calculation-rate sweep (edit the paths at
  the top for your machine)

Each model script writes `model.xml` into the working directory:

    python godiva.py --particles 20000 --batches 170 --inactive 20 --tallies
    openmc                     # CPU
    OPENMC_GPU=1 openmc        # GPU
    python compare_tallies.py cpu/statepoint.170.h5 gpu/statepoint.170.h5
