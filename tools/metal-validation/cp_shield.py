"""Deep-penetration acceptance test: ST centre-post shield slab.

GPU-envelope adaptation of the PROCESS-audit CP-shield model
(audit/repro/scripts/openmc_cp_shield.py): a planar 14.06 MeV isotropic
point source impinging on a 0.60 m shield slab, reflective lateral
boundaries (infinite slab), vacuum beyond. Depth profiles of total and
E > 0.1 MeV neutron flux over 60 x 1 cm bins for four shield materials
(W, WC+13%H2O, W2B5, W2B5+13%H2O; densities per Windsor et al. 2021).

Differences from the audit script, all required by the GPU envelope and
neutron-ratio-neutral:
  - 60 explicit 1 cm depth cells + CellFilter instead of a mesh tally
    (tracklength mesh track-splitting is not ported; for a 1-D slab the
    cell binning is geometrically identical)
  - neutron-only: no photon transport, no heating score, no ParticleFilter
    (all particles are neutrons)
  - analog transport (survival_biasing off): the GPU brute-forces deep
    attenuation with raw histories; run the CPU comparison leg with the
    same settings so the comparison is estimator-identical

This probes the regime the reactor-shaped validation suite does not:
~1e7 flux attenuation over 55+ cm, where fp32 errors would compound
exponentially. Acceptance = GPU reproduces the same-tree CPU depth
profiles bin-by-bin within statistics.

Usage:
  python cp_shield.py <xs_xml> <out_dir> [particles] [batches] [materials] [seed]
  (run with OPENMC_GPU=1 for the GPU leg; the model is identical. Use
  different seeds for the two legs — with the same seed the host-sampled
  fp64 source states are identical and the depth profiles are correlated,
  which defeats the independent z-test.)
"""

import json
import sys
from pathlib import Path

import numpy as np
import openmc

XS_XML = sys.argv[1]
OUT = Path(sys.argv[2])
PARTICLES = int(float(sys.argv[3])) if len(sys.argv) > 3 else int(2e7)
BATCHES = int(sys.argv[4]) if len(sys.argv) > 4 else 50

openmc.config["cross_sections"] = XS_XML

T_SHIELD = 0.60  # m
N_BINS = 60  # 1 cm resolution

ATOMIC_MASS = {"W": 183.84, "C": 12.011, "B": 10.811, "H": 1.008, "O": 15.999}


def _wt_fractions(formula):
    total = sum(ATOMIC_MASS[el] * n for el, n in formula.items())
    return {el: ATOMIC_MASS[el] * n / total for el, n in formula.items()}


def homogenise(name, components, *, sab=None):
    rho = sum(d * v for _, d, v in components)
    m = openmc.Material(name=name)
    elements = {}
    for formula, d, v in components:
        mass = d * v / rho
        for el, w in _wt_fractions(formula).items():
            elements[el] = elements.get(el, 0.0) + mass * w
    for el, w in elements.items():
        m.add_element(el, w, "wo")
    m.set_density("g/cm3", rho)
    if sab:
        m.add_s_alpha_beta(sab)
    return m


WC_F = {"W": 1.0, "C": 1.0}
W2B5_F = {"W": 2.0, "B": 5.0}
H2O_F = {"H": 2.0, "O": 1.0}

MATERIALS = {
    "W": lambda: homogenise("W", [({"W": 1.0}, 18.91, 1.0)]),
    "WC_H2O": lambda: homogenise(
        "WC_H2O", [(WC_F, 15.32, 0.87), (H2O_F, 1.0, 0.13)], sab="c_H_in_H2O"
    ),
    "W2B5": lambda: homogenise("W2B5", [(W2B5_F, 12.91, 1.0)]),
    "W2B5_H2O": lambda: homogenise(
        "W2B5_H2O", [(W2B5_F, 12.91, 0.87), (H2O_F, 1.0, 0.13)], sab="c_H_in_H2O"
    ),
}


def run_case(tag, make_mat, case_dir):
    case_dir.mkdir(parents=True, exist_ok=True)
    mat = make_mat()
    mat.name = tag

    dx = T_SHIELD * 100 / N_BINS
    planes = [openmc.XPlane(i * dx) for i in range(N_BINS + 1)]
    x_front = openmc.XPlane(-5.0, boundary_type="vacuum")
    x_back = openmc.XPlane(T_SHIELD * 100 + 5.0, boundary_type="vacuum")

    src_cell = openmc.Cell(region=+x_front & -planes[0])
    depth_cells = [
        openmc.Cell(fill=mat, region=+planes[i] & -planes[i + 1])
        for i in range(N_BINS)
    ]
    back_cell = openmc.Cell(region=+planes[N_BINS] & -x_back)
    geom = openmc.Geometry([src_cell] + depth_cells + [back_cell])

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = BATCHES
    settings.particles = PARTICLES // BATCHES
    settings.survival_biasing = False  # analog on both engines
    settings.photon_transport = False  # neutron-only legs
    settings.output = {"tallies": False}
    if len(sys.argv) > 6:
        settings.seed = int(sys.argv[6])
    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((-0.01, 0.0, 0.0))
    src.angle = openmc.stats.Isotropic()
    src.energy = openmc.stats.Discrete([14.06e6], [1.0])
    settings.source = src

    cf = openmc.CellFilter(depth_cells)

    t_fast = openmc.Tally(name="fast_flux")
    t_fast.filters = [cf, openmc.EnergyFilter([1.0e5, 20.0e6])]
    t_fast.scores = ["flux"]

    t_tot = openmc.Tally(name="total_flux")
    t_tot.filters = [cf]
    t_tot.scores = ["flux"]

    model = openmc.Model(
        geometry=geom,
        settings=settings,
        tallies=openmc.Tallies([t_fast, t_tot]),
    )
    sp_path = model.run(cwd=case_dir)

    with openmc.StatePoint(sp_path) as sp:
        out = {}
        for name in ["fast_flux", "total_flux"]:
            t = sp.get_tally(name=name)
            out[name] = {
                "mean": t.mean.ravel().tolist(),
                "std_dev": t.std_dev.ravel().tolist(),
            }
    out["x_cm"] = list(np.linspace(0.5, T_SHIELD * 100 - 0.5, N_BINS))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = sys.argv[5].split(",") if len(sys.argv) > 5 else list(MATERIALS)
    results = {}
    if (OUT / "results.json").exists():
        results = json.loads((OUT / "results.json").read_text())
    for tag, maker in MATERIALS.items():
        if tag not in selected or tag in results:
            continue
        print(f"=== running {tag}", flush=True)
        results[tag] = run_case(tag, maker, OUT / tag)
        (OUT / "results.json").write_text(json.dumps(results))
    print("all cases done ->", OUT / "results.json")


if __name__ == "__main__":
    main()
