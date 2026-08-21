"""GPU optimisation benchmark + acceptance gate: the CP-shield slab.

A fixed-source, deep-penetration, continuous-energy workload (60 cm slab of
W / WC+13%H2O / W2B5 / W2B5+13%H2O under a 14.06 MeV planar isotropic
source; 60 x 1 cm depth cells; E > 0.1 MeV and total neutron-flux cell
tallies; analog, neutron-only). It is the workload the PROCESS-audit
shielding work runs, and it is the regime where the engine's speedup is
smallest (4-5x over an 8-thread CPU on an M3 Ultra, vs 6-26x on the
reactor-shaped validation cases): histories are long and divergent
(~10^3 events for a 14 MeV neutron slowing down in tungsten), the thin
cells make boundary crossings a large share of the event stream, and the
material set spans short (boron capture) to long (pure-W) walks.

Every run is scored two ways:

1. PERFORMANCE: active calculation rate (histories/s), GPU device time,
   and speedup vs the frozen CPU 8-thread baseline.
2. CORRECTNESS GATE: the depth profiles are z-tested bin-by-bin against a
   frozen fp64 CPU reference (reference/cp_shield_cpu_fp64_1e8.json;
   same tree, same model, seed 7, 1e8 histories). PASS requires, per
   material and per tally: |mean z| < 1.0, RMS z < 1.5, max |z| < 4.5
   over the 60 bins, slope of ln(GPU/CPU) over 10-40 cm within 5 sigma OR
   below 0.03 %/cm in magnitude (0.9% over 30 cm), and zero lost
   particles / event-cap kills. (Adjacent depth bins are fed by the same
   histories, so the naive per-bin errors understate the spread of mean z
   and of the slope; the cuts are calibrated on the 2026-08-21 full-
   statistics passes, which reached |mean z| 0.56 and 2.8 sigma of slope.) Bit-identity with the CPU is
   NOT required (re-ordering work across lanes changes tally accumulation
   order); statistical identity IS. Run at a seed different from the
   reference's 7 so the z-test is independent.

Usage
-----
  python bench_cp_shield.py <cross_sections.xml> <out_dir> [options]
    --materials W,WC_H2O,W2B5,W2B5_H2O   (default: all four)
    --histories 4e7        total histories per material
    --batches 20           batches (histories/batches per batch)
    --seed 42
    --sweep 1e5,1e6,2e6,1e7  instead of --batches: run the same total
                           histories at each particles-per-batch value
                           (occupancy map; correctness gated at each)
    --cpu                  run the CPU engine instead (fresh baseline)
    --reference <json>     override the reference file
    --label <text>         tag stored in the results JSON (e.g. git sha)

Outputs: <out_dir>/bench_<label>.json (all metrics) and a markdown table
on stdout. Exit status 1 if any correctness gate fails.

Environment: the `openmc` executable of THIS tree on PATH and its Python
package importable (e.g. the repo's venv). OMP_NUM_THREADS=8 is a sensible
host-side setting (source sampling bursts; CPU tally scoring collapses
above ~8 threads on Apple Silicon).
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import openmc

T_SHIELD = 0.60  # m
N_BINS = 60

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
    "WC_H2O": lambda: homogenise("WC_H2O", [(WC_F, 15.32, 0.87), (H2O_F, 1.0, 0.13)], sab="c_H_in_H2O"),
    "W2B5": lambda: homogenise("W2B5", [(W2B5_F, 12.91, 1.0)]),
    "W2B5_H2O": lambda: homogenise("W2B5_H2O", [(W2B5_F, 12.91, 0.87), (H2O_F, 1.0, 0.13)], sab="c_H_in_H2O"),
}

GATES = {"mean_z": 1.0, "rms_z": 1.5, "max_z": 4.5, "slope_sigma": 5.0, "slope_pct_per_cm": 0.03}


def build_model(tag, particles, batches, seed):
    mat = MATERIALS[tag]()
    dx = T_SHIELD * 100 / N_BINS
    planes = [openmc.XPlane(i * dx) for i in range(N_BINS + 1)]
    x_front = openmc.XPlane(-5.0, boundary_type="vacuum")
    x_back = openmc.XPlane(T_SHIELD * 100 + 5.0, boundary_type="vacuum")
    src_cell = openmc.Cell(region=+x_front & -planes[0])
    depth = [openmc.Cell(fill=mat, region=+planes[i] & -planes[i + 1]) for i in range(N_BINS)]
    back = openmc.Cell(region=+planes[N_BINS] & -x_back)
    geom = openmc.Geometry([src_cell] + depth + [back])

    st = openmc.Settings()
    st.run_mode = "fixed source"
    st.batches = batches
    st.particles = particles
    st.survival_biasing = False
    st.photon_transport = False
    st.output = {"tallies": False}
    st.seed = seed
    src = openmc.IndependentSource()
    src.space = openmc.stats.Point((-0.01, 0.0, 0.0))
    src.angle = openmc.stats.Isotropic()
    src.energy = openmc.stats.Discrete([14.06e6], [1.0])
    st.source = src

    cf = openmc.CellFilter(depth)
    t_fast = openmc.Tally(name="fast_flux")
    t_fast.filters = [cf, openmc.EnergyFilter([1.0e5, 20.0e6])]
    t_fast.scores = ["flux"]
    t_tot = openmc.Tally(name="total_flux")
    t_tot.filters = [cf]
    t_tot.scores = ["flux"]
    return openmc.Model(geometry=geom, settings=st, tallies=openmc.Tallies([t_fast, t_tot]))


def run_case(model, case_dir, use_gpu):
    case_dir.mkdir(parents=True, exist_ok=True)
    model.export_to_model_xml(path=case_dir / "model.xml")
    env = dict(os.environ)
    env["OPENMC_GPU"] = "1" if use_gpu else "0"
    t0 = time.time()
    proc = subprocess.run(["openmc"], cwd=case_dir, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    out = proc.stdout + proc.stderr
    (case_dir / "run.log").write_text(out)
    if proc.returncode != 0:
        raise RuntimeError(f"openmc failed in {case_dir}:\n{out[-2000:]}")
    m = re.search(r"Calculation Rate \(active\)\s*=\s*([0-9.eE+]+)", out)
    rate = float(m.group(1)) if m else float("nan")
    m = re.search(r"GPU transport device time:\s*([0-9.]+)", out)
    dev = float(m.group(1)) if m else float("nan")
    gpu_active = "GPU transport engine active" in out
    fallback = re.search(r"GPU transport disabled[^\n]*", out)
    # GPU warnings say "transport lost N particles"; CPU losses print
    # per-particle messages; neither contains the phrase "lost particle"
    lost = sum(int(x) for x in re.findall(r"transport lost (\d+) particles", out))
    lost += len(re.findall(r"could not be located|Couldn't find particle|"
                           r"Maximum number of lost particles", out))
    capped = len(re.findall(r"(?i)event cap|max_events|maximum number of events", out))
    sp = case_dir / f"statepoint.{model.settings.batches}.h5"
    with openmc.StatePoint(sp) as s:
        prof = {}
        for name in ["fast_flux", "total_flux"]:
            t = s.get_tally(name=name)
            prof[name] = {"mean": t.mean.ravel().tolist(), "std_dev": t.std_dev.ravel().tolist()}
    return {"rate": rate, "device_s": dev, "wall_s": wall, "gpu_active": gpu_active,
            "fallback": fallback.group(0) if fallback else None,
            "lost_warnings": lost, "event_cap_warnings": capped, "profiles": prof}


def gate(prof, ref, x):
    """z statistics of prof vs ref; returns dict with pass flag."""
    res = {}
    ok = True
    for key in ["fast_flux", "total_flux"]:
        gm, gs = prof[key]["mean"], prof[key]["std_dev"]
        cm, cs = ref[key]["mean"], ref[key]["std_dev"]
        zs, pts = [], []
        for i in range(len(x)):
            den = math.sqrt(gs[i] ** 2 + cs[i] ** 2)
            if cm[i] > 0 and gm[i] > 0 and den > 0:
                zs.append((gm[i] - cm[i]) / den)
                if 10 <= x[i] <= 40:
                    r = gm[i] / cm[i]
                    pts.append((x[i], math.log(r), den / cm[i]))
        mean_z = sum(zs) / len(zs)
        rms_z = math.sqrt(sum(z * z for z in zs) / len(zs))
        max_z = max(abs(z) for z in zs)
        w = [1 / e ** 2 for _, _, e in pts]
        sw = sum(w)
        xb = sum(wi * p[0] for wi, p in zip(w, pts)) / sw
        yb = sum(wi * p[1] for wi, p in zip(w, pts)) / sw
        sxx = sum(wi * (p[0] - xb) ** 2 for wi, p in zip(w, pts))
        b = sum(wi * (p[0] - xb) * (p[1] - yb) for wi, p in zip(w, pts)) / sxx
        eb = 1 / math.sqrt(sxx)
        stat = {"mean_z": mean_z, "rms_z": rms_z, "max_z": max_z,
                "slope_pct_per_cm": 100 * b, "slope_sigma": b / eb, "n_bins": len(zs)}
        stat["pass"] = (abs(mean_z) < GATES["mean_z"] and rms_z < GATES["rms_z"]
                        and max_z < GATES["max_z"]
                        and (abs(b / eb) < GATES["slope_sigma"]
                             or abs(100 * b) < GATES["slope_pct_per_cm"]))
        ok = ok and stat["pass"]
        res[key] = stat
    res["pass"] = ok
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xs")
    ap.add_argument("out")
    ap.add_argument("--materials", default="W,WC_H2O,W2B5,W2B5_H2O")
    ap.add_argument("--histories", type=float, default=4e7)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--reference", default=str(Path(__file__).parent / "reference" / "cp_shield_cpu_fp64_1e8.json"))
    ap.add_argument("--label", default="run")
    a = ap.parse_args()

    openmc.config["cross_sections"] = a.xs
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = json.load(open(a.reference))
    cpu_base = ref["_meta"]["cpu_rate_8threads_particles_per_s"]
    gpu_prev = ref["_meta"].get("gpu_rate_2026_08_21_particles_per_s", {})
    use_gpu = not a.cpu
    total = int(a.histories)
    configs = ([("ppb=%g" % p, int(p), max(1, total // int(p))) for p in map(float, a.sweep.split(","))]
               if a.sweep else [("batches=%d" % a.batches, total // a.batches, a.batches)])

    results = {"label": a.label, "engine": "GPU" if use_gpu else "CPU", "seed": a.seed,
               "histories": total, "cases": []}
    rows = []
    any_fail = False
    for tag in a.materials.split(","):
        for cfg_name, particles, batches in configs:
            model = build_model(tag, particles, batches, a.seed)
            case_dir = out / f"{tag}_{cfg_name.replace('=', '')}"
            r = run_case(model, case_dir, use_gpu)
            g = gate(r["profiles"], ref[tag], ref[tag]["x_cm"])
            clean = r["lost_warnings"] == 0 and r["event_cap_warnings"] == 0 and (r["fallback"] is None)
            passed = g["pass"] and clean and (r["gpu_active"] or not use_gpu)
            any_fail = any_fail or not passed
            speed = r["rate"] / cpu_base[tag] if cpu_base.get(tag) else float("nan")
            vs_prev = r["rate"] / gpu_prev[tag] if gpu_prev.get(tag) else float("nan")
            rec = {"material": tag, "config": cfg_name, "particles_per_batch": particles, "batches": batches,
                   **{k: v for k, v in r.items() if k != "profiles"}, "gate": g, "speedup_vs_cpu8": speed,
                   "vs_gpu_2026_08_21": vs_prev, "pass": passed}
            results["cases"].append(rec)
            rows.append((tag, cfg_name, r["rate"] / 1e6, r["device_s"], speed, vs_prev,
                         g["fast_flux"]["mean_z"], g["fast_flux"]["rms_z"], g["fast_flux"]["max_z"],
                         g["fast_flux"]["slope_sigma"], "PASS" if passed else "FAIL",
                         "" if r["fallback"] is None else "FALLBACK"))
            print(f"[{tag} {cfg_name}] {r['rate']/1e6:.3f} M/s  dev {r['device_s']:.1f}s  "
                  f"x{speed:.2f} vs CPU8  x{vs_prev:.2f} vs GPU-baseline  "
                  f"gate {'PASS' if passed else 'FAIL'} (mean z {g['fast_flux']['mean_z']:+.2f}, "
                  f"rms {g['fast_flux']['rms_z']:.2f}, max {g['fast_flux']['max_z']:.1f}, "
                  f"slope {g['fast_flux']['slope_sigma']:+.1f} sigma; lost {r['lost_warnings']}, "
                  f"cap {r['event_cap_warnings']})", flush=True)

    (out / f"bench_{a.label}.json").write_text(json.dumps(results, indent=1))
    print("\n| material | config | M hist/s | device s | x vs CPU8 | x vs GPU-base | mean z | rms z | max|z| | slope sigma | gate |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in rows:
        print(f"| {t[0]} | {t[1]} | {t[2]:.3f} | {t[3]:.1f} | {t[4]:.2f} | {t[5]:.2f} | {t[6]:+.2f} | {t[7]:.2f} | {t[8]:.1f} | {t[9]:+.1f} | {t[10]}{(' ' + t[11]) if t[11] else ''} |")
    print(f"\nresults -> {out / f'bench_{a.label}.json'}")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
