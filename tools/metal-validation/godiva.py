"""Godiva-like bare HEU sphere (HEU-MET-FAST-001 style) — fast spectrum,
no S(a,b), no URR self-shielding importance. Differential CPU-vs-GPU reference."""
import argparse
import openmc

p = argparse.ArgumentParser()
p.add_argument("--particles", type=int, default=10000)
p.add_argument("--batches", type=int, default=120)
p.add_argument("--inactive", type=int, default=20)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--tallies", action="store_true")
args = p.parse_args()

heu = openmc.Material(name="HEU")
heu.set_density("g/cm3", 18.74)
heu.add_nuclide("U235", 0.9371, "wo")
heu.add_nuclide("U238", 0.0527, "wo")
heu.add_nuclide("U234", 0.0102, "wo")

mats = openmc.Materials([heu])

sph = openmc.Sphere(r=8.7407, boundary_type="vacuum")
cell = openmc.Cell(fill=heu, region=-sph)
geom = openmc.Geometry([cell])

settings = openmc.Settings()
settings.run_mode = "eigenvalue"
settings.particles = args.particles
settings.batches = args.batches
settings.inactive = args.inactive
settings.seed = args.seed
settings.source = openmc.IndependentSource(
    space=openmc.stats.Point((0, 0, 0)),
    energy=openmc.stats.Watt(0.988e6, 2.249e-6),
)

model = openmc.Model(geometry=geom, materials=mats, settings=settings)

if args.tallies:
    t_flux = openmc.Tally(name="flux-spectrum")
    ef = openmc.EnergyFilter(
        [0.0, 1e3, 1e4, 1e5, 5e5, 1e6, 2e6, 4e6, 6e6, 10e6, 20e6])
    t_flux.filters = [ef, openmc.CellFilter([cell])]
    t_flux.scores = ["flux", "fission", "nu-fission", "absorption", "elastic"]
    t_rate = openmc.Tally(name="rates")
    t_rate.filters = [openmc.CellFilter([cell])]
    t_rate.scores = ["flux", "total", "fission", "nu-fission", "absorption",
                     "elastic", "scatter"]
    model.tallies = openmc.Tallies([t_flux, t_rate])

model.export_to_model_xml()
