"""Classic PWR pincell (UO2 / Zr clad / borated light water, reflective x-y).
Exercises thermal physics incl. S(a,b) for hydrogen in water and URR ptables."""
import argparse
import openmc

p = argparse.ArgumentParser()
p.add_argument("--particles", type=int, default=10000)
p.add_argument("--batches", type=int, default=120)
p.add_argument("--inactive", type=int, default=20)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--no-sab", action="store_true", help="omit S(a,b) tables")
p.add_argument("--tallies", action="store_true")
args = p.parse_args()

uo2 = openmc.Material(name="UO2 (3.1%)")
uo2.set_density("g/cm3", 10.29769)
uo2.add_nuclide("U234", 5.7987e-06)
uo2.add_nuclide("U235", 7.2175e-04)
uo2.add_nuclide("U238", 2.2253e-02)
uo2.add_nuclide("O16", 4.5940e-02)

zirc = openmc.Material(name="Zircaloy")
zirc.set_density("g/cm3", 6.55)
zirc.add_nuclide("Zr90", 2.1827e-02)
zirc.add_nuclide("Zr91", 4.7600e-03)
zirc.add_nuclide("Zr92", 7.2758e-03)
zirc.add_nuclide("Zr94", 7.3734e-03)
zirc.add_nuclide("Zr96", 1.1879e-03)

water = openmc.Material(name="Borated water")
water.set_density("g/cm3", 0.740582)
water.add_nuclide("H1", 4.9457e-02)
water.add_nuclide("O16", 2.4672e-02)
water.add_nuclide("B10", 8.0042e-06)
water.add_nuclide("B11", 3.2218e-05)
if not args.no_sab:
    water.add_s_alpha_beta("c_H_in_H2O")

mats = openmc.Materials([uo2, zirc, water])

fuel_or = openmc.ZCylinder(r=0.39218)
clad_ir = openmc.ZCylinder(r=0.40005)
clad_or = openmc.ZCylinder(r=0.45720)
pitch = 1.25984
box = openmc.model.RectangularPrism(pitch, pitch, boundary_type="reflective")

fuel = openmc.Cell(name="fuel", fill=uo2, region=-fuel_or)
gap = openmc.Cell(name="gap", region=+fuel_or & -clad_ir)
clad = openmc.Cell(name="clad", fill=zirc, region=+clad_ir & -clad_or)
mod = openmc.Cell(name="moderator", fill=water, region=+clad_or & -box)
geom = openmc.Geometry([fuel, gap, clad, mod])

settings = openmc.Settings()
settings.run_mode = "eigenvalue"
settings.particles = args.particles
settings.batches = args.batches
settings.inactive = args.inactive
settings.seed = args.seed
settings.source = openmc.IndependentSource(
    space=openmc.stats.Box((-0.4, -0.4, -1), (0.4, 0.4, 1)),
    constraints={"fissionable": True},
)

model = openmc.Model(geometry=geom, materials=mats, settings=settings)

if args.tallies:
    ef = openmc.EnergyFilter([0.0, 0.625, 1e2, 1e4, 1e6, 20e6])
    t1 = openmc.Tally(name="cell-rates")
    t1.filters = [openmc.CellFilter([fuel, mod]), ef]
    t1.scores = ["flux", "fission", "nu-fission", "absorption", "elastic"]
    t2 = openmc.Tally(name="mat-rates")
    t2.filters = [openmc.MaterialFilter([uo2, water])]
    t2.scores = ["flux", "total", "absorption", "fission"]
    model.tallies = openmc.Tallies([t1, t2])

model.export_to_model_xml()
