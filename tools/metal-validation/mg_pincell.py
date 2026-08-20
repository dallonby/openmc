"""7-group multigroup pincell-lattice problem (3x3 rectangular lattice) —
exercises the MG engine, CSG universes/lattices, no nuclear data files needed.
Cross sections loosely based on C5G7 UO2 / moderator values."""
import argparse
import numpy as np
import openmc

p = argparse.ArgumentParser()
p.add_argument("--particles", type=int, default=10000)
p.add_argument("--batches", type=int, default=120)
p.add_argument("--inactive", type=int, default=20)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--tallies", action="store_true")
args = p.parse_args()

groups = openmc.mgxs.EnergyGroups(
    [0.0, 0.0635, 10.0, 1.0e2, 1.0e3, 0.5e6, 1.0e6, 20.0e6])

# --- C5G7 UO2 fuel data (7 groups) ---
uo2_data = openmc.XSdata("uo2", groups)
uo2_data.order = 0
uo2_data.set_total(
    [0.177949, 0.329805, 0.480388, 0.554367, 0.311801, 0.395168, 0.564406])
uo2_data.set_absorption(
    [8.0248e-03, 3.7174e-03, 2.6769e-02, 9.6236e-02, 3.0020e-02, 1.1126e-01,
     2.8278e-01])
uo2_data.set_fission(
    [7.21206e-03, 8.19301e-04, 6.45320e-03, 1.85648e-02, 1.78084e-02,
     8.30348e-02, 2.16004e-01])
uo2_data.set_nu_fission(
    (np.array([7.21206e-03, 8.19301e-04, 6.45320e-03, 1.85648e-02, 1.78084e-02,
               8.30348e-02, 2.16004e-01])
     * np.array([2.78145, 2.47443, 2.43383, 2.43380, 2.43380, 2.43380,
                 2.43380])))
uo2_data.set_chi(
    [5.87910e-01, 4.11760e-01, 3.39060e-04, 1.17610e-07, 0.0, 0.0, 0.0])
scat = np.array(
    [[[1.27537e-01, 4.23780e-02, 9.43740e-06, 5.51630e-09, 0.0, 0.0, 0.0],
      [0.0, 3.24456e-01, 1.63140e-03, 3.14270e-09, 0.0, 0.0, 0.0],
      [0.0, 0.0, 4.50940e-01, 2.67920e-03, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 4.52565e-01, 5.56640e-03, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.25250e-04, 2.71401e-01, 1.02550e-02, 1.00210e-08],
      [0.0, 0.0, 0.0, 0.0, 1.29680e-03, 2.65802e-01, 1.68090e-02],
      [0.0, 0.0, 0.0, 0.0, 0.0, 8.54580e-03, 2.73080e-01]]])
uo2_data.set_scatter_matrix(scat[0][:, :, np.newaxis])

# --- C5G7 moderator data ---
mod_data = openmc.XSdata("mod", groups)
mod_data.order = 0
mod_data.set_total(
    [0.159206, 0.412970, 0.590310, 0.584350, 0.718000, 1.254450, 2.650380])
mod_data.set_absorption(
    [6.0105e-04, 1.5793e-05, 3.3716e-04, 1.9406e-03, 5.7416e-03, 1.5001e-02,
     3.7239e-02])
scat_m = np.array(
    [[[4.44777e-02, 1.13400e-01, 7.23470e-04, 3.74990e-06, 5.31840e-08, 0.0, 0.0],
      [0.0, 2.82334e-01, 1.29940e-01, 6.23400e-04, 4.80020e-05, 7.44860e-06, 1.04550e-06],
      [0.0, 0.0, 3.45256e-01, 2.24570e-01, 1.69990e-02, 2.64430e-03, 5.03440e-04],
      [0.0, 0.0, 0.0, 9.10284e-02, 4.15510e-01, 6.37320e-02, 1.21390e-02],
      [0.0, 0.0, 0.0, 7.14370e-05, 1.39138e-01, 5.11820e-01, 6.12290e-02],
      [0.0, 0.0, 0.0, 0.0, 2.21570e-03, 6.99913e-01, 5.37320e-01],
      [0.0, 0.0, 0.0, 0.0, 0.0, 1.32440e-01, 2.48070e+00]]])
mod_data.set_scatter_matrix(scat_m[0][:, :, np.newaxis])

mg_lib = openmc.MGXSLibrary(groups)
mg_lib.add_xsdatas([uo2_data, mod_data])
mg_lib.export_to_hdf5("mgxs.h5")

# --- materials referencing macroscopic data ---
uo2 = openmc.Material(name="UO2")
uo2.set_density("macro", 1.0)
uo2.add_macroscopic("uo2")
water = openmc.Material(name="water")
water.set_density("macro", 1.0)
water.add_macroscopic("mod")
mats = openmc.Materials([uo2, water])
mats.cross_sections = "mgxs.h5"

# --- 3x3 pin lattice geometry ---
pitch = 1.26
fuel_or = openmc.ZCylinder(r=0.54)

fuel_cell = openmc.Cell(fill=uo2, region=-fuel_or)
mod_cell = openmc.Cell(fill=water, region=+fuel_or)
pin = openmc.Universe(cells=[fuel_cell, mod_cell])

wf_cell = openmc.Cell(fill=water)
wu = openmc.Universe(cells=[wf_cell])

lat = openmc.RectLattice()
lat.lower_left = (-1.5 * pitch, -1.5 * pitch)
lat.pitch = (pitch, pitch)
lat.universes = [[pin, pin, pin], [pin, wu, pin], [pin, pin, pin]]
lat.outer = wu

box = openmc.model.RectangularPrism(3 * pitch, 3 * pitch,
                                    boundary_type="reflective")
root_cell = openmc.Cell(fill=lat, region=-box)
geom = openmc.Geometry([root_cell])

settings = openmc.Settings()
settings.energy_mode = "multi-group"
settings.run_mode = "eigenvalue"
settings.particles = args.particles
settings.batches = args.batches
settings.inactive = args.inactive
settings.seed = args.seed
settings.source = openmc.IndependentSource(
    space=openmc.stats.Box((-1.8, -1.8, -1), (1.8, 1.8, 1)))

model = openmc.Model(geometry=geom, materials=mats, settings=settings)

if args.tallies:
    t = openmc.Tally(name="pin-rates")
    t.filters = [openmc.CellFilter([fuel_cell, mod_cell]),
                 openmc.EnergyFilter(groups.group_edges)]
    t.scores = ["flux", "nu-fission", "absorption", "fission", "total"]
    model.tallies = openmc.Tallies([t])

model.export_to_model_xml()
