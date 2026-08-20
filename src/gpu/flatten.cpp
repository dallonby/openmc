//! \file flatten.cpp
//! Builds the device-ready flat model from OpenMC's loaded data model.
//! Any feature outside the v1 GPU envelope sets reject_reason and returns
//! false — the caller then leaves the engine inactive and the run proceeds
//! on the CPU with identical semantics.

#include "flatten.h"

#include <fmt/format.h>

#include "openmc/boundary_condition.h"
#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/geometry.h"
#include "openmc/lattice.h"
#include "openmc/material.h"
#include "openmc/mgxs_interface.h"
#include "openmc/scattdata.h"
#include "openmc/settings.h"
#include "openmc/surface.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/filter_cell.h"
#include "openmc/tallies/filter_energy.h"
#include "openmc/tallies/filter_material.h"
#include "openmc/tallies/filter_universe.h"
#include "openmc/tallies/tally.h"
#include "openmc/universe.h"
#include "openmc/xsdata.h"

namespace openmc {
namespace gpu {

namespace {

bool reject(FlatModel& m, std::string why)
{
  m.reject_reason = std::move(why);
  return false;
}

uint32_t push_f32(FlatModel& m, const double* vals, size_t n)
{
  uint32_t off = (uint32_t)m.f32.size();
  for (size_t i = 0; i < n; ++i)
    m.f32.push_back((float)vals[i]);
  return off;
}

bool flatten_surfaces(FlatModel& m)
{
  m.surfaces.clear();
  for (const auto& sp : model::surfaces) {
    const Surface* s = sp.get();
    GpuSurface gs {};
    gs.albedo = -1.0f;

    // boundary condition
    if (!s->bc_) {
      gs.bc = GPU_BC_TRANSMISSION;
    } else if (dynamic_cast<const VacuumBC*>(s->bc_.get())) {
      gs.bc = GPU_BC_VACUUM;
    } else if (dynamic_cast<const ReflectiveBC*>(s->bc_.get())) {
      gs.bc = GPU_BC_REFLECTIVE;
    } else if (dynamic_cast<const WhiteBC*>(s->bc_.get())) {
      gs.bc = GPU_BC_WHITE;
    } else {
      return reject(m,
        fmt::format("surface {} has an unsupported boundary condition "
                    "(periodic BCs are not yet in the GPU envelope)",
          s->id_));
    }
    if (s->bc_ && s->bc_->has_albedo()) {
      return reject(
        m, fmt::format("surface {} uses a boundary albedo", s->id_));
    }

    double c[10] = {};
    size_t nc = 0;
    if (auto* p = dynamic_cast<const SurfaceXPlane*>(s)) {
      gs.type = GPU_SURF_X_PLANE;
      c[0] = p->x0_;
      nc = 1;
    } else if (auto* p = dynamic_cast<const SurfaceYPlane*>(s)) {
      gs.type = GPU_SURF_Y_PLANE;
      c[0] = p->y0_;
      nc = 1;
    } else if (auto* p = dynamic_cast<const SurfaceZPlane*>(s)) {
      gs.type = GPU_SURF_Z_PLANE;
      c[0] = p->z0_;
      nc = 1;
    } else if (auto* p = dynamic_cast<const SurfacePlane*>(s)) {
      gs.type = GPU_SURF_PLANE;
      c[0] = p->A_;
      c[1] = p->B_;
      c[2] = p->C_;
      c[3] = p->D_;
      nc = 4;
    } else if (auto* p = dynamic_cast<const SurfaceXCylinder*>(s)) {
      gs.type = GPU_SURF_X_CYLINDER;
      c[0] = p->y0_;
      c[1] = p->z0_;
      c[2] = p->radius_;
      nc = 3;
    } else if (auto* p = dynamic_cast<const SurfaceYCylinder*>(s)) {
      gs.type = GPU_SURF_Y_CYLINDER;
      c[0] = p->x0_;
      c[1] = p->z0_;
      c[2] = p->radius_;
      nc = 3;
    } else if (auto* p = dynamic_cast<const SurfaceZCylinder*>(s)) {
      gs.type = GPU_SURF_Z_CYLINDER;
      c[0] = p->x0_;
      c[1] = p->y0_;
      c[2] = p->radius_;
      nc = 3;
    } else if (auto* p = dynamic_cast<const SurfaceSphere*>(s)) {
      gs.type = GPU_SURF_SPHERE;
      c[0] = p->x0_;
      c[1] = p->y0_;
      c[2] = p->z0_;
      c[3] = p->radius_;
      nc = 4;
    } else if (auto* p = dynamic_cast<const SurfaceXCone*>(s)) {
      gs.type = GPU_SURF_X_CONE;
      c[0] = p->x0_;
      c[1] = p->y0_;
      c[2] = p->z0_;
      c[3] = p->radius_sq_;
      nc = 4;
    } else if (auto* p = dynamic_cast<const SurfaceYCone*>(s)) {
      gs.type = GPU_SURF_Y_CONE;
      c[0] = p->x0_;
      c[1] = p->y0_;
      c[2] = p->z0_;
      c[3] = p->radius_sq_;
      nc = 4;
    } else if (auto* p = dynamic_cast<const SurfaceZCone*>(s)) {
      gs.type = GPU_SURF_Z_CONE;
      c[0] = p->x0_;
      c[1] = p->y0_;
      c[2] = p->z0_;
      c[3] = p->radius_sq_;
      nc = 4;
    } else if (auto* p = dynamic_cast<const SurfaceQuadric*>(s)) {
      gs.type = GPU_SURF_QUADRIC;
      c[0] = p->A_;
      c[1] = p->B_;
      c[2] = p->C_;
      c[3] = p->D_;
      c[4] = p->E_;
      c[5] = p->F_;
      c[6] = p->G_;
      c[7] = p->H_;
      c[8] = p->J_;
      c[9] = p->K_;
      nc = 10;
    } else {
      return reject(m,
        fmt::format("surface {} has a type outside the GPU envelope "
                    "(tori are not yet supported)",
          s->id_));
    }
    gs.coeff_off = push_f32(m, c, nc);
    m.surfaces.push_back(gs);
  }
  return true;
}

bool flatten_cells(FlatModel& m)
{
  m.cells.clear();
  for (const auto& cp : model::cells) {
    const Cell* c = cp.get();
    auto* csg = dynamic_cast<const CSGCell*>(c);
    if (!csg) {
      return reject(
        m, fmt::format("cell {} is not a CSG cell (DAGMC unsupported)",
             c->id_));
    }
    if (c->material_.size() > 1 || c->sqrtkT_.size() > 1 ||
        c->density_mult_.size() > 1) {
      return reject(m,
        fmt::format("cell {} uses distributed materials/temperatures/"
                    "densities (unsupported in GPU v1)",
          c->id_));
    }

    GpuCell gc {};
    switch (c->type_) {
    case Fill::MATERIAL:
      gc.fill_type = GPU_FILL_MATERIAL;
      break;
    case Fill::UNIVERSE:
      gc.fill_type = GPU_FILL_UNIVERSE;
      break;
    case Fill::LATTICE:
      gc.fill_type = GPU_FILL_LATTICE;
      break;
    }
    gc.fill = c->fill_;
    gc.material = c->material_.empty() ? GPU_MATERIAL_VOID : c->material_[0];
    if (gc.material == MATERIAL_VOID)
      gc.material = GPU_MATERIAL_VOID;
    gc.universe = c->universe_;
    gc.sqrtkT = c->sqrtkT_.empty() ? 0.0f : (float)c->sqrtkT_[0];
    gc.density_mult =
      c->density_mult_.empty() ? 1.0f : (float)c->density_mult_[0];

    const auto& expr = csg->region().expression();
    gc.token_off = (uint32_t)m.i32.size();
    gc.n_tokens = (uint32_t)expr.size();
    for (int32_t tok : expr)
      m.i32.push_back(tok);
    gc.simple = csg->is_simple() ? 1u : 0u;

    if (c->translation_.x != 0.0 || c->translation_.y != 0.0 ||
        c->translation_.z != 0.0) {
      double t[3] = {c->translation_.x, c->translation_.y, c->translation_.z};
      gc.trans_off = (int32_t)push_f32(m, t, 3);
    } else {
      gc.trans_off = -1;
    }
    if (!c->rotation_.empty()) {
      gc.rot_off = (int32_t)push_f32(m, c->rotation_.data(), 9);
    } else {
      gc.rot_off = -1;
    }
    m.cells.push_back(gc);
  }
  return true;
}

bool flatten_universes_lattices(FlatModel& m)
{
  m.universes.clear();
  for (const auto& up : model::universes) {
    GpuUniverse gu {};
    gu.cells_off = (uint32_t)m.i32.size();
    gu.n_cells = (uint32_t)up->cells_.size();
    for (int32_t ic : up->cells_)
      m.i32.push_back(ic);
    m.universes.push_back(gu);
  }

  m.lattices.clear();
  for (const auto& lp : model::lattices) {
    auto* rect = dynamic_cast<const RectLattice*>(lp.get());
    if (!rect) {
      return reject(m,
        fmt::format(
          "lattice {} is not rectangular (hex lattices unsupported in v1)",
          lp->id_));
    }
    GpuLattice gl {};
    gl.nx = rect->n_cells()[0];
    gl.ny = rect->n_cells()[1];
    gl.nz = rect->n_cells()[2];
    gl.llx = (float)rect->lower_left().x;
    gl.lly = (float)rect->lower_left().y;
    gl.llz = (float)rect->lower_left().z;
    gl.px = (float)rect->pitch().x;
    gl.py = (float)rect->pitch().y;
    gl.pz = (float)rect->pitch().z;
    gl.is_3d = rect->is_3d() ? 1u : 0u;
    gl.outer = rect->outer_;
    gl.univ_off = (uint32_t)m.i32.size();
    for (int32_t iu : rect->universes_)
      m.i32.push_back(iu);
    m.lattices.push_back(gl);
  }
  return true;
}

bool flatten_mg(FlatModel& m)
{
  const auto& mgi = data::mg;
  int G = mgi.num_energy_groups_;

  // group mean energies for tally energy filters / particle E bookkeeping
  m.mg_bin_avg_off =
    push_f32(m, mgi.energy_bin_avg_.data(), mgi.energy_bin_avg_.size());

  m.mgmats.clear();
  m.materials.clear();
  for (size_t im = 0; im < model::materials.size(); ++im) {
    if (im >= mgi.macro_xs_.size()) {
      return reject(m, "material index outside MG macro data");
    }
    const Mgxs& mx = mgi.macro_xs_[im];
    if (!mx.is_isotropic) {
      return reject(m,
        fmt::format("MG data '{}' is angle-dependent (unsupported in v1)",
          mx.name));
    }
    if (mx.get_scatter_format() == AngleDistributionType::LEGENDRE) {
      return reject(m,
        fmt::format("MG data '{}' retains Legendre scattering; enable "
                    "tabular_legendre conversion (default) for GPU runs",
          mx.name));
    }
    if (mx.xs_data().size() != 1) {
      return reject(m,
        fmt::format("MG data '{}' has multiple temperatures (unsupported "
                    "in GPU v1)",
          mx.name));
    }
    const XsData& xd = mx.xs_data()[0];
    int ndg = mx.n_delayed_groups();

    GpuMgMat gm {};
    gm.fissionable = mx.fissionable ? 1u : 0u;
    gm.n_delayed = (uint32_t)ndg;

    // vector block
    gm.xs_off = (uint32_t)m.f32.size();
    auto push_vec = [&](const tensor::Tensor<double>& t) {
      for (int g = 0; g < G; ++g)
        m.f32.push_back(t.size() ? (float)t(0, g) : 0.0f);
    };
    push_vec(xd.total);
    push_vec(xd.absorption);
    push_vec(xd.nu_fission);
    push_vec(xd.prompt_nu_fission);
    push_vec(xd.fission);
    // scatt_xs from ScattData
    const ScattData* sd = xd.scatter.empty() ? nullptr : xd.scatter[0].get();
    if (!sd)
      return reject(m, fmt::format("MG data '{}' missing scatter", mx.name));
    for (int g = 0; g < G; ++g)
      m.f32.push_back((float)sd->scattxs(g));
    push_vec(xd.inverse_velocity);

    // chi_prompt [a][gin][gout]
    gm.chi_p_off = (uint32_t)m.f32.size();
    for (int gi = 0; gi < G; ++gi)
      for (int go = 0; go < G; ++go)
        m.f32.push_back(
          xd.chi_prompt.size() ? (float)xd.chi_prompt(0, gi, go) : 0.0f);

    // delayed blocks
    gm.dnf_off = (uint32_t)m.f32.size();
    for (int dg = 0; dg < ndg; ++dg)
      for (int gi = 0; gi < G; ++gi)
        m.f32.push_back((float)xd.delayed_nu_fission(0, dg, gi));
    gm.chi_d_off = (uint32_t)m.f32.size();
    for (int dg = 0; dg < ndg; ++dg)
      for (int gi = 0; gi < G; ++gi)
        for (int go = 0; go < G; ++go)
          m.f32.push_back((float)xd.chi_delayed(0, dg, gi, go));
    gm.decay_off = (uint32_t)m.f32.size();
    for (int dg = 0; dg < ndg; ++dg)
      m.f32.push_back((float)xd.decay_rate(0, dg));

    // scattering: bounds, row_ptr, probs, mult, angle descriptors
    gm.sc_bounds_off = (uint32_t)m.i32.size();
    for (int g = 0; g < G; ++g)
      m.i32.push_back(sd->gmin(g));
    for (int g = 0; g < G; ++g)
      m.i32.push_back(sd->gmax(g));

    gm.sc_rowptr_off = (uint32_t)m.i32.size();
    int32_t acc = 0;
    for (int g = 0; g < G; ++g) {
      m.i32.push_back(acc);
      acc += (int32_t)sd->energy[g].size();
    }
    m.i32.push_back(acc);

    gm.sc_prob_off = (uint32_t)m.f32.size();
    for (int g = 0; g < G; ++g)
      for (double v : sd->energy[g])
        m.f32.push_back((float)v);
    gm.sc_mult_off = (uint32_t)m.f32.size();
    for (int g = 0; g < G; ++g)
      for (double v : sd->mult[g])
        m.f32.push_back((float)v);

    // angle distributions per (gin, i_gout) pair
    const auto* tab = dynamic_cast<const ScattDataTabular*>(sd);
    const auto* hist = dynamic_cast<const ScattDataHistogram*>(sd);
    if (!tab && !hist)
      return reject(m,
        fmt::format("MG data '{}' has an unsupported scattering "
                    "representation",
          mx.name));

    gm.sc_ang_off = (uint32_t)m.i32.size();
    // reserve descriptor space first (3 ints per pair)
    size_t n_pairs = (size_t)acc;
    size_t desc_base = m.i32.size();
    m.i32.resize(desc_base + 3 * n_pairs);
    size_t pair = 0;
    const double_3dvec& fmu = tab ? tab->fmu_data() : hist->fmu_data();
    for (int g = 0; g < G; ++g) {
      for (size_t j = 0; j < sd->energy[g].size(); ++j, ++pair) {
        const auto& f = fmu[g][j];
        const auto& cdf = sd->dist[g][j];
        int n_mu = (int)f.size();
        // flat distribution → isotropic shortcut (saves table walks)
        bool flat = true;
        for (double v : f)
          if (std::abs(v - 0.5) > 1e-12) {
            flat = false;
            break;
          }
        int32_t type =
          flat ? GPU_MG_ANGLE_ISOTROPIC
               : (tab ? GPU_MG_ANGLE_TABULAR : GPU_MG_ANGLE_HISTOGRAM);
        int32_t aoff = 0;
        if (!flat) {
          aoff = (int32_t)m.f32.size();
          for (double v : f)
            m.f32.push_back((float)v);
          for (double v : cdf)
            m.f32.push_back((float)v);
        }
        m.i32[desc_base + 3 * pair] = type;
        m.i32[desc_base + 3 * pair + 1] = n_mu;
        m.i32[desc_base + 3 * pair + 2] = aoff;
      }
    }

    m.mgmats.push_back(gm);

    GpuMaterial mat {};
    mat.mg_off = (int32_t)(m.mgmats.size() - 1);
    mat.fissionable = mx.fissionable ? 1u : 0u;
    m.materials.push_back(mat);
  }
  return true;
}

} // namespace

bool flatten_tallies(FlatModel& m)
{
  m.tallies.clear();
  m.filters.clear();
  m.meshes.clear();
  m.tally_host_index.clear();
  m.tally_accum_size = 0;

  for (size_t it = 0; it < model::tallies.size(); ++it) {
    const Tally* t = model::tallies[it].get();

    uint32_t est;
    switch (t->estimator_) {
    case TallyEstimator::TRACKLENGTH:
      est = GPU_ESTIMATOR_TRACKLENGTH;
      break;
    case TallyEstimator::COLLISION:
      est = GPU_ESTIMATOR_COLLISION;
      break;
    default:
      return reject(m,
        fmt::format("tally {} uses the analog estimator (unsupported in "
                    "GPU v1)",
          t->id()));
    }

    // nuclide bins: only total
    if (t->nuclides_.size() > 1 ||
        (t->nuclides_.size() == 1 && t->nuclides_[0] != -1)) {
      return reject(m,
        fmt::format("tally {} has nuclide bins (unsupported in GPU v1)",
          t->id()));
    }

    GpuTallyDesc td {};
    td.estimator = est;
    td.filter_off = (uint32_t)m.filters.size();
    td.n_filters = (uint32_t)t->filters().size();

    for (int fi = 0; fi < (int)t->filters().size(); ++fi) {
      const Filter* f = model::tally_filters[t->filters(fi)].get();
      GpuFilterDesc fd {};
      fd.stride = (uint32_t)t->strides(fi);
      fd.mesh = -1;
      if (auto* cf = dynamic_cast<const CellFilter*>(f)) {
        fd.type = GPU_FILTER_CELL;
        fd.n_bins = (uint32_t)cf->cells().size();
        fd.map_off = (uint32_t)m.i32.size();
        m.i32.resize(m.i32.size() + model::cells.size(), -1);
        for (size_t b = 0; b < cf->cells().size(); ++b)
          m.i32[fd.map_off + cf->cells()[b]] = (int32_t)b;
      } else if (auto* mf = dynamic_cast<const MaterialFilter*>(f)) {
        fd.type = GPU_FILTER_MATERIAL;
        fd.n_bins = (uint32_t)mf->materials().size();
        fd.map_off = (uint32_t)m.i32.size();
        m.i32.resize(m.i32.size() + model::materials.size(), -1);
        for (size_t b = 0; b < mf->materials().size(); ++b)
          m.i32[fd.map_off + mf->materials()[b]] = (int32_t)b;
      } else if (auto* uf = dynamic_cast<const UniverseFilter*>(f)) {
        fd.type = GPU_FILTER_UNIVERSE;
        fd.n_bins = (uint32_t)uf->universes().size();
        fd.map_off = (uint32_t)m.i32.size();
        m.i32.resize(m.i32.size() + model::universes.size(), -1);
        for (size_t b = 0; b < uf->universes().size(); ++b)
          m.i32[fd.map_off + uf->universes()[b]] = (int32_t)b;
      } else if (auto* ef = dynamic_cast<const EnergyFilter*>(f)) {
        // matches_transport_groups needs no special handling here: MG
        // particles carry the group mean energy, which bins identically
        fd.type = GPU_FILTER_ENERGY;
        fd.n_bins = (uint32_t)ef->bins().size() - 1;
        fd.map_off = push_f32(m, ef->bins().data(), ef->bins().size());
      } else {
        return reject(m,
          fmt::format("tally {} has a filter type outside the GPU v1 "
                      "envelope",
            t->id()));
      }
      m.filters.push_back(fd);
    }

    // scores
    td.n_scores = (uint32_t)t->scores_.size();
    td.score_off = (uint32_t)m.i32.size();
    for (int sc : t->scores_) {
      int32_t code;
      switch (sc) {
      case SCORE_FLUX:
        code = GPU_SCORE_FLUX;
        break;
      case SCORE_TOTAL:
        code = GPU_SCORE_TOTAL;
        break;
      case SCORE_ABSORPTION:
        code = GPU_SCORE_ABSORPTION;
        break;
      case SCORE_FISSION:
        code = GPU_SCORE_FISSION;
        break;
      case SCORE_NU_FISSION:
        code = GPU_SCORE_NU_FISSION;
        break;
      case SCORE_SCATTER:
        code = GPU_SCORE_SCATTER;
        break;
      case 2: // ELASTIC (MT number)
        code = GPU_SCORE_ELASTIC;
        break;
      default:
        return reject(m,
          fmt::format(
            "tally {} has a score outside the GPU v1 envelope", t->id()));
      }
      m.i32.push_back(code);
    }

    td.n_filter_bins = (uint32_t)t->n_filter_bins();
    td.accum_off = m.tally_accum_size;
    m.tally_accum_size += td.n_filter_bins * td.n_scores;
    m.tallies.push_back(td);
    m.tally_host_index.push_back((int32_t)it);
  }
  return true;
}

bool flatten_model(FlatModel& m)
{
  m.reject_reason.clear();

  if (settings::photon_transport)
    return reject(m, "photon transport is not in the GPU envelope");
  if (settings::survival_biasing)
    return reject(m, "survival biasing is not in the GPU v1 envelope");
  if (settings::weight_windows_on)
    return reject(m, "weight windows are not in the GPU v1 envelope");
  if (settings::solver_type != SolverType::MONTE_CARLO)
    return reject(m, "random ray solver cannot run on the GPU engine");
  if (settings::ufs_on)
    return reject(m, "uniform fission source is not in the GPU v1 envelope");
  if (settings::ifp_on)
    return reject(m, "IFP is not in the GPU v1 envelope");
  if (model::n_coord_levels > GPU_MAX_COORD)
    return reject(m,
      fmt::format("geometry has {} coordinate levels (GPU supports {})",
        model::n_coord_levels, GPU_MAX_COORD));

  if (!flatten_surfaces(m))
    return false;
  if (!flatten_cells(m))
    return false;
  if (!flatten_universes_lattices(m))
    return false;
  if (settings::run_CE) {
    if (!flatten_ce(m))
      return false;
  } else {
    if (!flatten_mg(m))
      return false;
  }
  if (!flatten_tallies(m))
    return false;
  return true;
}

} // namespace gpu
} // namespace openmc
