//! \file flatten_ce.cpp
//! Flattens the continuous-energy nuclear data model (data::nuclides) into
//! the device arenas defined in device/ce.h. All heavy lifting stays on the
//! host: temperature selection, distribution tree walks, fp64->fp32
//! conversion with duplicate-grid-point collapsing.

#include "flatten.h"

#include <cmath>
#include <set>

#include <fmt/format.h>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/distribution.h"
#include "openmc/distribution_angle.h"
#include "openmc/distribution_energy.h"
#include "openmc/endf.h"
#include "openmc/error.h"
#include "openmc/material.h"
#include "openmc/nuclide.h"
#include "openmc/reaction.h"
#include "openmc/reaction_product.h"
#include "openmc/secondary_correlated.h"
#include "openmc/secondary_kalbach.h"
#include "openmc/secondary_nbody.h"
#include "openmc/secondary_uncorrelated.h"
#include "openmc/secondary_thermal.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/thermal.h"
#include "openmc/urr.h"

namespace openmc {

//! Friend of the distribution classes: extracts private tables.
struct GpuCeFlatten {
  gpu::FlatModel& m;
  std::string err;

  explicit GpuCeFlatten(gpu::FlatModel& model) : m(model) {}

  uint32_t f32_off() const { return (uint32_t)m.f32.size(); }
  uint32_t i32_off() const { return (uint32_t)m.i32.size(); }

  uint32_t push_f(const double* v, size_t n)
  {
    uint32_t off = f32_off();
    for (size_t i = 0; i < n; ++i)
      m.f32.push_back((float)v[i]);
    return off;
  }
  template<typename C>
  uint32_t push_fc(const C& c)
  {
    uint32_t off = f32_off();
    for (double v : c)
      m.f32.push_back((float)v);
    return off;
  }

  // ---- Function1D ----
  int32_t f1d(const Function1D* f)
  {
    if (!f)
      return -1;
    if (auto* p = dynamic_cast<const Polynomial*>(f)) {
      int32_t blob = (int32_t)i32_off();
      m.i32.push_back(GPU_F1D_POLY);
      m.i32.push_back((int32_t)p->coef_.size());
      m.i32.push_back((int32_t)push_fc(p->coef_));
      m.i32.push_back(0);
      m.i32.push_back(0);
      return blob;
    }
    if (auto* t = dynamic_cast<const Tabulated1D*>(f)) {
      size_t n = t->x().size();
      uint32_t xoff = push_fc(t->x());
      push_fc(t->y());
      int32_t reg_off = (int32_t)i32_off();
      for (size_t j = 0; j < t->n_regions_; ++j) {
        m.i32.push_back(t->nbt_[j]);
        m.i32.push_back((int32_t)t->int_[j]);
      }
      int32_t blob = (int32_t)i32_off();
      m.i32.push_back(GPU_F1D_TAB);
      m.i32.push_back((int32_t)n);
      m.i32.push_back((int32_t)xoff);
      m.i32.push_back((int32_t)t->n_regions_);
      m.i32.push_back(reg_off);
      return blob;
    }
    err = "unsupported Function1D subtype";
    return -1;
  }

  // ---- tabular mu table (Tabular) -> [n, interp, f32off(mu,p,c)] ----
  int32_t tabular_blob(const Tabular* t)
  {
    int32_t blob = (int32_t)i32_off();
    size_t n = t->x_.size();
    m.i32.push_back((int32_t)n);
    m.i32.push_back((int32_t)t->interp_);
    uint32_t off = push_fc(t->x_);
    push_fc(t->p_);
    push_fc(t->c_);
    m.i32.push_back((int32_t)off);
    return blob;
  }

  // ---- AngleDistribution -> [n_E, Egrid_off, table blobs...] ----
  int32_t angle_dist(const AngleDistribution& ad)
  {
    if (ad.empty())
      return -1;
    size_t n = ad.energy_.size();
    // table blobs first
    std::vector<int32_t> blobs(n);
    for (size_t i = 0; i < n; ++i)
      blobs[i] = tabular_blob(ad.distribution_[i].get());
    int32_t hdr = (int32_t)i32_off();
    m.i32.push_back((int32_t)n);
    m.i32.push_back((int32_t)push_fc(ad.energy_));
    for (auto b : blobs)
      m.i32.push_back(b);
    return hdr;
  }

  // ---- energy distributions ----
  int32_t energy_dist(const EnergyDistribution* ed)
  {
    if (auto* lv = dynamic_cast<const LevelInelastic*>(ed)) {
      int32_t hdr = (int32_t)i32_off();
      double d[2] = {lv->threshold_, lv->mass_ratio_};
      m.i32.push_back(GPU_DIST_LEVEL);
      m.i32.push_back((int32_t)push_f(d, 2));
      return hdr;
    }
    if (auto* ct = dynamic_cast<const ContinuousTabular*>(ed)) {
      return cont_tab(ct);
    }
    if (auto* mx = dynamic_cast<const MaxwellEnergy*>(ed)) {
      int32_t th = f1d(&mx->theta_);
      int32_t hdr = (int32_t)i32_off();
      m.i32.push_back(GPU_DIST_MAXWELL);
      m.i32.push_back(th);
      m.i32.push_back((int32_t)push_f(&mx->u_, 1));
      return hdr;
    }
    if (auto* ev = dynamic_cast<const Evaporation*>(ed)) {
      int32_t th = f1d(&ev->theta_);
      int32_t hdr = (int32_t)i32_off();
      m.i32.push_back(GPU_DIST_EVAPORATION);
      m.i32.push_back(th);
      m.i32.push_back((int32_t)push_f(&ev->u_, 1));
      return hdr;
    }
    if (auto* wt = dynamic_cast<const WattEnergy*>(ed)) {
      int32_t fa = f1d(&wt->a_);
      int32_t fb = f1d(&wt->b_);
      int32_t hdr = (int32_t)i32_off();
      m.i32.push_back(GPU_DIST_WATT);
      m.i32.push_back(fa);
      m.i32.push_back(fb);
      m.i32.push_back((int32_t)push_f(&wt->u_, 1));
      return hdr;
    }
    err = "unsupported energy distribution (discrete photon?)";
    return -1;
  }

  int32_t cont_tab(const ContinuousTabular* ct)
  {
    bool hist =
      (ct->n_region_ == 1 && ct->interpolation_[0] == Interpolation::histogram);
    size_t n = ct->energy_.size();
    std::vector<int32_t> tbl(n);
    for (size_t i = 0; i < n; ++i) {
      const auto& t = ct->distribution_[i];
      int32_t b = (int32_t)i32_off();
      int32_t n_out = (int32_t)t.e_out.size();
      m.i32.push_back(t.n_discrete);
      m.i32.push_back((int32_t)t.interpolation);
      m.i32.push_back(n_out);
      uint32_t off = push_fc(t.e_out);
      push_fc(t.p);
      push_fc(t.c);
      m.i32.push_back((int32_t)off);
      tbl[i] = b;
    }
    int32_t hdr = (int32_t)i32_off();
    m.i32.push_back(GPU_DIST_CONT_TAB);
    m.i32.push_back(hist ? 1 : 0);
    m.i32.push_back((int32_t)n);
    m.i32.push_back((int32_t)push_fc(ct->energy_));
    for (auto b : tbl)
      m.i32.push_back(b);
    return hdr;
  }

  int32_t kalbach(const KalbachMann* km)
  {
    size_t n = km->energy_.size();
    std::vector<int32_t> tbl(n);
    for (size_t i = 0; i < n; ++i) {
      const auto& t = km->distribution_[i];
      int32_t b = (int32_t)i32_off();
      int32_t n_out = (int32_t)t.e_out.size();
      m.i32.push_back(t.n_discrete);
      m.i32.push_back((int32_t)t.interpolation);
      m.i32.push_back(n_out);
      uint32_t off = push_fc(t.e_out);
      push_fc(t.p);
      push_fc(t.c);
      push_fc(t.r);
      push_fc(t.a);
      m.i32.push_back((int32_t)off);
      tbl[i] = b;
    }
    int32_t hdr = (int32_t)i32_off();
    m.i32.push_back(GPU_DIST_KALBACH);
    m.i32.push_back(0);
    m.i32.push_back((int32_t)n);
    m.i32.push_back((int32_t)push_fc(km->energy_));
    for (auto b : tbl)
      m.i32.push_back(b);
    return hdr;
  }

  int32_t correlated(const CorrelatedAngleEnergy* ca)
  {
    size_t n = ca->energy_.size();
    std::vector<int32_t> tbl(n);
    for (size_t i = 0; i < n; ++i) {
      const auto& t = ca->distribution_[i];
      int32_t n_out = (int32_t)t.e_out.size();
      // per-outgoing-energy mu tables first
      std::vector<int32_t> mu_blobs(n_out);
      for (int32_t j = 0; j < n_out; ++j)
        mu_blobs[j] = tabular_blob(t.angle[j].get());
      int32_t b = (int32_t)i32_off();
      m.i32.push_back(t.n_discrete);
      m.i32.push_back((int32_t)t.interpolation);
      m.i32.push_back(n_out);
      uint32_t off = push_fc(t.e_out);
      push_fc(t.p);
      push_fc(t.c);
      m.i32.push_back((int32_t)off);
      for (auto mb : mu_blobs)
        m.i32.push_back(mb);
      tbl[i] = b;
    }
    int32_t hdr = (int32_t)i32_off();
    m.i32.push_back(GPU_DIST_CORRELATED);
    m.i32.push_back((int32_t)n);
    m.i32.push_back((int32_t)push_fc(ca->energy_));
    for (auto b : tbl)
      m.i32.push_back(b);
    return hdr;
  }

  int32_t uncorr_angle(const UncorrelatedAngleEnergy* d)
  {
    return angle_dist(d->angle_);
  }

  // ---- S(a,b) thermal laws ----
  int32_t coherent_el_blob(const CoherentElasticXS& xs)
  {
    size_t n = xs.bragg_edges().size();
    uint32_t eoff = push_fc(xs.bragg_edges());
    uint32_t foff = push_fc(xs.factors());
    int32_t hdr = (int32_t)i32_off();
    m.i32.push_back(GPU_TDIST_COH_EL);
    m.i32.push_back((int32_t)n);
    m.i32.push_back((int32_t)eoff);
    m.i32.push_back((int32_t)foff);
    return hdr;
  }

  int32_t sab_dist(const AngleEnergy* ae, int table_index)
  {
    if (auto* c = dynamic_cast<const CoherentElasticAE*>(ae)) {
      return coherent_el_blob(c->xs_);
    }
    if (auto* ie = dynamic_cast<const IncoherentElasticAE*>(ae)) {
      double d[2] = {0.0, ie->debye_waller_};
      int32_t hdr = (int32_t)i32_off();
      m.i32.push_back(GPU_TDIST_INCOH_EL);
      m.i32.push_back((int32_t)push_f(d, 2));
      return hdr;
    }
    if (auto* ied = dynamic_cast<const IncoherentElasticAEDiscrete*>(ae)) {
      int n_e = (int)ied->energy_.size();
      int n_mu = (int)ied->mu_out_.shape(1);
      uint32_t eoff = push_fc(ied->energy_);
      uint32_t moff = f32_off();
      for (int i = 0; i < n_e; ++i)
        for (int k = 0; k < n_mu; ++k)
          m.f32.push_back((float)ied->mu_out_(i, k));
      int32_t hdr = (int32_t)i32_off();
      m.i32.push_back(GPU_TDIST_INCOH_EL_DISC);
      m.i32.push_back(n_e);
      m.i32.push_back((int32_t)eoff);
      m.i32.push_back(n_mu);
      m.i32.push_back((int32_t)moff);
      return hdr;
    }
    if (auto* id = dynamic_cast<const IncoherentInelasticAEDiscrete*>(ae)) {
      int n_e = (int)id->energy_.size();
      int n_out = (int)id->energy_out_.shape(1);
      int n_mu = (int)id->mu_out_.shape(2);
      uint32_t eoff = push_fc(id->energy_);
      uint32_t eooff = f32_off();
      for (int i = 0; i < n_e; ++i)
        for (int j = 0; j < n_out; ++j)
          m.f32.push_back((float)id->energy_out_(i, j));
      uint32_t moff = f32_off();
      for (int i = 0; i < n_e; ++i)
        for (int j = 0; j < n_out; ++j)
          for (int k = 0; k < n_mu; ++k)
            m.f32.push_back((float)id->mu_out_(i, j, k));
      int32_t hdr = (int32_t)i32_off();
      m.i32.push_back(GPU_TDIST_INCOH_INEL_DISC);
      m.i32.push_back(n_e);
      m.i32.push_back((int32_t)eoff);
      m.i32.push_back(n_out);
      m.i32.push_back(n_mu);
      m.i32.push_back(id->skewed_ ? 1 : 0);
      m.i32.push_back((int32_t)eooff);
      m.i32.push_back((int32_t)moff);
      return hdr;
    }
    if (auto* ic = dynamic_cast<const IncoherentInelasticAE*>(ae)) {
      int n_e = (int)ic->energy_.size();
      std::vector<int32_t> blobs(n_e);
      for (int l = 0; l < n_e; ++l) {
        const auto& d = ic->distribution_[l];
        int n = (int)d.n_e_out;
        int n_mu = (int)d.mu.shape(1);
        uint32_t off = f32_off();
        for (int j = 0; j < n; ++j)
          m.f32.push_back((float)d.e_out(j));
        for (int j = 0; j < n; ++j)
          m.f32.push_back((float)d.e_out_pdf(j));
        for (int j = 0; j < n; ++j)
          m.f32.push_back((float)d.e_out_cdf(j));
        for (int j = 0; j < n; ++j)
          for (int k = 0; k < n_mu; ++k)
            m.f32.push_back((float)d.mu(j, k));
        int32_t b = (int32_t)i32_off();
        m.i32.push_back(n);
        m.i32.push_back(n_mu);
        m.i32.push_back((int32_t)off);
        blobs[l] = b;
      }
      int32_t hdr = (int32_t)i32_off();
      m.i32.push_back(GPU_TDIST_INCOH_INEL_CONT);
      m.i32.push_back(n_e);
      m.i32.push_back((int32_t)push_fc(ic->energy_));
      for (auto b : blobs)
        m.i32.push_back(b);
      return hdr;
    }
    if (auto* mx = dynamic_cast<const MixedElasticAE*>(ae)) {
      int32_t coh = coherent_el_blob(mx->coherent_xs_);
      int32_t incoh = sab_dist(mx->incoherent_dist_.get(), table_index);
      if (incoh < 0)
        return -1;
      int32_t ixs = f1d(&mx->incoherent_xs_);
      if (ixs < 0)
        return -1;
      int32_t hdr = (int32_t)i32_off();
      m.i32.push_back(GPU_TDIST_MIXED_EL);
      m.i32.push_back(coh);
      m.i32.push_back(incoh);
      m.i32.push_back(table_index);
      m.i32.push_back(ixs);
      return hdr;
    }
    err = "unsupported thermal scattering law";
    return -1;
  }

  //! Flatten one thermal table (friend access to ThermalData internals)
  bool sab_table(const ThermalScattering& ts, double model_kT,
    GpuSabTable& gt, int table_index)
  {
    int i_temp = 0;
    double best = 1e300;
    for (size_t t = 0; t < ts.kTs_.size(); ++t) {
      double d = std::abs(ts.kTs_[t] - model_kT);
      if (d < best) {
        best = d;
        i_temp = (int)t;
      }
    }
    const ThermalData& td = ts.data_[i_temp];
    gt.awr = (float)ts.awr_;
    gt.kT = (float)ts.kTs_[i_temp];
    gt.energy_max = (float)ts.energy_max_;
    gt.inelastic_xs_f1d = f1d(td.inelastic_.xs.get());
    if (gt.inelastic_xs_f1d < 0) {
      err = "inelastic xs form unsupported";
      return false;
    }
    gt.inelastic_dist = sab_dist(td.inelastic_.distribution.get(),
      table_index);
    if (gt.inelastic_dist < 0)
      return false;
    gt.elastic_xs_type = GPU_SABXS_NONE;
    gt.elastic_xs_blob = -1;
    gt.elastic_dist = -1;
    if (td.elastic_.xs) {
      if (auto* cx =
            dynamic_cast<const CoherentElasticXS*>(td.elastic_.xs.get())) {
        size_t n = cx->bragg_edges().size();
        uint32_t eoff = push_fc(cx->bragg_edges());
        uint32_t foff = push_fc(cx->factors());
        gt.elastic_xs_type = GPU_SABXS_COHERENT;
        gt.elastic_xs_blob = (int32_t)i32_off();
        m.i32.push_back((int32_t)n);
        m.i32.push_back((int32_t)eoff);
        m.i32.push_back((int32_t)foff);
      } else if (auto* ix = dynamic_cast<const IncoherentElasticXS*>(
                   td.elastic_.xs.get())) {
        double d[2] = {ix->bound_xs_, ix->debye_waller_};
        gt.elastic_xs_type = GPU_SABXS_INCOHERENT;
        gt.elastic_xs_blob = (int32_t)push_f(d, 2);
      } else {
        int32_t f = f1d(td.elastic_.xs.get());
        if (f < 0) {
          err = "elastic xs form unsupported";
          return false;
        }
        gt.elastic_xs_type = GPU_SABXS_TAB;
        gt.elastic_xs_blob = f;
      }
      gt.elastic_dist = sab_dist(td.elastic_.distribution.get(), table_index);
      if (gt.elastic_dist < 0)
        return false;
    }
    return true;
  }

  // ---- AngleEnergy dispatch ----
  int32_t angle_energy(const AngleEnergy* ae)
  {
    if (auto* un = dynamic_cast<const UncorrelatedAngleEnergy*>(ae)) {
      int32_t ang = angle_dist(un->angle_);
      int32_t en = energy_dist(un->energy_.get());
      if (en < 0)
        return -1;
      int32_t hdr = (int32_t)i32_off();
      m.i32.push_back(GPU_DIST_UNCORR);
      m.i32.push_back(ang);
      m.i32.push_back(en);
      return hdr;
    }
    if (auto* km = dynamic_cast<const KalbachMann*>(ae))
      return kalbach(km);
    if (auto* ca = dynamic_cast<const CorrelatedAngleEnergy*>(ae))
      return correlated(ca);
    if (auto* nb = dynamic_cast<const NBodyPhaseSpace*>(ae)) {
      if (nb->n_bodies_ < 3 || nb->n_bodies_ > 5) {
        // CPU fatals on other counts; the device would sample 5-body
        err = "N-body phase space with n_bodies outside 3..5";
        return -1;
      }
      int32_t hdr = (int32_t)i32_off();
      double d[3] = {nb->mass_ratio_, nb->A_, nb->Q_};
      m.i32.push_back(GPU_DIST_NBODY);
      m.i32.push_back(nb->n_bodies_);
      m.i32.push_back((int32_t)push_f(d, 3));
      return hdr;
    }
    err = "unsupported angle-energy law";
    return -1;
  }

  // ---- ReactionProduct neutron distribution (with applicability) ----
  int32_t product_dist(const ReactionProduct& prod)
  {
    size_t n = prod.distribution_.size();
    if (n == 1)
      return angle_energy(prod.distribution_[0].get());
    std::vector<int32_t> parts;
    for (size_t i = 0; i < n; ++i) {
      int32_t app = f1d(&prod.applicability_[i]);
      int32_t d = angle_energy(prod.distribution_[i].get());
      if (d < 0)
        return -1;
      parts.push_back(app);
      parts.push_back(d);
    }
    int32_t hdr = (int32_t)i32_off();
    m.i32.push_back(GPU_DIST_MULTI);
    m.i32.push_back((int32_t)n);
    for (auto v : parts)
      m.i32.push_back(v);
    return hdr;
  }
};

namespace gpu {

namespace {

bool reject_ce(FlatModel& m, std::string why)
{
  m.reject_reason = std::move(why);
  return false;
}

} // namespace

bool flatten_ce(FlatModel& m)
{
  GpuCeFlatten fx(m);

  // one temperature per nuclide, chosen against the model's cell
  // temperatures — v1 requires a single distinct kT in the model
  std::set<double> kts;
  for (const auto& c : model::cells) {
    if (c->type_ != Fill::MATERIAL || c->sqrtkT_.empty())
      continue;
    // void cells carry no meaningful temperature
    if (c->material_.empty() || c->material_[0] == MATERIAL_VOID)
      continue;
    kts.insert(c->sqrtkT_[0] * c->sqrtkT_[0]);
  }
  if (kts.size() > 1)
    return reject_ce(
      m, "multiple cell temperatures (single-temperature CE only in GPU v1)");
  double model_kT = kts.empty() ? 2.53e-2 : *kts.begin();

  m.nuclides.clear();
  for (const auto& np : data::nuclides) {
    const Nuclide& nuc = *np;
    GpuNuclide gn {};
    gn.awr = (float)nuc.awr_;
    gn.index = (uint32_t)nuc.index_;
    gn.fissionable = nuc.fissionable_ ? 1u : 0u;

    if (nuc.multipole_)
      return reject_ce(m, "windowed multipole data is outside the GPU "
                          "envelope");

    // temperature selection: nearest available
    int i_temp = 0;
    double best = 1e300;
    for (size_t t = 0; t < nuc.kTs_.size(); ++t) {
      double d = std::abs(nuc.kTs_[t] - model_kT);
      if (d < best) {
        best = d;
        i_temp = (int)t;
      }
    }
    gn.kT = (float)nuc.kTs_[i_temp];

    // energy grid (+ dedupe of fp32-collapsed points is handled by the
    // device's duplicate-point guard)
    const auto& grid = nuc.grid_[i_temp];
    size_t ng = grid.energy.size();
    gn.n_grid = (uint32_t)ng;
    gn.grid_off = fx.push_fc(grid.energy);

    // log hash table
    gn.loggrid_off = fx.i32_off();
    for (int v : grid.grid_index)
      m.i32.push_back(v);

    // xs table [n][4]: total, absorption, fission, nu_fission
    // column indices fixed upstream (nuclide.cpp:45): 0 total,
    // 1 absorption, 2 fission, 3 nu-fission, 4 photon production
    const auto& xs = nuc.xs_[i_temp];
    gn.xs_off = fx.f32_off();
    for (size_t i = 0; i < ng; ++i) {
      m.f32.push_back((float)xs(i, 0));
      m.f32.push_back((float)xs(i, 1));
      m.f32.push_back((float)xs(i, 2));
      m.f32.push_back((float)xs(i, 3));
    }

    // elastic xs (reactions_[0], threshold 0)
    {
      const auto& el = nuc.reactions_[0]->xs_[i_temp];
      if (el.threshold != 0 || el.value.size() != ng)
        return reject_ce(
          m, fmt::format("nuclide {} elastic grid mismatch", nuc.name_));
      gn.elastic_off = fx.push_fc(el.value);
    }

    // elastic angular distribution
    {
      auto* d = dynamic_cast<UncorrelatedAngleEnergy*>(
        nuc.reactions_[0]->products_[0].distribution_[0].get());
      if (!d)
        return reject_ce(m,
          fmt::format("nuclide {} elastic law is not uncorrelated", nuc.name_));
      gn.elastic_angle = fx.uncorr_angle(d);
    }

    // fission (including partial fission reactions)
    gn.total_nu_f1d = -1;
    gn.n_delayed = 0;
    gn.n_fission_rx = 0;
    gn.fis_rx_off = 0;
    if (nuc.fissionable_) {
      const Reaction* frx0 = nuc.fission_rx_[0];
      // nu: total if available (and delayed neutrons on), else prompt
      if (nuc.total_nu_ && settings::create_delayed_neutrons) {
        gn.total_nu_f1d = fx.f1d(nuc.total_nu_.get());
      } else {
        gn.total_nu_f1d = fx.f1d(frx0->products_[0].yield_.get());
      }
      if (gn.total_nu_f1d < 0)
        return reject_ce(
          m, fmt::format("nuclide {}: no usable nu function ({})", nuc.name_,
               fx.err.empty() ? "missing yield" : fx.err));
      int n_del = settings::create_delayed_neutrons ? nuc.n_precursor_ : 0;
      gn.n_delayed = (uint32_t)n_del;

      std::vector<int32_t> rx_recs;
      for (const Reaction* frx : nuc.fission_rx_) {
        // reaction header (xs for partial-fission selection)
        const auto& rxs = frx->xs_[i_temp];
        double q = frx->q_value_;
        int32_t hdr = fx.i32_off();
        m.i32.push_back(frx->mt_);
        m.i32.push_back(frx->scatter_in_cm_ ? 1 : 0);
        m.i32.push_back(rxs.threshold);
        m.i32.push_back((int32_t)rxs.value.size());
        m.i32.push_back((int32_t)fx.push_fc(rxs.value));
        m.i32.push_back((int32_t)fx.push_f(&q, 1));
        m.i32.push_back(-1);
        m.i32.push_back(-1);

        // neutron products: index 0 prompt, then delayed groups present
        // on this reaction
        std::vector<int32_t> prods;
        int n_have = 0;
        for (const auto& prod : frx->products_) {
          if (!prod.particle_.is_neutron())
            continue;
          if (n_have > n_del)
            break;
          int32_t y = fx.f1d(prod.yield_.get());
          int32_t d = fx.product_dist(prod);
          if (d < 0)
            return reject_ce(m,
              fmt::format("nuclide {} fission product: {}", nuc.name_, fx.err));
          double dr = prod.decay_rate_;
          prods.push_back(y);
          prods.push_back(d);
          prods.push_back((int32_t)fx.push_f(&dr, 1));
          ++n_have;
        }
        int32_t rec = fx.i32_off();
        m.i32.push_back(hdr);
        m.i32.push_back(n_have);
        for (auto v : prods)
          m.i32.push_back(v);
        rx_recs.push_back(rec);
      }
      gn.n_fission_rx = (uint32_t)rx_recs.size();
      gn.fis_rx_off = fx.i32_off();
      for (auto v : rx_recs)
        m.i32.push_back(v);
    }

    // inelastic scattering reactions
    std::vector<int32_t> rx_offs;
    for (int idx : nuc.index_inelastic_scatter_) {
      const Reaction& rx = *nuc.reactions_[idx];
      const auto& rxs = rx.xs_[i_temp];
      int32_t y = fx.f1d(rx.products_[0].yield_.get());
      int32_t d = fx.product_dist(rx.products_[0]);
      if (d < 0)
        return reject_ce(
          m, fmt::format("nuclide {} MT={}: {}", nuc.name_, rx.mt_, fx.err));
      double q = rx.q_value_;
      int32_t hdr = fx.i32_off();
      m.i32.push_back(rx.mt_);
      m.i32.push_back(rx.scatter_in_cm_ ? 1 : 0);
      m.i32.push_back(rxs.threshold);
      m.i32.push_back((int32_t)rxs.value.size());
      m.i32.push_back((int32_t)fx.push_fc(rxs.value));
      m.i32.push_back((int32_t)fx.push_f(&q, 1));
      m.i32.push_back(y);
      m.i32.push_back(d);
      rx_offs.push_back(hdr);
    }
    gn.n_inelastic = (uint32_t)rx_offs.size();
    gn.inelastic_off = fx.i32_off();
    for (auto v : rx_offs)
      m.i32.push_back(v);

    // URR probability tables
    gn.urr_off = -1;
    if (settings::urr_ptables_on && nuc.urr_present_) {
      const UrrData& u = nuc.urr_data_[i_temp];
      if (u.interp_ != Interpolation::lin_lin &&
          u.interp_ != Interpolation::log_log)
        return reject_ce(m,
          fmt::format("nuclide {} URR interpolation unsupported", nuc.name_));
      int n_e = (int)u.n_energy();
      int n_b = (int)u.n_cdf();
      // inelastic competition reaction header
      int32_t inel = -1;
      if (u.inelastic_flag_ != C_NONE) {
        const Reaction& rx = *nuc.reactions_[nuc.urr_inelastic_];
        const auto& rxs = rx.xs_[i_temp];
        double q = rx.q_value_;
        inel = fx.i32_off();
        m.i32.push_back(rx.mt_);
        m.i32.push_back(rx.scatter_in_cm_ ? 1 : 0);
        m.i32.push_back(rxs.threshold);
        m.i32.push_back((int32_t)rxs.value.size());
        m.i32.push_back((int32_t)fx.push_fc(rxs.value));
        m.i32.push_back((int32_t)fx.push_f(&q, 1));
        m.i32.push_back(-1);
        m.i32.push_back(-1);
      }
      uint32_t eoff = fx.push_fc(u.energy_);
      uint32_t coff = fx.f32_off();
      for (int i = 0; i < n_e; ++i)
        for (int b = 0; b < n_b; ++b)
          m.f32.push_back((float)u.cdf_values_(i, b));
      uint32_t xoff = fx.f32_off();
      for (int i = 0; i < n_e; ++i)
        for (int b = 0; b < n_b; ++b) {
          const auto& s = u.xs_values_(i, b);
          m.f32.push_back((float)s.elastic);
          m.f32.push_back((float)s.fission);
          m.f32.push_back((float)s.n_gamma);
        }
      gn.urr_off = fx.i32_off();
      m.i32.push_back(n_e);
      m.i32.push_back(n_b);
      m.i32.push_back((int32_t)u.interp_);
      m.i32.push_back(inel);
      m.i32.push_back(u.multiply_smooth_ ? 1 : 0);
      m.i32.push_back((int32_t)eoff);
      m.i32.push_back((int32_t)coff);
      m.i32.push_back((int32_t)xoff);
    }

    // catch-all for helpers whose failure a call site did not check
    // (e.g. an unsupported applicability function inside a nested law):
    // err is sticky, so nothing unsupported can slip into the arena
    if (!fx.err.empty())
      return reject_ce(
        m, fmt::format("nuclide {}: {}", nuc.name_, fx.err));

    m.nuclides.push_back(gn);
  }

  // S(a,b) thermal scattering tables
  m.sab_tables.clear();
  for (size_t it = 0; it < data::thermal_scatt.size(); ++it) {
    const ThermalScattering& ts = *data::thermal_scatt[it];
    GpuSabTable gt {};
    if (!fx.sab_table(ts, model_kT, gt, (int)m.sab_tables.size()))
      return reject_ce(
        m, fmt::format("thermal table {}: {}", ts.name_, fx.err));
    m.sab_tables.push_back(gt);
  }

  // materials
  m.materials.clear();
  m.mgmats.clear();
  for (const auto& mp : model::materials) {
    const Material& mat = *mp;
    if (mat.nuclide_.size() > 32)
      return reject_ce(
        m, fmt::format("material {} has more than 32 nuclides", mat.id_));
    GpuMaterial gm {};
    gm.n_nuclides = (uint32_t)mat.nuclide_.size();
    gm.nuclide_off = fx.i32_off();
    for (int in : mat.nuclide_)
      m.i32.push_back(in);
    gm.density_off = fx.f32_off();
    for (size_t i = 0; i < mat.nuclide_.size(); ++i)
      m.f32.push_back((float)mat.atom_density_(i));
    gm.fissionable = mat.fissionable() ? 1u : 0u;
    gm.mg_off = -1;
    gm.sab_off = -1;
    gm.sab_frac_off = -1;
    if (!mat.thermal_tables_.empty()) {
      std::vector<int32_t> idx(mat.nuclide_.size(), -1);
      std::vector<float> frac(mat.nuclide_.size(), 0.0f);
      for (const auto& tt : mat.thermal_tables_) {
        idx[tt.index_nuclide] = tt.index_table;
        frac[tt.index_nuclide] = (float)tt.fraction;
      }
      gm.sab_off = (int32_t)fx.i32_off();
      for (auto v : idx)
        m.i32.push_back(v);
      gm.sab_frac_off = (int32_t)fx.f32_off();
      for (auto v : frac)
        m.f32.push_back(v);
    }
    m.materials.push_back(gm);
  }
  return true;
}

} // namespace gpu
} // namespace openmc
