// SPDX-License-Identifier: BSD-3-Clause
/**
 * Parameters, priors, gradient, MAP fit and Laplace uncertainty of a FRET
 * network model. See FRETNetwork.h.
 *
 * The gradient is the adjoint of the segment filter with respect to the
 * builders' outputs -- generator entries on their sparsity pattern, photon
 * factors, start distribution -- contracted with central differences of
 * those builders in each parameter's transformed coordinate. The builders
 * are photon-free, so a derivative costs two small rebuilds, not a pass over
 * the photons.
 */
#include <IMP/bff/FRETNetwork.h>
#include <IMP/bff/FRETLandscapeGrid.h>
#include <IMP/bff/States.h>
#include <IMP/bff/internal/FRETNetworkFilter.h>

#include <Eigen/Dense>

#if IMP_BFF_HAS_TTTRLIB
#include <tttrlib/i_lbfgs.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

IMPBFF_BEGIN_NAMESPACE

namespace fret_network_detail {

inline double to_z(double v, int t) {
  if (t == FRET_TRANSFORM_LOG) return std::log(v);
  if (t == FRET_TRANSFORM_LOGIT) return std::log(v / (1.0 - v));
  return v;
}

inline double from_z(double z, int t) {
  if (t == FRET_TRANSFORM_LOG) return std::exp(z);
  if (t == FRET_TRANSFORM_LOGIT) return 1.0 / (1.0 + std::exp(-z));
  return z;
}

//! d(natural)/dz.
inline double dv_dz(double v, int t) {
  if (t == FRET_TRANSFORM_LOG) return v;
  if (t == FRET_TRANSFORM_LOGIT) return v * (1.0 - v);
  return 1.0;
}

inline bool on_boundary(double v, int t) {
  if (t == FRET_TRANSFORM_LOG) return !(v > 0.0);
  if (t == FRET_TRANSFORM_LOGIT) return !(v > 0.0 && v < 1.0);
  return !std::isfinite(v);
}

}  // namespace fret_network_detail

namespace fnd = fret_network_detail;

// --- structure helpers ---------------------------------------------------------------

double FRETMeasurement::set_state_distance_from_clouds(int state,
                                                       const std::vector<double>& donor_cloud,
                                                       const std::vector<double>& acceptor_cloud,
                                                       const std::vector<double>& axis,
                                                       int n_samples, int seed) {
  const std::vector<double> h =
      cloud_distance_distribution(donor_cloud, acceptor_cloud, axis, n_samples, seed, true);
  double mean = 0.0, tot = 0.0;
  std::vector<double> centres(h.size());
  for (std::size_t i = 0; i < h.size(); ++i) {
    centres[i] = 0.5 * (axis[i] + axis[i + 1]);
    mean += h[i] * centres[i];
    tot += h[i];
  }
  if (!(tot > 0.0))
    IMP_THROW("set_state_distance_from_clouds: no pair distance falls on the axis",
              IMP::ValueException);
  mean /= tot;
  std::vector<double> off, w;
  for (std::size_t i = 0; i < h.size(); ++i)
    if (h[i] > 0.0) {
      off.push_back(centres[i] - mean);
      w.push_back(h[i]);
    }
  set_state_distance(state, mean, off, w);
  return mean;
}

// --- parameter registry ----------------------------------------------------------------

void FRETNetworkModel::rebuild_parameters() {
  std::vector<Param> old = params_;
  params_.clear();
  auto add = [&](int owner, const std::vector<std::string>& names, const std::vector<int>& tr,
                 const std::vector<int>& kinds, const std::vector<double>& vals) {
    for (std::size_t j = 0; j < names.size(); ++j) {
      Param p;
      p.owner = owner;
      p.local = static_cast<int>(j);
      p.name = names[j];
      p.transform = tr[j];
      p.kind = kinds[j];
      p.prior_mean = 0.0;
      p.prior_sd = 0.0;
      // default: hidden process, distance maps, dye rates
      const std::string& n = names[j];
      p.free = owner < 0 || n.find(".mean[") != std::string::npos ||
               n.find(".map[") != std::string::npos || n.find(".rate[") != std::string::npos;
      for (const Param& o : old)
        if (o.name == n) {
          p.free = o.free;
          p.prior_mean = o.prior_mean;
          p.prior_sd = o.prior_sd;
        }
      if (fnd::on_boundary(vals[j], tr[j])) p.free = false;
      params_.push_back(p);
    }
  };
  add(-1, process_.get_parameter_names(), process_.get_parameter_transforms(),
      process_.get_parameter_kinds(), process_.get_parameter_values());
  for (std::size_t i = 0; i < meas_.size(); ++i)
    add(static_cast<int>(i), meas_[i].get_parameter_names(process_),
        meas_[i].get_parameter_transforms(process_), meas_[i].get_parameter_kinds(process_),
        meas_[i].get_parameter_values(process_));
}

int FRETNetworkModel::parameter_index(const std::string& name) const {
  for (std::size_t i = 0; i < params_.size(); ++i)
    if (params_[i].name == name) return static_cast<int>(i);
  IMP_THROW("FRETNetworkModel: no parameter " << name, IMP::ValueException);
}

std::vector<std::string> FRETNetworkModel::get_parameter_names() const {
  std::vector<std::string> n;
  for (const Param& p : params_) n.push_back(p.name);
  return n;
}

std::vector<double> FRETNetworkModel::get_parameter_values() const {
  std::vector<double> v = process_.get_parameter_values();
  for (const FRETMeasurement& m : meas_)
    for (double x : m.get_parameter_values(process_)) v.push_back(x);
  return v;
}

double FRETNetworkModel::get_parameter_value(const std::string& name) const {
  return get_parameter_values()[parameter_index(name)];
}

void FRETNetworkModel::set_parameter_value(const std::string& name, double value) {
  const Param& p = params_[parameter_index(name)];
  if (p.owner < 0) {
    std::vector<double> v = process_.get_parameter_values();
    v[p.local] = value;
    process_.set_parameter_values(v);
  } else {
    std::vector<double> v = meas_[p.owner].get_parameter_values(process_);
    v[p.local] = value;
    meas_[p.owner].set_parameter_values(process_, v);
  }
  if (p.free && fnd::on_boundary(value, p.transform))
    params_[parameter_index(name)].free = false;
}

void FRETNetworkModel::set_parameter_free(const std::string& name, bool free) {
  const int i = parameter_index(name);
  if (free && fnd::on_boundary(get_parameter_values()[i], params_[i].transform))
    IMP_THROW("set_parameter_free: " << name << " sits on its transform's boundary",
              IMP::ValueException);
  params_[i].free = free;
}

bool FRETNetworkModel::get_parameter_free(const std::string& name) const {
  return params_[parameter_index(name)].free;
}

std::vector<std::string> FRETNetworkModel::get_free_parameter_names() const {
  std::vector<std::string> n;
  for (const Param& p : params_)
    if (p.free) n.push_back(p.name);
  return n;
}

std::vector<double> FRETNetworkModel::get_theta() const {
  const std::vector<double> v = get_parameter_values();
  std::vector<double> t;
  for (std::size_t i = 0; i < params_.size(); ++i)
    if (params_[i].free) t.push_back(fnd::to_z(v[i], params_[i].transform));
  return t;
}

void FRETNetworkModel::configure(const std::vector<double>& theta, FRETHiddenProcess& process,
                                 std::vector<FRETMeasurement>& meas) const {
  process = process_;
  meas = meas_;
  std::vector<double> pv = process.get_parameter_values();
  std::vector<std::vector<double> > mv(meas.size());
  for (std::size_t i = 0; i < meas.size(); ++i) mv[i] = meas[i].get_parameter_values(process);
  std::size_t j = 0;
  for (const Param& p : params_) {
    if (!p.free) continue;
    if (j >= theta.size()) IMP_THROW("FRETNetworkModel: theta too short", IMP::ValueException);
    const double v = fnd::from_z(theta[j++], p.transform);
    if (p.owner < 0) pv[p.local] = v;
    else mv[p.owner][p.local] = v;
  }
  if (j != theta.size())
    IMP_THROW("FRETNetworkModel: theta has " << theta.size() << " entries, " << j
                                             << " parameters are free",
              IMP::ValueException);
  process.set_parameter_values(pv);
  for (std::size_t i = 0; i < meas.size(); ++i) meas[i].set_parameter_values(process, mv[i]);
}

void FRETNetworkModel::set_theta(const std::vector<double>& theta) {
  FRETHiddenProcess p;
  std::vector<FRETMeasurement> m;
  configure(theta, p, m);
  process_ = p;
  meas_ = m;
}

void FRETNetworkModel::set_parameter_prior(const std::string& name, double mean, double sd) {
  Param& p = params_[parameter_index(name)];
  p.prior_mean = mean;
  p.prior_sd = sd > 0.0 ? sd : 0.0;
}

void FRETNetworkModel::set_map_prior_from_path(int measurement,
                                               const std::vector<double>& path_q,
                                               const std::vector<double>& path_distances,
                                               double sd) {
  if (!process_.get_is_landscape())
    IMP_THROW("set_map_prior_from_path: the hidden process is not a landscape",
              IMP::ValueException);
  if (path_q.size() != path_distances.size() || path_q.size() < 2)
    IMP_THROW("set_map_prior_from_path: need >= 2 frames with a distance each",
              IMP::ValueException);
  for (std::size_t i = 1; i < path_q.size(); ++i)
    if (!(path_q[i] > path_q[i - 1]))
      IMP_THROW("set_map_prior_from_path: path_q must increase", IMP::ValueException);
  FRETMeasurement& m = meas_.at(measurement);
  const std::vector<double> q = process_.get_coordinates();
  const int k = static_cast<int>(m.get_map_values().size());
  const std::vector<double> knots = NaturalCubicSpline(q.front(), q.back(), k).get_knots();
  std::vector<double> vals(k);
  for (int i = 0; i < k; ++i) {
    const double x = knots[i];
    std::size_t j = 1;
    while (j + 1 < path_q.size() && path_q[j] < x) ++j;
    const double f = (x - path_q[j - 1]) / (path_q[j] - path_q[j - 1]);
    vals[i] = path_distances[j - 1] + f * (path_distances[j] - path_distances[j - 1]);
  }
  m.set_distance_map(vals, m.get_spread());
  rebuild_parameters();
  for (int i = 0; i < k; ++i)
    set_parameter_prior(m.get_name() + ".map[" + std::to_string(i) + "]", vals[i], sd);
}

// --- priors ------------------------------------------------------------------------

namespace {

//! -log p over the full natural value vector; gradient in the free z.
double fret_network_neg_log_prior(const std::vector<double>& v, const FRETHiddenProcess& process,
                                  const std::vector<FRETMeasurement>& meas,
                                  const std::vector<int>& owner, const std::vector<int>& transform,
                                  const std::vector<char>& free, const std::vector<double>& pm,
                                  const std::vector<double>& psd, double omega, double anchor,
                                  double omega_map, std::vector<double>* grad_full) {
  const std::size_t P = v.size();
  std::vector<double> g(P, 0.0);  // d(-log p)/d(natural v), then converted
  std::vector<double> gz(P, 0.0);  // direct z-scale terms
  double f = 0.0;
  for (std::size_t i = 0; i < P; ++i) {
    if (!(psd[i] > 0.0) || fnd::on_boundary(v[i], transform[i])) continue;
    const double z = fnd::to_z(v[i], transform[i]);
    const double d = (z - pm[i]) / psd[i];
    f += 0.5 * d * d;
    gz[i] += d / psd[i];
  }
  auto second_differences = [&](std::size_t first, int k, double h, double w) {
    if (!(w > 0.0) || k < 3) return;
    const double c = w / (h * h * h * h);
    for (int j = 1; j + 1 < k; ++j) {
      const double d = v[first + j + 1] - 2.0 * v[first + j] + v[first + j - 1];
      f += c * d * d;
      g[first + j + 1] += 2.0 * c * d;
      g[first + j] += -4.0 * c * d;
      g[first + j - 1] += 2.0 * c * d;
    }
  };
  if (process.get_is_landscape()) {
    const int k = static_cast<int>(process.get_knot_heights().size());
    const std::vector<double> kn = process.get_knots();
    second_differences(0, k, kn[1] - kn[0], omega);
    if (anchor > 0.0) {
      double mean = 0.0;
      for (int j = 0; j < k; ++j) mean += v[j];
      mean /= k;
      f += 0.5 * mean * mean / (anchor * anchor);
      for (int j = 0; j < k; ++j) g[j] += mean / (anchor * anchor * k);
    }
    // maps
    std::size_t off = process.get_parameter_values().size();
    const std::vector<double> q = process.get_coordinates();
    for (const FRETMeasurement& m : meas) {
      const int km = static_cast<int>(m.get_map_values().size());
      second_differences(off, km, (q.back() - q.front()) / (km - 1), omega_map);
      off += m.get_parameter_values(process).size();
    }
  }
  (void)owner;
  if (grad_full) {
    grad_full->assign(P, 0.0);
    for (std::size_t i = 0; i < P; ++i)
      if (free[i]) (*grad_full)[i] = g[i] * fnd::dv_dz(v[i], transform[i]) + gz[i];
  }
  return f;
}

}  // namespace

double FRETNetworkModel::log_prior(const std::vector<double>& theta) const {
  FRETHiddenProcess p;
  std::vector<FRETMeasurement> m;
  configure(theta, p, m);
  std::vector<double> v = p.get_parameter_values();
  for (const FRETMeasurement& x : m)
    for (double y : x.get_parameter_values(p)) v.push_back(y);
  std::vector<int> own, tr;
  std::vector<char> fr;
  std::vector<double> pm, ps;
  for (const Param& q : params_) {
    own.push_back(q.owner);
    tr.push_back(q.transform);
    fr.push_back(q.free);
    pm.push_back(q.prior_mean);
    ps.push_back(q.prior_sd);
  }
  return -fret_network_neg_log_prior(v, p, m, own, tr, fr, pm, ps, omega_, anchor_, omega_map_,
                                     nullptr);
}

std::vector<double> FRETNetworkModel::log_prior_gradient(const std::vector<double>& theta) const {
  FRETHiddenProcess p;
  std::vector<FRETMeasurement> m;
  configure(theta, p, m);
  std::vector<double> v = p.get_parameter_values();
  for (const FRETMeasurement& x : m)
    for (double y : x.get_parameter_values(p)) v.push_back(y);
  std::vector<int> own, tr;
  std::vector<char> fr;
  std::vector<double> pm, ps;
  for (const Param& q : params_) {
    own.push_back(q.owner);
    tr.push_back(q.transform);
    fr.push_back(q.free);
    pm.push_back(q.prior_mean);
    ps.push_back(q.prior_sd);
  }
  std::vector<double> gf;
  fret_network_neg_log_prior(v, p, m, own, tr, fr, pm, ps, omega_, anchor_, omega_map_, &gf);
  std::vector<double> g;
  for (std::size_t i = 0; i < params_.size(); ++i)
    if (params_[i].free) g.push_back(-gf[i]);
  return g;
}

// --- likelihood and gradient ---------------------------------------------------------

double FRETNetworkModel::evaluate(const std::vector<double>& theta,
                                  std::vector<double>* gradient,
                                  std::vector<double>* scores) const {
  FRETHiddenProcess proc;
  std::vector<FRETMeasurement> meas;
  configure(theta, proc, meas);
  const bool adj = gradient || scores;
  const int nfree = static_cast<int>(theta.size());
  // free parameter j -> global index
  std::vector<int> gidx;
  for (std::size_t i = 0; i < params_.size(); ++i)
    if (params_[i].free) gidx.push_back(static_cast<int>(i));
  if (gradient) gradient->assign(nfree, 0.0);
  std::size_t total_segments = 0;
  for (const FRETPhotonData& d : data_) total_segments += d.get_n_segments();
  if (scores) scores->assign(total_segments * nfree, 0.0);
  double total = 0.0;
  std::size_t seg_row = 0;
  for (std::size_t i = 0; i < meas.size(); ++i) {
    const bool cond = arrival_[i] == FRET_ARRIVAL_CONDITIONAL;
    const fnd::NetOp op = fnd::build_operator(proc, meas[i], cond, detection_start_[i],
                                              joint_start_[i]);
    const FRETPhotonData& d = data_[i];
    const int ns = d.get_n_segments();
    std::vector<fnd::SegmentOut> outs(adj ? ns : 0);
    std::vector<double> ll(ns);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int k = 0; k < ns; ++k) {
      fnd::SegmentOut* o = nullptr;
      if (adj) {
        outs[k].want_adjoint = true;
        o = &outs[k];
      }
      ll[k] = fnd::segment_pass(op, d, k, o);
    }
    for (double x : ll) total += x;
    if (!adj || !std::isfinite(total)) {
      seg_row += ns;
      continue;
    }
    // builders differenced in each free parameter that touches this measurement
    for (int j = 0; j < nfree; ++j) {
      const Param& p = params_[gidx[j]];
      if (p.owner >= 0 && p.owner != static_cast<int>(i)) continue;
      const double h = 1e-5 * std::max(1.0, std::fabs(theta[j]));
      std::vector<double> tp(theta), tm(theta);
      tp[j] += h;
      tm[j] -= h;
      FRETHiddenProcess pp, pm_;
      std::vector<FRETMeasurement> mp, mm;
      configure(tp, pp, mp);
      configure(tm, pm_, mm);
      const fnd::NetOp a = fnd::build_operator(pp, mp[i], cond, detection_start_[i], joint_start_[i]);
      const fnd::NetOp b = fnd::build_operator(pm_, mm[i], cond, detection_start_[i], joint_start_[i]);
      if (a.fi != op.fi || b.fi != op.fi || a.fp != op.fp || b.fp != op.fp)
        IMP_THROW("FRETNetworkModel: parameter " << p.name
                                                 << " changes the generator's sparsity pattern",
                  IMP::ValueException);
      const int n = op.n;
      for (int k = 0; k < ns; ++k) {
        const fnd::SegmentOut& o = outs[k];
        double v = 0.0;
        for (std::size_t e = 0; e < o.gA.size(); ++e) v += o.gA[e] * (a.av[e] - b.av[e]);
        for (int r : o.rows) {
          const std::size_t base = static_cast<std::size_t>(r) * n;
          for (int s = 0; s < n; ++s)
            v += o.gfactor[base + s] * (a.factor[base + s] - b.factor[base + s]);
        }
        for (int s = 0; s < n; ++s) v += o.gstart[s] * (a.start[s] - b.start[s]);
        v /= 2.0 * h;
        if (gradient) (*gradient)[j] += v;
        if (scores) (*scores)[(seg_row + k) * nfree + j] = v;
      }
    }
    seg_row += ns;
  }
  return total;
}

double FRETNetworkModel::log_likelihood_at(const std::vector<double>& theta) const {
  return evaluate(theta, nullptr, nullptr);
}

double FRETNetworkModel::log_posterior(const std::vector<double>& theta) const {
  return log_likelihood_at(theta) + log_prior(theta);
}

std::vector<double> FRETNetworkModel::log_likelihood_gradient(
    const std::vector<double>& theta) const {
  std::vector<double> g;
  evaluate(theta, &g, nullptr);
  return g;
}

std::vector<double> FRETNetworkModel::log_posterior_gradient(
    const std::vector<double>& theta) const {
  std::vector<double> g = log_likelihood_gradient(theta);
  const std::vector<double> gp = log_prior_gradient(theta);
  for (std::size_t i = 0; i < g.size(); ++i) g[i] += gp[i];
  return g;
}

std::vector<double> FRETNetworkModel::segment_scores(const std::vector<double>& theta) const {
  std::vector<double> s;
  evaluate(theta, nullptr, &s);
  return s;
}

// --- precision, fit, Laplace -----------------------------------------------------------

namespace {

//! -d^2 log p / dz^2 by central differences of the prior gradient.
Eigen::MatrixXd fret_network_prior_precision(const FRETNetworkModel& m,
                                             const std::vector<double>& theta) {
  const int n = static_cast<int>(theta.size());
  Eigen::MatrixXd H(n, n);
  for (int j = 0; j < n; ++j) {
    const double h = 1e-5 * std::max(1.0, std::fabs(theta[j]));
    std::vector<double> tp(theta), tm(theta);
    tp[j] += h;
    tm[j] -= h;
    const std::vector<double> a = m.log_prior_gradient(tp), b = m.log_prior_gradient(tm);
    for (int i = 0; i < n; ++i) H(i, j) = -(a[i] - b[i]) / (2.0 * h);
  }
  return 0.5 * (H + H.transpose());
}

Eigen::MatrixXd fret_network_precision(const FRETNetworkModel& m,
                                       const std::vector<double>& theta) {
  const int n = static_cast<int>(theta.size());
  const std::vector<double> sc = m.segment_scores(theta);
  const int ns = n > 0 ? static_cast<int>(sc.size() / n) : 0;
  const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
      S(sc.data(), ns, n);
  Eigen::MatrixXd H = S.transpose() * S;
  H += fret_network_prior_precision(m, theta);
  return H;
}

#if IMP_BFF_HAS_TTTRLIB
struct FRETNetworkFitContext {
  const FRETNetworkModel* model;
  std::vector<double> theta0;
  Eigen::MatrixXd T;
  std::vector<double> theta_of(const double* z) const {
    const int n = static_cast<int>(theta0.size());
    std::vector<double> th(theta0);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) th[i] += T(i, j) * z[j];
    return th;
  }
};

//! The log-posterior, or -inf where the parameters make no valid model.
/*! A line search probes wherever the step takes it: rates driven to zero can
    leave the hidden chain reducible (no unique stationary distribution), and
    the builders then throw. That point is as bad as a zero likelihood, and
    saying so lets the search back off instead of ending the fit. */
double fret_network_safe_log_posterior(const FRETNetworkModel& model,
                                       const std::vector<double>& theta) {
  try {
    return model.log_posterior(theta);
  } catch (const std::exception&) {
    return -std::numeric_limits<double>::infinity();
  }
}

double fret_network_fit_target(double* z, void* p) {
  const FRETNetworkFitContext* c = static_cast<const FRETNetworkFitContext*>(p);
  const double v = -fret_network_safe_log_posterior(*c->model, c->theta_of(z));
  return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
}

double fret_network_fit_gradient(double* z, double* g, void* p) {
  const FRETNetworkFitContext* c = static_cast<const FRETNetworkFitContext*>(p);
  const std::vector<double> th = c->theta_of(z);
  const int n = static_cast<int>(th.size());
  std::vector<double> gp;
  try {
    gp = c->model->log_posterior_gradient(th);
  } catch (const std::exception&) {
    std::fill(g, g + n, 0.0);
    return std::numeric_limits<double>::infinity();
  }
  for (int j = 0; j < n; ++j) {
    double v = 0.0;
    for (int i = 0; i < n; ++i) v -= c->T(i, j) * gp[i];
    g[j] = v;
  }
  const double v = -fret_network_safe_log_posterior(*c->model, th);
  return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
}
#endif

}  // namespace

FRETLandscapeFit FRETNetworkModel::fit(const std::vector<double>& theta0,
                                       const FRETLandscapeFitOptions& options) const {
#if !IMP_BFF_HAS_TTTRLIB
  IMP_THROW("FRETNetworkModel::fit needs tttrlib's L-BFGS (tttrlib/i_lbfgs.h); this IMP.bff "
            "was built without tttrlib",
            IMP::ValueException);
#else
  const int n = static_cast<int>(theta0.size());
  if (options.patience < 1 || options.max_iterations < 1)
    IMP_THROW("fit: patience and max_iterations must be >= 1", IMP::ValueException);
  if (!options.fixed.empty())
    IMP_THROW("FRETNetworkModel::fit: fix parameters with set_parameter_free",
              IMP::ValueException);
  FRETLandscapeFit out;
  std::vector<double> x(theta0);
  double best = log_posterior(x);
  out.history_.push_back(best);
  out.status_ = "max_iterations";
  if (!std::isfinite(best) || n == 0) {
    out.theta_ = x;
    out.log_posterior_ = best;
    out.log_likelihood_ = log_likelihood_at(x);
    out.status_ = std::isfinite(best) ? "converged" : "failed";
    return out;
  }
  FRETNetworkFitContext ctx;
  ctx.model = this;
  int done = 0;
  while (done < options.max_iterations) {
    ctx.theta0 = x;
    ctx.T = Eigen::MatrixXd::Identity(n, n);
    if (options.precondition) {
      Eigen::MatrixXd H = fret_network_precision(*this, x);
      const double ridge =
          std::max(options.precondition_damping, 1e-8 * std::max(1.0, H.diagonal().maxCoeff()));
      for (int i = 0; i < n; ++i) H(i, i) += ridge;
      Eigen::LLT<Eigen::MatrixXd> llt(H);
      if (llt.info() == Eigen::Success)
        ctx.T = llt.matrixU().solve(Eigen::MatrixXd::Identity(n, n));
    }
    const int iters = std::min(options.patience, options.max_iterations - done);
    bfgs opt(fret_network_fit_target, n);
    opt.set_gradient(fret_network_fit_gradient);
    opt.maxiter = iters;
    std::vector<double> z(n, 0.0);
    const int info = opt.minimize(z.data(), &ctx);
    done += iters;
    const std::vector<double> trial = ctx.theta_of(z.data());
    const double v = fret_network_safe_log_posterior(*this, trial);
    const double gain = std::isfinite(v) ? v - best : -1.0;
    if (std::isfinite(v) && v > best) {
      x = trial;
      best = v;
    }
    out.history_.push_back(best);
    if (gain < options.min_delta) {
      out.status_ = (info == 1 || info == 2 || info == 4) ? "converged" : "patience";
      break;
    }
  }
  out.n_iterations_ = done;
  out.theta_ = x;
  out.log_posterior_ = best;
  out.log_likelihood_ = log_likelihood_at(x);
  return out;
#endif
}

FRETNetworkLaplace FRETNetworkModel::laplace(const std::vector<double>& theta) const {
  const int n = static_cast<int>(theta.size());
  const Eigen::MatrixXd H = fret_network_precision(*this, theta);
  const Eigen::MatrixXd S = H.ldlt().solve(Eigen::MatrixXd::Identity(n, n));
  FRETNetworkLaplace out;
  out.names_ = get_free_parameter_names();
  out.theta_ = theta;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      out.precision_.push_back(H(i, j));
      out.covariance_.push_back(S(i, j));
    }
  std::vector<int> gidx;
  for (std::size_t i = 0; i < params_.size(); ++i)
    if (params_[i].free) gidx.push_back(static_cast<int>(i));
  for (int j = 0; j < n; ++j) {
    const int t = params_[gidx[j]].transform;
    const double v = fnd::from_z(theta[j], t), sd = std::sqrt(std::max(0.0, S(j, j)));
    out.sigmas_.push_back(sd);
    out.values_.push_back(v);
    out.natural_sigmas_.push_back(fnd::dv_dz(v, t) * sd);
  }
  FRETHiddenProcess proc;
  std::vector<FRETMeasurement> meas;
  configure(theta, proc, meas);
  // a spline over the q grid with the given free knot indices: value and band
  auto band = [&](const std::vector<double>& knots_values, const std::string& prefix,
                  bool centre, std::vector<double>& value, std::vector<double>& sigma) {
    const std::vector<double> q = proc.get_coordinates();
    const int k = static_cast<int>(knots_values.size()), M = static_cast<int>(q.size());
    const NaturalCubicSpline sp(q.front(), q.back(), k);
    const std::vector<double> phi = sp.get_basis(q);
    Eigen::MatrixXd P(M, k);
    for (int i = 0; i < M; ++i)
      for (int c = 0; c < k; ++c) P(i, c) = phi[static_cast<std::size_t>(i) * k + c];
    if (centre) P.rowwise() -= P.colwise().mean();
    std::vector<int> col(k, -1);
    for (int j = 0; j < n; ++j)
      for (int c = 0; c < k; ++c)
        if (out.names_[j] == prefix + "[" + std::to_string(c) + "]") col[c] = j;
    value = sp.evaluate(knots_values, q);
    if (centre) {
      double mean = 0.0;
      for (double x : value) mean += x;
      mean /= M;
      for (double& x : value) x -= mean;
    }
    sigma.assign(M, 0.0);
    for (int i = 0; i < M; ++i) {
      double s2 = 0.0;
      for (int a = 0; a < k; ++a)
        for (int b = 0; b < k; ++b)
          if (col[a] >= 0 && col[b] >= 0) s2 += P(i, a) * S(col[a], col[b]) * P(i, b);
      sigma[i] = std::sqrt(std::max(0.0, s2));
    }
  };
  out.maps_.resize(meas.size());
  out.map_sigmas_.resize(meas.size());
  if (proc.get_is_landscape()) {
    band(proc.get_knot_heights(), "hidden.mu", true, out.landscape_, out.landscape_sigma_);
    for (std::size_t i = 0; i < meas.size(); ++i)
      band(meas[i].get_map_values(), meas[i].get_name() + ".map", false, out.maps_[i],
           out.map_sigmas_[i]);
  }
  return out;
}

IMPBFF_END_NAMESPACE
