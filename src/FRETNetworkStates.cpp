// SPDX-License-Identifier: BSD-3-Clause
/**
 * The factored state of a FRET network measurement: the shared hidden
 * process, the dyes' photophysics, and their joint generator through
 * KineticNetwork amalgamation. See FRETNetwork.h.
 */
#include <IMP/bff/FRETNetwork.h>
#include <IMP/bff/FRETLandscapeGrid.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

namespace fret_network_detail {

//! A KineticNetwork dense output (malloc'ed view) as a vector.
std::vector<double> take_view(double* p, int n) {
  std::vector<double> v(p, p + n);
  std::free(p);
  return v;
}

std::vector<double> stationary_of(const std::vector<double>& k) {
  if (k.size() == 1) return std::vector<double>(1, 1.0);
  double* p = nullptr;
  int n = 0;
  kinetic_stationary_distribution(k, &p, &n);
  return take_view(p, n);
}

//! Gauss-Hermite nodes and weights (probabilists', normalised) for n = 5.
void gauss_hermite5(std::vector<double>& z, std::vector<double>& w) {
  z = {-2.856970013872806, -1.355626179974266, 0.0, 1.355626179974266, 2.856970013872806};
  w = {0.011257411327721, 0.222075922005613, 0.533333333333333, 0.222075922005613,
       0.011257411327721};
}

}  // namespace fret_network_detail

// --- hidden process ----------------------------------------------------------------

FRETHiddenProcess::FRETHiddenProcess(int n_states) : landscape_(false), n_(n_states) {
  if (n_states < 1) IMP_THROW("FRETHiddenProcess: n_states must be >= 1", IMP::ValueException);
  rates_.assign(static_cast<std::size_t>(n_) * n_, 0.0);
}

FRETHiddenProcess::FRETHiddenProcess(double q_min, double q_max, int n_grid, int n_knots)
    : landscape_(true), n_(n_grid), q_min_(q_min), q_max_(q_max), n_knots_(n_knots) {
  if (n_grid < 2 || n_knots < 2)
    IMP_THROW("FRETHiddenProcess: n_grid and n_knots must be >= 2", IMP::ValueException);
  if (!(q_max > q_min)) IMP_THROW("FRETHiddenProcess: q_max must exceed q_min", IMP::ValueException);
  mu_.assign(n_knots, 0.0);
}

std::vector<double> FRETHiddenProcess::get_coordinates() const {
  std::vector<double> q(n_);
  for (int i = 0; i < n_; ++i)
    q[i] = landscape_ ? q_min_ + (q_max_ - q_min_) * i / (n_ - 1) : static_cast<double>(i);
  return q;
}

void FRETHiddenProcess::set_rate(int source, int target, double rate) {
  if (landscape_) IMP_THROW("set_rate: a landscape has no explicit rates", IMP::ValueException);
  if (source < 0 || target < 0 || source >= n_ || target >= n_ || source == target)
    IMP_THROW("set_rate: bad states " << source << " -> " << target, IMP::ValueException);
  if (!(rate >= 0.0)) IMP_THROW("set_rate: rate must be >= 0", IMP::ValueException);
  rates_[static_cast<std::size_t>(target) * n_ + source] = rate;
  rebuild_rate_index();
}

double FRETHiddenProcess::get_rate(int source, int target) const {
  if (landscape_ || source < 0 || target < 0 || source >= n_ || target >= n_) return 0.0;
  return rates_[static_cast<std::size_t>(target) * n_ + source];
}

void FRETHiddenProcess::rebuild_rate_index() {
  rate_index_.clear();
  for (int s = 0; s < n_; ++s)
    for (int t = 0; t < n_; ++t)
      if (s != t && rates_[static_cast<std::size_t>(t) * n_ + s] > 0.0)
        rate_index_.push_back(std::make_pair(s, t));
}

void FRETHiddenProcess::set_landscape(const std::vector<double>& knot_heights) {
  if (!landscape_ || static_cast<int>(knot_heights.size()) != n_knots_)
    IMP_THROW("set_landscape: expected " << n_knots_ << " knot heights", IMP::ValueException);
  mu_ = knot_heights;
}

std::vector<double> FRETHiddenProcess::get_knots() const {
  if (!landscape_) return std::vector<double>();
  return NaturalCubicSpline(q_min_, q_max_, n_knots_).get_knots();
}

void FRETHiddenProcess::set_diffusion(double diffusion) {
  if (!(diffusion > 0.0)) IMP_THROW("set_diffusion: must be > 0", IMP::ValueException);
  diffusion_ = diffusion;
}

std::vector<double> FRETHiddenProcess::get_landscape() const {
  if (!landscape_) return std::vector<double>();
  return NaturalCubicSpline(q_min_, q_max_, n_knots_).evaluate(mu_, get_coordinates());
}

std::vector<double> FRETHiddenProcess::get_generator() const {
  if (!landscape_) {
    std::vector<double> k(rates_);
    for (int s = 0; s < n_; ++s) {
      double out = 0.0;
      for (int t = 0; t < n_; ++t)
        if (t != s) out += k[static_cast<std::size_t>(t) * n_ + s];
      k[static_cast<std::size_t>(s) * n_ + s] = -out;
    }
    return k;
  }
  return sqra_generator(get_landscape(), diffusion_, (q_max_ - q_min_) / (n_ - 1));
}

std::vector<double> FRETHiddenProcess::get_stationary() const {
  if (landscape_) return sqra_stationary_distribution(get_landscape());
  return fret_network_detail::stationary_of(get_generator());
}

std::vector<std::string> FRETHiddenProcess::get_parameter_names() const {
  std::vector<std::string> n;
  if (landscape_) {
    for (int k = 0; k < n_knots_; ++k) n.push_back("hidden.mu[" + std::to_string(k) + "]");
    n.push_back("hidden.D");
  } else {
    for (const auto& p : rate_index_)
      n.push_back("hidden.rate[" + std::to_string(p.first) + "->" + std::to_string(p.second) + "]");
  }
  return n;
}

std::vector<double> FRETHiddenProcess::get_parameter_values() const {
  std::vector<double> v;
  if (landscape_) {
    v = mu_;
    v.push_back(diffusion_);
  } else {
    for (const auto& p : rate_index_)
      v.push_back(rates_[static_cast<std::size_t>(p.second) * n_ + p.first]);
  }
  return v;
}

void FRETHiddenProcess::set_parameter_values(const std::vector<double>& v) {
  if (v.size() != get_parameter_names().size())
    IMP_THROW("FRETHiddenProcess: wrong number of parameter values", IMP::ValueException);
  if (landscape_) {
    std::copy(v.begin(), v.begin() + n_knots_, mu_.begin());
    diffusion_ = v[n_knots_];
  } else {
    for (std::size_t i = 0; i < rate_index_.size(); ++i)
      rates_[static_cast<std::size_t>(rate_index_[i].second) * n_ + rate_index_[i].first] = v[i];
  }
}

std::vector<int> FRETHiddenProcess::get_parameter_transforms() const {
  std::vector<int> t;
  if (landscape_) {
    t.assign(n_knots_, FRET_TRANSFORM_IDENTITY);
    t.push_back(FRET_TRANSFORM_LOG);
  } else {
    t.assign(rate_index_.size(), FRET_TRANSFORM_LOG);
  }
  return t;
}

std::vector<int> FRETHiddenProcess::get_parameter_kinds() const {
  return std::vector<int>(get_parameter_names().size(),
                          FRET_PARAMETER_GENERATOR | FRET_PARAMETER_START);
}

// --- dye -----------------------------------------------------------------------

FRETDye::FRETDye(const std::string& name) : name_(name) {}

int FRETDye::add_state(const std::string& name, double excitation, double quantum_yield,
                       double lifetime, double accepts) {
  if (get_state_index(name) >= 0)
    IMP_THROW("FRETDye " << name_ << ": state " << name << " exists", IMP::ValueException);
  if (!(excitation >= 0.0) || !(quantum_yield >= 0.0 && quantum_yield <= 1.0) ||
      !(lifetime > 0.0) || !(accepts >= 0.0 && accepts <= 1.0))
    IMP_THROW("FRETDye " << name_ << ": need excitation >= 0, 0 <= quantum_yield <= 1, "
                         "lifetime > 0, 0 <= accepts <= 1",
              IMP::ValueException);
  names_.push_back(name);
  excitation_.push_back(excitation);
  qy_.push_back(quantum_yield);
  lifetime_.push_back(lifetime);
  accepts_.push_back(accepts);
  return static_cast<int>(names_.size()) - 1;
}

int FRETDye::get_state_index(const std::string& name) const {
  for (std::size_t i = 0; i < names_.size(); ++i)
    if (names_[i] == name) return static_cast<int>(i);
  return -1;
}

void FRETDye::set_rate(const std::string& source, const std::string& target, double rate,
                       bool light, int hidden_state) {
  const int s = get_state_index(source), t = get_state_index(target);
  if (s < 0 || t < 0 || s == t)
    IMP_THROW("FRETDye " << name_ << ": bad transition " << source << " -> " << target,
              IMP::ValueException);
  if (!(rate > 0.0)) IMP_THROW("FRETDye: a declared rate must be > 0", IMP::ValueException);
  for (Rate& r : rates_)
    if (r.source == s && r.target == t && r.light == light && r.hidden == hidden_state) {
      r.value = rate;
      return;
    }
  rates_.push_back(Rate{s, t, hidden_state < 0 ? -1 : hidden_state, light, rate});
}

bool FRETDye::get_is_conditioned() const {
  for (const Rate& r : rates_)
    if (r.hidden >= 0) return true;
  return false;
}

void FRETDye::set_initial(const std::vector<double>& weights) {
  if (static_cast<int>(weights.size()) != get_n_states())
    IMP_THROW("FRETDye " << name_ << ": one initial weight per state", IMP::ValueException);
  double s = 0.0;
  for (double w : weights) {
    if (!(w >= 0.0)) IMP_THROW("FRETDye: initial weights must be >= 0", IMP::ValueException);
    s += w;
  }
  if (!(s > 0.0)) IMP_THROW("FRETDye: initial weights sum to zero", IMP::ValueException);
  initial_ = weights;
}

std::vector<double> FRETDye::get_generator(double power, int hidden_state) const {
  const int n = get_n_states();
  std::vector<double> k(static_cast<std::size_t>(n) * n, 0.0);
  for (const Rate& r : rates_) {
    if (r.hidden >= 0 && r.hidden != hidden_state) continue;
    k[static_cast<std::size_t>(r.target) * n + r.source] += r.value * (r.light ? power : 1.0);
  }
  for (int s = 0; s < n; ++s) {
    double out = 0.0;
    for (int t = 0; t < n; ++t)
      if (t != s) out += k[static_cast<std::size_t>(t) * n + s];
    k[static_cast<std::size_t>(s) * n + s] = -out;
  }
  return k;
}

std::vector<double> FRETDye::get_start(double power, int hidden_state) const {
  if (get_n_states() == 0) IMP_THROW("FRETDye " << name_ << " has no states", IMP::ValueException);
  if (!initial_.empty()) {
    std::vector<double> p(initial_);
    double s = 0.0;
    for (double v : p) s += v;
    for (double& v : p) v /= s;
    return p;
  }
  // States without any rate in this hidden state (a PET state that exists
  // only in other conformers) are not part of the chain here: start at zero.
  const int n = get_n_states();
  const std::vector<double> k = get_generator(power, hidden_state);
  std::vector<int> live;
  for (int s = 0; s < n; ++s) {
    bool any = false;
    for (int t = 0; t < n && !any; ++t)
      any = t != s && (k[static_cast<std::size_t>(t) * n + s] > 0.0 ||
                       k[static_cast<std::size_t>(s) * n + t] > 0.0);
    if (any) live.push_back(s);
  }
  std::vector<double> p(n, 0.0);
  if (live.empty()) {
    if (n != 1)
      IMP_THROW("FRETDye " << name_ << ": no rates, so no stationary start; set_initial",
                IMP::ValueException);
    p[0] = 1.0;
    return p;
  }
  const int m = static_cast<int>(live.size());
  std::vector<double> sub(static_cast<std::size_t>(m) * m);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j)
      sub[static_cast<std::size_t>(i) * m + j] = k[static_cast<std::size_t>(live[i]) * n + live[j]];
  const std::vector<double> ps = fret_network_detail::stationary_of(sub);
  for (int i = 0; i < m; ++i) p[live[i]] = ps[i];
  return p;
}

std::vector<std::string> FRETDye::get_parameter_names() const {
  std::vector<std::string> n;
  const std::string p = name_ + ".";
  for (const std::string& s : names_) {
    n.push_back(p + "lifetime[" + s + "]");
    n.push_back(p + "quantum_yield[" + s + "]");
    n.push_back(p + "excitation[" + s + "]");
    n.push_back(p + "accepts[" + s + "]");
  }
  for (const Rate& r : rates_) {
    std::string nm = p + "rate[" + names_[r.source] + "->" + names_[r.target] + "]";
    if (r.light) nm += "|light";
    if (r.hidden >= 0) nm += "@" + std::to_string(r.hidden);
    n.push_back(nm);
  }
  if (!initial_.empty())
    for (const std::string& s : names_) n.push_back(p + "initial[" + s + "]");
  return n;
}

std::vector<double> FRETDye::get_parameter_values() const {
  std::vector<double> v;
  for (std::size_t i = 0; i < names_.size(); ++i) {
    v.push_back(lifetime_[i]);
    v.push_back(qy_[i]);
    v.push_back(excitation_[i]);
    v.push_back(accepts_[i]);
  }
  for (const Rate& r : rates_) v.push_back(r.value);
  for (double w : initial_) v.push_back(w);
  return v;
}

void FRETDye::set_parameter_values(const std::vector<double>& v) {
  if (v.size() != get_parameter_names().size())
    IMP_THROW("FRETDye " << name_ << ": wrong number of parameter values", IMP::ValueException);
  std::size_t j = 0;
  for (std::size_t i = 0; i < names_.size(); ++i) {
    lifetime_[i] = v[j++];
    qy_[i] = v[j++];
    excitation_[i] = v[j++];
    accepts_[i] = v[j++];
  }
  for (Rate& r : rates_) r.value = v[j++];
  for (double& w : initial_) w = v[j++];
}

std::vector<int> FRETDye::get_parameter_transforms() const {
  std::vector<int> t;
  for (std::size_t i = 0; i < names_.size(); ++i) {
    t.push_back(FRET_TRANSFORM_LOG);
    t.push_back(FRET_TRANSFORM_LOGIT);
    t.push_back(FRET_TRANSFORM_LOG);
    t.push_back(FRET_TRANSFORM_LOGIT);
  }
  for (std::size_t i = 0; i < rates_.size(); ++i) t.push_back(FRET_TRANSFORM_LOG);
  for (std::size_t i = 0; i < initial_.size(); ++i) t.push_back(FRET_TRANSFORM_LOG);
  return t;
}

std::vector<int> FRETDye::get_parameter_kinds() const {
  std::vector<int> k;
  for (std::size_t i = 0; i < names_.size(); ++i)
    for (int j = 0; j < 4; ++j) k.push_back(FRET_PARAMETER_EMISSION);
  const int rate_kind = FRET_PARAMETER_GENERATOR | (initial_.empty() ? FRET_PARAMETER_START : 0);
  for (std::size_t i = 0; i < rates_.size(); ++i) k.push_back(rate_kind);
  for (std::size_t i = 0; i < initial_.size(); ++i) k.push_back(FRET_PARAMETER_START);
  return k;
}

// --- measurement -----------------------------------------------------------------

FRETMeasurement::FRETMeasurement(const std::string& name, const FRETDye& donor,
                                 const FRETDye& acceptor, double forster_radius)
    : name_(name), donor_(donor), acceptor_(acceptor) {
  set_forster_radius(forster_radius);
}

void FRETMeasurement::set_forster_radius(double r0) {
  if (!(r0 > 0.0)) IMP_THROW("FRETMeasurement: forster_radius must be > 0", IMP::ValueException);
  r0_ = r0;
}

double FRETMeasurement::get_reference_lifetime() const {
  if (tau0_ > 0.0) return tau0_;
  if (donor_.get_n_states() == 0) return 4.0;
  return donor_.get_lifetimes()[0];
}

void FRETMeasurement::set_power(double power) {
  if (!(power >= 0.0)) IMP_THROW("set_power: must be >= 0", IMP::ValueException);
  power_ = power;
}

void FRETMeasurement::set_state_distance(int state, double mean,
                                         const std::vector<double>& offsets,
                                         const std::vector<double>& weights) {
  if (state < 0) IMP_THROW("set_state_distance: state must be >= 0", IMP::ValueException);
  if (offsets.size() != weights.size())
    IMP_THROW("set_state_distance: offsets and weights differ in length", IMP::ValueException);
  if (static_cast<int>(means_.size()) <= state) {
    means_.resize(state + 1, 0.0);
    offsets_.resize(state + 1);
    weights_.resize(state + 1);
  }
  means_[state] = mean;
  double s = 0.0;
  for (double w : weights) {
    if (!(w >= 0.0)) IMP_THROW("set_state_distance: weights must be >= 0", IMP::ValueException);
    s += w;
  }
  offsets_[state] = offsets;
  weights_[state] = weights;
  if (s > 0.0)
    for (double& w : weights_[state]) w /= s;
}

void FRETMeasurement::set_distance_map(const std::vector<double>& knot_values, double spread) {
  if (knot_values.size() < 2)
    IMP_THROW("set_distance_map: at least two knots", IMP::ValueException);
  if (!(spread >= 0.0)) IMP_THROW("set_distance_map: spread must be >= 0", IMP::ValueException);
  map_ = knot_values;
  spread_ = spread;
}

void FRETMeasurement::check_process(const FRETHiddenProcess& process) const {
  if (donor_.get_n_states() == 0 || acceptor_.get_n_states() == 0)
    IMP_THROW("FRETMeasurement " << name_ << ": both dyes need at least one state",
              IMP::ValueException);
  if (process.get_is_landscape()) {
    if (map_.size() < 2)
      IMP_THROW("FRETMeasurement " << name_ << ": set_distance_map for a landscape process",
                IMP::ValueException);
    if (donor_.get_is_conditioned() || acceptor_.get_is_conditioned())
      IMP_THROW("FRETMeasurement " << name_
                                   << ": dye rates conditioned on a landscape grid point are "
                                      "not supported; condition on discrete conformers",
                IMP::ValueException);
  } else if (static_cast<int>(means_.size()) != process.get_n_states()) {
    IMP_THROW("FRETMeasurement " << name_ << ": set_state_distance for each of the "
                                 << process.get_n_states() << " conformers",
              IMP::ValueException);
  }
}

std::vector<double> FRETMeasurement::get_state_distances(const FRETHiddenProcess& process,
                                                         int h) const {
  check_process(process);
  if (h < 0 || h >= process.get_n_states())
    IMP_THROW("get_state_distances: no hidden state " << h, IMP::ValueException);
  std::vector<double> out;
  if (process.get_is_landscape()) {
    const std::vector<double> q = process.get_coordinates();
    const NaturalCubicSpline sp(q.front(), q.back(), static_cast<int>(map_.size()));
    const double r = sp.evaluate(map_, std::vector<double>(1, q[h]))[0];
    if (spread_ > 0.0) {
      std::vector<double> z, w;
      fret_network_detail::gauss_hermite5(z, w);
      for (std::size_t j = 0; j < z.size(); ++j) {
        out.push_back(r + spread_ * z[j]);
        out.push_back(w[j]);
      }
    } else {
      out = {r, 1.0};
    }
  } else if (offsets_[h].empty()) {
    out = {means_[h], 1.0};
  } else {
    for (std::size_t j = 0; j < offsets_[h].size(); ++j) {
      out.push_back(means_[h] + offsets_[h][j]);
      out.push_back(weights_[h][j]);
    }
  }
  return out;
}

KineticNetwork FRETMeasurement::get_network(const FRETHiddenProcess& process) const {
  check_process(process);
  KineticNetwork net;
  const int nh = process.get_n_states(), nd = donor_.get_n_states(),
            na = acceptor_.get_n_states();
  net.add_variable("hidden", nh);
  net.add_variable("donor", nd);
  net.add_variable("acceptor", na);
  const std::vector<double> kh = process.get_generator();
  for (int s = 0; s < nh; ++s)
    for (int t = 0; t < nh; ++t)
      if (s != t && kh[static_cast<std::size_t>(t) * nh + s] > 0.0)
        net.set_rate("hidden", s, t, kh[static_cast<std::size_t>(t) * nh + s]);
  const FRETDye* dyes[2] = {&donor_, &acceptor_};
  const char* names[2] = {"donor", "acceptor"};
  for (int d = 0; d < 2; ++d) {
    const int n = dyes[d]->get_n_states();
    if (dyes[d]->get_is_conditioned()) {
      net.add_arc("hidden", names[d]);
      for (int h = 0; h < nh; ++h) {
        const std::vector<double> k = dyes[d]->get_generator(power_, h);
        for (int s = 0; s < n; ++s)
          for (int t = 0; t < n; ++t)
            if (s != t && k[static_cast<std::size_t>(t) * n + s] > 0.0)
              net.set_rate(names[d], s, t, k[static_cast<std::size_t>(t) * n + s],
                           std::vector<int>(1, h));
      }
    } else {
      const std::vector<double> k = dyes[d]->get_generator(power_, -1);
      for (int s = 0; s < n; ++s)
        for (int t = 0; t < n; ++t)
          if (s != t && k[static_cast<std::size_t>(t) * n + s] > 0.0)
            net.set_rate(names[d], s, t, k[static_cast<std::size_t>(t) * n + s]);
    }
  }
  return net;
}

std::vector<double> FRETMeasurement::get_generator(const FRETHiddenProcess& process) const {
  const KineticNetwork net = get_network(process);
  double* p = nullptr;
  int n = 0;
  net.get_generator(&p, &n);
  return fret_network_detail::take_view(p, n);
}

std::vector<double> FRETMeasurement::get_start(const FRETHiddenProcess& process) const {
  check_process(process);
  const std::vector<double> ph = process.get_stationary();
  const int nh = process.get_n_states(), nd = donor_.get_n_states(),
            na = acceptor_.get_n_states();
  std::vector<double> p(static_cast<std::size_t>(nh) * nd * na, 0.0);
  std::vector<double> pd = donor_.get_start(power_, -1), pa = acceptor_.get_start(power_, -1);
  for (int h = 0; h < nh; ++h) {
    if (donor_.get_is_conditioned() && donor_.get_initial_is_stationary())
      pd = donor_.get_start(power_, h);
    if (acceptor_.get_is_conditioned() && acceptor_.get_initial_is_stationary())
      pa = acceptor_.get_start(power_, h);
    for (int d = 0; d < nd; ++d)
      for (int a = 0; a < na; ++a)
        p[(static_cast<std::size_t>(h) * nd + d) * na + a] = ph[h] * pd[d] * pa[a];
  }
  return p;
}

std::vector<double> FRETMeasurement::get_joint_stationary(
    const FRETHiddenProcess& process) const {
  return fret_network_detail::stationary_of(get_generator(process));
}

int FRETMeasurement::get_n_states(const FRETHiddenProcess& process) const {
  return process.get_n_states() * donor_.get_n_states() * acceptor_.get_n_states();
}

std::vector<int> FRETMeasurement::get_state(const FRETHiddenProcess& process, int s) const {
  const int nd = donor_.get_n_states(), na = acceptor_.get_n_states();
  if (s < 0 || s >= get_n_states(process))
    IMP_THROW("get_state: no joint state " << s, IMP::ValueException);
  return {s / (nd * na), (s / na) % nd, s % na};
}

std::vector<std::string> FRETMeasurement::get_parameter_names(
    const FRETHiddenProcess& process) const {
  check_process(process);
  std::vector<std::string> n;
  const std::string p = name_ + ".";
  if (process.get_is_landscape()) {
    for (std::size_t k = 0; k < map_.size(); ++k) n.push_back(p + "map[" + std::to_string(k) + "]");
    n.push_back(p + "spread");
  } else {
    for (std::size_t k = 0; k < means_.size(); ++k)
      n.push_back(p + "mean[" + std::to_string(k) + "]");
  }
  n.push_back(p + "forster_radius");
  n.push_back(p + "reference_lifetime");
  for (const std::string& s : donor_.get_parameter_names()) n.push_back(p + s);
  for (const std::string& s : acceptor_.get_parameter_names()) n.push_back(p + s);
  return n;
}

std::vector<double> FRETMeasurement::get_parameter_values(
    const FRETHiddenProcess& process) const {
  check_process(process);
  std::vector<double> v;
  if (process.get_is_landscape()) {
    v = map_;
    v.push_back(spread_);
  } else {
    v = means_;
  }
  v.push_back(r0_);
  v.push_back(get_reference_lifetime());
  for (double x : donor_.get_parameter_values()) v.push_back(x);
  for (double x : acceptor_.get_parameter_values()) v.push_back(x);
  return v;
}

void FRETMeasurement::set_parameter_values(const FRETHiddenProcess& process,
                                           const std::vector<double>& v) {
  if (v.size() != get_parameter_names(process).size())
    IMP_THROW("FRETMeasurement " << name_ << ": wrong number of parameter values",
              IMP::ValueException);
  std::size_t j = 0;
  if (process.get_is_landscape()) {
    for (double& x : map_) x = v[j++];
    spread_ = v[j++];
  } else {
    for (double& x : means_) x = v[j++];
  }
  r0_ = v[j++];
  tau0_ = v[j++];
  const std::size_t nd = donor_.get_parameter_names().size();
  donor_.set_parameter_values(std::vector<double>(v.begin() + j, v.begin() + j + nd));
  j += nd;
  acceptor_.set_parameter_values(std::vector<double>(v.begin() + j, v.end()));
}

std::vector<int> FRETMeasurement::get_parameter_transforms(
    const FRETHiddenProcess& process) const {
  check_process(process);
  std::vector<int> t;
  if (process.get_is_landscape()) {
    t.assign(map_.size(), FRET_TRANSFORM_IDENTITY);
    t.push_back(FRET_TRANSFORM_LOG);
  } else {
    t.assign(means_.size(), FRET_TRANSFORM_IDENTITY);
  }
  t.push_back(FRET_TRANSFORM_LOG);
  t.push_back(FRET_TRANSFORM_LOG);
  for (int x : donor_.get_parameter_transforms()) t.push_back(x);
  for (int x : acceptor_.get_parameter_transforms()) t.push_back(x);
  return t;
}

std::vector<int> FRETMeasurement::get_parameter_kinds(const FRETHiddenProcess& process) const {
  check_process(process);
  std::vector<int> k(process.get_is_landscape() ? map_.size() + 1 : means_.size(),
                     FRET_PARAMETER_EMISSION);
  k.push_back(FRET_PARAMETER_EMISSION);
  k.push_back(FRET_PARAMETER_EMISSION);
  for (int x : donor_.get_parameter_kinds()) k.push_back(x);
  for (int x : acceptor_.get_parameter_kinds()) k.push_back(x);
  return k;
}

IMPBFF_END_NAMESPACE
