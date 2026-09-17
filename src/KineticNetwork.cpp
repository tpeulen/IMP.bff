/**
 * \file KineticNetwork.cpp
 * \brief Conditional intensity matrices, amalgamation and exact inference.
 *
 * Port of aGrUM's pyagrum.ctbn (CIM.py, CTBN.py, CTBNInference.py; dual
 * LGPL-3.0-or-later / MIT). See KineticNetwork.h for the conventions.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/KineticNetwork.h>

#include <IMP/bff/IMPCompatibility.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

IMPBFF_BEGIN_NAMESPACE

namespace {

void kinetic_emit(const std::vector<double>& values, double** out_view,
          int* n_out_view) {
  const std::size_t bytes = std::max<std::size_t>(1, values.size()) * sizeof(double);
  double* buffer = static_cast<double*>(std::malloc(bytes));
  if (buffer == nullptr) {
    IMP_THROW("out of memory for " << values.size() << " values",
              ValueException);
  }
  if (!values.empty()) std::memcpy(buffer, values.data(), values.size() * sizeof(double));
  *out_view = buffer;
  *n_out_view = static_cast<int>(values.size());
}

int kinetic_product(const std::vector<int>& sizes) {
  long long n = 1;
  for (int s : sizes) {
    n *= s;
    if (n > 100000000LL) {
      IMP_THROW("the joint state space is too large (" << n << " states)",
                ValueException);
    }
  }
  return static_cast<int>(n);
}

//! C-order index of \p values over \p sizes.
int kinetic_ravel(const std::vector<int>& values, const std::vector<int>& sizes) {
  if (values.size() != sizes.size()) {
    IMP_THROW("expected " << sizes.size() << " state values, got "
                          << values.size(),
              ValueException);
  }
  int index = 0;
  for (std::size_t k = 0; k < sizes.size(); ++k) {
    if (values[k] < 0 || values[k] >= sizes[k]) {
      IMP_THROW("state " << values[k] << " out of range [0, " << sizes[k]
                         << ")",
                ValueException);
    }
    index = index * sizes[k] + values[k];
  }
  return index;
}

std::vector<int> kinetic_unravel(int index, const std::vector<int>& sizes) {
  std::vector<int> values(sizes.size(), 0);
  for (std::size_t k = sizes.size(); k-- > 0;) {
    values[k] = index % sizes[k];
    index /= sizes[k];
  }
  return values;
}

int kinetic_find(const std::vector<std::string>& names, const std::string& name) {
  for (std::size_t k = 0; k < names.size(); ++k) {
    if (names[k] == name) return static_cast<int>(k);
  }
  return -1;
}

Eigen::MatrixXd kinetic_generator_matrix(const std::vector<double>& generator) {
  const double root = std::sqrt(static_cast<double>(generator.size()));
  const int n = static_cast<int>(std::lround(root));
  if (n < 1 || static_cast<std::size_t>(n) * n != generator.size()) {
    IMP_THROW("the generator must be square, not " << generator.size()
                                                   << " values",
              ValueException);
  }
  Eigen::MatrixXd k(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) k(i, j) = generator[static_cast<std::size_t>(i) * n + j];
  }
  // K[target, source]: the diagonal is minus each column's off-diagonal sum.
  k.diagonal().setZero();
  const Eigen::VectorXd exit = k.colwise().sum().transpose();
  k.diagonal() = -exit;
  return k;
}

}  // namespace

// ---------------------------------------------------------------------------
// KineticIntensityMatrix

KineticIntensityMatrix::KineticIntensityMatrix()
    : n_(1), n_configurations_(1), rates_(1, 0.0) {}

KineticIntensityMatrix::KineticIntensityMatrix(
        const std::string& variable, int n_states,
        const std::vector<std::string>& parents,
        const std::vector<int>& parent_states) {
  if (variable.empty()) {
    IMP_THROW("a variable needs a name", ValueException);
  }
  if (n_states < 1) {
    IMP_THROW("variable " << variable << " needs at least one state, not "
                          << n_states,
              ValueException);
  }
  if (parents.size() != parent_states.size()) {
    IMP_THROW("one state count per parent: " << parents.size() << " parents, "
                                             << parent_states.size()
                                             << " counts",
              ValueException);
  }
  for (std::size_t k = 0; k < parents.size(); ++k) {
    if (parents[k] == variable) {
      IMP_THROW("variable " << variable << " cannot be its own parent",
                ValueException);
    }
    if (kinetic_find(std::vector<std::string>(parents.begin(), parents.begin() + k),
             parents[k]) >= 0) {
      IMP_THROW("parent " << parents[k] << " given twice", ValueException);
    }
    if (parent_states[k] < 1) {
      IMP_THROW("parent " << parents[k] << " needs at least one state",
                ValueException);
    }
  }
  variables_.push_back(variable);
  variable_states_.push_back(n_states);
  parents_ = parents;
  parent_states_ = parent_states;
  n_ = n_states;
  n_configurations_ = kinetic_product(parent_states_);
  rates_.assign(static_cast<std::size_t>(n_configurations_) * n_ * n_, 0.0);
}

std::size_t KineticIntensityMatrix::at(int configuration, int target,
                                       int source) const {
  return (static_cast<std::size_t>(configuration) * n_ + target) * n_ + source;
}

void KineticIntensityMatrix::check_configuration(int c) const {
  if (c < 0 || c >= n_configurations_) {
    IMP_THROW("parent configuration " << c << " out of range [0, "
                                      << n_configurations_ << ")",
              ValueException);
  }
}

void KineticIntensityMatrix::check_state(int s) const {
  if (s < 0 || s >= n_) {
    IMP_THROW("state " << s << " out of range [0, " << n_ << ")",
              ValueException);
  }
}

int KineticIntensityMatrix::variable_position(const std::string& name) const {
  return kinetic_find(variables_, name);
}

int KineticIntensityMatrix::parent_position(const std::string& name) const {
  return kinetic_find(parents_, name);
}

int KineticIntensityMatrix::get_state_index(const std::vector<int>& states) const {
  return kinetic_ravel(states, variable_states_);
}

std::vector<int> KineticIntensityMatrix::get_states(int index) const {
  check_state(index);
  return kinetic_unravel(index, variable_states_);
}

int KineticIntensityMatrix::get_parent_configuration(
        const std::vector<int>& parent_values) const {
  return kinetic_ravel(parent_values, parent_states_);
}

void KineticIntensityMatrix::set_rate(int source, int target, double rate,
                                      int parent_configuration) {
  check_state(source);
  check_state(target);
  check_configuration(parent_configuration);
  if (source == target) {
    IMP_THROW("the diagonal is derived from the exit rates; set the rates "
              "out of state " << source << " instead",
              ValueException);
  }
  if (!(rate >= 0.0) || !std::isfinite(rate)) {
    IMP_THROW("a rate must be finite and >= 0, not " << rate, ValueException);
  }
  rates_[at(parent_configuration, target, source)] = rate;
}

double KineticIntensityMatrix::get_rate(int source, int target,
                                        int parent_configuration) const {
  check_state(source);
  check_state(target);
  check_configuration(parent_configuration);
  if (source != target) return rates_[at(parent_configuration, target, source)];
  double exit = 0.0;
  for (int t = 0; t < n_; ++t) {
    if (t != source) exit += rates_[at(parent_configuration, t, source)];
  }
  return -exit;
}

void KineticIntensityMatrix::set_matrix(const std::vector<double>& matrix,
                                        int parent_configuration) {
  check_configuration(parent_configuration);
  if (matrix.size() != static_cast<std::size_t>(n_) * n_) {
    IMP_THROW("expected " << n_ << " x " << n_ << " rates, got "
                          << matrix.size(),
              ValueException);
  }
  for (int t = 0; t < n_; ++t) {
    for (int s = 0; s < n_; ++s) {
      if (s != t) set_rate(s, t, matrix[static_cast<std::size_t>(t) * n_ + s], parent_configuration);
    }
  }
}

void KineticIntensityMatrix::get_matrix(int parent_configuration,
                                        double** out_view,
                                        int* n_out_view) const {
  check_configuration(parent_configuration);
  std::vector<double> out(static_cast<std::size_t>(n_) * n_, 0.0);
  for (int t = 0; t < n_; ++t) {
    for (int s = 0; s < n_; ++s) {
      out[static_cast<std::size_t>(t) * n_ + s] = get_rate(s, t, parent_configuration);
    }
  }
  kinetic_emit(out, out_view, n_out_view);
}

void KineticIntensityMatrix::get_generator(double** out_view,
                                           int* n_out_view) const {
  if (get_is_conditional()) {
    IMP_THROW("the CIM is conditional: parents remain (first: "
                      << parents_.front() << "); amalgamate or extract them",
              ValueException);
  }
  get_matrix(0, out_view, n_out_view);
}

KineticIntensityMatrix KineticIntensityMatrix::extract(
        const std::vector<std::string>& parents,
        const std::vector<int>& values) const {
  if (parents.size() != values.size()) {
    IMP_THROW("one value per fixed parent: " << parents.size() << " names, "
                                             << values.size() << " values",
              ValueException);
  }
  std::vector<int> fixed(parents_.size(), -1);
  for (std::size_t k = 0; k < parents.size(); ++k) {
    const int p = parent_position(parents[k]);
    if (p < 0) {
      IMP_THROW(parents[k] << " is not a parent of this CIM", ValueException);
    }
    if (values[k] < 0 || values[k] >= parent_states_[p]) {
      IMP_THROW("parent " << parents[k] << " has no state " << values[k],
                ValueException);
    }
    fixed[p] = values[k];
  }
  KineticIntensityMatrix out;
  out.variables_ = variables_;
  out.variable_states_ = variable_states_;
  for (std::size_t p = 0; p < parents_.size(); ++p) {
    if (fixed[p] < 0) {
      out.parents_.push_back(parents_[p]);
      out.parent_states_.push_back(parent_states_[p]);
    }
  }
  out.n_ = n_;
  out.n_configurations_ = kinetic_product(out.parent_states_);
  out.rates_.assign(static_cast<std::size_t>(out.n_configurations_) * n_ * n_, 0.0);
  for (int c = 0; c < out.n_configurations_; ++c) {
    const std::vector<int> free_values = kinetic_unravel(c, out.parent_states_);
    std::vector<int> full(parents_.size());
    std::size_t next = 0;
    for (std::size_t p = 0; p < parents_.size(); ++p) {
      full[p] = fixed[p] >= 0 ? fixed[p] : free_values[next++];
    }
    const int source_configuration = kinetic_ravel(full, parent_states_);
    std::copy(rates_.begin() + at(source_configuration, 0, 0),
              rates_.begin() + at(source_configuration, 0, 0) +
                      static_cast<std::ptrdiff_t>(n_) * n_,
              out.rates_.begin() + out.at(c, 0, 0));
  }
  return out;
}

KineticIntensityMatrix KineticIntensityMatrix::amalgamate(
        const KineticIntensityMatrix& other) const {
  // CIM.py: an empty operand returns the other.
  if (variables_.empty()) return other;
  if (other.variables_.empty()) return *this;
  const KineticIntensityMatrix& x = *this;
  const KineticIntensityMatrix& y = other;
  for (const std::string& v : y.variables_) {
    if (x.variable_position(v) >= 0) {
      IMP_THROW("both CIMs carry variable " << v
                                            << "; amalgamation joins "
                                               "disjoint variables",
                ValueException);
    }
  }

  KineticIntensityMatrix out;
  out.variables_ = x.variables_;
  out.variables_.insert(out.variables_.end(), y.variables_.begin(), y.variables_.end());
  out.variable_states_ = x.variable_states_;
  out.variable_states_.insert(out.variable_states_.end(), y.variable_states_.begin(),
                              y.variable_states_.end());
  // Parents that are not variables of the other operand remain parents;
  // CIM.py adds X's then Y's (a parent both share is added once).
  for (std::size_t p = 0; p < x.parents_.size(); ++p) {
    if (y.variable_position(x.parents_[p]) < 0) {
      out.parents_.push_back(x.parents_[p]);
      out.parent_states_.push_back(x.parent_states_[p]);
    }
  }
  for (std::size_t p = 0; p < y.parents_.size(); ++p) {
    if (x.variable_position(y.parents_[p]) >= 0) continue;
    const int already = kinetic_find(out.parents_, y.parents_[p]);
    if (already >= 0) {
      if (out.parent_states_[already] != y.parent_states_[p]) {
        IMP_THROW("parent " << y.parents_[p]
                            << " has a different number of states in the two "
                               "CIMs",
                  ValueException);
      }
      continue;
    }
    out.parents_.push_back(y.parents_[p]);
    out.parent_states_.push_back(y.parent_states_[p]);
  }
  // A parent of one operand that is a variable of the other must agree on
  // its number of states.
  for (std::size_t p = 0; p < x.parents_.size(); ++p) {
    const int v = y.variable_position(x.parents_[p]);
    if (v >= 0 && y.variable_states_[v] != x.parent_states_[p]) {
      IMP_THROW("parent " << x.parents_[p] << " has " << x.parent_states_[p]
                          << " states, the variable " << y.variable_states_[v],
                ValueException);
    }
  }
  for (std::size_t p = 0; p < y.parents_.size(); ++p) {
    const int v = x.variable_position(y.parents_[p]);
    if (v >= 0 && x.variable_states_[v] != y.parent_states_[p]) {
      IMP_THROW("parent " << y.parents_[p] << " has " << y.parent_states_[p]
                          << " states, the variable " << x.variable_states_[v],
                ValueException);
    }
  }
  out.n_ = x.n_ * y.n_;
  out.n_configurations_ = kinetic_product(out.parent_states_);
  out.rates_.assign(static_cast<std::size_t>(out.n_configurations_) * out.n_ * out.n_, 0.0);

  // Where each operand's parent value comes from: a result parent (>= 0), or
  // a variable of the other operand at its source state (-1 - position).
  auto sources_of = [&](const KineticIntensityMatrix& self,
                        const KineticIntensityMatrix& partner) {
    std::vector<int> from(self.parents_.size());
    for (std::size_t p = 0; p < self.parents_.size(); ++p) {
      const int v = partner.variable_position(self.parents_[p]);
      from[p] = v >= 0 ? -1 - v : kinetic_find(out.parents_, self.parents_[p]);
    }
    return from;
  };
  const std::vector<int> x_from = sources_of(x, y);
  const std::vector<int> y_from = sources_of(y, x);

  std::vector<int> x_parent(x.parents_.size()), y_parent(y.parents_.size());
  for (int c = 0; c < out.n_configurations_; ++c) {
    const std::vector<int> result_parents = kinetic_unravel(c, out.parent_states_);
    for (int xs = 0; xs < x.n_; ++xs) {
      const std::vector<int> x_states = kinetic_unravel(xs, x.variable_states_);
      for (int ys = 0; ys < y.n_; ++ys) {
        const std::vector<int> y_states = kinetic_unravel(ys, y.variable_states_);
        for (std::size_t p = 0; p < x_from.size(); ++p) {
          x_parent[p] = x_from[p] >= 0 ? result_parents[x_from[p]] : y_states[-1 - x_from[p]];
        }
        for (std::size_t p = 0; p < y_from.size(); ++p) {
          y_parent[p] = y_from[p] >= 0 ? result_parents[y_from[p]] : x_states[-1 - y_from[p]];
        }
        const int xc = x.parents_.empty() ? 0 : kinetic_ravel(x_parent, x.parent_states_);
        const int yc = y.parents_.empty() ? 0 : kinetic_ravel(y_parent, y.parent_states_);
        const int source = xs * y.n_ + ys;
        // Only X changes (y unchanged): X's rate given X's parents.
        for (int xt = 0; xt < x.n_; ++xt) {
          if (xt == xs) continue;
          out.rates_[out.at(c, xt * y.n_ + ys, source)] = x.rates_[x.at(xc, xt, xs)];
        }
        // Only Y changes (x unchanged).
        for (int yt = 0; yt < y.n_; ++yt) {
          if (yt == ys) continue;
          out.rates_[out.at(c, xs * y.n_ + yt, source)] = y.rates_[y.at(yc, yt, ys)];
        }
        // Both unchanged: the diagonal, Q_X(x->x) + Q_Y(y->y), is derived
        // from the two fills above, which is that sum. Both changing: 0.
      }
    }
  }
  return out;
}

void KineticIntensityMatrix::get_marginal(const std::vector<double>& joint,
                                          const std::string& variable,
                                          double** out_view,
                                          int* n_out_view) const {
  if (joint.size() != static_cast<std::size_t>(n_)) {
    IMP_THROW("the joint distribution has " << joint.size()
                                            << " entries, the CIM " << n_
                                            << " states",
              ValueException);
  }
  const int v = variable_position(variable);
  if (v < 0) {
    IMP_THROW(variable << " is not a variable of this CIM", ValueException);
  }
  std::vector<double> out(variable_states_[v], 0.0);
  for (int s = 0; s < n_; ++s) out[kinetic_unravel(s, variable_states_)[v]] += joint[s];
  kinetic_emit(out, out_view, n_out_view);
}

// ---------------------------------------------------------------------------
// KineticNetwork

KineticNetwork::KineticNetwork() {}

int KineticNetwork::position(const std::string& name) const {
  for (std::size_t k = 0; k < cims_.size(); ++k) {
    if (cims_[k].get_variables().front() == name) return static_cast<int>(k);
  }
  IMP_THROW(name << " is not a variable of the network", ValueException);
  return -1;
}

int KineticNetwork::add_variable(const std::string& name, int n_states) {
  for (const KineticIntensityMatrix& c : cims_) {
    if (c.get_variables().front() == name) {
      IMP_THROW("a variable named " << name << " already exists",
                ValueException);
    }
  }
  cims_.push_back(KineticIntensityMatrix(name, n_states));
  return static_cast<int>(cims_.size()) - 1;
}

void KineticNetwork::add_arc(const std::string& parent,
                             const std::string& child) {
  const int p = position(parent);
  const int c = position(child);
  const KineticIntensityMatrix& old = cims_[c];
  std::vector<std::string> parents = old.get_parents();
  std::vector<int> parent_states = old.get_parent_states();
  if (kinetic_find(parents, parent) >= 0) {
    IMP_THROW(parent << " is already a parent of " << child, ValueException);
  }
  parents.push_back(parent);
  parent_states.push_back(cims_[p].get_variable_states().front());
  KineticIntensityMatrix updated(child, old.get_number_of_states(), parents,
                                 parent_states);
  const int k = parent_states.back();
  const int n = old.get_number_of_states();
  for (int oc = 0; oc < old.get_number_of_parent_configurations(); ++oc) {
    for (int value = 0; value < k; ++value) {
      const int nc = oc * k + value;  // the new parent is least significant
      for (int t = 0; t < n; ++t) {
        for (int s = 0; s < n; ++s) {
          if (s != t) updated.set_rate(s, t, old.get_rate(s, t, oc), nc);
        }
      }
    }
  }
  cims_[c] = updated;
}

void KineticNetwork::set_rate(const std::string& variable, int source,
                              int target, double rate,
                              const std::vector<int>& parent_values) {
  KineticIntensityMatrix& cim = cims_[position(variable)];
  cim.set_rate(source, target, rate,
               cim.get_parent_configuration(parent_values));
}

double KineticNetwork::get_rate(const std::string& variable, int source,
                                int target,
                                const std::vector<int>& parent_values) const {
  const KineticIntensityMatrix& cim = cims_[position(variable)];
  return cim.get_rate(source, target,
                      cim.get_parent_configuration(parent_values));
}

std::vector<std::string> KineticNetwork::get_variables() const {
  std::vector<std::string> out;
  for (const KineticIntensityMatrix& c : cims_) out.push_back(c.get_variables().front());
  return out;
}

std::vector<int> KineticNetwork::get_variable_states() const {
  std::vector<int> out;
  for (const KineticIntensityMatrix& c : cims_) out.push_back(c.get_variable_states().front());
  return out;
}

std::vector<std::string> KineticNetwork::get_parents(
        const std::string& variable) const {
  return cims_[position(variable)].get_parents();
}

KineticIntensityMatrix KineticNetwork::get_intensity_matrix(
        const std::string& variable) const {
  return cims_[position(variable)];
}

void KineticNetwork::set_intensity_matrix(const KineticIntensityMatrix& cim) {
  if (cim.get_variables().size() != 1) {
    IMP_THROW("a network holds one CIM per variable, not an amalgamation of "
                      << cim.get_variables().size(),
              ValueException);
  }
  KineticIntensityMatrix& current = cims_[position(cim.get_variables().front())];
  if (cim.get_variable_states() != current.get_variable_states() ||
      cim.get_parents() != current.get_parents() ||
      cim.get_parent_states() != current.get_parent_states()) {
    IMP_THROW("the CIM of " << cim.get_variables().front()
                            << " must keep its states and parents",
              ValueException);
  }
  current = cim;
}

int KineticNetwork::get_number_of_states() const {
  return kinetic_product(get_variable_states());
}

KineticIntensityMatrix KineticNetwork::get_joint() const {
  KineticIntensityMatrix joint;
  for (const KineticIntensityMatrix& c : cims_) joint = joint.amalgamate(c);
  return joint;
}

void KineticNetwork::get_generator(double** out_view, int* n_out_view) const {
  if (cims_.empty()) {
    IMP_THROW("the network has no variables", ValueException);
  }
  get_joint().get_generator(out_view, n_out_view);
}

int KineticNetwork::get_state_index(const std::vector<int>& states) const {
  return kinetic_ravel(states, get_variable_states());
}

std::vector<int> KineticNetwork::get_states(int index) const {
  const std::vector<int> sizes = get_variable_states();
  if (index < 0 || index >= kinetic_product(sizes)) {
    IMP_THROW("state " << index << " out of range", ValueException);
  }
  return kinetic_unravel(index, sizes);
}

void KineticNetwork::get_marginal(const std::vector<double>& joint,
                                  const std::string& variable,
                                  double** out_view, int* n_out_view) const {
  const std::vector<int> sizes = get_variable_states();
  const int v = position(variable);
  if (joint.size() != static_cast<std::size_t>(kinetic_product(sizes))) {
    IMP_THROW("the joint distribution has " << joint.size()
                                            << " entries, the network "
                                            << kinetic_product(sizes) << " states",
              ValueException);
  }
  std::vector<double> out(sizes[v], 0.0);
  for (std::size_t s = 0; s < joint.size(); ++s) {
    out[kinetic_unravel(static_cast<int>(s), sizes)[v]] += joint[s];
  }
  kinetic_emit(out, out_view, n_out_view);
}

// ---------------------------------------------------------------------------
// Inference on a generator

void kinetic_stationary_distribution(const std::vector<double>& generator,
                                     double** out_view, int* n_out_view) {
  const Eigen::MatrixXd k = kinetic_generator_matrix(generator);
  const int n = static_cast<int>(k.rows());
  const double scale = std::max(1.0, k.cwiseAbs().maxCoeff());
  Eigen::FullPivLU<Eigen::MatrixXd> lu(k);
  lu.setThreshold(1e-12 * n);
  if (n > 1 && lu.rank() < n - 1) {
    IMP_THROW("the stationary distribution is not unique: the generator has "
              "rank " << lu.rank() << " for " << n << " states",
              ValueException);
  }
  // K p = 0 with the normalisation appended; the system is consistent, so
  // the least-squares solution is exact.
  Eigen::MatrixXd a(n + 1, n);
  a.topRows(n) = k / scale;
  a.row(n).setOnes();
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n + 1);
  b[n] = 1.0;
  const Eigen::VectorXd p = a.colPivHouseholderQr().solve(b);
  std::vector<double> out(p.data(), p.data() + n);
  kinetic_emit(out, out_view, n_out_view);
}

void kinetic_transient_distribution(const std::vector<double>& generator,
                                    const std::vector<double>& p0,
                                    double time, double** out_view,
                                    int* n_out_view) {
  const Eigen::MatrixXd k = kinetic_generator_matrix(generator);
  const int n = static_cast<int>(k.rows());
  if (p0.size() != static_cast<std::size_t>(n)) {
    IMP_THROW("p0 has " << p0.size() << " entries, the generator " << n
                        << " states",
              ValueException);
  }
  if (!(time >= 0.0) || !std::isfinite(time)) {
    IMP_THROW("time must be finite and >= 0, not " << time, ValueException);
  }
  const Eigen::MatrixXd propagator = (k * time).exp();
  const Eigen::VectorXd p =
          propagator * Eigen::Map<const Eigen::VectorXd>(p0.data(), n);
  std::vector<double> out(p.data(), p.data() + n);
  kinetic_emit(out, out_view, n_out_view);
}

IMPBFF_END_NAMESPACE
