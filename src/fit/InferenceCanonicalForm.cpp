/**
 *  \file InferenceCanonicalForm.cpp
 *  \brief A Gaussian factor in canonical form (see InferenceCanonicalForm.h).
 *
 *  Line references are to ../chisurf/junk/aGrUM at 9f2905b60,
 *  wrappers/pyagrum/pyLibs/clg/canonicalForm.py.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */

#include <IMP/bff/InferenceCanonicalForm.h>
#include <IMP/bff/IMPCompatibility.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

// Named, not anonymous: the module is a unity build, where an anonymous
// namespace does not keep helpers of different files apart.
namespace inference_canonical_form_detail {

const double PI = 3.141592653589793238462643383279502884;
const double LOG_2PI = std::log(2.0 * PI);

//! Copy a vector into a malloc'd buffer, the ARGOUTVIEWM contract.
void emit(const std::vector<double>& values, double** out_view,
          int* n_out_view) {
  const std::size_t n = values.size();
  *out_view = static_cast<double*>(std::malloc(std::max<std::size_t>(n, 1) *
                                               sizeof(double)));
  if (*out_view == nullptr) {
    IMP_THROW("out of memory for " << n << " values", IMP::ValueException);
  }
  if (n) std::memcpy(*out_view, values.data(), n * sizeof(double));
  *n_out_view = static_cast<int>(n);
}

Eigen::MatrixXd square(const std::vector<double>& flat, int d) {
  Eigen::MatrixXd m(d, d);
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) m(i, j) = flat[std::size_t(i) * d + j];
  return m;
}

//! Cholesky of a symmetric matrix after Jacobi scaling.
/*!
    K = D^1/2 S D^1/2 with S unit-diagonal. Deciding positive definiteness on
    S rather than K makes the decision independent of the units of each
    variable -- a posterior over an amplitude of 1e6 and a lifetime of 1e-9
    has a K whose eigenvalues span 30 decades and is perfectly proper.
*/
struct ScaledCholesky {
  bool ok = false;
  Eigen::VectorXd scale;  // sqrt of the diagonal
  Eigen::LLT<Eigen::MatrixXd> llt;

  explicit ScaledCholesky(const Eigen::MatrixXd& k) {
    const Eigen::Index d = k.rows();
    scale.resize(d);
    for (Eigen::Index i = 0; i < d; ++i) {
      if (!(k(i, i) > 0.0) || !std::isfinite(k(i, i))) return;
      scale(i) = std::sqrt(k(i, i));
    }
    Eigen::MatrixXd s = scale.cwiseInverse().asDiagonal() * k *
                        scale.cwiseInverse().asDiagonal();
    llt.compute(s);
    if (llt.info() != Eigen::Success) return;
    // A unit-diagonal S has eigenvalues in (0, d]; a pivot at rounding level
    // is a singular direction, not a very well determined one.
    const double floor = 64.0 * std::numeric_limits<double>::epsilon() *
                         static_cast<double>(std::max<Eigen::Index>(d, 1));
    const Eigen::VectorXd pivots = llt.matrixLLT().diagonal();
    for (Eigen::Index i = 0; i < d; ++i) {
      if (!(pivots(i) * pivots(i) > floor)) return;
    }
    ok = true;
  }
  //! K^-1 v.
  Eigen::MatrixXd solve(const Eigen::MatrixXd& v) const {
    Eigen::MatrixXd scaled = scale.cwiseInverse().asDiagonal() * v;
    return scale.cwiseInverse().asDiagonal() * llt.solve(scaled);
  }
  double log_det() const {
    double out = 0.0;
    const Eigen::VectorXd pivots = llt.matrixLLT().diagonal();
    for (Eigen::Index i = 0; i < pivots.size(); ++i)
      out += 2.0 * std::log(pivots(i)) + 2.0 * std::log(scale(i));
    return out;
  }
};

Eigen::MatrixXd block(const std::vector<double>& flat, int d,
                      const std::vector<int>& rows,
                      const std::vector<int>& cols) {
  Eigen::MatrixXd out(rows.size(), cols.size());
  for (std::size_t i = 0; i < rows.size(); ++i)
    for (std::size_t j = 0; j < cols.size(); ++j)
      out(i, j) = flat[std::size_t(rows[i]) * d + cols[j]];
  return out;
}

Eigen::VectorXd pick(const std::vector<double>& v, const std::vector<int>& idx) {
  Eigen::VectorXd out(idx.size());
  for (std::size_t i = 0; i < idx.size(); ++i) out(i) = v[idx[i]];
  return out;
}

std::vector<double> to_std(const Eigen::VectorXd& v) {
  return std::vector<double>(v.data(), v.data() + v.size());
}

std::vector<double> to_flat(const Eigen::MatrixXd& m) {
  std::vector<double> out(std::size_t(m.rows()) * m.cols());
  for (Eigen::Index i = 0; i < m.rows(); ++i)
    for (Eigen::Index j = 0; j < m.cols(); ++j)
      out[std::size_t(i) * m.cols() + j] = m(i, j);
  return out;
}

}  // namespace inference_canonical_form_detail

namespace icf = inference_canonical_form_detail;

InferenceCanonicalForm::InferenceCanonicalForm() { finish(); }

InferenceCanonicalForm::InferenceCanonicalForm(
    const std::vector<std::string>& names, const std::vector<double>& precision,
    const std::vector<double>& information, double log_constant,
    const std::vector<int>& sizes)
    : names_(names),
      sizes_(sizes.empty() ? std::vector<int>(names.size(), 1) : sizes),
      K_(precision),
      h_(information),
      g_(log_constant) {
  finish();
}

void InferenceCanonicalForm::finish() {
  if (sizes_.size() != names_.size()) {
    IMP_THROW("a canonical form needs one size per variable: "
                  << names_.size() << " names, " << sizes_.size() << " sizes",
              IMP::ValueException);
  }
  std::set<std::string> seen;
  std::set<std::string> repeated;
  offsets_.assign(names_.size(), 0);
  int d = 0;
  for (std::size_t i = 0; i < names_.size(); ++i) {
    if (!seen.insert(names_[i]).second) repeated.insert(names_[i]);
    if (sizes_[i] < 1) {
      IMP_THROW("variable '" << names_[i] << "' must hold at least one number",
                IMP::ValueException);
    }
    offsets_[i] = d;
    d += sizes_[i];
  }
  // Every operation addresses the scope by name, so a repeated name would
  // silently resolve to one of its occurrences and answer for the wrong
  // variable (ChiSurf's canonical form found this the hard way).
  if (!repeated.empty()) {
    std::ostringstream msg;
    msg << "a canonical form needs a unique name per variable; repeated:";
    for (const auto& r : repeated) msg << " " << r;
    IMP_THROW(msg.str(), IMP::ValueException);
  }
  if (h_.size() != std::size_t(d) || K_.size() != std::size_t(d) * d) {
    IMP_THROW("a canonical form of dimension " << d << " needs " << d * d
                  << " precision and " << d << " information numbers, got "
                  << K_.size() << " and " << h_.size(),
              IMP::ValueException);
  }
  for (int i = 0; i < d; ++i) {
    for (int j = i + 1; j < d; ++j) {
      const double s = 0.5 * (K_[std::size_t(i) * d + j] + K_[std::size_t(j) * d + i]);
      K_[std::size_t(i) * d + j] = s;
      K_[std::size_t(j) * d + i] = s;
    }
  }
}

bool InferenceCanonicalForm::has_variable(const std::string& name) const {
  return std::find(names_.begin(), names_.end(), name) != names_.end();
}

int InferenceCanonicalForm::get_offset(const std::string& name) const {
  auto it = std::find(names_.begin(), names_.end(), name);
  return it == names_.end() ? -1 : offsets_[it - names_.begin()];
}

std::vector<int> InferenceCanonicalForm::indices_of(
    const std::vector<std::string>& names) const {
  std::vector<int> out;
  std::vector<std::string> missing;
  for (const auto& n : names) {
    auto it = std::find(names_.begin(), names_.end(), n);
    if (it == names_.end()) {
      missing.push_back(n);
      continue;
    }
    const std::size_t k = it - names_.begin();
    for (int j = 0; j < sizes_[k]; ++j) out.push_back(offsets_[k] + j);
  }
  if (!missing.empty()) {
    std::ostringstream msg;
    msg << "not in the scope of " << get_description() << ":";
    for (const auto& m : missing) msg << " " << m;
    IMP_THROW(msg.str(), IMP::ValueException);
  }
  return out;
}

void InferenceCanonicalForm::get_precision(double** out_view,
                                           int* n_out_view) const {
  icf::emit(K_, out_view, n_out_view);
}

void InferenceCanonicalForm::get_information(double** out_view,
                                             int* n_out_view) const {
  icf::emit(h_, out_view, n_out_view);
}

InferenceCanonicalForm InferenceCanonicalForm::from_moments(
    const std::vector<std::string>& names, const std::vector<double>& mean,
    const std::vector<double>& covariance, double log_mass,
    const std::vector<int>& sizes) {
  const int d = static_cast<int>(mean.size());
  if (covariance.size() != std::size_t(d) * d) {
    IMP_THROW("a covariance for a mean of " << d << " numbers needs " << d * d
                                            << ", got " << covariance.size(),
              IMP::ValueException);
  }
  Eigen::MatrixXd cov = icf::square(covariance, d);
  cov = 0.5 * (cov + cov.transpose()).eval();
  icf::ScaledCholesky chol(cov);
  if (!chol.ok) {
    IMP_THROW("the covariance is not positive definite", IMP::ValueException);
  }
  const Eigen::VectorXd mu = Eigen::Map<const Eigen::VectorXd>(mean.data(), d);
  Eigen::MatrixXd k = chol.solve(Eigen::MatrixXd::Identity(d, d));
  k = 0.5 * (k + k.transpose()).eval();
  const Eigen::VectorXd h = k * mu;
  const double g = log_mass - 0.5 * (d * icf::LOG_2PI + chol.log_det()) -
                   0.5 * mu.dot(h);
  return InferenceCanonicalForm(names, icf::to_flat(k), icf::to_std(h), g, sizes);
}

InferenceCanonicalForm InferenceCanonicalForm::from_linear_gaussian(
    const std::string& variable, const std::vector<std::string>& parents,
    double mu, double sigma, const std::vector<double>& weights) {
  if (weights.size() != parents.size()) {
    IMP_THROW("one weight per parent: " << parents.size() << " parents, "
                                        << weights.size() << " weights",
              IMP::ValueException);
  }
  if (!(sigma > 0.0)) {
    IMP_THROW("the standard deviation must be positive, not " << sigma,
              IMP::ValueException);
  }
  // canonicalForm.py 120-132 (Lauritzen 1992).
  const double gamma = sigma * sigma;
  const int d = static_cast<int>(parents.size()) + 1;
  Eigen::VectorXd b(d);
  b(0) = 1.0;
  for (int i = 1; i < d; ++i) b(i) = -weights[i - 1];
  const Eigen::MatrixXd k = (b * b.transpose()) / gamma;
  const Eigen::VectorXd h = (mu / gamma) * b;
  const double g = -(mu * mu / gamma + std::log(2.0 * icf::PI * gamma)) / 2.0;
  std::vector<std::string> scope{variable};
  scope.insert(scope.end(), parents.begin(), parents.end());
  return InferenceCanonicalForm(scope, icf::to_flat(k), icf::to_std(h), g);
}

InferenceCanonicalForm InferenceCanonicalForm::extend(
    const std::vector<std::string>& names,
    const std::vector<int>& sizes) const {
  if (!sizes.empty() && sizes.size() != names.size()) {
    IMP_THROW("one size per name", IMP::ValueException);
  }
  std::vector<std::string> scope = names_;
  std::vector<int> scope_sizes = sizes_;
  for (std::size_t i = 0; i < names.size(); ++i) {
    const int size = sizes.empty() ? 1 : sizes[i];
    auto it = std::find(scope.begin(), scope.end(), names[i]);
    if (it != scope.end()) {
      if (!sizes.empty() && scope_sizes[it - scope.begin()] != size) {
        IMP_THROW("variable '" << names[i] << "' holds "
                               << scope_sizes[it - scope.begin()]
                               << " numbers, not " << size,
                  IMP::ValueException);
      }
      continue;
    }
    scope.push_back(names[i]);
    scope_sizes.push_back(size);
  }
  int d = 0;
  for (int s : scope_sizes) d += s;
  const int d0 = get_dimension();
  std::vector<double> k(std::size_t(d) * d, 0.0), h(d, 0.0);
  // The original scope is a prefix of the extended one, so its block is the
  // leading one (canonicalForm.py 212-259 inserts zeros instead, because its
  // scope is sorted by id).
  for (int i = 0; i < d0; ++i) {
    h[i] = h_[i];
    for (int j = 0; j < d0; ++j)
      k[std::size_t(i) * d + j] = K_[std::size_t(i) * d0 + j];
  }
  return InferenceCanonicalForm(scope, k, h, g_, scope_sizes);
}

InferenceCanonicalForm InferenceCanonicalForm::combine(
    const InferenceCanonicalForm& other, double sign) const {
  // canonicalForm.py 261-287 (product) and 289-314 (division): both forms
  // extended to the union scope, then K, h and g added or subtracted. Here
  // the union keeps this scope first and places `other` by name.
  InferenceCanonicalForm out = extend(other.names_, other.sizes_);
  const int d = out.get_dimension();
  const int db = other.get_dimension();
  std::vector<int> position(db);
  for (std::size_t v = 0; v < other.names_.size(); ++v) {
    const int to = out.get_offset(other.names_[v]);
    for (int j = 0; j < other.sizes_[v]; ++j)
      position[other.offsets_[v] + j] = to + j;
  }
  for (int i = 0; i < db; ++i) {
    out.h_[position[i]] += sign * other.h_[i];
    for (int j = 0; j < db; ++j)
      out.K_[std::size_t(position[i]) * d + position[j]] +=
          sign * other.K_[std::size_t(i) * db + j];
  }
  out.g_ += sign * other.g_;
  return out;
}

InferenceCanonicalForm InferenceCanonicalForm::product(
    const InferenceCanonicalForm& other) const {
  return combine(other, 1.0);
}

InferenceCanonicalForm InferenceCanonicalForm::divide(
    const InferenceCanonicalForm& other) const {
  return combine(other, -1.0);
}

InferenceCanonicalForm InferenceCanonicalForm::reduce(
    const std::vector<std::string>& names, const std::vector<double>& values,
    const std::vector<int>& sizes) const {
  if (!sizes.empty() && sizes.size() != names.size()) {
    IMP_THROW("evidence needs one size per variable", IMP::ValueException);
  }
  // canonicalForm.py 386: evidence on a variable out of scope is dropped --
  // a factor sees only its own.
  std::vector<std::string> held;
  std::vector<double> held_values;
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < names.size(); ++i) {
    const int size = sizes.empty() ? 1 : sizes[i];
    if (size < 1) {
      IMP_THROW("variable '" << names[i] << "' must hold at least one number",
                IMP::ValueException);
    }
    if (cursor + size > values.size()) {
      IMP_THROW("evidence for " << names.size()
                                << " variables ran out of values at '"
                                << names[i] << "'",
                IMP::ValueException);
    }
    auto it = std::find(names_.begin(), names_.end(), names[i]);
    if (it != names_.end()) {
      if (sizes_[it - names_.begin()] != size) {
        IMP_THROW("variable '" << names[i] << "' holds "
                               << sizes_[it - names_.begin()]
                               << " numbers here, evidence gives " << size,
                  IMP::ValueException);
      }
      held.push_back(names[i]);
      held_values.insert(held_values.end(), values.begin() + cursor,
                         values.begin() + cursor + size);
    }
    cursor += size;
  }
  if (cursor != values.size()) {
    IMP_THROW("evidence for " << names.size() << " variables needs " << cursor
                              << " values, got " << values.size(),
              IMP::ValueException);
  }
  if (held.empty()) return *this;

  const int d = get_dimension();
  const std::vector<int> b = indices_of(held);
  std::vector<int> a;
  std::vector<std::string> scope;
  std::vector<int> scope_sizes;
  for (std::size_t v = 0; v < names_.size(); ++v) {
    if (std::find(held.begin(), held.end(), names_[v]) != held.end()) continue;
    scope.push_back(names_[v]);
    scope_sizes.push_back(sizes_[v]);
    for (int j = 0; j < sizes_[v]; ++j) a.push_back(offsets_[v] + j);
  }
  const Eigen::VectorXd v =
      Eigen::Map<const Eigen::VectorXd>(held_values.data(), held_values.size());
  // canonicalForm.py 404-415.
  const Eigen::MatrixXd k_bb = icf::block(K_, d, b, b);
  const double g = g_ + icf::pick(h_, b).dot(v) - 0.5 * v.dot(k_bb * v);
  if (a.empty()) {
    return InferenceCanonicalForm(std::vector<std::string>(),
                                  std::vector<double>(), std::vector<double>(),
                                  g);
  }
  const Eigen::MatrixXd k_aa = icf::block(K_, d, a, a);
  const Eigen::VectorXd h = icf::pick(h_, a) - icf::block(K_, d, a, b) * v;
  return InferenceCanonicalForm(scope, icf::to_flat(k_aa), icf::to_std(h), g,
                                scope_sizes);
}

InferenceCanonicalForm InferenceCanonicalForm::condition(
    const std::vector<std::string>& names,
    const std::vector<double>& values) const {
  indices_of(names);  // every name must be in scope
  std::set<std::string> distinct(names.begin(), names.end());
  if (distinct.size() != names.size()) {
    IMP_THROW("a variable can be held at one value only", IMP::ValueException);
  }
  std::vector<int> sizes;
  for (const auto& n : names)
    sizes.push_back(sizes_[std::find(names_.begin(), names_.end(), n) -
                           names_.begin()]);
  return reduce(names, values, sizes);
}

InferenceCanonicalForm InferenceCanonicalForm::marginalize(
    const std::vector<std::string>& names) const {
  // canonicalForm.py 341-345.
  if (names.empty()) return *this;
  const std::vector<int> b = indices_of(names);
  std::set<std::string> gone(names.begin(), names.end());
  if (gone.size() != names.size()) {
    IMP_THROW("a variable can be integrated out once", IMP::ValueException);
  }
  const int d = get_dimension();
  std::vector<int> a;
  std::vector<std::string> scope;
  std::vector<int> scope_sizes;
  for (std::size_t v = 0; v < names_.size(); ++v) {
    if (gone.count(names_[v])) continue;
    scope.push_back(names_[v]);
    scope_sizes.push_back(sizes_[v]);
    for (int j = 0; j < sizes_[v]; ++j) a.push_back(offsets_[v] + j);
  }
  // canonicalForm.py 354-369. aGrUM inverts K_yy; a singular one raises
  // there too, but only by numpy's accident. It is a statement: the variables
  // being integrated out have a direction nothing constrains, and the
  // integral along it is infinite.
  const Eigen::MatrixXd k_bb = icf::block(K_, d, b, b);
  icf::ScaledCholesky chol(k_bb);
  if (!chol.ok) {
    std::ostringstream msg;
    msg << "cannot integrate out";
    for (const auto& n : names) msg << " " << n;
    msg << ": their block of the precision is not positive definite, so the "
           "integral along an unconstrained direction diverges";
    IMP_THROW(msg.str(), IMP::ValueException);
  }
  const Eigen::VectorXd h_b = icf::pick(h_, b);
  const Eigen::VectorXd solved_h = chol.solve(h_b);
  const double g = g_ + 0.5 * (static_cast<double>(b.size()) * icf::LOG_2PI -
                               chol.log_det() + h_b.dot(solved_h));
  if (a.empty()) {
    return InferenceCanonicalForm(std::vector<std::string>(),
                                  std::vector<double>(), std::vector<double>(),
                                  g);
  }
  const Eigen::MatrixXd k_ab = icf::block(K_, d, a, b);
  const Eigen::MatrixXd k = icf::block(K_, d, a, a) - k_ab * chol.solve(k_ab.transpose());
  const Eigen::VectorXd h = icf::pick(h_, a) - k_ab * solved_h;
  return InferenceCanonicalForm(scope, icf::to_flat(k), icf::to_std(h), g, scope_sizes);
}

InferenceCanonicalForm InferenceCanonicalForm::marginal(
    const std::vector<std::string>& keep) const {
  std::set<std::string> kept(keep.begin(), keep.end());
  if (kept.size() != keep.size()) {
    IMP_THROW("a canonical form needs a unique name per variable; the kept "
              "list repeats one",
              IMP::ValueException);
  }
  indices_of(keep);
  std::vector<std::string> gone;
  for (const auto& n : names_)
    if (!kept.count(n)) gone.push_back(n);
  const InferenceCanonicalForm reduced = marginalize(gone);
  // Reorder to `keep`.
  const std::vector<int> idx = reduced.indices_of(keep);
  const int d = reduced.get_dimension();
  std::vector<int> sizes;
  for (const auto& n : keep) {
    auto it = std::find(names_.begin(), names_.end(), n);
    sizes.push_back(sizes_[it - names_.begin()]);
  }
  return InferenceCanonicalForm(keep, icf::to_flat(icf::block(reduced.K_, d, idx, idx)),
                                icf::to_std(icf::pick(reduced.h_, idx)), reduced.g_,
                                sizes);
}

bool InferenceCanonicalForm::get_is_proper() const {
  const int d = get_dimension();
  if (d == 0) return true;
  return icf::ScaledCholesky(icf::square(K_, d)).ok;
}

namespace inference_canonical_form_detail {

//! Eigen-decomposition of the Jacobi-scaled K, and the scale used.
void scaled_eigen(const std::vector<double>& flat, int d,
                  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>* solver,
                  Eigen::VectorXd* inverse_scale) {
  const Eigen::MatrixXd k = square(flat, d);
  inverse_scale->resize(d);
  for (int i = 0; i < d; ++i) {
    (*inverse_scale)(i) = k(i, i) > 0.0 ? 1.0 / std::sqrt(k(i, i)) : 1.0;
  }
  solver->compute(inverse_scale->asDiagonal() * k * inverse_scale->asDiagonal());
}

}  // namespace inference_canonical_form_detail

int InferenceCanonicalForm::get_rank(double relative_tolerance) const {
  const int d = get_dimension();
  if (d == 0) return 0;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
  Eigen::VectorXd inverse_scale;
  icf::scaled_eigen(K_, d, &solver, &inverse_scale);
  const Eigen::VectorXd ev = solver.eigenvalues();
  const double top = ev.cwiseAbs().maxCoeff();
  int rank = 0;
  for (int i = 0; i < d; ++i)
    if (std::abs(ev(i)) > relative_tolerance * std::max(top, 1e-300)) ++rank;
  return rank;
}

void InferenceCanonicalForm::get_null_space(double** out_view, int* n_out_view,
                                            double relative_tolerance) const {
  const int d = get_dimension();
  if (d == 0) {
    icf::emit(std::vector<double>(), out_view, n_out_view);
    return;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
  Eigen::VectorXd inverse_scale;
  icf::scaled_eigen(K_, d, &solver, &inverse_scale);
  const Eigen::VectorXd ev = solver.eigenvalues();
  const double top = ev.cwiseAbs().maxCoeff();
  std::vector<int> null;
  for (int i = 0; i < d; ++i)
    if (!(std::abs(ev(i)) > relative_tolerance * std::max(top, 1e-300)))
      null.push_back(i);
  if (null.empty()) {
    icf::emit(std::vector<double>(), out_view, n_out_view);
    return;
  }
  // S v = 0 with S = D^-1/2 K D^-1/2 means K (D^-1/2 v) = 0; orthonormalise
  // the mapped vectors, since the scaling does not preserve angles.
  Eigen::MatrixXd basis(d, null.size());
  for (std::size_t j = 0; j < null.size(); ++j)
    basis.col(j) = inverse_scale.asDiagonal() * solver.eigenvectors().col(null[j]);
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(basis);
  Eigen::MatrixXd q =
      qr.householderQ() * Eigen::MatrixXd::Identity(d, null.size());
  icf::emit(icf::to_flat(q), out_view, n_out_view);
}

void InferenceCanonicalForm::get_mean(double** out_view,
                                      int* n_out_view) const {
  const int d = get_dimension();
  if (d == 0) {
    icf::emit(std::vector<double>(), out_view, n_out_view);
    return;
  }
  icf::ScaledCholesky chol(icf::square(K_, d));
  if (!chol.ok) {
    IMP_THROW(get_description() << " has no mean: its precision has rank "
                                << get_rank() << " of " << d
                                << "; get_null_space() names the directions "
                                   "the factor leaves unconstrained",
              IMP::ValueException);
  }
  const Eigen::VectorXd h = Eigen::Map<const Eigen::VectorXd>(h_.data(), d);
  icf::emit(icf::to_std(chol.solve(h)), out_view, n_out_view);
}

void InferenceCanonicalForm::get_covariance(double** out_view,
                                            int* n_out_view) const {
  const int d = get_dimension();
  if (d == 0) {
    icf::emit(std::vector<double>(), out_view, n_out_view);
    return;
  }
  icf::ScaledCholesky chol(icf::square(K_, d));
  if (!chol.ok) {
    IMP_THROW(get_description() << " has no covariance: its precision has rank "
                                << get_rank() << " of " << d
                                << "; get_null_space() names the directions "
                                   "the factor leaves unconstrained",
              IMP::ValueException);
  }
  Eigen::MatrixXd cov = chol.solve(Eigen::MatrixXd::Identity(d, d));
  cov = 0.5 * (cov + cov.transpose()).eval();
  icf::emit(icf::to_flat(cov), out_view, n_out_view);
}

double InferenceCanonicalForm::get_log_normalizer() const {
  const int d = get_dimension();
  if (d == 0) return g_;
  icf::ScaledCholesky chol(icf::square(K_, d));
  if (!chol.ok) return std::numeric_limits<double>::infinity();
  const Eigen::VectorXd h = Eigen::Map<const Eigen::VectorXd>(h_.data(), d);
  const Eigen::VectorXd mu = chol.solve(h);
  return g_ + 0.5 * (d * icf::LOG_2PI - chol.log_det() + h.dot(mu));
}

double InferenceCanonicalForm::get_log_density(
    const std::vector<double>& x) const {
  const int d = get_dimension();
  if (x.size() != std::size_t(d)) {
    IMP_THROW("a point of " << get_description() << " has " << d
                            << " numbers, not " << x.size(),
              IMP::ValueException);
  }
  double quad = 0.0, lin = 0.0;
  for (int i = 0; i < d; ++i) {
    lin += h_[i] * x[i];
    for (int j = 0; j < d; ++j) quad += x[i] * K_[std::size_t(i) * d + j] * x[j];
  }
  return g_ + lin - 0.5 * quad;
}

std::string InferenceCanonicalForm::get_description() const {
  std::ostringstream out;
  out << "InferenceCanonicalForm(";
  for (std::size_t i = 0; i < names_.size(); ++i) {
    if (i) out << ", ";
    out << names_[i];
    if (sizes_[i] != 1) out << "[" << sizes_[i] << "]";
  }
  out << ")";
  return out.str();
}

IMPBFF_END_NAMESPACE
