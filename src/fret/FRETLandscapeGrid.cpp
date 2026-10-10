// SPDX-License-Identifier: BSD-3-Clause
/**
 * The discretisation of the photon-by-photon landscape likelihood: SqRA
 * generator, its symmetrisation, the tridiagonal eigensolver and the natural
 * cubic spline. See FRETLandscapeGrid.h.
 */
#include <IMP/bff/FRETLandscapeGrid.h>
#include <IMP/bff/internal/TridiagonalEigen.h>

#include <algorithm>
#include <cmath>
#include <numeric>

IMPBFF_BEGIN_NAMESPACE

TridiagonalEigenSystem symmetric_tridiagonal_eigen(
    const std::vector<double>& diagonal, const std::vector<double>& off_diagonal) {
  const int m = static_cast<int>(diagonal.size());
  if (m < 1)
    IMP_THROW("symmetric_tridiagonal_eigen: empty matrix", IMP::ValueException);
  if (static_cast<int>(off_diagonal.size()) != m - 1)
    IMP_THROW("symmetric_tridiagonal_eigen: off_diagonal must have n-1 entries",
              IMP::ValueException);
  std::vector<double> d(diagonal), e(off_diagonal), z(static_cast<std::size_t>(m) * m, 0.0);
  for (int i = 0; i < m; ++i) z[static_cast<std::size_t>(i) * m + i] = 1.0;
  internal::tridiagonal_eigen_ql(d, e, z, m);
  std::vector<int> order(m);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return d[a] > d[b]; });
  std::vector<double> values(m), vectors(static_cast<std::size_t>(m) * m);
  for (int k = 0; k < m; ++k) {
    values[k] = d[order[k]];
    for (int i = 0; i < m; ++i)
      vectors[static_cast<std::size_t>(i) * m + k] = z[static_cast<std::size_t>(i) * m + order[k]];
  }
  return TridiagonalEigenSystem(values, vectors);
}

namespace {
void sqra_check_grid(const std::vector<double>& u, double diffusion, double spacing,
                const char* who) {
  if (u.empty()) IMP_THROW(who << ": empty landscape", IMP::ValueException);
  if (!(diffusion > 0.0)) IMP_THROW(who << ": diffusion must be > 0", IMP::ValueException);
  if (!(spacing > 0.0)) IMP_THROW(who << ": spacing must be > 0", IMP::ValueException);
}
}  // namespace

std::vector<double> sqra_generator(const std::vector<double>& u, double diffusion,
                                   double spacing) {
  sqra_check_grid(u, diffusion, spacing, "sqra_generator");
  const std::size_t m = u.size();
  const double o = diffusion / (spacing * spacing);
  std::vector<double> q(m * m, 0.0);
  for (std::size_t j = 0; j < m; ++j) {
    double out = 0.0;
    if (j + 1 < m) {
      const double r = o * std::exp(-0.5 * (u[j + 1] - u[j]));
      q[(j + 1) * m + j] = r;
      out += r;
    }
    if (j > 0) {
      const double r = o * std::exp(-0.5 * (u[j - 1] - u[j]));
      q[(j - 1) * m + j] = r;
      out += r;
    }
    q[j * m + j] = -out;
  }
  return q;
}

std::vector<double> sqra_stationary_distribution(const std::vector<double>& u) {
  if (u.empty())
    IMP_THROW("sqra_stationary_distribution: empty landscape", IMP::ValueException);
  const double umin = *std::min_element(u.begin(), u.end());
  std::vector<double> p(u.size());
  double s = 0.0;
  for (std::size_t i = 0; i < u.size(); ++i) s += (p[i] = std::exp(-(u[i] - umin)));
  for (double& v : p) v /= s;
  return p;
}

std::vector<double> sqra_symmetric_diagonal(const std::vector<double>& u, double diffusion,
                                            double spacing,
                                            const std::vector<double>& killing) {
  sqra_check_grid(u, diffusion, spacing, "sqra_symmetric_diagonal");
  const std::size_t m = u.size();
  if (!killing.empty() && killing.size() != m)
    IMP_THROW("sqra_symmetric_diagonal: killing must match the grid", IMP::ValueException);
  const double o = diffusion / (spacing * spacing);
  std::vector<double> d(m, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    double c = 0.0;
    if (i + 1 < m) c += std::exp(-0.5 * (u[i + 1] - u[i]));
    if (i > 0) c += std::exp(-0.5 * (u[i - 1] - u[i]));
    d[i] = -o * c - (killing.empty() ? 0.0 : killing[i]);
  }
  return d;
}

// --- natural cubic spline ---------------------------------------------------

NaturalCubicSpline::NaturalCubicSpline(double x_min, double x_max, int n_knots)
    : x0_(x_min), h_(0.0), n_(n_knots) {
  if (n_knots < 2) IMP_THROW("NaturalCubicSpline: n_knots must be >= 2", IMP::ValueException);
  if (!(x_max > x_min)) IMP_THROW("NaturalCubicSpline: x_max must exceed x_min", IMP::ValueException);
  h_ = (x_max - x_min) / (n_knots - 1);
}

std::vector<double> NaturalCubicSpline::get_knots() const {
  std::vector<double> k(n_);
  for (int i = 0; i < n_; ++i) k[i] = x0_ + h_ * i;
  return k;
}

std::vector<double> NaturalCubicSpline::knot_curvatures(const std::vector<double>& mu) const {
  if (static_cast<int>(mu.size()) != n_)
    IMP_THROW("NaturalCubicSpline: expected " << n_ << " knot heights", IMP::ValueException);
  std::vector<double> m2(n_, 0.0);
  const int n = n_ - 2;  // interior unknowns
  if (n <= 0) return m2;
  // m_{j-1} + 4 m_j + m_{j+1} = 6/h^2 (y_{j+1} - 2 y_j + y_{j-1}), Thomas algorithm.
  std::vector<double> c(n), r(n);
  const double s = 6.0 / (h_ * h_);
  for (int j = 0; j < n; ++j) r[j] = s * (mu[j + 2] - 2.0 * mu[j + 1] + mu[j]);
  double b = 4.0;
  c[0] = 1.0 / b;
  r[0] /= b;
  for (int j = 1; j < n; ++j) {
    b = 4.0 - c[j - 1];
    c[j] = 1.0 / b;
    r[j] = (r[j] - r[j - 1]) / b;
  }
  for (int j = n - 2; j >= 0; --j) r[j] -= c[j] * r[j + 1];
  for (int j = 0; j < n; ++j) m2[j + 1] = r[j];
  return m2;
}

void NaturalCubicSpline::evaluate_point(const std::vector<double>& mu,
                                        const std::vector<double>& m2, double x,
                                        double& value, double& slope) const {
  const double xl = x0_, xr = x0_ + h_ * (n_ - 1);
  if (x <= xl || x >= xr) {
    // linear continuation with the end slope (zero curvature at the ends)
    const bool left = x <= xl;
    const int j = left ? 0 : n_ - 2;
    const double t = left ? 0.0 : 1.0;
    const double y0 = mu[j], y1 = mu[j + 1], a = m2[j], bb = m2[j + 1];
    slope = (y1 - y0) / h_ + h_ / 6.0 * (-(3.0 * (1 - t) * (1 - t) - 1.0) * a +
                                         (3.0 * t * t - 1.0) * bb);
    value = (left ? y0 : y1) + slope * (x - (left ? xl : xr));
    return;
  }
  int j = static_cast<int>(std::floor((x - x0_) / h_));
  j = std::max(0, std::min(n_ - 2, j));
  const double t = (x - (x0_ + h_ * j)) / h_, t1 = 1.0 - t;
  const double y0 = mu[j], y1 = mu[j + 1], a = m2[j], bb = m2[j + 1];
  value = t1 * y0 + t * y1 + h_ * h_ / 6.0 * ((t1 * t1 * t1 - t1) * a + (t * t * t - t) * bb);
  slope = (y1 - y0) / h_ + h_ / 6.0 * (-(3.0 * t1 * t1 - 1.0) * a + (3.0 * t * t - 1.0) * bb);
}

std::vector<double> NaturalCubicSpline::evaluate(const std::vector<double>& mu,
                                                 const std::vector<double>& x) const {
  const std::vector<double> m2 = knot_curvatures(mu);
  std::vector<double> out(x.size());
  double s;
  for (std::size_t i = 0; i < x.size(); ++i) evaluate_point(mu, m2, x[i], out[i], s);
  return out;
}

std::vector<double> NaturalCubicSpline::derivative(const std::vector<double>& mu,
                                                   const std::vector<double>& x) const {
  const std::vector<double> m2 = knot_curvatures(mu);
  std::vector<double> out(x.size());
  double v;
  for (std::size_t i = 0; i < x.size(); ++i) evaluate_point(mu, m2, x[i], v, out[i]);
  return out;
}

std::vector<double> NaturalCubicSpline::get_basis(const std::vector<double>& x) const {
  std::vector<double> phi(x.size() * n_);
  std::vector<double> e(n_, 0.0);
  for (int k = 0; k < n_; ++k) {
    e[k] = 1.0;
    const std::vector<double> col = evaluate(e, x);
    for (std::size_t i = 0; i < x.size(); ++i) phi[i * n_ + k] = col[i];
    e[k] = 0.0;
  }
  return phi;
}

std::vector<double> NaturalCubicSpline::get_basis_derivative(
    const std::vector<double>& x) const {
  std::vector<double> phi(x.size() * n_);
  std::vector<double> e(n_, 0.0);
  for (int k = 0; k < n_; ++k) {
    e[k] = 1.0;
    const std::vector<double> col = derivative(e, x);
    for (std::size_t i = 0; i < x.size(); ++i) phi[i * n_ + k] = col[i];
    e[k] = 0.0;
  }
  return phi;
}

IMPBFF_END_NAMESPACE
