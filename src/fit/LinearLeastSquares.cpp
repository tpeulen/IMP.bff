/**
 * \file LinearLeastSquares.cpp
 * \brief NNLS (Lawson & Hanson) and BVLS (Stark & Parker), one active-set core.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/LinearLeastSquares.h>
#include <IMP/bff/IMPCompatibility.h>

#include <algorithm>
#include <cmath>
#include <limits>

IMPBFF_BEGIN_NAMESPACE

// Named, not anonymous: the IMP build compiles every source of the module
// into one translation unit (bff_all.cpp), where anonymous helpers collide.
namespace lls_internal {

const double EPS = std::numeric_limits<double>::epsilon();

//! A dot product with four independent accumulators.
/*! Without -ffast-math the compiler may not reorder a reduction, so the
    one-accumulator loop runs at one multiply-add per latency; the dot
    products are most of the work here (the dual test is `A^T r` every
    iteration), and splitting the sum is what lets them vectorise. */
double dot(const double* x, const double* y, int m) {
  double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
  int i = 0;
  for (; i + 4 <= m; i += 4) {
    s0 += x[i] * y[i];
    s1 += x[i + 1] * y[i + 1];
    s2 += x[i + 2] * y[i + 2];
    s3 += x[i + 3] * y[i + 3];
  }
  for (; i < m; ++i) s0 += x[i] * y[i];
  return (s0 + s1) + (s2 + s3);
}

//! A thin QR factorisation of the free columns, updated one column at a time.
/*!
    `A[:, vars] = Q R` with `Q` `m x p` orthonormal and `R` `p x p` upper
    triangular. Each active-set step frees or binds one variable, so the
    factorisation is *updated* rather than recomputed: adding a column is a
    re-orthogonalised Gram-Schmidt step and removing one is a sweep of Givens
    rotations, both `O(m p)`. Refactorising per step instead cost `O(m p^2)`
    each and made a 4096 x 200 problem a hundred times slower than scipy.

    Orthogonalising against `Q` (rather than forming `A^T A`) keeps the
    accuracy QR is used for: the normal equations square the condition
    number, which is exactly what Lawson & Hanson set out to avoid.
*/
struct ThinQR {
  int m = 0;
  const std::vector<double>* acol = nullptr;  // A, column-major
  std::vector<std::vector<double> > q;        // columns of Q
  std::vector<std::vector<double> > r;        // r[c]: column c of R, length c+1
  std::vector<int> vars;                      // the variable behind each column

  const double* column(int j) const { return &(*acol)[static_cast<std::size_t>(j) * m]; }

  //! Append variable j's column; false (and nothing changed) if it is
  //! numerically dependent on the columns already there.
  bool add(int j) {
    const int p = static_cast<int>(q.size());
    std::vector<double> v(column(j), column(j) + m);
    const double norm0 = std::sqrt(dot(v.data(), v.data(), m));
    if (!(norm0 > 0.0)) return false;
    std::vector<double> rc(p + 1, 0.0);
    // Two passes of modified Gram-Schmidt: one pass loses orthogonality in
    // proportion to the condition number, the second restores it to working
    // precision ("twice is enough").
    for (int pass = 0; pass < 2; ++pass) {
      for (int c = 0; c < p; ++c) {
        const double d = dot(q[c].data(), v.data(), m);
        rc[c] += d;
        const double* qc = q[c].data();
        for (int i = 0; i < m; ++i) v[i] -= d * qc[i];
      }
    }
    const double rho = std::sqrt(dot(v.data(), v.data(), m));
    if (!(rho > 10.0 * EPS * std::max(m, p + 1) * norm0)) return false;
    for (int i = 0; i < m; ++i) v[i] /= rho;
    rc[p] = rho;
    q.push_back(std::move(v));
    r.push_back(std::move(rc));
    vars.push_back(j);
    return true;
  }

  int position(int j) const {
    for (std::size_t c = 0; c < vars.size(); ++c)
      if (vars[c] == j) return static_cast<int>(c);
    return -1;
  }

  //! Drop variable j's column and restore R to triangular form.
  void remove(int j) {
    const int pos = position(j);
    if (pos < 0) return;
    r.erase(r.begin() + pos);
    vars.erase(vars.begin() + pos);
    const int np = static_cast<int>(r.size());
    // Columns pos.. now carry one sub-diagonal entry each (upper Hessenberg);
    // a Givens rotation of rows (c, c+1) zeroes it, and the same rotation of
    // Q's columns (c, c+1) keeps A = Q R.
    for (int c = pos; c < np; ++c) {
      const double a = r[c][c], b = r[c][c + 1];
      const double h = std::hypot(a, b);
      const double cs = (h > 0.0) ? a / h : 1.0;
      const double sn = (h > 0.0) ? b / h : 0.0;
      for (int cc = c; cc < np; ++cc) {
        const double x = r[cc][c], y = r[cc][c + 1];
        r[cc][c] = cs * x + sn * y;
        r[cc][c + 1] = -sn * x + cs * y;
      }
      r[c].pop_back();
      double* qa = q[c].data();
      double* qb = q[c + 1].data();
      for (int i = 0; i < m; ++i) {
        const double x = qa[i], y = qb[i];
        qa[i] = cs * x + sn * y;
        qb[i] = -sn * x + cs * y;
      }
    }
    q.pop_back();
  }

  //! The least-squares solution over the free columns, scattered into n.
  std::vector<double> solve(const std::vector<double>& rhs, int n) const {
    const int p = static_cast<int>(vars.size());
    std::vector<double> y(p), z(n, 0.0);
    for (int c = 0; c < p; ++c) y[c] = dot(q[c].data(), rhs.data(), m);
    for (int c = p - 1; c >= 0; --c) {
      double s = y[c];
      for (int k = c + 1; k < p; ++k) s -= r[k][c] * y[k];
      y[c] = s / r[c][c];
    }
    for (int c = 0; c < p; ++c) z[vars[c]] = y[c];
    return z;
  }
};

enum BoundState { FREE = 0, AT_LOWER = -1, AT_UPPER = 1, PINNED = 2 };

void check_shapes(int m, int n, int nb) {
  if (m < 0 || n < 0)
    IMP_THROW("least squares: negative matrix dimension", ValueException);
  if (nb != m)
    IMP_THROW("least squares: A has " << m << " rows and b has " << nb
                                       << " entries; they must agree",
              ValueException);
}

//! The active-set loop shared by NNLS and BVLS.
/*!
    \param[in] x, state the start: every variable is FREE, on a bound, or
               PINNED (held at its value because its column is dependent on
               the free ones and it has no bound to sit on)
*/
LinearLeastSquaresResult active_set(const double* a, int m, int n,
                                    const double* b,
                                    const std::vector<double>& lb,
                                    const std::vector<double>& ub,
                                    std::vector<double> x,
                                    std::vector<int> state, int maxiter,
                                    double tol) {
  std::vector<double> acol(static_cast<std::size_t>(m) * n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) acol[static_cast<std::size_t>(j) * m + i] = a[i * n + j];

  if (!(tol > 0.0)) {
    // The descent direction scales as ||A|| ||r||, so the tolerance carries
    // both: a fixed tolerance would read every gradient of a problem with
    // b ~ 1e6 as "nonzero" and every gradient with b ~ 1e-6 as zero.
    double norm1 = 0.0;
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int i = 0; i < m; ++i) s += std::fabs(acol[static_cast<std::size_t>(j) * m + i]);
      norm1 = std::max(norm1, s);
    }
    const double bn = std::sqrt(dot(b, b, m));
    tol = 10.0 * EPS * std::max(m, n) * norm1 * std::max(bn, 1e-300);
  }

  ThinQR qr;
  qr.m = m;
  qr.acol = &acol;
  for (int j = 0; j < n; ++j) {
    if (state[j] != FREE || qr.add(j)) continue;
    // A dependent column in the start: hold the variable where it is, on a
    // bound when it has one.
    if (std::isfinite(lb[j]) && (!std::isfinite(ub[j]) || x[j] - lb[j] <= ub[j] - x[j])) {
      x[j] = lb[j];
      state[j] = AT_LOWER;
    } else if (std::isfinite(ub[j])) {
      x[j] = ub[j];
      state[j] = AT_UPPER;
    } else {
      state[j] = PINNED;
    }
  }

  auto residual = [&]() {
    std::vector<double> res(m);
    for (int i = 0; i < m; ++i) res[i] = -b[i];
    for (int j = 0; j < n; ++j) {
      if (x[j] == 0.0) continue;
      const double* cj = qr.column(j);
      for (int i = 0; i < m; ++i) res[i] += cj[i] * x[j];
    }
    return res;
  };
  // b minus the contribution of every variable held fixed.
  auto reduced_rhs = [&]() {
    std::vector<double> rhs(b, b + m);
    for (int j = 0; j < n; ++j) {
      if (state[j] == FREE || x[j] == 0.0) continue;
      const double* cj = qr.column(j);
      for (int i = 0; i < m; ++i) rhs[i] -= cj[i] * x[j];
    }
    return rhs;
  };

  int iterations = 0;
  bool exhausted = false;
  std::vector<bool> excluded(n, false);

  // The inner loop: solve over the free set, stepping back into the box while
  // the solve leaves it. Returns false when the freed variable `t` went the
  // wrong way and was rebound.
  auto settle = [&](int t, int t_side) -> bool {
    bool first = true;
    while (true) {
      const std::vector<double> z = qr.solve(reduced_rhs(), n);
      // The freed variable must move off its bound into the box; if the
      // unconstrained solve sends it the wrong way, freeing it was a
      // round-off artefact of the dual test. Rebind it and try the next
      // candidate -- Stark & Parker's step 6, and what stops NNLS cycling.
      if (first && t >= 0 && t_side != PINNED) {
        const bool wrong = (t_side == AT_LOWER) ? !(z[t] > x[t]) : !(z[t] < x[t]);
        if (wrong) {
          qr.remove(t);
          state[t] = t_side;
          excluded[t] = true;
          return false;
        }
      }
      first = false;
      double alpha = 1.0;
      bool feasible = true;
      for (int j : qr.vars) {
        if (z[j] < lb[j]) {
          feasible = false;
          const double d = x[j] - z[j];
          if (d > 0.0) alpha = std::min(alpha, (x[j] - lb[j]) / d);
        } else if (z[j] > ub[j]) {
          feasible = false;
          const double d = z[j] - x[j];
          if (d > 0.0) alpha = std::min(alpha, (ub[j] - x[j]) / d);
        }
      }
      if (feasible) {
        for (int j : qr.vars) x[j] = z[j];
        return true;
      }
      if (++iterations > maxiter) {
        exhausted = true;
        return true;
      }
      // Step towards z as far as the box allows, and bind whatever reached
      // a bound on the way.
      alpha = std::max(0.0, std::min(1.0, alpha));
      const std::vector<int> cols = qr.vars;
      for (int j : cols) {
        x[j] += alpha * (z[j] - x[j]);
        const double span = std::max(1.0, std::fabs(x[j]));
        if (x[j] <= lb[j] + 10.0 * EPS * span) {
          x[j] = lb[j];
          state[j] = AT_LOWER;
          qr.remove(j);
        } else if (x[j] >= ub[j] - 10.0 * EPS * span) {
          x[j] = ub[j];
          state[j] = AT_UPPER;
          qr.remove(j);
        }
      }
    }
  };

  // A start with free variables (BVLS's projected start) is not optimal over
  // its own free set yet; settle it before the first dual test.
  if (!qr.vars.empty()) settle(-1, FREE);

  while (!exhausted) {
    const std::vector<double> res = residual();
    // The fixed variable whose release most decreases the objective: one at
    // its lower bound wants to grow (w > 0), one at its upper bound to shrink
    // (w < 0), a pinned one either way. w = -A^T (A x - b).
    int t = -1;
    double best = tol;
    for (int j = 0; j < n; ++j) {
      if (state[j] == FREE || excluded[j]) continue;
      const double w = -dot(qr.column(j), res.data(), m);
      const double g = (state[j] == AT_LOWER) ? w : (state[j] == AT_UPPER) ? -w : std::fabs(w);
      if (g > best) {
        best = g;
        t = j;
      }
    }
    if (t < 0) break;
    if (++iterations > maxiter) {
      exhausted = true;
      break;
    }
    const int t_side = state[t];
    if (!qr.add(t)) {
      // Dependent on the free set: freeing it cannot lower the residual in
      // exact arithmetic, whatever round-off says.
      excluded[t] = true;
      continue;
    }
    state[t] = FREE;
    if (settle(t, t_side)) std::fill(excluded.begin(), excluded.end(), false);
  }

  LinearLeastSquaresResult out;
  out.x = x;
  out.residuals = residual();
  out.rnorm = std::sqrt(dot(out.residuals.data(), out.residuals.data(), m));
  out.iterations = iterations;
  out.status = exhausted ? 1 : 0;
  out.active_mask.assign(n, 0);
  for (int j = 0; j < n; ++j) {
    if (state[j] == AT_LOWER && std::isfinite(lb[j])) out.active_mask[j] = -1;
    else if (state[j] == AT_UPPER && std::isfinite(ub[j])) out.active_mask[j] = 1;
  }
  return out;
}

//! Unconstrained least squares by Householder QR with column pivoting.
/*! Only BVLS's start uses it, once; dependent columns get zero (the basic
    solution). */
std::vector<double> lstsq(const double* a, int m, int n, const double* b) {
  std::vector<double> q(static_cast<std::size_t>(m) * n);  // column-major
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) q[static_cast<std::size_t>(j) * m + i] = a[i * n + j];
  std::vector<double> y(b, b + m);
  std::vector<int> perm(n);
  for (int j = 0; j < n; ++j) perm[j] = j;
  const int kmax = std::min(m, n);
  int rank = 0;
  double r00 = 0.0;
  for (int k = 0; k < kmax; ++k) {
    int best = k;
    double bestn = -1.0;
    for (int j = k; j < n; ++j) {
      const double* cj = &q[static_cast<std::size_t>(j) * m];
      double s = 0.0;
      for (int i = k; i < m; ++i) s += cj[i] * cj[i];
      if (s > bestn) {
        bestn = s;
        best = j;
      }
    }
    if (best != k) {
      std::swap_ranges(q.begin() + static_cast<std::size_t>(k) * m,
                       q.begin() + static_cast<std::size_t>(k + 1) * m,
                       q.begin() + static_cast<std::size_t>(best) * m);
      std::swap(perm[k], perm[best]);
    }
    const double alpha = std::sqrt(std::max(bestn, 0.0));
    if (k == 0) r00 = alpha;
    if (!(alpha > 10.0 * EPS * std::max(m, n) * r00)) break;
    double* col = &q[static_cast<std::size_t>(k) * m];
    const double beta = (col[k] > 0.0) ? -alpha : alpha;
    col[k] -= beta;
    double vnorm2 = 0.0;
    for (int i = k; i < m; ++i) vnorm2 += col[i] * col[i];
    if (vnorm2 > 0.0) {
      for (int j = k + 1; j < n; ++j) {
        double* cj = &q[static_cast<std::size_t>(j) * m];
        double s = 0.0;
        for (int i = k; i < m; ++i) s += col[i] * cj[i];
        s = 2.0 * s / vnorm2;
        for (int i = k; i < m; ++i) cj[i] -= s * col[i];
      }
      double s = 0.0;
      for (int i = k; i < m; ++i) s += col[i] * y[i];
      s = 2.0 * s / vnorm2;
      for (int i = k; i < m; ++i) y[i] -= s * col[i];
    }
    col[k] = beta;
    ++rank;
  }
  std::vector<double> w(rank, 0.0), x(n, 0.0);
  for (int k = rank - 1; k >= 0; --k) {
    double s = y[k];
    for (int j = k + 1; j < rank; ++j) s -= q[static_cast<std::size_t>(j) * m + k] * w[j];
    w[k] = s / q[static_cast<std::size_t>(k) * m + k];
  }
  for (int k = 0; k < rank; ++k) x[perm[k]] = w[k];
  return x;
}

}  // namespace lls_internal

using namespace lls_internal;

std::string LinearLeastSquaresResult::get_message() const {
  return status == 0 ? "converged"
                     : "the iteration limit was reached before convergence";
}

LinearLeastSquaresResult nnls(const double* in_matrix, int n_rows, int n_cols,
                              const double* in_b, int n_b, int maxiter,
                              double tol) {
  check_shapes(n_rows, n_cols, n_b);
  const int m = n_rows, n = n_cols;
  if (maxiter <= 0) maxiter = 3 * n;
  std::vector<double> lb(n, 0.0);
  std::vector<double> ub(n, std::numeric_limits<double>::infinity());
  // Lawson & Hanson start with every variable on its (zero) bound.
  return active_set(in_matrix, m, n, in_b, lb, ub, std::vector<double>(n, 0.0),
                    std::vector<int>(n, AT_LOWER), maxiter, tol);
}

LinearLeastSquaresResult bvls(const double* in_matrix, int n_rows, int n_cols,
                              const double* in_b, int n_b,
                              const double* in_lower, int n_lower,
                              const double* in_upper, int n_upper, int maxiter,
                              double tol) {
  check_shapes(n_rows, n_cols, n_b);
  const int m = n_rows, n = n_cols;
  if (n_lower != n || n_upper != n)
    IMP_THROW("bvls: the bounds must have one entry per unknown", ValueException);
  std::vector<double> lb(in_lower, in_lower + n), ub(in_upper, in_upper + n);
  for (int j = 0; j < n; ++j) {
    if (std::isnan(lb[j]) || std::isnan(ub[j]) || lb[j] > ub[j])
      IMP_THROW("bvls: each lower bound must be <= its upper", ValueException);
  }
  if (maxiter <= 0) maxiter = std::max(100, n);

  // Start from the unconstrained solution projected onto the box, as
  // scipy's BVLS does; a variable whose projection hit a bound starts bound.
  std::vector<double> x = lstsq(in_matrix, m, n, in_b);
  std::vector<int> state(n, FREE);
  for (int j = 0; j < n; ++j) {
    if (x[j] <= lb[j]) {
      x[j] = lb[j];
      state[j] = AT_LOWER;
    } else if (x[j] >= ub[j]) {
      x[j] = ub[j];
      state[j] = AT_UPPER;
    }
  }
  return active_set(in_matrix, m, n, in_b, lb, ub, x, state, maxiter, tol);
}

IMPBFF_END_NAMESPACE
