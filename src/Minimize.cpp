/**
 *  \file src/Minimize.cpp
 *  \brief L-BFGS-B and Nelder-Mead drivers, shaped as scipy.optimize's.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/Minimize.h>
#include <IMP/bff/internal/Lbfgsb.h>
#include <IMP/exception.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

// Named, not anonymous: bff builds as one unity translation unit, where
// every anonymous namespace is the same one.
namespace minimize_impl {

// The simplex and finite-difference arithmetic mirrors NumPy expressions term
// for term, and NumPy never fuses `a * b - c * d` into an FMA. Scoped to the
// function bodies: a file-level pragma would leak into the rest of the unity
// build.
#if defined(__clang__)
#define BFF_MINIMIZE_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define BFF_MINIMIZE_NO_CONTRACT
#endif

const double kMinimizeInf = std::numeric_limits<double>::infinity();
const double kMinimizeEps = std::numeric_limits<double>::epsilon();

//! `lower`/`upper` expanded to n entries, `-inf`/`inf` where not given.
void expand_bounds(const std::vector<double>& lower,
                   const std::vector<double>& upper, std::size_t n,
                   std::vector<double>* lb, std::vector<double>* ub,
                   bool* bounded, const char* who) {
  *bounded = !(lower.empty() && upper.empty());
  lb->assign(n, -kMinimizeInf);
  ub->assign(n, kMinimizeInf);
  if (!lower.empty()) {
    if (lower.size() != n) {
      IMP_THROW(who << ": length of x0 != length of the lower bounds",
                ValueException);
    }
    for (std::size_t i = 0; i < n; ++i)
      if (!std::isnan(lower[i])) (*lb)[i] = lower[i];
  }
  if (!upper.empty()) {
    if (upper.size() != n) {
      IMP_THROW(who << ": length of x0 != length of the upper bounds",
                ValueException);
    }
    for (std::size_t i = 0; i < n; ++i)
      if (!std::isnan(upper[i])) (*ub)[i] = upper[i];
  }
}

double clip(double v, double lo, double hi) {
  // np.clip(v, lo, hi) == minimum(maximum(v, lo), hi)
  double r = v < lo ? lo : v;
  if (std::isnan(v)) r = v;
  return r > hi ? hi : r;
}

// --------------------------------------------------------------- L-BFGS-B

//! SciPy's ScalarFunction: memoised f and g at the last x, nfev/ngev counts.
class ScalarFunction {
 public:
  ScalarFunction(MinimizeObjective* f, bool use_gradient, double eps,
                 const std::vector<double>& lb, const std::vector<double>& ub)
      : f_(f), use_gradient_(use_gradient), eps_(eps), lb_(lb), ub_(ub) {
    unbounded_ = true;
    for (std::size_t i = 0; i < lb.size(); ++i)
      if (!(lb[i] == -kMinimizeInf && ub[i] == kMinimizeInf)) unbounded_ = false;
  }

  //! Evaluate f and g at `x` (cached when `x` is the last point).
  void fun_and_grad(const std::vector<double>& x, double* fx,
                    std::vector<double>* gx) {
    if (!have_ || x != x_) {
      x_ = x;
      have_ = true;
      update();
    }
    *fx = f_x_;
    *gx = g_x_;
  }

  int nfev() const { return nfev_; }
  int ngev() const { return ngev_; }

 private:
  void update() {
    BFF_MINIMIZE_NO_CONTRACT
    const std::size_t n = x_.size();
    if (use_gradient_) {
      std::vector<double> fg = f_->evaluate_with_gradient(x_);
      if (fg.size() != n + 1) {
        IMP_THROW("minimize_lbfgsb: evaluate_with_gradient must return "
                      << n + 1 << " values (f, then the gradient), got "
                      << fg.size(),
                  ValueException);
      }
      ++nfev_;
      ++ngev_;
      f_x_ = fg[0];
      g_x_.assign(fg.begin() + 1, fg.end());
      return;
    }
    f_x_ = f_->evaluate(x_);
    ++nfev_;
    // scipy.optimize._numdiff.approx_derivative, '2-point', abs_step = eps.
    std::vector<double> h(n, eps_);
    for (std::size_t i = 0; i < n; ++i) {
      if ((x_[i] + h[i]) - x_[i] == 0.0) {
        const double sign = x_[i] >= 0.0 ? 1.0 : -1.0;
        h[i] = std::sqrt(kMinimizeEps) * sign * std::max(1.0, std::fabs(x_[i]));
      }
    }
    if (!unbounded_) {
      // _adjust_scheme_to_bounds(x0, h, 1, '1-sided', lb, ub)
      for (std::size_t i = 0; i < n; ++i) {
        const double lower_dist = x_[i] - lb_[i];
        const double upper_dist = ub_[i] - x_[i];
        const double xh = x_[i] + h[i];
        const bool violated = (xh < lb_[i]) || (xh > ub_[i]);
        const bool fitting =
            std::fabs(h[i]) <= std::max(lower_dist, upper_dist);
        double hi = h[i];
        if (violated && fitting) hi = -hi;
        if (!fitting) {
          if (upper_dist >= lower_dist) {
            hi = upper_dist;
          } else {
            hi = -lower_dist;
          }
        }
        h[i] = hi;
      }
    }
    g_x_.assign(n, 0.0);
    std::vector<double> x1(x_);
    for (std::size_t i = 0; i < n; ++i) {
      x1[i] = x_[i] + h[i];
      const double f1 = f_->evaluate(x1);
      x1[i] = x_[i];
      const double dx = (x_[i] + h[i]) - x_[i];
      g_x_[i] = (f1 - f_x_) / dx;
    }
    nfev_ += static_cast<int>(n);
    ++ngev_;
  }

  MinimizeObjective* f_;
  bool use_gradient_;
  double eps_;
  std::vector<double> lb_, ub_;
  bool unbounded_;
  bool have_ = false;
  std::vector<double> x_;
  double f_x_ = 0.0;
  std::vector<double> g_x_;
  int nfev_ = 0;
  int ngev_ = 0;
};

const char* lbfgsb_status_message(int s) {
  switch (s) {
    case 0: return "START";
    case 1: return "NEW_X";
    case 2: return "RESTART";
    case 3: return "FG";
    case 4: return "CONVERGENCE";
    case 5: return "STOP";
    case 6: return "WARNING";
    case 7: return "ERROR";
    case 8: return "ABNORMAL";
  }
  return "";
}

const char* lbfgsb_task_message(int t) {
  switch (t) {
    case 401: return "NORM OF PROJECTED GRADIENT <= PGTOL";
    case 402: return "RELATIVE REDUCTION OF F <= FACTR*EPSMCH";
    case 501: return "CPU EXCEEDING THE TIME LIMIT";
    case 502: return "TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT";
    case 503: return "PROJECTED GRADIENT IS SUFFICIENTLY SMALL";
    case 504: return "TOTAL NO. OF ITERATIONS REACHED LIMIT";
    case 505: return "CALLBACK REQUESTED HALT";
    case 601: return "ROUNDING ERRORS PREVENT PROGRESS";
    case 602: return "STP = STPMAX";
    case 603: return "STP = STPMIN";
    case 604: return "XTOL TEST SATISFIED";
    case 701: return "NO FEASIBLE SOLUTION";
    case 702: return "FACTR < 0";
    case 703: return "FTOL < 0";
    case 704: return "GTOL < 0";
    case 705: return "XTOL < 0";
    case 706: return "STP < STPMIN";
    case 707: return "STP > STPMAX";
    case 708: return "STPMIN < 0";
    case 709: return "STPMAX < STPMIN";
    case 710: return "INITIAL G >= 0";
    case 711: return "M <= 0";
    case 712: return "N <= 0";
    case 713: return "INVALID NBD";
  }
  return "";
}

// ------------------------------------------------------------ Nelder-Mead

//! np.argsort order for the simplex values: ascending, NaN last, stable.
std::vector<std::size_t> argsort(const std::vector<double>& v) {
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
    const double x = v[a], y = v[b];
    if (std::isnan(x)) return false;
    if (std::isnan(y)) return true;
    return x < y;
  });
  return idx;
}

}  // namespace minimize_impl

MinimizeResult minimize_lbfgsb(MinimizeObjective* objective,
                               const std::vector<double>& x0,
                               const std::vector<double>& lower,
                               const std::vector<double>& upper,
                               bool use_gradient, int maxcor, double ftol,
                               double gtol, double eps, int maxfun,
                               int maxiter, int maxls) {
  using namespace minimize_impl;
  const char* who = "minimize_lbfgsb";
  if (objective == nullptr) IMP_THROW(who << ": no objective", ValueException);
  const int n = static_cast<int>(x0.size());
  if (n == 0) IMP_THROW(who << ": x0 is empty", ValueException);
  if (!(maxls > 0)) IMP_THROW("maxls must be positive.", ValueException);
  const int m = maxcor;

  std::vector<double> lb, ub;
  bool bounded = false;
  expand_bounds(lower, upper, x0.size(), &lb, &ub, &bounded, who);
  std::vector<double> x(x0);
  if (bounded) {
    for (int i = 0; i < n; ++i) {
      if (lb[i] > ub[i]) {
        IMP_THROW("LBFGSB - one of the lower bounds is greater than an "
                  "upper bound.",
                  ValueException);
      }
      x[i] = clip(x[i], lb[i], ub[i]);
    }
  }

  // SciPy builds the ScalarFunction -- and with it f and g at x0 -- before
  // the first setulb call, which then asks for exactly that point.
  ScalarFunction sf(objective, use_gradient, eps, lb, ub);
  double f = 0.0;
  std::vector<double> g(n, 0.0);
  sf.fun_and_grad(x, &f, &g);

  std::vector<int> nbd(n, 0);
  std::vector<double> low(n, 0.0), upp(n, 0.0);
  for (int i = 0; i < n; ++i) {
    const bool has_l = !std::isinf(lb[i]);
    const bool has_u = !std::isinf(ub[i]);
    if (has_l) low[i] = lb[i];
    if (has_u) upp[i] = ub[i];
    nbd[i] = has_l ? (has_u ? 2 : 1) : (has_u ? 3 : 0);
  }

  const double factr = ftol / kMinimizeEps;
  const double pgtol = gtol;
  std::vector<double> wa(2 * static_cast<std::size_t>(m) * n + 5 * n +
                         11 * static_cast<std::size_t>(m) * m + 8 * m, 0.0);
  std::vector<int> iwa(3 * n, 0);
  int task[2] = {0, 0};
  int ln_task[2] = {0, 0};
  int lsave[4] = {0, 0, 0, 0};
  int isave[44] = {0};
  double dsave[29] = {0.0};

  int n_iterations = 0;
  double fval = 0.0;
  while (true) {
    internal::lbfgsb::setulb(n, m, x.data(), low.data(), upp.data(),
                             nbd.data(), &fval, g.data(), factr, pgtol,
                             wa.data(), iwa.data(), task, lsave, isave, dsave,
                             maxls, ln_task);
    if (task[0] == 3) {
      sf.fun_and_grad(x, &fval, &g);
    } else if (task[0] == 1) {
      ++n_iterations;
      if (n_iterations >= maxiter) {
        task[0] = 5;
        task[1] = 504;
      } else if (sf.nfev() > maxfun) {
        task[0] = 5;
        task[1] = 502;
      }
    } else {
      break;
    }
  }

  MinimizeResult r;
  if (task[0] == 4) {
    r.status = 0;
  } else if (sf.nfev() > maxfun || n_iterations >= maxiter) {
    r.status = 1;
  } else {
    r.status = 2;
  }
  r.success = r.status == 0;
  r.message = std::string(lbfgsb_status_message(task[0])) + ": " +
              lbfgsb_task_message(task[1]);
  r.x = x;
  r.fun = fval;
  r.jac = g;
  r.nit = n_iterations;
  r.nfev = sf.nfev();
  r.njev = sf.ngev();
  const int n_updates = isave[30];
  r.n_corrections = std::min(n_updates, m);
  r.correction_s.assign(wa.begin(),
                        wa.begin() + static_cast<long>(r.n_corrections) * n);
  r.correction_y.assign(
      wa.begin() + static_cast<long>(m) * n,
      wa.begin() + static_cast<long>(m) * n +
          static_cast<long>(r.n_corrections) * n);
  (void)f;
  return r;
}

MinimizeResult minimize_nelder_mead(MinimizeObjective* objective,
                                    const std::vector<double>& x0_in,
                                    const std::vector<double>& lower,
                                    const std::vector<double>& upper,
                                    int maxiter_in, int maxfev_in,
                                    double xatol, double fatol, bool adaptive,
                                    const std::vector<double>& initial_simplex) {
  BFF_MINIMIZE_NO_CONTRACT
  using namespace minimize_impl;
  const char* who = "minimize_nelder_mead";
  if (objective == nullptr) IMP_THROW(who << ": no objective", ValueException);
  std::vector<double> x0(x0_in);
  const std::size_t N = x0.size();
  if (N == 0) IMP_THROW(who << ": x0 is empty", ValueException);

  double rho, chi, psi, sigma;
  if (adaptive) {
    const double dim = static_cast<double>(N);
    rho = 1;
    chi = 1 + 2 / dim;
    psi = 0.75 - 1 / (2 * dim);
    sigma = 1 - 1 / dim;
  } else {
    rho = 1;
    chi = 2;
    psi = 0.5;
    sigma = 0.5;
  }
  const double nonzdelt = 0.05;
  const double zdelt = 0.00025;

  std::vector<double> lb, ub;
  bool bounded = false;
  expand_bounds(lower, upper, N, &lb, &ub, &bounded, who);
  if (bounded) {
    for (std::size_t i = 0; i < N; ++i) {
      if (lb[i] > ub[i]) {
        IMP_THROW("Nelder Mead - one of the lower bounds is greater than an "
                  "upper bound.",
                  ValueException);
      }
    }
    for (std::size_t i = 0; i < N; ++i) x0[i] = clip(x0[i], lb[i], ub[i]);
  }

  // sim[k] is row k of the (N + 1) x N simplex.
  std::vector<std::vector<double>> sim(N + 1, std::vector<double>(N));
  if (initial_simplex.empty()) {
    sim[0] = x0;
    for (std::size_t k = 0; k < N; ++k) {
      std::vector<double> y(x0);
      if (y[k] != 0) {
        y[k] = (1 + nonzdelt) * y[k];
      } else {
        y[k] = zdelt;
      }
      sim[k + 1] = y;
    }
  } else {
    if (initial_simplex.size() != (N + 1) * N) {
      IMP_THROW("`initial_simplex` should be an array of shape (N+1,N)",
                ValueException);
    }
    for (std::size_t k = 0; k <= N; ++k)
      for (std::size_t j = 0; j < N; ++j) sim[k][j] = initial_simplex[k * N + j];
  }

  const long kUnlimited = std::numeric_limits<long>::max();
  long maxiter = maxiter_in > 0 ? maxiter_in : 0;
  long maxfun = maxfev_in > 0 ? maxfev_in : 0;
  if (maxiter == 0 && maxfun == 0) {
    maxiter = static_cast<long>(N) * 200;
    maxfun = static_cast<long>(N) * 200;
  } else if (maxiter == 0) {
    maxiter = kUnlimited;
  } else if (maxfun == 0) {
    maxfun = kUnlimited;
  }

  if (bounded) {
    for (auto& row : sim) {
      for (std::size_t j = 0; j < N; ++j) {
        if (row[j] > ub[j]) row[j] = 2 * ub[j] - row[j];
        row[j] = clip(row[j], lb[j], ub[j]);
      }
    }
  }
  auto clip_row = [&](std::vector<double>* v) {
    if (!bounded) return;
    for (std::size_t j = 0; j < N; ++j) (*v)[j] = clip((*v)[j], lb[j], ub[j]);
  };

  long fcalls = 0;
  // _wrap_scalar_function_maxfun_validation: refuse once the budget is spent.
  auto func = [&](const std::vector<double>& v, double* fx) -> bool {
    if (fcalls >= maxfun) return false;
    ++fcalls;
    *fx = objective->evaluate(v);
    return true;
  };
  std::vector<double> fsim(N + 1, kMinimizeInf);
  auto sort_simplex = [&]() {
    const std::vector<std::size_t> ind = argsort(fsim);
    std::vector<std::vector<double>> s2(N + 1);
    std::vector<double> f2(N + 1);
    for (std::size_t k = 0; k <= N; ++k) {
      s2[k] = sim[ind[k]];
      f2[k] = fsim[ind[k]];
    }
    sim.swap(s2);
    fsim.swap(f2);
  };

  for (std::size_t k = 0; k <= N; ++k) {
    double fx;
    if (!func(sim[k], &fx)) break;
    fsim[k] = fx;
  }
  sort_simplex();
  sort_simplex();

  long iterations = 1;
  std::vector<double> xbar(N), xr(N), xe(N), xc(N), xcc(N);
  while (fcalls < maxfun && iterations < maxiter) {
    // The body of SciPy's `try:`; `false` from func() is the
    // _MaxFuncCallError, which skips the rest of the iteration.
    bool converged = false;
    [&]() {
      double max_dx = 0.0, max_df = 0.0;
      bool nan_dx = false, nan_df = false;
      for (std::size_t k = 1; k <= N; ++k) {
        for (std::size_t j = 0; j < N; ++j) {
          const double d = std::fabs(sim[k][j] - sim[0][j]);
          if (std::isnan(d)) nan_dx = true;
          else if (d > max_dx) max_dx = d;
        }
        const double d = std::fabs(fsim[0] - fsim[k]);
        if (std::isnan(d)) nan_df = true;
        else if (d > max_df) max_df = d;
      }
      if (!nan_dx && !nan_df && max_dx <= xatol && max_df <= fatol) {
        converged = true;
        return;
      }
      for (std::size_t j = 0; j < N; ++j) {
        double s = 0.0;
        for (std::size_t k = 0; k < N; ++k) s += sim[k][j];
        xbar[j] = s / static_cast<double>(N);
      }
      for (std::size_t j = 0; j < N; ++j)
        xr[j] = (1 + rho) * xbar[j] - rho * sim[N][j];
      clip_row(&xr);
      double fxr;
      if (!func(xr, &fxr)) return;
      bool doshrink = false;
      if (fxr < fsim[0]) {
        for (std::size_t j = 0; j < N; ++j)
          xe[j] = (1 + rho * chi) * xbar[j] - rho * chi * sim[N][j];
        clip_row(&xe);
        double fxe;
        if (!func(xe, &fxe)) return;
        if (fxe < fxr) {
          sim[N] = xe;
          fsim[N] = fxe;
        } else {
          sim[N] = xr;
          fsim[N] = fxr;
        }
      } else {
        if (fxr < fsim[N - 1]) {
          sim[N] = xr;
          fsim[N] = fxr;
        } else {
          if (fxr < fsim[N]) {
            for (std::size_t j = 0; j < N; ++j)
              xc[j] = (1 + psi * rho) * xbar[j] - psi * rho * sim[N][j];
            clip_row(&xc);
            double fxc;
            if (!func(xc, &fxc)) return;
            if (fxc <= fxr) {
              sim[N] = xc;
              fsim[N] = fxc;
            } else {
              doshrink = true;
            }
          } else {
            for (std::size_t j = 0; j < N; ++j)
              xcc[j] = (1 - psi) * xbar[j] + psi * sim[N][j];
            clip_row(&xcc);
            double fxcc;
            if (!func(xcc, &fxcc)) return;
            if (fxcc < fsim[N]) {
              sim[N] = xcc;
              fsim[N] = fxcc;
            } else {
              doshrink = true;
            }
          }
          if (doshrink) {
            for (std::size_t k = 1; k <= N; ++k) {
              for (std::size_t j = 0; j < N; ++j)
                sim[k][j] = sim[0][j] + sigma * (sim[k][j] - sim[0][j]);
              clip_row(&sim[k]);
              double fk;
              if (!func(sim[k], &fk)) return;
              fsim[k] = fk;
            }
          }
        }
      }
      ++iterations;
    }();
    if (converged) break;
    sort_simplex();
  }

  MinimizeResult r;
  r.x = sim[0];
  bool any_nan = false;
  double fval = kMinimizeInf;
  for (double v : fsim) {
    if (std::isnan(v)) any_nan = true;
    else if (v < fval) fval = v;
  }
  r.fun = any_nan ? std::numeric_limits<double>::quiet_NaN() : fval;
  if (fcalls >= maxfun) {
    r.status = 1;
    r.message = "Maximum number of function evaluations has been exceeded.";
  } else if (iterations >= maxiter) {
    r.status = 2;
    r.message = "Maximum number of iterations has been exceeded.";
  } else {
    r.status = 0;
    r.message = "Optimization terminated successfully.";
  }
  r.success = r.status == 0;
  r.nit = static_cast<int>(std::min<long>(iterations, std::numeric_limits<int>::max()));
  r.nfev = static_cast<int>(std::min<long>(fcalls, std::numeric_limits<int>::max()));
  r.final_simplex.reserve((N + 1) * N);
  for (const auto& row : sim) r.final_simplex.insert(r.final_simplex.end(), row.begin(), row.end());
  r.final_simplex_values = fsim;
  return r;
}

#undef BFF_MINIMIZE_NO_CONTRACT

RootResult root_brentq(MinimizeObjective* f, double a, double b, double xtol,
                       double rtol, int maxiter) {
  // scipy/optimize/Zeros/brentq.c, kept line for line so the iterates are
  // SciPy's; only the callback and the result record differ.
  if (!f) IMP_THROW("root_brentq: no function", ValueException);
  if (xtol <= 0) IMP_THROW("root_brentq: xtol too small (" << xtol << " <= 0)", ValueException);
  if (rtol < 8.881784197001252e-16) {
    IMP_THROW("root_brentq: rtol too small (" << rtol << " < 8.88e-16)", ValueException);
  }
  std::vector<double> xv(1);
  auto eval = [&](double x) {
    xv[0] = x;
    return f->evaluate(xv);
  };
  RootResult r;
  double xpre = a, xcur = b;
  double xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis;
  double delta, stry, dpre, dblk;
  fpre = eval(xpre);
  fcur = eval(xcur);
  r.function_calls = 2;
  if (fpre == 0) {
    r.root = xpre; r.converged = true; r.flag = "converged";
    return r;
  }
  if (fcur == 0) {
    r.root = xcur; r.converged = true; r.flag = "converged";
    return r;
  }
  if (std::signbit(fpre) == std::signbit(fcur)) {
    IMP_THROW("f(a) and f(b) must have different signs", ValueException);
  }
  for (int i = 0; i < maxiter; i++) {
    r.iterations++;
    if (fpre != 0 && fcur != 0 && (std::signbit(fpre) != std::signbit(fcur))) {
      xblk = xpre;
      fblk = fpre;
      spre = scur = xcur - xpre;
    }
    if (std::fabs(fblk) < std::fabs(fcur)) {
      xpre = xcur; xcur = xblk; xblk = xpre;
      fpre = fcur; fcur = fblk; fblk = fpre;
    }
    delta = (xtol + rtol * std::fabs(xcur)) / 2;
    sbis = (xblk - xcur) / 2;
    if (fcur == 0 || std::fabs(sbis) < delta) {
      r.root = xcur; r.converged = true; r.flag = "converged";
      return r;
    }
    if (std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre)) {
      if (xpre == xblk) {
        stry = -fcur * (xcur - xpre) / (fcur - fpre);  // interpolate
      } else {  // extrapolate
        dpre = (fpre - fcur) / (xpre - xcur);
        dblk = (fblk - fcur) / (xblk - xcur);
        stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
      }
      if (2 * std::fabs(stry) < std::min(std::fabs(spre), 3 * std::fabs(sbis) - delta)) {
        spre = scur;  // good short step
        scur = stry;
      } else {
        spre = sbis;  // bisect
        scur = sbis;
      }
    } else {
      spre = sbis;  // bisect
      scur = sbis;
    }
    xpre = xcur; fpre = fcur;
    if (std::fabs(scur) > delta) {
      xcur += scur;
    } else {
      xcur += (sbis > 0 ? delta : -delta);
    }
    fcur = eval(xcur);
    r.function_calls++;
  }
  r.root = xcur; r.converged = false; r.flag = "convergence error";
  return r;
}

IMPBFF_END_NAMESPACE
