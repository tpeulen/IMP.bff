/**
 *  \file IMP/bff/Minimize.h
 *  \brief Bounded scalar minimisation: L-BFGS-B and Nelder-Mead.
 *
 * The two `scipy.optimize.minimize` methods ChiSurf uses, here so a caller
 * that needs one does not need SciPy:
 *
 * - minimize_lbfgsb(): L-BFGS-B 3.0 (Byrd, Lu, Nocedal and Zhu 1995), run
 *   through SciPy's own C translation of the algorithm
 *   (internal/Lbfgsb.h), driven exactly as `scipy.optimize._lbfgsb_py`
 *   drives it -- the same defaults, the same `factr = ftol / eps`, the same
 *   stopping tests and messages. Without a gradient the objective is
 *   differenced forward with SciPy's absolute step `eps` (1e-8), flipped or
 *   shortened at a bound the way `scipy.optimize._numdiff` does, so the
 *   gradients -- and therefore the iterates -- are SciPy's.
 * - root_brentq(): `scipy.optimize.brentq` (what `root_scalar` with a
 *   bracket runs), ported from SciPy's C.
 * - minimize_nelder_mead(): the downhill simplex of
 *   `scipy.optimize._minimize_neldermead`, step for step: the same initial
 *   simplex (5 % / 0.00025 perturbations), reflection, expansion,
 *   contraction and shrink, clipping to the bounds, and `xatol`/`fatol`
 *   convergence.
 *
 * The objective is a MinimizeObjective, a director: Python subclasses it
 * (ChiSurf's `chisurf.core.math.numerics.minimize` does) and each evaluation
 * crosses the language boundary once.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_MINIMIZE_H
#define IMPBFF_MINIMIZE_H

#include <IMP/bff/bff_config.h>
#include <IMP/Object.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A scalar objective, optionally with its gradient.
/*! Override evaluate(); a caller that has an analytic gradient also
    overrides evaluate_with_gradient() and passes `use_gradient = true` to
    minimize_lbfgsb(). */
class IMPBFFEXPORT MinimizeObjective : public IMP::Object {
 public:
  explicit MinimizeObjective(std::string name = "MinimizeObjective%1%")
      : IMP::Object(name) {}

  //! The objective at `x`.
  virtual double evaluate(const std::vector<double>& x) { return 0.0; }

  //! `[f, df/dx_0, ..., df/dx_{n-1}]` at `x`, in one call.
  virtual std::vector<double> evaluate_with_gradient(
      const std::vector<double>& x) {
    return std::vector<double>();
  }

  IMP_OBJECT_METHODS(MinimizeObjective);
};

//! What a minimisation found, shaped like SciPy's `OptimizeResult`.
struct IMPBFFEXPORT MinimizeResult {
  //! The solution.
  std::vector<double> x;
  //! The objective at the solution.
  double fun = 0.0;
  //! The gradient at the solution (L-BFGS-B only; empty for Nelder-Mead).
  std::vector<double> jac;
  //! Iterations.
  int nit = 0;
  //! Objective evaluations, finite-difference evaluations included.
  int nfev = 0;
  //! Gradient evaluations (L-BFGS-B only).
  int njev = 0;
  //! SciPy's `status`: 0 converged, 1 an evaluation or iteration limit,
  //! 2 anything else (L-BFGS-B) / the iteration limit (Nelder-Mead).
  int status = 0;
  //! `status == 0`.
  bool success = false;
  //! SciPy's message, verbatim.
  std::string message;
  //! L-BFGS-B: the correction pairs behind the inverse-Hessian
  //! approximation, `n_corrections x n` row-major each (`hess_inv` in SciPy).
  std::vector<double> correction_s;
  std::vector<double> correction_y;
  int n_corrections = 0;
  //! Nelder-Mead: the final simplex, `(n + 1) x n` row-major, sorted by value.
  std::vector<double> final_simplex;
  std::vector<double> final_simplex_values;
};

//! Minimise `objective` over the box `[lower, upper]` with L-BFGS-B.
/*!
    \param[in] objective the function to minimise.
    \param[in] x0 the start; clipped into the box first, as SciPy does.
    \param[in] lower,upper the box; empty means unbounded, and `-inf`/`inf`
               entries leave that side open.
    \param[in] use_gradient call evaluate_with_gradient() for the gradient
               (SciPy's `jac=True`); otherwise difference evaluate().
    \param[in] maxcor,ftol,gtol,eps,maxfun,maxiter,maxls SciPy's options,
               with SciPy's defaults.
*/
IMPBFFEXPORT MinimizeResult minimize_lbfgsb(
    MinimizeObjective* objective, const std::vector<double>& x0,
    const std::vector<double>& lower = std::vector<double>(),
    const std::vector<double>& upper = std::vector<double>(),
    bool use_gradient = false, int maxcor = 10,
    double ftol = 2.2204460492503131e-09, double gtol = 1e-5,
    double eps = 1e-8, int maxfun = 15000, int maxiter = 15000,
    int maxls = 20);

//! Minimise `objective` with the Nelder-Mead simplex, clipped to the box.
/*!
    \param[in] maxiter,maxfev limits; a value `<= 0` means "not given", with
               SciPy's rule: neither given -> both `200 n`; one given -> the
               other unlimited.
    \param[in] initial_simplex `(n + 1) x n` row-major; empty means SciPy's
               default simplex around `x0`.
*/
IMPBFFEXPORT MinimizeResult minimize_nelder_mead(
    MinimizeObjective* objective, const std::vector<double>& x0,
    const std::vector<double>& lower = std::vector<double>(),
    const std::vector<double>& upper = std::vector<double>(),
    int maxiter = 0, int maxfev = 0, double xatol = 1e-4,
    double fatol = 1e-4, bool adaptive = false,
    const std::vector<double>& initial_simplex = std::vector<double>());

//! What a bracketed root search found, shaped like SciPy's `RootResults`.
struct IMPBFFEXPORT RootResult {
  //! The root estimate.
  double root = 0.0;
  //! Iterations.
  int iterations = 0;
  //! Function evaluations.
  int function_calls = 0;
  //! True when the tolerance was met.
  bool converged = false;
  //! SciPy's flag: "converged" or "convergence error".
  std::string flag;
};

//! A root of the scalar `f` in `[a, b]` by Brent's method.
/*! SciPy's `brentq` (Charles Harris' C, `scipy/optimize/Zeros/brentq.c`),
    step for step, with its defaults. `f` is a MinimizeObjective evaluated
    at the one-element vector `[x]`. Throws ValueException when `f(a)` and
    `f(b)` have the same sign, as SciPy raises ValueError.
*/
IMPBFFEXPORT RootResult root_brentq(MinimizeObjective* f, double a, double b,
                                    double xtol = 2e-12,
                                    double rtol = 8.881784197001252e-16,
                                    int maxiter = 100);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_MINIMIZE_H */
