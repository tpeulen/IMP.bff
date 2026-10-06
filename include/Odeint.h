/**
 *  \file IMP/bff/Odeint.h
 *  \brief odeint: LSODA, stiff-or-not, on a grid of output times.
 *
 * `scipy.integrate.odeint`, here so a caller that needs it does not need
 * SciPy. It runs SciPy's own C translation of ODEPACK's LSODA
 * (internal/Lsoda.h) exactly as `scipy/integrate/_odepackmodule.c` drives
 * it: the same work-array sizes, defaults (rtol = atol = 1.49012e-8,
 * mxordn 12, mxords 5), output loop, critical times and full-output
 * diagnostics. LSODA switches between Adams (non-stiff) and BDF (stiff)
 * methods on its own, which is why it, and not an explicit Runge-Kutta, is
 * the replacement: a fast pre-equilibrium in a kinetic scheme makes the
 * system stiff, and an explicit method then needs orders of magnitude more
 * steps.
 *
 * The Jacobian is always the internally differenced full matrix (`jt = 2`,
 * `odeint` without `Dfun`).
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_ODEINT_H
#define IMPBFF_ODEINT_H

#include <IMP/bff/bff_config.h>
#include <IMP/Object.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The right-hand side `dy/dt = f(y, t)` of an ODE system.
class IMPBFFEXPORT OdeFunction : public IMP::Object {
 public:
  explicit OdeFunction(std::string name = "OdeFunction%1%")
      : IMP::Object(name) {}

  //! `dy/dt` at `(y, t)`; must return `y.size()` values.
  virtual std::vector<double> evaluate(const std::vector<double>& y,
                                       double t) {
    return std::vector<double>();
  }

  IMP_OBJECT_METHODS(OdeFunction);
};

//! What odeint() returns: the solution and scipy's full-output `infodict`.
struct IMPBFFEXPORT OdeintResult {
  //! `y(t[i])`, `len(t) x len(y0)` row-major. Rows after a failure are NaN.
  std::vector<double> y;
  int n_times = 0;
  int n_equations = 0;
  //! LSODA's final `istate`: 2 success; negative a failure (see `message`).
  int istate = 0;
  bool success = false;
  //! scipy's `_msgs[istate]`, verbatim.
  std::string message;
  //! Per output time after the first (length `len(t) - 1`), as scipy's
  //! infodict: last step size, time reached, tolerance scale factor, last
  //! method switch, steps, f evaluations, Jacobian evaluations, last order,
  //! last method (1 Adams, 2 BDF).
  std::vector<double> hu, tcur, tolsf, tsw;
  std::vector<int> nst, nfe, nje, nqu, mused;
  //! The component with the largest error on an error-test or convergence
  //! failure (istate -4/-5), else -1; work-array lengths used.
  int imxer = -1;
  int lenrw = 0;
  int leniw = 0;
};

//! Integrate `dy/dt = f(y, t)` from `t[0]`, reporting `y` at every `t`.
/*!
    \param[in] f the right-hand side.
    \param[in] y0 the state at `t[0]`.
    \param[in] t output times, monotonic; `y(t[0]) = y0`.
    \param[in] rtol,atol empty for scipy's 1.49012e-8, one value, or one per
               equation.
    \param[in] tcrit critical times the integrator must not step past.
    \param[in] h0,hmax,hmin,ixpr,mxstep,mxhnil,mxordn,mxords scipy's
               options, with scipy's defaults (0 means LSODA's own default).
*/
IMPBFFEXPORT OdeintResult odeint(
    OdeFunction* f, const std::vector<double>& y0,
    const std::vector<double>& t,
    const std::vector<double>& rtol = std::vector<double>(),
    const std::vector<double>& atol = std::vector<double>(),
    const std::vector<double>& tcrit = std::vector<double>(), double h0 = 0.0,
    double hmax = 0.0, double hmin = 0.0, int ixpr = 0, int mxstep = 0,
    int mxhnil = 0, int mxordn = 12, int mxords = 5);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_ODEINT_H */
