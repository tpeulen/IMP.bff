/**
 *  \file src/Odeint.cpp
 *  \brief odeint(), the scipy driver around LSODA.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/Odeint.h>
#include <IMP/bff/internal/Lsoda.h>
#include <IMP/exception.h>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

namespace odeint_impl {

// LSODA calls a plain C function with no user pointer; the function being
// integrated is reached through this, set for the duration of one odeint()
// call and restored afterwards (so a nested call would also work).
thread_local OdeFunction* current_function = nullptr;

void rhs_thunk(int* neq, double* t, double* y, double* ydot) {
  const int n = *neq;
  std::vector<double> yv(y, y + n);
  const std::vector<double> d = current_function->evaluate(yv, *t);
  if (static_cast<int>(d.size()) != n) {
    IMP_THROW("odeint: the function returned " << d.size()
                  << " values for " << n << " equations",
              ValueException);
  }
  std::copy(d.begin(), d.end(), ydot);
}

void jac_thunk(int*, double*, double*, int*, int*, double*, int*) {
  // jt = 2 differences the Jacobian internally; LSODA never calls this.
  IMP_THROW("odeint: the user Jacobian is not supported", ValueException);
}

struct FunctionScope {
  OdeFunction* previous;
  explicit FunctionScope(OdeFunction* f) : previous(current_function) {
    current_function = f;
  }
  ~FunctionScope() { current_function = previous; }
};

const char* message(int istate) {
  switch (istate) {
    case 2: return "Integration successful.";
    case 1: return "Nothing was done; the integration time was 0.";
    case -1: return "Excess work done on this call (perhaps wrong Dfun type).";
    case -2: return "Excess accuracy requested (tolerances too small).";
    case -3: return "Illegal input detected (internal error).";
    case -4: return "Repeated error test failures (internal error).";
    case -5:
      return "Repeated convergence failures (perhaps bad Jacobian or "
             "tolerances).";
    case -6: return "Error weight became zero during problem.";
    case -7: return "Internal workspace insufficient to finish (internal error).";
    case -8: return "Run terminated (internal error).";
  }
  return "";
}

}  // namespace odeint_impl

OdeintResult odeint(OdeFunction* f, const std::vector<double>& y0,
                    const std::vector<double>& t,
                    const std::vector<double>& rtol_in,
                    const std::vector<double>& atol_in,
                    const std::vector<double>& tcrit_in, double h0,
                    double hmax, double hmin, int ixpr, int mxstep,
                    int mxhnil, int mxordn, int mxords) {
  using namespace odeint_impl;
  if (f == nullptr) IMP_THROW("odeint: no function", ValueException);
  const int neq = static_cast<int>(y0.size());
  const long ntimes = static_cast<long>(t.size());
  if (neq == 0) IMP_THROW("odeint: y0 is empty", ValueException);
  if (mxordn < 0) IMP_THROW("Incorrect value for mxordn.", ValueException);
  if (mxords < 0) IMP_THROW("Incorrect value for mxords.", ValueException);

  const int jt = 2;  // full Jacobian, differenced internally
  const int ml = 0, mu = 0;

  // setup_extra_inputs: scalar or per-equation tolerances -> itol 1..4.
  int itol = 0;
  std::vector<double> rtol(1, 1.49012e-8), atol(1, 1.49012e-8);
  const char* tol_error =
      "Tolerances must be an array of the same length as the\n     number of "
      "equations or a scalar.";
  if (!rtol_in.empty()) {
    if (rtol_in.size() == 1) {
      rtol = rtol_in;
    } else if (static_cast<int>(rtol_in.size()) == neq) {
      rtol = rtol_in;
      itol |= 2;
    } else {
      IMP_THROW(tol_error, ValueException);
    }
  }
  if (!atol_in.empty()) {
    if (atol_in.size() == 1) {
      atol = atol_in;
    } else if (static_cast<int>(atol_in.size()) == neq) {
      atol = atol_in;
      itol |= 1;
    } else {
      IMP_THROW(tol_error, ValueException);
    }
  }
  ++itol;
  std::vector<double> tcrit(tcrit_in);
  const long numcrit = static_cast<long>(tcrit.size());

  // compute_lrw_liw
  const int lmat = neq * neq + 2;
  const int nyh = neq;
  const int lrn = 20 + nyh * (mxordn + 1) + 3 * neq;
  const int lrs = 20 + nyh * (mxords + 1) + 3 * neq + lmat;
  const int lrw = std::max(lrn, lrs);
  const int liw = 20 + neq;
  std::vector<double> rwork(lrw, 0.0);
  std::vector<int> iwork(liw, 0);
  iwork[0] = ml;
  iwork[1] = mu;
  int iopt = 0;
  if (h0 != 0.0 || hmax != 0.0 || hmin != 0.0 || ixpr != 0 || mxstep != 0 ||
      mxhnil != 0 || mxordn != 0 || mxords != 0) {
    rwork[4] = h0;
    rwork[5] = hmax;
    rwork[6] = hmin;
    iwork[4] = ixpr;
    iwork[5] = mxstep;
    iwork[6] = mxhnil;
    iwork[7] = mxordn;
    iwork[8] = mxords;
    iopt = 1;
  }

  OdeintResult r;
  r.n_times = static_cast<int>(ntimes);
  r.n_equations = neq;
  r.y.assign(static_cast<std::size_t>(ntimes) * neq,
             std::numeric_limits<double>::quiet_NaN());
  const long out_sz = ntimes > 0 ? ntimes - 1 : 0;
  r.hu.assign(out_sz, 0.0);
  r.tcur.assign(out_sz, 0.0);
  r.tolsf.assign(out_sz, 0.0);
  r.tsw.assign(out_sz, 0.0);
  r.nst.assign(out_sz, 0);
  r.nfe.assign(out_sz, 0);
  r.nje.assign(out_sz, 0);
  r.nqu.assign(out_sz, 0);
  r.mused.assign(out_sz, 0);

  std::vector<double> y(y0);
  double tt = 0.0;
  long t0count = 0;
  if (ntimes > 0) {
    tt = t[0];
    t0count = 1;
    while (t0count < ntimes && t[t0count] == tt) ++t0count;
  }
  for (long k = 0; k < t0count; ++k)
    std::copy(y.begin(), y.end(), r.y.begin() + k * neq);

  int itask = 1;
  int istate = 1;
  long crit_ind = 0;
  if (numcrit > 0) {
    itask = 4;
    rwork[0] = tcrit[0];
  }
  internal::lsoda::lsoda_common_struct_t S = internal::lsoda::lsoda_common_struct_t();
  FunctionScope scope(f);
  long k = t0count;
  while (k < ntimes && istate > 0) {
    double tout = t[k];
    if (itask == 4) {
      if (tout > tcrit[crit_ind]) {
        ++crit_ind;
        if (crit_ind < numcrit) rwork[0] = tcrit[crit_ind];
      }
    }
    if (crit_ind >= numcrit) itask = 1;
    internal::lsoda::lsoda(rhs_thunk, neq, y.data(), &tt, &tout, itol,
                           rtol.data(), atol.data(), &itask, &istate, &iopt,
                           rwork.data(), lrw, iwork.data(), liw, jac_thunk, jt,
                           &S);
    r.hu[k - 1] = rwork[10];
    r.tcur[k - 1] = rwork[12];
    r.tolsf[k - 1] = rwork[13];
    r.tsw[k - 1] = rwork[14];
    r.nst[k - 1] = iwork[10];
    r.nfe[k - 1] = iwork[11];
    r.nje[k - 1] = iwork[12];
    r.nqu[k - 1] = iwork[13];
    r.imxer = (istate == -5 || istate == -4) ? iwork[15] : -1;
    r.lenrw = iwork[16];
    r.leniw = iwork[17];
    r.mused[k - 1] = iwork[18];
    // scipy copies y even for the failing call, then stops.
    std::copy(y.begin(), y.end(), r.y.begin() + k * neq);
    ++k;
  }
  r.istate = istate;
  r.success = istate > 0;
  r.message = message(istate);
  return r;
}

IMPBFF_END_NAMESPACE
