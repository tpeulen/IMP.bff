/**
 *  \file IMP/bff/LinearLeastSquares.h
 *  \brief Linear least squares with sign and box constraints: NNLS and BVLS.
 *
 * `min ||A x - b||` subject to `x >= 0` (nnls(), Lawson & Hanson 1974,
 * chapter 23) or `lb <= x <= ub` (bvls(), Stark & Parker 1995). These are
 * the two solvers scipy offers as `scipy.optimize.nnls` and
 * `scipy.optimize.lsq_linear(..., method="bvls")`, and they are here so a
 * caller that needs one does not need scipy: ChiSurf's lifetime-amplitude
 * unmixing, Tikhonov-regularised distance inversions and phasor component
 * fractions all reduce to one of them.
 *
 * Both are active-set methods. Each iteration solves an unconstrained least
 * squares problem over the free ("passive") columns by Householder QR, so
 * the cost per iteration is `O(m p^2)` for `p` free columns and the answer
 * is accurate even when `A` is badly conditioned -- the normal equations
 * would square the condition number, which is the whole reason Lawson and
 * Hanson used QR.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_LINEARLEASTSQUARES_H
#define IMPBFF_LINEARLEASTSQUARES_H

#include <IMP/bff/bff_config.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The outcome of a constrained linear least-squares solve.
struct IMPBFFEXPORT LinearLeastSquaresResult {
  //! The solution.
  std::vector<double> x;
  //! `A x - b` at the solution.
  std::vector<double> residuals;
  //! `||A x - b||_2`, scipy's `rnorm`.
  double rnorm = 0.0;
  //! Active-set iterations used.
  int iterations = 0;
  //! 0 is converged; 1 is the iteration limit reached (`x` is the last
  //! feasible iterate, which is scipy's `nnls` `info == 3`).
  int status = 0;
  //! Per variable: 0 free, -1 at its lower bound, +1 at its upper bound.
  //! scipy's `lsq_linear(...).active_mask`.
  std::vector<int> active_mask;
  bool get_success() const { return status == 0; }
  std::string get_message() const;
};

//! Non-negative least squares: `min ||A x - b||_2` subject to `x >= 0`.
/*!
    Lawson & Hanson's algorithm NNLS, the one behind
    `scipy.optimize.nnls`, and the same answer to working precision on a
    problem with a unique solution.

    \param[in] in_matrix `A`, row-major `n_rows x n_cols`
    \param[in] n_rows `m`, the number of observations
    \param[in] n_cols `n`, the number of unknowns
    \param[in] in_b `b`, length `m`
    \param[in] n_b must equal \p n_rows
    \param[in] maxiter iteration limit; 0 or negative means `3 n`, scipy's
               default
    \param[in] tol the dual-feasibility tolerance below which a gradient
               component counts as zero; 0 or negative means
               `10 eps max(m, n) ||A||_1`, Lawson & Hanson's own choice
*/
IMPBFFEXPORT LinearLeastSquaresResult nnls(const double* in_matrix, int n_rows,
                                           int n_cols, const double* in_b,
                                           int n_b, int maxiter = 0,
                                           double tol = 0.0);

//! Bounded-variable least squares: `min ||A x - b||_2`, `lb <= x <= ub`.
/*!
    Stark & Parker's BVLS, the generalisation of NNLS to two-sided bounds
    and `scipy.optimize.lsq_linear(..., method="bvls")`. An infinite bound
    is no bound, so `lb = 0, ub = inf` is NNLS and both infinite is ordinary
    least squares.

    The start is the unconstrained solution projected onto the box, as
    scipy's BVLS does, which usually leaves few constraints to release.

    \param[in] in_lower length `n`; `-inf` for none
    \param[in] in_upper length `n`; `+inf` for none
    \param[in] maxiter 0 or negative means `max(100, n)`, scipy's `None`
*/
IMPBFFEXPORT LinearLeastSquaresResult bvls(
    const double* in_matrix, int n_rows, int n_cols, const double* in_b,
    int n_b, const double* in_lower, int n_lower, const double* in_upper,
    int n_upper, int maxiter = 0, double tol = 0.0);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_LINEARLEASTSQUARES_H */
