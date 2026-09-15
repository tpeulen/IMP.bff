/**
 *  \file IMP/bff/MaxEnt.h
 *  \brief Maximum-entropy solutions of linear problems.
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_MAXENT_H
#define IMPBFF_MAXENT_H

#include <IMP/bff/bff_config.h>

#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The answer of a maximum-entropy solve.
/*!
    The amplitudes `p` maximise the Skilling-Bryan entropy relative to the
    prior under the quadratic chi-square, both at the regularised optimum and
    at the early-stop maximum (the last iterate before the entropy weight
    stopped improving the objective). `converged` is the stopping test
    (`gradient_angle <= tol` at exit); `success` only says the solve ran to
    completion, which an exhausted iteration budget also does.
*/
class IMPBFFEXPORT MaxEntSolution {
 public:
  std::vector<double> amplitudes;
  double chisq = 0.0;
  double entropy = 0.0;
  //! chisq - nu * entropy at the amplitudes.
  double objective = 0.0;
  std::vector<double> early_stop_amplitudes;
  double early_stop_chisq = 0.0;
  double early_stop_entropy = 0.0;
  double early_stop_objective = 0.0;
  int iterations = 0;
  bool success = false;
  bool converged = false;
  //! The Skilling-Bryan angle between the chi-square and entropy gradients.
  double gradient_angle = 0.0;

  std::vector<double> get_amplitudes() const { return amplitudes; }
  std::vector<double> get_early_stop_amplitudes() const {
    return early_stop_amplitudes;
  }
};

//! Maximum entropy under a quadratic chi-square.
/*!
    `chi^2(p) = 1/2 p^T H p - g^T p + chisq_constant`, maximised against the
    entropy relative to `prior` (strictly positive; the entropy peaks at
    p = prior) with weight `nu`; `nu = 0` is the bound-constrained least
    squares solution.

    \param[in] hessian n x n, row-major
    \param[in] gradient the linear term g, length n
    \param[in] prior length n
    \param[in] max_iter, tol, min_prob the iteration cap, the tolerance on the
               gradient angle, and the positivity floor
*/
IMPBFFEXPORT MaxEntSolution maxent_solve(const std::vector<double>& hessian,
                                         const std::vector<double>& gradient,
                                         const std::vector<double>& prior,
                                         double chisq_constant, double nu,
                                         int max_iter = 200, double tol = 1e-4,
                                         double min_prob = 1e-12);

//! Maximum-entropy inversion of a linear model.
/*!
    Minimises `sum_i w_i (A x - b)_i^2 - nu^2 S(x; prior)` over `x >= 0`.

    \param[in] design A, n_rows x n_cols, row-major
    \param[in] measurements b, length n_rows
    \param[in] weights w (1/sigma^2 per row); empty weights every row 1
    \param[in] prior length n_cols, strictly positive; empty is uniform
*/
IMPBFFEXPORT MaxEntSolution maxent_invert(const std::vector<double>& design,
                                          const std::vector<double>& measurements,
                                          const std::vector<double>& weights,
                                          const std::vector<double>& prior,
                                          double nu, int n_rows, int n_cols,
                                          int max_iter = 500, double tol = 1e-8);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_MAXENT_H */
