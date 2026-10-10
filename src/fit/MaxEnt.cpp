/**
 * \file MaxEnt.cpp
 * \brief Maximum-entropy solutions of linear problems.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/MaxEnt.h>

#include <IMP/bff/internal/MaxEntQp.h>

#include <stdexcept>
#include <string>

IMPBFF_BEGIN_NAMESPACE

namespace {

MaxEntSolution solution_from(const internal::MaxEntResult& r) {
  MaxEntSolution s;
  s.amplitudes = r.p;
  s.chisq = r.chisq;
  s.entropy = r.S;
  s.objective = r.Q;
  s.early_stop_amplitudes = r.p_esm;
  s.early_stop_chisq = r.chisq_esm;
  s.early_stop_entropy = r.S_esm;
  s.early_stop_objective = r.Q_esm;
  s.iterations = r.niter;
  s.success = r.success;
  s.converged = r.converged;
  s.gradient_angle = r.dgrad;
  return s;
}

void require_size(const std::vector<double>& v, std::size_t n, const char* what) {
  if (v.size() != n) {
    throw std::domain_error(std::string("maxent: ") + what + " has " +
                                std::to_string(v.size()) + " values, expected " +
                                std::to_string(n));
  }
}

}  // namespace

MaxEntSolution maxent_solve(const std::vector<double>& hessian,
                            const std::vector<double>& gradient,
                            const std::vector<double>& prior,
                            double chisq_constant, double nu, int max_iter,
                            double tol, double min_prob) {
  const std::size_t n = gradient.size();
  require_size(hessian, n * n, "the Hessian");
  require_size(prior, n, "the prior");
  return solution_from(internal::run_mem(hessian, gradient, prior, chisq_constant,
                                        nu, max_iter, tol, min_prob));
}

MaxEntSolution maxent_invert(const std::vector<double>& design,
                             const std::vector<double>& measurements,
                             const std::vector<double>& weights,
                             const std::vector<double>& prior, double nu,
                             int n_rows, int n_cols, int max_iter, double tol) {
  if (n_rows <= 0 || n_cols <= 0) {
    throw std::domain_error("maxent: the design needs rows and columns");
  }
  require_size(design, std::size_t(n_rows) * std::size_t(n_cols), "the design");
  require_size(measurements, std::size_t(n_rows), "the measurements");
  if (!weights.empty()) require_size(weights, std::size_t(n_rows), "the weights");
  if (!prior.empty()) require_size(prior, std::size_t(n_cols), "the prior");
  std::vector<double> hessian, gradient;
  double constant = 0.0;
  internal::build_normal_equations(design, measurements, weights, n_rows, n_cols,
                                  hessian, gradient, constant);
  const std::vector<double> m =
      prior.empty() ? std::vector<double>(std::size_t(n_cols), 1.0) : prior;
  // H = 2 A^T W A in build_normal_equations, so nu^2 S on ||Ax-b||^2 is
  // 2 nu^2 on the engine's 1/2 p^T H p form.
  return solution_from(internal::run_mem(hessian, gradient, m, constant,
                                        2.0 * nu * nu, max_iter, tol, 1e-12));
}

IMPBFF_END_NAMESPACE
