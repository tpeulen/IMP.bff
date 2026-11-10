/**\file InferenceFitting.cpp
 * One generic fit of an evaluable factor graph (PRD-155). Written 2026-11-10 by
 * claude-code/Opus 5.5 for tpeulen.
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/fit/InferenceFitting.h>
#include <IMP/bff/fit/Optimization.h>
#include <IMP/bff/bayesian/BayesianLaplace.h>

#include <cmath>
#include <limits>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

InferenceFit inference_fitting(const InferenceFactorGraph& graph,
                                     const std::vector<double>& theta0, int max_iterations,
                                     double tolerance, double line_search_below) {
  if (!graph.get_is_evaluable()) {
    throw std::domain_error("inference_fitting: every factor needs a score");
  }
  if (max_iterations < 1 || !(tolerance > 0.0)) {
    throw std::invalid_argument("inference_fitting: max_iterations >= 1 and tolerance > 0");
  }
  const std::size_t dim = graph.get_parameter_dimension();
  std::vector<double> th = theta0;
  InferenceFit R;
  R.decrement = std::nan("");
  InferenceScore pt = graph.eval(th, true);
  //: the loop of bayesian_decay_fit_node, on the graph's sums: A is the information, grad the
  //: gradient of the log posterior, the objective minus the log posterior
  auto objective = [&](const double* t) {
    return -graph.eval(std::vector<double>(t, t + dim), false).log_density;
  };
  DampedNewton<decltype(objective)> stepper;
  stepper.line_search_below = line_search_below;
  int it = 0;
  for (; it < max_iterations; ++it) {
    const OptimizationStepResult r =
        stepper.step(pt.information.data(), pt.gradient.data(), dim, -pt.log_density, th.data(), objective);
    if (!r.accepted) { R.converged = std::isfinite(R.decrement) && R.decrement < 1e-3; break; }
    R.n_gradient_fallback += r.gradient_fallback ? 1 : 0;
    R.decrement = r.decrement;
    pt = graph.eval(th, true);
    if (R.decrement < tolerance) { R.converged = true; ++it; break; }
  }
  if (!R.converged && std::isfinite(R.decrement) && R.decrement < 1e-3) R.converged = true;
  R.iterations = it;
  double ev = 0.0;
  R.laplace_ok = bayesian_laplace_log_evidence(pt.log_density, pt.information.data(), dim, &ev);
  R.laplace_evidence = R.laplace_ok ? ev : -std::numeric_limits<double>::infinity();
  R.theta = th;
  R.at_mode = pt;
  return R;
}

IMPBFF_END_NAMESPACE
