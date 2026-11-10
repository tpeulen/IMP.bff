/**
 *  \file IMP/bff/fit/InferenceFitting.h
 *  \brief One generic fit of an evaluable factor graph: the mode and its Laplace evidence (PRD-155).
 *
 *  Written 2026-11-10 by claude-code/Opus 5.5 for tpeulen.
 *
 *  The graph (InferenceFactorGraph with an InferenceFactorScore on every
 *  factor) is the log posterior: the sum of its factors. This finds its mode by
 *  damped Newton steps on the summed gradient and information -- Fisher scoring
 *  when the factors give expected information -- with a line search on the log
 *  posterior itself (`DampedNewton`, the stepper `bayesian_decay_fit_node` uses),
 *  and reports the Laplace approximation of the evidence at the mode,
 *  `log p(y, theta*) + d/2 log 2 pi - 1/2 log det I`.
 *
 *  Nothing here knows a model: the same call fits a Bayesian decay analysis, a
 *  global fit of several datasets, or a two-factor toy, and the log posterior
 *  comes back split by factor and by kind (likelihood, prior, hyper).
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_INFERENCEFITTING_H
#define IMPBFF_INFERENCEFITTING_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/fit/InferenceFactorGraph.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! What `inference_fitting` found.
struct IMPBFFEXPORT InferenceFit {
  std::vector<double> theta;          //!< the mode
  InferenceScore at_mode;        //!< the log posterior there, by factor and kind, with gradient and information
  bool converged = false;             //!< the Newton decrement fell below the tolerance
  int iterations = 0;
  double decrement = 0.0;             //!< the last accepted step's `grad . step`
  int n_gradient_fallback = 0;        //!< steps the stepper had to take along the gradient
  double laplace_evidence = 0.0;      //!< log evidence of the Laplace approximation at the mode
  bool laplace_ok = false;            //!< the information was positive definite there
};

//! Fit the mode of an evaluable graph from `theta0`.
/*! Stops when the decrement is below `tolerance` (then `converged`), when a step
    is not accepted (converged if the last decrement was below 1e-3, as
    `bayesian_decay_fit_node` decides), or after `max_iterations`.
    \throws std::domain_error if the graph is not evaluable */
IMPBFFEXPORT InferenceFit inference_fitting(const InferenceFactorGraph& graph,
                                                  const std::vector<double>& theta0,
                                                  int max_iterations = 300,
                                                  double tolerance = 1e-6,
                                                  double line_search_below = 1.0);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_INFERENCEFITTING_H */
