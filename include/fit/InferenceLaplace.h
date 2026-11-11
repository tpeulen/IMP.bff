/**
 *  \file IMP/bff/fit/InferenceLaplace.h
 *  \brief The curvature, covariance, exact-curvature polish and evidence, and Tierney-Kadane moments
 *         of any evaluable factor graph at a point (PRD-156 step 4).
 *
 *  Written 2026-11-11 by claude-code/Opus 5.5 for tpeulen ("BayesianDecayPosterior - I do not like
 *  it in bff, change make generic"). These were written for the Bayesian decay model only
 *  (`bayesian_decay_hessian`, `_covariance`, `_polish_exact`, `_laplace_evidence_exact`,
 *  `_fit_node_tilted`, `_tk_mean_rel`); here they take a graph, so they serve every model built as
 *  factors. Every function takes the `fixed` variables of `inference_fitting`: the curvature,
 *  covariance and evidence are then over the free coordinates only (a hyperparameter held).
 *
 *  Matrices are row-major over the FREE coordinates, in theta order.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_INFERENCELAPLACE_H
#define IMPBFF_INFERENCELAPLACE_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/fit/InferenceFactorGraph.h>
#include <IMP/bff/fit/InferenceFitting.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The positions of the free coordinates of theta, in order (all but the `fixed` variables).
/*! \throws std::invalid_argument for an unknown variable */
IMPBFFEXPORT std::vector<int> inference_free_coordinates(const InferenceFactorGraph& graph,
                                                         const std::vector<std::string>& fixed =
                                                             std::vector<std::string>());

//! The exact curvature of minus the log posterior, by central differences of the graph's gradient.
/*! `H_ij = -(g_i(theta + h e_j) - g_i(theta - h e_j)) / 2h`, `h = rel * max(1, |theta_j|)`,
    symmetrised; `2 n_free` gradient evaluations. The factors' information is Fisher's (expected)
    where a likelihood gives it; this is the observed curvature, which defines the Laplace
    approximation -- they differ where a mode is not a perfect fit (PRD-143 A4.9: 11.7 nats of
    evidence on CBM56). */
IMPBFFEXPORT std::vector<double> inference_hessian(const InferenceFactorGraph& graph,
                                                   const std::vector<double>& theta,
                                                   const std::vector<std::string>& fixed =
                                                       std::vector<std::string>(),
                                                   double rel = 1e-5);

//! The Laplace covariance at theta: the inverse of the information (or, `exact`, of inference_hessian).
/*! Empty if the matrix is not positive definite. */
IMPBFFEXPORT std::vector<double> inference_covariance(const InferenceFactorGraph& graph,
                                                      const std::vector<double>& theta,
                                                      const std::vector<std::string>& fixed =
                                                          std::vector<std::string>(),
                                                      bool exact = false);

//! The Laplace log evidence at theta: `log p(y, theta) + n/2 log 2 pi - 1/2 log det A` over the free coordinates.
/*! `A` is the factors' summed information, or with `exact` inference_hessian. `-inf` if `A` is not
    positive definite. */
IMPBFFEXPORT double inference_laplace_evidence(const InferenceFactorGraph& graph,
                                               const std::vector<double>& theta,
                                               const std::vector<std::string>& fixed =
                                                   std::vector<std::string>(),
                                               bool exact = false);

//! Polish a fit with the exact curvature: damped Newton steps on inference_hessian.
/*! A scoring fit stops where its Fisher decrement is small, which is not always a point where the
    exact curvature is negative definite -- and the exact-curvature evidence is only defined there.
    Stops at decrement `tolerance` or after `max_iterations` steps; `laplace_evidence` of the result
    is the exact-curvature one, `iterations` adds the polish steps to the fit's. */
IMPBFFEXPORT InferenceFit inference_polish(const InferenceFactorGraph& graph, const InferenceFit& fit,
                                           int max_iterations = 20, double tolerance = 1e-8);

//! Tierney-Kadane moments of a positive functional of theta, with what the two tilted fits did.
struct IMPBFFEXPORT InferenceTierneyKadane {
  double mean = 0.0, sd = 0.0;
  bool ok = false;
  int iterations = 0;                  //!< both tilted fits together
  //! per power (1, 2): the tilted mode's distance from the mode in the mode's metric (1 = one sd)
  std::vector<double> shift;
  //! per power: the log-determinant the tilt added
  std::vector<double> dlogdet;
  //! the smallest eigenvalue of the mode's curvature, and per power the tilt's curvature along that
  //! direction relative to it; where this ratio is not small the correction is the volume of a
  //! direction the data do not constrain, not a better moment (ucfret 2026-09-06; PRD-149)
  double flat_curvature = 0.0;
  std::vector<double> flat_ratio;
  //! the fraction of that direction's curvature supplied by the PRIOR and HYPER factors alone
  double flat_prior_share = 0.0;
};

//! The posterior mean and sd of a positive functional f(theta) by Tierney & Kadane (1986).
/*! `functional` is a score over `scope` (variable keys) whose eval returns `[f, df/dx, -d2f/dx2]`
    -- the layout of a factor score, with f in place of the log density and its exact second
    derivative (negated) in place of the information. Two tilted fits, `power * log(f + eps)` added
    as a factor for power 1 and 2, start at `mode.theta` (the graph's mode, from inference_fitting,
    with its `fixed` variables held); both tilted integrals get the mode's own curvature plus the
    tilt's exact curvature at the tilted mode (`bayesian_tierney_kadane`, BayesianLaplace.h). The
    base curvature is the mode's information, or with `exact_curvature` inference_hessian. */
IMPBFFEXPORT InferenceTierneyKadane inference_tierney_kadane(const InferenceFactorGraph& graph,
                                                             const InferenceFit& mode,
                                                             InferenceFactorScore* functional,
                                                             const std::vector<std::string>& scope,
                                                             double eps = 1e-6, int max_iterations = 100,
                                                             bool exact_curvature = false);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_INFERENCELAPLACE_H */
