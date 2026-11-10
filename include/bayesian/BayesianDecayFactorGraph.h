/**
 *  \file IMP/bff/bayesian/BayesianDecayFactorGraph.h
 *  \brief The Bayesian decay model as an evaluable factor graph (PRD-155).
 *
 *  Written 2026-11-10 by claude-code/Opus 5.5 for tpeulen.
 *
 *  A Bayesian decay analysis turns time-resolved fluorescence decays -- photon
 *  counts per arrival-time bin after each laser pulse, one histogram per sample
 *  and detector -- into the posterior of the donor-acceptor distance
 *  distribution p(R), every calibration constant carrying a prior. The model is
 *  an experiment directory (`manifest.json` plus float64 arrays, the format
 *  `bayesian_decay_experiment_load` reads).
 *
 *  bayesian_decay_factor_graph() returns that posterior as an
 *  InferenceFactorGraph whose factors carry scores: no model class, the
 *  factors are the model. Fit it with `inference_fitting`, query its structure
 *  like any other graph, add factors to it (another dataset, another
 *  measurement) like any other graph.
 *
 *  Factors:
 *  - `counts` (LIKELIHOOD): the Poisson log-likelihood of every histogram, with
 *    the expected (Fisher) information; predict() = expected counts of all
 *    histograms concatenated, get_data() = the counts.
 *  - `prior` (PRIOR): the log prior of every variable, the roughness prior of
 *    p(R) included, with minus its Hessian.
 *  - `observation:<name>` (LIKELIHOOD, scope `c`): each external FRET measurement in the
 *    manifest's `observations`, in the docking restraint's vocabulary (fps.json):
 *    `{name, distance_type, distance, error_neg, error_pos, forster_radius}` with
 *    `distance_type` `RDAMean` (<R_DA>), `RDAMeanE` (R_E, the efficiency-averaged
 *    distance) or `Efficiency` (<E>). The model's value is
 *    `distance_sample_statistics` of p(R) at `R = (R/R0) * forster_radius`, the
 *    score `-chi2_score / 2` (asymmetric errors) with the split normal's
 *    normalisation -- the same statistic and score docking uses. An ensemble FRET
 *    efficiency measured elsewhere is `Efficiency` with its error on both sides.
 *    predict() = the model's value, get_data() = the measured one.
 *
 *  The histograms and the priors are one factor each, not one per histogram or
 *  per variable: the model evaluates all histograms in one pass (shared
 *  amplitudes) and all priors in one function, and that is what this exposes;
 *  splitting them is a refinement for later and changes no number.
 *
 *  Variables are the manifest's, keyed by name, at their offsets in theta, with
 *  their free size and with the prior family as role.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_BAYESIANDECAYFACTORGRAPH_H
#define IMPBFF_BAYESIANDECAYFACTORGRAPH_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/fit/InferenceFactorGraph.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The posterior of the experiment in `directory` as an evaluable factor graph.
/*! \throws std::domain_error in a build without the Bayesian decay code (no pocketfft) */
IMPBFFEXPORT InferenceFactorGraph bayesian_decay_factor_graph(const std::string& directory);

//! p(R) at `theta` on the experiment's grid (bayesian_decay_distance_grid).
IMPBFFEXPORT std::vector<double> bayesian_decay_distribution_at(const std::string& directory,
                                                                const std::vector<double>& theta);

//! The experiment's distance grid, R/R0.
IMPBFFEXPORT std::vector<double> bayesian_decay_distance_grid(const std::string& directory);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_BAYESIANDECAYFACTORGRAPH_H */
