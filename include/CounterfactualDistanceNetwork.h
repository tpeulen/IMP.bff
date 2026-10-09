/**
 *  \file IMP/bff/CounterfactualDistanceNetwork.h
 *  \brief Can a correct analysis of a distance network give the wrong answer?
 *         Site biases as exogenous variables, abducted, intervened on and
 *         replayed.
 *
 *  A distance network (FRET pairs, crosslinks, any label-to-label distance)
 *  is compared with a set of candidate structures: a table of model distances
 *  (candidates x pairs). The usual analysis scores each candidate by
 *  chi^2 = sum_p (y_p - M_fp)^2 / sigma_p^2 and picks the best. When one label
 *  misbehaves -- a dye that sticks, a linker the accessible-volume model gets
 *  wrong, kappa^2 far from 2/3 -- every pair that uses the site carries the
 *  same offset, and the fit absorbs it into the structure.
 *
 *  This class makes the assumption explicit. Every site s gets a bias
 *  b_s ~ N(0, tau_s^2), an exogenous variable, and a pair (i, j) reads
 *
 *      y_p = M_fp + b_i + b_j + eps_p,     eps_p ~ N(0, sigma_p^2).
 *
 *  Integrating the biases out gives each candidate a Gaussian marginal
 *  likelihood with covariance S = diag(sigma^2) + A diag(tau^2) A^T (A the
 *  pair-site incidence); network redundancy -- a site in several pairs -- is
 *  what makes b_s partly identifiable. On top of that posterior:
 *
 *  - **abduction**: E[b | y] per candidate and mixed over candidates, and the
 *    residual noise of the best candidate;
 *  - **do(b = 0)**: the measurement this experiment would have produced with
 *    ideal labels (same noise), y - A E[b | y];
 *  - **hinge**: do(b_s = 0) for one site at a time, then the usual analysis --
 *    which single site the conclusion depends on;
 *  - **counterfactual replay**: had the structure been candidate f', with this
 *    experiment's abducted biases and noise, which candidate would the usual
 *    analysis report? Against a level-2 replay with fresh noise and no bias,
 *    this shows where *this* data set is blind.
 *
 *  A bias at a site that appears in one or two pairs is not distinguishable
 *  from structure; tau then decides, and the posterior stays broad, which is
 *  the honest answer. See okf/counterfactuals.md section 3.9.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_COUNTERFACTUALDISTANCENETWORK_H
#define IMPBFF_COUNTERFACTUALDISTANCENETWORK_H

#include <IMP/bff/bff_config.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Site-bias abduction, interventions and replay for a distance network.
class IMPBFFEXPORT CounterfactualDistanceNetwork {
 public:
  //! \param[in] model row-major (n_candidates, n_pairs) model distances
  //! \param[in] n_candidates number of candidate structures
  //! \param[in] pair_sites the two site indices of every pair, flat
  //!            (a0, b0, a1, b1, ...); a site index is in [0, n_sites)
  //! \param[in] n_sites number of labelling sites
  CounterfactualDistanceNetwork(const std::vector<double>& model, int n_candidates,
                                const std::vector<int>& pair_sites, int n_sites);

  int get_number_of_candidates() const;
  int get_number_of_pairs() const;
  int get_number_of_sites() const;

  //! The measured distances and their error bars, one per pair.
  void set_measurement(const std::vector<double>& values, const std::vector<double>& sigma);
  //! Prior width of each site's bias (the same unit as the distances).
  void set_site_bias_prior(const std::vector<double>& tau);
  //! Prior probability of each candidate (default: uniform).
  void set_candidate_prior(const std::vector<double>& prior);

  //! Posterior over candidates: the usual analysis (biases assumed zero) or with biases integrated out.
  std::vector<double> get_candidate_posterior(bool bias_aware = true) const;
  //! The usual analysis' posterior over candidates for any measurement vector.
  std::vector<double> get_standard_posterior(const std::vector<double>& values) const;
  //! chi^2 of the usual analysis for every candidate.
  std::vector<double> get_chi2() const;

  //! Abduction: E[b | y, candidate], row-major (n_candidates, n_sites).
  std::vector<double> get_site_biases_given_candidates() const;
  //! Abduction: E[b | y], mixed over the bias-aware candidate posterior.
  std::vector<double> get_abducted_site_biases() const;
  //! Posterior standard deviation of each site's bias (mixture over candidates).
  std::vector<double> get_abducted_site_bias_sd() const;
  //! The noise of the most probable candidate: y - M_f - A E[b | y, f].
  std::vector<double> get_abducted_noise() const;

  //! do(b = 0): the measurement with ideal labels and this experiment's noise.
  std::vector<double> get_ideal_measurement() const;
  //! do(b_s = 0) one site at a time: the usual analysis' posterior, row-major (n_sites, n_candidates).
  std::vector<double> get_hinge_posteriors() const;

  //! Counterfactual replay: for every candidate f', the usual analysis' answer
  //! for M_f' + A E[b | y, f_hat] + eps_hat (this experiment's biases and noise).
  std::vector<int> get_counterfactual_replay() const;
  //! Level-2 replay: the usual analysis' answer for M_f' + fresh noise, no bias.
  std::vector<int> get_fresh_noise_replay(unsigned int seed) const;

 private:
  void check_ready(const char* what) const;
  int n_candidates_, n_pairs_, n_sites_;
  std::vector<double> model_;
  std::vector<int> pair_sites_;
  std::vector<double> y_, sigma_, tau_, prior_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_COUNTERFACTUALDISTANCENETWORK_H
