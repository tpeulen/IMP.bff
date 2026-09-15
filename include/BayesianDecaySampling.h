/**
 * \file IMP/bff/BayesianDecaySampling.h
 * \brief The Bayesian decay posterior sampled: No-U-Turn chains (tttrlib `NoUTurnSampler`) on the
 *        decay model's log posterior and gradient, and the convergence analysis of summaries
 *        (tttrlib `McmcDiagnostics`).
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 * PRD-146 (2026-09-15, ucfret prompt 463). The two samplers bff had (blocked Metropolis, ensemble
 * slice) did not mix on the CBM56 posterior (PRD-145); this one uses the analytic gradient.
 */
#ifndef IMPBFF_BAYESIANDECAYSAMPLING_H
#define IMPBFF_BAYESIANDECAYSAMPLING_H

#include <IMP/bff/BayesianDecayPosterior.h>

#if __has_include("pocketfft/pocketfft_hdronly.h")

#include <IMP/bff/internal/McmcDiagnostics.h>
#include <IMP/bff/internal/NoUTurnSampler.h>
#include <cstdint>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! \\name Sampling the Bayesian decay posterior
//! @{

//! How one chain is run.
struct BayesianDecayNutsOptions {
  int warmup = 500;              //!< transitions with step-size adaptation, discarded
  int draws = 1000;              //!< transitions recorded
  int max_depth = 10;            //!< tree depth limit (at most 2^depth leapfrog steps)
  double target_accept = 0.8;    //!< dual-averaging target of the acceptance statistic
  std::uint64_t seed = 1;
};

//! One chain: the recorded states and the sampler's per-draw statistics.
struct BayesianDecayChain {
  std::vector<std::vector<double>> theta;          //!< draws x dim
  std::vector<double> log_posterior, accept_stat, energy;
  std::vector<int> tree_depth, n_leapfrog;
  std::vector<char> divergent;
  double step_size = 0.0;
  int warmup_divergences = 0;
};

/**
 * \brief One NUTS chain on the decay posterior from `start`, with `inverse_metric` (a covariance,
 *        dim x dim; the Laplace covariance at a mode is the natural choice) as the dense metric.
 */
inline BayesianDecayChain bayesian_decay_nuts_chain(const BayesianDecayExperiment& f, const BayesianDecayTensors& env,
                                                    const std::vector<double>& start, const std::vector<double>& inverse_metric,
                                                    const BayesianDecayNutsOptions& opt = BayesianDecayNutsOptions()) {
  const std::size_t dim = f.dim;
  ::tttrlib::NoUTurnSampler s(dim, [&](const std::vector<double>& q, std::vector<double>& g) {
    return bayesian_decay_log_posterior_and_gradient(f, env, q, g);
  }, opt.seed);
  s.set_inverse_metric(inverse_metric);
  s.set_max_depth(opt.max_depth);
  s.set_target_accept(opt.target_accept);
  s.set_state(start);
  s.init_step_size();
  BayesianDecayChain out;
  s.begin_adaptation();
  for (int k = 0; k < opt.warmup; ++k) out.warmup_divergences += s.transition().divergent ? 1 : 0;
  s.end_adaptation();
  out.step_size = s.step_size();
  for (int k = 0; k < opt.draws; ++k) {
    const ::tttrlib::NutsTransition t = s.transition();
    out.theta.push_back(t.q);
    out.log_posterior.push_back(t.log_density);
    out.accept_stat.push_back(t.accept_stat);
    out.energy.push_back(t.energy);
    out.tree_depth.push_back(t.tree_depth);
    out.n_leapfrog.push_back(t.n_leapfrog);
    out.divergent.push_back(t.divergent ? 1 : 0);
  }
  return out;
}

//! The convergence analysis of one scalar summary over chains (draws of equal length).
struct BayesianConvergence {
  double mean = 0.0, sd = 0.0, mcse = 0.0, rhat = 0.0, ess_bulk = 0.0, ess_tail = 0.0, tau = 0.0;
};

//! R-hat (rank-normalised, bulk and tail), bulk and tail ESS, MCSE of the mean, and the integrated
//! autocorrelation time per chain `tau = draws per chain * chains / ess_mean`.
inline BayesianConvergence bayesian_convergence(const std::vector<std::vector<double>>& chains) {
  BayesianConvergence c;
  std::size_t N = 0;
  for (const auto& ch : chains) for (double v : ch) { c.mean += v; ++N; }
  c.mean /= double(N);
  for (const auto& ch : chains) for (double v : ch) c.sd += (v - c.mean) * (v - c.mean);
  c.sd = std::sqrt(c.sd / double(N - 1));
  c.rhat = ::tttrlib::rhat_rank(chains);
  c.ess_bulk = ::tttrlib::ess_bulk(chains);
  c.ess_tail = ::tttrlib::ess_tail(chains);
  c.mcse = ::tttrlib::mcse_mean(chains);
  c.tau = double(N) / ::tttrlib::ess_mean(chains);
  return c;
}

//! @}

IMPBFF_END_NAMESPACE

#endif  // __has_include("pocketfft/pocketfft_hdronly.h")

#endif /* IMPBFF_BAYESIANDECAYSAMPLING_H */
