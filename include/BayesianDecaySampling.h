/**
 * \file IMP/bff/BayesianDecaySampling.h
 * \brief The Bayesian decay posterior as a SamplingTarget: run any gradient kernel on it through
 *        run_sampler (Sampling.h) -- NUTS (NutsKernel.h) -- and summarise with SamplerDiagnostics.h.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 * PRD-146 (2026-09-15, ucfret prompt 463) sampled this posterior with a NUTS chain of its own; PRD-147
 * step 5 replaced that chain loop and its convergence summary by the one sampler interface: this
 * header now only says what the decay posterior is as a target. The posterior is unconstrained (every
 * variable carries its own transform) and has an analytic gradient, so every kernel whose registry
 * entry declares `requires_gradient` can run on it.
 */
#ifndef IMPBFF_BAYESIANDECAYSAMPLING_H
#define IMPBFF_BAYESIANDECAYSAMPLING_H

#include <IMP/bff/BayesianDecayPosterior.h>

#if __has_include("pocketfft/pocketfft_hdronly.h")

#include <IMP/bff/NutsKernel.h>
#include <IMP/bff/SamplerDiagnostics.h>
#include <IMP/bff/Sampling.h>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! \\name Sampling the Bayesian decay posterior
//! @{

/**
 * \brief The decay posterior as a SamplingTarget: log posterior and its analytic gradient in the
 *        unconstrained coordinates.
 *
 * The target refers to \p f and \p env; both must outlive every run on it.
 */
inline SamplingTarget bayesian_decay_sampling_target(const BayesianDecayExperiment& f, const BayesianDecayTensors& env) {
  SamplingTarget t = sampling_target(f.dim, std::function<double(const std::vector<double>&, std::vector<double>&)>(
                                                [&f, &env](const std::vector<double>& q, std::vector<double>& g) {
                                                  return bayesian_decay_log_posterior_and_gradient(f, env, q, g);
                                                }));
  return t;
}

//! @}

IMPBFF_END_NAMESPACE

#endif  // __has_include("pocketfft/pocketfft_hdronly.h")

#endif /* IMPBFF_BAYESIANDECAYSAMPLING_H */
