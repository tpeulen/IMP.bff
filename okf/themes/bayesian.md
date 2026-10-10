# bayesian: Bayesian inference: decay posteriors, samplers, causal models

Bayesian layer: decay model and posterior, Laplace and delta-method bands, Fisher scoring, MCMC samplers (stretch, slice, DE, NUTS, ensemble, blocked), diagnostics and warm-up, causal linear-Gaussian and counterfactual networks.

- **Inputs:** curves, priors, parameter vectors.
- **Relations:** The sampler registry (`util/Registry`) dispatches by name, never by hard-coded algorithm.
- **Layout:** sources in `src/bayesian/`; the public headers stay flat in `include/` (IMP links only `include/*.h`) and are assigned to this theme in `src/bayesian/Headers.cmake`.

Public headers: `CausalLinearGaussian.h`, `CounterfactualDistanceNetwork.h`, `CounterfactualMarkovChain.h`, `BayesianDecayModel.h`, `BayesianDecayPosterior.h`, `BayesianDecaySampling.h`, `BayesianDeltaMethod.h`, `BayesianFisherScoring.h`, `BayesianLaplace.h`, `BayesianMeasuredResponse.h`, `BayesianPSpline.h`, `BayesianTransferTensors.h`, `BayesianTransformedGaussianPrior.h`, `BayesianTransforms.h`, `Distributions.h`, `MCMCSampler.h`, `NutsKernel.h`, `SamplerDiagnostics.h`, `SamplerKernels.h`, `SamplerWarmup.h`, `Sampling.h`.
