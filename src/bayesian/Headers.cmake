# Public headers of the `bayesian` theme. IMP links only include/*.h and
# include/internal/*.h into its build tree, so headers stay flat and this
# list is what assigns them (test/test_theme_headers.py checks it).
set(imp_bff_bayesian_headers "CausalLinearGaussian.h;CounterfactualDistanceNetwork.h;CounterfactualMarkovChain.h;BayesianDecayModel.h;BayesianDecayPosterior.h;BayesianDecaySampling.h;BayesianDeltaMethod.h;BayesianFisherScoring.h;BayesianLaplace.h;BayesianMeasuredResponse.h;BayesianPSpline.h;BayesianTransferTensors.h;BayesianTransformedGaussianPrior.h;BayesianTransforms.h;Distributions.h;MCMCSampler.h;NutsKernel.h;SamplerDiagnostics.h;SamplerKernels.h;SamplerWarmup.h;Sampling.h")
# Sources without a public header of their own.
set(imp_bff_bayesian_private_sources "")
