# fit: Curve fitting: objectives, minimizers, factor-graph inference, MaxEnt

Objectives and minimizers (chi-squared, joint chi-squared, L-BFGS-B, NNLS/BVLS), the factor-graph inference core (canonical forms, Gaussian elimination), MaxEnt and spectrum grids.

- **Inputs:** curves (decay, FCS) and their weights.
- **Relations:** `bayesian`, `spectroscopy`, `fret` use it for their likelihoods.
- **Layout:** sources in `src/fit/`; the public headers stay flat in `include/` (IMP links only `include/*.h`) and are assigned to this theme in `src/fit/Headers.cmake`.

Public headers: `Convolution.h`, `FitChiSquared.h`, `FitDataset.h`, `FitJointChiSquared.h`, `FitMinimizer.h`, `FitObjective.h`, `FitStatistics.h`, `GeneralizedNormalCurve.h`, `InferenceCanonicalForm.h`, `InferenceFactorGraph.h`, `InferenceFactorGraphFromFit.h`, `InferenceGaussianElimination.h`, `LinearLeastSquares.h`, `MaxEnt.h`, `MaxEntSpectrum.h`, `Minimize.h`, `Optimization.h`, `SpectrumGrid.h`, `StripMask.h`.
