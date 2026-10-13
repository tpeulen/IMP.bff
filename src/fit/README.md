# fit: Curve fitting: objectives, minimizers, factor-graph inference, MaxEnt

Objectives and minimizers (chi-squared, joint chi-squared, L-BFGS-B, NNLS/BVLS), the factor-graph inference core (canonical forms, Gaussian elimination), MaxEnt and spectrum grids.

- **Inputs:** curves (decay, FCS) and their weights.
- **Relations:** `bayesian`, `spectroscopy`, `fret` use it for their likelihoods.
- **Layout:** sources in `src/fit/`, public headers in `include/fit/` (reached as `<IMP/bff/fit/X.h>`; `<IMP/bff/fit.h>` includes them all). The directory is the assignment; `cmake/ThemeHeaders.cmake` links the headers into the build tree.

Public headers: `Convolution.h`, `FitChiSquared.h`, `FitDataset.h`, `FitJointChiSquared.h`, `FitMinimizer.h`, `FitObjective.h`, `FitStatistics.h`, `GeneralizedNormalCurve.h`, `InferenceCanonicalForm.h`, `InferenceFactorGraph.h`, `InferenceFactorGraphFromFit.h`, `InferenceGaussianElimination.h`, `LinearLeastSquares.h`, `MaxEnt.h`, `MaxEntSpectrum.h`, `Minimize.h`, `Optimization.h`, `SpectrumGrid.h`, `StripMask.h`.
