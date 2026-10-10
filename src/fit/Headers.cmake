# Public headers of the `fit` theme. IMP links only include/*.h and
# include/internal/*.h into its build tree, so headers stay flat and this
# list is what assigns them (test/test_theme_headers.py checks it).
set(imp_bff_fit_headers "Convolution.h;FitChiSquared.h;FitDataset.h;FitJointChiSquared.h;FitMinimizer.h;FitObjective.h;FitStatistics.h;GeneralizedNormalCurve.h;InferenceCanonicalForm.h;InferenceFactorGraph.h;InferenceFactorGraphFromFit.h;InferenceGaussianElimination.h;LinearLeastSquares.h;MaxEnt.h;MaxEntSpectrum.h;Minimize.h;Optimization.h;SpectrumGrid.h;StripMask.h")
# Sources without a public header of their own.
set(imp_bff_fit_private_sources "")
