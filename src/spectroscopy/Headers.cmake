# Public headers of the `spectroscopy` theme. IMP links only include/*.h and
# include/internal/*.h into its build tree, so headers stay flat and this
# list is what assigns them (test/test_theme_headers.py checks it).
set(imp_bff_spectroscopy_headers "AcceptorDensityDecay.h;FCS.h;KineticNetwork.h;KineticSchemeNode.h;LifetimeSpectrumMixture.h;PhotophysicsAnisotropySpectrumNode.h;PhotophysicsCrosstalkMatrix.h;PhotophysicsInteractionTerms.h;PhotophysicsLifetimeSpectrum.h;PhotophysicsLifetimeSpectrumNode.h;PhotophysicsPhotonSimulation.h;PhotophysicsPolarisation.h;PhotophysicsQuenching.h;PhotophysicsQuenchingModel.h;PhotophysicsTransferKinetics.h;PhotophysicsTransferKineticsNode.h;States.h;TCSPCDecay.h")
# Sources without a public header of their own.
set(imp_bff_spectroscopy_private_sources "")
