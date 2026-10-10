# Public headers of the `fret` theme. IMP links only include/*.h and
# include/internal/*.h into its build tree, so headers stay flat and this
# list is what assigns them (test/test_theme_headers.py checks it).
set(imp_bff_fret_headers "DiscreteDistances.h;FRET.h;FRETAcceptorDensity.h;FRETExchange.h;FRETLandscape.h;FRETLandscapeGrid.h;FRETNetwork.h;FRETNetworkSimulation.h;FRETOrientationFactor.h;FRETRotamer.h;FRETSpectrumNode.h;GaussianDistances.h;LabelizerFeatures.h;LabelizerFRET.h;LabelizerIO.h;LabelizerScore.h;PolymerChain.h;PolymerDistances.h;TabulatedDistances.h")
# Sources without a public header of their own.
set(imp_bff_fret_private_sources "CommandLineLabelizer.cpp;FRETLandscapeSimulation.cpp;FRETNetworkEmission.cpp;FRETNetworkFilter.cpp;FRETNetworkFit.cpp;FRETNetworkStates.cpp")
