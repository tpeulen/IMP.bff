# fret: FRET: transfer, networks, landscapes, labelizer, polymer distances

FRET physics: R0/kappa^2, orientation factors, distance distributions (Gaussian, tabulated, polymer), exchange, landscapes, FRET networks, rotamer FRET and the Labelizer label-site score.

- **Inputs:** coordinates and distance distributions in, efficiencies and photon likelihoods out.
- **Relations:** Uses `spectroscopy` decays and `fit` objectives.
- **Layout:** sources in `src/fret/`; the public headers stay flat in `include/` (IMP links only `include/*.h`) and are assigned to this theme in `src/fret/Headers.cmake`.

Public headers: `DiscreteDistances.h`, `FRET.h`, `FRETAcceptorDensity.h`, `FRETExchange.h`, `FRETLandscape.h`, `FRETLandscapeGrid.h`, `FRETNetwork.h`, `FRETNetworkSimulation.h`, `FRETOrientationFactor.h`, `FRETRotamer.h`, `FRETSpectrumNode.h`, `GaussianDistances.h`, `LabelizerFeatures.h`, `LabelizerFRET.h`, `LabelizerIO.h`, `LabelizerScore.h`, `PolymerChain.h`, `PolymerDistances.h`, `TabulatedDistances.h`.

Sources without a public header of their own: `CommandLineLabelizer.cpp`, `FRETLandscapeSimulation.cpp`, `FRETNetworkEmission.cpp`, `FRETNetworkFilter.cpp`, `FRETNetworkFit.cpp`, `FRETNetworkStates.cpp`.
