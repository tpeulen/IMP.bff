# spectroscopy: Photophysics and decays: lifetimes, quenching, kinetics, FCS

Photophysics and decay models: lifetime spectra, quenching, crosstalk, polarisation and anisotropy, transfer kinetics, kinetic networks, states, TCSPC decay and FCS, photon simulation.

- **Inputs:** photons and curves (tttrlib supplies the kernels).
- **Relations:** `fret` builds on the transfer kinetics.
- **Layout:** sources in `src/spectroscopy/`; the public headers stay flat in `include/` (IMP links only `include/*.h`) and are assigned to this theme in `src/spectroscopy/Headers.cmake`.

Public headers: `AcceptorDensityDecay.h`, `FCS.h`, `KineticNetwork.h`, `KineticSchemeNode.h`, `LifetimeSpectrumMixture.h`, `PhotophysicsAnisotropySpectrumNode.h`, `PhotophysicsCrosstalkMatrix.h`, `PhotophysicsInteractionTerms.h`, `PhotophysicsLifetimeSpectrum.h`, `PhotophysicsLifetimeSpectrumNode.h`, `PhotophysicsPhotonSimulation.h`, `PhotophysicsPolarisation.h`, `PhotophysicsQuenching.h`, `PhotophysicsQuenchingModel.h`, `PhotophysicsTransferKinetics.h`, `PhotophysicsTransferKineticsNode.h`, `States.h`, `TCSPCDecay.h`.
