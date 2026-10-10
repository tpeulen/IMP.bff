# smlm: Single-molecule localisation: particles, likelihoods, registration

Single-molecule localisation: particles, CSV I/O, Gaussian overlap, the LocMoFit likelihood and rigid registration. The IMP restraint (`SMLMRestraint`) sits in `imp/`.

- **Inputs:** localisations with per-axis precision.
- **Relations:** See [okf/smlm.md](../smlm.md).
- **Layout:** sources in `src/smlm/`; the public headers stay flat in `include/` (IMP links only `include/*.h`) and are assigned to this theme in `src/smlm/Headers.cmake`.

Public headers: `SMLM.h`, `SMLMGaussianOverlap.h`, `SMLMIO.h`, `SMLMLikelihood.h`, `SMLMParticleRegistration.h`, `SMLMParticles.h`.
