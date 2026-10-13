# smlm: Single-molecule localisation: particles, likelihoods, registration

Single-molecule localisation: particles, CSV I/O, Gaussian overlap, the LocMoFit likelihood and rigid registration. The IMP restraint (`SMLMRestraint`) sits in `imp/`.

- **Inputs:** localisations with per-axis precision.
- **Relations:** See [okf/smlm.md](../../okf/smlm.md).
- **Layout:** sources in `src/smlm/`, public headers in `include/smlm/` (reached as `<IMP/bff/smlm/X.h>`; `<IMP/bff/smlm.h>` includes them all). The directory is the assignment; `cmake/ThemeHeaders.cmake` links the headers into the build tree.

Public headers: `SMLM.h`, `SMLMGaussianOverlap.h`, `SMLMIO.h`, `SMLMLikelihood.h`, `SMLMParticleRegistration.h`, `SMLMParticles.h`.
