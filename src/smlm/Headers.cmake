# Public headers of the `smlm` theme. IMP links only include/*.h and
# include/internal/*.h into its build tree, so headers stay flat and this
# list is what assigns them (test/test_theme_headers.py checks it).
set(imp_bff_smlm_headers "SMLM.h;SMLMGaussianOverlap.h;SMLMIO.h;SMLMLikelihood.h;SMLMParticleRegistration.h;SMLMParticles.h")
# Sources without a public header of their own.
set(imp_bff_smlm_private_sources "")
