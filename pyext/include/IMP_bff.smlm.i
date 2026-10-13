/* Native SMLM localization index, likelihood and selective particle average.
   The common vector typemaps marshal input once and expose output as arrays.
   All search, refinement and map accumulation loops stay in C++. */
%include "IMP/bff/smlm/SMLM.h"
%include "IMP/bff/smlm/SMLMParticles.h"
%include "IMP/bff/smlm/SMLMLikelihood.h"
%include "IMP/bff/smlm/SMLMGaussianOverlap.h"
%include "IMP/bff/smlm/SMLMParticleRegistration.h"
%include "IMP/bff/smlm/SMLMIO.h"
