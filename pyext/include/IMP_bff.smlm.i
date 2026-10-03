/* Native SMLM localization index, likelihood and selective particle average.
   The common vector typemaps marshal input once and expose output as arrays.
   All search, refinement and map accumulation loops stay in C++. */
%include "IMP/bff/SMLM.h"
%include "IMP/bff/SMLMParticles.h"
%include "IMP/bff/SMLMLikelihood.h"
%include "IMP/bff/SMLMGaussianOverlap.h"
%include "IMP/bff/SMLMParticleRegistration.h"
%include "IMP/bff/SMLMIO.h"
