/* IMP-only structural modelling bridge. Core SMLM value types are already
   declared by core.i; ownership and derivative evaluation are native IMP. */
IMP_SWIG_OBJECT(IMP::bff, SMLMRestraint, SMLMRestraints);
%include "IMP/bff/SMLMRestraint.h"
