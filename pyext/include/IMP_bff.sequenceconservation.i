/*
 * Per-site evolutionary rates of an alignment (the Rate4Site method, an
 * independent implementation) and ConSurf's conservation grades.
 */

IMP_SWIG_VALUE(IMP::bff, SequenceConservationOptions, SequenceConservationOptionsList);
IMP_SWIG_VALUE(IMP::bff, SequenceConservation, SequenceConservations);

%feature("compactdefaultargs") IMP::bff::compute_sequence_conservation;

%include "IMP/bff/SequenceConservation.h"
