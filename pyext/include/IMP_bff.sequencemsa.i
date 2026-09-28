/*
 * An encoded protein multiple sequence alignment: the shared input of the
 * conservation and co-evolution analyses.
 */

IMP_SWIG_VALUE(IMP::bff, SequenceMSA, SequenceMSAs);

%feature("compactdefaultargs") IMP::bff::SequenceMSA::SequenceMSA;
// Overloads turn keyword arguments off; the empty one is for containers only.
%ignore IMP::bff::SequenceMSA::SequenceMSA();
// The raw int8 buffer is C++'s; Python reads states with get_state / get_sequence.
%ignore IMP::bff::SequenceMSA::get_data;

%include "IMP/bff/SequenceMSA.h"
