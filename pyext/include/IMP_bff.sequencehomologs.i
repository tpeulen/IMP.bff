/*
 * Choosing homologues by ConSurf's rules.
 */

IMP_SWIG_VALUE(IMP::bff, SequenceHomologOptions, SequenceHomologOptionsList);

%feature("compactdefaultargs") IMP::bff::select_sequence_homologs;

%include "IMP/bff/SequenceHomologs.h"
