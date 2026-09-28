/*
 * Homology search of a sequence database (MMseqs2's method, an independent
 * implementation) and the query-anchored alignment of its hits.
 */

IMP_SWIG_VALUE(IMP::bff, SequenceSearchOptions, SequenceSearchOptionsList);
IMP_SWIG_VALUE(IMP::bff, SequenceSearchHit, SequenceSearchHits);

%feature("compactdefaultargs") IMP::bff::search_sequence_database;
%feature("compactdefaultargs") IMP::bff::align_sequences;
%feature("compactdefaultargs") IMP::bff::get_query_msa;

%include "IMP/bff/SequenceSearch.h"
