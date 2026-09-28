/*
 * A two-stage search: cluster representatives first, then the members of the
 * clusters found.
 */

IMP_SWIG_VALUE(IMP::bff, SequenceClusters, SequenceClustersList);
IMP_SWIG_VALUE(IMP::bff, SequenceClusterSearchOptions, SequenceClusterSearchOptionsList);

%feature("compactdefaultargs") IMP::bff::create_sequence_clusters;
%feature("compactdefaultargs") IMP::bff::search_clustered_sequence_database;
%ignore IMP::bff::SequenceClusters::SequenceClusters();

%include "IMP/bff/SequenceClusters.h"
