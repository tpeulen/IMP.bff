/*
 * A protein sequence database as binary FASTA in a .pto container.
 */

IMP_SWIG_VALUE(IMP::bff, SequenceDatabase, SequenceDatabases);

%feature("compactdefaultargs") IMP::bff::create_sequence_database;
// Overloads turn keyword arguments off; the empty one is for containers only.
%ignore IMP::bff::SequenceDatabase::SequenceDatabase();

%include "IMP/bff/SequenceDatabase.h"
