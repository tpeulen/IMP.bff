/*
 * Mean-field direct coupling analysis of an alignment; direct information per
 * pair of columns, and per pair of structure residues for probe selection.
 */

IMP_SWIG_VALUE(IMP::bff, SequenceCoevolution, SequenceCoevolutions);

%ignore IMP::bff::SequenceCoevolution::SequenceCoevolution();

%apply(int* IN_ARRAY2, int DIM1, int DIM2) {
    (int* pair_residues, int n_pair_rows, int n_pair_cols)
};
%apply(double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {
    (double** out_matrix, int* n_out_rows, int* n_out_cols)
};

%include "IMP/bff/SequenceCoevolution.h"
