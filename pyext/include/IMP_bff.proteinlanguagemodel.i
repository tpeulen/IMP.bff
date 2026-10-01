/*
 * A protein language model (ESM-2 from GGUF), run natively: embeddings for
 * the embedding prefilter of the sequence search, and residue contacts as an
 * alternative to co-evolution for probe selection.
 */

IMP_SWIG_VALUE(IMP::bff, ProteinLanguageModel, ProteinLanguageModels);

%apply(int* IN_ARRAY2, int DIM1, int DIM2) {
    (int* pair_residues, int n_pair_rows, int n_pair_cols)
};
%apply(double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {
    (double** out_matrix, int* n_out_rows, int* n_out_cols)
};
%feature("compactdefaultargs") IMP::bff::probe_pair_contacts;

%include "IMP/bff/ProteinLanguageModel.h"
