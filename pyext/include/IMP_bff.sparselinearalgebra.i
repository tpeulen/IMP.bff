/*
 * Sparse LU (scipy.sparse.linalg.spsolve / factorized) over compressed
 * arrays. Index arrays are int32, values float64; the solution is an owned
 * numpy view. The scipy-shaped layer (CSR/CSC objects, spsolve, factorized)
 * lives in the callers, e.g. imp-tricks' IMP/finite/_sparse.py.
 */
%apply(int* IN_ARRAY1, int DIM1) {
    (const int* in_indptr, int n_indptr),
    (const int* in_indices, int n_indices)
};
%apply(double* IN_ARRAY1, int DIM1) {
    (const double* in_data, int n_data),
    (const double* in_b, int n_b)
};
%apply(double** ARGOUTVIEWM_ARRAY1, int* DIM1) {
    (double** out_view, int* n_out_view)
};
IMP_SWIG_OBJECT(IMP::bff, SparseLU, SparseLUs);
%include "IMP/bff/SparseLinearAlgebra.h"
