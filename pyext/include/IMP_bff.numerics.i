/*
 * Special functions, distribution functions and expm over numpy arrays.
 *
 * Inputs are IN_ARRAY1 (any array-like, converted to contiguous float64,
 * no copy when it already is); results are owned numpy views. Broadcasting
 * and reshaping live in the Python shim, chisurf.core.math.numerics.
 */
%apply(double* IN_ARRAY1, int DIM1) {
    (const double* in_x, int n_x),
    (const double* in_a, int n_a),
    (const double* in_b, int n_b),
    (const double* in_s, int n_s),
    (const double* in_q, int n_q),
    (const double* in_p, int n_p),
    (const double* in_k, int n_k),
    (const double* in_n, int n_n),
    (const double* in_t, int n_t),
    (const double* in_df, int n_df),
    (const double* in_nc, int n_nc),
    (const double* in_dfn, int n_dfn),
    (const double* in_dfd, int n_dfd)
};
%apply(double** ARGOUTVIEWM_ARRAY1, int* DIM1) {
    (double** out_view, int* n_out_view),
    (double** out_second, int* n_out_second)
};
%apply(double* IN_ARRAY2, int DIM1, int DIM2) {(const double* in_matrix, int n_rows, int n_cols)};
%apply(double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {
    (double** out_matrix, int* n_out_rows, int* n_out_cols)
};
%include "IMP/bff/Numerics.h"
