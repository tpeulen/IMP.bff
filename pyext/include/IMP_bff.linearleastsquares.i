/*
 * Constrained linear least squares: nnls() and bvls().
 *
 * A and b come in as numpy arrays (IN_ARRAY2 / IN_ARRAY1: any array-like,
 * converted to C-contiguous float64), so a caller hands over what it has.
 * The result is a LinearLeastSquaresResult; its vectors read back as
 * tuples, which the chisurf shim turns into ndarrays.
 */
%apply(double* IN_ARRAY2, int DIM1, int DIM2) {(const double* in_matrix, int n_rows, int n_cols)};
%apply(double* IN_ARRAY1, int DIM1) {
    (const double* in_b, int n_b),
    (const double* in_lower, int n_lower),
    (const double* in_upper, int n_upper)
};
%include "IMP/bff/LinearLeastSquares.h"
