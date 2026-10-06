/**
 *  \file IMP/bff/SparseLinearAlgebra.h
 *  \brief Sparse LU: factorise a compressed sparse matrix once, solve often.
 *
 * The `scipy.sparse.linalg.spsolve` / `factorized` pair, here so a caller
 * that needs it does not need SciPy. The factorisation is Eigen's
 * supernodal `SparseLU` with COLAMD column ordering -- the same algorithm
 * family as SuperLU, which SciPy uses -- over the caller's CSR or CSC
 * arrays (`indptr`, `indices`, `data`; duplicates are summed, as SciPy
 * does). A SparseLU object keeps the factors, so a time stepper that solves
 * the same left-hand side every step factorises it once.
 *
 * A singular matrix throws ValueException instead of returning NaNs with a
 * warning (SciPy's MatrixRankWarning): every caller in the stack treats
 * that as an error, and a NaN field that propagates through a simulation
 * is harder to trace than an exception at the solve.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SPARSE_LINEAR_ALGEBRA_H
#define IMPBFF_SPARSE_LINEAR_ALGEBRA_H

#include <IMP/bff/bff_config.h>
#include <IMP/Object.h>

#include <memory>
#include <string>

IMPBFF_BEGIN_NAMESPACE

//! An LU factorisation of a square sparse matrix, reusable for many solves.
class IMPBFFEXPORT SparseLU : public IMP::Object {
 public:
  explicit SparseLU(std::string name = "SparseLU%1%");

  //! Factorise the `n x n` matrix given in compressed form.
  /*!
      \param[in] n the matrix order.
      \param[in] in_indptr `n + 1` row (CSR) or column (CSC) pointers.
      \param[in] in_indices column (CSR) or row (CSC) index of each entry.
      \param[in] in_data the entries; repeated (row, col) pairs are summed.
      \param[in] row_major true for CSR, false for CSC.
      Throws ValueException on malformed arrays or a singular matrix.
  */
  void factorize(int n, const int* in_indptr, int n_indptr,
                 const int* in_indices, int n_indices, const double* in_data,
                 int n_data, bool row_major = true);

  //! Solve `A x = b` with the stored factors.
  void solve(const double* in_b, int n_b, double** out_view,
             int* n_out_view) const;

  //! The matrix order (0 before factorize()).
  int get_size() const;

  IMP_OBJECT_METHODS(SparseLU);

 private:
  struct Impl;
  // shared_ptr: its deleter is bound where Impl is complete (the .cpp), so
  // the inline destructor IMP_OBJECT_METHODS declares can stay in the header.
  std::shared_ptr<Impl> impl_;
  int n_ = 0;
};

//! Solve `A x = b` for one right-hand side: factorise, solve, discard.
IMPBFFEXPORT void sparse_solve(int n, const int* in_indptr, int n_indptr,
                               const int* in_indices, int n_indices,
                               const double* in_data, int n_data,
                               const double* in_b, int n_b, double** out_view,
                               int* n_out_view, bool row_major = true);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_SPARSE_LINEAR_ALGEBRA_H */
