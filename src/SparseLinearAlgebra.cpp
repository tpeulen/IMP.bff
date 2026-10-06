/**
 *  \file SparseLinearAlgebra.cpp
 *  \brief Sparse LU over Eigen's SparseLU (COLAMD ordering).
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SparseLinearAlgebra.h>
#include <IMP/Pointer.h>
#include <IMP/bff/IMPCompatibility.h>

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <cstdlib>
#include <cstring>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

struct SparseLU::Impl {
  Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor>,
                  Eigen::COLAMDOrdering<int> >
      lu;
};

SparseLU::SparseLU(std::string name)
    : IMP::Object(name), impl_(std::make_shared<Impl>()) {}

namespace {

Eigen::SparseMatrix<double, Eigen::ColMajor> sparse_lu_matrix(
    int n, const int* indptr, int n_indptr, const int* indices, int n_indices,
    const double* data, int n_data, bool row_major, const char* who) {
  if (n <= 0) IMP_THROW(who << ": the matrix order must be positive", ValueException);
  if (n_indptr != n + 1) {
    IMP_THROW(who << ": indptr needs n + 1 = " << n + 1 << " entries, got " << n_indptr,
              ValueException);
  }
  if (n_indices != n_data) {
    IMP_THROW(who << ": indices and data differ in length", ValueException);
  }
  if (indptr[0] != 0 || indptr[n] != n_data) {
    IMP_THROW(who << ": indptr must start at 0 and end at nnz", ValueException);
  }
  std::vector<Eigen::Triplet<double> > triplets;
  triplets.reserve(static_cast<std::size_t>(n_data));
  for (int major = 0; major < n; ++major) {
    if (indptr[major + 1] < indptr[major]) {
      IMP_THROW(who << ": indptr must not decrease", ValueException);
    }
    for (int k = indptr[major]; k < indptr[major + 1]; ++k) {
      const int minor = indices[k];
      if (minor < 0 || minor >= n) {
        IMP_THROW(who << ": index " << minor << " outside [0, " << n << ")",
                  ValueException);
      }
      if (row_major) {
        triplets.emplace_back(major, minor, data[k]);
      } else {
        triplets.emplace_back(minor, major, data[k]);
      }
    }
  }
  Eigen::SparseMatrix<double, Eigen::ColMajor> a(n, n);
  a.setFromTriplets(triplets.begin(), triplets.end());  // sums duplicates
  a.makeCompressed();
  return a;
}

void publish(const Eigen::VectorXd& x, double** out_view, int* n_out_view) {
  const int n = static_cast<int>(x.size());
  double* out = static_cast<double*>(std::malloc(sizeof(double) * (n > 0 ? n : 1)));
  if (!out) IMP_THROW("sparse solve: out of memory", ValueException);
  if (n > 0) std::memcpy(out, x.data(), sizeof(double) * n);
  *out_view = out;
  *n_out_view = n;
}

}  // namespace

void SparseLU::factorize(int n, const int* in_indptr, int n_indptr,
                         const int* in_indices, int n_indices,
                         const double* in_data, int n_data, bool row_major) {
  Eigen::SparseMatrix<double, Eigen::ColMajor> a =
      sparse_lu_matrix(n, in_indptr, n_indptr, in_indices, n_indices, in_data,
                       n_data, row_major, "SparseLU::factorize");
  impl_->lu.analyzePattern(a);
  impl_->lu.factorize(a);
  if (impl_->lu.info() != Eigen::Success) {
    n_ = 0;
    IMP_THROW("SparseLU::factorize: the matrix is singular ("
                  << impl_->lu.lastErrorMessage() << ")",
              ValueException);
  }
  n_ = n;
}

void SparseLU::solve(const double* in_b, int n_b, double** out_view,
                     int* n_out_view) const {
  if (n_ == 0) IMP_THROW("SparseLU::solve: factorize() first", ValueException);
  if (n_b != n_) {
    IMP_THROW("SparseLU::solve: b has " << n_b << " entries, the matrix order is "
                                        << n_,
              ValueException);
  }
  Eigen::Map<const Eigen::VectorXd> b(in_b, n_b);
  Eigen::VectorXd x = impl_->lu.solve(b);
  if (impl_->lu.info() != Eigen::Success) {
    IMP_THROW("SparseLU::solve: the solve failed", ValueException);
  }
  publish(x, out_view, n_out_view);
}

int SparseLU::get_size() const { return n_; }

void sparse_solve(int n, const int* in_indptr, int n_indptr,
                  const int* in_indices, int n_indices, const double* in_data,
                  int n_data, const double* in_b, int n_b, double** out_view,
                  int* n_out_view, bool row_major) {
  IMP::Pointer<SparseLU> lu = new SparseLU();
  lu->factorize(n, in_indptr, n_indptr, in_indices, n_indices, in_data, n_data,
                row_major);
  lu->solve(in_b, n_b, out_view, n_out_view);
}

IMPBFF_END_NAMESPACE
