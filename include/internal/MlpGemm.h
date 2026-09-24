/**
 *  \file IMP/bff/internal/MlpGemm.h
 *  \brief The `MatGemm` policy: MlpCore's batch products through Mat.h.
 *
 * `internal/MlpCore.h` takes its three dense products -- `nn` (C = A B),
 * `nt` (C = A B^T) and `tn` (C = A^T B), row-major, C overwritten -- as a
 * template policy. Its default, `PortableGemm`, is three plain loops with an
 * `omp simd` hint: std-only, so the header stands alone. This policy routes
 * the same products through the vendored `internal/Mat.h`'s packed,
 * register-blocked micro-kernel, threaded over 32-row tiles when the build
 * has OpenMP (`gemm_nn_core_mt`). The arithmetic is the same up to
 * reassociation (test/test_neural_net_derivatives.py pins forward and
 * backward to 1e-12 relative against PortableGemm).
 *
 * `NeuralNet` (predict on the CPU and every derivative) and
 * train_neural_net() use this policy. MlpCore.h's default stays the
 * portable one; the policy is a template argument at the call site.
 *
 * History, from tttrlib's `modules/math/README.md` (the policy was extracted
 * there as `MlpGemm.h` and lost before it was committed; this is its
 * bff-owned successor, the struct being the one tttrlib's NeuralNet.cpp
 * carried): the pre-tile OpenMP path bypassed the micro-kernel for a
 * per-row SAXPY, so `gemm_nt` / `gemm_tn` -- the MLP hot path -- ran on one
 * core of an 8-core machine. Measured then on wall clock, OpenMP builds,
 * median over three net draws: 2.7x the portable path at batch 512, parity
 * at batch 256 (the portable loops get `omp simd` too), and 1.7-2.6x behind
 * Eigen; closing that gap means a true blocked GEMM, a project of its own,
 * not an adapter. A Kx2 unroll of the micro-kernel was measured at noise
 * level (~2 %). Batch-1 shapes stay serial by threshold (packing overhead
 * dominates a single row).
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_MLPGEMM_H
#define IMPBFF_INTERNAL_MLPGEMM_H

#include <IMP/bff/internal/Mat.h>

namespace IMP {
namespace bff {
namespace internal {

//! MlpCore's GEMM policy over Mat.h's blocked kernels (see the file comment).
struct MatGemm {
    //! C(M x N) = A(M x K) B(K x N)
    static void nn(int M, int N, int K, const double* A, const double* B, double* C) {
        tttrlib::mat_detail::gemm_nn(M, N, K, A, B, C);
    }
    //! C(M x N) = A(M x K) B(N x K)^T
    static void nt(int M, int N, int K, const double* A, const double* B, double* C) {
        tttrlib::mat_detail::gemm_nt(M, N, K, A, B, C);
    }
    //! C(M x N) = A(K x M)^T B(K x N)
    static void tn(int M, int N, int K, const double* A, const double* B, double* C) {
        tttrlib::mat_detail::gemm_tn(M, N, K, A, B, C);
    }
};

}  // namespace internal
}  // namespace bff
}  // namespace IMP

#endif  // IMPBFF_INTERNAL_MLPGEMM_H
