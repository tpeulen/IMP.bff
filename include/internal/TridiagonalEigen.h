/**
 *  \file IMP/bff/internal/TridiagonalEigen.h
 *  \brief Eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
 *
 * QL with implicit shifts. It was an anonymous-namespace helper of
 * `DiffusionSolverKrylov.cpp` (the Lanczos projection) and is shared now with
 * the photon-by-photon landscape likelihood (`FRETLandscape.h`), whose
 * symmetrised SqRA generator is tridiagonal too. The arithmetic is unchanged,
 * so the Krylov propagation is bit-identical to what it was.
 *
 * `m` is at most a few hundred for both callers, so O(m^3) costs nothing
 * beside what they do with the result, and it saves a LAPACK dependency in a
 * library that does not otherwise have one.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_INTERNAL_TRIDIAGONAL_EIGEN_H
#define IMPBFF_INTERNAL_TRIDIAGONAL_EIGEN_H

#include <IMP/bff/bff_config.h>

#include <cmath>
#include <vector>

IMPBFF_BEGIN_INTERNAL_NAMESPACE

//! Diagonalise a symmetric tridiagonal matrix in place.
/*! `diag` (length m) and `off` (length m-1, the sub-diagonal) are consumed:
    `diag` leaves holding the eigenvalues, unsorted. `z` comes in as the
    identity (row-major m x m) and leaves holding the eigenvectors in its
    columns, `z[q*m + k]` being component q of eigenvector k. */
inline void tridiagonal_eigen_ql(std::vector<double>& diag, std::vector<double>& off,
                                 std::vector<double>& z, int m) {
    off.push_back(0.0);
    for (int l = 0; l < m; ++l) {
        for (int iter = 0; iter < 50; ++iter) {
            int mm = l;
            for (; mm < m - 1; ++mm) {
                const double dd = std::fabs(diag[mm]) + std::fabs(diag[mm + 1]);
                if (std::fabs(off[mm]) <= 1e-300 + 1e-16 * dd) break;
            }
            if (mm == l) break;
            double g = (diag[l + 1] - diag[l]) / (2.0 * off[l]);
            double r = std::hypot(g, 1.0);
            g = diag[mm] - diag[l] + off[l] / (g + (g >= 0 ? std::fabs(r) : -std::fabs(r)));
            double s = 1.0, c = 1.0, p = 0.0;
            int i = mm - 1;
            for (; i >= l; --i) {
                double f = s * off[i], b = c * off[i];
                r = std::hypot(f, g);
                off[i + 1] = r;
                if (r == 0.0) { diag[i + 1] -= p; off[mm] = 0.0; break; }
                s = f / r; c = g / r;
                g = diag[i + 1] - p;
                r = (diag[i] - g) * s + 2.0 * c * b;
                p = s * r;
                diag[i + 1] = g + p;
                g = c * r - b;
                for (int q = 0; q < m; ++q) {        // rotate the eigenvectors along
                    f = z[q * m + i + 1];
                    z[q * m + i + 1] = s * z[q * m + i] + c * f;
                    z[q * m + i] = c * z[q * m + i] - s * f;
                }
            }
            if (r == 0.0 && i >= l) continue;
            diag[l] -= p; off[l] = g; off[mm] = 0.0;
        }
    }
}

IMPBFF_END_INTERNAL_NAMESPACE

#endif  // IMPBFF_INTERNAL_TRIDIAGONAL_EIGEN_H
