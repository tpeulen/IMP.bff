/**
 *  \file IMP/bff/FRETLandscapeGrid.h
 *  \brief The grid a one-dimensional free-energy landscape lives on: the
 *         square-root-approximation (SqRA) generator, its symmetrisation, the
 *         symmetric tridiagonal eigensolver, and the natural cubic spline
 *         that parameterises the landscape.
 *
 * These are the discretisation pieces of the photon-by-photon landscape
 * likelihood of Dingeldein & Covino (arXiv:2608.21061, Sec. III A-B); the
 * likelihood itself is FRETLandscapeModel (FRETLandscape.h).
 *
 * **SqRA.** Brownian motion on `u(x)` (in kT) with diffusion coefficient `D`
 * is discretised on a uniform grid `x_i`, spacing `h`, as hopping between
 * neighbouring cells with rates
 *
 *     q_{i -> i+-1} = D/h^2 exp(-(u_{i+-1} - u_i)/2),
 *
 * reflecting at both ends. Columns of `Q` sum to zero and detailed balance
 * holds with `pi_i ~ exp(-u_i)` exactly, on the grid and not only as h -> 0.
 *
 * **Symmetrisation.** With `Pi = diag(pi)`,
 * `A = Pi^(-1/2) (Q - Lambda) Pi^(1/2)` is symmetric tridiagonal for any
 * diagonal killing `Lambda`. Its off-diagonal is the constant `D/h^2` (the
 * Boltzmann factors cancel), and its diagonal is
 * `-(D/h^2)(exp(-(u_{i+1}-u_i)/2) + exp(-(u_{i-1}-u_i)/2)) - Lambda_i`.
 *
 * \authors Thomas-Otavio Peulen
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_FRETLANDSCAPEGRID_H
#define IMPBFF_FRETLANDSCAPEGRID_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The eigen-decomposition of a real symmetric tridiagonal matrix.
/*! Eigenvalues are sorted in descending order. `get_vectors()` is row-major
    `n x n` with the eigenvectors in the columns: element `[i*n + k]` is
    component `i` of eigenvector `k`, so `T = V diag(values) V^T`. */
class IMPBFFEXPORT TridiagonalEigenSystem {
 public:
  TridiagonalEigenSystem() {}
  TridiagonalEigenSystem(const std::vector<double>& values,
                         const std::vector<double>& vectors)
      : values_(values), vectors_(vectors) {}
  //! Eigenvalues, descending.
  const std::vector<double>& get_values() const { return values_; }
  //! Eigenvectors in columns, row-major `n x n`.
  const std::vector<double>& get_vectors() const { return vectors_; }
  int get_n() const { return static_cast<int>(values_.size()); }

  IMP_SHOWABLE_INLINE(TridiagonalEigenSystem,
                      out << "TridiagonalEigenSystem(n " << get_n() << ")");

 private:
  std::vector<double> values_, vectors_;
};
IMP_VALUES(TridiagonalEigenSystem, TridiagonalEigenSystems);

//! Diagonalise a real symmetric tridiagonal matrix (QL, implicit shifts).
/*! \param diagonal the n diagonal entries
    \param off_diagonal the n-1 sub- (= super-) diagonal entries
    The same kernel serves the Krylov diffusion propagation
    (`diffusion_propagate_krylov`). */
IMPBFFEXPORT TridiagonalEigenSystem symmetric_tridiagonal_eigen(
    const std::vector<double>& diagonal, const std::vector<double>& off_diagonal);

//! The SqRA rate matrix of 1-D Brownian motion on a grid landscape.
/*! \param u the landscape on the grid, in kT
    \param diffusion the diffusion coefficient D (length^2 / time)
    \param spacing the grid spacing h (length)
    \return row-major `M x M`, `Q[target*M + source]`: column `j` holds the
            rates out of cell `j`, and every column sums to zero. */
IMPBFFEXPORT std::vector<double> sqra_generator(const std::vector<double>& u,
                                                double diffusion, double spacing);

//! The Boltzmann weights `exp(-u_i) / sum_j exp(-u_j)` on the grid.
/*! The exact stationary distribution of sqra_generator(). */
IMPBFFEXPORT std::vector<double> sqra_stationary_distribution(
    const std::vector<double>& u);

//! Diagonal of the symmetrised killed generator `Pi^-1/2 (Q - Lambda) Pi^1/2`.
/*! The off-diagonal of that matrix is `diffusion / spacing^2` everywhere.
    \param killing the per-cell killing rate `Lambda_i`; empty for none. */
IMPBFFEXPORT std::vector<double> sqra_symmetric_diagonal(
    const std::vector<double>& u, double diffusion, double spacing,
    const std::vector<double>& killing = std::vector<double>());

//! A natural cubic spline on uniformly spaced knots.
/*! The free parameters are the knot heights `mu_k = u(x_k)`; the spline is
    linear in them, `u(x) = sum_k mu_k phi_k(x)`, and the basis functions
    `phi_k` depend on the knot geometry only. Outside the knot range the
    spline continues linearly (a natural spline has zero curvature at its
    ends). */
class IMPBFFEXPORT NaturalCubicSpline {
 public:
  NaturalCubicSpline(double x_min = 0.0, double x_max = 1.0, int n_knots = 4);
  int get_n_knots() const { return n_; }
  double get_knot_spacing() const { return h_; }
  std::vector<double> get_knots() const;
  //! The basis `Phi[i*K + k] = phi_k(x_i)`, row-major `len(x) x K`.
  std::vector<double> get_basis(const std::vector<double>& x) const;
  //! Derivatives `phi_k'(x_i)`, same layout as get_basis().
  std::vector<double> get_basis_derivative(const std::vector<double>& x) const;
  //! Spline values at `x` for knot heights `mu`.
  std::vector<double> evaluate(const std::vector<double>& mu,
                               const std::vector<double>& x) const;
  //! Spline derivative at `x` for knot heights `mu`.
  std::vector<double> derivative(const std::vector<double>& mu,
                                 const std::vector<double>& x) const;
  //! Second derivatives at the knots for heights `mu` (zero at both ends).
  std::vector<double> knot_curvatures(const std::vector<double>& mu) const;
  //! One point: value and first derivative, from precomputed curvatures.
  void evaluate_point(const std::vector<double>& mu, const std::vector<double>& m2,
                      double x, double& value, double& slope) const;

  IMP_SHOWABLE_INLINE(NaturalCubicSpline,
                      out << "NaturalCubicSpline(" << x0_ << ", "
                          << x0_ + h_ * (n_ - 1) << ", n_knots " << n_ << ")");

 private:
  double x0_, h_;
  int n_;
};
IMP_VALUES(NaturalCubicSpline, NaturalCubicSplines);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_FRETLANDSCAPEGRID_H
