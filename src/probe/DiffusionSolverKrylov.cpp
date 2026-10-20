// SPDX-License-Identifier: BSD-3-Clause
/**
 * The whole propagation as one matrix exponential.
 *
 * `diffusion_propagate()` takes `n_steps` explicit Euler sweeps because
 * forward Euler on a diffusion operator is stable only for
 * `dt <= dg^2/(6D)`. That bound is why the solver costs `dg^-5`. This file
 * does not take a better step -- it stops stepping.
 *
 * Two properties of *this* operator make that possible, and a general PDE has
 * neither:
 *
 *  * **It does not change over a run.** `d`, `decay` and `bounds` are fixed,
 *    so the propagation is `p(t) = exp(-L t) p0` for one `L`, not a sequence
 *    of different problems.
 *  * **It is symmetrisable.** The Smoluchowski flux form is symmetric as it
 *    stands. The Ito form is not, but `q = sqrt(d) p` makes it so: with
 *    `A[c,m] = -d_m b_m`, the similar matrix `D^(1/2) A D^(-1/2)` has entries
 *    `-sqrt(d_c d_m) b_m`, which is symmetric. The detailed-balance form
 *    becomes so under `u = p / sqrt(w)`: off-diagonal `-0.5 (d_c + d_m) b_m`,
 *    the plain Smoluchowski one. So one Lanczos serves all three.
 *
 * So: project `L` onto an `m`-dimensional Krylov space, take the exponential
 * of the resulting `m x m` tridiagonal, and read **every** reported time out
 * of the same projection. Measured at 64-128 operator applications where the
 * stepping route needs four thousand, at better accuracy than the stepping
 * route has (okf/validation/larger_time_steps.md).
 *
 * Three things this exploits that are worth naming, because each one removes
 * a cost that a textbook Lanczos would pay:
 *
 *  * **No reorthogonalisation.** It would cost `m^2 n` flops -- at m = 256,
 *    n = 92k that is more work than the four thousand steps being replaced --
 *    and it is measurably unnecessary here: identical to fourteen digits at
 *    every `m` tested. For `exp(-tL)v` the ghost copies belong to
 *    already-converged eigenvalues.
 *  * **No stored basis for the trace.** The observable is a plain sum, so all
 *    a Krylov vector has to leave behind is one scalar, its (weighted) column
 *    sum. Three vectors of memory for any `m`.
 *  * **A second pass instead of storage for the density.** Storing the basis
 *    is 544 MB at ng = 81, m = 128. Re-running the recursion with the
 *    coefficients already known costs `m` more matvecs and no memory, so the
 *    density is `2m` rather than `m` -- still 15-30x fewer than stepping.
 *
 * What this is *not*: a reproduction of `diffusion_propagate()`. It solves the
 * semi-discrete equation, where the stepping route adds a forward-Euler and a
 * Lie-splitting error of about 2e-6. This is the more accurate answer, and
 * therefore a different one.
 */
#include <IMP/bff/probe/DiffusionSolver.h>
// IMP_THROW and IMP::ValueException: IMP's real ones in the module build,
// the standalone definitions otherwise. Both live behind IMPCompatibility.h.
#include <IMP/bff/util/IMPCompatibility.h>
#include <IMP/bff/internal/TridiagonalEigen.h>
#include <IMP/bff/internal/OutputView.h>
#ifndef TTTRLIB_LATTICE_NAMESPACE
#define TTTRLIB_LATTICE_NAMESPACE IMP::bff::internal
#endif
#include <IMP/bff/internal/LatticeDiffusion.h>


#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

namespace {

/// Which voxels the operator acts on: interior, and inside the domain.
/*! The outer shell is excluded for the same reason the sweep excludes it --
    the 7-point stencil cannot be evaluated there. */
inline bool active(std::size_t ix, std::size_t iy, std::size_t iz, std::size_t n,
                   const std::vector<double>& bounds, std::size_t c) {
    return ix >= 1 && iy >= 1 && iz >= 1 && ix + 1 < n && iy + 1 < n && iz + 1 < n
           && bounds[c] != 0.0;
}

/// y = L x, with L = A + K the generator of dp/dt = -L p.
/*!
    Smoluchowski: `(Ax)[c] = sum_q w_q (x_c - x_m)`, `w_q = 0.5(d_c+d_m) b_m`.
    Ito, already symmetrised by `q = sqrt(d) p`:
    `(Ax)[c] = d_c (sum_q b_m) x_c - sum_q sqrt(d_c d_m) b_m x_m`.
    Detailed balance, symmetrised by `u = p / r` with `r = sqrt(w)`:
    `(Ax)[c] = (sum_q h_q r_m / r_c) x_c - sum_q h_q x_m`, `h_q = 0.5(d_c+d_m) b_m`.
    All plus `k_c x_c`. Rows outside the active set are zero. `aux` is
    `sqrt(d)` for Ito and `r` for detailed balance.

    This is the same memory traffic as one `lattice_sweep`, so a matvec here
    and a step there cost the same -- which is what makes the count of them
    the whole story.
*/
template <int kForm>
void generator_apply(const double* x, const std::vector<double>& d,
                     const std::vector<double>& k, const std::vector<double>& bounds,
                     const std::vector<double>& aux, std::size_t n, double* y) {
    // Zeroing the whole grid every matvec is a full pass of writes for the
    // sake of the shell: 1 GB over 256 matvecs at ng = 81. The interior loop
    // below writes every interior voxel, so only the shell needs it.
    const long ngl = static_cast<long>(n);
    for (std::size_t ix = 0; ix < n; ++ix)
        for (std::size_t iy = 0; iy < n; ++iy)
            for (std::size_t iz = 0; iz < n; ++iz)
                if (ix == 0 || iy == 0 || iz == 0 ||
                    ix + 1 == n || iy + 1 == n || iz + 1 == n)
                    y[(ix * n + iy) * n + iz] = 0.0;
    const long stride[6] = {-ngl * ngl, ngl * ngl, -ngl, ngl, -1, 1};
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long ixl = 1; ixl < ngl - 1; ++ixl) {
        const std::size_t ix = static_cast<std::size_t>(ixl);
        for (std::size_t iy = 1; iy + 1 < n; ++iy) {
            for (std::size_t iz = 1; iz + 1 < n; ++iz) {
                const std::size_t c = (ix * n + iy) * n + iz;
                if (bounds[c] == 0.0) { y[c] = 0.0; continue; }
                const double x0 = x[c], d0 = d[c];
                const double inv0 = kForm == FLUX_DETAILED_BALANCE ? 1.0 / aux[c] : 0.0;
                double diag = 0.0, off = 0.0;
                for (int q = 0; q < 6; ++q) {
                    const std::size_t m = static_cast<std::size_t>(
                        static_cast<long>(c) + stride[q]);
                    const double bm = bounds[m];
                    if (bm == 0.0) continue;              // reflecting wall
                    if (kForm == FLUX_SMOLUCHOWSKI) {
                        const double w = 0.5 * (d0 + d[m]) * bm;
                        diag += w;
                        off += w * x[m];
                    } else if (kForm == FLUX_ITO) {
                        diag += d0 * bm;
                        off += aux[c] * aux[m] * bm * x[m];
                    } else {
                        const double h = 0.5 * (d0 + d[m]) * bm;
                        diag += h * aux[m] * inv0;
                        off += h * x[m];
                    }
                }
                y[c] = (diag + k[c]) * x0 - off;
            }
        }
    }
}

/// The diagonal of the symmetric generator, for a Jacobi preconditioner.
template <int kForm>
double generator_diagonal(std::size_t c, const std::vector<double>& d,
                          const std::vector<double>& k, const std::vector<double>& bounds,
                          const std::vector<double>& aux, std::size_t n) {
    const long ngl = static_cast<long>(n);
    const long stride[6] = {-ngl * ngl, ngl * ngl, -ngl, ngl, -1, 1};
    double diag = 0.0;
    for (int q = 0; q < 6; ++q) {
        const std::size_t m = static_cast<std::size_t>(static_cast<long>(c) + stride[q]);
        const double bm = bounds[m];
        if (bm == 0.0) continue;
        if (kForm == FLUX_SMOLUCHOWSKI) diag += 0.5 * (d[c] + d[m]) * bm;
        else if (kForm == FLUX_ITO)     diag += d[c] * bm;
        else                            diag += 0.5 * (d[c] + d[m]) * bm * aux[m] / aux[c];
    }
    return diag + k[c];
}

/// The generator in its symmetric variables: `v = p * to_sym`, `p = v * from_sym`.
/*! `from_sym` is also what the population sum weighs a voxel by, and is zero
    outside the active set, so it masks the shell in the same multiply. */
struct SymmetricGenerator {
    std::size_t n = 0, nv = 0;
    int form = FLUX_SMOLUCHOWSKI;
    const std::vector<double>* d = nullptr;
    const std::vector<double>* bounds = nullptr;
    std::vector<double> k, aux, to_sym, from_sym;
    std::vector<std::size_t> act;

    void apply(const double* x, double* y) const {
        if (form == FLUX_SMOLUCHOWSKI) generator_apply<FLUX_SMOLUCHOWSKI>(x, *d, k, *bounds, aux, n, y);
        else if (form == FLUX_ITO)     generator_apply<FLUX_ITO>(x, *d, k, *bounds, aux, n, y);
        else                           generator_apply<FLUX_DETAILED_BALANCE>(x, *d, k, *bounds, aux, n, y);
    }
    double diagonal(std::size_t c) const {
        if (form == FLUX_SMOLUCHOWSKI) return generator_diagonal<FLUX_SMOLUCHOWSKI>(c, *d, k, *bounds, aux, n);
        if (form == FLUX_ITO)          return generator_diagonal<FLUX_ITO>(c, *d, k, *bounds, aux, n);
        return generator_diagonal<FLUX_DETAILED_BALANCE>(c, *d, k, *bounds, aux, n);
    }
};

/// Set up the symmetric generator; `rate` is k in the units `d` carries.
/*! \param name the caller, for the messages */
SymmetricGenerator make_generator(const char* name, const std::vector<double>& d,
                                  const std::vector<double>& rate,
                                  const std::vector<double>& bounds,
                                  const std::vector<double>& weight, int ng,
                                  int flux_form) {
    SymmetricGenerator g;
    const std::size_t n = static_cast<std::size_t>(ng);
    const std::size_t nv = n * n * n;
    if (ng < 3) IMP_THROW(name << ": ng must be >= 3", IMP::ValueException);
    if (d.size() != nv || rate.size() != nv || bounds.size() != nv)
        IMP_THROW(name << ": grids must be ng^3 long", IMP::ValueException);
    if (flux_form != FLUX_SMOLUCHOWSKI && flux_form != FLUX_ITO &&
        flux_form != FLUX_DETAILED_BALANCE)
        IMP_THROW(name << ": unknown flux form " << flux_form, IMP::ValueException);
    g.n = n; g.nv = nv; g.form = flux_form; g.d = &d; g.bounds = &bounds;
    g.k = rate;
    // Ito's sqrt(d) is set on the active set only, zero elsewhere, as it
    // always was; the weight's root is 1 outside the domain.
    g.aux.assign(nv, flux_form == FLUX_ITO ? 0.0 : 1.0);
    if (flux_form == FLUX_DETAILED_BALANCE) {
        try {
            g.aux = internal::lattice_root_weight(weight, bounds);
        } catch (const std::invalid_argument& e) {
            IMP_THROW(name << ": " << e.what(), IMP::ValueException);
        }
    }
    g.to_sym.assign(nv, 0.0);
    g.from_sym.assign(nv, 0.0);
    g.act.reserve(nv / 2);
    for (std::size_t ix = 0; ix < n; ++ix)
        for (std::size_t iy = 0; iy < n; ++iy)
            for (std::size_t iz = 0; iz < n; ++iz) {
                const std::size_t c = (ix * n + iy) * n + iz;
                if (!active(ix, iy, iz, n, bounds, c)) { g.k[c] = 0.0; continue; }
                g.act.push_back(c);
                if (flux_form == FLUX_ITO) {
                    if (d[c] <= 0.0)
                        IMP_THROW(name << ": the Ito flux form needs d > 0 "
                            "everywhere in the domain -- the change of variables that makes "
                            "it symmetric is q = sqrt(d) p. Use FLUX_SMOLUCHOWSKI, or "
                            "diffusion_propagate().", IMP::ValueException);
                    g.aux[c] = std::sqrt(d[c]);
                    g.to_sym[c] = g.aux[c];
                    g.from_sym[c] = 1.0 / g.aux[c];
                } else if (flux_form == FLUX_DETAILED_BALANCE) {
                    g.to_sym[c] = 1.0 / g.aux[c];
                    g.from_sym[c] = g.aux[c];
                } else {
                    g.to_sym[c] = 1.0;
                    g.from_sym[c] = 1.0;
                }
            }
    return g;
}

/// Eigenvalues and eigenvectors of a symmetric tridiagonal, QL with implicit shifts.
/*! Shared with the landscape likelihood; see internal/TridiagonalEigen.h. */
inline void tridiagonal_eigen(std::vector<double>& diag, std::vector<double>& off,
                              std::vector<double>& z, int m) {
    internal::tridiagonal_eigen_ql(diag, off, z, m);
}

}  // namespace

namespace internal {

void lattice_krylov(
        const std::vector<double>& cur, const std::vector<double>& d,
        const std::vector<double>& decay, const std::vector<double>& bounds,
        const std::vector<double>& weight, int ng, int flux_form, int n_steps,
        int n_out, int krylov_dim, std::vector<double>& fluorescence,
        std::vector<double>* density) {
    const std::size_t n = static_cast<std::size_t>(ng);
    const std::size_t nv = n * n * n;
    if (cur.size() != nv || decay.size() != nv)
        IMP_THROW("diffusion_propagate_krylov: grids must be ng^3 long", IMP::ValueException);
    if (n_steps < 0 || n_out <= 0)
        IMP_THROW("diffusion_propagate_krylov: n_steps >= 0, n_out > 0", IMP::ValueException);

    // k from the decay factor: decay = exp(-k dt), and the generator wants k.
    std::vector<double> k(nv, 0.0);
    for (std::size_t c = 0; c < nv; ++c) {
        if (bounds.size() == nv && bounds[c] != 0.0) {
            if (decay[c] <= 0.0)
                IMP_THROW("diffusion_propagate_krylov: decay must be positive in the domain "
                    "(it is exp(-k dt))", IMP::ValueException);
            k[c] = -std::log(decay[c]);
        }
    }
    const SymmetricGenerator g = make_generator("diffusion_propagate_krylov", d, k,
                                                bounds, weight, ng, flux_form);

    // The initial vector, in whichever space the operator is symmetric in,
    // masked to the active set so nothing outside can leak into the sums.
    std::vector<double> v0(nv, 0.0);
    for (std::size_t c : g.act) v0[c] = cur[c] * g.to_sym[c];

    const int n_rep = n_steps / n_out + 1;
    fluorescence.assign(static_cast<std::size_t>(n_rep), 0.0);
    if (density) density->assign(nv, 0.0);

    double beta0 = 0.0;
    for (std::size_t c = 0; c < nv; ++c) beta0 += v0[c] * v0[c];
    beta0 = std::sqrt(beta0);
    if (beta0 == 0.0) return;                 // nothing to propagate

    int m = krylov_dim > 0 ? krylov_dim : 128;
    m = std::min<int>(m, static_cast<int>(nv));

    // The Lanczos vectors are zero outside the domain, so every dot product,
    // axpy and norm walks a list of the active voxels rather than the cube --
    // 43% of it at ng = 81 -- and walks it in parallel. Serial full-grid
    // versions of these made eight threads worth nothing here: the generator
    // scaled and they did not, so they became the whole cost.
    const long na = static_cast<long>(g.act.size());
    const std::size_t* AC = g.act.data();

    // --- pass one: alpha, beta, and one scalar per Krylov vector ------------
    std::vector<double> vj(nv), vp(nv, 0.0), w(nv);
    for (std::size_t c = 0; c < nv; ++c) vj[c] = v0[c] / beta0;
    std::vector<double> al, be, u;
    int mm = m;
    double* VJ = vj.data(); double* VP = vp.data(); double* W = w.data();
    const double* WT = g.from_sym.data();
    for (int j = 0; j < m; ++j) {
        g.apply(VJ, W);
        // One pass, two accumulators: the projection alpha_j and the scalar
        // this Krylov vector leaves behind for the trace.
        double a = 0.0, s = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : a, s) schedule(static)
#endif
        for (long t = 0; t < na; ++t) {
            const std::size_t c = AC[t];
            a += VJ[c] * W[c];
            s += WT[c] * VJ[c];
        }
        al.push_back(a); u.push_back(s);
        const double bprev = j ? be[j - 1] : 0.0;
        if (j + 1 >= m) break;
        double b2 = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : b2) schedule(static)
#endif
        for (long t = 0; t < na; ++t) {
            const std::size_t c = AC[t];
            const double wv = W[c] - a * VJ[c] - bprev * VP[c];
            W[c] = wv; b2 += wv * wv;
        }
        const double b = std::sqrt(b2);
        // An exhausted subspace is an exact answer, not a failure.
        if (b <= 1e-12 * (std::fabs(a) + 1.0)) { mm = j + 1; break; }
        be.push_back(b);
        std::swap(VP, VJ);
        const double inv = 1.0 / b;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long t = 0; t < na; ++t) VJ[AC[t]] = W[AC[t]] * inv;
    }
    al.resize(mm); be.resize(std::max(mm - 1, 0)); u.resize(mm);

    // --- the m x m exponential ---------------------------------------------
    std::vector<double> lam(al), offd(be), z(static_cast<std::size_t>(mm) * mm, 0.0);
    for (int i = 0; i < mm; ++i) z[static_cast<std::size_t>(i) * mm + i] = 1.0;
    tridiagonal_eigen(lam, offd, z, mm);
    // L is positive semidefinite; a tiny negative eigenvalue is round-off, and
    // exp(-lam t) with t in the thousands would turn it into a large one.
    for (int i = 0; i < mm; ++i) if (lam[i] < 0.0) lam[i] = 0.0;

    // F(t) = beta0 * sum_i c_i exp(-lam_i t): the decay comes out as a sum of
    // exponentials with explicit rates and amplitudes, which is the form a
    // fluorescence decay is fitted with.
    std::vector<double> amp(mm, 0.0);
    for (int i = 0; i < mm; ++i) {
        double uq = 0.0;
        for (int j = 0; j < mm; ++j) uq += u[j] * z[static_cast<std::size_t>(j) * mm + i];
        amp[i] = beta0 * uq * z[i];             // z[0*mm + i] = first row
    }
    for (int r = 0; r < n_rep; ++r) {
        const double t = static_cast<double>(r) * n_out;
        double f = 0.0;
        for (int i = 0; i < mm; ++i) f += amp[i] * std::exp(-lam[i] * t);
        fluorescence[static_cast<std::size_t>(r)] = f;
    }

    if (!density) return;

    // --- pass two: the density at the final time, without storing the basis --
    const double tend = static_cast<double>(n_steps);
    std::vector<double> y(mm, 0.0);
    for (int j = 0; j < mm; ++j) {
        double s = 0.0;
        for (int i = 0; i < mm; ++i)
            s += z[static_cast<std::size_t>(j) * mm + i] * std::exp(-lam[i] * tend) * z[i];
        y[j] = beta0 * s;
    }
    std::vector<double> acc(nv, 0.0);
    double* ACC = acc.data();
    VJ = vj.data(); VP = vp.data(); W = w.data();
    std::fill(vp.begin(), vp.end(), 0.0);
    const double invb0 = 1.0 / beta0;
    for (std::size_t c = 0; c < nv; ++c) vj[c] = v0[c] * invb0;
    for (int j = 0; j < mm; ++j) {
        const double yj = y[j];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long t = 0; t < na; ++t) ACC[AC[t]] += yj * VJ[AC[t]];
        if (j + 1 >= mm) break;
        g.apply(VJ, W);
        const double aj = al[j], bp = j ? be[j - 1] : 0.0, inv = 1.0 / be[j];
        double* NV = VP;                      // the vector about to be overwritten
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long t = 0; t < na; ++t) {
            const std::size_t c = AC[t];
            NV[c] = (W[c] - aj * VJ[c] - bp * VP[c]) * inv;
        }
        std::swap(VP, VJ);
    }
    // `from_sym` undoes the change of variables and masks the shell in one
    // multiply: zero outside the domain, 1, 1/sqrt(d) or sqrt(w) inside.
    for (std::size_t c = 0; c < nv; ++c) (*density)[c] = acc[c] * g.from_sym[c];
}

LatticeMoments lattice_moment_solve(
        const std::vector<double>& cur, const std::vector<double>& d,
        const std::vector<double>& rate, const std::vector<double>& bounds,
        const std::vector<double>& weight, int ng, int flux_form,
        double tolerance, int max_iterations) {
    const SymmetricGenerator g = make_generator("lattice_moment_solve", d, rate,
                                                bounds, weight, ng, flux_form);
    const std::size_t nv = g.nv;
    if (cur.size() != nv)
        IMP_THROW("lattice_moment_solve: the density must be ng^3 long", IMP::ValueException);
    if (tolerance <= 0.0) tolerance = 1e-10;
    if (max_iterations <= 0) max_iterations = 20000;
    const long na = static_cast<long>(g.act.size());
    const std::size_t* AC = g.act.data();

    std::vector<double> inv_diag(nv, 0.0);
    for (long t = 0; t < na; ++t) {
        const std::size_t c = AC[t];
        const double dg = g.diagonal(c);
        if (!(dg > 0.0))
            IMP_THROW("lattice_moment_solve: the generator is singular -- a voxel "
                      "with no rate and no neighbour; every voxel needs a rate "
                      "(the radiative one) for the moments to exist",
                      IMP::ValueException);
        inv_diag[c] = 1.0 / dg;
    }

    LatticeMoments out{0.0, 0.0, 0, 0.0};
    // Jacobi-preconditioned conjugate gradient for A y = b, A symmetric
    // positive definite, on the active set; vectors are full grids so that
    // apply() can read neighbours, zero outside.
    auto solve = [&](const std::vector<double>& b, std::vector<double>& yv) {
        yv.assign(nv, 0.0);
        std::vector<double> r = b, z(nv, 0.0), pv(nv, 0.0), ap(nv, 0.0);
        double bnorm = 0.0, rz = 0.0;
        for (long t = 0; t < na; ++t) {
            const std::size_t c = AC[t];
            bnorm += b[c] * b[c];
            z[c] = r[c] * inv_diag[c];
            pv[c] = z[c];
            rz += r[c] * z[c];
        }
        bnorm = std::sqrt(bnorm);
        if (bnorm == 0.0) return 0.0;
        double rel = 1.0;
        int it = 0;
        for (; it < max_iterations; ++it) {
            g.apply(pv.data(), ap.data());
            double pap = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : pap) schedule(static)
#endif
            for (long t = 0; t < na; ++t) pap += pv[AC[t]] * ap[AC[t]];
            const double alpha = rz / pap;
            double rr = 0.0, rz_new = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : rr, rz_new) schedule(static)
#endif
            for (long t = 0; t < na; ++t) {
                const std::size_t c = AC[t];
                yv[c] += alpha * pv[c];
                r[c] -= alpha * ap[c];
                z[c] = r[c] * inv_diag[c];
                rr += r[c] * r[c];
                rz_new += r[c] * z[c];
            }
            rel = std::sqrt(rr) / bnorm;
            if (rel < tolerance) { ++it; break; }
            const double beta = rz_new / rz;
            rz = rz_new;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (long t = 0; t < na; ++t) {
                const std::size_t c = AC[t];
                pv[c] = z[c] + beta * pv[c];
            }
        }
        out.iterations += it;
        return rel;
    };

    // M^-1 p0 in the symmetric variables, then M^-2 p0 = M^-1 (M^-1 p0):
    // since to_sym * from_sym = 1 on the active set, the second right-hand
    // side is the first solution as it stands.
    std::vector<double> b(nv, 0.0), y0, y1;
    for (long t = 0; t < na; ++t) b[AC[t]] = cur[AC[t]] * g.to_sym[AC[t]];
    const double r0 = solve(b, y0);
    const double r1 = solve(y0, y1);
    for (long t = 0; t < na; ++t) {
        const std::size_t c = AC[t];
        out.zeroth += g.from_sym[c] * y0[c];
        out.first += g.from_sym[c] * y1[c];
    }
    out.residual = std::max(r0, r1);
    return out;
}

}  // namespace internal

void diffusion_propagate_krylov(
        const std::vector<double>& cur, const std::vector<double>& d,
        const std::vector<double>& decay, const std::vector<double>& bounds,
        int ng, int flux_form, int n_steps, int n_out, int krylov_dim,
        double** out_fluorescence, int* n_out_fluorescence,
        double** out_view, int* n_out_view) {
    const bool want_density = (out_view != nullptr && n_out_view != nullptr);
    std::vector<double> fluorescence, density;
    internal::lattice_krylov(cur, d, decay, bounds, std::vector<double>(), ng,
                             flux_form, n_steps, n_out, krylov_dim, fluorescence,
                             want_density ? &density : nullptr);
    internal::copy_to_view(fluorescence, out_fluorescence, n_out_fluorescence);
    if (want_density) internal::copy_to_view(density, out_view, n_out_view);
}

void diffusion_trace_krylov(
        const std::vector<double>& cur, const std::vector<double>& d,
        const std::vector<double>& decay, const std::vector<double>& bounds,
        int ng, int flux_form, int n_steps, int n_out, int krylov_dim,
        double** out_fluorescence, int* n_out_fluorescence) {
    diffusion_propagate_krylov(cur, d, decay, bounds, ng, flux_form, n_steps,
                               n_out, krylov_dim, out_fluorescence,
                               n_out_fluorescence, nullptr, nullptr);
}

IMPBFF_END_NAMESPACE
