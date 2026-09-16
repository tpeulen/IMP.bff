/**
 * \file IMP/bff/BayesianDecayPosterior.h
 * \brief The posterior of a Bayesian decay model: prior, Poisson likelihood,
 *        Fisher scoring to its mode, the Laplace evidence, and p(R/R0) with
 *        its delta-method band.
 *
 * Copyright 2007-2023 IMP Inventors. All rights reserved.
 *
 * Ported 2026-09-14 from ucfret's `s89_cpp/cbm56_fit.h` (PRD-142 step 4), where it
 * was gated against the Python prototype (`s88_laplace_posterior`: the prior and
 * posterior at 3e-14 / 4e-7 nats of 1.7e8, the gradient at 4e-9 of its peak) and,
 * fitted from the same start, landed on the Newton-polished Python mode: evidence
 * to 1e-4 nats, p(R/R0) to 7e-6 of its peak, D/dof to 7e-6, in 2.2 s.
 *
 * * **The prior**: per variable, from the experiment's table -- a Gaussian on the
 *   coordinate (`gaussian`, `gaussian_on_z`), a log-normal on a log coordinate, a
 *   uniform on a logit one -- stated on the constrained value with the transform's
 *   log |det| added; the lifetime spectrum's continuity prior (a Gaussian with
 *   precision `spec_smooth_P`); the P-spline on the coefficients of log p(R/R0)
 *   (Eilers & Marx 1996: second differences with precision lambda held at
 *   `fixed_values["log10_lam"]` or free as the variable `log10_lam`, and a weak Gaussian on the linear tilt). Analytic
 *   gradient and Hessian.
 * * **The likelihood**: Poisson over the masked bins, `lgamma(y + 1)` dropped.
 * * **The mode**: Fisher scoring -- `A = J' diag(mask / m) J - H_prior` -- stepped by
 *   `DampedNewton` with `line_search_below` (Optimization.h), McCullagh & Nelder
 *   1989 section 2.5.
 * * **The evidence**: `log p(y, theta*) + d/2 log 2 pi - 1/2 log det A`. With the
 *   Fisher matrix; at the CBM56 mode the exact Hessian's evidence was 11.7 nats
 *   lower, which matters when evidences of different models are compared.
 * * **p(R/R0)** with its first-order delta-method sd, the covariance `A^-1`.
 *
 * A caveat that travels with every band from here: on CBM56 the Gaussian at the
 * mode is not the posterior in its flattest directions (the far-tail spline
 * coefficients fell 74-177 nats within one sd where the Gaussian predicts 0.5;
 * ucfret `okf/prd-cbm56-mode-mixture.md`, M4b).
 */

#ifndef IMPBFF_BAYESIANDECAYPOSTERIOR_H
#define IMPBFF_BAYESIANDECAYPOSTERIOR_H

//! The FFT is pocketfft, reached through tttrlib (`"pocketfft/pocketfft_hdronly.h"`
//! with tttrlib's `thirdparty/` on the include path, as tttrlib's own sources
//! include it). Where it is not on the path -- the IMP module build today -- this
//! header declares nothing, so that `IMP/bff.h`, which includes every public header,
//! still compiles.
#if __has_include("pocketfft/pocketfft_hdronly.h")

#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/BayesianDecayModel.h>
#include <IMP/bff/Optimization.h>
#include <IMP/bff/BayesianPSpline.h>
#include <IMP/bff/BayesianFisherScoring.h>
#include <IMP/bff/BayesianLaplace.h>
#include <IMP/bff/BayesianDeltaMethod.h>
#include <IMP/bff/internal/AdamUpdate.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! \name The posterior of a Bayesian decay model
//! @{

struct BayesianDecayPrior { double lp = 0.0; std::vector<double> g, H; };

//! the log prior in theta, with its gradient and Hessian (dim x dim)
inline BayesianDecayPrior bayesian_decay_log_prior(const BayesianDecayExperiment& f, const std::vector<double>& th, bool derivs = true) {
    const std::size_t dim = f.dim;
    const double l2pi = std::log(2.0 * 3.14159265358979323846);
    BayesianDecayPrior P;
    if (derivs) { P.g.assign(dim, 0.0); P.H.assign(dim * dim, 0.0); }
    for (const BayesianDecayVariable& v : f.variables) {
        const std::string& fam = v.family, & tr = v.transform, & nm = v.name;
        const std::size_t o = v.offset, sz = v.size;
        if (fam == "pspline") continue;                          // c: the P-spline term below
        if (fam == "spectrum_smoothness") {
            const BayesianDecayArray& Pm = f["spec_smooth_P"];
            const double logdet = f.spec_smooth_logdet;
            double q = 0.0;
            for (std::size_t i = 0; i < sz; ++i) {
                double r = 0.0; for (std::size_t j = 0; j < sz; ++j) r += Pm.d[i * sz + j] * th[o + j];
                q += th[o + i] * r;
                if (derivs) { P.g[o + i] -= r; for (std::size_t j = 0; j < sz; ++j) P.H[(o + i) * dim + o + j] -= Pm.d[i * sz + j]; }
            }
            P.lp += 0.5 * logdet - 0.5 * q - 0.5 * double(sz) * l2pi;
            continue;
        }
        for (std::size_t i = 0; i < sz; ++i) {
            const double z = th[o + i], a = v.a[i], b = v.b[i];
            double lp = 0.0, g = 0.0, h = 0.0;
            if (fam == "gaussian_on_z" || (fam == "gaussian" && tr == "identity")) {
                lp = -0.5 * ((z - a) / b) * ((z - a) / b) - std::log(b) - 0.5 * l2pi; g = -(z - a) / (b * b); h = -1.0 / (b * b);
            } else if (fam == "lognormal" && tr == "log") {
                //: log x = z: the prior's -log x and the transform's +z cancel
                const double u = (z - std::log(a)) / b;
                lp = -0.5 * u * u - std::log(b) - 0.5 * l2pi; g = -u / b; h = -1.0 / (b * b);
            } else if (fam == "uniform" && tr == "logit") {
                const double lo = v.lo, hi = v.hi;
                //: -log(hi - lo) from the prior, log(hi - lo) + logsigmoid(z) + logsigmoid(-z) from the transform
                lp = -bayesian_soft_positive(-z, 1.0, BAYESIAN_TORCH_SOFTPLUS_THRESHOLD) - bayesian_soft_positive(z, 1.0, BAYESIAN_TORCH_SOFTPLUS_THRESHOLD);
                const double s = 1.0 / (1.0 + std::exp(-z));
                g = 1.0 - 2.0 * s; h = -2.0 * s * (1.0 - s);
                (void)lo; (void)hi;
            } else {
                throw std::runtime_error("bayesian_decay_log_prior: family " + fam + " on transform " + tr + " (" + nm + ") is not implemented");
            }
            P.lp += lp;
            if (derivs) { P.g[o + i] += g; P.H[(o + i) * dim + o + i] += h; }
        }
    }
    // the P-spline on c = Q z
    const BayesianDecayPSpline& ps = f.pspline;
    if (ps.order != 2 || ps.family != "gaussian" || ps.space != "log")
        throw std::runtime_error("bayesian_decay_log_prior: only the order-2 Gaussian P-spline in log space is implemented");
    const BayesianDecayVariableInfo vc = bayesian_decay_variable_info(f, "c");
    const BayesianDecayArray& Q = f["Q_c"];
    const std::size_t n = Q.shape[0], nz = Q.shape[1];
    //: lambda: held (fixed_values) or free (a variable `log10_lam`, logit on [lo, hi] -- its own
    //: prior handled with the other variables above); PRD-143 A4.6
    const BayesianDecayVariableInfo vl = bayesian_decay_variable_info(f, "log10_lam");
    const auto fixed_lam = f.fixed_values.find("log10_lam");
    if (vl.present == (fixed_lam != f.fixed_values.end()))
        throw std::runtime_error("bayesian_decay_log_prior: log10_lam must be either held or free, exactly one");
    double x_lam = 0.0, dL_dz = 0.0, d2L_dz2 = 0.0;           // L = ln lambda as a function of its z
    if (vl.present) {
        const BayesianDecayVariable* var = f.variable("log10_lam");
        if (var->transform != "logit") throw std::runtime_error("bayesian_decay_log_prior: a free log10_lam needs the logit transform");
        const double z = th[vl.off], sg = 1.0 / (1.0 + std::exp(-z)), w = var->hi - var->lo, ln10 = std::log(10.0);
        x_lam = var->lo + w * sg;
        dL_dz = ln10 * w * sg * (1.0 - sg);
        d2L_dz2 = ln10 * w * sg * (1.0 - sg) * (1.0 - 2.0 * sg);
    } else {
        x_lam = fixed_lam->second[0];
    }
    const double lam = std::pow(10.0, x_lam);
    const double tilt_sd = ps.tilt_sd, rank = ps.rank;
    std::vector<double> c(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < nz; ++j) c[i] += Q.d[i * nz + j] * th[vc.off + j];
    //: BayesianPSplinePrior (Eilers & Marx 1996): second differences with precision lambda,
    //: a weak Gaussian on the tilt, (rank/2) log lambda; its gradient and Hessian in c,
    //: carried to z through Q
    const BayesianPSplinePrior<double> prior(n, ps.order, tilt_sd, ps.quad_sd);
    if (std::size_t(rank) != std::size_t(prior.rank()))
        throw std::runtime_error("bayesian_decay_log_prior: the manifest's P-spline rank disagrees with the prior's");
    P.lp += prior.log_prob(c.data(), lam);
    if (derivs) {
        std::vector<double> gc(n), Hc(n * n);
        prior.gradient(c.data(), lam, gc.data());
        prior.hessian(c.data(), lam, Hc.data());
        for (std::size_t j = 0; j < nz; ++j) { double s = 0.0; for (std::size_t i = 0; i < n; ++i) s += Q.d[i * nz + j] * gc[i]; P.g[vc.off + j] += s; }
        std::vector<double> HQ(n * nz, 0.0);
        for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < nz; ++j) { double s = 0.0; for (std::size_t k = 0; k < n; ++k) s += Hc[i * n + k] * Q.d[k * nz + j]; HQ[i * nz + j] = s; }
        for (std::size_t a = 0; a < nz; ++a) for (std::size_t b = 0; b < nz; ++b) {
            double s = 0.0; for (std::size_t i = 0; i < n; ++i) s += Q.d[i * nz + a] * HQ[i * nz + b];
            P.H[(vc.off + a) * dim + vc.off + b] += s;
        }
    }
    if (vl.present && derivs) {
        //: in L = ln lambda the prior is (rank/2) L - (lambda/2) q(c) - tilt(c), q = |D c|^2:
        //: f' = rank/2 - lambda q / 2, f'' = -lambda q / 2, d f'/dc = -lambda D'D c
        const std::vector<double>& D = prior.difference_matrix();
        const std::size_t nr = n - std::size_t(ps.order);
        std::vector<double> d(nr, 0.0), DtDc(n, 0.0);
        double q = 0.0;
        for (std::size_t k = 0; k < nr; ++k) { for (std::size_t i = 0; i < n; ++i) d[k] += D[k * n + i] * c[i]; q += d[k] * d[k]; }
        for (std::size_t k = 0; k < nr; ++k) for (std::size_t i = 0; i < n; ++i) DtDc[i] += D[k * n + i] * d[k];
        const double f1 = 0.5 * rank - 0.5 * lam * q, f2 = -0.5 * lam * q;
        const std::size_t zl = vl.off;
        P.g[zl] += f1 * dL_dz;
        P.H[zl * dim + zl] += f2 * dL_dz * dL_dz + f1 * d2L_dz2;
        for (std::size_t j = 0; j < nz; ++j) {
            double s = 0.0; for (std::size_t i = 0; i < n; ++i) s += Q.d[i * nz + j] * DtDc[i];
            const double h = -lam * s * dL_dz;
            P.H[(vc.off + j) * dim + zl] += h;
            P.H[zl * dim + vc.off + j] += h;
        }
    }
    return P;
}

inline double bayesian_decay_log_likelihood(const BayesianDecayExperiment& f, const std::vector<double>& lam) {
    const BayesianDecayArray& y = f["y"], & mask = f["mask"];
    double ll = 0.0;
    for (std::size_t i = 0; i < lam.size(); ++i) {
        if (mask.d[i] == 0.0) continue;
        const double l = std::max(lam[i], 1e-12);
        ll += mask.d[i] * (y.d[i] * std::log(l) - l);
    }
    return ll;
}

inline double bayesian_decay_log_posterior(const BayesianDecayExperiment& f, const BayesianDecayTensors& e, const std::vector<double>& th) {
    const BayesianDecayValues v = bayesian_decay_unpack(f, th);
    return bayesian_decay_log_likelihood(f, bayesian_decay_expected_counts(f, e, v)) + bayesian_decay_log_prior(f, th, false).lp;
}

//! everything one scoring step needs at theta
struct BayesianDecayPoint {
    std::vector<double> theta, lam, J, grad, A;   // grad of log posterior; A = J'WJ - H_prior
    double logpost = 0.0, logprior = 0.0;
};

inline BayesianDecayPoint bayesian_decay_evaluate(const BayesianDecayExperiment& f, const BayesianDecayTensors& e, const std::vector<double>& th) {
    const std::size_t dim = f.dim;
    BayesianDecayPoint p; p.theta = th;
    const BayesianDecayValues v = bayesian_decay_unpack(f, th);
    const std::vector<double> a2 = bayesian_decay_amplitudes(f, e, v);
    const std::vector<double> Ja = bayesian_decay_amplitude_jacobian(f, e, v);
    std::vector<std::vector<std::size_t>> cols;
    p.lam = bayesian_decay_expected_counts(f, e, v, &p.J, &Ja, &a2, &cols);
    const BayesianDecayPrior P = bayesian_decay_log_prior(f, th, true);
    p.logprior = P.lp;
    p.logpost = bayesian_decay_log_likelihood(f, p.lam) + P.lp;
    //: the likelihood's gradient and Fisher matrix through tttrlib::poisson_score_blocks:
    //: each histogram on its live columns, the histograms on the thread pool
    const BayesianDecayArray& y = f["y"], & mask = f["mask"];
    p.grad.assign(dim, 0.0);
    p.A.assign(dim * dim, 0.0);
    ::tttrlib::poisson_score_blocks(y.d.data(), p.lam.data(), p.J.data(), mask.d.data(), p.lam.size(), dim,
                                  ::tttrlib::POISSON_INFORMATION_EXPECTED, cols, f["y"].shape[1], p.grad.data(), p.A.data(),
                                  [](std::size_t n, const std::function<void(std::size_t)>& body) { internal::bayesian_parallel_for(n, body); });
    for (std::size_t a = 0; a < dim; ++a) p.grad[a] += P.g[a];
    for (std::size_t k = 0; k < dim * dim; ++k) p.A[k] -= P.H[k];
    return p;
}

//! The log posterior and its gradient, without the information matrix `bayesian_decay_evaluate`
//! also accumulates: what a gradient-based sampler needs per leapfrog step (PRD-146).
//! `grad = sum_b w_b (y_b/m_b - 1) dm_b/dtheta + d log prior/dtheta`, with the same floor on m.
inline double bayesian_decay_log_posterior_and_gradient(const BayesianDecayExperiment& f, const BayesianDecayTensors& e,
                                                        const std::vector<double>& th, std::vector<double>& grad) {
    const std::size_t dim = f.dim;
    const BayesianDecayValues v = bayesian_decay_unpack(f, th);
    const std::vector<double> a2 = bayesian_decay_amplitudes(f, e, v);
    const std::vector<double> Ja = bayesian_decay_amplitude_jacobian(f, e, v);
    std::vector<double> J;
    const std::vector<double> lam = bayesian_decay_expected_counts(f, e, v, &J, &Ja, &a2);
    const BayesianDecayPrior P = bayesian_decay_log_prior(f, th, true);
    const BayesianDecayArray& y = f["y"], & mask = f["mask"];
    grad.assign(dim, 0.0);
    double ll = 0.0;
    for (std::size_t b = 0; b < lam.size(); ++b) {
        const double w = mask.d[b];
        if (w == 0.0) continue;
        const double m = std::max(lam[b], 1e-12);
        ll += w * (y.d[b] * std::log(m) - m);
        const double u = w * (y.d[b] / m - 1.0);
        if (u == 0.0) continue;
        const double* row = &J[b * dim];
        for (std::size_t c = 0; c < dim; ++c) grad[c] += u * row[c];
    }
    for (std::size_t c = 0; c < dim; ++c) grad[c] += P.g[c];
    return ll + P.lp;
}

//! the Laplace evidence at a point: log p(y | lambda) = log p(y, theta*) + d/2 log 2 pi - 1/2 log det A
inline double bayesian_decay_laplace_evidence(const BayesianDecayPoint& p, std::size_t dim, bool* ok = nullptr) {
    double ev = 0.0;
    const bool good = bayesian_laplace_log_evidence(p.logpost, p.A.data(), dim, &ev);
    if (ok) *ok = good;
    return good ? ev : -std::numeric_limits<double>::infinity();
}

// ---------------------------------------------------------------------------
// Rule 0 and the posterior summaries (B8)
// ---------------------------------------------------------------------------

//! the Laplace bayesian_decay_covariance A^-1 by one Cholesky factor and dim solves
inline std::vector<double> bayesian_decay_covariance(const BayesianDecayPoint& p, std::size_t dim) {
    CholeskyFactor ch;
    std::vector<double> S(dim * dim, 0.0), e(dim, 0.0), col(dim);
    if (!ch.factor(p.A.data(), dim)) return {};
    for (std::size_t j = 0; j < dim; ++j) {
        std::fill(e.begin(), e.end(), 0.0); e[j] = 1.0;
        ch.solve(e.data(), col.data());
        for (std::size_t i = 0; i < dim; ++i) S[i * dim + j] = col[i];
    }
    return S;
}

//! p(R/R0) at theta and its delta-method sd (s88.delta_bands, one node):
//! J_p = (diag p - p p') spl Q on the c block, var_j = J_p Sigma_cc J_p'
inline void bayesian_decay_distribution_with_sd(const BayesianDecayExperiment& f, const BayesianDecayTensors& e, const std::vector<double>& th, const std::vector<double>& Sig,
                      std::vector<double>& p, std::vector<double>& sd, double* mean_rel = nullptr, double* mean_rel_sd = nullptr) {
    const std::size_t dim = f.dim;
    p = bayesian_decay_distribution(e, bayesian_decay_unpack(f, th));
    const BayesianDecayVariableInfo vc = bayesian_decay_variable_info(f, "c");
    const BayesianDecayArray& Q = f["Q_c"];
    const std::size_t nR = p.size(), nc = e.spl->shape[1], nz = Q.shape[1];
    std::vector<double> BQ(nR * nz, 0.0), pBQ(nz, 0.0), Jp(nR * nz);
    for (std::size_t j = 0; j < nR; ++j) for (std::size_t a = 0; a < nc; ++a) {
        const double s_ = e.spl->d[j * nc + a];
        if (s_ != 0.0) for (std::size_t b = 0; b < nz; ++b) BQ[j * nz + b] += s_ * Q.d[a * nz + b];
    }
    for (std::size_t j = 0; j < nR; ++j) for (std::size_t b = 0; b < nz; ++b) pBQ[b] += p[j] * BQ[j * nz + b];
    for (std::size_t j = 0; j < nR; ++j) for (std::size_t b = 0; b < nz; ++b) Jp[j * nz + b] = p[j] * (BQ[j * nz + b] - pBQ[b]);
    //: the delta method through bayesian_delta_variance, on the c block of Sigma
    std::vector<double> Scc(nz * nz);
    for (std::size_t a = 0; a < nz; ++a) for (std::size_t b = 0; b < nz; ++b) Scc[a * nz + b] = Sig[(vc.off + a) * dim + vc.off + b];
    sd.assign(nR, 0.0);
    bayesian_delta_variance(Jp.data(), Scc.data(), nR, nz, sd.data());
    for (double& t : sd) t = std::sqrt(t);
    if (mean_rel) {
        //: the mean of R/R0 and its delta-method sd: d mean / d z = rel' J_p
        const std::vector<double>& rel = f["rel"].d;
        std::vector<double> jm(nz, 0.0);
        double m = 0.0;
        for (std::size_t j = 0; j < nR; ++j) { m += p[j] * rel[j]; for (std::size_t b = 0; b < nz; ++b) jm[b] += rel[j] * Jp[j * nz + b]; }
        double v = 0.0;
        bayesian_delta_variance(jm.data(), Scc.data(), 1, nz, &v);
        *mean_rel = m;
        if (mean_rel_sd) *mean_rel_sd = std::sqrt(v);
    }
}

/**
 * \brief The tilt `power * log(mean R/R0 + eps)` with its exact gradient and Hessian.
 *
 * Tierney-Kadane needs the posterior tilted by a power of the summary, and needs the tilt's
 * curvature exactly (see `bayesian_tierney_kadane`). For the mean of `R/R0` that curvature is
 * analytic and cheap: `p = softmax(spl Q z)` depends on `theta` only through the spline block,
 * so with `u = spl Q z`, `f = sum_j p_j rel_j` and `g_j = p_j (rel_j - f)`,
 *
 *     df/du_k    = g_k
 *     d2f/du_k du_l = delta_kl g_k - g_k p_l - p_k g_l
 *
 * and the tilt's own derivatives follow from `d log(f + eps)`. Everything outside the spline
 * block is exactly zero, which is also why the tilted fit costs no more than an untilted one.
 */
struct BayesianDecayTilt {
    double f = 0.0;                //!< the summary itself, mean R/R0
    double value = 0.0;            //!< power * log(f + eps)
    std::vector<double> grad;      //!< d value / d theta, `dim`
    std::vector<double> hess;      //!< d2 value / d theta2, `dim x dim` row-major (empty if not asked for)
};

inline BayesianDecayTilt bayesian_decay_mean_rel_tilt(const BayesianDecayExperiment& f, const BayesianDecayTensors& e,
                                                      const std::vector<double>& th, double power, double eps,
                                                      bool want_hessian = true) {
    const std::size_t dim = f.dim;
    BayesianDecayTilt T;
    T.grad.assign(dim, 0.0);
    if (want_hessian) T.hess.assign(dim * dim, 0.0);
    const std::vector<double> p = bayesian_decay_distribution(e, bayesian_decay_unpack(f, th));
    const std::vector<double>& rel = f["rel"].d;
    const BayesianDecayVariableInfo vc = bayesian_decay_variable_info(f, "c");
    const BayesianDecayArray& Q = f["Q_c"];
    const std::size_t nR = p.size(), nc = e.spl->shape[1], nz = Q.shape[1];
    std::vector<double> M(nR * nz, 0.0);                      // spl @ Q, so u = M z
    for (std::size_t j = 0; j < nR; ++j) for (std::size_t a = 0; a < nc; ++a) {
        const double s_ = e.spl->d[j * nc + a];
        if (s_ != 0.0) for (std::size_t b = 0; b < nz; ++b) M[j * nz + b] += s_ * Q.d[a * nz + b];
    }
    double fv = 0.0;
    for (std::size_t j = 0; j < nR; ++j) fv += p[j] * rel[j];
    T.f = fv;
    std::vector<double> gj(nR), G(nz, 0.0), P(nz, 0.0);
    for (std::size_t j = 0; j < nR; ++j) gj[j] = p[j] * (rel[j] - fv);
    for (std::size_t j = 0; j < nR; ++j) for (std::size_t b = 0; b < nz; ++b) {
        G[b] += gj[j] * M[j * nz + b];
        P[b] += p[j] * M[j * nz + b];
    }
    const double d = fv + eps, k1 = power / d;
    T.value = power * std::log(d);
    for (std::size_t b = 0; b < nz; ++b) T.grad[vc.off + b] = k1 * G[b];
    if (want_hessian) {
        std::vector<double> W(nz * nz, 0.0);                  // M' diag(g) M
        for (std::size_t j = 0; j < nR; ++j) {
            if (gj[j] == 0.0) continue;
            for (std::size_t a = 0; a < nz; ++a) {
                const double t = gj[j] * M[j * nz + a];
                for (std::size_t b = 0; b < nz; ++b) W[a * nz + b] += t * M[j * nz + b];
            }
        }
        const double k2 = power / (d * d);
        for (std::size_t a = 0; a < nz; ++a) for (std::size_t b = 0; b < nz; ++b) {
            const double d2f = W[a * nz + b] - G[a] * P[b] - P[a] * G[b];
            T.hess[(vc.off + a) * dim + vc.off + b] = k1 * d2f - k2 * G[a] * G[b];
        }
    }
    return T;
}

//! one node fit: Fisher scoring stepped by imp.bff's DampedNewton, then the
//! Laplace evidence (the loop cbm56_fit.cpp ran inline until the mixture needed
//! it several times)
struct BayesianDecayFit {
    std::vector<double> theta;
    BayesianDecayPoint pt;
    double evidence = -std::numeric_limits<double>::infinity(), dec = std::nan(""), t_eval = 0.0, t_search = 0.0, t_fit = 0.0;
    bool converged = false, chol = false;
    int iterations = 0, n_obj = 0, n_fallback = 0;
};

inline BayesianDecayFit bayesian_decay_fit_node(const BayesianDecayExperiment& f, const BayesianDecayTensors& env, std::vector<double> th, double line_search_below = 1.0,
                          int max_iter = 300, double tol = 1e-6, bool verbose = false) {
    using clk_ = std::chrono::steady_clock;
    auto secs_ = [](clk_::time_point a) { return std::chrono::duration<double>(clk_::now() - a).count(); };
    const auto t0 = clk_::now();
    const std::size_t dim = f.dim;
    BayesianDecayFit R;
    BayesianDecayPoint pt = bayesian_decay_evaluate(f, env, th);
    int it = 0;
    auto objective = [&](const double* t) { ++R.n_obj; return -bayesian_decay_log_posterior(f, env, std::vector<double>(t, t + dim)); };
    DampedNewton<decltype(objective)> stepper;
    stepper.line_search_below = line_search_below;
    for (; it < max_iter; ++it) {
        const auto ts = clk_::now();
        const OptimizationStepResult r = stepper.step(pt.A.data(), pt.grad.data(), dim, -pt.logpost, th.data(), objective);
        R.t_search += secs_(ts);
        if (!r.accepted) { R.converged = std::isfinite(R.dec) && R.dec < 1e-3; break; }
        R.n_fallback += r.gradient_fallback ? 1 : 0;
        R.dec = r.decrement;
        const auto te = clk_::now();
        pt = bayesian_decay_evaluate(f, env, th);
        R.t_eval += secs_(te);
        if (verbose) std::printf("  step %3d: log post %.4f, decrement %.3e, mu %.1e, alpha %.3g%s\n", it, pt.logpost, R.dec, r.mu, r.alpha, r.gradient_fallback ? " (gradient)" : "");
        if (R.dec < tol) { R.converged = true; ++it; break; }
    }
    if (!R.converged && std::isfinite(R.dec) && R.dec < 1e-3) R.converged = true;
    R.iterations = it;
    R.evidence = bayesian_decay_laplace_evidence(pt, dim, &R.chol);
    R.theta = std::move(th); R.pt = std::move(pt);
    R.t_fit = secs_(t0);
    return R;
}

/**
 * \brief The exact Hessian of minus the log posterior, by central differences of the
 *        analytic gradient, symmetrised: `H_ij = (g_i(theta + h e_j) - g_i(theta - h e_j)) / 2h`.
 *
 * **Why it is needed.** Fisher scoring's matrix `J' diag(w/m) J - prior` drops the term
 * `sum_b w_b (y_b/m_b - 1) d2m_b/dtheta2`. At a mode that fits it is small per bin but not
 * in total, and on CBM56 it is where the Laplace evidence differed: 11.7 nats, all of it that
 * term (observed vs expected information: 0.04), concentrated in weakly identified,
 * nonlinearly parameterised blocks -- the donor spectrum (5.3), the acceptor weights (2.4),
 * the rotational weights, the spline -- with the exact curvature up to 5.6x Fisher's there
 * (PRD-143 A4.9). The Laplace approximation is defined by the exact Hessian.
 *
 * `h = rel * max(1, |theta_j|)`; `2 dim` gradient evaluations, run serially over coordinates
 * (each evaluation already uses the pool).
 */
inline std::vector<double> bayesian_decay_hessian(const BayesianDecayExperiment& f, const BayesianDecayTensors& env,
                                                  const std::vector<double>& theta, double rel = 1e-5) {
    const std::size_t dim = f.dim;
    std::vector<double> H(dim * dim, 0.0);
    for (std::size_t j = 0; j < dim; ++j) {
        const double h = rel * std::max(1.0, std::fabs(theta[j]));
        std::vector<double> tp = theta, tm = theta;
        tp[j] += h; tm[j] -= h;
        const BayesianDecayPoint gp = bayesian_decay_evaluate(f, env, tp), gm = bayesian_decay_evaluate(f, env, tm);
        for (std::size_t i = 0; i < dim; ++i) H[i * dim + j] = -(gp.grad[i] - gm.grad[i]) / (2.0 * h);   // grad is of +log posterior
    }
    for (std::size_t i = 0; i < dim; ++i)
        for (std::size_t j = i + 1; j < dim; ++j) {
            const double v = 0.5 * (H[i * dim + j] + H[j * dim + i]);
            H[i * dim + j] = H[j * dim + i] = v;
        }
    return H;
}

/**
 * \brief Polish a mode with the exact Hessian: damped Newton steps (`DampedNewton`, which
 *        adds to the diagonal until the matrix is positive definite) on `bayesian_decay_hessian`.
 *
 * A scoring fit stops where its Fisher decrement is small. On a posterior with ripples that
 * is not always a point where the exact curvature is negative definite, and the exact-Hessian
 * evidence is only defined at such a point (on the CBM56 lambda grid one node's exact Hessian
 * was indefinite after scoring). Stops at decrement `tol` (g' H^-1 g) or `max_iter` steps.
 */
inline BayesianDecayFit bayesian_decay_polish_exact(const BayesianDecayExperiment& f, const BayesianDecayTensors& env,
                                                    BayesianDecayFit fit, int max_iter = 20, double tol = 1e-8) {
    const std::size_t dim = f.dim;
    std::vector<double> th = fit.theta;
    auto objective = [&](const double* t) { return -bayesian_decay_log_posterior(f, env, std::vector<double>(t, t + dim)); };
    DampedNewton<decltype(objective)> stepper;
    BayesianDecayPoint pt = fit.pt;
    int it = 0;
    for (; it < max_iter; ++it) {
        const std::vector<double> H = bayesian_decay_hessian(f, env, th);
        const OptimizationStepResult r = stepper.step(H.data(), pt.grad.data(), dim, -pt.logpost, th.data(), objective);
        if (!r.accepted) break;
        pt = bayesian_decay_evaluate(f, env, th);
        fit.dec = r.decrement;
        if (r.decrement < tol) { ++it; break; }
    }
    fit.theta = th; fit.pt = std::move(pt); fit.iterations += it;
    return fit;
}

//! The Laplace evidence at a mode with the exact Hessian (`bayesian_decay_hessian`) instead of
//! the Fisher information: `log p(y, theta*) + dim/2 log 2 pi - 1/2 log det H`.
inline double bayesian_decay_laplace_evidence_exact(const BayesianDecayExperiment& f, const BayesianDecayTensors& env,
                                                    const BayesianDecayPoint& mode, bool* ok = nullptr) {
    BayesianDecayPoint q = mode;
    q.A = bayesian_decay_hessian(f, env, mode.theta);
    return bayesian_decay_laplace_evidence(q, f.dim, ok);
}

/**
 * \brief The mode of the tilted posterior `log p(y, theta) + power * log(mean R/R0 + eps)`.
 *
 * The same damped Newton loop as `bayesian_decay_fit_node`, with the tilt's exact gradient and
 * curvature (`bayesian_decay_mean_rel_tilt`) added to the scoring matrix. The returned `pt` is
 * the UNTILTED evaluation at the tilted mode, because that is what the Tierney-Kadane ratio
 * needs; `dec` is the tilted decrement.
 */
inline BayesianDecayFit bayesian_decay_fit_node_tilted(const BayesianDecayExperiment& f, const BayesianDecayTensors& env,
                                                       std::vector<double> th, double power, double eps,
                                                       int max_iter = 100, double tol = 1e-8) {
    const std::size_t dim = f.dim;
    BayesianDecayFit R;
    auto tilted_point = [&](const std::vector<double>& t) {
        BayesianDecayPoint pt = bayesian_decay_evaluate(f, env, t);
        const BayesianDecayTilt T = bayesian_decay_mean_rel_tilt(f, env, t, power, eps);
        pt.logpost += T.value;
        for (std::size_t i = 0; i < dim; ++i) pt.grad[i] += T.grad[i];
        for (std::size_t i = 0; i < dim * dim; ++i) pt.A[i] -= T.hess[i];      //: A is the curvature of MINUS the log posterior
        return pt;
    };
    auto objective = [&](const double* t) {
        const std::vector<double> v(t, t + dim);
        ++R.n_obj;
        return -(bayesian_decay_log_posterior(f, env, v) + bayesian_decay_mean_rel_tilt(f, env, v, power, eps, false).value);
    };
    BayesianDecayPoint pt = tilted_point(th);
    DampedNewton<decltype(objective)> stepper;
    int it = 0;
    for (; it < max_iter; ++it) {
        const OptimizationStepResult r = stepper.step(pt.A.data(), pt.grad.data(), dim, -pt.logpost, th.data(), objective);
        if (!r.accepted) { R.converged = std::isfinite(R.dec) && R.dec < 1e-3; break; }
        R.dec = r.decrement;
        R.n_fallback += r.gradient_fallback ? 1 : 0;
        pt = tilted_point(th);
        if (R.dec < tol) { R.converged = true; ++it; break; }
    }
    if (!R.converged && std::isfinite(R.dec) && R.dec < 1e-3) R.converged = true;
    R.iterations = it;
    R.theta = th;
    R.pt = bayesian_decay_evaluate(f, env, th);
    return R;
}

//! Tierney-Kadane moments of mean R/R0 at one node, with what the two tilted fits did.
struct BayesianDecayTkSummary {
    double mean = std::nan(""), sd = std::nan("");
    bool ok = false;
    int iterations = 0;            //!< both tilted fits together
    double seconds = 0.0;
    //! per power (1, 2): how far the tilted mode moved from the node's, in the node's own metric
    //! (`sqrt(d' A0 d)`, so 1 is one posterior sd), and the log-determinant the tilt added. The
    //! displacement should be about `power * sd(f) / f`: a Laplace ratio whose modes are far apart is
    //! outside the approximation's range, which is why it is reported and gated on.
    double shift[2] = {std::nan(""), std::nan("")};
    double dlogdet[2] = {std::nan(""), std::nan("")};
    //! **The flattest direction, which is where this approximation dies.** `flat_curvature` is the
    //! smallest eigenvalue of the node's own precision and `tilt_on_flat` the tilt's curvature along
    //! that same direction; `flat_ratio` is their ratio per power. Where the ratio approaches one the
    //! log-determinant change -- which IS the correction, once the tilted mode barely moves -- is a change
    //! of order one in the log-volume of a direction the data do not constrain and where the posterior is
    //! not Gaussian at all. ucfret measured exactly that and rejected TK on those grounds
    //! (`okf/log.md` 2026-09-06 23:55: an eigenvalue of 2.7e-3 in a spline direction where p(R) is empty,
    //! a tilt curvature of 7e-3 there, and a 4 % shift of the mean that belongs to the volume, not to the
    //! mean). A TK number whose `flat_ratio` is not small is not a better moment; it is an artefact.
    double flat_curvature = std::nan("");
    double tilt_on_flat[2] = {std::nan(""), std::nan("")};
    double flat_ratio[2] = {std::nan(""), std::nan("")};
    //! the share of the whole log-determinant change that this ONE flattest direction contributes,
    //! `log1p(tilt_on_flat / flat_curvature) / dlogdet`. Once the tilted mode barely moves, the
    //! log-determinant change IS the correction, so a share near one says the correction is the volume of a
    //! single direction the data do not constrain -- and the number should not be quoted as a moment.
    double flat_share[2] = {std::nan(""), std::nan("")};
    //! **What holds the flattest direction.** `flat_prior_share` is the fraction of that direction's
    //! curvature supplied by the PRIOR alone, `v' (-H_prior) v / flat_curvature`. At one, the data say
    //! nothing there and the direction's width is whatever the roughness prior chose; a correction that
    //! lives in it is a property of the prior and the Gaussian approximation, not of the posterior mean.
    //! ucfret's 2026-09-06 rejection of TK is exactly this case ("a direction of the spline coefficients
    //! where p(R) is empty and only the roughness prior at lambda 1 holds the coefficient").
    double flat_prior_share = std::nan("");
};

/**
 * \brief The posterior mean and sd of mean R/R0 at one node by Tierney-Kadane (PRD-149 step 3).
 *
 * Two tilted fits (`power` 1 and 2) started from the node's own mode, combined by
 * `bayesian_tierney_kadane`. Both tilted integrals are given the curvature rule that ratio
 * needs: the node's own matrix plus the exact Hessian of the tilt at the tilted mode, never a
 * second independent estimate of the whole curvature.
 *
 * Where the summary is nearly linear over the posterior's width this returns the delta method's
 * answer; where it is not -- a distribution's mean near the edge of its support, a node whose
 * p(R/R0) is sharply peaked -- it is the one with the smaller error (`O(n^-2)` against
 * `O(n^-1)`), which is why the prototype applies it to the nodes carrying real weight.
 */
inline BayesianDecayTkSummary bayesian_decay_tk_mean_rel(const BayesianDecayExperiment& f, const BayesianDecayTensors& env,
                                                         const BayesianDecayFit& node, double eps = 1e-6, int max_iter = 100,
                                                         bool exact_curvature = false) {
    using clk_ = std::chrono::steady_clock;
    const auto t0 = clk_::now();
    BayesianDecayTkSummary out;
    const std::size_t dim = f.dim;
    //: the base curvature both integrals share: Fisher's matrix (the prototype's rule) or the exact Hessian
    const std::vector<double> A0 = exact_curvature ? bayesian_decay_hessian(f, env, node.theta) : node.pt.A;
    double logdet0 = 0.0;
    if (!bayesian_log_det_spd(A0.data(), dim, &logdet0)) return out;
    //: the node's flattest direction, against which the tilt's curvature is measured below
    std::vector<double> vflat(dim, 0.0);
    double lam_min = std::nan("");
    const bool have_flat = bayesian_smallest_eigenpair(A0.data(), dim, vflat.data(), &lam_min);
    out.flat_curvature = have_flat ? lam_min : std::nan("");
    if (have_flat && lam_min > 0.0) {
        //: how much of that direction's curvature the prior alone supplies
        const BayesianDecayPrior pr = bayesian_decay_log_prior(f, node.theta, true);
        double q = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            double si = 0.0;
            for (std::size_t j = 0; j < dim; ++j) si -= pr.H[i * dim + j] * vflat[j];
            q += vflat[i] * si;
        }
        out.flat_prior_share = q / lam_min;
    }
    const double L0 = node.pt.logpost;
    double lr[2] = {std::nan(""), std::nan("")};
    for (int k = 0; k < 2; ++k) {
        const double power = double(k + 1);
        const BayesianDecayFit t = bayesian_decay_fit_node_tilted(f, env, node.theta, power, eps, max_iter);
        out.iterations += t.iterations;
        if (!t.converged || !std::isfinite(t.pt.logpost)) return out;
        const BayesianDecayTilt T = bayesian_decay_mean_rel_tilt(f, env, t.theta, power, eps);
        std::vector<double> H = A0;                                 //: the NODE's matrix ...
        for (std::size_t i = 0; i < dim * dim; ++i) H[i] -= T.hess[i];   //: ... plus the tilt's exact curvature at the tilted mode
        double logdet1 = 0.0;
        if (!bayesian_log_det_spd(H.data(), dim, &logdet1)) return out;
        lr[k] = (t.pt.logpost + T.value) - L0 - 0.5 * (logdet1 - logdet0);
        out.dlogdet[k] = logdet1 - logdet0;
        double d2 = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            const double di = t.theta[i] - node.theta[i];
            if (di == 0.0) continue;
            for (std::size_t j = 0; j < dim; ++j) d2 += di * A0[i * dim + j] * (t.theta[j] - node.theta[j]);
        }
        out.shift[k] = std::sqrt(d2 > 0.0 ? d2 : 0.0);
        if (have_flat) {
            //: v' (-d2 tilt) v, the tilt's curvature along the direction the data leave flat
            double q = 0.0;
            for (std::size_t i = 0; i < dim; ++i) {
                double si = 0.0;
                for (std::size_t j = 0; j < dim; ++j) si -= T.hess[i * dim + j] * vflat[j];
                q += vflat[i] * si;
            }
            out.tilt_on_flat[k] = q;
            out.flat_ratio[k] = lam_min > 0.0 ? std::fabs(q) / lam_min : std::numeric_limits<double>::infinity();
            if (lam_min > 0.0 && std::fabs(out.dlogdet[k]) > 0.0)
                out.flat_share[k] = std::log1p(q / lam_min) / out.dlogdet[k];
        }
    }
    const BayesianTierneyKadane r = bayesian_tierney_kadane(lr[0], lr[1], eps);
    out.mean = r.mean; out.sd = r.sd; out.ok = r.ok;
    out.seconds = std::chrono::duration<double>(clk_::now() - t0).count();
    return out;
}

/**
 * \brief A declared start: where a fit begins and how it walks from there.
 *
 * `adam_steps > 0` runs that many Adam steps (tttrlib `adam_update`, Kingma & Ba 2015,
 * on minus the log posterior with the full gradient, step size `adam_lr`) before the
 * scoring fit. On CBM56 every start without such a phase -- three line-search schedules
 * and ten Fisher-scaled perturbations of the data start -- ends in the posterior's
 * second mode, 0.48 nats below the first; 150 Adam steps reach the first (PRD-143 A4.7).
 */
struct BayesianDecayStart {
    std::string name;
    std::vector<double> theta;
    double line_search_below = 1.0;
    int adam_steps = 0;
    double adam_lr = 0.02;
};

//! The declared starts from one data-derived start: the start itself, and three Adam
//! phases (150 steps at 0.02 and 0.05, 50 at 0.1) -- each depends only on the model and
//! the data, as MCTS's declared starts do (`ModelSearch.h`, `add_structure_start`).
inline std::vector<BayesianDecayStart> bayesian_decay_declared_starts(const std::vector<double>& data_start) {
    return {{"data start", data_start, 1.0, 0, 0.0},
            {"data start + Adam 150 x 0.02", data_start, 1.0, 150, 0.02},
            {"data start + Adam 150 x 0.05", data_start, 1.0, 150, 0.05},
            {"data start + Adam 50 x 0.1", data_start, 1.0, 50, 0.1}};
}

//! Every declared start's fit, and which is best.
struct BayesianDecayMultiFit {
    std::vector<BayesianDecayFit> fits;
    std::size_t best = 0;
};

/**
 * \brief Fit every declared start and keep the highest log posterior.
 *
 * The starts run `n_concurrent` at a time, each on its own thread with the pool's
 * loops inline (`bayesian_serial_here`). Every fit is returned, so a caller reports
 * where each start ended -- two starts that end apart are a second mode, and one
 * start alone cannot show that.
 */
inline BayesianDecayMultiFit bayesian_decay_fit_best_of_starts(const BayesianDecayExperiment& f, const BayesianDecayTensors& env,
                                                              const std::vector<BayesianDecayStart>& starts, int max_iter = 1000,
                                                              std::size_t n_concurrent = 4) {
    const std::size_t dim = f.dim;
    BayesianDecayMultiFit out;
    out.fits.resize(starts.size());
    std::atomic<std::size_t> next{0};
    auto work = [&] {
        internal::bayesian_serial_here() = true;
        for (;;) {
            const std::size_t i = next.fetch_add(1);
            if (i >= starts.size()) return;
            std::vector<double> th = starts[i].theta;
            if (starts[i].adam_steps > 0) {
                ::tttrlib::AdamState st;
                st.reset(dim);
                std::vector<double> neg(dim);
                for (int k = 0; k < starts[i].adam_steps; ++k) {
                    const BayesianDecayPoint pt = bayesian_decay_evaluate(f, env, th);
                    for (std::size_t a = 0; a < dim; ++a) neg[a] = -pt.grad[a];
                    ::tttrlib::adam_update(th.data(), neg.data(), dim, st, starts[i].adam_lr);
                }
            }
            out.fits[i] = bayesian_decay_fit_node(f, env, th, starts[i].line_search_below, max_iter);
        }
    };
    std::vector<std::thread> pool;
    for (std::size_t w = 0; w < std::max<std::size_t>(1, std::min(n_concurrent, starts.size())); ++w) pool.emplace_back(work);
    for (auto& t : pool) t.join();
    //: a start whose fit failed (a non-finite log posterior) never wins, wherever it sits in the list
    auto score = [&](std::size_t i) { const double v = out.fits[i].pt.logpost; return std::isfinite(v) ? v : -std::numeric_limits<double>::infinity(); };
    for (std::size_t i = 1; i < out.fits.size(); ++i)
        if (score(i) > score(out.best)) out.best = i;
    return out;
}

//! One node of the lambda grid: the fit at that log10 lambda and its p(R/R0) summary.
struct BayesianDecayLambdaNode {
    double log10_lam = 0.0;
    BayesianDecayFit fit;
    std::vector<double> p, p_sd;
    double mean_rel = 0.0, mean_rel_sd = 0.0;
    //! the same summary by Tierney-Kadane, where it was asked for and succeeded (`summary_method = "tk"`);
    //! the delta-method pair above is always filled, so the two can be reported side by side
    double tk_mean = std::nan(""), tk_sd = std::nan("");
    bool tk_ok = false;
    //! what the two tilted fits did there (`BayesianDecayTkSummary::shift`, `dlogdet`), so a node whose
    //! tilted modes ran far from its own can be seen and not quoted
    double tk_shift[2] = {std::nan(""), std::nan("")}, tk_dlogdet[2] = {std::nan(""), std::nan("")};
    //! the flattest direction's curvature at this node, and the tilt's curvature along it relative to it
    double tk_flat_curvature = std::nan(""), tk_flat_ratio[2] = {std::nan(""), std::nan("")};
    double tk_flat_share[2] = {std::nan(""), std::nan("")};
    double tk_flat_prior_share = std::nan("");
};

//! The grid, its evidence weights and the mixture's p(R/R0).
struct BayesianDecayLambdaGrid {
    std::vector<BayesianDecayLambdaNode> nodes;
    std::vector<double> weights, p_mean, p_sd;
    double mean_rel = 0.0, mean_rel_sd = 0.0;
    std::size_t n_coarse = 0;        //!< nodes of the first pass; the rest were added by refinement
    std::size_t n_dropped = 0;       //!< nodes whose evidence was not finite (weight zero)
    std::size_t n_improved = 0;      //!< nodes a sweep moved to a better mode found from a neighbour
    std::size_t n_tk = 0;            //!< nodes whose mean R/R0 came out of a Tierney-Kadane pair of tilted fits
    std::size_t n_screened = 0;      //!< sweep trials considered
    std::size_t n_swept_fits = 0;    //!< sweep trials that actually cost a fit (the rest were screened out)
    //! quantiles of the mixture (not of the moments): mean R/R0 at 16/50/84 %, and p(R/R0) bin by bin
    double mean_rel_q16 = 0.0, mean_rel_q50 = 0.0, mean_rel_q84 = 0.0;
    std::vector<double> p_q16, p_q84;
};

/**
 * \brief How the lambda grid is laid out and weighted -- the rules of ucfret's Python prototype
 *        (`s88_laplace_posterior.py`), which this is the C++ side of.
 */
struct BayesianDecayLambdaOptions {
    //! the roughness hyperparameter's tilt: the weight of a node is `exp(evidence + prior_slope * log10 lambda)`,
    //! so a node one decade rougher must beat the smoother one by `prior_slope` nats ("smooth unless the data
    //! insist"). Zero: the evidence alone decides.
    double prior_slope = 0.0;
    //! after the coarse pass, nodes this far apart are added across the range where the tilted evidence lies
    //! within `refine_window` nats of its maximum (half a decade beyond the kept nodes), each warm-started from
    //! its nearest fitted node -- the hyperparameter explored around its mode, as INLA does it. 0 turns it off.
    double refine_step = 0.25;
    double refine_window = 10.0;
    int max_refine = 64;             //!< a cap on added nodes, so a flat evidence cannot run away
    //! after every node is fitted, sweep the grid again refitting each node from its neighbours' end points and
    //! keeping the higher posterior. A node warm-started from one side only can stay in a worse mode, and its
    //! evidence -- which is what weights the mixture -- is then wrong: on CBM56 D0+DA the one-directional grid
    //! had a 17-nat cliff between neighbouring nodes. 0 turns the sweeps off (the walk outward only).
    int sweeps = 2;
    //! before a sweep trial costs a fit, score the neighbour's end point at this node's lambda and skip the
    //! trial unless it is already better than the node's own mode -- one evaluation against a whole fit.
    //! **Off by default, because it is not free**: on CBM56 D0+DA it took the grid from 89 s to 62 s but
    //! found 9 of the 18 moves, shifting single-node evidences by up to 0.16 nats and the mixture's mean
    //! R/R0 from 1.07957 to 1.07991 (PRD-149 A1). A trial started from a point that is worse at this node
    //! can still descend into a better mode, which is what the screen throws away.
    bool sweep_screen = false;
    //! the evidence with the exact Hessian (`bayesian_decay_polish_exact` + `bayesian_decay_laplace_evidence_exact`)
    //! instead of Fisher's; on CBM56 it jumps ~2 nats between neighbours and is indefinite at one node (A4.9)
    bool exact_evidence = false;
    int max_iter = 1000;
    //! the node fit's stopping rule: `g' A^-1 g / 2` below this. It is not a formality -- against the Python
    //! prototype on CBM56 both sides stopped 0.07-0.6 nats short of their modes at the default, which is
    //! larger than the 0.05 nats an evidence comparison between the two was declared to need (PRD-149 A3).
    double tol = 1e-6;
    //! `"delta"` (the linearisation at each node's mode) or `"tk"` (Tierney-Kadane, two tilted fits per node).
    //! TK is run only where a node carries at least `tk_min_weight` of the mixture -- it doubles that node's
    //! fitting cost, and a node with no weight cannot move the answer. `tk_eps` keeps `log(f + eps)` finite.
    std::string summary_method = "delta";
    double tk_min_weight = 0.02;
    double tk_eps = 1e-6;
    int tk_max_iter = 100;
    //! the curvature both tilted integrals share: Fisher's matrix (false, the prototype's rule and the
    //! default, because the exact Hessian is indefinite at some CBM56 nodes -- A4.9) or the exact Hessian.
    //! Measured on the simulated fixture against a converged NUTS reference (PRD-149 A2): the exact base
    //! gives the sd to 0.6 % and the mean to 0.25 posterior sd, Fisher's 2.5 % and 0.39.
    bool tk_exact_curvature = false;
};

/**
 * \brief The P-spline weight varied, not held: a fit per log10 lambda node, weighted
 *        by its Laplace evidence p(y | lambda) (Rue, Martino & Chopin 2009's grid).
 *
 * **Why not lambda as a free parameter of one fit.** `p(c | lambda)` carries
 * `lambda^{rank/2}` and grows without bound as the coefficients approach the penalty's
 * null space, so a joint maximum over (c, lambda) runs to large lambda: on CBM56 it went
 * to log10 lambda 3.9 while the evidence, with c integrated out, put 40 % of its weight at
 * -1 (PRD-143 A4.6). The evidence is the density of lambda given the data (with its flat
 * prior on log10 lambda), so the grid integrates lambda out; the mixture's sd includes
 * the spread between nodes (`BayesianEvidenceMixture`).
 *
 * Evidence from the Fisher information by default. `exact_evidence` polishes each node with
 * the exact Hessian and uses it -- the Laplace approximation's definition -- but on CBM56 that
 * evidence jumps by ~2 nats between neighbouring nodes and is undefined (indefinite) at one,
 * after the polish too: the posterior is not locally Gaussian there, and neither matrix gives
 * reliable weights; sampling does (PRD-143 A4.9/A4.10).
 *
 * The node nearest `theta_centre`'s lambda is fitted from `theta_centre`, the others from
 * their inner neighbour's end, outwards. `f` is copied; its `fixed_values["log10_lam"]`
 * is set per node, and `env` (the tensors) does not depend on it.
 */
inline BayesianDecayLambdaGrid bayesian_decay_fit_lambda_grid(const BayesianDecayExperiment& f, const BayesianDecayTensors& env,
                                                             const std::vector<double>& log10_nodes, const std::vector<double>& theta_centre,
                                                             double log10_centre, const BayesianDecayLambdaOptions& opt) {
    BayesianDecayLambdaGrid G;
    std::vector<double> nodes = log10_nodes;
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    BayesianDecayExperiment g = f;
    //: one node, fitted from `start`, inserted so that `G.nodes` stays sorted by lambda
    auto fit_into = [&](double x, const std::vector<double>& start) {
        BayesianDecayLambdaNode N;
        N.log10_lam = x;
        g.fixed_values["log10_lam"] = {x};
        N.fit = bayesian_decay_fit_node(g, env, start, 1.0, opt.max_iter, opt.tol);
        //: the node's evidence with the exact Hessian (bayesian_decay_hessian): Fisher's drops the
        //: second-derivative term, 11.7 nats on CBM56, and the weights are differences of these
        if (opt.exact_evidence) {
            N.fit = bayesian_decay_polish_exact(g, env, N.fit);
            N.fit.evidence = bayesian_decay_laplace_evidence_exact(g, env, N.fit.pt);
        }
        const std::vector<double> Sig = bayesian_decay_covariance(N.fit.pt, g.dim);
        bayesian_decay_distribution_with_sd(g, env, N.fit.theta, Sig, N.p, N.p_sd, &N.mean_rel, &N.mean_rel_sd);
        const auto at = std::lower_bound(G.nodes.begin(), G.nodes.end(), x,
                                         [](const BayesianDecayLambdaNode& a, double b) { return a.log10_lam < b; });
        G.nodes.insert(at, N);
    };
    //: the coarse pass: the node nearest the centre from `theta_centre`, the others warm-started from
    //: their neighbour, walking outward
    std::size_t c0 = 0;
    for (std::size_t i = 0; i < nodes.size(); ++i)
        if (std::fabs(nodes[i] - log10_centre) < std::fabs(nodes[c0] - log10_centre)) c0 = i;
    fit_into(nodes[c0], theta_centre);
    for (std::size_t i = c0 + 1; i < nodes.size(); ++i) fit_into(nodes[i], G.nodes.back().fit.theta);
    for (std::size_t i = c0; i-- > 0;) fit_into(nodes[i], G.nodes.front().fit.theta);
    G.n_coarse = G.nodes.size();

    //: the tilted evidence, which is what the weights and the refinement window are read from
    auto tilted = [&](const BayesianDecayLambdaNode& N) { return N.fit.evidence + opt.prior_slope * N.log10_lam; };
    if (opt.refine_step > 0.0 && opt.refine_window > 0.0 && G.nodes.size() > 1) {
        double best = -std::numeric_limits<double>::infinity();
        double lo = 0.0, hi = 0.0;
        bool any = false;
        for (const auto& N : G.nodes)
            if (std::isfinite(tilted(N)) && tilted(N) > best) best = tilted(N);
        for (const auto& N : G.nodes)
            if (std::isfinite(tilted(N)) && tilted(N) >= best - opt.refine_window) {
                lo = any ? std::min(lo, N.log10_lam) : N.log10_lam;
                hi = any ? std::max(hi, N.log10_lam) : N.log10_lam;
                any = true;
            }
        if (any) {
            //: half a decade beyond the kept nodes, inside the coarse grid's own range
            lo = std::max(nodes.front(), lo - 0.5);
            hi = std::min(nodes.back(), hi + 0.5);
            std::vector<double> wanted;
            for (double x = lo; x <= hi + 1e-9; x += opt.refine_step) {
                bool have = false;
                for (const auto& N : G.nodes) have = have || std::fabs(N.log10_lam - x) < 1e-9;
                if (!have) wanted.push_back(x);
            }
            if (int(wanted.size()) > opt.max_refine) wanted.resize(std::size_t(opt.max_refine));
            for (double x : wanted) {
                //: warm start from the nearest node already fitted
                std::size_t near = 0;
                for (std::size_t i = 0; i < G.nodes.size(); ++i)
                    if (std::fabs(G.nodes[i].log10_lam - x) < std::fabs(G.nodes[near].log10_lam - x)) near = i;
                fit_into(x, G.nodes[near].fit.theta);
            }
        }
    }

    //: the sweeps: a node fitted from its neighbour's end point may find a better mode than its own start did
    for (int pass = 0; pass < opt.sweeps; ++pass) {
        std::size_t improved = 0;
        for (std::size_t i = 0; i < G.nodes.size(); ++i) {
            for (int side = 0; side < 2; ++side) {
                const std::size_t j = side == 0 ? (i == 0 ? i : i - 1) : (i + 1 < G.nodes.size() ? i + 1 : i);
                if (j == i) continue;
                const double before = G.nodes[i].fit.pt.logpost;
                BayesianDecayLambdaNode trial;
                trial.log10_lam = G.nodes[i].log10_lam;
                g.fixed_values["log10_lam"] = {trial.log10_lam};
                //: the optional screen (off by default): score the neighbour's end point at THIS node's lambda
                //: first -- one evaluation against a whole fit -- and skip the trial unless it is already better.
                //: It catches the cliff case the sweep exists for, but NOT every move: measured on CBM56 it kept
                //: 9 of 18 and moved the answer (see the option's own note).
                ++G.n_screened;
                if (opt.sweep_screen && !(bayesian_decay_log_posterior(g, env, G.nodes[j].fit.theta) > before)) continue;
                ++G.n_swept_fits;
                trial.fit = bayesian_decay_fit_node(g, env, G.nodes[j].fit.theta, 1.0, opt.max_iter, opt.tol);
                if (!(trial.fit.pt.logpost > before + 1e-6)) continue;
                if (opt.exact_evidence) {
                    trial.fit = bayesian_decay_polish_exact(g, env, trial.fit);
                    trial.fit.evidence = bayesian_decay_laplace_evidence_exact(g, env, trial.fit.pt);
                }
                const std::vector<double> Sig = bayesian_decay_covariance(trial.fit.pt, g.dim);
                bayesian_decay_distribution_with_sd(g, env, trial.fit.theta, Sig, trial.p, trial.p_sd, &trial.mean_rel, &trial.mean_rel_sd);
                G.nodes[i] = trial;
                ++improved;
            }
        }
        G.n_improved += improved;
        if (improved == 0) break;
    }

    //: Tierney-Kadane where the node carries weight (the weights are the evidences, so they are known
    //: before any summary is formed): two tilted fits per node, the delta-method pair kept beside it
    if (opt.summary_method == "tk") {
        BayesianEvidenceMixture w_only;
        for (const auto& N : G.nodes) w_only.add(tilted(N), 0.0, 0.0);
        const std::vector<double> w = w_only.weights();
        for (std::size_t i = 0; i < G.nodes.size(); ++i) {
            if (!(w[i] >= opt.tk_min_weight)) continue;
            g.fixed_values["log10_lam"] = {G.nodes[i].log10_lam};
            const BayesianDecayTkSummary tk = bayesian_decay_tk_mean_rel(g, env, G.nodes[i].fit, opt.tk_eps, opt.tk_max_iter,
                                                                         opt.tk_exact_curvature);
            G.nodes[i].tk_mean = tk.mean; G.nodes[i].tk_sd = tk.sd; G.nodes[i].tk_ok = tk.ok;
            G.nodes[i].tk_flat_curvature = tk.flat_curvature;
            G.nodes[i].tk_flat_prior_share = tk.flat_prior_share;
            for (int k = 0; k < 2; ++k) {
                G.nodes[i].tk_shift[k] = tk.shift[k]; G.nodes[i].tk_dlogdet[k] = tk.dlogdet[k];
                G.nodes[i].tk_flat_ratio[k] = tk.flat_ratio[k];
                G.nodes[i].tk_flat_share[k] = tk.flat_share[k];
            }
            if (tk.ok) ++G.n_tk;
        }
    }

    BayesianEvidenceMixture scalar, vec;
    for (const auto& N : G.nodes) {
        if (!std::isfinite(N.fit.evidence)) ++G.n_dropped;
        //: a node TK did not reach (too little weight, or a tilted fit that did not converge) keeps the
        //: delta method, which is what the prototype does
        const bool use_tk = N.tk_ok && opt.summary_method == "tk";
        scalar.add(tilted(N), use_tk ? N.tk_mean : N.mean_rel,
                   use_tk ? N.tk_sd * N.tk_sd : N.mean_rel_sd * N.mean_rel_sd);
        std::vector<double> v(N.p_sd.size());
        for (std::size_t j = 0; j < v.size(); ++j) v[j] = N.p_sd[j] * N.p_sd[j];
        vec.add_vector(tilted(N), N.p, v);
    }
    G.weights = scalar.weights();
    scalar.moments(&G.mean_rel, &G.mean_rel_sd);
    vec.moments_vector(&G.p_mean, &G.p_sd);
    G.mean_rel_q16 = scalar.quantile(0.15865525393145705);
    G.mean_rel_q50 = scalar.quantile(0.5);
    G.mean_rel_q84 = scalar.quantile(0.8413447460685429);
    G.p_q16 = vec.quantile_vector(0.15865525393145705);
    G.p_q84 = vec.quantile_vector(0.8413447460685429);
    return G;
}

//! The grid with the default rules; `max_iter` and `exact_evidence` as before.
inline BayesianDecayLambdaGrid bayesian_decay_fit_lambda_grid(const BayesianDecayExperiment& f, const BayesianDecayTensors& env,
                                                             const std::vector<double>& log10_nodes, const std::vector<double>& theta_centre,
                                                             double log10_centre, int max_iter = 1000, bool exact_evidence = false) {
    BayesianDecayLambdaOptions opt;
    opt.max_iter = max_iter;
    opt.exact_evidence = exact_evidence;
    opt.refine_step = 0.0;          //!< the old behaviour: the given nodes, no refinement, no tilt, one pass
    opt.sweeps = 0;
    return bayesian_decay_fit_lambda_grid(f, env, log10_nodes, theta_centre, log10_centre, opt);
}

//! @}

IMPBFF_END_NAMESPACE

#endif  // __has_include("pocketfft/pocketfft_hdronly.h")

#endif /* IMPBFF_BAYESIANDECAYPOSTERIOR_H */
