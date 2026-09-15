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
 *   `fixed_values["log10_lam"]`, and a weak Gaussian on the linear tilt). Analytic
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
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
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
    const double lam = std::pow(10.0, f.fixed_values.at("log10_lam")[0]);
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

//! @}

IMPBFF_END_NAMESPACE

#endif  // __has_include("pocketfft/pocketfft_hdronly.h")

#endif /* IMPBFF_BAYESIANDECAYPOSTERIOR_H */
