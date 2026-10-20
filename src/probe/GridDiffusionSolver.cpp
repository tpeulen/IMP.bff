/**
 * \file GridDiffusionSolver.cpp
 * \brief The field picture of a tethered dye: an occupancy density, propagated.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/probe/GridDiffusionSolver.h>
#include <IMP/bff/internal/GridShape.h>
#include <IMP/bff/internal/OutputView.h>
#include <IMP/bff/util/IMPCompatibility.h>

#include <algorithm>
#include <string>
#include <utility>
#include <cmath>
#include <cstdlib>
#include <limits>

IMPBFF_BEGIN_NAMESPACE

GridDiffusionSolver::GridDiffusionSolver(
        const std::vector<double>& diffusion_map,
        const std::vector<double>& bounds, const std::vector<double>& density,
        const std::vector<double>& rate_map, double dg, double t_step,
        const std::string& flux_form, bool check_stability,
        const std::vector<double>& weight_map)
        : density_(density), bounds_(bounds),
          diffusion_map_(diffusion_map), rate_map_(rate_map),
          weight_map_(weight_map), dg_(dg),
          t_step_(t_step),
          ng_(internal::cube_side(bounds.size())),
          check_stability_(check_stability), n_iterations_(0) {
    if (flux_form == "smoluchowski") {
        flux_form_ = FLUX_SMOLUCHOWSKI;
    } else if (flux_form == "ito") {
        flux_form_ = FLUX_ITO;
    } else if (flux_form == "detailed_balance") {
        flux_form_ = FLUX_DETAILED_BALANCE;
    } else {
        IMP_THROW("flux_form must be 'smoluchowski', 'ito' or "
                  "'detailed_balance', not '" << flux_form << "'",
                  IMP::ValueException);
    }
    if (!bounds_.empty() && ng_ == 0) {
        IMP_THROW("the bounds grid must be cubic; " << bounds_.size()
                          << " values are not a whole cube",
                  IMP::ValueException);
    }
    // The mask is used as a multiplier everywhere, so it is stored as one:
    // a caller may hand in a count, a probability or a bool array, and only
    // "inside or not" is meant.
    for (std::size_t i = 0; i < bounds_.size(); ++i) {
        bounds_[i] = bounds_[i] > 0.0 ? 1.0 : 0.0;
    }
    if (rate_map_.empty()) rate_map_.assign(bounds_.size(), 0.0);
}

const std::vector<double>& GridDiffusionSolver::kernel_weight() const {
    static const std::vector<double> none;
    return flux_form_ == FLUX_DETAILED_BALANCE ? weight_map_ : none;
}

std::vector<double> GridDiffusionSolver::folded_d() const {
    std::vector<double> d(diffusion_map_.size());
    const double scale = t_step_ / (dg_ * dg_);
    for (std::size_t i = 0; i < d.size(); ++i) d[i] = diffusion_map_[i] * scale;
    return d;
}

std::vector<double> GridDiffusionSolver::folded_decay(double extra_rate) const {
    std::vector<double> decay(rate_map_.size());
    for (std::size_t i = 0; i < decay.size(); ++i) {
        decay[i] = std::exp(-(rate_map_[i] + extra_rate) * t_step_);
    }
    return decay;
}

std::vector<double> GridDiffusionSolver::stationary() const {
    const std::size_t nv = bounds_.size();
    std::vector<double> p(nv, 0.0);
    double total = 0.0;
    for (std::size_t i = 0; i < nv; ++i) {
        if (bounds_[i] == 0.0) continue;
        if (flux_form_ == FLUX_DETAILED_BALANCE) {
            p[i] = weight_map_.empty() ? 1.0 : weight_map_[i];
        } else if (flux_form_ == FLUX_ITO) {
            p[i] = diffusion_map_[i] > 0.0 ? 1.0 / diffusion_map_[i] : 0.0;
        } else {
            p[i] = 1.0;
        }
        total += p[i];
    }
    if (total > 0.0) for (double& v : p) v /= total;
    return p;
}

void GridDiffusionSolver::get_stationary_occupancy(double** out_view,
                                                   int* n_out_view) const {
    internal::copy_to_view(stationary(), out_view, n_out_view);
}

void GridDiffusionSolver::get_weight_map(double** o, int* n) const {
    internal::copy_to_view(weight_map_, o, n);
}

void GridDiffusionSolver::validate() const {
    // The 7-point stencil cannot be evaluated on the outer shell, so a domain
    // touching it would lose population there with no warning.
    const std::size_t n = static_cast<std::size_t>(ng_);
    for (int ix = 0; ix < ng_; ++ix) {
        for (int iy = 0; iy < ng_; ++iy) {
            for (int iz = 0; iz < ng_; ++iz) {
                const bool edge = ix == 0 || iy == 0 || iz == 0 ||
                                  ix == ng_ - 1 || iy == ng_ - 1 || iz == ng_ - 1;
                if (!edge) continue;
                if (bounds_[(static_cast<std::size_t>(ix) * n + iy) * n + iz] != 0.0) {
                    IMP_THROW(
                            "The domain reaches the outer shell of the grid, "
                            "where the 7-point stencil cannot be evaluated and "
                            "population is discarded. Enlarge the grid so the "
                            "accessible volume is surrounded by at least one "
                            "empty voxel.",
                            IMP::ValueException);
                }
            }
        }
    }
    const std::size_t nv = bounds_.size();
    if (diffusion_map_.size() != nv || rate_map_.size() != nv) {
        IMP_THROW("the diffusion and rate maps must have the bounds' "
                  << nv << " voxels, not " << diffusion_map_.size() << " and "
                  << rate_map_.size(), IMP::ValueException);
    }
    if (flux_form_ == FLUX_DETAILED_BALANCE && !weight_map_.empty()) {
        if (weight_map_.size() != nv) {
            IMP_THROW("the weight map must have the bounds' " << nv
                      << " voxels, not " << weight_map_.size(),
                      IMP::ValueException);
        }
        for (std::size_t i = 0; i < nv; ++i) {
            if (bounds_[i] != 0.0 && !(weight_map_[i] > 0.0 && std::isfinite(weight_map_[i]))) {
                IMP_THROW("the equilibrium weight must be positive and finite "
                          "inside the domain; voxel " << i << " has "
                          << weight_map_[i], IMP::ValueException);
            }
        }
    }
    if (!check_stability_) return;
    if (flux_form_ == FLUX_DETAILED_BALANCE && !weight_map_.empty()) {
        // The self coefficient of the sweep is 1 - dt/dg^2 sum_m D_cm
        // sqrt(w_m/w_c): a voxel whose neighbours weigh more empties faster,
        // so the bound is per voxel, not 6 D_max.
        const long ngl = ng_;
        const long stride[6] = {-ngl * ngl, ngl * ngl, -ngl, ngl, -1, 1};
        double worst = 0.0;
        for (std::size_t c = 0; c < nv; ++c) {
            if (bounds_[c] == 0.0) continue;
            double out = 0.0;
            for (int q = 0; q < 6; ++q) {
                const long m = static_cast<long>(c) + stride[q];
                if (m < 0 || m >= static_cast<long>(nv) || bounds_[m] == 0.0) continue;
                out += 0.5 * (diffusion_map_[c] + diffusion_map_[m]) *
                       std::sqrt(weight_map_[m] / weight_map_[c]);
            }
            worst = std::max(worst, out);
        }
        const double limit = worst > 0.0 ? dg_ * dg_ / worst
                                         : std::numeric_limits<double>::infinity();
        if (t_step_ > limit) {
            IMP_THROW("t_step " << t_step_ << " ns exceeds the detailed-balance "
                      << "stability limit dg^2 / max_c sum_m D_cm sqrt(w_m/w_c) = "
                      << limit << " ns. The scheme would diverge rather than "
                      << "lose accuracy; take a smaller step or run_krylov().",
                      IMP::ValueException);
        }
        return;
    }
    double d_max = 0.0;
    for (std::size_t i = 0; i < diffusion_map_.size(); ++i) {
        d_max = std::max(d_max, diffusion_map_[i]);
    }
    // **Without k_max, deliberately.** The rate is carried as exp(-k dt),
    // applied as a factor -- the exact solution of dp/dt = -k p over the step
    // -- so it contributes no stability constraint at all here. It is the
    // `1 - k dt` form that diverges once k dt > 1, and this scheme does not
    // use it. Passing k_max would reject steps that are perfectly stable:
    // on a site with k_max = 96 1/ns it cuts the allowed step by a third.
    const double limit = diffusion_stability_limit(d_max, dg_);
    if (t_step_ > limit) {
        IMP_THROW("t_step " << t_step_ << " ns exceeds the explicit-scheme "
                            << "stability limit dg^2 / (6 D_max) = "
                            << limit << " ns for dg = " << dg_
                            << " A and D_max = " << d_max
                            << " A^2/ns. The scheme would diverge rather "
                            << "than lose accuracy.",
                  IMP::ValueException);
    }
}

std::vector<double> GridDiffusionSolver::start_density(
        const std::vector<double>& p) const {
    std::vector<double> cur(bounds_.size(), 0.0);
    double total = 0.0;
    for (std::size_t i = 0; i < p.size() && i < cur.size(); ++i) {
        cur[i] = p[i] * bounds_[i];
        total += cur[i];
    }
    // Nothing to start from: an ensemble excited out of its equilibrium.
    if (!(total > 0.0)) return stationary();
    for (std::size_t i = 0; i < cur.size(); ++i) cur[i] /= total;
    return cur;
}

GridDiffusionResult GridDiffusionSolver::run(int n_steps, int n_out) {
    validate();
    if (n_out < 1) n_out = 1;
    const std::vector<double> cur = start_density(density_);

    // Fold dt into the coefficients so the inner loop is pure arithmetic, and
    // carry the rate as exp(-k dt) -- the exact solution of dp/dt = -k p over
    // the step, so it contributes no stability constraint of its own.
    std::vector<double> fluorescence, final;
    internal::lattice_propagate_any(cur, folded_d(), folded_decay(), bounds_,
                                    kernel_weight(), ng_, flux_form_, n_steps,
                                    n_out, fluorescence, final);
    n_iterations_ += n_steps;

    const int n_reports = n_steps / n_out + 1;
    const std::size_t n_fluo =
            std::min<std::size_t>(fluorescence.size(),
                                  static_cast<std::size_t>(std::max(0, n_reports)));

    // The time axis is the caller's report grid, in ns.
    std::vector<double> time;
    time.reserve(n_fluo);
    for (std::size_t i = 0; i < n_fluo; ++i) {
        time.push_back(static_cast<double>(i) * t_step_ * n_out);
    }
    density_ = final;
    fluorescence.resize(n_fluo);
    return GridDiffusionResult(std::move(time), std::move(fluorescence),
                               std::move(final));
}

GridDiffusionResult GridDiffusionSolver::run_krylov(int n_steps, int n_out,
                                                    int krylov_dim) const {
    // The shell and the weights are checked; the step is not -- there is none.
    GridDiffusionSolver probe(*this);
    probe.check_stability_ = false;
    probe.validate();
    if (n_out < 1) n_out = 1;
    std::vector<double> fluorescence, final;
    internal::lattice_krylov(start_density(density_), folded_d(), folded_decay(),
                             bounds_, kernel_weight(), ng_, flux_form_, n_steps,
                             n_out, krylov_dim, fluorescence, &final);
    std::vector<double> time(fluorescence.size());
    for (std::size_t i = 0; i < time.size(); ++i) {
        time[i] = static_cast<double>(i) * t_step_ * n_out;
    }
    return GridDiffusionResult(std::move(time), std::move(fluorescence),
                               std::move(final));
}

GridDiffusionResult GridDiffusionSolver::donor_decay(double tau0, int n_steps,
                                                     int n_out,
                                                     int krylov_dim) const {
    if (!(tau0 > 0.0)) {
        IMP_THROW("tau0 must be positive, not " << tau0, IMP::ValueException);
    }
    GridDiffusionSolver probe(*this);
    probe.check_stability_ = false;
    probe.validate();
    if (n_out < 1) n_out = 1;
    // 1/tau0 is uniform, so it commutes with transport: propagate the sink
    // alone and multiply the trace by exp(-t/tau0), rather than put a rate
    // into every voxel that the Lanczos basis has to resolve.
    std::vector<double> fluorescence, final;
    internal::lattice_krylov(start_density(density_), folded_d(), folded_decay(),
                             bounds_, kernel_weight(), ng_, flux_form_, n_steps,
                             n_out, krylov_dim, fluorescence, &final);
    std::vector<double> time(fluorescence.size());
    for (std::size_t i = 0; i < time.size(); ++i) {
        time[i] = static_cast<double>(i) * t_step_ * n_out;
        fluorescence[i] *= std::exp(-time[i] / tau0);
    }
    const double radiative = std::exp(-static_cast<double>(n_steps) * t_step_ / tau0);
    for (double& v : final) v *= radiative;
    return GridDiffusionResult(std::move(time), std::move(fluorescence),
                               std::move(final));
}

GridLifetime GridDiffusionSolver::lifetime_static(double tau0) const {
    if (!(tau0 > 0.0)) {
        IMP_THROW("tau0 must be positive, not " << tau0, IMP::ValueException);
    }
    const std::vector<double> p = start_density(density_);
    double tx = 0.0, t2 = 0.0;
    for (std::size_t c = 0; c < p.size(); ++c) {
        if (p[c] == 0.0) continue;
        const double tau = 1.0 / (1.0 / tau0 + rate_map_[c]);
        tx += p[c] * tau;
        t2 += p[c] * tau * tau;
    }
    return GridLifetime(tx, tx > 0.0 ? t2 / tx : 0.0, tx / tau0, 0, 0.0,
                        "static");
}

GridLifetime GridDiffusionSolver::lifetime_fast_exchange(double tau0) const {
    if (!(tau0 > 0.0)) {
        IMP_THROW("tau0 must be positive, not " << tau0, IMP::ValueException);
    }
    const std::vector<double> p = stationary();
    double k = 0.0;
    for (std::size_t c = 0; c < p.size(); ++c) k += p[c] * rate_map_[c];
    const double tau = 1.0 / (1.0 / tau0 + k);
    return GridLifetime(tau, tau, tau / tau0, 0, 0.0, "fast_exchange");
}

GridLifetime GridDiffusionSolver::lifetime(double tau0, double tolerance,
                                           int max_iterations) const {
    if (!(tau0 > 0.0)) {
        IMP_THROW("tau0 must be positive, not " << tau0, IMP::ValueException);
    }
    GridDiffusionSolver probe(*this);
    probe.check_stability_ = false;
    probe.validate();
    // In ns rather than steps: d = D/dg^2 (1/ns) and the rate k + 1/tau0
    // (1/ns), so the moments come out in ns and ns^2 and t_step plays no part.
    std::vector<double> d(diffusion_map_.size()), rate(rate_map_.size());
    for (std::size_t i = 0; i < d.size(); ++i) d[i] = diffusion_map_[i] / (dg_ * dg_);
    for (std::size_t i = 0; i < rate.size(); ++i) rate[i] = rate_map_[i] + 1.0 / tau0;
    const internal::LatticeMoments m = internal::lattice_moment_solve(
            start_density(density_), d, rate, bounds_, kernel_weight(), ng_,
            flux_form_, tolerance, max_iterations);
    return GridLifetime(m.zeroth, m.zeroth > 0.0 ? m.first / m.zeroth : 0.0,
                        m.zeroth / tau0, m.iterations, m.residual, "dynamic");
}

GridDiffusionGradient GridDiffusionSolver::gradient(
        const std::vector<double>& dL_dF, int n_steps, int n_out,
        const std::vector<double>& density) {
    validate();
    if (n_out < 1) n_out = 1;
    const int n_reports = n_steps / n_out + 1;
    if (static_cast<int>(dL_dF.size()) != n_reports) {
        IMP_THROW("dL_dF has " << dL_dF.size() << " entries; run(n_steps="
                               << n_steps << ", n_out=" << n_out
                               << ") reports " << n_reports << " points",
                  IMP::ValueException);
    }
    const std::vector<double>& p = density.empty() ? density_ : density;
    double total = 0.0;
    for (std::size_t i = 0; i < p.size() && i < bounds_.size(); ++i) {
        total += p[i] * bounds_[i];
    }
    const std::vector<double> cur = start_density(p);
    const std::vector<double> d = folded_d();
    const std::vector<double> decay = folded_decay();
    const double scale = t_step_ / (dg_ * dg_);

    std::vector<double> g_d, g_decay, g_cur;
    internal::lattice_propagate_adjoint_any(cur, d, decay, bounds_,
                                            kernel_weight(), ng_, flux_form_,
                                            n_steps, n_out, dL_dF, g_d, g_decay,
                                            g_cur);
    const std::size_t nv = bounds_.size();
    std::vector<double> gd(nv, 0.0), gk(nv, 0.0), gp(nv, 0.0);
    // The chain rule through the folding run() applies:
    //   d = D dt/dg^2      -> dL/dD  = dL/dd * dt/dg^2
    //   decay = exp(-k dt) -> dL/dk  = dL/ddecay * (-dt) * decay
    //   cur = p b / sum(p b) -> dL/dp = b (g - <g, cur>) / total
    // (a start that fell back to the stationary occupancy has no dL/dp).
    double dot = 0.0;
    for (std::size_t i = 0; i < nv; ++i) dot += g_cur[i] * cur[i];
    for (std::size_t i = 0; i < nv; ++i) {
        gd[i] = g_d[i] * scale;
        gk[i] = g_decay[i] * (-t_step_) * decay[i];
        gp[i] = total > 0.0 ? bounds_[i] * (g_cur[i] - dot) / total : 0.0;
    }
    return GridDiffusionGradient(std::move(gd), std::move(gk), std::move(gp));
}

void GridDiffusionSolver::equilibrium(int n_steps, double tolerance,
                                      int n_check, double** out_view,
                                      int* n_out_view) {
    validate();
    std::vector<double> cur = start_density(density_);
    const std::vector<double> d = folded_d();
    const std::vector<double> one(cur.size(), 1.0);   // exp(-0 dt): no decay

    std::vector<double> previous = cur;
    int remaining = n_steps;
    const int chunk = std::max(1, n_check);
    while (remaining > 0) {
        const int take = std::min(chunk, remaining);
        std::vector<double> fluo, next;
        internal::lattice_propagate_any(cur, d, one, bounds_, kernel_weight(),
                                        ng_, flux_form_, take, take, fluo, next);
        if (next.empty()) break;
        cur.swap(next);
        n_iterations_ += take;
        remaining -= take;
        double drift = 0.0;
        for (std::size_t i = 0; i < cur.size(); ++i) {
            drift += std::fabs(cur[i] - previous[i]);
        }
        if (drift < tolerance) break;
        previous = cur;
    }
    double total = 0.0;
    for (std::size_t i = 0; i < cur.size(); ++i) total += cur[i];
    if (total > 0.0) {
        for (std::size_t i = 0; i < cur.size(); ++i) cur[i] /= total;
    }
    density_ = cur;
    internal::copy_to_view(cur, out_view, n_out_view);
}

void GridDiffusionSolver::get_density(double** o, int* n) const {
    internal::copy_to_view(density_, o, n);
}
void GridDiffusionSolver::get_bounds(double** o, int* n) const {
    internal::copy_to_view(bounds_, o, n);
}
void GridDiffusionSolver::get_diffusion_map(double** o, int* n) const {
    internal::copy_to_view(diffusion_map_, o, n);
}
void GridDiffusionSolver::get_rate_map(double** o, int* n) const {
    internal::copy_to_view(rate_map_, o, n);
}

void GridDiffusionResult::get_time(double** out_view, int* n_out_view) const {
    internal::copy_to_view(time_, out_view, n_out_view);
}
void GridDiffusionResult::get_fluorescence(double** out_view, int* n_out_view) const {
    internal::copy_to_view(fluorescence_, out_view, n_out_view);
}
void GridDiffusionResult::get_density(double** out_view, int* n_out_view) const {
    internal::copy_to_view(density_, out_view, n_out_view);
}

void GridDiffusionGradient::get_d_diffusion(double** out_view, int* n_out_view) const {
    internal::copy_to_view(d_diffusion_, out_view, n_out_view);
}
void GridDiffusionGradient::get_d_rate(double** out_view, int* n_out_view) const {
    internal::copy_to_view(d_rate_, out_view, n_out_view);
}
void GridDiffusionGradient::get_d_density(double** out_view, int* n_out_view) const {
    internal::copy_to_view(d_density_, out_view, n_out_view);
}

IMPBFF_END_NAMESPACE
