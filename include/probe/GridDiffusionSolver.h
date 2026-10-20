/**
 *  \file IMP/bff/probe/GridDiffusionSolver.h
 *  \brief The field picture of a tethered dye: an occupancy density, propagated.
 *
 * The counterpart of #ProbeDiffusionSimulation. That one walks a particle and
 * resolves the dye's history; this one propagates the density of where the dye
 * is, which is cheaper and smoother but cannot produce a correlation function
 * because it has thrown the history away. They agree on equilibrium and on the
 * mean squared displacement.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_GRIDDIFFUSIONSOLVER_H
#define IMPBFF_GRIDDIFFUSIONSOLVER_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/probe/DiffusionSolver.h>

#include <IMP/bff/util/IMPCompatibility.h>

#include <string>
#include <utility>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Propagate an excited-state density on a masked 3-D grid.
/*!
    \param density initial occupancy, flat `ng^3`
    \param bounds non-zero inside the accessible volume
    \param diffusion_map per-voxel `D`, A^2/ns
    \param rate_map per-voxel deactivation rate, 1/ns
    \param dg voxel edge, A
    \param t_step time step, ns
    \param flux_form "smoluchowski", "ito" or "detailed_balance"
    \param check_stability refuse a step past the explicit scheme's limit
    \param weight_map equilibrium weight w, flat `ng^3`, positive in the
           domain; read by "detailed_balance" only (empty: w = 1)
*/

//! The donor decay `run()` leaves behind.
/*!
    A value carrying the time axis, the surviving excited-state fraction at
    each reported step, and the final density. Its arrays are read back as
    managed numpy views through the `get_*()` methods (ARGOUTVIEWM_ARRAY1), so
    a caller treats `result.get_time()`, `result.get_fluorescence()` and
    `result.get_density()` as ndarrays.
*/
class IMPBFFEXPORT GridDiffusionResult {
    std::vector<double> time_;
    std::vector<double> fluorescence_;
    std::vector<double> density_;

public:
    GridDiffusionResult() {}
    GridDiffusionResult(std::vector<double> time, std::vector<double> fluorescence,
                        std::vector<double> density)
        : time_(std::move(time)), fluorescence_(std::move(fluorescence)),
          density_(std::move(density)) {}

    void get_time(double** out_view, int* n_out_view) const;
    void get_fluorescence(double** out_view, int* n_out_view) const;
    void get_density(double** out_view, int* n_out_view) const;

    IMP_SHOWABLE_INLINE(
            GridDiffusionResult,
            out << "GridDiffusionResult(" << time_.size() << " points, "
                << density_.size() << " density voxels)");
};
IMP_VALUES(GridDiffusionResult, GridDiffusionResults);

//! What `gradient()` returns: dL/dD, dL/dk and dL/dp0, per voxel.
class IMPBFFEXPORT GridDiffusionGradient {
    std::vector<double> d_diffusion_;
    std::vector<double> d_rate_;
    std::vector<double> d_density_;

public:
    GridDiffusionGradient() {}
    GridDiffusionGradient(std::vector<double> d_diffusion,
                          std::vector<double> d_rate,
                          std::vector<double> d_density)
        : d_diffusion_(std::move(d_diffusion)), d_rate_(std::move(d_rate)),
          d_density_(std::move(d_density)) {}

    void get_d_diffusion(double** out_view, int* n_out_view) const;
    void get_d_rate(double** out_view, int* n_out_view) const;
    void get_d_density(double** out_view, int* n_out_view) const;

    IMP_SHOWABLE_INLINE(
            GridDiffusionGradient,
            out << "GridDiffusionGradient(" << d_diffusion_.size()
                << " voxels)");
};
IMP_VALUES(GridDiffusionGradient, GridDiffusionGradients);

//! The donor's lifetime averages over the grid ensemble.
/*!
    For an excited-state population \f$F(t)\f$ with \f$F(0) = 1\f$:
    \f$\tau_x = \int_0^\infty F\,dt\f$ (species- or amplitude-weighted, ns),
    \f$\tau_F = \int_0^\infty t F\,dt / \tau_x\f$ (fluorescence- or
    intensity-weighted, ns), and the quantum yield relative to the unquenched
    dye, \f$\Phi/\Phi_0 = \tau_x/\tau_0\f$ (the radiative rate is
    \f$1/\tau_0\f$'s share and does not depend on position).
*/
class IMPBFFEXPORT GridLifetime {
    double tau_x_, tau_f_, quantum_yield_, residual_;
    int iterations_;
    std::string method_;

public:
    GridLifetime(double tau_x = 0.0, double tau_f = 0.0,
                 double quantum_yield = 0.0, int iterations = 0,
                 double residual = 0.0, const std::string& method = "")
        : tau_x_(tau_x), tau_f_(tau_f), quantum_yield_(quantum_yield),
          residual_(residual), iterations_(iterations), method_(method) {}
    //! \f$\int F dt\f$, ns
    double get_tau_x() const { return tau_x_; }
    //! \f$\int t F dt / \int F dt\f$, ns
    double get_tau_f() const { return tau_f_; }
    //! \f$\Phi/\Phi_0 = \tau_x / \tau_0\f$
    double get_quantum_yield() const { return quantum_yield_; }
    //! conjugate-gradient iterations ("dynamic" only)
    int get_iterations() const { return iterations_; }
    //! the solves' final relative residual ("dynamic" only)
    double get_residual() const { return residual_; }
    //! "static", "fast_exchange" or "dynamic"
    std::string get_method() const { return method_; }

    IMP_SHOWABLE_INLINE(GridLifetime,
                        out << "GridLifetime(" << method_ << ": tau_x = "
                            << tau_x_ << " ns, tau_f = " << tau_f_
                            << " ns, QY/QY0 = " << quantum_yield_ << ")");
};
IMP_VALUES(GridLifetime, GridLifetimes);

class IMPBFFEXPORT GridDiffusionSolver {
    std::vector<double> density_;
    std::vector<double> bounds_;
    std::vector<double> diffusion_map_;
    std::vector<double> rate_map_;
    std::vector<double> weight_map_;
    double dg_;
    double t_step_;
    int ng_;
    int flux_form_;
    bool check_stability_;
    int n_iterations_;

    //! \throws IMP::ValueException on a domain touching the outer shell, or a
    //!         time step past `dg^2 / (6 D_max)` -- for "detailed_balance"
    //!         `dg^2 / max_c sum_m D_cm sqrt(w_m / w_c)`, the step at which the
    //!         self coefficient of the sweep reaches zero. The rate does **not** enter
    //!         that bound: it is carried as `exp(-k dt)`, the exact solution
    //!         over the step, and so constrains nothing.
    void validate() const;
    //! `p b / sum(p b)` -- the normalised start both run() and gradient() use.
    //! An empty or all-zero \p p starts from get_stationary_occupancy().
    std::vector<double> start_density(const std::vector<double>& p) const;
    //! The weight field the kernels read: the caller's, for detailed balance.
    const std::vector<double>& kernel_weight() const;
    //! \f$D\,dt/dg^2\f$ and \f$e^{-k\,dt}\f$, the folded coefficients.
    std::vector<double> folded_d() const;
    std::vector<double> folded_decay(double extra_rate = 0.0) const;
    std::vector<double> stationary() const;

public:
    GridDiffusionSolver(
            const std::vector<double>& diffusion_map = std::vector<double>(),
            const std::vector<double>& bounds = std::vector<double>(),
            const std::vector<double>& density = std::vector<double>(),
            const std::vector<double>& rate_map = std::vector<double>(),
            double dg = 1.0, double t_step = 0.01,
            const std::string& flux_form = "smoluchowski",
            bool check_stability = true,
            const std::vector<double>& weight_map = std::vector<double>());

    //! Integrate \p n_steps steps, reporting every \p n_out.
    /*!
        \return a #GridDiffusionResult: the time axis in ns, the surviving
                excited-state fraction at each reported step, and the final
                density -- the same three arrays `run()` used to hand back as
                one flat buffer for Python to split.
    */
    GridDiffusionResult run(int n_steps, int n_out = 1);

    //! Adjoint of run(): the gradient of a loss on the decay, every voxel at once.
    /*!
        For \f$L = \sum_k \bar F_k F(t_k)\f$, returns dL/dD, dL/dk and dL/dp0
        for about four times the cost of one run(). Finite differences need one
        run *per parameter*.

        The gradient is of the *discrete* scheme actually run, so it agrees with
        a central difference of run() to roundoff -- which is also why the time
        step must not depend on the parameters being differentiated.

        \param[in] dL_dF sensitivity of the loss to each reported population
        \param[in] density the start of the run being differentiated; run()
                   replaces the solver's own with the final one, so a completed
                   run has to be handed the density it began with
        \return a #GridDiffusionGradient of dL/dD, dL/dk and dL/dp0, per voxel
    */
    GridDiffusionGradient gradient(const std::vector<double>& dL_dF,
                                   int n_steps, int n_out,
                                   const std::vector<double>& density);

    //! The same propagation as one matrix exponential (Lanczos), no stepping.
    /*!
        The semi-discrete solution \f$p(t) = e^{-Lt}p_0\f$ at the times run()
        would report, `n_steps / n_out + 1` of them, in \p krylov_dim operator
        applications instead of \p n_steps sweeps, and with no stability limit
        on `t_step` (which only sets the report grid). It agrees with run() to
        the stepping scheme's own error. Advances get_n_iterations() by nothing
        and leaves the density alone.
    */
    GridDiffusionResult run_krylov(int n_steps, int n_out = 1,
                                   int krylov_dim = 128) const;

    //! The donor decay \f$I(t)/I(0)\f$ with the radiative lifetime included.
    /*!
        \f$I(t) = e^{-t/\tau_0} \sum_c p_c(t)\f$, the sum propagated with the
        rate map as the sink, by Lanczos (run_krylov()) -- so `t_step` is only
        the report spacing. The start is the solver's density, or the
        stationary occupancy when that is empty: an ensemble excited out of
        its equilibrium.
        \param[in] tau0 unquenched donor lifetime, ns
        \param[in] n_steps,n_out report every `n_out t_step` up to
                   `n_steps t_step`, ns
    */
    GridDiffusionResult donor_decay(double tau0, int n_steps, int n_out = 1,
                                    int krylov_dim = 128) const;

    //! Lifetimes with no transport: each voxel decays on its own.
    /*! \f$\tau_x = \sum_c p_c \tau_c\f$, \f$\tau_c = 1/(1/\tau_0 + k_c)\f$,
        over the start density -- the \f$D \to 0\f$ limit. */
    GridLifetime lifetime_static(double tau0) const;

    //! Lifetimes with instant transport: one rate, the equilibrium mean.
    /*! \f$\tau_x = \tau_F = 1/(1/\tau_0 + \langle k\rangle_{eq})\f$ over the
        stationary occupancy, whatever the start -- the \f$D \to \infty\f$
        limit. */
    GridLifetime lifetime_fast_exchange(double tau0) const;

    //! Lifetimes with the transport resolved, by two linear solves.
    /*!
        \f$\tau_x = \mathbf 1^T M^{-1} p_0\f$ and
        \f$\tau_F = \mathbf 1^T M^{-2} p_0 / \tau_x\f$ with
        \f$M = L + K + 1/\tau_0\f$ the generator (\f$dp/dt = -Mp\f$), solved by
        preconditioned conjugate gradients in the variables that make it
        symmetric. Exact for the semi-discrete equation, no time step, no
        truncation of the decay. The start is as for donor_decay().
        \param[in] tolerance relative residual of each solve
        \param[in] max_iterations per solve; <= 0 takes 20000
    */
    GridLifetime lifetime(double tau0, double tolerance = 1e-10,
                          int max_iterations = 0) const;

    //! The closed-form stationary state with no sink, normalised.
    /*! \f$w b\f$ for "detailed_balance", \f$b\f$ for "smoluchowski",
        \f$b/D\f$ for "ito". */
    void get_stationary_occupancy(double** out_view, int* n_out_view) const;

    //! Propagate with no decay until the occupancy stops moving.
    /*! Prefer equilibrium_occupancy(), which is the same answer in closed form.
        This iterates toward it, and on a real site -- where `D` spans orders of
        magnitude through the compounding slow factor -- it can fail to converge
        in any practical number of steps. */
    void equilibrium(int n_steps, double tolerance, int n_check = 100,
                     double** out_view = 0, int* n_out_view = 0);

    void get_density(double** out_view, int* n_out_view) const;
    void set_density(const std::vector<double>& d) { density_ = d; }
    void get_bounds(double** out_view, int* n_out_view) const;
    void get_diffusion_map(double** out_view, int* n_out_view) const;
    void set_diffusion_map(const std::vector<double>& d) { diffusion_map_ = d; }
    void get_rate_map(double** out_view, int* n_out_view) const;
    void set_rate_map(const std::vector<double>& k) { rate_map_ = k; }
    void get_weight_map(double** out_view, int* n_out_view) const;
    void set_weight_map(const std::vector<double>& w) { weight_map_ = w; }

    int get_ng() const { return ng_; }
    double get_dg() const { return dg_; }
    double get_t_step() const { return t_step_; }
    int get_n_iterations() const { return n_iterations_; }
    std::string get_flux_form() const {
        return flux_form_ == FLUX_ITO ? "ito"
             : flux_form_ == FLUX_DETAILED_BALANCE ? "detailed_balance"
             : "smoluchowski";
    }

    IMP_SHOWABLE_INLINE(GridDiffusionSolver,
                        out << "GridDiffusionSolver(ng = " << ng_
                            << ", dg = " << dg_ << " A, dt = " << t_step_
                            << " ns, " << n_iterations_ << " steps taken)");
};
IMP_VALUES(GridDiffusionSolver, GridDiffusionSolvers);

IMPBFF_END_NAMESPACE

#endif //IMPBFF_GRIDDIFFUSIONSOLVER_H
