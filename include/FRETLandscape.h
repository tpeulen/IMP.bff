/**
 *  \file IMP/bff/FRETLandscape.h
 *  \brief The photon-by-photon likelihood of a continuous free-energy
 *         landscape and diffusion coefficient from single-molecule FRET.
 *
 * Dingeldein & Covino, "A differentiable photon-by-photon likelihood for
 * continuous free-energy landscapes and diffusion coefficients from
 * single-molecule FRET", arXiv:2608.21061.
 *
 * The donor-acceptor distance `x(t)` diffuses on `u(x)` (kT) with diffusion
 * coefficient `D`; a photon in channel `c` is detected at rate
 *
 *     lambda_c(x) = a_c (d_c (1 - E(x)) + e_c E(x)) + beta_c,
 *
 * `E(x) = 1/(1 + (x/R0)^6)`, where `d_c` and `e_c` are the fixed fractions of
 * donor and acceptor emission that land in channel `c` (crosstalk and
 * detection branching; the paper's `C_{g->c}`, `C_{r->c}`), `a_c` is the
 * channel's brightness (`eta_c k_em`) and `beta_c` its background. The
 * likelihood of a trace, with the dark gaps before the first and after the
 * last photon dropped (paper Eq. 14), is
 *
 *     L = < 1, Lambda_{c_N} S(tau_N) ... S(tau_2) Lambda_{c_1} pi >,
 *     S(tau) = exp((Q - Lambda_tot) tau),
 *
 * with `Q` the SqRA generator (FRETLandscapeGrid.h) and `pi` its Boltzmann
 * state. It is evaluated through one eigen-decomposition of the symmetrised
 * killed generator `A = Psi diag(nu) Psi^T` per parameter vector, as a
 * normalised forward filter; traces are processed in batches so each step is
 * a matrix-matrix product.
 *
 * **Gradient.** Exact, by a backward (adjoint) sweep: the derivative of
 * `exp(A tau)` along `dA` is `Psi (G o Psi^T dA Psi) Psi^T` with
 * `G_kl = (e^{nu_k tau} - e^{nu_l tau})/(nu_k - nu_l)` (Daleckii-Krein), so
 * all photons of all traces reduce to one `M x M` matrix in the eigenbasis,
 * transformed back once. Cost: about 2.5x a likelihood evaluation.
 *
 * **Parameters** `theta` (length `K + 1 + 2C`, see pack_parameters()):
 * the `K` spline knot heights `mu` of `u(x)` (kT), `log D`, `log a_c` for
 * each channel, then `log beta_c` for each channel.
 *
 * Units are the caller's: with `x` in nm and times in ms, `D` is nm^2/ms and
 * the rates are per ms.
 *
 * \authors Thomas-Otavio Peulen
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_FRETLANDSCAPE_H
#define IMPBFF_FRETLANDSCAPE_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/FRETLandscapeGrid.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! How FRETLandscapeModel::fit() runs its L-BFGS.
/*! Every field is public and plain. Defaults follow the paper's Table II. */
class IMPBFFEXPORT FRETLandscapeFitOptions {
 public:
  //! Maximum L-BFGS iterations over the whole fit.
  int max_iterations = 500;
  //! Stop once the log posterior improved by less than `min_delta` (nats)
  //! over the last `patience` iterations.
  int patience = 50;
  double min_delta = 0.1;
  //! Indices of theta held at their start values.
  std::vector<int> fixed;

  IMP_SHOWABLE_INLINE(FRETLandscapeFitOptions,
                      out << "FRETLandscapeFitOptions(max_iterations " << max_iterations
                          << ", patience " << patience << ")");
};
IMP_VALUES(FRETLandscapeFitOptions, FRETLandscapeFitOptionsList);

//! What FRETLandscapeModel::fit() found.
class IMPBFFEXPORT FRETLandscapeFit {
 public:
  FRETLandscapeFit() {}
  //! The MAP parameters.
  const std::vector<double>& get_theta() const { return theta_; }
  double get_log_posterior() const { return log_posterior_; }
  double get_log_likelihood() const { return log_likelihood_; }
  int get_n_iterations() const { return n_iterations_; }
  //! Log posterior after each block of `patience` iterations (start first).
  const std::vector<double>& get_history() const { return history_; }
  //! `patience`: stopped on the improvement criterion; `converged`: the
  //! optimiser's own test (gradient, step or function change);
  //! `max_iterations`; or `failed`.
  const std::string& get_status() const { return status_; }

  IMP_SHOWABLE_INLINE(FRETLandscapeFit,
                      out << "FRETLandscapeFit(log_posterior " << log_posterior_
                          << ", iterations " << n_iterations_ << ", " << status_ << ")");

 private:
  friend class FRETLandscapeModel;
  std::vector<double> theta_, history_;
  double log_posterior_ = 0.0, log_likelihood_ = 0.0;
  int n_iterations_ = 0;
  std::string status_;
};
IMP_VALUES(FRETLandscapeFit, FRETLandscapeFits);

//! A free-energy landscape over a distance coordinate, scored photon by photon.
/*! Holds the grid, the spline, the fixed photophysics (Forster radius and
    channel fractions), the priors, and the photon traces it is scored on.
    The photons are handed over once (set_photons()) and stay in C++. */
class IMPBFFEXPORT FRETLandscapeModel {
 public:
  //! \param x_min, x_max the grid (and spline knot) range of the distance
  //! \param n_grid grid points M of the SqRA discretisation
  //! \param n_knots spline knots K
  //! \param forster_radius R0, in the units of x
  //! \param donor_fraction fraction of donor emission detected in each
  //!        channel; empty for the paper's `{0.97, 0.03}` (green, red)
  //! \param acceptor_fraction fraction of acceptor emission per channel;
  //!        empty for the paper's `{0.08, 0.92}`
  FRETLandscapeModel(double x_min = 3.75, double x_max = 8.75, int n_grid = 200,
                     int n_knots = 25, double forster_radius = 6.0,
                     const std::vector<double>& donor_fraction = std::vector<double>(),
                     const std::vector<double>& acceptor_fraction = std::vector<double>());

  // --- geometry --------------------------------------------------------------
  int get_n_grid() const { return m_; }
  int get_n_knots() const { return k_; }
  int get_n_channels() const { return c_; }
  //! `K + 1 + 2C`.
  int get_n_parameters() const { return k_ + 1 + 2 * c_; }
  //! Index of `log D` in theta; `log a_c` follow, then `log beta_c`.
  int get_diffusion_index() const { return k_; }
  const std::vector<double>& get_grid() const { return x_; }
  double get_grid_spacing() const { return h_; }
  std::vector<double> get_knots() const { return spline_.get_knots(); }
  const NaturalCubicSpline& get_spline() const { return spline_; }
  //! Spline basis on the grid, row-major `M x K`.
  const std::vector<double>& get_basis() const { return phi_; }
  //! `E(x_i)` on the grid.
  const std::vector<double>& get_fret_efficiency() const { return eff_; }
  double get_forster_radius() const { return r0_; }
  const std::vector<double>& get_donor_fraction() const { return dfrac_; }
  const std::vector<double>& get_acceptor_fraction() const { return afrac_; }

  // --- parameters ------------------------------------------------------------
  //! theta from knot heights, `D`, brightnesses `a_c` and backgrounds `beta_c`.
  std::vector<double> pack_parameters(const std::vector<double>& knot_heights,
                                      double diffusion,
                                      const std::vector<double>& amplitudes,
                                      const std::vector<double>& backgrounds) const;
  //! `u(x_i)` on the grid for theta.
  std::vector<double> get_landscape(const std::vector<double>& theta) const;
  double get_diffusion(const std::vector<double>& theta) const;
  std::vector<double> get_amplitudes(const std::vector<double>& theta) const;
  std::vector<double> get_backgrounds(const std::vector<double>& theta) const;
  //! Detection rates on the grid, row-major `C x M`: `lambda_c(x_i)`.
  std::vector<double> get_rates(const std::vector<double>& theta) const;

  // --- data ------------------------------------------------------------------
  //! The photon traces as plain arrays.
  /*! \param times arrival times, non-decreasing within each trace
      \param channels channel index of each photon, `0 <= c < C`
      \param offsets trace boundaries, CSR: trace `m` is photons
             `offsets[m] .. offsets[m+1]-1`; length `n_traces + 1` */
  void set_photons(const std::vector<double>& times, const std::vector<int>& channels,
                   const std::vector<int>& offsets);
  int get_n_traces() const { return static_cast<int>(offsets_.empty() ? 0 : offsets_.size() - 1); }
  int get_n_photons() const { return static_cast<int>(times_.size()); }
  const std::vector<double>& get_times() const { return times_; }
  const std::vector<int>& get_channels() const { return channels_; }
  const std::vector<int>& get_offsets() const { return offsets_; }
  //! Traces per matrix-matrix batch (default 16). Memory is
  //! `batch * longest_trace * M` doubles during a gradient.
  void set_batch_size(int b);
  int get_batch_size() const { return batch_; }

  // --- likelihood ------------------------------------------------------------
  //! `log L(theta)` summed over all traces.
  double log_likelihood(const std::vector<double>& theta) const;
  //! `log L` of each trace.
  std::vector<double> trace_log_likelihoods(const std::vector<double>& theta) const;
  //! Exact gradient of log_likelihood() with respect to theta.
  std::vector<double> log_likelihood_gradient(const std::vector<double>& theta) const;
  //! `log L` and its gradient in one pass: `[log L, d/dtheta_0, ...]`.
  std::vector<double> log_likelihood_and_gradient(const std::vector<double>& theta) const;

  // --- priors (paper Sec. III D) ----------------------------------------------
  //! Roughness weight `omega` of `-log p = omega sum_k ((mu_{k+1} - 2 mu_k +
  //! mu_{k-1}) / h_s^2)^2`; default 2.15e-4 (paper Table II). 0 disables it.
  void set_roughness_weight(double omega);
  double get_roughness_weight() const { return omega_; }
  //! Width (kT) of the Gaussian anchor on the mean knot height; default 1.
  //! It fixes the offset the likelihood leaves free. <= 0 disables it.
  void set_anchor_sigma(double sigma);
  double get_anchor_sigma() const { return anchor_; }
  //! Gamma priors on the backgrounds, one mode and width per channel:
  //! `-log p = (mode/sd)^2 (r - log r)`, `r = beta/mode`. A channel with
  //! `sd <= 0` has none; empty vectors disable all (the default).
  void set_background_prior(const std::vector<double>& modes, const std::vector<double>& sds);
  //! `log p(theta)` up to a constant.
  double log_prior(const std::vector<double>& theta) const;
  std::vector<double> log_prior_gradient(const std::vector<double>& theta) const;
  //! `-d^2 log p / dtheta^2`, row-major `P x P`.
  std::vector<double> prior_precision(const std::vector<double>& theta) const;
  //! `log L + log p`.
  double log_posterior(const std::vector<double>& theta) const;
  std::vector<double> log_posterior_gradient(const std::vector<double>& theta) const;

  // --- fit ---------------------------------------------------------------------
  //! Maximise the log posterior from `theta0` with L-BFGS.
  /*! The optimiser is tttrlib's header-only L-BFGS (`tttrlib/i_lbfgs.h`,
      two-loop recursion with Armijo backtracking), driven with the exact
      gradient; it is available when IMP.bff links tttrlib
      (`IMP_BFF_HAS_TTTRLIB`) and throws otherwise. The patience criterion is
      applied between blocks of `patience` iterations, each block restarting
      the L-BFGS history. */
  FRETLandscapeFit fit(const std::vector<double>& theta0,
                       const FRETLandscapeFitOptions& options = FRETLandscapeFitOptions()) const;

  IMP_SHOWABLE_INLINE(FRETLandscapeModel,
                      out << "FRETLandscapeModel(M " << m_ << ", K " << k_
                          << ", channels " << c_ << ", traces " << get_n_traces()
                          << ")");

 private:
  void check_theta(const std::vector<double>& theta) const;
  //! Runs the filter over `traces`; fills logL (and the gradient if asked).
  double evaluate(const std::vector<double>& theta, const std::vector<int>& traces,
                  std::vector<double>* gradient, std::vector<double>* per_trace) const;

  double xmin_, xmax_, h_, r0_;
  int m_, k_, c_, batch_ = 16;
  std::vector<double> x_, phi_, eff_, dfrac_, afrac_;
  NaturalCubicSpline spline_;
  std::vector<double> times_;
  std::vector<int> channels_, offsets_;
  double tau_max_ = 0.0;
  double omega_ = 2.15e-4, anchor_ = 1.0;
  std::vector<double> bg_mode_, bg_sd_;
};
IMP_VALUES(FRETLandscapeModel, FRETLandscapeModels);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_FRETLANDSCAPE_H
