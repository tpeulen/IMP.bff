/**
 * \file IMP/bff/SamplerKernels.h
 * \brief The derivative-free samplers as kernels: affine-invariant stretch, differential evolution,
 *        ensemble slice and blocked random-walk Metropolis.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 * PRD-147 step 4. These are the samplers MCMCSampler had inline (ported 1:1 from chisurf's
 * sample.py / ensemble.py, see MCMCSampler.h for the semantics kept), moved behind the SamplerKernel
 * interface (Sampling.h) and registered in the registry (category `sampler`) beside their code, so
 * MCMCSampler, run_sampler and every registry reader reach the same implementations by name. The
 * moves, their constants and their chisurf-specific conventions are unchanged: temperature divides
 * the log-density difference in DE and the blocked walk only, the stretch and slice moves are
 * untempered, DE numbers its generations from zero in warm-up and from one when recording, the
 * blocked walk seeds its proposal from a curvature or a diagonal scaled by the start.
 *
 * Kernels receive their starting walkers; how a start is spread (chisurf's walker cloud, DE's
 * population) is the caller's policy -- MCMCSampler keeps chisurf's.
 */

#ifndef IMPBFF_SAMPLERKERNELS_H
#define IMPBFF_SAMPLERKERNELS_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/Sampling.h>
#include <IMP/bff/SamplerWarmup.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

namespace internal {

//! Walkers, their log densities and blobs, and the target they are evaluated on.
class IMPBFFEXPORT WalkerKernelBase : public SamplerKernel {
 public:
  void initialize(const SamplingTarget& target, const std::vector<std::vector<double>>& walkers,
                  std::mt19937_64& rng) override;
  const std::vector<std::vector<double>>& walkers() const override { return walkers_; }
  const std::vector<double>& log_density() const override { return lps_; }
  const std::vector<std::vector<double>>& blobs() const override { return blobs_; }
  long evaluations() const override { return evaluations_; }
  long accepted() const override { return accepted_; }
  long proposed() const override { return proposed_; }

 protected:
  //! log density at x, filling blobs when the target provides them; -inf outside the box
  double eval(const std::vector<double>& x, std::vector<double>& blobs);
  //! kernel-specific set-up after the walkers are evaluated
  virtual void on_initialize(std::mt19937_64& rng) { (void)rng; }

  SamplingTarget target_;
  std::size_t ndim_ = 0;
  std::vector<std::vector<double>> walkers_;
  std::vector<double> lps_;
  std::vector<std::vector<double>> blobs_;
  long evaluations_ = 0;
  long accepted_ = 0, proposed_ = 0;   //!< since the end of warm-up
  std::vector<double> last_accepted_;  //!< per walker, the last transition
};

}  // namespace internal

//! The affine-invariant ensemble stretch move (Goodman & Weare 2010; chisurf's EnsembleSampler).
class IMPBFFEXPORT StretchKernel : public internal::WalkerKernelBase {
 public:
  explicit StretchKernel(double stretch_scale = 2.0, bool live_dangerously = false);
  std::string name() const override { return "stretch"; }
  bool is_ensemble() const override { return true; }
  void transition(std::mt19937_64& rng) override;
  void end_warmup() override;
  std::vector<std::string> stat_names() const override { return {"accepted"}; }
  void stats(std::vector<double>& out) const override { out = last_accepted_; }
  std::unique_ptr<SamplerKernel> clone() const override;
  double stretch_scale() const { return a_; }

 protected:
  void on_initialize(std::mt19937_64& rng) override;

 private:
  double a_;
  bool live_dangerously_;
};

//! The ensemble slice sampler (Karamanis & Beutler 2021; what zeus runs), differential directions.
class IMPBFFEXPORT EnsembleSliceKernel : public internal::WalkerKernelBase {
 public:
  //! \param mu direction scale; tuned during warm-up unless \p tune is false
  EnsembleSliceKernel(double mu = 1.0, bool tune = true, int max_steps = 10000);
  std::string name() const override { return "slice"; }
  bool is_ensemble() const override { return true; }
  void transition(std::mt19937_64& rng) override;
  void begin_warmup(int n_warmup) override { tuning_ = tune_ && n_warmup > 0; }
  void end_warmup() override;
  std::vector<std::string> stat_names() const override { return {"accepted", "expansions", "contractions", "truncated"}; }
  void stats(std::vector<double>& out) const override;
  std::unique_ptr<SamplerKernel> clone() const override;
  double mu() const { return mu_; }
  long truncations() const { return truncations_; }
  void set_live_dangerously(bool v) { live_dangerously_ = v; }

 protected:
  void on_initialize(std::mt19937_64& rng) override;

 private:
  std::vector<double> slice_along(const std::vector<double>& x, const std::vector<double>& direction, double log_p_x,
                                  int* expansions, int* contractions, bool* truncated, std::mt19937_64& rng);
  double mu_;
  bool tune_, tuning_ = false, live_dangerously_ = false;
  int max_steps_;
  long truncations_ = 0, expansions_ = 0, contractions_ = 0;
  long last_expansions_ = 0, last_contractions_ = 0, last_truncated_ = 0;
};

//! Differential-evolution MCMC (ter Braak 2006) with chisurf's gamma = 1 every tenth generation and
//! snooker updates (ter Braak & Vrugt 2008).
class IMPBFFEXPORT DifferentialEvolutionKernel : public internal::WalkerKernelBase {
 public:
  DifferentialEvolutionKernel(double jitter = 1e-4, double snooker = 0.1, double temp = 1.0);
  std::string name() const override { return "de"; }
  bool is_ensemble() const override { return true; }
  void transition(std::mt19937_64& rng) override;
  //! chisurf numbers warm-up generations from zero and recorded ones from one
  void begin_warmup(int n_warmup) override { if (n_warmup > 0) generation_ = 0; }
  void end_warmup() override;
  std::vector<std::string> stat_names() const override { return {"accepted"}; }
  void stats(std::vector<double>& out) const override { out = last_accepted_; }
  std::unique_ptr<SamplerKernel> clone() const override;

 protected:
  void on_initialize(std::mt19937_64& rng) override;

 private:
  double jitter_, snooker_, temp_;
  long generation_ = 1;
  std::vector<double> noise_;
};

//! Blocked random-walk Metropolis (chisurf's walk_mcmc_blocked): a full-covariance proposal per
//! block, adapted during warm-up on growing windows (SamplerWarmup.h) with dual averaging of each
//! block's log scale, frozen afterwards.
class IMPBFFEXPORT BlockedMetropolisKernel : public internal::WalkerKernelBase {
 public:
  /**
   * \param blocks index partition of the parameter vector; empty = one block over everything
   * \param proposal_covariance dim x dim curvature seed (empty = diagonal |start| * step_size); a
   *        block whose submatrix is finite and positive definite keeps its shape through warm-up
   */
  BlockedMetropolisKernel(const std::vector<std::vector<int>>& blocks = {},
                          const std::vector<std::vector<double>>& proposal_covariance = {},
                          double step_size = 0.1, double temp = 1.0);
  std::string name() const override { return "metropolis"; }
  void transition(std::mt19937_64& rng) override;
  void begin_warmup(int n_warmup) override;
  void end_warmup() override;
  std::vector<std::string> stat_names() const override { return {"accepted"}; }
  void stats(std::vector<double>& out) const override { out = last_accepted_; }
  std::unique_ptr<SamplerKernel> clone() const override;
  std::vector<std::vector<int>> blocks() const;
  std::vector<double> block_acceptance_rates() const;

 protected:
  void on_initialize(std::mt19937_64& rng) override;

 private:
  struct Block {
    std::vector<int> indices;
    std::vector<double> factor;       // flat lower-triangular Cholesky
    double log_scale = 0.0;
    DualAveragingStepSize adapter;
    long accepted = 0, proposed = 0;  // since the end of warm-up
    bool from_curvature = false;
  };
  void sweep(bool adapt, std::mt19937_64& rng);
  std::vector<std::vector<int>> partition_;
  std::vector<std::vector<double>> covariance_;
  double step_size_, temp_;
  std::vector<Block> blocks_;
  // warm-up
  int warmup_total_ = 0, warmup_done_ = 0;
  WarmupWindows windows_;
  std::vector<std::vector<double>> warmup_draws_;
};

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_SAMPLERKERNELS_H */
