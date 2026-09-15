/**
 * \file IMP/bff/Sampling.h
 * \brief One sampler interface: what is sampled (SamplingTarget), one transition (SamplerKernel),
 *        the driver (run_sampler) and its result (SampleResult).
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 * PRD-147 (2026-09-15, tpeulen: "cant you design a common sampler interface?"). Every sampler bff
 * has is a kernel behind this interface, and every kernel is an entry in the registry (category
 * `sampler`, Registry.h), so a caller -- chisurf's selector, a script, MCMCSampler -- finds what
 * exists and what each one needs by reading data, not a list.
 *
 * - **SamplingTarget**: a log density, optionally its gradient, an optional box. A kernel that needs
 *   a gradient refuses a target without one; finite differences only through the named adapter
 *   with_finite_difference_gradient().
 * - **SamplerKernel**: advances every walker it holds by one transition. A *chain* kernel (NUTS,
 *   blocked Metropolis) holds one walker, and independent chains are independent kernels; an
 *   *ensemble* kernel (stretch, differential evolution, ensemble slice) holds interacting walkers.
 * - **run_sampler**: independent chains, or independent ensembles, from their starts, each on its own
 *   seed and thread; warm-up, then draws. SampleResult::independent_group says which rows of the
 *   result are independent of which, because convergence diagnostics (SamplerDiagnostics.h) are only
 *   valid across independent chains -- never across the walkers of one ensemble.
 */

#ifndef IMPBFF_SAMPLING_H
#define IMPBFF_SAMPLING_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/internal/BayesianParallel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A sampler asked to do something it cannot: a gradient it does not have, a start outside the box,
//! starts of the wrong shape, an unknown kernel.
class SamplerConfigurationError : public std::domain_error {
 public:
  explicit SamplerConfigurationError(const std::string& what_arg) : std::domain_error(what_arg) {}
};

//! What is sampled.
struct SamplingTarget {
  std::size_t dim = 0;
  //! log density up to a constant; -inf (or NaN) rejects
  std::function<double(const std::vector<double>& x)> log_density;
  //! log density and its gradient into grad (resized by the callee or pre-sized to dim); optional
  std::function<double(const std::vector<double>& x, std::vector<double>& grad)> log_density_gradient;
  //! optional: the log density and auxiliary values ("blobs", as emcee's) a kernel keeps per walker --
  //! MCMCSampler's log prior and chi^2, say; used instead of log_density when set
  std::function<double(const std::vector<double>& x, std::vector<double>& blobs)> log_density_blobs;
  //! optional box; empty = unbounded; a point outside has log density -inf (never clipped)
  std::vector<double> lower, upper;
  std::vector<std::string> names;

  bool has_gradient() const { return static_cast<bool>(log_density_gradient); }
  bool in_box(const std::vector<double>& x) const {
    for (std::size_t i = 0; i < lower.size() && i < x.size(); ++i)
      if (x[i] < lower[i]) return false;
    for (std::size_t i = 0; i < upper.size() && i < x.size(); ++i)
      if (x[i] > upper[i]) return false;
    return true;
  }
};

//! A target from a log density alone.
inline SamplingTarget sampling_target(std::size_t dim, std::function<double(const std::vector<double>&)> f) {
  SamplingTarget t;
  t.dim = dim;
  t.log_density = std::move(f);
  return t;
}

//! A target from a log density and its gradient (the value-only call uses the gradient's).
inline SamplingTarget sampling_target(std::size_t dim,
                                      std::function<double(const std::vector<double>&, std::vector<double>&)> f_and_grad) {
  SamplingTarget t;
  t.dim = dim;
  t.log_density_gradient = f_and_grad;
  t.log_density = [f_and_grad, dim](const std::vector<double>& x) {
    std::vector<double> g(dim);
    return f_and_grad(x, g);
  };
  return t;
}

//! The explicit, named way to give a gradient-free target a gradient: central differences of step h
//! per coordinate (2 dim evaluations per gradient). Never applied implicitly.
inline SamplingTarget with_finite_difference_gradient(SamplingTarget t, double h = 1e-6) {
  const auto f = t.log_density;
  const std::size_t n = t.dim;
  t.log_density_gradient = [f, n, h](const std::vector<double>& x, std::vector<double>& g) {
    g.assign(n, 0.0);
    std::vector<double> y = x;
    for (std::size_t i = 0; i < n; ++i) {
      y[i] = x[i] + h;
      const double up = f(y);
      y[i] = x[i] - h;
      const double dn = f(y);
      y[i] = x[i];
      g[i] = (up - dn) / (2.0 * h);
    }
    return f(x);
  };
  return t;
}

//! One transition of every walker a kernel holds.
class SamplerKernel {
 public:
  virtual ~SamplerKernel() = default;
  //! the registry key
  virtual std::string name() const = 0;
  virtual bool needs_gradient() const { return false; }
  //! walkers interact (stretch, DE, ensemble slice); false for a chain kernel (one walker)
  virtual bool is_ensemble() const { return false; }
  //! take the walkers and evaluate the target there; throws SamplerConfigurationError
  virtual void initialize(const SamplingTarget& target, const std::vector<std::vector<double>>& walkers,
                          std::mt19937_64& rng) = 0;
  //! advance every walker once
  virtual void transition(std::mt19937_64& rng) = 0;
  virtual const std::vector<std::vector<double>>& walkers() const = 0;
  virtual const std::vector<double>& log_density() const = 0;
  //! warm-up: the kernel tunes itself over the next n_warmup transitions
  virtual void begin_warmup(int n_warmup) { (void)n_warmup; }
  //! warm-up ends; tuned quantities are frozen
  virtual void end_warmup() {}
  //! per-draw statistics, Stan-style names
  virtual std::vector<std::string> stat_names() const = 0;
  //! the last transition's statistics, walker x stat, row-major
  virtual void stats(std::vector<double>& out) const = 0;
  //! the blobs of each walker's current state (empty when the target has none)
  virtual const std::vector<std::vector<double>>& blobs() const {
    static const std::vector<std::vector<double>> none;
    return none;
  }
  //! Metropolis-type counts since the end of warm-up (a slice move counts as proposed and accepted)
  virtual long accepted() const { return 0; }
  virtual long proposed() const { return 0; }
  //! log-density evaluations so far
  virtual long evaluations() const = 0;
  //! a fresh kernel with the same settings and no state
  virtual std::unique_ptr<SamplerKernel> clone() const = 0;
};

//! How run_sampler runs.
struct SamplerOptions {
  int warmup = 1000;
  int draws = 1000;
  int thin = 1;
  //! threads for independent chains or ensembles; 0 = one per chain
  int threads = 0;
  std::uint64_t seed = 1;
  //! called after every recorded draw of chain 0: observer(done, total)
  std::function<void(int, int)> observer;
};

//! Draws and statistics of a run.
struct SampleResult {
  std::string kernel;
  std::vector<std::string> names, stat_names;
  //! [row][draw][dim]; a row is a chain, or one walker of an ensemble
  std::vector<std::vector<std::vector<double>>> draws;
  std::vector<std::vector<double>> log_density;                    //!< [row][draw]
  std::vector<std::vector<std::vector<double>>> stats;             //!< [row][draw][stat]
  //! each statistic summed over warm-up, [row][stat] (warm-up divergences, say)
  std::vector<std::vector<double>> warmup_stat_sums;
  //! rows with different groups are independent; walkers of one ensemble share a group
  std::vector<int> independent_group;
  long evaluations = 0;
  double seconds = 0.0;
  std::vector<std::string> warnings;

  //! One scalar summary per draw, as chains for SamplerDiagnostics.h: one row per independent group
  //! when every group is one row (chain kernels); otherwise throws, because the walkers of an
  //! ensemble are not independent chains.
  std::vector<std::vector<double>> independent_chains(const std::function<double(const std::vector<double>&)>& summary) const {
    for (std::size_t r = 0; r < independent_group.size(); ++r)
      if (independent_group[r] != static_cast<int>(r))
        throw SamplerConfigurationError("independent_chains: rows share an ensemble; summarise each ensemble (e.g. its walker mean) instead");
    std::vector<std::vector<double>> out(draws.size());
    for (std::size_t r = 0; r < draws.size(); ++r) {
      out[r].reserve(draws[r].size());
      for (const auto& x : draws[r]) out[r].push_back(summary(x));
    }
    return out;
  }
};

//! SplitMix64: independent per-chain seeds from one base seed.
inline std::uint64_t sampler_split_seed(std::uint64_t base, std::uint64_t index) {
  std::uint64_t z = base + 0x9E3779B97F4A7C15ull * (index + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/**
 * \brief Run independent chains (chain kernel) or independent ensembles (ensemble kernel).
 *
 * \param starts one entry per chain or ensemble; each is its walkers (one walker for a chain kernel)
 */
inline SampleResult run_sampler(const SamplingTarget& target, const SamplerKernel& prototype,
                                const std::vector<std::vector<std::vector<double>>>& starts,
                                const SamplerOptions& opt = SamplerOptions()) {
  if (starts.empty()) throw SamplerConfigurationError("run_sampler: no starts");
  if (prototype.needs_gradient() && !target.has_gradient())
    throw SamplerConfigurationError("sampler '" + prototype.name() + "' needs a gradient; the target has none "
                                    "(with_finite_difference_gradient() gives it one explicitly)");
  if (!target.log_density) throw SamplerConfigurationError("run_sampler: the target has no log density");
  if (opt.draws < 0 || opt.warmup < 0 || opt.thin < 1) throw SamplerConfigurationError("run_sampler: warmup, draws >= 0 and thin >= 1");
  const std::size_t G = starts.size();
  for (const auto& s : starts) {
    if (s.empty()) throw SamplerConfigurationError("run_sampler: a start without walkers");
    if (!prototype.is_ensemble() && s.size() != 1)
      throw SamplerConfigurationError("run_sampler: a chain kernel takes one walker per start");
    for (const auto& w : s)
      if (w.size() != target.dim) throw SamplerConfigurationError("run_sampler: a start of the wrong dimension");
  }
  struct Group {
    std::vector<std::vector<std::vector<double>>> draws;
    std::vector<std::vector<double>> lp;
    std::vector<std::vector<std::vector<double>>> stats;
    std::vector<std::vector<double>> warmup_sums;
    long evaluations = 0;
    std::string error;
  };
  std::vector<Group> groups(G);
  const auto t0 = std::chrono::steady_clock::now();
  auto run_group = [&](std::size_t g) {
    try {
      std::mt19937_64 rng(sampler_split_seed(opt.seed, g));
      std::unique_ptr<SamplerKernel> k = prototype.clone();
      k->initialize(target, starts[g], rng);
      const std::size_t W = k->walkers().size();
      const std::size_t S = k->stat_names().size();
      Group& out = groups[g];
      out.draws.assign(W, {});
      out.lp.assign(W, {});
      out.stats.assign(W, {});
      out.warmup_sums.assign(W, std::vector<double>(S, 0.0));
      std::vector<double> st;
      k->begin_warmup(opt.warmup);
      for (int i = 0; i < opt.warmup; ++i) {
        k->transition(rng);
        k->stats(st);
        for (std::size_t w = 0; w < W; ++w)
          for (std::size_t j = 0; j < S && (w + 1) * S <= st.size(); ++j) out.warmup_sums[w][j] += st[w * S + j];
      }
      k->end_warmup();
      for (int i = 0; i < opt.draws * opt.thin; ++i) {
        k->transition(rng);
        if ((i + 1) % opt.thin) continue;
        k->stats(st);
        for (std::size_t w = 0; w < W; ++w) {
          out.draws[w].push_back(k->walkers()[w]);
          out.lp[w].push_back(k->log_density()[w]);
          out.stats[w].emplace_back(st.begin() + w * S, st.begin() + (w + 1) * S);
        }
        if (g == 0 && opt.observer) opt.observer((i + 1) / opt.thin, opt.draws);
      }
      out.evaluations = k->evaluations();
    } catch (const std::exception& e) {
      groups[g].error = e.what();
    }
  };
  const std::size_t T = opt.threads > 0 ? std::min<std::size_t>(opt.threads, G) : G;
  if (T <= 1) {
    for (std::size_t g = 0; g < G; ++g) run_group(g);
  } else {
    std::vector<std::thread> pool;
    std::size_t next = 0;
    std::mutex m;
    for (std::size_t t = 0; t < T; ++t)
      pool.emplace_back([&] {
        // a chain thread evaluates its density inline rather than contending for the one decay pool
        internal::bayesian_serial_here() = true;
        for (;;) {
          std::size_t g;
          {
            std::lock_guard<std::mutex> lock(m);
            if (next >= G) return;
            g = next++;
          }
          run_group(g);
        }
      });
    for (auto& th : pool) th.join();
  }
  SampleResult res;
  res.kernel = prototype.name();
  res.names = target.names;
  res.stat_names = prototype.stat_names();
  for (std::size_t g = 0; g < G; ++g) {
    if (!groups[g].error.empty()) throw SamplerConfigurationError("run_sampler: chain " + std::to_string(g) + ": " + groups[g].error);
    for (std::size_t w = 0; w < groups[g].draws.size(); ++w) {
      res.draws.push_back(std::move(groups[g].draws[w]));
      res.log_density.push_back(std::move(groups[g].lp[w]));
      res.stats.push_back(std::move(groups[g].stats[w]));
      res.warmup_stat_sums.push_back(std::move(groups[g].warmup_sums[w]));
      res.independent_group.push_back(static_cast<int>(g));
    }
    res.evaluations += groups[g].evaluations;
  }
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return res;
}

//! Builds a kernel from its options (a JSON object whose keys the entry's params_schema declares).
using SamplerKernelFactory = std::function<std::unique_ptr<SamplerKernel>(const std::string& options_json)>;

/**
 * \brief A kernel by registry name or alias (category `sampler`), configured from JSON options.
 *
 * Options are checked against the entry's `params_schema` before the kernel sees them: an
 * undeclared key, a value of the wrong type or outside `minimum`/`maximum` is refused with the
 * option's own description in the message. Compiled (src/Sampling.cpp): the registry is in the
 * library.
 */
IMPBFFEXPORT std::unique_ptr<SamplerKernel> create_sampler_kernel(const std::string& name_or_alias,
                                                                  const std::string& options_json = "{}");

//! Register a kernel: its registry entry (category `sampler`) and its factory, from a static initialiser.
IMPBFFEXPORT bool register_sampler_kernel(const std::string& key, const std::string& entry_json,
                                         SamplerKernelFactory factory);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_SAMPLING_H */
