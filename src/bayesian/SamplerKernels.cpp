/**
 * \file SamplerKernels.cpp
 * \brief Stretch, differential evolution, ensemble slice and blocked Metropolis as kernels, and their
 *        registry entries.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 * Moved from MCMCSampler.cpp by PRD-147 step 4; each move is the code that was there, with the
 * walker state and the evaluation now the kernel's. The comments name the chisurf function each
 * piece is ported from. Helpers live in kernel_detail: this file is compiled into a unity build.
 */

#include <IMP/bff/SamplerKernels.h>

#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/ordered_map.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

IMPBFF_BEGIN_NAMESPACE

namespace kernel_detail {

typedef std::vector<std::vector<double> > Matrix;

//! Lower-triangular Cholesky factor of A, or an empty matrix on failure.
Matrix cholesky(const Matrix& a) {
  const std::size_t n = a.size();
  Matrix l(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double sum = a[i][j];
      for (std::size_t k = 0; k < j; ++k) sum -= l[i][k] * l[j][k];
      if (i == j) {
        if (!(sum > 0.0)) return Matrix();
        l[i][j] = std::sqrt(sum);
      } else {
        l[i][j] = sum / l[j][j];
      }
    }
  }
  return l;
}

//! chisurf's _cholesky_or_diagonal: ridge, then drop the correlations.
Matrix cholesky_or_diagonal(const Matrix& cov) {
  const std::size_t k = cov.size();
  double scale = 0.0;
  for (std::size_t i = 0; i < k; ++i) scale += cov[i][i];
  scale /= (k > 0 ? static_cast<double>(k) : 1.0);
  const double ridges[4] = {0.0, 1e-10, 1e-6, 1e-3};
  for (int r = 0; r < 4; ++r) {
    Matrix ridged = cov;
    for (std::size_t i = 0; i < k; ++i) ridged[i][i] += ridges[r] * scale;
    Matrix factor = cholesky(ridged);
    if (!factor.empty()) return factor;
  }
  Matrix diag(k, std::vector<double>(k, 0.0));
  for (std::size_t i = 0; i < k; ++i) diag[i][i] = std::sqrt(std::max(cov[i][i], 1e-30));
  return diag;
}

//! chisurf's _regularised_covariance: a sample covariance shrunk toward its own diagonal by n/(n+5).
/*!  Returns an empty matrix when the window is unusable (too short, not finite, or a zero variance),
     as chisurf returns None. */
Matrix regularised_covariance(const std::vector<const double*>& draws, std::size_t n_draws, const std::vector<int>& idx) {
  const std::size_t k = idx.size();
  if (n_draws < 3 || k == 0) return Matrix();
  Matrix cov(k, std::vector<double>(k, 0.0));
  std::vector<double> mean(k, 0.0);
  for (std::size_t d = 0; d < n_draws; ++d)
    for (std::size_t j = 0; j < k; ++j) mean[j] += draws[d][static_cast<std::size_t>(idx[j])];
  for (std::size_t j = 0; j < k; ++j) mean[j] /= static_cast<double>(n_draws);
  for (std::size_t d = 0; d < n_draws; ++d)
    for (std::size_t i = 0; i < k; ++i)
      for (std::size_t j = 0; j < k; ++j)
        cov[i][j] += (draws[d][static_cast<std::size_t>(idx[i])] - mean[i]) *
                     (draws[d][static_cast<std::size_t>(idx[j])] - mean[j]);
  const double denom = static_cast<double>(n_draws) - 1.0;
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j) cov[i][j] /= denom;
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j)
      if (!std::isfinite(cov[i][j])) return Matrix();
  for (std::size_t i = 0; i < k; ++i)
    if (!(cov[i][i] > 0.0)) return Matrix();
  const double weight = static_cast<double>(n_draws) / (static_cast<double>(n_draws) + 5.0);
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j) cov[i][j] = weight * cov[i][j] + (1.0 - weight) * (i == j ? cov[i][i] : 0.0);
  return cov;
}

//! A lower-triangular matrix flattened row-major.
std::vector<double> flatten_factor(const Matrix& m) {
  std::vector<double> flat;
  flat.reserve(m.size() * m.size());
  for (std::size_t i = 0; i < m.size(); ++i)
    for (std::size_t j = 0; j < m[i].size(); ++j) flat.push_back(m[i][j]);
  return flat;
}

//! k distinct entries of a population, uniformly (a partial Fisher-Yates, standing in for numpy's
//! rng.choice(..., replace=False)).
std::vector<int> sample_without_replacement(const std::vector<int>& population, std::size_t k, std::mt19937_64& rng) {
  std::vector<int> pool = population;
  if (k > pool.size()) k = pool.size();
  for (std::size_t i = 0; i < k; ++i) {
    std::uniform_int_distribution<std::size_t> pick(i, pool.size() - 1);
    const std::size_t j = pick(rng);
    std::swap(pool[i], pool[j]);
  }
  pool.resize(k);
  return pool;
}

//! Eigenvalues of a symmetric matrix by cyclic Jacobi rotations.
void symmetric_eigenvalues(Matrix a, std::vector<double>& values) {
  const std::size_t n = a.size();
  values.assign(n, 0.0);
  if (n == 0) return;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = p + 1; q < n; ++q) off += a[p][q] * a[p][q];
    if (off <= 1e-300) break;
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        if (std::fabs(a[p][q]) <= 1e-300) continue;
        const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;
        for (std::size_t k = 0; k < n; ++k) {
          const double akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
      }
    }
  }
  for (std::size_t i = 0; i < n; ++i) values[i] = a[i][i];
}

//! chisurf's walkers_independent, through the Gram matrix eigenvalues.
bool walkers_independent(const Matrix& coords, std::size_t ndim) {
  const std::size_t n = coords.size();
  if (n < 2 || ndim == 0) return false;
  std::vector<double> mean(ndim, 0.0);
  for (std::size_t w = 0; w < n; ++w)
    for (std::size_t i = 0; i < ndim; ++i) {
      if (!std::isfinite(coords[w][i])) return false;
      mean[i] += coords[w][i];
    }
  for (std::size_t i = 0; i < ndim; ++i) mean[i] /= static_cast<double>(n);
  Matrix c(n, std::vector<double>(ndim, 0.0));
  std::vector<double> col_max(ndim, 0.0);
  for (std::size_t w = 0; w < n; ++w)
    for (std::size_t i = 0; i < ndim; ++i) {
      c[w][i] = coords[w][i] - mean[i];
      col_max[i] = std::max(col_max[i], std::fabs(c[w][i]));
    }
  for (std::size_t i = 0; i < ndim; ++i)
    if (!(col_max[i] > 0.0)) return false;
  std::vector<double> col_norm(ndim, 0.0);
  for (std::size_t w = 0; w < n; ++w)
    for (std::size_t i = 0; i < ndim; ++i) {
      c[w][i] /= col_max[i];
      col_norm[i] += c[w][i] * c[w][i];
    }
  for (std::size_t i = 0; i < ndim; ++i) {
    if (!(col_norm[i] > 0.0)) return false;
    col_norm[i] = std::sqrt(col_norm[i]);
  }
  for (std::size_t w = 0; w < n; ++w)
    for (std::size_t i = 0; i < ndim; ++i) c[w][i] /= col_norm[i];
  // cond(c) = sqrt(largest/smallest eigenvalue of c^T c)
  Matrix gram(ndim, std::vector<double>(ndim, 0.0));
  for (std::size_t w = 0; w < n; ++w)
    for (std::size_t i = 0; i < ndim; ++i)
      for (std::size_t j = 0; j < ndim; ++j) gram[i][j] += c[w][i] * c[w][j];
  std::vector<double> values;
  symmetric_eigenvalues(gram, values);
  double lmin = values[0], lmax = values[0];
  for (std::size_t i = 1; i < values.size(); ++i) {
    lmin = std::min(lmin, values[i]);
    lmax = std::max(lmax, values[i]);
  }
  if (!(lmin > 0.0)) return false;
  return std::sqrt(lmax / lmin) <= 1e8;
}

//! chisurf's ensemble size checks, shared by the stretch and slice moves.
void check_ensemble(const Matrix& walkers, std::size_t ndim, bool live_dangerously) {
  const std::size_t n = walkers.size();
  if (n < 4)
    throw SamplerConfigurationError("an ensemble of " + std::to_string(n) +
                                    " walkers is too small to be split into two halves that propose "
                                    "from each other; use at least 4");
  if (n < 2 * ndim && !live_dangerously)
    throw SamplerConfigurationError("an ensemble of " + std::to_string(n) + " walkers cannot span " +
                                    std::to_string(ndim) + " dimensions; use at least " + std::to_string(2 * ndim) +
                                    " walkers");
  if (!walkers_independent(walkers, ndim))
    throw SamplerConfigurationError("Initial state has a large condition number. The walkers span "
                                    "less than the full parameter space, so the chain cannot explore "
                                    "it -- spread them out.");
}

}  // namespace kernel_detail

// ------------------------------------------------------------------ the base

namespace internal {

void WalkerKernelBase::initialize(const SamplingTarget& target, const std::vector<std::vector<double>>& walkers,
                                  std::mt19937_64& rng) {
  if (walkers.empty()) throw SamplerConfigurationError("sampler '" + name() + "': no walkers");
  target_ = target;
  ndim_ = target.dim;
  walkers_ = walkers;
  lps_.assign(walkers_.size(), 0.0);
  blobs_.assign(walkers_.size(), std::vector<double>());
  last_accepted_.assign(walkers_.size(), 0.0);
  accepted_ = proposed_ = 0;
  on_initialize(rng);
  for (std::size_t w = 0; w < walkers_.size(); ++w) lps_[w] = eval(walkers_[w], blobs_[w]);
}

double WalkerKernelBase::eval(const std::vector<double>& x, std::vector<double>& blobs) {
  ++evaluations_;
  if (!target_.in_box(x)) return -std::numeric_limits<double>::infinity();
  if (target_.log_density_blobs) return target_.log_density_blobs(x, blobs);
  return target_.log_density(x);
}

}  // namespace internal

// ------------------------------------------------------------------- stretch

StretchKernel::StretchKernel(double stretch_scale, bool live_dangerously) : a_(stretch_scale), live_dangerously_(live_dangerously) {
  if (!(a_ > 1.0)) throw SamplerConfigurationError("the stretch scale must exceed one");
}

std::unique_ptr<SamplerKernel> StretchKernel::clone() const {
  return std::unique_ptr<SamplerKernel>(new StretchKernel(a_, live_dangerously_));
}

void StretchKernel::on_initialize(std::mt19937_64& rng) {
  (void)rng;
  kernel_detail::check_ensemble(walkers_, ndim_, live_dangerously_);
}

void StretchKernel::end_warmup() { accepted_ = proposed_ = 0; }

void StretchKernel::transition(std::mt19937_64& rng) {
  // EnsembleSampler._step: two randomly assigned halves, a stretch per active walker along the line
  // to a complementary walker, accepted in log space against (ndim - 1) ln z.
  const int n = static_cast<int>(walkers_.size());
  const int half = n / 2;
  std::vector<int> order(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
  std::shuffle(order.begin(), order.end(), rng);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<double> q(ndim_), blobs;
  last_accepted_.assign(walkers_.size(), 0.0);
  for (int pass = 0; pass < 2; ++pass) {
    for (int s = 0; s < (pass == 0 ? half : n - half); ++s) {
      const int a = pass == 0 ? order[static_cast<std::size_t>(s)] : order[static_cast<std::size_t>(half + s)];
      const int comp_offset = pass == 0 ? half : 0;
      const int comp_size = pass == 0 ? n - half : half;
      std::uniform_int_distribution<int> comp_pick(0, comp_size - 1);
      const int p = order[static_cast<std::size_t>(comp_offset + comp_pick(rng))];
      // z ~ g(z) propto 1/sqrt(z) on [1/a, a], by inverting its CDF
      const double u = uniform(rng);
      const double z = ((a_ - 1.0) * u + 1.0) * ((a_ - 1.0) * u + 1.0) / a_;
      const double factor = (static_cast<double>(ndim_) - 1.0) * std::log(z);
      for (std::size_t d = 0; d < ndim_; ++d) {
        const double partner = walkers_[static_cast<std::size_t>(p)][d];
        const double s_v = walkers_[static_cast<std::size_t>(a)][d];
        q[d] = partner - (partner - s_v) * z;
      }
      const double lp = eval(q, blobs);
      ++proposed_;
      const double delta = factor + lp - lps_[static_cast<std::size_t>(a)];
      if (delta > std::log(uniform(rng))) {
        walkers_[static_cast<std::size_t>(a)] = q;
        lps_[static_cast<std::size_t>(a)] = lp;
        blobs_[static_cast<std::size_t>(a)] = blobs;
        last_accepted_[static_cast<std::size_t>(a)] += 1.0;
        ++accepted_;
      }
    }
  }
}

// --------------------------------------------------------------------- slice

EnsembleSliceKernel::EnsembleSliceKernel(double mu, bool tune, int max_steps) : mu_(mu), tune_(tune), max_steps_(max_steps > 0 ? max_steps : 10000) {
  if (!(mu_ > 0.0)) throw SamplerConfigurationError("the slice direction scale must be positive");
}

std::unique_ptr<SamplerKernel> EnsembleSliceKernel::clone() const {
  auto k = std::unique_ptr<EnsembleSliceKernel>(new EnsembleSliceKernel(mu_, tune_, max_steps_));
  k->live_dangerously_ = live_dangerously_;
  return std::unique_ptr<SamplerKernel>(k.release());
}

void EnsembleSliceKernel::on_initialize(std::mt19937_64& rng) {
  (void)rng;
  kernel_detail::check_ensemble(walkers_, ndim_, live_dangerously_);
}

void EnsembleSliceKernel::end_warmup() {
  // Frozen: a scale that keeps adapting on the recorded chain makes the chain non-Markovian. The
  // counters answer questions about the recorded chain, so warm-up's -- taken with an untuned scale,
  // where a long stepping-out is expected -- do not count against it.
  tuning_ = false;
  truncations_ = expansions_ = contractions_ = 0;
  accepted_ = proposed_ = 0;
}

void EnsembleSliceKernel::stats(std::vector<double>& out) const {
  out.clear();
  for (std::size_t w = 0; w < walkers_.size(); ++w) {
    out.push_back(1.0);
    out.push_back(double(last_expansions_));
    out.push_back(double(last_contractions_));
    out.push_back(double(last_truncated_));
  }
}

std::vector<double> EnsembleSliceKernel::slice_along(const std::vector<double>& x, const std::vector<double>& direction,
                                                     double log_p_x, int* expansions, int* contractions, bool* truncated,
                                                     std::mt19937_64& rng) {
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<double> scratch;
  // The slice height, in log space: y = log p(x) + log u, u ~ U(0, 1).
  const double threshold = log_p_x + std::log(uniform(rng));
  // The initial interval straddles x at a random offset, so the move is reversible: [L, R) of unit
  // length with x somewhere inside it.
  double left = -uniform(rng);
  double right = left + 1.0;
  std::vector<double> probe(x.size());
  const std::size_t n = x.size();
  auto fill = [&](double t) {
    for (std::size_t i = 0; i < n; ++i) probe[i] = x[i] + t * direction[i];
  };
  // Stepping out. This is the half that must be allowed to overshoot: a cap that binds leaves the
  // interval short of the slice and every draw comes from a truncated line, which narrows the
  // posterior without ever producing a rejected point to notice.
  int steps = 0;
  while (steps < max_steps_) {
    fill(left);
    if (eval(probe, scratch) <= threshold) break;
    left -= 1.0;
    ++steps;
    ++(*expansions);
  }
  if (steps >= max_steps_) *truncated = true;
  steps = 0;
  while (steps < max_steps_) {
    fill(right);
    if (eval(probe, scratch) <= threshold) break;
    right += 1.0;
    ++steps;
    ++(*expansions);
  }
  if (steps >= max_steps_) *truncated = true;
  // Shrinkage. A draw outside the slice replaces the end it came from, so the interval closes on the
  // slice and the walk is exact.
  for (int attempt = 0; attempt < max_steps_; ++attempt) {
    const double t = left + uniform(rng) * (right - left);
    fill(t);
    if (eval(probe, scratch) > threshold) return probe;
    if (t < 0.0) {
      left = t;
    } else {
      right = t;
    }
    ++(*contractions);
  }
  // The interval collapsed onto x without a point above the threshold, which happens when the
  // density is flat to numerical precision. Staying is the correct answer: the walker is already in
  // the slice.
  return x;
}

void EnsembleSliceKernel::transition(std::mt19937_64& rng) {
  // Karamanis & Beutler's ensemble slice sampler, which is what zeus runs: the halves and the
  // differential direction of "stretch", and a slice along that direction instead of a Metropolis
  // proposal.
  const int n = static_cast<int>(walkers_.size());
  const int half = n / 2;
  std::vector<int> order(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
  std::shuffle(order.begin(), order.end(), rng);
  long expansions = 0, contractions = 0, truncated_now = 0;
  std::vector<double> direction(ndim_);
  for (int pass = 0; pass < 2; ++pass) {
    for (int s = 0; s < (pass == 0 ? half : n - half); ++s) {
      const int a = pass == 0 ? order[static_cast<std::size_t>(s)] : order[static_cast<std::size_t>(half + s)];
      const int comp_offset = pass == 0 ? half : 0;
      const int comp_size = pass == 0 ? n - half : half;
      // The direction is the difference of **two other** walkers, not the line from this one to a
      // partner. A direction built from the current point makes the move's geometry depend on where
      // the walker is, the slice is then not a slice of a fixed line, and the chain is not the target.
      // Measured on the correlated Gaussian the (partner - self) version samples 20% narrow with no
      // truncation and no other symptom.
      if (comp_size < 2) continue;
      std::uniform_int_distribution<int> comp_pick(0, comp_size - 1);
      const int i1 = comp_pick(rng);
      int i2 = comp_pick(rng);
      while (i2 == i1) i2 = comp_pick(rng);
      const int p1 = order[static_cast<std::size_t>(comp_offset + i1)];
      const int p2 = order[static_cast<std::size_t>(comp_offset + i2)];
      for (std::size_t d = 0; d < ndim_; ++d)
        direction[d] = mu_ * (walkers_[static_cast<std::size_t>(p1)][d] - walkers_[static_cast<std::size_t>(p2)][d]);
      int e = 0, c = 0;
      bool truncated = false;
      const std::vector<double> moved =
          slice_along(walkers_[static_cast<std::size_t>(a)], direction, lps_[static_cast<std::size_t>(a)], &e, &c, &truncated, rng);
      walkers_[static_cast<std::size_t>(a)] = moved;
      lps_[static_cast<std::size_t>(a)] = eval(moved, blobs_[static_cast<std::size_t>(a)]);
      ++proposed_;
      ++accepted_;
      expansions += e;
      contractions += c;
      if (truncated) {
        ++truncations_;
        ++truncated_now;
      }
    }
  }
  expansions_ += expansions;
  contractions_ += contractions;
  last_expansions_ = expansions;
  last_contractions_ = contractions;
  last_truncated_ = truncated_now;
  // Tuning: the scale is right when the interval neither has to be grown nor shrunk much, so mu is
  // moved toward the ratio the two counts imply.
  if (tuning_ && (expansions + contractions) > 0) {
    const double ratio = 2.0 * static_cast<double>(expansions) / static_cast<double>(expansions + contractions);
    mu_ *= std::pow(ratio > 0.0 ? ratio : 0.5, 0.25);
    if (!(mu_ > 1e-8)) mu_ = 1e-8;
    if (mu_ > 1e8) mu_ = 1e8;
  }
}

// ------------------------------------------------------ differential evolution

DifferentialEvolutionKernel::DifferentialEvolutionKernel(double jitter, double snooker, double temp)
    : jitter_(jitter), snooker_(snooker), temp_(temp) {
  if (!(jitter_ >= 0.0)) throw SamplerConfigurationError("the jitter must not be negative");
  if (!(snooker_ >= 0.0 && snooker_ <= 1.0)) throw SamplerConfigurationError("the snooker fraction must lie in [0, 1]");
  if (!(temp_ > 0.0)) throw SamplerConfigurationError("the temperature must be positive");
}

std::unique_ptr<SamplerKernel> DifferentialEvolutionKernel::clone() const {
  return std::unique_ptr<SamplerKernel>(new DifferentialEvolutionKernel(jitter_, snooker_, temp_));
}

void DifferentialEvolutionKernel::on_initialize(std::mt19937_64& rng) {
  (void)rng;
  if (walkers_.size() < 3) throw SamplerConfigurationError("differential evolution needs at least 3 chains");
  // chisurf: the jitter widths come from the start, which is population member 0
  noise_.resize(ndim_);
  for (std::size_t i = 0; i < ndim_; ++i) {
    noise_[i] = std::fabs(walkers_[0][i]) * jitter_;
    if (noise_[i] < 1e-15) noise_[i] = jitter_;
  }
  generation_ = 1;
}

void DifferentialEvolutionKernel::end_warmup() {
  accepted_ = proposed_ = 0;
  generation_ = 1;  // chisurf numbers recordings from 1
}

void DifferentialEvolutionKernel::transition(std::mt19937_64& rng) {
  // sample_differential_evolution._generation, including the every-tenth gamma = 1 jump and the
  // snooker updates with their line Jacobian.
  const long generation_index = generation_++;
  const double gamma0 = 2.38 / std::sqrt(2.0 * static_cast<double>(std::max<std::size_t>(1, ndim_)));
  const double gamma = (generation_index % 10 == 9) ? 1.0 : gamma0;
  const int n = static_cast<int>(walkers_.size());
  std::vector<int> order(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
  std::shuffle(order.begin(), order.end(), rng);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::vector<double> proposal(ndim_, 0.0), direction(ndim_, 0.0), blobs;
  last_accepted_.assign(walkers_.size(), 0.0);
  for (int oi = 0; oi < n; ++oi) {
    const int c = order[static_cast<std::size_t>(oi)];
    // Three (or two) distinct others, as rng.choice(..., replace=False)
    std::vector<int> others;
    others.reserve(static_cast<std::size_t>(n - 1));
    for (int o = 0; o < n; ++o)
      if (o != c) others.push_back(o);
    double log_jacobian = 0.0;
    std::vector<int> picked;
    if (uniform(rng) < snooker_ && n >= 4) {
      picked = kernel_detail::sample_without_replacement(others, 3, rng);
      const int z = picked[0], j = picked[1], k = picked[2];
      for (std::size_t d = 0; d < ndim_; ++d)
        direction[d] = walkers_[static_cast<std::size_t>(c)][d] - walkers_[static_cast<std::size_t>(z)][d];
      double norm = 0.0;
      for (std::size_t d = 0; d < ndim_; ++d) norm += direction[d] * direction[d];
      if (!(norm > 0.0)) continue;
      double proj = 0.0;
      for (std::size_t d = 0; d < ndim_; ++d)
        proj += (walkers_[static_cast<std::size_t>(j)][d] - walkers_[static_cast<std::size_t>(k)][d]) * direction[d];
      proj /= norm;
      const double w = 1.2 + uniform(rng) * (2.2 - 1.2);
      for (std::size_t d = 0; d < ndim_; ++d) proposal[d] = walkers_[static_cast<std::size_t>(c)][d] + w * proj * direction[d];
      double new_norm = 0.0;
      for (std::size_t d = 0; d < ndim_; ++d) {
        const double r = proposal[d] - walkers_[static_cast<std::size_t>(z)][d];
        new_norm += r * r;
      }
      if (!(new_norm > 0.0)) continue;
      log_jacobian = 0.5 * (static_cast<double>(ndim_) - 1.0) * (std::log(new_norm) - std::log(norm));
    } else {
      picked = kernel_detail::sample_without_replacement(others, 2, rng);
      const int j = picked[0], k = picked[1];
      for (std::size_t d = 0; d < ndim_; ++d)
        proposal[d] = walkers_[static_cast<std::size_t>(c)][d] +
                      gamma * (walkers_[static_cast<std::size_t>(j)][d] - walkers_[static_cast<std::size_t>(k)][d]) +
                      normal(rng) * noise_[d];
    }
    const double lp = eval(proposal, blobs);
    ++proposed_;
    if (!std::isfinite(lp)) continue;
    const double delta = (lp - lps_[static_cast<std::size_t>(c)]) / temp_ + log_jacobian;
    if (delta > std::log(uniform(rng))) {
      walkers_[static_cast<std::size_t>(c)] = proposal;
      lps_[static_cast<std::size_t>(c)] = lp;
      blobs_[static_cast<std::size_t>(c)] = blobs;
      last_accepted_[static_cast<std::size_t>(c)] = 1.0;
      ++accepted_;
    }
  }
}

// --------------------------------------------------------- blocked Metropolis

BlockedMetropolisKernel::BlockedMetropolisKernel(const std::vector<std::vector<int>>& blocks,
                                                 const std::vector<std::vector<double>>& proposal_covariance,
                                                 double step_size, double temp)
    : partition_(blocks), covariance_(proposal_covariance), step_size_(step_size), temp_(temp) {
  if (!(step_size_ > 0.0)) throw SamplerConfigurationError("the step size must be positive");
  if (!(temp_ > 0.0)) throw SamplerConfigurationError("the temperature must be positive");
}

std::unique_ptr<SamplerKernel> BlockedMetropolisKernel::clone() const {
  return std::unique_ptr<SamplerKernel>(new BlockedMetropolisKernel(partition_, covariance_, step_size_, temp_));
}

std::vector<std::vector<int>> BlockedMetropolisKernel::blocks() const {
  std::vector<std::vector<int>> out;
  for (const Block& b : blocks_) out.push_back(b.indices);
  return out;
}

std::vector<double> BlockedMetropolisKernel::block_acceptance_rates() const {
  std::vector<double> out;
  for (const Block& b : blocks_)
    out.push_back(b.proposed > 0 ? static_cast<double>(b.accepted) / static_cast<double>(b.proposed) : std::nan(""));
  return out;
}

void BlockedMetropolisKernel::on_initialize(std::mt19937_64& rng) {
  (void)rng;
  if (walkers_.size() != 1) throw SamplerConfigurationError("the blocked Metropolis walk is a chain kernel: one walker");
  const std::vector<double>& start = walkers_[0];
  blocks_.clear();
  std::vector<std::vector<int>> partition = partition_;
  if (partition.empty()) {
    std::vector<int> all(ndim_);
    std::iota(all.begin(), all.end(), 0);
    partition.push_back(all);
  }
  for (const auto& p : partition) {
    if (p.empty()) continue;
    for (int i : p)
      if (i < 0 || i >= static_cast<int>(ndim_)) throw SamplerConfigurationError("blocks: index out of range");
    Block b;
    b.indices = p;
    blocks_.push_back(b);
  }
  // chisurf's _seed_block_covariances: the caller's curvature where it is finite and positive
  // definite for a block, the diagonal otherwise.
  const bool have_full = covariance_.size() == ndim_;
  for (Block& block : blocks_) {
    const std::size_t k = block.indices.size();
    block.from_curvature = false;
    kernel_detail::Matrix cov;
    if (have_full) {
      kernel_detail::Matrix candidate(k, std::vector<double>(k, 0.0));
      bool usable = true;
      for (std::size_t i = 0; i < k && usable; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
          const std::vector<double>& row = covariance_[static_cast<std::size_t>(block.indices[i])];
          const double v = row.size() == ndim_ ? row[static_cast<std::size_t>(block.indices[j])] : std::nan("");
          if (!std::isfinite(v)) {
            usable = false;
            break;
          }
          candidate[i][j] = v;
        }
        if (usable && !(candidate[i][i] > 0.0)) usable = false;
      }
      if (usable && !kernel_detail::cholesky(candidate).empty()) {
        cov = candidate;
        block.from_curvature = true;
      }
    }
    if (cov.empty()) {
      cov.assign(k, std::vector<double>(k, 0.0));
      for (std::size_t i = 0; i < k; ++i) {
        double scale = std::fabs(start[static_cast<std::size_t>(block.indices[i])]) * step_size_;
        if (scale < 1e-15) scale = step_size_;
        cov[i][i] = scale * scale;
      }
    }
    block.factor = kernel_detail::flatten_factor(kernel_detail::cholesky_or_diagonal(cov));
    // chisurf's per-block target: 0.44 for a singleton, 0.44/sqrt(k) (floored at 0.234) otherwise,
    // and the theoretically optimal start 2.38/sqrt(k) for the log scale -- OPTIMAL_RWM_SCALING.
    const double target = k == 1 ? 0.44 : std::max(0.234, 0.44 / std::sqrt(static_cast<double>(k)));
    const double log_scale = std::log(2.38 / std::sqrt(static_cast<double>(std::max<std::size_t>(1, k))));
    block.adapter.set_target(target);
    block.adapter.restart_log(log_scale);
    block.log_scale = log_scale;
    block.accepted = 0;
    block.proposed = 0;
  }
}

void BlockedMetropolisKernel::begin_warmup(int n_warmup) {
  warmup_total_ = std::max(0, n_warmup);
  warmup_done_ = 0;
  windows_ = warmup_windows(warmup_total_);
  warmup_draws_.clear();
  warmup_draws_.reserve(static_cast<std::size_t>(warmup_total_));
}

void BlockedMetropolisKernel::end_warmup() {
  if (warmup_total_ > 0)
    for (Block& b : blocks_) b.log_scale = b.adapter.final_log();
  for (Block& b : blocks_) b.accepted = b.proposed = 0;
  accepted_ = proposed_ = 0;
  warmup_total_ = warmup_done_ = 0;
  warmup_draws_.clear();
}

void BlockedMetropolisKernel::transition(std::mt19937_64& rng) {
  const bool adapting = warmup_done_ < warmup_total_;
  sweep(adapting, rng);
  if (!adapting) return;
  warmup_draws_.push_back(walkers_[0]);
  const int i = warmup_done_++;
  if (!windows_.closes(i + 1)) return;
  // chisurf's growing windows: every estimate uses all draws since the end of the initial buffer,
  // never a sliding one.
  std::vector<const double*> visited;
  for (std::size_t d = static_cast<std::size_t>(windows_.init_buffer); d <= static_cast<std::size_t>(i); ++d)
    visited.push_back(&warmup_draws_[d][0]);
  for (Block& block : blocks_) {
    // An exact curvature seed cannot be improved on by a short warm-up chain; only its scale adapts
    // (chisurf's from_curvature contract).
    if (block.from_curvature) continue;
    const std::size_t k = block.indices.size();
    kernel_detail::Matrix empirical = kernel_detail::regularised_covariance(visited, visited.size(), block.indices);
    if (empirical.empty()) continue;
    kernel_detail::Matrix current_cov(k, std::vector<double>(k, 0.0));
    double size = 0.0, empirical_size = 0.0;
    for (std::size_t r = 0; r < k; ++r)
      for (std::size_t cc = 0; cc < k; ++cc) {
        double dot = 0.0;
        for (std::size_t j = 0; j < k; ++j) dot += block.factor[r * k + j] * block.factor[cc * k + j];
        current_cov[r][cc] = std::exp(2.0 * block.log_scale) * dot;
      }
    for (std::size_t r = 0; r < k; ++r) {
      size += current_cov[r][r];
      empirical_size += empirical[r][r];
    }
    if (!(empirical_size > 0.0 && std::isfinite(size) && size > 0.0)) continue;
    for (std::size_t r = 0; r < k; ++r)
      for (std::size_t cc = 0; cc < k; ++cc) empirical[r][cc] *= size / empirical_size;
    block.factor = kernel_detail::flatten_factor(kernel_detail::cholesky_or_diagonal(empirical));
    block.log_scale = 0.0;
    block.adapter.restart_log(0.0);
  }
}

void BlockedMetropolisKernel::sweep(bool adapt, std::mt19937_64& rng) {
  // walk_mcmc_blocked._sweep: a correlated proposal per block, accepted in log space, with dual
  // averaging of the per-block log scale in warm-up.
  std::vector<double> current = walkers_[0];
  double lp = lps_[0];
  std::vector<double> blobs = blobs_[0], trial_blobs;
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::normal_distribution<double> normal(0.0, 1.0);
  double accepted_blocks = 0.0;
  for (Block& block : blocks_) {
    const std::size_t k = block.indices.size();
    std::vector<double> trial = current;
    std::vector<double> draw(k, 0.0);
    // One standard-normal vector per proposal, shared across the rows of L: draw = L z. Drawing a
    // fresh normal per matrix *entry* yields a diagonal proposal with the right marginal widths and
    // zero correlation, which on a strongly correlated posterior mixes orders of magnitude worse
    // (found 2026-09-02 by the curvature seed).
    std::vector<double> z(k);
    for (std::size_t j = 0; j < k; ++j) z[j] = normal(rng);
    for (std::size_t i = 0; i < k; ++i) {
      double value = 0.0;
      for (std::size_t j = 0; j <= i; ++j) value += block.factor[i * k + j] * z[j];
      draw[i] = value;
    }
    const double scale = std::exp(block.log_scale);
    for (std::size_t i = 0; i < k; ++i)
      trial[static_cast<std::size_t>(block.indices[i])] = current[static_cast<std::size_t>(block.indices[i])] + scale * draw[i];
    const double trial_lp = eval(trial, trial_blobs);
    ++block.proposed;
    ++proposed_;
    double alpha = 0.0;
    if (std::isfinite(trial_lp)) {
      const double delta = (trial_lp - lp) / temp_;
      alpha = delta >= 0.0 ? 1.0 : std::exp(delta);
      if (delta > std::log(uniform(rng))) {
        current = trial;
        lp = trial_lp;
        blobs = trial_blobs;
        ++block.accepted;
        ++accepted_;
        accepted_blocks += 1.0;
      }
    }
    if (adapt) block.log_scale = block.adapter.learn_log(alpha);
  }
  walkers_[0] = current;
  lps_[0] = lp;
  blobs_[0] = blobs;
  last_accepted_.assign(1, blocks_.empty() ? 0.0 : accepted_blocks / static_cast<double>(blocks_.size()));
}

// ----------------------------------------------------------- registry entries

namespace {

using KernelJson = nlohmann::basic_json<nlohmann::ordered_map>;

const char* const kStretchEntry = R"JSON({
  "label": "Ensemble (stretch)",
  "summary": "Affine-invariant walkers that stretch towards each other; no gradient, no tuning.",
  "description": "The affine-invariant ensemble stretch move of Goodman & Weare, as chisurf's EnsembleSampler (and emcee) run it: the walkers are split into two halves, and each walker of one half is proposed along the line to a random walker of the other, stretched by z ~ 1/sqrt(z) on [1/a, a]. The proposal takes its scale and correlations from the ensemble, so nothing has to be known about the posterior in advance. It degrades in high dimension and on curved posteriors. The walkers of one ensemble are not independent chains: R-hat over them is no convergence test -- run several independent ensembles.",
  "references": [
    {"type": "article", "authors": "Goodman J, Weare J", "title": "Ensemble samplers with affine invariance", "year": 2010, "journal": "Commun Appl Math Comput Sci", "volume": "5", "pages": "65-80"},
    {"type": "article", "authors": "Foreman-Mackey D, Hogg DW, Lang D, Goodman J", "title": "emcee: The MCMC Hammer", "year": 2013, "journal": "PASP", "volume": "125", "pages": "306-312"}
  ],
  "params_schema": {"type": "object", "properties": {
    "stretch_scale": {"type": "number", "title": "Stretch scale a", "description": "z is drawn on [1/a, a]", "default": 2.0, "exclusiveMinimum": 1.0},
    "live_dangerously": {"type": "boolean", "title": "Allow fewer than 2 dim walkers", "description": "skip the check that the ensemble can span the space", "default": false, "advanced": true}
  }},
  "kind": "ensemble", "requires_gradient": false, "supports_bounds": true, "uses_covariance_seed": false, "uses_blocks": false,
  "aliases": ["ensemble", "sample_ensemble", "affine", "emcee"],
  "population": "walkers",
  "default_walkers": {"rule": "max", "per_dim": 2, "offset": 2, "minimum": 10},
  "default_warmup": {"rule": "fixed", "value": 0},
  "acceptance_rate": "walker_fractions",
  "statistics": {"accepted": "number of accepted proposals of the walker in the transition"},
  "required_checks": ["several independent ensembles; R-hat across ensembles, not walkers", "acceptance fraction in a usable range (about 0.2-0.5)"]
})JSON";

const char* const kSliceEntry = R"JSON({
  "label": "Ensemble slice",
  "summary": "The stretch move's ensemble directions, sampled by slice sampling: every walker moves every step.",
  "description": "Karamanis & Beutler's ensemble slice sampler (what zeus runs): walkers split into halves, each moved along the difference of two walkers of the other half by slice sampling instead of accept/reject -- stepping out until both ends are below the slice height, then shrinking. No acceptance rate to tune; the direction scale mu is tuned during warm-up and frozen. Several model evaluations per move. The stepping-out cap must not bind: a non-zero truncation count on the recorded chain means its width is not to be believed.",
  "references": [
    {"type": "article", "authors": "Karamanis M, Beutler F", "title": "Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions", "year": 2021, "journal": "Stat Comput", "volume": "31", "pages": "61"},
    {"type": "article", "authors": "Neal RM", "title": "Slice sampling", "year": 2003, "journal": "Ann Stat", "volume": "31", "pages": "705-767"}
  ],
  "params_schema": {"type": "object", "properties": {
    "mu": {"type": "number", "title": "Direction scale", "description": "fixes the scale; when absent it starts at 1 and is tuned during warm-up", "exclusiveMinimum": 0.0},
    "max_steps": {"type": "integer", "title": "Stepping-out cap", "description": "expansions per side before the cap binds (generous on purpose)", "default": 10000, "minimum": 1, "advanced": true},
    "live_dangerously": {"type": "boolean", "title": "Allow fewer than 2 dim walkers", "description": "skip the check that the ensemble can span the space", "default": false, "advanced": true}
  }},
  "kind": "ensemble", "requires_gradient": false, "supports_bounds": true, "uses_covariance_seed": false, "uses_blocks": false,
  "aliases": ["zeus", "ensemble_slice", "sample_slice", "sample_ensemble_slice"],
  "population": "walkers",
  "default_walkers": {"rule": "max", "per_dim": 2, "offset": 2, "minimum": 10},
  "default_warmup": {"rule": "clip", "divisor": 20, "min": 20, "max": 200},
  "acceptance_rate": "proposals",
  "statistics": {"accepted": "1: every slice move is accepted", "expansions": "stepping-out expansions in the transition (whole ensemble)", "contractions": "shrinkage contractions in the transition (whole ensemble)", "truncated": "moves whose stepping-out hit the cap in the transition (whole ensemble)"},
  "required_checks": ["no truncated moves on the recorded chain", "several independent ensembles; R-hat across ensembles, not walkers"]
})JSON";

const char* const kDeEntry = R"JSON({
  "label": "Differential evolution",
  "summary": "Proposes from differences between a population of chains; needs neither a gradient nor a covariance.",
  "description": "Differential-evolution MCMC (ter Braak) as chisurf runs it: each chain proposes a jump along the difference of two other chains scaled by 2.38/sqrt(2 dim), with a small jitter, every tenth generation with scale 1 (a mode-to-mode jump), and a fraction of snooker updates along the line to a third chain with their Jacobian. It cannot be misled by a covariance taken at the wrong point. The population's chains interact: they are not independent chains for R-hat.",
  "references": [
    {"type": "article", "authors": "ter Braak CJF", "title": "A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution", "year": 2006, "journal": "Stat Comput", "volume": "16", "pages": "239-249"},
    {"type": "article", "authors": "ter Braak CJF, Vrugt JA", "title": "Differential Evolution Markov Chain with snooker updater and fewer chains", "year": 2008, "journal": "Stat Comput", "volume": "18", "pages": "435-446"}
  ],
  "params_schema": {"type": "object", "properties": {
    "jitter": {"type": "number", "title": "Jitter", "description": "relative width of the Gaussian noise added to each jump", "default": 0.0001, "minimum": 0.0},
    "snooker": {"type": "number", "title": "Snooker fraction", "description": "fraction of snooker updates", "default": 0.1, "minimum": 0.0, "maximum": 1.0},
    "temp": {"type": "number", "title": "Temperature", "description": "divides the log-density difference; above one flattens the posterior", "default": 1.0, "exclusiveMinimum": 0.0, "advanced": true}
  }},
  "kind": "ensemble", "requires_gradient": false, "supports_bounds": true, "uses_covariance_seed": false, "uses_blocks": false,
  "aliases": ["differential_evolution", "sample_differential_evolution"],
  "population": "chains",
  "default_walkers": {"rule": "max", "per_dim": 2, "offset": 0, "minimum": 8},
  "default_warmup": {"rule": "clip", "divisor": 4, "min": 50, "max": 500},
  "acceptance_rate": "proposals",
  "restores_parameters": true,
  "statistics": {"accepted": "1 when the chain's proposal was accepted in the generation"},
  "required_checks": ["several independent populations; R-hat across populations, not chains", "acceptance rate above a few percent"]
})JSON";

const char* const kMetropolisEntry = R"JSON({
  "label": "Blocked Metropolis (covariance)",
  "summary": "Random-walk Metropolis with a full-covariance proposal per block, adapted during warm-up.",
  "description": "chisurf's walk_mcmc_blocked: parameters are partitioned into blocks (from the fit's factor graph -- parameters sharing the same datasets -- or given explicitly; one block otherwise), each proposed from a Gaussian with the block's covariance. The covariance is seeded from the curvature at the optimum when given (its shape is then kept and only the scale adapts), otherwise from a diagonal scaled by the start, and adapted during warm-up on Stan's windowed schedule with dual averaging of each block's log scale towards 0.44/sqrt(k) acceptance. The right choice on a correlated posterior with a good curvature; random-walk mixing still degrades with dimension.",
  "references": [
    {"type": "article", "authors": "Roberts GO, Gelman A, Gilks WR", "title": "Weak convergence and optimal scaling of random walk Metropolis algorithms", "year": 1997, "journal": "Ann Appl Probab", "volume": "7", "pages": "110-120"},
    {"type": "article", "authors": "Haario H, Saksman E, Tamminen J", "title": "An adaptive Metropolis algorithm", "year": 2001, "journal": "Bernoulli", "volume": "7", "pages": "223-242"}
  ],
  "params_schema": {"type": "object", "properties": {
    "blocks": {"type": "array", "title": "Blocks", "description": "partition of the parameter indices, e.g. [[0, 1], [2]]; empty for one block", "items": {"type": "array", "items": {"type": "integer"}}, "default": [], "advanced": true},
    "proposal_covariance": {"type": "array", "title": "Proposal covariance", "description": "dim x dim curvature seed (rows); empty for the diagonal seed", "items": {"type": "array", "items": {"type": "number"}}, "default": [], "advanced": true},
    "step_size": {"type": "number", "title": "Relative step", "description": "diagonal seed width relative to |start|", "default": 0.1, "exclusiveMinimum": 0.0},
    "temp": {"type": "number", "title": "Temperature", "description": "divides the log-density difference; above one flattens the posterior", "default": 1.0, "exclusiveMinimum": 0.0, "advanced": true}
  }},
  "kind": "chain", "requires_gradient": false, "supports_bounds": true, "uses_covariance_seed": true, "uses_blocks": true,
  "aliases": ["walk", "walk_mcmc", "blocked", "walk_mcmc_blocked"],
  "population": "single",
  "default_walkers": {"rule": "fixed", "value": 1},
  "default_warmup": {"rule": "clip", "divisor": 20, "min": 100, "max": 500},
  "acceptance_rate": "proposals",
  "statistics": {"accepted": "fraction of blocks whose proposal was accepted in the sweep"},
  "required_checks": ["several independent chains with rank R-hat and bulk/tail ESS against declared thresholds", "per-block acceptance away from 0 and 1"]
})JSON";

std::vector<std::vector<int>> int_rows(const KernelJson& v) {
  std::vector<std::vector<int>> out;
  for (const auto& r : v) out.push_back(r.get<std::vector<int>>());
  return out;
}
std::vector<std::vector<double>> double_rows(const KernelJson& v) {
  std::vector<std::vector<double>> out;
  for (const auto& r : v) out.push_back(r.get<std::vector<double>>());
  return out;
}

const bool registered_derivative_free_kernels = [] {
  register_sampler_kernel("stretch", kStretchEntry, [](const std::string& options) {
    const KernelJson o = KernelJson::parse(options);
    return std::unique_ptr<SamplerKernel>(new StretchKernel(o.value("stretch_scale", 2.0), o.value("live_dangerously", false)));
  });
  register_sampler_kernel("slice", kSliceEntry, [](const std::string& options) {
    const KernelJson o = KernelJson::parse(options);
    const bool given = o.contains("mu");
    auto k = new EnsembleSliceKernel(o.value("mu", 1.0), !given, o.value("max_steps", 10000));
    k->set_live_dangerously(o.value("live_dangerously", false));
    return std::unique_ptr<SamplerKernel>(k);
  });
  register_sampler_kernel("de", kDeEntry, [](const std::string& options) {
    const KernelJson o = KernelJson::parse(options);
    return std::unique_ptr<SamplerKernel>(
        new DifferentialEvolutionKernel(o.value("jitter", 1e-4), o.value("snooker", 0.1), o.value("temp", 1.0)));
  });
  register_sampler_kernel("metropolis", kMetropolisEntry, [](const std::string& options) {
    const KernelJson o = KernelJson::parse(options);
    return std::unique_ptr<SamplerKernel>(new BlockedMetropolisKernel(
        o.contains("blocks") ? int_rows(o["blocks"]) : std::vector<std::vector<int>>(),
        o.contains("proposal_covariance") ? double_rows(o["proposal_covariance"]) : std::vector<std::vector<double>>(),
        o.value("step_size", 0.1), o.value("temp", 1.0)));
  });
  return true;
}();

}  // namespace

IMPBFF_END_NAMESPACE
