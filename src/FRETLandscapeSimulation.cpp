// SPDX-License-Identifier: BSD-3-Clause
/**
 * Synthetic photon streams from a landscape model and the data-driven start
 * values of a fit (Dingeldein & Covino, arXiv:2608.21061, Sec. III C and F).
 * See FRETLandscape.h.
 */
#include <IMP/bff/FRETLandscape.h>
#include <IMP/bff/States.h>
#include <IMP/bff/internal/pcg_random.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

IMPBFF_BEGIN_NAMESPACE

namespace {

//! Uniform in (0, 1), 53 bits from two draws.
inline double fret_landscape_uniform(pcg32& rng) {
  const std::uint64_t a = rng() >> 5, b = rng() >> 6;
  return (static_cast<double>(a * 67108864ULL + b) + 0.5) / 9007199254740992.0;
}

inline double fret_landscape_exponential(pcg32& rng) {
  return -std::log(fret_landscape_uniform(rng));
}

//! Standard normal pair by Box-Muller; returns one, caches the other.
struct FRETLandscapeNormal {
  bool has = false;
  double next = 0.0;
  double operator()(pcg32& rng) {
    if (has) { has = false; return next; }
    const double r = std::sqrt(-2.0 * std::log(fret_landscape_uniform(rng)));
    const double t = 6.283185307179586 * fret_landscape_uniform(rng);
    next = r * std::sin(t);
    has = true;
    return r * std::cos(t);
  }
};

}  // namespace

FRETLandscapePhotons FRETLandscapeModel::simulate(const std::vector<double>& theta,
                                                  int n_traces, double duration, double dt,
                                                  int seed) const {
  check_theta(theta);
  if (n_traces < 1) IMP_THROW("simulate: n_traces must be >= 1", IMP::ValueException);
  if (!(duration > 0.0) || !(dt > 0.0) || dt > duration)
    IMP_THROW("simulate: need 0 < dt <= duration", IMP::ValueException);
  const std::vector<double> mu(theta.begin(), theta.begin() + k_);
  const std::vector<double> m2 = spline_.knot_curvatures(mu);
  const double D = std::exp(theta[k_]);
  std::vector<double> amp(c_), bg(c_);
  for (int c = 0; c < c_; ++c) {
    amp[c] = std::exp(theta[k_ + 1 + c]);
    bg[c] = std::exp(theta[k_ + 1 + c_ + c]);
  }
  // Boltzmann start on the grid, spread uniformly within the cell
  const std::vector<double> ug = get_landscape(theta);
  const std::vector<double> pi = sqra_stationary_distribution(ug);
  std::vector<double> cdf(m_);
  std::partial_sum(pi.begin(), pi.end(), cdf.begin());
  const long long n_steps = static_cast<long long>(std::llround(duration / dt));
  const double noise = std::sqrt(2.0 * D * dt);

  std::vector<std::vector<double> > tt(n_traces);
  std::vector<std::vector<int> > cc(n_traces);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int m = 0; m < n_traces; ++m) {
    pcg32 rng(static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed)),
              static_cast<std::uint64_t>(m));
    FRETLandscapeNormal normal;
    const double r = fret_landscape_uniform(rng) * cdf.back();
    const int i0 = static_cast<int>(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
    double x = x_[std::min(i0, m_ - 1)] + (fret_landscape_uniform(rng) - 0.5) * h_;
    x = std::min(xmax_, std::max(xmin_, x));
    std::vector<double> lam(c_);
    double need = fret_landscape_exponential(rng);
    for (long long j = 0; j < n_steps; ++j) {
      const double t0 = j * dt;
      double u, du;
      spline_.evaluate_point(mu, m2, x, u, du);
      const double e = fret_efficiency(x, r0_);
      double rate = 0.0;
      for (int c = 0; c < c_; ++c) {
        lam[c] = amp[c] * (dfrac_[c] * (1.0 - e) + afrac_[c] * e) + bg[c];
        rate += lam[c];
      }
      // photons of the step: unit exponentials of integrated rate
      double left = rate * dt, t = t0;
      while (need <= left) {
        t += need / rate;
        left -= need;
        double pick = fret_landscape_uniform(rng) * rate;
        int ch = 0;
        while (ch + 1 < c_ && pick >= lam[ch]) { pick -= lam[ch]; ++ch; }
        tt[m].push_back(t);
        cc[m].push_back(ch);
        need = fret_landscape_exponential(rng);
      }
      need -= left;
      // Euler-Maruyama, reflecting walls
      x += -D * du * dt + noise * normal(rng);
      for (int it = 0; it < 4 && (x < xmin_ || x > xmax_); ++it) {
        if (x < xmin_) x = 2.0 * xmin_ - x;
        if (x > xmax_) x = 2.0 * xmax_ - x;
      }
      x = std::min(xmax_, std::max(xmin_, x));
    }
  }
  FRETLandscapePhotons out;
  out.offsets_.push_back(0);
  for (int m = 0; m < n_traces; ++m) {
    out.times_.insert(out.times_.end(), tt[m].begin(), tt[m].end());
    out.channels_.insert(out.channels_.end(), cc[m].begin(), cc[m].end());
    out.offsets_.push_back(static_cast<int>(out.times_.size()));
  }
  return out;
}

// --- initial guess ---------------------------------------------------------------

namespace {

//! Knot heights from a Boltzmann-inverted kernel density of binned distances.
std::vector<double> fret_landscape_knots_from_bins(
    const std::vector<double>& times, const std::vector<int>& channels,
    const std::vector<int>& offsets, const std::vector<int>& traces, double width,
    const std::vector<double>& rates /* C x M */, int C, const std::vector<double>& x,
    const std::vector<double>& phi, int K, double floor) {
  const int M = static_cast<int>(x.size());
  // log colour probability of each channel at each grid point
  std::vector<double> lp(static_cast<std::size_t>(C) * M);
  for (int i = 0; i < M; ++i) {
    double tot = 0.0;
    for (int c = 0; c < C; ++c) tot += rates[static_cast<std::size_t>(c) * M + i];
    for (int c = 0; c < C; ++c)
      lp[static_cast<std::size_t>(c) * M + i] = std::log(rates[static_cast<std::size_t>(c) * M + i] / tot);
  }
  std::vector<double> samples;
  std::vector<int> counts(C);
  for (int m : traces) {
    const int a = offsets[m], b = offsets[m + 1];
    if (b <= a) continue;
    const double t0 = times[a];
    long long bin = 0;
    std::fill(counts.begin(), counts.end(), 0);
    auto flush = [&]() {
      int n = 0;
      for (int c = 0; c < C; ++c) n += counts[c];
      if (n == 0) return;
      int best = 0;
      double bv = -std::numeric_limits<double>::infinity();
      for (int i = 0; i < M; ++i) {
        double v = 0.0;
        for (int c = 0; c < C; ++c) v += counts[c] * lp[static_cast<std::size_t>(c) * M + i];
        if (v > bv) { bv = v; best = i; }
      }
      samples.push_back(x[best]);
      std::fill(counts.begin(), counts.end(), 0);
    };
    for (int n = a; n < b; ++n) {
      const long long k = static_cast<long long>(std::floor((times[n] - t0) / width));
      if (k != bin) { flush(); bin = k; }
      ++counts[channels[n]];
    }
    flush();
  }
  std::vector<double> mu(K, 0.0);
  if (samples.size() < 2) return mu;
  // Silverman's rule of thumb
  const double n = static_cast<double>(samples.size());
  double mean = 0.0;
  for (double v : samples) mean += v;
  mean /= n;
  double var = 0.0;
  for (double v : samples) var += (v - mean) * (v - mean);
  const double sd = std::sqrt(var / (n - 1.0));
  std::vector<double> sorted(samples);
  std::sort(sorted.begin(), sorted.end());
  const double iqr = sorted[static_cast<std::size_t>(0.75 * (n - 1))] -
                     sorted[static_cast<std::size_t>(0.25 * (n - 1))];
  double spread = std::min(sd, iqr / 1.34);
  if (!(spread > 0.0)) spread = sd > 0.0 ? sd : (x[1] - x[0]);
  const double bw = std::max(0.9 * spread * std::pow(n, -0.2), 0.5 * (x[1] - x[0]));
  // the samples sit on grid points: histogram them, then smooth
  std::vector<double> hist(M, 0.0), dens(M, 0.0);
  const double h = x[1] - x[0];
  for (double v : samples) {
    const int i = std::max(0, std::min(M - 1, static_cast<int>(std::lround((v - x[0]) / h))));
    hist[i] += 1.0;
  }
  for (int i = 0; i < M; ++i) {
    if (hist[i] == 0.0) continue;
    for (int j = 0; j < M; ++j) {
      const double z = (x[j] - x[i]) / bw;
      if (std::fabs(z) < 8.0) dens[j] += hist[i] * std::exp(-0.5 * z * z);
    }
  }
  const double peak = *std::max_element(dens.begin(), dens.end());
  Eigen::VectorXd u(M);
  for (int i = 0; i < M; ++i) u[i] = -std::log(std::max(dens[i], floor * peak) / peak);
  // least squares onto the spline, a whisper of curvature penalty for empty knots
  Eigen::MatrixXd P(M, K);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) P(i, k) = phi[static_cast<std::size_t>(i) * K + k];
  Eigen::MatrixXd A = P.transpose() * P;
  for (int k = 1; k + 1 < K; ++k) {
    const int idx[3] = {k - 1, k, k + 1};
    const double cf[3] = {1.0, -2.0, 1.0};
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) A(idx[a], idx[b]) += 1e-6 * cf[a] * cf[b];
  }
  const Eigen::VectorXd sol = A.ldlt().solve(P.transpose() * u);
  const double off = sol.mean();
  for (int k = 0; k < K; ++k) mu[k] = sol[k] - off;
  return mu;
}

}  // namespace

FRETLandscapeInitialGuess FRETLandscapeModel::initial_guess(
    const FRETLandscapeInitialGuessOptions& options) const {
  if (offsets_.empty()) IMP_THROW("initial_guess: no photons (set_photons)", IMP::ValueException);
  if (options.bin_widths.empty() || options.diffusions.empty())
    IMP_THROW("initial_guess: need candidate bin widths and diffusions", IMP::ValueException);
  std::vector<double> bg = options.backgrounds;
  if (bg.empty()) bg = bg_mode_;
  if (static_cast<int>(bg.size()) != c_)
    IMP_THROW("initial_guess: give backgrounds, or set a background prior", IMP::ValueException);
  const int nt = get_n_traces();
  // count rates per channel over the observed spans
  std::vector<double> rate(c_, 0.0);
  double span = 0.0;
  for (int m = 0; m < nt; ++m) {
    const int a = offsets_[m], b = offsets_[m + 1];
    if (b - a < 2) continue;
    span += times_[b - 1] - times_[a];
    for (int n = a; n < b; ++n) rate[channels_[n]] += 1.0;
  }
  if (!(span > 0.0)) IMP_THROW("initial_guess: traces too short", IMP::ValueException);
  for (double& r : rate) r /= span;
  std::vector<int> train, held, all(nt);
  std::iota(all.begin(), all.end(), 0);
  const int every = std::max(2, options.holdout_every);
  for (int m = 0; m < nt; ++m) (m % every == every - 1 ? held : train).push_back(m);
  if (held.empty() || train.empty()) train = held = all;

  // amplitudes from rates under a landscape (flat to begin with)
  auto amplitudes = [&](const std::vector<double>& pi) {
    std::vector<double> a(c_);
    for (int c = 0; c < c_; ++c) {
      double sh = 0.0;
      for (int i = 0; i < m_; ++i)
        sh += pi[i] * (dfrac_[c] * (1.0 - eff_[i]) + afrac_[c] * eff_[i]);
      a[c] = std::max(rate[c] - bg[c], 0.05 * rate[c]) / std::max(sh, 1e-6);
    }
    return a;
  };
  auto estimate = [&](double width, const std::vector<int>& tr, double D) {
    std::vector<double> pi(m_, 1.0 / m_), mu(k_, 0.0), amp;
    for (int pass = 0; pass < 2; ++pass) {
      amp = amplitudes(pi);
      const std::vector<double> th0 = pack_parameters(mu, D, amp, bg);
      mu = fret_landscape_knots_from_bins(times_, channels_, offsets_, tr, width, get_rates(th0),
                                          c_, x_, phi_, k_, options.density_floor);
      pi = sqra_stationary_distribution(get_landscape(pack_parameters(mu, D, amp, bg)));
    }
    return pack_parameters(mu, D, amplitudes(pi), bg);
  };

  FRETLandscapeInitialGuess out;
  double best = -std::numeric_limits<double>::infinity();
  for (double w : options.bin_widths) {
    if (!(w > 0.0)) IMP_THROW("initial_guess: bin widths must be > 0", IMP::ValueException);
    const std::vector<double> th = estimate(w, train, options.diffusion);
    const double s = evaluate(th, held, nullptr, nullptr);
    out.bin_scores_.push_back(s);
    if (s > best) {
      best = s;
      out.bin_width_ = w;
    }
  }
  std::vector<double> th = estimate(out.bin_width_, all, options.diffusion);
  best = -std::numeric_limits<double>::infinity();
  double bestD = options.diffusion;
  for (double D : options.diffusions) {
    if (!(D > 0.0)) IMP_THROW("initial_guess: diffusions must be > 0", IMP::ValueException);
    th[k_] = std::log(D);
    const double s = evaluate(th, all, nullptr, nullptr);
    out.diffusion_scores_.push_back(s);
    if (s > best) {
      best = s;
      bestD = D;
    }
  }
  th[k_] = std::log(bestD);
  out.theta_ = th;
  return out;
}

IMPBFF_END_NAMESPACE
