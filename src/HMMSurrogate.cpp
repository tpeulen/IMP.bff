/**
 * \file HMMSurrogate.cpp
 * \brief A neural network that estimates H2MM parameters from burst features.
 *
 * Ported from tttrlib's HMMSurrogate.cpp (tttrlib 2026-09). The feature
 * extractor, encode/decode and the training-set simulation are unchanged, so
 * a surrogate's features match the NumPy reference and tttrlib's to the bit;
 * the network is IMP::bff::NeuralNet, fitted by train_neural_net().
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/bff_config.h>

#if IMP_BFF_HAS_TTTRLIB

// bff's json first: tttrlib's headers include nlohmann/json_fwd.hpp, which
// then finds its guard defined and stays out of the way.
#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/NetworkDocument.h>

#include <IMP/bff/HMMSurrogate.h>

#include <tttrlib/HMM.h>
#include <tttrlib/Mat.h>
#include <tttrlib/SimPcgRandom.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

// No FMA contraction in this file. The features must reproduce the NumPy
// reference bit for bit, and fusing `wsum += streams[k] * scale` into an FMA
// moves the sliding-window sum by ~1e-16 -- enough to push a windowed value
// across a histogram bin edge (measured in tttrlib: 33 of 200 photons moved
// with contraction on, 0 with it off). Scoped, because IMP compiles bff
// as one unity translation unit: clang's pragma goes at the
// top of each function body that computes features or draws models, where it
// is scoped to that body; GCC's is pushed and popped around the file.
#if defined(__clang__)
#define IMPBFF_HMMSURROGATE_NO_FMA _Pragma("clang fp contract(off)")
#elif defined(__GNUC__)
#define IMPBFF_HMMSURROGATE_NO_FMA
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#else
#define IMPBFF_HMMSURROGATE_NO_FMA
#endif

IMPBFF_BEGIN_NAMESPACE

// Named, not anonymous: IMP compiles bff as one unity translation unit.
namespace hmm_surrogate_detail {

/// Photon-lag offsets of the emission autocorrelation feature.
const long long kAcLags[] = {1, 2, 4, 8, 16, 32};
const int kNAcLags = 6;
/// Half-width of the sliding local-FRET window (in photons).
const int kWindow = 12;
/// Histogram bins over the local-FRET range [0, 1].
const int kNBins = 10;
/// Quantiles of the local-FRET distribution.
const double kQuantiles[] = {0.1, 0.25, 0.5, 0.75, 0.9};
const int kNQuantiles = 5;
/// Transition probabilities below this are stored as this in log10 space.
const double kLogFloor = 1e-6;

/// `numpy.quantile(a, q, method="linear")` on an already-sorted array.
inline double quantile_sorted(const std::vector<double>& sorted, double q) {
  IMPBFF_HMMSURROGATE_NO_FMA
  const std::size_t n = sorted.size();
  if (n == 0) return 0.0;
  if (n == 1) return sorted[0];
  const double idx = q * static_cast<double>(n - 1);
  const double lo = std::floor(idx);
  const double hi = std::ceil(idx);
  const double frac = idx - lo;
  const double a = sorted[static_cast<std::size_t>(lo)];
  const double b = sorted[static_cast<std::size_t>(hi)];
  return a + frac * (b - a);
}

/// Reorder states by descending stream-0 emission, making labels canonical.
inline tttrlib::HmmModel canonical_order(const tttrlib::HmmModel& model) {
  const int n = model.n_states();
  const int p = model.n_symbols();
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  // Stable, so ties keep their order; NumPy's argsort on the distinct
  // emission values these models carry gives the same permutation.
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return model.obs[a * p] > model.obs[b * p]; });
  tttrlib::HmmModel out;
  out.prior.resize(n);
  out.trans.resize(static_cast<std::size_t>(n) * n);
  out.obs.resize(static_cast<std::size_t>(n) * p);
  for (int i = 0; i < n; ++i) {
    out.prior[i] = model.prior[order[i]];
    for (int j = 0; j < n; ++j) out.trans[i * n + j] = model.trans[order[i] * n + order[j]];
    for (int k = 0; k < p; ++k) out.obs[i * p + k] = model.obs[order[i] * p + k];
  }
  out.loglik = model.loglik;
  out.n_iter = model.n_iter;
  out.n_phot = model.n_phot;
  out.converged = model.converged;
  return out;
}

/// Draw a random, well-ordered model over a realistic FRET/kinetics range.
inline tttrlib::HmmModel random_model(int n, int p, tttrlib::SimPcgRandom& rng) {
  IMPBFF_HMMSURROGATE_NO_FMA
  std::vector<double> fret(n);
  for (int i = 0; i < n; ++i) fret[i] = 0.08 + rng.random0i1e() * (0.92 - 0.08);
  std::sort(fret.begin(), fret.end());

  std::vector<double> obs(static_cast<std::size_t>(n) * p);
  if (p == 2) {
    for (int i = 0; i < n; ++i) {
      obs[i * p + 0] = 1.0 - fret[i];
      obs[i * p + 1] = fret[i];
    }
  } else {
    // Dirichlet(1,...,1) == uniform on the simplex, sampled via exponentials.
    for (int i = 0; i < n; ++i) {
      double s = 0.0;
      for (int k = 0; k < p; ++k) {
        const double u = std::max(rng.random0e1e(), 1e-300);
        obs[i * p + k] = -std::log(u);
        s += obs[i * p + k];
      }
      for (int k = 0; k < p; ++k) obs[i * p + k] /= s;
    }
  }

  std::vector<double> trans(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    double off = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      trans[i * n + j] = std::pow(10.0, -2.8 + rng.random0i1e() * (-1.3 + 2.8));
      off += trans[i * n + j];
    }
    trans[i * n + i] = 1.0 - off;
  }

  std::vector<double> prior(n, 1.0 / n);
  tttrlib::row_normalize(trans, n, n);
  tttrlib::row_normalize(obs, n, p);
  return tttrlib::HmmModel(std::move(prior), std::move(trans), std::move(obs));
}

/// Index of the first element of `cum` not less than `u`.
inline int searchsorted(const double* cum, int n, double u) {
  int lo = 0, hi = n;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (cum[mid] < u) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/// The features from an HMM's CSR layout; templated on the index vector types
/// so tttrlib's `int32_t`/`int64_t` vectors and the bindings' `int`/`long long`
/// ones are read without a copy.
template <class IV, class LV, class DV>
std::vector<double> features_from_layout(const IV& streams, const LV& offsets,
                                         const IV& gap_slot, const DV& unique_dt,
                                         int p) {
  IMPBFF_HMMSURROGATE_NO_FMA
  const int n_features = HmmSurrogate::N_FEATURES;
  const std::size_t n = streams.size();
  const double scale = 1.0 / std::max(p - 1, 1);

  // --- mean per-photon FRET signal
  double mu = 0.0;
  if (n > 0) {
    double tot = 0.0;
    for (std::size_t i = 0; i < n; ++i) tot += streams[i] * scale;
    mu = tot / static_cast<double>(n);
  }

  // --- windowed local FRET, via a sliding two-pointer sum (O(N), not O(N*win))
  std::vector<double> loc(n, 0.0);
  const std::size_t n_bursts = offsets.empty() ? 0 : offsets.size() - 1;
  for (std::size_t b = 0; b < n_bursts; ++b) {
    const long long s = offsets[b], e = offsets[b + 1];
    long long a = s;
    long long c = (s + kWindow + 1 > e) ? e : s + kWindow + 1;
    double wsum = 0.0;
    for (long long k = a; k < c; ++k) wsum += streams[k] * scale;
    for (long long j = s; j < e; ++j) {
      const long long na = (j - kWindow < s) ? s : j - kWindow;
      const long long nc = (j + kWindow + 1 > e) ? e : j + kWindow + 1;
      while (a < na) { wsum -= streams[a] * scale; ++a; }
      while (c < nc) { wsum += streams[c] * scale; ++c; }
      loc[j] = wsum / static_cast<double>(c - a);
    }
  }

  // --- photon-lag autocorrelation (per-burst mean, averaged over bursts)
  std::vector<double> ac(kNAcLags, 0.0);
  for (int li = 0; li < kNAcLags; ++li) {
    const long long lag = kAcLags[li];
    double num = 0.0;
    long long cnt = 0;
    for (std::size_t b = 0; b < n_bursts; ++b) {
      const long long s = offsets[b], e = offsets[b + 1];
      const long long m = e - s;
      if (m > lag) {
        double ss = 0.0;
        for (long long j = s; j < e - lag; ++j)
          ss += (streams[j] * scale - mu) * (streams[j + lag] * scale - mu);
        num += ss / static_cast<double>(m - lag);
        ++cnt;
      }
    }
    ac[li] = cnt > 0 ? num / static_cast<double>(cnt) : 0.0;
  }

  std::vector<double> feats;
  feats.reserve(n_features);
  feats.push_back(mu);

  // --- density-normalised histogram over [0, 1], numpy.histogram's
  // uniform-bin path exactly: the index is (v-lo)/(hi-lo)*nbins (not
  // (v-lo)/width; the two disagree on edge values) and then corrected
  // against the edges linspace produced; density divides by the in-range
  // count, grouped as (n/db)/n.sum().
  {
    const double lo = 0.0, hi = 1.0;
    const double width = (hi - lo) / kNBins;
    std::vector<double> edges(kNBins + 1);
    for (int i = 0; i <= kNBins; ++i) edges[i] = lo + i * width;
    edges[kNBins] = hi;

    std::vector<double> counts(kNBins, 0.0);
    double in_range = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double v = loc[i];
      if (v < lo || v > hi) continue;
      int bin = static_cast<int>((v - lo) / (hi - lo) * kNBins);
      if (bin == kNBins) --bin;
      if (v < edges[bin]) --bin;
      if (bin != kNBins - 1 && v >= edges[bin + 1]) ++bin;
      counts[bin] += 1.0;
      in_range += 1.0;
    }
    for (int i = 0; i < kNBins; ++i)
      feats.push_back(in_range > 0.0 ? (counts[i] / width) / in_range : 0.0);
  }

  // --- quantiles of the local-FRET distribution
  {
    std::vector<double> sorted = loc;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < kNQuantiles; ++i) feats.push_back(quantile_sorted(sorted, kQuantiles[i]));
  }

  for (int i = 0; i < kNAcLags; ++i) feats.push_back(ac[i]);

  // --- inter-photon gap statistics (population std, NumPy's default)
  {
    double sum = 0.0, sumsq = 0.0;
    long long cnt = 0;
    if (!unique_dt.empty()) {
      for (std::size_t i = 0; i < gap_slot.size(); ++i) {
        if (gap_slot[i] < 0) continue;
        const double v = static_cast<double>(unique_dt[gap_slot[i]]);
        sum += v;
        sumsq += v * v;
        ++cnt;
      }
    }
    if (cnt > 0) {
      const double mean = sum / static_cast<double>(cnt);
      const double var = std::max(sumsq / static_cast<double>(cnt) - mean * mean, 0.0);
      feats.push_back(mean);
      feats.push_back(std::sqrt(var));
    } else {
      feats.push_back(0.0);
      feats.push_back(0.0);
    }
  }
  return feats;
}

inline void check_layout(std::size_t n_streams_v, const std::vector<long long>& offsets,
                         std::size_t n_gap_slot, const std::vector<int>& gap_slot,
                         std::size_t n_unique) {
  if (n_gap_slot != n_streams_v)
    IMP_THROW("HmmSurrogate: gap_slot and streams differ in length", IMP::ValueException);
  if (!offsets.empty()) {
    if (offsets.front() != 0 ||
        offsets.back() != static_cast<long long>(n_streams_v))
      IMP_THROW("HmmSurrogate: offsets must run from 0 to the photon count",
                IMP::ValueException);
    for (std::size_t i = 1; i < offsets.size(); ++i)
      if (offsets[i] < offsets[i - 1])
        IMP_THROW("HmmSurrogate: offsets must not decrease", IMP::ValueException);
  } else if (n_streams_v != 0) {
    IMP_THROW("HmmSurrogate: photons without burst offsets", IMP::ValueException);
  }
  for (int g : gap_slot)
    if (g >= static_cast<int>(n_unique))
      IMP_THROW("HmmSurrogate: gap_slot indexes past unique_dt", IMP::ValueException);
}

inline HmmSurrogateEstimate to_estimate(const tttrlib::HmmModel& m) {
  return HmmSurrogateEstimate(m.prior, m.trans, m.obs, m.n_phot);
}

inline void build_hmm(tttrlib::HMM& engine,
                      const std::vector<std::vector<long long> >& times,
                      const std::vector<std::vector<int> >& streams, int n_streams) {
  try {
    engine.set_bursts(times, streams, n_streams);
  } catch (const std::exception& e) {
    IMP_THROW(std::string("HmmSurrogate: ") + e.what(), IMP::ValueException);
  }
}

}  // namespace hmm_surrogate_detail

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

HmmSurrogate::HmmSurrogate(const MsgpackBytes& net, int n_states, int n_streams,
                           int features_version)
    : net_document_(net),
      net_(net),
      n_states_(n_states),
      n_streams_(n_streams),
      features_version_(features_version) {
  if (n_states_ < 1 || n_streams_ < 1)
    IMP_THROW("HmmSurrogate: n_states and n_streams must be positive", IMP::ValueException);
  if (net_.get_n_inputs() != N_FEATURES)
    IMP_THROW("HmmSurrogate: net takes " << net_.get_n_inputs() << " inputs, expected "
                                         << N_FEATURES,
              IMP::ValueException);
  const int want = n_targets(n_states_, n_streams_);
  if (net_.get_n_outputs() != want)
    IMP_THROW("HmmSurrogate: net produces " << net_.get_n_outputs()
                                            << " outputs, expected " << want
                                            << " for n_states=" << n_states_
                                            << ", n_streams=" << n_streams_,
              IMP::ValueException);
}

int HmmSurrogate::n_targets(int n, int p) { return n * p + n * (n - 1) + n; }

// ---------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------

std::vector<double> HmmSurrogate::extract_features(const tttrlib::HMM& data) {
  return hmm_surrogate_detail::features_from_layout(
      data.get_streams(), data.get_offsets(), data.get_gap_slot(), data.get_unique_dt(),
      data.get_n_streams());
}

std::vector<double> HmmSurrogate::extract_features_from_bursts(
    const std::vector<std::vector<long long> >& times,
    const std::vector<std::vector<int> >& streams, int n_streams) {
  tttrlib::HMM engine;
  hmm_surrogate_detail::build_hmm(engine, times, streams, n_streams);
  return extract_features(engine);
}

std::vector<double> HmmSurrogate::extract_features_from_layout(
    const std::vector<int>& streams, const std::vector<long long>& offsets,
    const std::vector<int>& gap_slot, const std::vector<long long>& unique_dt,
    int n_streams) {
  if (n_streams < 1) IMP_THROW("HmmSurrogate: n_streams must be positive", IMP::ValueException);
  hmm_surrogate_detail::check_layout(streams.size(), offsets, gap_slot.size(), gap_slot,
                                     unique_dt.size());
  return hmm_surrogate_detail::features_from_layout(streams, offsets, gap_slot, unique_dt,
                                                    n_streams);
}

// ---------------------------------------------------------------------------
// Encode / decode
// ---------------------------------------------------------------------------

std::vector<double> HmmSurrogate::encode(const tttrlib::HmmModel& model) {
  const tttrlib::HmmModel m = hmm_surrogate_detail::canonical_order(model);
  const int n = m.n_states(), p = m.n_symbols();
  std::vector<double> out;
  out.reserve(n_targets(n, p));
  for (int i = 0; i < n * p; ++i) out.push_back(m.obs[i]);
  // Off-diagonal transitions span orders of magnitude, so they are regressed
  // in log10 space; a linear target would effectively ignore them.
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i != j)
        out.push_back(std::log10(std::max(m.trans[i * n + j], hmm_surrogate_detail::kLogFloor)));
  for (int i = 0; i < n; ++i) out.push_back(m.prior[i]);
  return out;
}

std::vector<double> HmmSurrogate::encode_arrays(const std::vector<double>& prior,
                                                const std::vector<double>& trans,
                                                const std::vector<double>& obs) {
  const std::size_t n = prior.size();
  if (n == 0 || trans.size() != n * n || obs.empty() || obs.size() % n != 0)
    IMP_THROW("HmmSurrogate::encode: need prior (n), trans (n*n) and obs (n*p)",
              IMP::ValueException);
  return encode(tttrlib::HmmModel(prior, trans, obs));
}

tttrlib::HmmModel HmmSurrogate::decode(const std::vector<double>& vec, int n, int p) {
  IMPBFF_HMMSURROGATE_NO_FMA
  if (n < 1 || p < 1)
    IMP_THROW("HmmSurrogate::decode: n_states and n_streams must be positive",
              IMP::ValueException);
  if (static_cast<int>(vec.size()) != n_targets(n, p))
    IMP_THROW("HmmSurrogate::decode: vector has " << vec.size() << " entries, expected "
                                                  << n_targets(n, p),
              IMP::ValueException);
  std::size_t k = 0;

  std::vector<double> obs(static_cast<std::size_t>(n) * p);
  for (int i = 0; i < n * p; ++i) obs[i] = std::max(vec[k + i], 1e-6);
  k += static_cast<std::size_t>(n) * p;
  tttrlib::row_normalize(obs, n, p);

  std::vector<double> trans(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) trans[i * n + i] = 1.0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i != j) trans[i * n + j] = std::pow(10.0, vec[k++]);
  for (int i = 0; i < n; ++i) {
    double off = 0.0;
    for (int j = 0; j < n; ++j)
      if (i != j) off += trans[i * n + j];
    trans[i * n + i] = std::max(1.0 - off, 1e-6);
  }
  tttrlib::row_normalize(trans, n, n);

  std::vector<double> prior(n);
  for (int i = 0; i < n; ++i) prior[i] = std::max(vec[k + i], 1e-9);
  tttrlib::row_normalize(prior, 1, n);

  return hmm_surrogate_detail::canonical_order(
      tttrlib::HmmModel(std::move(prior), std::move(trans), std::move(obs)));
}

HmmSurrogateEstimate HmmSurrogate::decode_arrays(const std::vector<double>& vec, int n, int p) {
  return hmm_surrogate_detail::to_estimate(decode(vec, n, p));
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

HmmSurrogateEstimate HmmSurrogate::predict_from_features(
    const std::vector<double>& feats) const {
  if (static_cast<int>(feats.size()) != N_FEATURES)
    IMP_THROW("HmmSurrogate: got " << feats.size() << " features, expected " << N_FEATURES,
              IMP::ValueException);
  double* y = nullptr;
  int n_y = 0;
  net_.predict(feats, 1, &y, &n_y);
  std::vector<double> out(y, y + n_y);
  std::free(y);
  return decode_arrays(out, n_states_, n_streams_);
}

tttrlib::HmmModel HmmSurrogate::predict(const tttrlib::HMM& data) const {
  if (data.get_n_streams() != n_streams_)
    IMP_THROW("HmmSurrogate: trained for n_streams=" << n_streams_ << ", got "
                                                     << data.get_n_streams(),
              IMP::ValueException);
  const HmmSurrogateEstimate e = predict_from_features(extract_features(data));
  tttrlib::HmmModel out(e.get_prior(), e.get_trans(), e.get_obs());
  out.n_phot = data.get_n_photons();
  return out;
}

HmmSurrogateEstimate HmmSurrogate::predict_from_bursts(
    const std::vector<std::vector<long long> >& times,
    const std::vector<std::vector<int> >& streams) const {
  tttrlib::HMM engine;
  hmm_surrogate_detail::build_hmm(engine, times, streams, n_streams_);
  return hmm_surrogate_detail::to_estimate(predict(engine));
}

HmmSurrogateEstimate HmmSurrogate::predict_from_layout(
    const std::vector<int>& streams, const std::vector<long long>& offsets,
    const std::vector<int>& gap_slot, const std::vector<long long>& unique_dt,
    int n_streams) const {
  if (n_streams != n_streams_)
    IMP_THROW("HmmSurrogate: trained for n_streams=" << n_streams_ << ", got " << n_streams,
              IMP::ValueException);
  const HmmSurrogateEstimate e = predict_from_features(
      extract_features_from_layout(streams, offsets, gap_slot, unique_dt, n_streams));
  return HmmSurrogateEstimate(e.get_prior(), e.get_trans(), e.get_obs(),
                              static_cast<long long>(streams.size()));
}

// ---------------------------------------------------------------------------
// Training-set generation and training
// ---------------------------------------------------------------------------

void HmmSurrogate::generate_training_set(int n_states, int n_streams, int n_samples,
                                         int n_bursts, int burst_len, double mean_dt,
                                         int seed, std::vector<double>& X,
                                         std::vector<double>& Y) {
  IMPBFF_HMMSURROGATE_NO_FMA
  namespace d = hmm_surrogate_detail;
  if (n_states < 1 || n_streams < 1)
    IMP_THROW("HmmSurrogate::generate_training_set: n_states and n_streams must be positive",
              IMP::ValueException);
  if (n_samples <= 0 || n_bursts <= 0 || burst_len < 2)
    IMP_THROW("HmmSurrogate::generate_training_set: degenerate problem size",
              IMP::ValueException);
  if (!(mean_dt > 0.0))
    IMP_THROW("HmmSurrogate::generate_training_set: mean_dt must be positive",
              IMP::ValueException);

  const int n_y = n_targets(n_states, n_streams);
  X.assign(static_cast<std::size_t>(n_samples) * N_FEATURES, 0.0);
  Y.assign(static_cast<std::size_t>(n_samples) * n_y, 0.0);

  tttrlib::SimPcgRandom rng;
  rng.reset(static_cast<uint32_t>(seed), 0, 0);

  for (int s = 0; s < n_samples; ++s) {
    const tttrlib::HmmModel model = d::random_model(n_states, n_streams, rng);

    // --- Poisson-spaced macro times per burst
    std::vector<std::vector<long long> > times(n_bursts);
    for (int b = 0; b < n_bursts; ++b) {
      std::vector<long long>& t = times[b];
      t.resize(burst_len);
      t[0] = 0;
      for (int j = 1; j < burst_len; ++j) {
        // Knuth's method; mean_dt is small so this stays cheap.
        const double L = std::exp(-mean_dt);
        double prod = rng.random0e1e();
        long long k = 0;
        while (prod > L) { prod *= rng.random0e1e(); ++k; }
        t[j] = t[j - 1] + k + 1;
      }
    }

    // --- hidden state at each photon from the cached A^dt propagator:
    // O(photons), not O(clock ticks).
    std::vector<std::vector<int> > streams(n_bursts);
    std::vector<double> cum_obs(static_cast<std::size_t>(n_states) * n_streams);
    for (int i = 0; i < n_states; ++i) {
      double acc = 0.0;
      for (int k = 0; k < n_streams; ++k) {
        acc += model.obs[i * n_streams + k];
        cum_obs[i * n_streams + k] = acc;
      }
    }
    std::vector<double> cum_prior(n_states);
    {
      double acc = 0.0;
      for (int i = 0; i < n_states; ++i) { acc += model.prior[i]; cum_prior[i] = acc; }
    }

    std::map<long long, std::vector<double> > cum_pow;  // dt -> cumsum of A^dt rows
    for (int b = 0; b < n_bursts; ++b) {
      const std::vector<long long>& t = times[b];
      std::vector<int>& out = streams[b];
      out.resize(t.size());
      int st = d::searchsorted(cum_prior.data(), n_states, rng.random0i1e());
      if (st >= n_states) st = n_states - 1;
      for (std::size_t j = 0; j < t.size(); ++j) {
        if (j > 0) {
          const long long dt = t[j] - t[j - 1];
          auto it = cum_pow.find(dt);
          if (it == cum_pow.end()) {
            std::vector<double> P =
                tttrlib::mat_power(model.trans.data(), n_states, static_cast<int>(dt));
            for (int r = 0; r < n_states; ++r) {
              double acc = 0.0;
              for (int c = 0; c < n_states; ++c) {
                acc += P[r * n_states + c];
                P[r * n_states + c] = acc;
              }
            }
            it = cum_pow.emplace(dt, std::move(P)).first;
          }
          st = d::searchsorted(it->second.data() + st * n_states, n_states,
                               rng.random0i1e());
          if (st >= n_states) st = n_states - 1;
        }
        int k = d::searchsorted(cum_obs.data() + st * n_streams, n_streams,
                                rng.random0i1e());
        if (k >= n_streams) k = n_streams - 1;
        out[j] = k;
      }
    }

    // The engine's own CSR builder, so the surrogate sees exactly the layout
    // a real dataset produces.
    tttrlib::HMM engine;
    engine.set_bursts(times, streams, n_streams);

    const std::vector<double> feats = extract_features(engine);
    const std::vector<double> target = encode(model);
    std::copy(feats.begin(), feats.end(), X.begin() + static_cast<std::size_t>(s) * N_FEATURES);
    std::copy(target.begin(), target.end(), Y.begin() + static_cast<std::size_t>(s) * n_y);
  }
}

HmmSurrogate HmmSurrogate::train(int n_states, int n_streams, int n_samples, int n_bursts,
                                 int burst_len, double mean_dt,
                                 const NeuralNetTrainOptions& options, int seed) {
  std::vector<double> X, Y;
  generate_training_set(n_states, n_streams, n_samples, n_bursts, burst_len, mean_dt, seed,
                        X, Y);
  NeuralNetTrainOptions opt = options;
  opt.seed = seed;
  const NeuralNetTraining fit = train_neural_net_with_history(
      X, n_samples, N_FEATURES, Y, n_targets(n_states, n_streams), opt);
  HmmSurrogate out(fit.get_network(), n_states, n_streams, FEATURES_VERSION);
  out.loss_curve_ = fit.get_loss_curve();
  out.validation_curve_ = fit.get_validation_curve();
  return out;
}

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------

HmmSurrogate HmmSurrogate::from_msgpack(const MsgpackBytes& document) {
  const nlohmann::json j =
      internal::document_from_msgpack(document, "bff.hmm_surrogate", "HmmSurrogate");
  int fv = FEATURES_VERSION, n_states = 0, n_streams = 0;
  MsgpackBytes net;
  try {
    if (j.contains("features_version")) fv = j.at("features_version").get<int>();
    if (!j.contains("net") || !j.at("net").is_object())
      IMP_THROW("HmmSurrogate: document has no 'net' map", IMP::ValueException);
    // The network is nested as a map; re-encoded, it is the bytes
    // NeuralNet reads, and NeuralNet checks its format.
    net = internal::document_to_msgpack(j.at("net"));
    n_states = j.at("n_states").get<int>();
    n_streams = j.at("n_streams").get<int>();
  } catch (const IMP::ValueException&) {
    throw;
  } catch (const std::exception& e) {
    IMP_THROW("HmmSurrogate: malformed document: " << e.what(), IMP::ValueException);
  }
  if (fv != FEATURES_VERSION)
    IMP_THROW("HmmSurrogate: features_version " << fv << " != current " << FEATURES_VERSION
                                                << "; retrain the surrogate",
              IMP::ValueException);
  return HmmSurrogate(net, n_states, n_streams, fv);
}

MsgpackBytes HmmSurrogate::to_msgpack() const {
  nlohmann::json j = nlohmann::json::object();
  j["format"] = "bff.hmm_surrogate";
  j["version"] = 1;
  j["features_version"] = features_version_;
  j["n_states"] = n_states_;
  j["n_streams"] = n_streams_;
  j["net"] = internal::document_from_msgpack(net_document_, "bff.neural_net", "HmmSurrogate");
  return internal::document_to_msgpack(j);
}

HmmSurrogate HmmSurrogate::from_file(const std::string& path) {
  std::ifstream fh(path.c_str(), std::ios::binary);
  if (!fh) IMP_THROW("HmmSurrogate: cannot open '" << path << "'", IMP::IOException);
  const MsgpackBytes bytes((std::istreambuf_iterator<char>(fh)),
                           std::istreambuf_iterator<char>());
  return from_msgpack(bytes);
}

void HmmSurrogate::to_file(const std::string& path) const {
  const MsgpackBytes bytes = to_msgpack();
  std::ofstream fh(path.c_str(), std::ios::binary);
  if (!fh) IMP_THROW("HmmSurrogate: cannot write '" << path << "'", IMP::IOException);
  fh.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!fh) IMP_THROW("HmmSurrogate: cannot write '" << path << "'", IMP::IOException);
}

IMPBFF_END_NAMESPACE

#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC pop_options
#endif
#undef IMPBFF_HMMSURROGATE_NO_FMA

#endif  // IMP_BFF_HAS_TTTRLIB
