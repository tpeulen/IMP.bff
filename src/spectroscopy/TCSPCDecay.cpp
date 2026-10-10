/**
 * \file TCSPCDecay.cpp
 * \brief A time-correlated single-photon-counting decay, as a node.
 *
 * The kernels are tttrlib's, from the vendored copy of `DecayConvolution.h`:
 * `fconv_per_cs_ad<double>` for the periodic reconvolution and
 * `shift_lamp_ad<double>` for the timeshift. What is written here is the
 * *composition* -- the order the instrument model applies its terms in --
 * and the graph around it.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/TCSPCDecay.h>
#include <IMP/bff/internal/TCSPCInstrument.h>
#include <cstdlib>
#include <IMP/bff/internal/NodeConfig.h>

// The kernels. A byte-identical copy of tttrlib's
// `modules/spectroscopy/decay/include/DecayConvolution.h`, kept in step by
// `test/decay/test_decay_convolution_copy_is_identical.py` -- the same
// arrangement, and for the same reasons, as the expression engine (see
// `src/standalone/GraphExpression.cpp`). Only the header-only pieces are used;
// nothing here links tttrlib.
#include <IMP/bff/internal/DecayConvolution.h>
#include <IMP/bff/internal/ResponseFunction.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

namespace {

//! ChiSurf's `rescale_w_bg`, which is **not** the photon library's.
/*! Same least-squares solution, three deliberate differences that change the
    answer: this one guards on `e > 0` *and* a finite weight (an empty channel
    can carry an infinite weight, which would otherwise poison the whole sum),
    adds no epsilon to the squared weight, and returns the factor rather than
    rescaling the model as a side effect. Reproduced rather than unified,
    because unifying them moves fitted amplitudes and that is a decision about
    the fit, not about where the code lives. */
double rescale_factor(const std::vector<double>& model,
                      const std::vector<double>& data,
                      const std::vector<double>& errors, double background,
                      int begin, int end) {
  double sum_nom = 0.0, sum_denom = 0.0;
  for (int i = begin; i < end; ++i) {
    const double e = data[static_cast<std::size_t>(i)];
    if (!(e > 0.0)) continue;
    const double error = errors[static_cast<std::size_t>(i)];
    // The weight is 1/error; a zero error is an infinite weight, which is
    // what the `isfinite` guard on the Python side removes.
    if (!(error > 0.0) || !std::isfinite(error)) continue;
    const double w2 = 1.0 / (error * error);
    if (!std::isfinite(w2)) continue;
    const double m = model[static_cast<std::size_t>(i)];
    sum_nom += m * (e - background) * w2;
    sum_denom += m * m * w2;
  }
  if (sum_denom == 0.0) return 0.0;
  return sum_nom / sum_denom;
}

}  // namespace

std::vector<double> linearization_table(const std::vector<double>& data,
                                        int window_length,
                                        const std::string& window_type,
                                        int x_min, int x_max,
                                        double fill_value) {
  const int n = static_cast<int>(data.size());
  const int lo = std::max(0, x_min);
  const int hi = std::min(n, x_max < 0 ? n : x_max);
  if (hi <= lo) {
    throw std::domain_error("linearization_table: the range is empty");
  }
  // Divided by the mean over [x_min, x_max) ...
  double mean = 0.0;
  for (int i = lo; i < hi; ++i) mean += data[static_cast<std::size_t>(i)];
  mean /= (hi - lo);
  std::vector<double> x2(data.size());
  for (int i = 0; i < n; ++i) x2[static_cast<std::size_t>(i)] = data[static_cast<std::size_t>(i)] / mean;
  // ... then everything outside [x_min, x_max] -- inclusive, as the masked
  // array it was written with -- is the fill, and the rest divided by its mean.
  double masked_mean = 0.0;
  int kept = 0;
  for (int i = 0; i < n; ++i) {
    if (i < x_min || i > x_max) continue;
    masked_mean += x2[static_cast<std::size_t>(i)];
    ++kept;
  }
  if (kept == 0) throw std::domain_error("linearization_table: the range is empty");
  masked_mean /= kept;
  std::vector<double> yn(data.size());
  for (int i = 0; i < n; ++i) {
    yn[static_cast<std::size_t>(i)] =
        (i < x_min || i > x_max) ? fill_value : x2[static_cast<std::size_t>(i)] / masked_mean;
  }
  if (n < window_length) {
    throw std::domain_error("linearization_table: the curve is shorter than the window");
  }
  if (window_length < 3) return yn;
  const int m = window_length;
  std::vector<double> w(static_cast<std::size_t>(m));
  const double pi = 3.14159265358979323846;
  for (int k = 0; k < m; ++k) {
    const double phase = (m > 1) ? 2.0 * pi * k / (m - 1) : 0.0;
    double value;
    if (window_type == "flat") {
      value = 1.0;
    } else if (window_type == "hanning") {
      value = 0.5 - 0.5 * std::cos(phase);
    } else if (window_type == "hamming") {
      value = 0.54 - 0.46 * std::cos(phase);
    } else if (window_type == "bartlett") {
      value = 2.0 / (m - 1) * ((m - 1) / 2.0 - std::fabs(k - (m - 1) / 2.0));
    } else if (window_type == "blackman") {
      value = 0.42 - 0.5 * std::cos(phase) + 0.08 * std::cos(2.0 * phase);
    } else {
      throw std::domain_error("linearization_table: '" + window_type +
                              "' is not a window; the windows are flat, hanning, "
                              "hamming, bartlett and blackman");
    }
    w[static_cast<std::size_t>(k)] = value;
  }
  double total = 0.0;
  for (double v : w) total += v;
  for (double& v : w) v /= total;
  // Reflected copies of both ends, m - 1 samples each.
  std::vector<double> padded;
  padded.reserve(static_cast<std::size_t>(n + 2 * (m - 1)));
  // numpy: 2*d[0] - d[m:1:-1] and 2*d[-1] - d[-1:-m:-1].
  for (int k = m; k >= 2; --k) padded.push_back(2.0 * yn[0] - yn[static_cast<std::size_t>(k)]);
  padded.insert(padded.end(), yn.begin(), yn.end());
  for (int k = 0; k <= m - 2; ++k) {
    padded.push_back(2.0 * yn[static_cast<std::size_t>(n - 1)] - yn[static_cast<std::size_t>(n - 1 - k)]);
  }
  const int length = static_cast<int>(padded.size());
  // numpy's convolve(w, padded, "same") is the middle `length` of the full
  // convolution, offset (m - 1) / 2; the table is that, less m - 1 at each end.
  std::vector<double> out(static_cast<std::size_t>(n));
  const int offset = (m - 1) / 2;
  for (int i = 0; i < n; ++i) {
    const int centre = i + (m - 1) + offset;   // index into the full convolution
    double sum = 0.0;
    for (int k = 0; k < m; ++k) {
      const int j = centre - k;
      if (j >= 0 && j < length) sum += w[static_cast<std::size_t>(k)] * padded[static_cast<std::size_t>(j)];
    }
    out[static_cast<std::size_t>(i)] = sum;
  }
  return out;
}

// Deliberately empty. A `GraphNode` that owns ports has to be owned by a
// `shared_ptr` first -- `add_port` reaches for `shared_from_this()` -- so a
// constructor cannot create any, and every Python-wrapped node is
// constructed before it is owned. The ports are therefore made by
// set_number_of_lifetimes(), which is the one call a caller cannot skip.
TCSPCDecay::TCSPCDecay(const std::string& name) : GraphNode(name) {}

void TCSPCDecay::add_scalar_port(const std::string& key, double value,
                                 GraphPort** slot) {
  std::shared_ptr<GraphPort> port(new GraphPort(value));
  add_input_port(key, port);
  *slot = port.get();
}

void TCSPCDecay::set_number_of_lifetimes(int n) {
  if (n < 0) {
    throw std::domain_error(
        "TCSPCDecay::set_number_of_lifetimes: a negative component count");
  }
  // Ports cannot be removed from a node, so calling this with a *smaller*
  // count leaves the surplus `a`/`t` ports in the map. They feed nothing --
  // only the ports in `lifetime_ports_` are read -- but they are still
  // visible, so build the node once rather than reshaping it.
  if (scatter_port_ == nullptr) {
    // The interleaved spectrum, for a caller that computes it upstream. A
    // vector port, and one that does *not* sanitise: it is fit transport,
    // and a NaN lifetime has to reach `FitChiSquared` and make the misfit
    // infinite rather than be floored to `tiny` and read as a good fit.
    std::shared_ptr<GraphPort> spectrum(new GraphPort(std::vector<double>(1, 0.0)));
    spectrum->set_sanitize(false);
    add_input_port(spectrum_port_key(), spectrum);
    spectrum_port_ = spectrum.get();

    // The convolved curve, for a model that is already a decay. Fit
    // transport as well, so it does not sanitise either.
    std::shared_ptr<GraphPort> curve(new GraphPort(std::vector<double>(1, 0.0)));
    curve->set_sanitize(false);
    add_input_port(curve_port_key(), curve);
    curve_port_ = curve.get();

    add_scalar_port("scatter", 0.0, &scatter_port_);
    add_scalar_port("background", 0.0, &background_port_);
    add_scalar_port("n0", 1.0, &n0_port_);
    add_scalar_port("timeshift", 0.0, &timeshift_port_);
  }
  lifetime_ports_.clear();
  lifetime_ports_.reserve(static_cast<std::size_t>(2 * n));
  for (int i = 0; i < n; ++i) {
    std::ostringstream a, t;
    a << "a" << i;
    t << "t" << i;
    GraphPort* amplitude = nullptr;
    GraphPort* lifetime = nullptr;
    add_scalar_port(a.str(), 1.0, &amplitude);
    add_scalar_port(t.str(), 1.0, &lifetime);
    lifetime_ports_.push_back(amplitude);
    lifetime_ports_.push_back(lifetime);
  }
  n_lifetimes_ = n;
  spectrum_.assign(static_cast<std::size_t>(2 * n), 0.0);
  set_valid(false);
}

bool TCSPCDecay::get_basis_is_jacobian() const {
  // Both of these put the amplitudes through something before the
  // contraction, so the curve depends on each of them by a route the basis
  // does not carry.
  if (normalize_amplitudes_ || autoscale_) return false;
  // A background pattern rescales the model to the counts it leaves over.
  if (!background_pattern_.empty()) return false;
  // |a| is differentiable away from zero and its derivative is the sign, so
  // the basis is the Jacobian up to a per-species sign -- and only a caller
  // who knows that can use it. The spectrum here is the one the last
  // evaluation built, which is where the signs are.
  //
  // The sign cannot be read back out of `spectrum_`: `fabs` is applied in
  // place, so by the time anything can look, every amplitude there is
  // non-negative whatever the caller wrote. `build_spectrum` records it.
  if (absolute_amplitudes_ && saw_negative_amplitude_) return false;
  return true;
}

void TCSPCDecay::set_spectrum_from_port(bool v) {
  if (v && spectrum_port_ == nullptr) {
    throw std::domain_error(
        "TCSPCDecay::set_spectrum_from_port: the ports do not exist yet; "
        "call set_number_of_lifetimes() first, which is what builds them");
  }
  spectrum_from_port_ = v;
  set_valid(false);
}

void TCSPCDecay::set_response_range(int start, int stop) {
  if (start < 0) {
    throw std::domain_error(
        "TCSPCDecay::set_response_range: the window starts before channel 0");
  }
  response_preparation_.active = true;
  response_preparation_.start = start;
  response_preparation_.stop = stop;
  irf_valid_ = false;
  set_valid(false);
}

void TCSPCDecay::rebuild_linearization() {
  if (linearization_curve_.empty()) return;
  std::vector<double> table = linearization_table(
      linearization_curve_, lin_window_length_, lin_window_type_, lin_x_min_,
      lin_x_max_ < 0 ? static_cast<int>(linearization_curve_.size()) : lin_x_max_);
  if (lin_reverse_) std::reverse(table.begin(), table.end());
  set_linearization(table);
}

void TCSPCDecay::set_linearization_curve(const std::vector<double>& curve) {
  linearization_curve_ = curve;
  rebuild_linearization();
}

void TCSPCDecay::set_linearization_smoothing(int window_length,
                                             const std::string& window_type,
                                             int x_min, int x_max, bool reverse) {
  lin_window_length_ = window_length;
  lin_window_type_ = window_type;
  lin_x_min_ = x_min;
  lin_x_max_ = x_max;
  lin_reverse_ = reverse;
  rebuild_linearization();
}

void TCSPCDecay::set_background_pattern(const std::vector<double>& pattern) {
  background_pattern_ = pattern;
  set_valid(false);
}

void TCSPCDecay::set_background_pattern_array(double* in_pattern, int n_pattern) {
  set_background_pattern(std::vector<double>(in_pattern, in_pattern + n_pattern));
}

void TCSPCDecay::set_background_times(double t_background, double t_decay) {
  if (!(t_background > 0.0) || !(t_decay > 0.0)) {
    throw std::domain_error(
        "TCSPCDecay::set_background_times: both measurement times must be positive");
  }
  t_background_ = t_background;
  t_decay_ = t_decay;
  set_valid(false);
}

void TCSPCDecay::set_shift_background_with_response(bool v) {
  shift_background_with_response_ = v;
  set_valid(false);
}

void TCSPCDecay::set_convolution_mode(const std::string& mode) {
  if (mode != "periodic" && mode != "single") {
    throw std::domain_error("TCSPCDecay::set_convolution_mode: '" + mode +
                            "' is not a mode; the modes are periodic and single");
  }
  if (mode != "periodic" && emit_basis_) {
    throw std::domain_error(
        "TCSPCDecay::set_convolution_mode: the basis is the periodic one");
  }
  periodic_ = mode == "periodic";
  set_valid(false);
}

void TCSPCDecay::set_response_from_port(bool v) {
  response_from_port_ = v;
  set_valid(false);
}

void TCSPCDecay::set_convolve(bool v) {
  if (!v && emit_basis_) {
    throw std::domain_error(
        "TCSPCDecay::set_convolve: the basis is the convolved one");
  }
  convolve_ = v;
  set_valid(false);
}

void TCSPCDecay::set_curve_from_port(bool v) {
  // The ports are made with the components; a decay-from-a-port model has
  // none, and zero is a count the builder accepts.
  if (v && curve_port_ == nullptr) set_number_of_lifetimes(0);
  if (v && emit_basis_) {
    throw std::domain_error(
        "TCSPCDecay::set_curve_from_port: a curve from a port has no species "
        "to emit a basis for");
  }
  curve_from_port_ = v;
  set_valid(false);
}

void TCSPCDecay::set_response(const std::vector<double>& response) {
  if (response.empty()) {
    throw std::domain_error("TCSPCDecay::set_response: the response is empty");
  }
  response_ = response;
  shifted_.assign(response_.size(), 0.0);
  ++response_epoch_;
  irf_valid_ = false;
  set_valid(false);
}

void TCSPCDecay::set_data(const std::vector<double>& y,
                          const std::vector<double>& ey) {
  if (y.size() != ey.size()) {
    std::ostringstream m;
    m << "TCSPCDecay::set_data: " << y.size() << " values and " << ey.size()
      << " errors";
    throw std::domain_error(m.str());
  }
  data_y_ = y;
  data_ey_ = ey;
  set_valid(false);
}

void TCSPCDecay::set_timing(double dt, double period) {
  if (!(dt > 0.0)) {
    throw std::domain_error("TCSPCDecay::set_timing: the channel width is not "
                            "positive");
  }
  if (!(period > 0.0)) {
    throw std::domain_error("TCSPCDecay::set_timing: the excitation period is "
                            "not positive");
  }
  dt_ = dt;
  period_ = period;
  set_valid(false);
}

void TCSPCDecay::set_convolution_range(int convolution_stop, int stop) {
  convolution_stop_ = convolution_stop;
  stop_ = stop;
  set_valid(false);
}

void TCSPCDecay::set_scale_range(int start, int stop) {
  scale_start_ = start;
  scale_stop_ = stop;
  set_valid(false);
}

void TCSPCDecay::set_linearization(const std::vector<double>& table) {
  lin_table_ = table;
  set_valid(false);
}

void TCSPCDecay::set_linearization_array(double* in_table, int n_table) {
  set_linearization(std::vector<double>(in_table, in_table + n_table));
}

void TCSPCDecay::set_response_array(double* in_response, int n_response) {
  set_response(std::vector<double>(in_response, in_response + n_response));
}

void TCSPCDecay::set_data_arrays(double* in_data_y, int n_data_y,
                                 double* in_data_ey, int n_data_ey) {
  set_data(std::vector<double>(in_data_y, in_data_y + n_data_y),
           std::vector<double>(in_data_ey, in_data_ey + n_data_ey));
}

namespace {

//! Add the convolved `a t e^{-t/tau}` components of \p x (interleaved, `m` of
//! them) to \p curve. `t e^{-kt} = -d/dk e^{-kt}`, so each is minus the rate
//! derivative of the very kernel the plain components go through, taken in one
//! forward-mode pass: no second convolution to keep consistent, and under
//! periodic excitation the earlier pulses' `(t + nT) e^{-k(t + nT)}` come with it.
void add_t_exponential_components(double* curve, const std::vector<double>& x, const double* irf,
                                  int n_points, bool convolve, bool periodic, double period,
                                  int stop, int convolution_stop, double dt) {
  const int m = static_cast<int>(x.size() / 2);
  if (m == 0) return;
  if (!convolve) {
    for (int c = 0; c < m; ++c) {
      const double a = x[2 * c], tau = x[2 * c + 1];
      if (!(tau > 0.0)) continue;
      const double q = periodic ? std::exp(-period / tau) : 0.0;
      const double tail = periodic ? period * q / ((1.0 - q) * (1.0 - q)) : 0.0;
      const double head = 1.0 / (1.0 - q);
      for (int i = 0; i < n_points; ++i) {
        const double t = i * dt;
        curve[i] += a * std::exp(-t / tau) * (t * head + tail);
      }
    }
    return;
  }
  typedef tttrlib::Dual<double> D;
  std::vector<D> seeded(x.size()), fit(static_cast<std::size_t>(n_points), D(0.0));
  for (int c = 0; c < m; ++c) {
    const double tau = x[2 * c + 1];
    seeded[2 * c] = D(x[2 * c]);
    seeded[2 * c + 1] = D(tau, -tau * tau);  // d tau / d k, with k = 1 / tau
  }
  if (periodic) {
    fconv_per_cs_ad<D>(fit.data(), seeded.data(), irf, m, stop, n_points, period, convolution_stop, dt);
  } else {
    fconv_ad<D>(fit.data(), seeded.data(), irf, m, 0, std::min(convolution_stop + 1, n_points), dt);
  }
  for (int i = 0; i < n_points; ++i) curve[i] -= fit[static_cast<std::size_t>(i)].grad;
}

}  // namespace

void TCSPCDecay::build_spectrum() {
  if (spectrum_from_port_) {
    // Whatever the upstream node wrote, in the interleaved layout. An odd
    // length is a caller error rather than something to round down: the last
    // amplitude would silently lose its lifetime.
    const std::vector<double>& incoming = spectrum_port_->get_values_ref();
    if (incoming.size() < 2 || incoming.size() % 2 != 0) {
      std::ostringstream m;
      m << "TCSPCDecay '" << get_name() << "': the lifetime spectrum has "
        << incoming.size()
        << " entries; it is interleaved (a0, t0, a1, t1, ...) so the count "
           "is even and at least two";
      throw std::domain_error(m.str());
    }
    spectrum_ = incoming;
    n_lifetimes_ = static_cast<int>(incoming.size() / 2);
    // Which components are `t e^{-t/tau}` rather than `e^{-t/tau}`: an
    // optional parallel vector, as a transfer kinetics at a degeneracy gives.
    kinds_.clear();
    if (const std::shared_ptr<GraphPort> kinds = get_input_port(spectrum_kinds_port_key())) {
      const std::vector<double>& given = kinds->get_values_ref();
      const bool any = std::any_of(given.begin(), given.end(), [](double v) { return v != 0.0; });
      if (given.size() == static_cast<std::size_t>(n_lifetimes_)) {
        if (any) kinds_.assign(given.begin(), given.end());
      } else if (any) {
        std::ostringstream m;
        m << "TCSPCDecay '" << get_name() << "': " << given.size()
          << " component kinds for a spectrum of " << n_lifetimes_ << " components";
        throw std::domain_error(m.str());
      }
    }
  } else {
    kinds_.clear();
    spectrum_.resize(static_cast<std::size_t>(2 * n_lifetimes_));
    for (int i = 0; i < n_lifetimes_; ++i) {
      spectrum_[static_cast<std::size_t>(2 * i)] =
          lifetime_ports_[static_cast<std::size_t>(2 * i)]->get_value();
      spectrum_[static_cast<std::size_t>(2 * i + 1)] =
          lifetime_ports_[static_cast<std::size_t>(2 * i + 1)]->get_value();
    }
  }

  // `absolute_amplitudes` and `normalize_amplitudes` describe what the model
  // means by an amplitude, so they apply whichever way the pairs arrived.
  double sum = 0.0;
  saw_negative_amplitude_ = false;
  for (int i = 0; i < n_lifetimes_; ++i) {
    double amplitude = spectrum_[static_cast<std::size_t>(2 * i)];
    if (amplitude < 0.0) saw_negative_amplitude_ = true;
    // `fabs`, where ChiSurf writes `sqrt(a**2)`: the same number for every
    // finite amplitude, and it does not overflow on the way.
    if (absolute_amplitudes_) amplitude = std::fabs(amplitude);
    spectrum_[static_cast<std::size_t>(2 * i)] = amplitude;
    sum += amplitude;
    // Lifetimes are positive: a fit that walks one through zero would
    // otherwise put a growing exponential in the model. ChiSurf takes the
    // absolute value in the getter and writes it back; the write-back is the
    // application's business, the abs() is the model's.
    spectrum_[static_cast<std::size_t>(2 * i + 1)] =
        std::fabs(spectrum_[static_cast<std::size_t>(2 * i + 1)]);
  }
  if (normalize_amplitudes_) {
    const double scale = std::fabs(sum);
    for (int i = 0; i < n_lifetimes_; ++i) {
      spectrum_[static_cast<std::size_t>(2 * i)] /= scale;
    }
  }

  // Drop the species that cannot pay for themselves. The reconvolution below
  // is one serial recursion over every channel *per species*, so this is the
  // only knob on a decay whose spectrum came from a distribution: those have
  // as many species as the distribution has bins, whatever their weight.
  //
  // Compacted in place and `n_lifetimes_` moved down with it, because that
  // count is what the kernel is handed. At the default threshold of zero
  // this drops only exact zeros and is bit-exact -- and it still fires,
  // because a donor-only fraction of zero contributes a whole block of them.
  std::size_t kept = 0;
  double largest = 0.0;
  for (int i = 0; i < n_lifetimes_; ++i) {
    largest = std::max(largest,
                       std::fabs(spectrum_[static_cast<std::size_t>(2 * i)]));
  }
  // A relative threshold against nothing is meaningless: if every amplitude
  // is zero the spectrum is empty, and an empty spectrum is not something
  // the kernel can be handed. Keep it as it is and let the curve be zero.
  // Compaction is skipped when the basis is wanted, so that its columns line
  // up with the input spectrum one for one (see set_emit_basis).
  if (largest > 0.0 && !emit_basis_) {
    const double cutoff = amplitude_threshold_ * largest;
    for (int i = 0; i < n_lifetimes_; ++i) {
      const double amplitude = spectrum_[static_cast<std::size_t>(2 * i)];
      // `>` and not `>=`, so a threshold of exactly zero keeps everything a
      // non-zero amplitude contributes and drops only true zeros.
      if (std::fabs(amplitude) > cutoff) {
        spectrum_[2 * kept] = amplitude;
        spectrum_[2 * kept + 1] =
            spectrum_[static_cast<std::size_t>(2 * i + 1)];
        if (!kinds_.empty()) kinds_[kept] = kinds_[static_cast<std::size_t>(i)];
        ++kept;
      }
    }
    // Every amplitude at or below the cutoff, with a positive largest: only
    // reachable when the threshold is 1 or more. Keep the largest species so
    // there is still a decay rather than an empty spectrum.
    if (kept == 0) {
      std::size_t best = 0;
      for (int i = 0; i < n_lifetimes_; ++i) {
        if (std::fabs(spectrum_[static_cast<std::size_t>(2 * i)]) == largest) {
          best = static_cast<std::size_t>(i);
          break;
        }
      }
      spectrum_[0] = spectrum_[2 * best];
      spectrum_[1] = spectrum_[2 * best + 1];
      if (!kinds_.empty()) kinds_[0] = kinds_[best];
      kept = 1;
    }
    spectrum_.resize(2 * kept);
    if (!kinds_.empty()) kinds_.resize(kept);
  }
  n_active_ = static_cast<int>(spectrum_.size() / 2);
}

void TCSPCDecay::set_emit_basis(bool on) {
  if (on == emit_basis_) return;
  if (on && (!periodic_ || !convolve_)) {
    throw std::domain_error(
        "TCSPCDecay::set_emit_basis: the basis is the periodic convolution's");
  }
  if (on && curve_from_port_) {
    throw std::domain_error(
        "TCSPCDecay::set_emit_basis: a curve from a port has no species to "
        "emit a basis for");
  }
  emit_basis_ = on;
  if (on && !get_output_port(basis_port_key())) {
    add_output_port(basis_port_key(), std::make_shared<GraphPort>(
                        std::vector<double>{0.0}));
  }
  set_valid(false);
}

void TCSPCDecay::set_amplitude_threshold(double relative) {
  if (!(relative >= 0.0)) {
    throw std::domain_error(
        "TCSPCDecay::set_amplitude_threshold: the threshold is relative to "
        "the largest amplitude and cannot be negative");
  }
  amplitude_threshold_ = relative;
  set_valid(false);
}

void TCSPCDecay::evaluate() {
  if (response_from_port_) {
    const std::shared_ptr<GraphPort> port = get_input_port("response");
    if (!port) {
      throw std::domain_error("TCSPCDecay '" + get_name() +
                              "' reads its response from a port it does not have");
    }
    const std::vector<double>& modelled = port->get_values_ref();
    if (modelled != response_) set_response(modelled);
  }
  if (response_.empty()) {
    throw std::domain_error("TCSPCDecay '" + get_name() +
                            "' has no response function");
  }
  // Only when the pairs come from the scalar ports: reading them from the
  // spectrum port is how a node with *no* `a`/`t` ports drives this one, and
  // `build_spectrum` sets the count from what actually arrived.
  if (!curve_from_port_ && !spectrum_from_port_ && n_lifetimes_ <= 0) {
    throw std::domain_error("TCSPCDecay '" + get_name() +
                            "' has no lifetime components");
  }
  if (!curve_from_port_) build_spectrum();

  const int n_points = static_cast<int>(response_.size());

  // The shift, then the unit-sum normalisation -- that order, because the
  // shift zeroes what moves past either end and normalising first would
  // leave the model's area depending on the timeshift.
  const double timeshift = timeshift_port_->get_value();
  // The shift and the renormalisation depend on the response and the
  // timeshift and on nothing else, so an evaluation that moved neither --
  // every Jacobian column on an amplitude or a lifetime, which is most of
  // them -- can keep the copy it already has. A NaN timeshift compares
  // unequal to itself and rebuilds, which is the safe direction.
  internal::ResponsePreparation preparation = response_preparation_;
  if (const std::shared_ptr<GraphPort> bg =
          get_input_port(response_background_port_key())) {
    preparation.active = true;
    preparation.background = bg->get_value();
  }
  const bool irf_is_current =
      irf_valid_ && irf_epoch_ == response_epoch_ &&
      irf_timeshift_ == timeshift && irf_preparation_ == preparation &&
      irf_.size() == static_cast<std::size_t>(n_points);
  if (!irf_is_current) {
  internal::prepare_response(response_, preparation, timeshift, cleaned_,
                             shifted_, irf_);
  irf_preparation_ = preparation;
  irf_epoch_ = response_epoch_;
  irf_timeshift_ = timeshift;
  irf_valid_ = true;
  }
  const std::vector<double>& irf = irf_;
  // The response as the convolution uses it (shifted, background taken off,
  // unit sum), for a node that builds on this one's basis: a MaxEnt inversion
  // subtracts the scatter it carries.
  if (const std::shared_ptr<GraphPort> prepared = get_output_port("prepared_response")) {
    prepared->set_sanitize(false);
    prepared->set_value_vector(irf);
  }

  if (curve_from_port_) {
    const std::vector<double>& given = curve_port_->get_values_ref();
    if (given.size() != static_cast<std::size_t>(n_points)) {
      std::ostringstream m;
      m << "TCSPCDecay '" << get_name() << "' takes its curve from a port, "
        << "which holds " << given.size() << " values against a response of "
        << n_points;
      throw std::domain_error(m.str());
    }
    curve_.assign(given.begin(), given.end());
  } else {
    curve_.assign(static_cast<std::size_t>(n_points), 0.0);

    // tttrlib's periodic reconvolution. Its stop arguments are *inclusive*
    // indices, so the last valid one is `n_points - 1`; passing a length reads
    // and writes one element past both buffers.
    const int last = n_points - 1;
    int convolution_stop =
        convolution_stop_ < 0 ? last : std::min(convolution_stop_, last);
    int stop = stop_ < 0 ? last : std::min(stop_, last);
    convolution_stop = std::max(0, convolution_stop);
    stop = std::max(0, stop);
    // With `t e^{-t/tau}` components the spectrum splits by kind; without, the
    // plain spectrum is the whole of it and nothing below changes.
    const std::vector<double>* plain = &spectrum_;
    int n_plain = n_active_;
    if (!kinds_.empty()) {
      plain_spectrum_.clear();
      t_spectrum_.clear();
      for (int c = 0; c < n_active_; ++c) {
        std::vector<double>& to = kinds_[static_cast<std::size_t>(c)] != 0.0 ? t_spectrum_ : plain_spectrum_;
        to.push_back(spectrum_[static_cast<std::size_t>(2 * c)]);
        to.push_back(spectrum_[static_cast<std::size_t>(2 * c + 1)]);
      }
      plain = &plain_spectrum_;
      n_plain = static_cast<int>(plain_spectrum_.size() / 2);
    }
    if (!convolve_) {
      // No response to convolve with: the ideal decay on the channel axis,
      // ChiSurf's `decay_without_irf`. Under periodic excitation the unrelaxed
      // decay of the earlier pulses adds the geometric tail
      // 1 / (1 - exp(-period / tau)) -- the factor the periodic kernel applies.
      for (int s = 0; s < n_plain; ++s) {
        const double tau = (*plain)[static_cast<std::size_t>(2 * s + 1)];
        double amplitude = (*plain)[static_cast<std::size_t>(2 * s)];
        if (periodic_ && tau > 0.0) amplitude /= (1.0 - std::exp(-period_ / tau));
        for (int i = 0; i < n_points; ++i) {
          curve_[static_cast<std::size_t>(i)] += amplitude * std::exp(-(i * dt_) / tau);
        }
      }
    } else if (periodic_) {
      fconv_per_cs_ad<double>(curve_.data(), plain->data(), irf.data(),
                              n_plain, stop, n_points, period_,
                              convolution_stop, dt_);
    } else {
      // A single excitation: tttrlib's non-periodic recursion, whose stop is
      // one past the last channel.
      fconv_ad<double>(curve_.data(), plain->data(), irf.data(), n_plain,
                       0, std::min(convolution_stop + 1, n_points), dt_);
    }
    if (!kinds_.empty()) {
      add_t_exponential_components(curve_.data(), t_spectrum_, irf.data(), n_points, convolve_,
                                   periodic_, period_, stop, convolution_stop, dt_);
    }

    // The basis: each species reconvolved on its own, with unit amplitude, in
    // the order the input spectrum gave them.
    //
    // This is NOT the same work as the summed call above, though it recurses
    // over the same species and this comment claimed it was until it was
    // measured: 7.5x the curve at K = 33 over 1 563 channels. Each call here
    // passes `numexp = 1`, which is below FCONV_AD_BLOCK_MIN, so it takes the
    // serial recursion while the summed call takes the 8-way blocked body --
    // the basis loses the blocking entirely. Closing that needs a kernel that
    // writes K columns instead of accumulating them, which belongs in
    // tttrlib's DecayConvolution.h (this file's copy is vendored and pinned
    // byte-identical by test/test_vendored_headers.py), not here.
    if (emit_basis_) {
      const std::size_t n_species = static_cast<std::size_t>(n_active_);
      const std::size_t n_bins = static_cast<std::size_t>(n_points);
      // Species-major first: the kernel writes each column into its own
      // contiguous run, so there is no separate column buffer and no copy out
      // of one. Writing the port's bins x species layout directly would put
      // consecutive writes `n_species` doubles apart -- a different cache line
      // every time, and at 33 species over 1563 channels that is 51k of them
      // against a buffer far larger than L1.
      basis_columns_.assign(n_bins * n_species, 0.0);
      double single[2];
      for (std::size_t s = 0; s < n_species; ++s) {
        single[0] = 1.0;
        single[1] = spectrum_[2 * s + 1];
        if (!kinds_.empty() && kinds_[s] != 0.0) {
          add_t_exponential_components(basis_columns_.data() + s * n_bins,
                                       std::vector<double>(single, single + 2), irf.data(),
                                       n_points, true, true, period_, stop, convolution_stop, dt_);
        } else {
          fconv_per_cs_ad<double>(basis_columns_.data() + s * n_bins, single,
                                  irf.data(), 1, stop, n_points, period_,
                                  convolution_stop, dt_);
        }
      }
      // Then one blocked transpose into the contract the port promises. Tiled
      // because the naive loop is strided on whichever side it does not walk,
      // which is the cost this is here to avoid.
      basis_.assign(n_bins * n_species, 0.0);
      const std::size_t tile = 32;
      for (std::size_t b0 = 0; b0 < n_bins; b0 += tile) {
        const std::size_t b1 = std::min(b0 + tile, n_bins);
        for (std::size_t s0 = 0; s0 < n_species; s0 += tile) {
          const std::size_t s1 = std::min(s0 + tile, n_species);
          for (std::size_t b = b0; b < b1; ++b) {
            for (std::size_t s = s0; s < s1; ++s) {
              basis_[b * n_species + s] = basis_columns_[s * n_bins + b];
            }
          }
        }
      }
      const std::shared_ptr<GraphPort> bp = get_output_port(basis_port_key());
      if (!bp) {
        throw std::domain_error(
            "TCSPCDecay '" + get_name() +
            "' emits the basis but has no '" + basis_port_key() + "' port");
      }
      bp->set_sanitize(false);
      bp->set_value_vector(basis_);
    }
  }

  // The instrument: one implementation, internal/TCSPCInstrument.h, shared
  // with the Bayesian decay model. Scatter, the background pattern and the
  // constant background are fractions of the fluorescence total sum(F);
  // n0 is the counts per unit of F. ChiSurf's order: the components, Coates
  // pile-up on them, the background after pile-up, the scale, then the DNL
  // table on the finished curve.
  const std::size_t n = static_cast<std::size_t>(n_points);
  internal::TCSPCInstrumentSettings settings;
  internal::TCSPCInstrumentParameters<double> parameters;
  parameters.scale = 1.0;  // the scale is applied below, where autoscaling decides it
  // In counts (ChiSurf's units) the scatter is converted to the stage's
  // fraction at this evaluation's total, so a held count stays that count as
  // the decay changes; the background is added in counts after the scale.
  const bool counts = instrument_units_counts_;
  const double background_counts = counts ? background_port_->get_value() : 0.0;
  const double scatter_counts = counts ? scatter_port_->get_value() : 0.0;
  const double unit_total = internal::tcspc_instrument_total(curve_.data(), n);
  if (counts) {
    parameters.scatter =
        internal::tcspc_instrument_fractions_from_absolute(1.0, scatter_counts, 0.0, unit_total, n).scatter;
  } else {
    parameters.scatter = scatter_port_->get_value();
    parameters.background = background_port_->get_value();
  }
  settings.response = irf.data();
  flat_.assign(n, n > 0 ? 1.0 / static_cast<double>(n) : 0.0);
  settings.flat = flat_.data();

  // A measured background pattern: its shape (shifted with the response when
  // asked, unit sum) and its fraction -- given on the `pattern` port, or, as
  // ChiSurf derives it, the counts the measurement times assign it over the
  // fluorescence counts the data leave.
  if (!background_pattern_.empty()) {
    if (background_pattern_.size() != curve_.size()) {
      throw std::domain_error(
          "TCSPCDecay '" + get_name() +
          "' adds a background pattern, which must be as long as the response");
    }
    if (data_y_.size() != curve_.size()) {
      throw std::domain_error(
          "TCSPCDecay '" + get_name() +
          "' adds a background pattern, which needs data as long as the response");
    }
    pattern_shape_.assign(background_pattern_.begin(), background_pattern_.end());
    if (shift_background_with_response_ && timeshift != 0.0) {
      shift_lamp_ad<double>(pattern_shape_.data(), background_pattern_.data(), -timeshift,
                            n_points, 0.0);
    }
    double measured = 0.0, pattern_total = 0.0, recorded = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      measured += data_y_[i];
      pattern_total += pattern_shape_[i];
      recorded += background_pattern_[i];
    }
    if (pattern_total > 0.0) {
      for (double& v : pattern_shape_) v /= pattern_total;
      settings.pattern = pattern_shape_.data();
    }
    if (const std::shared_ptr<GraphPort> given = get_input_port("pattern")) {
      parameters.pattern = given->get_value();
    } else {
      const double n_background = recorded / t_background_ * t_decay_;
      const double n_fluorescence = std::max(measured - n_background, 1.0);
      if (counts) {
        // ChiSurf rescaled the model (scatter included) to the n_fl counts the
        // pattern leaves and added the pattern as n_bg counts, both before n0.
        const double model_total = unit_total + scatter_counts;
        parameters.scale = model_total != 0.0 ? n_fluorescence / model_total : 0.0;
        const double scaled_total = parameters.scale * unit_total;
        parameters.pattern = scaled_total != 0.0 ? n_background / scaled_total : 0.0;
      } else {
        parameters.pattern = n_background / n_fluorescence;
      }
    }
  }

  if (pile_up_) {
    if (data_y_.size() != response_.size()) {
      throw std::domain_error(
          "TCSPCDecay '" + get_name() +
          "' corrects pile-up, which needs data as long as the response");
    }
    if (period_ <= 0.0) {
      throw std::domain_error("TCSPCDecay '" + get_name() +
                              "' corrects pile-up, which needs a period");
    }
    // Lifetimes and the period are in ns, so the repetition rate in MHz is
    // 1000 / period -- the inverse of how chisurf builds the period.
    settings.pile_up_data = data_y_.data();
    settings.pile_up_n_data = static_cast<int>(data_y_.size());
    settings.repetition_rate_mhz = 1000.0 / period_;
    settings.dead_time_ns = pile_up_dead_time_ns_;
    settings.measurement_time_s = pile_up_measurement_time_s_;
  }

  instrument_background_.assign(n, 0.0);
  const double fluorescence_total = internal::tcspc_instrument_components(
      curve_.data(), n, settings, parameters, instrument_background_.data());
  internal::tcspc_instrument_detection(curve_.data(), n, settings,
                                       instrument_background_.data());
  if (const std::shared_ptr<GraphPort> total = get_output_port("fluorescence_total")) {
    total->set_value(fluorescence_total);
  }

  if (autoscale_) {
    if (data_y_.size() != response_.size()) {
      throw std::domain_error(
          "TCSPCDecay '" + get_name() +
          "' autoscales, which needs data as long as the response");
    }
    int begin = std::max(0, scale_start_);
    int end = scale_stop_ < 0 ? n_points : std::min(scale_stop_, n_points);
    end = std::min(end, static_cast<int>(data_y_.size()));
    if (end < begin) end = begin;
    // Every part of the stage is proportional to the scale, background
    // included, so the scale is the least-squares factor of the unit curve;
    // a background in counts is not, and is taken off the data first.
    n0_ = rescale_factor(curve_, data_y_, data_ey_, background_counts, begin, end);
    // Published, because a fit that autoscales still has to report the
    // amplitude it settled on. A port that follows another publishes to what
    // it follows.
    GraphPort* published = n0_port_;
    while (published->get_link()) published = published->get_link().get();
    published->set_value(n0_);
  } else {
    n0_ = n0_port_->get_value();
  }
  for (double& v : curve_) v = v * n0_ + background_counts;

  // DNL: the measured channel-width table multiplies the finished curve.
  if (!lin_table_.empty()) {
    if (lin_table_.size() != curve_.size()) {
      throw std::domain_error(
          "TCSPCDecay '" + get_name() +
          "' linearizes, which needs a table as long as the response");
    }
    for (std::size_t i = 0; i < curve_.size(); ++i) {
      curve_[i] *= lin_table_[i];
    }
  }

  // A negative expected count is not a decay. ChiSurf clamps, and the clamp
  // is part of the objective rather than a cosmetic step: without it a fit
  // can trade a negative channel against a positive one.
  //
  // `v < 0.0`, not `!(v > 0.0)`: the second spelling also catches NaN and
  // would turn it into a *zero*, which in a fit reads as a good fit near
  // zero -- the same trap the port sanitiser sets, one line earlier. NaN
  // compares false against everything, so this leaves it alone, which is
  // what `np.maximum` does too.
  for (double& v : curve_) {
    if (v < 0.0) v = 0.0;
  }

  // The spectrum the curve was built from, for a caller that shows or
  // summarises it (a lifetime distribution, averaged lifetimes) without
  // rebuilding it in its own words. Published only where a port asks for it.
  if (const std::shared_ptr<GraphPort> published = get_output_port("spectrum")) {
    published->set_value_vector(curve_from_port_ ? std::vector<double>()
                                                 : spectrum_);
  }
  const std::shared_ptr<GraphPort> out = get_output_port(get_name());
  if (!out) {
    throw std::domain_error(
        "TCSPCDecay '" + get_name() +
        "' writes its curve to the output port keyed by its own name, which "
        "this node does not have");
  }
  // Fit transport: a NaN must survive rather than be floored to `tiny`,
  // which in a fit reads as a *good* fit near zero. See FitChiSquared::update.
  out->set_sanitize(false);
  out->set_value_vector(curve_);
  set_valid(true);
}

std::string TCSPCDecay::describe() const {
  std::ostringstream out;
  out << "components     : " << n_lifetimes_ << "\n"
      << "channels       : " << response_.size() << "\n"
      << "dt / period    : " << dt_ << " / " << period_ << "\n"
      << "autoscale      : " << (autoscale_ ? "yes" : "no") << "\n"
      << "n0             : " << n0_ << "\n";
  return out.str();
}

std::string TCSPCDecay::get_node_type() const { return "TCSPCDecay"; }

void TCSPCDecay::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  // The instrument response and the measured decay are bound at runtime;
  // what a description fixes is the shape of the model over them.
  if (config.has("number_of_lifetimes")) {
    set_number_of_lifetimes(config.get_int("number_of_lifetimes"));
  }
  if (config.has("timing")) {
    const std::vector<double> timing = config.get_doubles("timing");
    if (timing.size() != 2) {
      throw std::domain_error(
          "node type 'TCSPCDecay': setting 'timing' must be "
          "[channel_width, excitation_period]");
    }
    set_timing(timing[0], timing[1]);
  }
  if (config.has("convolution_range")) {
    const std::vector<int> range = config.get_ints("convolution_range");
    if (range.size() != 2) {
      throw std::domain_error(
          "node type 'TCSPCDecay': setting 'convolution_range' must be "
          "[convolution_stop, stop]");
    }
    set_convolution_range(range[0], range[1]);
  }
  if (config.has("scale_range")) {
    const std::vector<int> range = config.get_ints("scale_range");
    if (range.size() != 2) {
      throw std::domain_error(
          "node type 'TCSPCDecay': setting 'scale_range' must be "
          "[start, stop]");
    }
    set_scale_range(range[0], range[1]);
  }
  if (config.has("normalize_amplitudes")) {
    set_normalize_amplitudes(config.get_bool("normalize_amplitudes"));
  }
  if (config.has("absolute_amplitudes")) {
    set_absolute_amplitudes(config.get_bool("absolute_amplitudes"));
  }
  if (config.has("autoscale")) set_autoscale(config.get_bool("autoscale"));
  if (config.has("instrument_units")) {
    set_instrument_units(config.get_string("instrument_units"));
  }
  if (config.has("pile_up")) set_pile_up(config.get_bool("pile_up"));
  if (config.has("pile_up_parameters")) {
    const std::vector<double> p = config.get_doubles("pile_up_parameters");
    if (p.size() != 2) {
      throw std::domain_error(
          "node type 'TCSPCDecay': setting 'pile_up_parameters' must be "
          "[dead_time_ns, measurement_time_s]");
    }
    set_pile_up_parameters(p[0], p[1]);
  }
  if (config.has("amplitude_threshold")) {
    set_amplitude_threshold(config.get_double("amplitude_threshold"));
  }
  if (config.has("spectrum_from_port")) {
    set_spectrum_from_port(config.get_bool("spectrum_from_port"));
  }
  if (config.has("convolution_mode")) {
    set_convolution_mode(config.get_string("convolution_mode"));
  }
  if (config.has("convolve")) set_convolve(config.get_bool("convolve"));
  if (config.has("response_from_port")) {
    set_response_from_port(config.get_bool("response_from_port"));
  }
  if (config.has("linearization_smoothing")) {
    // [window_length, x_min, x_max, reverse]; the window's name separately.
    const std::vector<double> v = config.get_doubles("linearization_smoothing");
    if (v.size() != 4) {
      throw std::domain_error(
          "node type 'TCSPCDecay': setting 'linearization_smoothing' must be "
          "[window_length, x_min, x_max, reverse]");
    }
    const std::string window = config.has("linearization_window")
                                   ? config.get_string("linearization_window")
                                   : lin_window_type_;
    set_linearization_smoothing(static_cast<int>(v[0]), window,
                                static_cast<int>(v[1]), static_cast<int>(v[2]),
                                v[3] != 0.0);
  } else if (config.has("linearization_window")) {
    set_linearization_smoothing(lin_window_length_,
                                config.get_string("linearization_window"),
                                lin_x_min_, lin_x_max_, lin_reverse_);
  }
  if (config.has("background_times")) {
    const std::vector<double> times = config.get_doubles("background_times");
    if (times.size() != 2) {
      throw std::domain_error(
          "node type 'TCSPCDecay': setting 'background_times' must be "
          "[t_background, t_decay]");
    }
    set_background_times(times[0], times[1]);
  }
  if (config.has("shift_background_with_response")) {
    set_shift_background_with_response(config.get_bool("shift_background_with_response"));
  }
  if (config.has("response_range")) {
    const std::vector<int> range = config.get_ints("response_range");
    if (range.size() != 2) {
      throw std::domain_error(
          "node type 'TCSPCDecay': setting 'response_range' must be "
          "[start, stop]");
    }
    set_response_range(range[0], range[1]);
  }
  if (config.has("curve_from_port")) {
    set_curve_from_port(config.get_bool("curve_from_port"));
  }
  // Last: the basis is only defined once the convolution mode is known.
  if (config.has("emit_basis")) set_emit_basis(config.get_bool("emit_basis"));
  config.apply_common(*this);
  config.require_all_used();
}

void TCSPCDecay::bind_dataset(const std::string& role,
                              const FitDataset& dataset) {
  // The response is an instrument function rather than a signal, but it is
  // measured too, so it arrives the same way. The decay is not compared to
  // anything itself -- that is the objective's job -- so it takes no "data".
  if (role == "response") {
    set_response(dataset.get_values());
    return;
  }
  if (role == "data") {
    // Used only when the node autoscales or corrects for pile-up; harmless
    // otherwise, since FitChiSquared holds its own copy for the misfit. The
    // errors come from the dataset's variance, so a caller passing measured
    // errors stores them as a variance and they arrive unchanged.
    const std::vector<double>& values = dataset.get_values();
    double* view = nullptr;
    int n = 0;
    dataset.variance(values, &view, &n);
    std::vector<double> errors(static_cast<std::size_t>(std::max(n, 0)));
    for (int i = 0; i < n; ++i) errors[i] = std::sqrt(std::max(view[i], 0.0));
    std::free(view);
    set_data(values, errors);
    return;
  }
  if (role == "linearization_curve") {
    // The measurement of uncorrelated light itself; the table is derived.
    set_linearization_curve(dataset.get_values());
    return;
  }
  if (role == "background_pattern") {
    // A measured background decay; how much of it the data holds follows
    // from the two measurement times.
    set_background_pattern(dataset.get_values());
    return;
  }
  if (role == "linearization") {
    // A measured table of each channel's effective width; configuration, not
    // a fitted port, so it arrives the way the response does.
    set_linearization(dataset.get_values());
    return;
  }
  throw std::domain_error("node type 'TCSPCDecay' has no role '" + role +
                          "'; it takes 'response', 'data', 'linearization', 'linearization_curve' or 'background_pattern'");
}

void TCSPCDecay::set_instrument_units(const std::string& units) {
  if (units != "counts" && units != "fractions") {
    throw std::domain_error("TCSPCDecay '" + get_name() + "': instrument units are 'counts' or 'fractions', not '" +
                            units + "'");
  }
  instrument_units_counts_ = units == "counts";
  set_valid(false);
}

std::vector<double> tcspc_fractions_from_absolute(double n0, double scatter_absolute,
                                                  double background_absolute,
                                                  double fluorescence_total, int n_channels) {
  const internal::TCSPCInstrumentParameters<double> p =
      internal::tcspc_instrument_fractions_from_absolute(
          n0, scatter_absolute, background_absolute, fluorescence_total,
          static_cast<std::size_t>(std::max(n_channels, 0)));
  return {p.scale, p.scatter, p.pattern, p.background};
}

IMPBFF_END_NAMESPACE
