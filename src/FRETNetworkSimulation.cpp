/**
 *  \file FRETNetworkSimulation.cpp
 *  \brief Simulated photons of a FRET network measurement.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/FRETNetworkSimulation.h>
#include <IMP/bff/internal/pcg_random.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

namespace {

double uniform(pcg32& rng) {
  // (0, 1): never 0, so a log of it is finite.
  return (static_cast<double>(rng()) + 0.5) / 4294967296.0;
}

int draw(pcg32& rng, const std::vector<double>& weights, double total) {
  double u = uniform(rng) * total;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    u -= weights[i];
    if (u <= 0.0) return static_cast<int>(i);
  }
  for (std::size_t i = weights.size(); i-- > 0;)
    if (weights[i] > 0.0) return static_cast<int>(i);
  return 0;
}

struct Simulated {
  std::vector<double> t;
  std::vector<int> c, b, state, start, stop;
};

Simulated simulate(const FRETHiddenProcess& process, const FRETMeasurement& m,
                   const FRETSimulationOptions& o) {
  if (o.n_molecules < 0 || !(o.duration > 0.0)) {
    throw std::invalid_argument("FRETSimulationOptions: molecules >= 0 and duration > 0");
  }
  const FRETInstrument& ins = m.get_instrument();
  const int n = m.get_n_states(process);
  const int C = ins.get_n_channels();
  const int nb = std::max(ins.get_n_bins(), 1);
  const std::vector<double> K = m.get_generator(process);  // K[target, source]
  const std::vector<double> lambda = m.get_detection_rates(process);  // C x n
  const std::vector<double> emission = m.get_emission(process);  // C x nb x n
  const std::vector<double> start =
      o.product_start ? m.get_start(process) : m.get_joint_stationary(process);

  std::vector<double> background(C), signal(static_cast<std::size_t>(C) * n);
  std::vector<std::vector<double> > bg_density(C);
  for (int c = 0; c < C; ++c) {
    background[c] = ins.get_background(c);
    bg_density[c] = ins.get_background_density(c);
    for (int s = 0; s < n; ++s)
      signal[c * n + s] = std::max(lambda[c * n + s] - background[c], 0.0);
  }
  // Per state: its exit rates, and each channel's signal microtime weights
  // (the emission table less the background's share).
  std::vector<double> exit(n, 0.0);
  for (int s = 0; s < n; ++s)
    for (int t = 0; t < n; ++t)
      if (t != s) exit[s] += std::max(K[t * n + s], 0.0);

  pcg32 rng(static_cast<std::uint64_t>(o.seed), 0x46524554ULL);
  Simulated out;
  double t0 = 0.0;
  const double w = o.duration / 4.0;
  for (int mol = 0; mol < o.n_molecules; ++mol) {
    out.start.push_back(static_cast<int>(out.t.size()));
    double total_start = 0.0;
    for (double v : start) total_start += v;
    int s = draw(rng, start, total_start);
    double t = 0.0;
    while (true) {
      double signal_total = 0.0;
      for (int c = 0; c < C; ++c) signal_total += signal[c * n + s];
      double bg_total = 0.0;
      for (int c = 0; c < C; ++c) bg_total += background[c];
      const double rate = exit[s] + signal_total + bg_total;
      if (!(rate > 0.0)) break;
      t += -std::log(uniform(rng)) / rate;
      if (t >= o.duration) break;
      double u = uniform(rng) * rate;
      if (u < exit[s]) {
        std::vector<double> to(n, 0.0);
        for (int x = 0; x < n; ++x)
          if (x != s) to[x] = std::max(K[x * n + s], 0.0);
        s = draw(rng, to, exit[s]);
        continue;
      }
      u -= exit[s];
      if (u < signal_total) {
        // Thinning: the focus lowers the rate below the state's own.
        if (o.focus) {
          const double z = (t - 0.5 * o.duration) / w;
          if (uniform(rng) > std::exp(-2.0 * z * z)) continue;
        }
        std::vector<double> per_channel(C);
        for (int c = 0; c < C; ++c) per_channel[c] = signal[c * n + s];
        const int c = draw(rng, per_channel, signal_total);
        int bin = 0;
        if (nb > 1) {
          std::vector<double> bins(nb);
          double sum = 0.0;
          for (int k = 0; k < nb; ++k) {
            const double e = emission[(static_cast<std::size_t>(c) * nb + k) * n + s] -
                             background[c] * bg_density[c][k];
            bins[k] = std::max(e, 0.0);
            sum += bins[k];
          }
          bin = draw(rng, bins, sum);
        }
        out.t.push_back(t0 + t);
        out.c.push_back(c);
        out.b.push_back(bin);
        out.state.push_back(s);
      } else {
        const int c = draw(rng, background, bg_total);
        const int bin = nb > 1 ? draw(rng, bg_density[c], 1.0) : 0;
        out.t.push_back(t0 + t);
        out.c.push_back(c);
        out.b.push_back(bin);
        out.state.push_back(s);
      }
    }
    out.stop.push_back(static_cast<int>(out.t.size()) - 1);
    if (out.stop.back() < out.start.back()) {  // no photon: no segment
      out.start.pop_back();
      out.stop.pop_back();
    }
    // Molecules apart by more than any burst-search gap.
    t0 += 2.0 * o.duration;
  }
  if (nb <= 1) out.b.clear();
  return out;
}

}  // namespace

FRETPhotonData simulate_fret_measurement(const FRETHiddenProcess& process,
                                         const FRETMeasurement& measurement,
                                         const FRETSimulationOptions& options) {
  Simulated s = simulate(process, measurement, options);
  return FRETPhotonData(s.t, s.c, s.b, s.start, s.stop);
}

std::vector<int> simulated_fret_states(const FRETHiddenProcess& process,
                                       const FRETMeasurement& measurement,
                                       const FRETSimulationOptions& options) {
  return simulate(process, measurement, options).state;
}

FRETPhotonData select_bursts(const FRETPhotonData& data, double max_gap, int min_photons) {
  const std::vector<double>& t = data.get_macrotimes();
  std::vector<int> starts, stops;
  for (int g = 0; g < data.get_n_segments(); ++g) {
    const int first = data.get_segment_starts()[g], last = data.get_segment_stops()[g];
    int run = first;
    for (int i = first + 1; i <= last + 1; ++i) {
      if (i <= last && t[i] - t[i - 1] <= max_gap) continue;
      if (i - run >= min_photons) {
        starts.push_back(run);
        stops.push_back(i - 1);
      }
      run = i;
    }
  }
  return FRETPhotonData(t, data.get_channels(), data.get_microtimes(), starts, stops);
}

IMPBFF_END_NAMESPACE
