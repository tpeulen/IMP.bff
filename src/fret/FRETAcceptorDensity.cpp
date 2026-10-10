/**
 * \file FRETAcceptorDensity.cpp
 * \brief A donor quenched by acceptors at a density rather than at a distance.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/FRETAcceptorDensity.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

namespace {
int checked_dimension(int dimension) {
  if (dimension < 1 || dimension > 3) {
    throw std::domain_error("acceptor density: dimension must be 1, 2 or 3, got " +
                            std::to_string(dimension));
  }
  return dimension;
}
}  // namespace

double acceptor_characteristic_density(double forster_radius, int dimension) {
  const int d = checked_dimension(dimension);
  if (!(forster_radius > 0.0)) {
    throw std::domain_error("acceptor_characteristic_density: R0 must be positive");
  }
  const double r0 = forster_radius;
  if (d == 3) return 1.0 / (4.0 / 3.0 * M_PI * r0 * r0 * r0);
  if (d == 2) return 1.0 / (M_PI * r0 * r0);
  return 1.0 / (2.0 * r0);
}

double acceptor_reduced_density(double c_over_c0, int dimension) {
  const int d = checked_dimension(dimension);
  if (c_over_c0 < 0.0) {
    throw std::domain_error("acceptor_reduced_density: C/C0 must be non-negative");
  }
  return 0.5 * std::tgamma(1.0 - d / 6.0) * c_over_c0;
}

std::vector<double> acceptor_quenching_factor(const std::vector<double>& time, double tau0,
                                              double c_over_c0, int dimension) {
  const int d = checked_dimension(dimension);
  if (!(tau0 > 0.0)) throw std::domain_error("acceptor_quenching_factor: tau0 must be positive");
  const double eta = acceptor_reduced_density(c_over_c0, d);
  std::vector<double> out(time.size());
  for (std::size_t i = 0; i < time.size(); ++i) {
    if (time[i] < 0.0) {
      throw std::domain_error("acceptor_quenching_factor: time must be non-negative");
    }
    out[i] = std::exp(-2.0 * eta * std::pow(time[i] / tau0, d / 6.0));
  }
  return out;
}

double acceptor_transfer_efficiency(double c_over_c0, int dimension, int n_points,
                                    double t_max_tau) {
  const int d = checked_dimension(dimension);
  if (n_points < 3) throw std::domain_error("acceptor_transfer_efficiency: n_points >= 3");
  const double eta = acceptor_reduced_density(c_over_c0, d);
  const double step = t_max_tau / (n_points - 1);
  double quenched = 0.0, reference = 0.0;
  double q_prev = 1.0, r_prev = 1.0;
  for (int i = 1; i < n_points; ++i) {
    const double t = i * step;
    const double r = std::exp(-t);
    const double q = std::exp(-t - 2.0 * eta * std::pow(t, d / 6.0));
    quenched += 0.5 * (q + q_prev) * step;
    reference += 0.5 * (r + r_prev) * step;
    q_prev = q;
    r_prev = r;
  }
  return 1.0 - quenched / reference;
}

std::vector<double> acceptor_density_decay(const std::vector<double>& donor_spectrum,
                                           int n_points, double dt, double tau0,
                                           double c_over_c0, int dimension, double period,
                                           int folded_periods) {
  const int d = checked_dimension(dimension);
  if (donor_spectrum.size() % 2) {
    throw std::domain_error("acceptor_density_decay: the donor spectrum is (a, tau) pairs");
  }
  if (!(tau0 > 0.0)) throw std::domain_error("acceptor_density_decay: tau0 must be positive");
  const double eta = acceptor_reduced_density(c_over_c0, d);
  std::vector<double> out(n_points > 0 ? n_points : 0, 0.0);
  const int n_fold = period > 0.0 ? std::max(0, folded_periods) : 0;
  for (int i = 0; i < n_points; ++i) {
    double value = 0.0;
    for (int k = 0; k <= n_fold; ++k) {
      const double t = i * dt + k * period;
      double donor = 0.0;
      for (std::size_t c = 0; c < donor_spectrum.size(); c += 2) {
        const double tau = donor_spectrum[c + 1];
        if (tau > 0.0) donor += donor_spectrum[c] * std::exp(-t / tau);
      }
      value += donor * std::exp(-2.0 * eta * std::pow(t / tau0, d / 6.0));
    }
    out[i] = value;
  }
  return out;
}

IMPBFF_END_NAMESPACE
