/**
 * \file PhotophysicsTransferKinetics.cpp
 * \brief Excited-state kinetics between two chromophores that transfer both ways.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/PhotophysicsTransferKinetics.h>
#include <IMP/bff/internal/GradVec.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

std::vector<double> transfer_pair_components(double tau_a, double tau_b, double k_ab,
                                             double k_ba, double p_a, double p_b,
                                             double m_a, double m_b, int mode,
                                             double eps) {
  internal::TransferComponent<double> parts[2];
  internal::transfer_pair_components_t(tau_a, tau_b, k_ab, k_ba, p_a, p_b, m_a, m_b, mode,
                                       eps, parts);
  return {parts[0].amplitude, parts[0].rate, static_cast<double>(parts[0].kind),
          parts[1].amplitude, parts[1].rate, static_cast<double>(parts[1].kind)};
}

std::vector<double> transfer_pair_components_jacobian(double tau_a, double tau_b,
                                                      double k_ab, double k_ba, double p_a,
                                                      double p_b, double m_a, double m_b,
                                                      int mode, double eps) {
  using G = tttrlib::GradVec<8>;
  using D = tttrlib::Dual<G>;
  const double at[8] = {tau_a, tau_b, k_ab, k_ba, p_a, p_b, m_a, m_b};
  D x[8];
  for (int i = 0; i < 8; ++i) x[i] = D::variable(at[i], G::Unit(i));
  internal::TransferComponent<D> parts[2];
  internal::transfer_pair_components_t(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], mode,
                                       eps, parts);
  std::vector<double> jacobian(6 * 8, 0.0);
  for (int c = 0; c < 2; ++c) {
    for (int j = 0; j < 8; ++j) {
      jacobian[(3 * c) * 8 + j] = parts[c].amplitude.grad[j];
      jacobian[(3 * c + 1) * 8 + j] = parts[c].rate.grad[j];
    }
  }
  return jacobian;
}

std::vector<double> transfer_pair_decay(const std::vector<double>& time, double tau_a,
                                        double tau_b, double k_ab, double k_ba, double p_a,
                                        double p_b, double m_a, double m_b) {
  std::vector<double> out(time.size());
  for (std::size_t i = 0; i < time.size(); ++i) {
    out[i] = internal::transfer_pair_decay_t(time[i], tau_a, tau_b, k_ab, k_ba, p_a, p_b,
                                             m_a, m_b);
  }
  return out;
}

std::vector<double> transfer_populations_from_pure_fractions(double pure_a, double pure_b) {
  const double denominator = 1.0 - pure_a * pure_b;
  const double pi_a = (pure_a * (1.0 - pure_b)) / denominator;
  const double pi_b = (pure_b * (1.0 - pure_a)) / denominator;
  return {pi_a, pi_b, 1.0 - pi_a - pi_b};
}

std::vector<double> transfer_kinetics_spectrum(
    const std::vector<double>& spectrum_a, const std::vector<double>& spectrum_b,
    const std::vector<double>& rates, double f_ab, double f_ba,
    const std::vector<double>& populations, const PhotophysicsCrosstalkMatrix& excitation,
    const PhotophysicsCrosstalkMatrix& emission, const std::string& pulse,
    const std::string& channel, const std::string& chromophore_a,
    const std::string& chromophore_b, int mode, double eps) {
  if (spectrum_a.size() % 2 || spectrum_b.size() % 2 || rates.size() % 2) {
    throw std::domain_error(
        "transfer_kinetics_spectrum: spectra and rates are interleaved pairs");
  }
  if (populations.size() != 3) {
    throw std::domain_error(
        "transfer_kinetics_spectrum: populations are (pi_A, pi_B, pi_AB)");
  }
  // The light path, from the one crosstalk definition.
  const double p_a = excitation.get(pulse, chromophore_a);
  const double p_b = excitation.get(pulse, chromophore_b);
  const double m_a = emission.get(chromophore_a, channel);
  const double m_b = emission.get(chromophore_b, channel);
  const std::size_t n_a = spectrum_a.size() / 2;
  const std::size_t n_b = spectrum_b.size() / 2;
  const std::size_t n_rates = rates.size() / 2;
  std::vector<double> out;
  out.reserve(2 * n_rates * (2 * n_a * n_b + n_a + n_b));
  const auto push = [&out](double amplitude, double rate) {
    out.push_back(amplitude);
    out.push_back(rate != 0.0 ? 1.0 / rate : 0.0);
  };
  for (std::size_t r = 0; r < n_rates; ++r) {
    const double w = rates[2 * r];
    const double k = rates[2 * r + 1];
    for (std::size_t i = 0; i < n_a; ++i) {
      const double c_a = spectrum_a[2 * i];
      const double tau_a = spectrum_a[2 * i + 1];
      for (std::size_t j = 0; j < n_b; ++j) {
        const double c_b = spectrum_b[2 * j];
        const double tau_b = spectrum_b[2 * j + 1];
        if (c_a == 0.0 || c_b == 0.0 || tau_a == 0.0 || tau_b == 0.0) {
          // An empty component carries nothing; the layout is kept.
          push(0.0, 0.0);
          push(0.0, 0.0);
          continue;
        }
        internal::TransferComponent<double> parts[2];
        internal::transfer_pair_components_t(tau_a, tau_b, k * f_ab, k * f_ba, p_a, p_b,
                                             m_a, m_b, mode, eps, parts);
        const double weight = w * populations[2] * c_a * c_b;
        if (parts[1].kind == TRANSFER_T_EXPONENTIAL) {
          // A sum of exponentials cannot hold t e^{-kt}: split it by eps |k|,
          // c t e^{-kt} ~ c (e^{-(k - h)t} - e^{-(k + h)t}) / (2h).
          const double rate = parts[0].rate;
          const double h = std::max(eps * std::fabs(rate), 1e-300);
          const double half = parts[1].amplitude / (2.0 * h);
          push(weight * (parts[0].amplitude + half), rate - h);
          push(-weight * half, rate + h);
        } else {
          push(weight * parts[0].amplitude, parts[0].rate);
          push(weight * parts[1].amplitude, parts[1].rate);
        }
      }
    }
    for (std::size_t i = 0; i < n_a; ++i) {
      out.push_back(w * m_a * p_a * populations[0] * spectrum_a[2 * i]);
      out.push_back(spectrum_a[2 * i + 1]);
    }
    for (std::size_t j = 0; j < n_b; ++j) {
      out.push_back(w * m_b * p_b * populations[1] * spectrum_b[2 * j]);
      out.push_back(spectrum_b[2 * j + 1]);
    }
  }
  return out;
}

IMPBFF_END_NAMESPACE
