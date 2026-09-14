"""PRD-142 step 1: a measured response prepared for a Bayesian decay fit.

`BayesianMeasuredResponse.h` removes a fitted background from a measured
response, shifts it by a fractional number of channels, and builds the periodic
lifetime basis the fit puts amplitudes on -- with the derivatives in the
background and the shift that a scoring step needs. What is checked, and what
would make each check fail:

* the tangents are the derivatives of the forward model (central differences,
  at a shift that is not a whole channel);
* the clamp at zero enters the tangents: dropping its mask from them must break
  the agreement (a check that passes either way would not be testing the clamp);
* every basis column sums to one, so amplitudes are counts;
* the exact periodic kernel is the bin-integrated periodic decay -- against an
  independent integration (the excitation time integrated numerically over the
  channel, the detection interval analytically, every earlier pulse folded in)
  -- and a kernel sampled at channel centres is not;
* a shift by whole channels is a circular roll.

Written 2026-09-14 (ucfret prompt 432).
"""

import pytest

from bayesian_cxx import run_driver

DRIVER = r"""
#include <IMP/bff/BayesianMeasuredResponse.h>
#include <cstdio>
#include <string>
using namespace IMP::bff;

static double max_abs(const std::vector<double>& v) { double m = 0; for (double x : v) m = std::max(m, std::fabs(x)); return m; }

int main() {
  const std::size_t n = 256;
  const BayesianDecayAxis ax{n, 0.064, n};
  std::vector<double> tau;
  for (int c = 0; c < 8; ++c) tau.push_back(0.064 * std::pow(20.0 / 0.064, c / 7.0));
  const BayesianPeriodicKernel kernel(ax, tau);
  // a skewed peak on [10, 90) with a flat floor inside its support, zero elsewhere
  std::vector<double> measured(n, 0.0);
  for (std::size_t i = 10; i < 90; ++i) {
    const double t = double(i) - 20.0;
    measured[i] = 1e4 * std::exp(-0.5 * t * t / 9.0) * (1.0 + 0.8 * std::tanh(t / 3.0)) + 30.0;
  }
  const double b = 0.05, s = 3.37;
  const BayesianResponseBasis at = bayesian_response_basis(kernel, measured, b, s, true);
  auto fd = [&](double hb, double hs, const BayesianResponseOptions& o) {
    auto p = bayesian_response_basis(kernel, measured, b + hb, s + hs, false, o).B;
    auto m = bayesian_response_basis(kernel, measured, b - hb, s - hs, false, o).B;
    std::vector<double> d(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) d[i] = (p[i] - m[i]) / (2.0 * (hb + hs));
    return d;
  };
  BayesianResponseOptions opt;
  const std::vector<double> fd_b = fd(1e-6, 0.0, opt), fd_s = fd(0.0, 1e-5, opt);
  auto rel = [](const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> d(a.size()); for (std::size_t i = 0; i < a.size(); ++i) d[i] = a[i] - b[i];
    return max_abs(d) / std::max(max_abs(b), 1e-300);
  };
  BayesianResponseOptions no_mask; no_mask.clamp_in_tangent = false;
  const BayesianResponseBasis nm = bayesian_response_basis(kernel, measured, b, s, true, no_mask);
  // column sums
  double worst_sum = 0.0;
  for (std::size_t k = 0; k < at.K; ++k) { double c = 0; for (std::size_t i = 0; i < n; ++i) c += at.B[i * at.K + k]; worst_sum = std::max(worst_sum, std::fabs(c - 1.0)); }
  // the kernel: a response in channel 0 only, no background, no shift
  std::vector<double> delta(n, 0.0); delta[0] = 1.0;
  const BayesianResponseBasis kb = bayesian_response_basis(kernel, delta, 0.0, 0.0, false);
  const double P = double(n) * ax.dt, dt = ax.dt;
  double exact_worst = 0.0, naive_worst = 0.0;
  for (std::size_t c = 0; c < tau.size(); ++c) {
    const double t = tau[c];
    std::vector<double> brute(n, 0.0), naive(n, 0.0);
    const int M = 512;                                    // excitation times across channel 0 (midpoint rule)
    const double g = std::exp(-P / t) / (1.0 - std::exp(-P / t));   // every earlier pulse
    for (int e = 0; e < M; ++e) {
      const double u = (e + 0.5) / M * dt;
      for (std::size_t j = 0; j < n; ++j) {
        const double a = j * dt, z = (j + 1) * dt;
        double f = g * (std::exp(-(a - u) / t) - std::exp(-(z - u) / t));
        if (j == 0) f += 1.0 - std::exp(-(z - u) / t);
        else f += std::exp(-(a - u) / t) - std::exp(-(z - u) / t);
        brute[j] += f / M;
      }
    }
    double sn = 0.0;
    for (std::size_t j = 0; j < n; ++j) { naive[j] = std::exp(-(j + 0.5) * dt / t) + g * std::exp(-(j + 0.5) * dt / t); sn += naive[j]; }
    for (double& x : naive) x /= sn;
    std::vector<double> col(n);
    for (std::size_t j = 0; j < n; ++j) col[j] = kb.B[j * kb.K + 1 + c];
    exact_worst = std::max(exact_worst, rel(col, brute));
    naive_worst = std::max(naive_worst, rel(naive, brute));
  }
  // a whole-channel shift is a roll
  const BayesianResponseBasis r0 = bayesian_response_basis(kernel, measured, b, 0.0, false);
  const BayesianResponseBasis r5 = bayesian_response_basis(kernel, measured, b, 5.0, false);
  std::vector<double> rolled(n), shifted(n);
  for (std::size_t i = 0; i < n; ++i) { rolled[(i + 5) % n] = r0.response[i]; shifted[i] = r5.response[i]; }
  std::printf("{\"tangent_background\": %.6e, \"tangent_shift\": %.6e, \"no_mask_shift\": %.6e, \"no_mask_background\": %.6e,"
              " \"column_sum\": %.6e, \"exact_kernel\": %.6e, \"naive_kernel\": %.6e, \"whole_channel_roll\": %.6e}\n",
              rel(at.dB_background, fd_b), rel(at.dB_shift, fd_s), rel(nm.dB_shift, fd_s), rel(nm.dB_background, fd_b),
              worst_sum, exact_worst, naive_worst, rel(shifted, rolled));
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    return run_driver(DRIVER)


def test_tangents_are_the_derivatives(result):
    assert result["tangent_background"] < 1e-7, result
    assert result["tangent_shift"] < 1e-7, result


def test_the_clamp_enters_the_tangents(result):
    # the negative control: the same comparison without the clamp's mask must fail
    assert max(result["no_mask_shift"], result["no_mask_background"]) > 1e-4, result


def test_every_column_sums_to_one(result):
    assert result["column_sum"] < 1e-12, result


def test_the_kernel_is_the_bin_integrated_periodic_decay(result):
    assert result["exact_kernel"] < 1e-5, result
    # and the check can tell: a kernel sampled at channel centres is not it
    assert result["naive_kernel"] > 1e-3, result


def test_a_whole_channel_shift_is_a_roll(result):
    assert result["whole_channel_roll"] < 1e-10, result
