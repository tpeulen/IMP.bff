"""PRD-144 step 1: a fitted response tail in the Bayesian response basis.

`bayesian_response_basis` with `tail_fraction = a` and `tail_log10_tau` builds the lifetime
columns on `(1 - a) u + a u (*) k(tau)` -- the fluorescence response as the scatter response
with an exponential diffusion tail -- and keeps the scatter column `u`. Checked:

* `a = 0` reproduces the basis without the tail bit for bit, also when the tail tangents
  are requested;
* every column still sums to one;
* the scatter column does not change with the tail;
* the tangents in a and in log10 tau equal central differences (best of four steps), and
  the background and shift tangents stay right with a tail present;
* a tangent with a planted 1e-3 error is caught (the check can fail).

Written 2026-09-15 (ucfret prompt 455).
"""

import pytest

from bayesian_cxx import run_driver

DRIVER = r"""
#include <IMP/bff/BayesianMeasuredResponse.h>
#include <cstdio>
using namespace IMP::bff;

static double rel_err(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0, m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) { d = std::max(d, std::fabs(a[i] - b[i])); m = std::max(m, std::fabs(b[i])); }
  return d / std::max(m, 1e-300);
}

int main() {
  const std::size_t n = 256;
  const BayesianDecayAxis ax{n, 0.064, n};
  std::vector<double> tau;
  for (int c = 0; c < 8; ++c) tau.push_back(0.064 * std::pow(20.0 / 0.064, c / 7.0));
  const BayesianPeriodicKernel kernel(ax, tau);
  std::vector<double> measured(n, 0.0);
  for (std::size_t i = 10; i < 90; ++i) {
    const double t = double(i) - 20.0;
    measured[i] = 1e4 * std::exp(-0.5 * t * t / 9.0) * (1.0 + 0.8 * std::tanh(t / 3.0)) + 30.0;
  }
  const double b = 0.05, s = 3.37, a = 0.12, lt = std::log10(0.4);
  BayesianResponseOptions plain, tail0, tail;
  tail0.tail_tangents = true;
  tail.tail_fraction = a; tail.tail_log10_tau = lt; tail.tail_tangents = true;
  const auto B_plain = bayesian_response_basis(kernel, measured, b, s, true, plain);
  const auto B_tail0 = bayesian_response_basis(kernel, measured, b, s, true, tail0);
  const bool identical = B_plain.B == B_tail0.B && B_plain.dB_shift == B_tail0.dB_shift && B_plain.dB_background == B_tail0.dB_background;
  const auto at = bayesian_response_basis(kernel, measured, b, s, true, tail);
  const std::size_t K = at.K;
  double sums = 0.0, scatter_diff = 0.0;
  for (std::size_t k = 0; k < K; ++k) { double c = 0; for (std::size_t i = 0; i < n; ++i) c += at.B[i * K + k]; sums = std::max(sums, std::fabs(c - 1.0)); }
  for (std::size_t i = 0; i < n; ++i) scatter_diff = std::max(scatter_diff, std::fabs(at.B[i * K] - B_plain.B[i * K]));
  auto basis_at = [&](double da, double dlt, double db, double ds) {
    BayesianResponseOptions o = tail; o.tail_tangents = false; o.tail_fraction = a + da; o.tail_log10_tau = lt + dlt;
    return bayesian_response_basis(kernel, measured, b + db, s + ds, false, o).B;
  };
  auto best_fd = [&](int which, const std::vector<double>& tangent) {
    double best = 1e300;
    for (double h : {1e-3, 1e-4, 1e-5, 1e-6}) {
      const double da = which == 0 ? h : 0, dlt = which == 1 ? h : 0, db = which == 2 ? h : 0, ds = which == 3 ? h : 0;
      const auto p = basis_at(da, dlt, db, ds), m = basis_at(-da, -dlt, -db, -ds);
      std::vector<double> fd(p.size());
      for (std::size_t i = 0; i < p.size(); ++i) fd[i] = (p[i] - m[i]) / (2 * h);
      best = std::min(best, rel_err(tangent, fd));
    }
    return best;
  };
  const double e_a = best_fd(0, at.dB_tail_fraction), e_t = best_fd(1, at.dB_tail_log10_tau);
  const double e_b = best_fd(2, at.dB_background), e_s = best_fd(3, at.dB_shift);
  std::vector<double> planted = at.dB_tail_log10_tau;
  for (double& v : planted) v *= 1.0 + 1e-3;
  const double e_planted = best_fd(1, planted);
  std::printf("{\"identical_at_zero\": %d, \"column_sum\": %.3e, \"scatter_diff\": %.3e, \"tail_changes_columns\": %.3e,"
              " \"tangent_a\": %.3e, \"tangent_log10_tau\": %.3e, \"tangent_b\": %.3e, \"tangent_shift\": %.3e, \"planted\": %.3e}\n",
              int(identical), sums, scatter_diff, rel_err(at.B, B_plain.B), e_a, e_t, e_b, e_s, e_planted);
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    return run_driver(DRIVER)


def test_zero_tail_is_bit_identical(result):
    assert result["identical_at_zero"] == 1, result


def test_unit_sums_and_scatter_column(result):
    assert result["column_sum"] < 1e-12, result
    assert result["scatter_diff"] == 0.0, result
    assert result["tail_changes_columns"] > 1e-3, result


def test_tail_tangents_are_derivatives(result):
    assert result["tangent_a"] < 1e-7, result
    assert result["tangent_log10_tau"] < 1e-7, result
    assert result["tangent_b"] < 1e-7, result
    assert result["tangent_shift"] < 1e-7, result


def test_a_planted_error_is_caught(result):
    assert result["planted"] > 5e-4, result
