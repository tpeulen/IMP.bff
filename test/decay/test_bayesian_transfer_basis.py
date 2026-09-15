"""PRD-143 #18a: the basis the transfer maps are projected onto.

`BayesianTransferTensors.h` builds the lifetime basis of a Bayesian FRET decay
fit on an instrument axis: a sampled Gaussian response, exact periodic columns
on a log-lifetime grid, a flat column. It is `bayesian_response_basis` with the
background removal switched off. What is checked, and what would make each
check fail:

* the grid has `per_decade` points per decade with both ends excluded -- on the
  CBM56 axis (488 channels of 64 ps) the 31 lifetimes the prototype used;
* column 0 is the response itself, untouched: the soft background floor a
  measured response goes through changes a Gaussian's tails, and switching it
  back on must break this check;
* every column sums to one;
* every lifetime column is the circular convolution of the response with the
  periodic kernel, computed here in the time domain (no FFT), clamped and
  renormalised;
* 18b: a decay built from known coefficients is reproduced by its ridge
  projection within the ridge's bias bound, sqrt(lambda) |c| / 2 in data space (the algorithm itself is tested in tttrlib's `test_damped_newton.cpp`).

The C++ basis also equals the Python prototype's (`s53_phase1_pseudolik.Basis`,
cached CBM56 environment) to 1.2e-13 of each column's peak: ucfret
`investigation/pinn_pR_anisotropy/s89_cpp/transfer_gate.cpp`.

Written 2026-09-15 (ucfret prompt 448).
"""

import pytest

from bayesian_cxx import run_driver

DRIVER = r"""
#include <IMP/bff/BayesianTransferTensors.h>
#include <cstdio>
using namespace IMP::bff;

int main() {
  BayesianTransferBasisSpec spec;
  spec.axis = BayesianDecayAxis{488, 0.064, 488};
  const BayesianTransferBasis tb = bayesian_transfer_basis(spec);
  const std::size_t n = tb.basis.n, K = tb.basis.K;
  const std::vector<double>& tau = tb.kernel->tau();
  double resp = 0.0, sums = 0.0, conv = 0.0, peak = 0.0;
  for (std::size_t i = 0; i < n; ++i) { resp = std::max(resp, std::fabs(tb.basis.B[i * K] - tb.response[i])); peak = std::max(peak, tb.response[i]); }
  BayesianResponseOptions floor_on;
  const BayesianResponseBasis nc = bayesian_response_basis(*tb.kernel, tb.response, 0.0, 0.0, false, floor_on);
  double floor_diff = 0.0;
  for (std::size_t i = 0; i < n; ++i) floor_diff = std::max(floor_diff, std::fabs(nc.B[i * K] - tb.response[i]));
  for (std::size_t k = 0; k < K; ++k) { double s = 0; for (std::size_t i = 0; i < n; ++i) s += tb.basis.B[i * K + k]; sums = std::max(sums, std::fabs(s - 1.0)); }
  std::vector<double> ker(n), col(n);
  for (std::size_t c = 0; c < tau.size(); ++c) {
    tb.kernel->kernel(c, ker.data());
    double s = 0.0, pk = 0.0, d = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      double v = 0.0;
      for (std::size_t j = 0; j < n; ++j) v += tb.response[j] * ker[(i + n - j) % n];
      col[i] = std::max(v, 0.0); s += col[i];
    }
    for (std::size_t i = 0; i < n; ++i) { col[i] /= s; pk = std::max(pk, col[i]); }
    for (std::size_t i = 0; i < n; ++i) d = std::max(d, std::fabs(col[i] - tb.basis.B[i * K + 1 + c]));
    conv = std::max(conv, d / pk);
  }
  // 18b: a decay built from known coefficients is reproduced by its projection
  const auto P = bayesian_transfer_projector(tb.basis);
  std::vector<double> c(K), y(n, 0.0), x(K);
  for (std::size_t k = 0; k < K; ++k) c[k] = 1.0 + 0.5 * std::sin(1.7 * double(k));
  for (std::size_t i = 0; i < n; ++i) for (std::size_t k = 0; k < K; ++k) y[i] += tb.basis.B[i * K + k] * c[k];
  P.project(y.data(), x.data());
  // the ridge's bias in data space is bounded: |B x - B c|_2 <= sqrt(lambda) |c|_2 / 2
  // (the SVD filter factors s^2/(s^2 + lambda) leave s lambda/(s^2 + lambda) <= sqrt(lambda)/2)
  double r2 = 0.0, c2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double v = 0.0;
    for (std::size_t k = 0; k < K; ++k) v += tb.basis.B[i * K + k] * x[k];
    r2 += (v - y[i]) * (v - y[i]);
  }
  for (double v : c) c2 += v * v;
  // and a projector with 1e4 times the ridge must violate the bound at the default lambda
  const auto P4 = bayesian_transfer_projector(tb.basis, 1e4 * BAYESIAN_TRANSFER_RIDGE);
  P4.project(y.data(), x.data());
  double r4 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double v = 0.0;
    for (std::size_t k = 0; k < K; ++k) v += tb.basis.B[i * K + k] * x[k];
    r4 += (v - y[i]) * (v - y[i]);
  }
  const double bound = 0.5 * std::sqrt(P.lambda() * c2);
  std::printf("{\"recon\": %.6e, \"recon_bound\": %.6e, \"recon_ridge_x1e4\": %.6e, \"lambda\": %.6e, ",
              std::sqrt(r2), bound, std::sqrt(r4), P.lambda());
  std::printf("\"n_tau\": %zu, \"tau_first\": %.15g, \"tau_last\": %.15g, \"K\": %zu, \"response\": %.6e, \"floor\": %.6e,"
              " \"column_sum\": %.6e, \"convolution\": %.6e}\n",
              tau.size(), tau.front(), tau.back(), K, resp / peak, floor_diff / peak, sums, conv);
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    return run_driver(DRIVER)


def test_the_grid_on_the_cbm56_axis(result):
    assert result["n_tau"] == 31 and result["K"] == 33, result
    assert result["tau_first"] == pytest.approx(10 ** (-1.30103 + 0.1), rel=1e-5), result
    assert result["tau_last"] == pytest.approx(10 ** (-1.30103 + 3.1), rel=1e-5), result


def test_column_zero_is_the_response_untouched(result):
    assert result["response"] < 1e-13, result
    # the negative control: with the background floor active the tails change
    assert result["floor"] > 1e-8, result


def test_every_column_sums_to_one(result):
    assert result["column_sum"] < 1e-12, result


def test_columns_are_the_circular_convolution(result):
    assert result["convolution"] < 1e-12, result


def test_the_projection_reproduces_a_decay_in_the_span(result):
    assert result["lambda"] > 0.0, result
    assert result["recon"] <= result["recon_bound"], result
    # the check can tell: 1e4 times the ridge leaves more than the default's bound
    assert result["recon_ridge_x1e4"] > result["recon_bound"], result
