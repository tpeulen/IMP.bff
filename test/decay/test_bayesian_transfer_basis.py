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
  projection within the ridge's bias bound, sqrt(lambda) |c| / 2 in data space;
* 18c: a map of rate 0 is the identity (each lifetime column maps onto itself,
  within the ridge's bias bound),
  and a quenched column carries the light the donor keeps, tau_s / tau =
  1 / (1 + k tau) -- a map that forgot that scaling would carry all of it.
  Against the Python maps (CBM56, 128 distances, 11 rotational times) the decays
  agree to 4.8e-12 of their peaks: ucfret `transfer_gate.cpp`;
* 18d: the sensitised acceptor light totals the transfer efficiency k/(1/tau + k),
  and is zero without transfer; the directly excited acceptor is one unit, and
  its rotational partner carries tau_s/tau_a of it (a partner normalised to one
  unit would not). Against the Python maps: decays 3.1e-10 of their peaks, the
  floor numpy's own LU vs Cholesky reaches on those cancelling maps. (the algorithm itself is tested in tttrlib's `test_damped_newton.cpp`).

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
  // 18c: maps of an added rate. Rate 0 is the identity; the light a column keeps is tau_s / tau
  const std::vector<double> rates = {0.0, 0.25, 4.0};
  const std::vector<double> S = bayesian_transfer_rate_maps(tb, P, rates);
  const std::size_t nt = tau.size();
  double identity = 0.0, area = 0.0, area_ctl = 0.0;
  for (std::size_t j = 0; j < rates.size(); ++j)
    for (std::size_t c = 0; c < nt; ++c) {
      double tot = 0.0, dev = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        double v = 0.0;
        for (std::size_t k = 0; k < K; ++k) v += tb.basis.B[i * K + k] * S[(j * K + k) * nt + c];
        tot += v;
        if (j == 0) dev += (v - tb.basis.B[i * K + 1 + c]) * (v - tb.basis.B[i * K + 1 + c]);
      }
      const double keep = 1.0 / (1.0 + rates[j] * tau[c]);
      // a unit coefficient: the ridge leaves at most sqrt(lambda)/2 in data space (2-norm)
      if (j == 0) identity = std::max(identity, std::sqrt(dev) / (0.5 * std::sqrt(P.lambda())));
      area = std::max(area, std::fabs(tot - keep));
      area_ctl = std::max(area_ctl, std::fabs(tot - 1.0));
    }
  // 18d: the acceptor maps. Sensitised light totals the transfer efficiency; no transfer, no light;
  // the direct decay is one unit and its rotational partner carries tau_s / tau_a of it
  const std::vector<double> ta = {2.0, 5.0}, rh = {0.5};
  const BayesianAcceptorMaps am = bayesian_transfer_acceptor_maps(tb, P, rates, ta, rh);
  auto total = [&](const double* coef, std::size_t stride, std::size_t col) {
    double t = 0.0;
    for (std::size_t i = 0; i < n; ++i) for (std::size_t k = 0; k < K; ++k) t += tb.basis.B[i * K + k] * coef[k * stride + col];
    return t;
  };
  double sens_eff = 0.0, sens_zero = 0.0, direct_one = 0.0, partner = 0.0, partner_ctl = 0.0;
  for (std::size_t j = 0; j < rates.size(); ++j)
    for (std::size_t l = 0; l < ta.size(); ++l)
      for (std::size_t c = 0; c < nt; ++c) {
        const double tot = total(am.sensitised.data() + (j * ta.size() + l) * K * nt, nt, c);
        const double eff = rates[j] / (1.0 / tau[c] + rates[j]);
        if (j == 0) sens_zero = std::max(sens_zero, std::fabs(tot));
        else sens_eff = std::max(sens_eff, std::fabs(tot - eff));
      }
  for (std::size_t l = 0; l < ta.size(); ++l) {
    direct_one = std::max(direct_one, std::fabs(total(am.direct.data() + l * K, 1, 0) - 1.0));
    const double p = total(am.direct_rot.data() + l * K, 1, 0), ts = 1.0 / (1.0 / ta[l] + 1.0 / rh[0]);
    partner = std::max(partner, std::fabs(p / (ts / ta[l]) - 1.0));
    partner_ctl = std::max(partner_ctl, std::fabs(p - 1.0));
  }
  std::printf("{\"acc_sens_efficiency\": %.6e, \"acc_sens_no_transfer\": %.6e, \"acc_direct_unit\": %.6e,"
              " \"acc_partner_area\": %.6e, \"acc_partner_if_unit\": %.6e, ", sens_eff, sens_zero, direct_one, partner, partner_ctl);
  std::printf("\"map_identity\": %.6e, \"map_area\": %.6e, \"map_area_if_unscaled\": %.6e, ", identity, area, area_ctl);
  std::printf("\"recon\": %.6e, \"recon_bound\": %.6e, \"recon_ridge_x1e4\": %.6e, \"lambda\": %.6e, ",
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


def test_a_rate_map_of_zero_is_the_identity(result):
    # in units of the ridge's bias bound for a unit coefficient
    assert result["map_identity"] <= 1.0, result


def test_a_rate_map_keeps_the_light_the_donor_keeps(result):
    assert result["map_area"] < 1e-6, result
    # the check can tell: the unscaled columns would all carry one unit
    assert result["map_area_if_unscaled"] > 0.1, result


def test_sensitised_light_is_the_transfer_efficiency(result):
    assert result["acc_sens_efficiency"] < 1e-6, result
    assert result["acc_sens_no_transfer"] == 0.0, result


def test_direct_acceptor_is_one_unit_and_its_partner_the_shorter_lifetime(result):
    assert result["acc_direct_unit"] < 1e-9, result
    # bin factors and the trapezoid make the partner's area approximate
    assert result["acc_partner_area"] < 1e-2, result
    assert result["acc_partner_if_unit"] > 0.5, result
