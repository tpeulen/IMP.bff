"""PRD-143 A4.3: the acceptor maps built exactly (tpeulen: "convolution periodic").

`bayesian_transfer_acceptor_maps_exact` builds the sensitised acceptor decay from the
one-way transfer kinetics (`PhotophysicsTransferKinetics.h`'s components) through the
basis's own periodic construction, and the directly excited acceptor the same way.
What is checked, against an INDEPENDENT construction written here:

* the sensitised decay column (before projection) equals a brute force: the acceptor
  population in closed form, summed over earlier pulses in closed form, integrated
  over the excitation time within channel 0 and over each detection channel by
  midpoint sums on two grids with Richardson's extrapolation, circularly convolved
  with the response -- for a donor and acceptor
  of different rates, for the degenerate case (quenched donor and acceptor at the
  same rate, the `t e^{-kt}` component), and for a rotational partner;
* the prototype's construction (sampled exponentials, non-periodic trapezoid
  convolution) does not equal it (the control);
* the maps are the projection of those columns and total the efficiency.

Written 2026-09-15 (ucfret prompt 452).
"""

import pytest

from bayesian_cxx import run_driver

DRIVER = r"""
#include <IMP/bff/BayesianTransferTensors.h>
#include <cstdio>
using namespace IMP::bff;

int main() {
  const std::size_t n = 256;
  const double dt = 0.064, P = double(n) * dt;
  BayesianTransferBasisSpec spec;
  spec.axis = BayesianDecayAxis{n, dt, n};
  spec.response_position = 1.6;
  const BayesianTransferBasis tb = bayesian_transfer_basis(spec);
  BayesianResponseOptions opt; opt.remove_background = false;
  // the column of a set of terms, unit sum
  auto column = [&](const std::vector<internal::BayesianDecayTerm>& terms) {
    std::vector<double> life, w; std::vector<int> ord;
    for (const auto& t : terms) { life.push_back(t.tau); ord.push_back(t.order); w.push_back(t.weight); }
    const BayesianPeriodicKernel k(spec.axis, life, ord);
    const BayesianResponseBasis c = bayesian_response_basis(k, tb.response, 0.0, 0.0, false, opt);
    std::vector<double> y(n, 0.0); double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) { for (std::size_t q = 0; q < w.size(); ++q) y[i] += w[q] * c.B[i * c.K + 1 + q]; s += y[i]; }
    for (double& v : y) v /= s;
    return std::make_pair(y, s);
  };
  // brute force: population a(t) periodic in closed form, integrated by midpoints
  auto brute = [&](double kq, double ka, double kr) {
    const bool degenerate = std::fabs(kq - ka) < 1e-12;
    auto periodic = [&](double t) {       // sum_p a(t + p P), t in [0, P)
      if (degenerate) {
        const double k = ka + kr, q = std::exp(-k * P);
        return std::exp(-k * t) * (t / (1.0 - q) + P * q / ((1.0 - q) * (1.0 - q)));
      }
      const double k1 = ka + kr, k2 = kq + kr;
      return (std::exp(-k1 * t) / (1.0 - std::exp(-k1 * P)) - std::exp(-k2 * t) / (1.0 - std::exp(-k2 * P))) / (kq - ka);
    };
    // midpoint sums are second order: two grids and Richardson's extrapolation
    auto integrate = [&](int Mu, int Mt) {
      std::vector<double> g(n, 0.0);
      for (int a = 0; a < Mu; ++a) {
        const double u = (a + 0.5) / Mu * dt;
        for (std::size_t j = 0; j < n; ++j)
          for (int b = 0; b < Mt; ++b) {
            double t = (double(j) + (b + 0.5) / Mt) * dt - u;
            if (t < 0.0) t += P;
            g[j] += periodic(t) / (double(Mu) * Mt);
          }
      }
      return g;
    };
    const std::vector<double> g1 = integrate(48, 160), g2 = integrate(96, 320);
    std::vector<double> f(n);
    for (std::size_t j = 0; j < n; ++j) f[j] = (4.0 * g2[j] - g1[j]) / 3.0;
    std::vector<double> y(n, 0.0); double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) { for (std::size_t j = 0; j < n; ++j) y[i] += tb.response[j] * f[(i + n - j) % n]; s += y[i]; }
    for (double& v : y) v /= s;
    return y;
  };
  auto rel = [&](const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0.0, p = 0.0;
    for (std::size_t i = 0; i < n; ++i) { d = std::max(d, std::fabs(a[i] - b[i])); p = std::max(p, std::fabs(b[i])); }
    return d / p;
  };
  const double td = 4.0, k = 0.5, ta = 3.0, rho = 1.0, kq = 1.0 / td + k;
  const auto plain = column(internal::bayesian_sensitised_terms(td, k, ta));
  const double e_plain = rel(plain.first, brute(kq, 1.0 / ta, 0.0));
  const double ta_deg = 1.0 / kq;                                         // acceptor at the quenched donor's rate
  const auto deg_terms = internal::bayesian_sensitised_terms(td, k, ta_deg);
  bool has_t_exp = false;
  for (const auto& t : deg_terms) has_t_exp = has_t_exp || t.order == 1;
  const double e_deg = rel(column(deg_terms).first, brute(kq, kq, 0.0));
  const auto rot = column(internal::bayesian_sensitised_terms(td, k, ta, 1.0 / rho));
  const double e_rot = rel(rot.first, brute(kq, 1.0 / ta, 1.0 / rho));
  // control: the prototype's construction of the plain case
  std::vector<double> d(n), kern(n), acc(n), col(n);
  const double tq = 1.0 / kq, bf = internal::bayesian_bin_factor(tq, dt);
  double sk = 0.0;
  for (std::size_t i = 0; i < n; ++i) { d[i] = std::exp(-double(i) * dt / tq) * bf; kern[i] = std::exp(-double(i) * dt / ta); sk += kern[i]; }
  for (double& v : kern) v /= sk;
  internal::BayesianTrapezoidConvolver(kern.data(), n, dt)(d.data(), acc.data());
  internal::BayesianTrapezoidConvolver(tb.response.data(), n, dt)(acc.data(), col.data());
  double sc = 0.0;
  for (double v : col) sc += v;
  for (double& v : col) v /= sc;
  const double e_faithful = rel(col, brute(kq, 1.0 / ta, 0.0));
  // the maps: projections of those columns, totalling the efficiency
  const auto P_ = bayesian_transfer_projector(tb.basis);
  const std::vector<double> tau = tb.kernel->tau();
  std::size_t c4 = 0;
  for (std::size_t c = 0; c < tau.size(); ++c) if (std::fabs(tau[c] - 4.0) < std::fabs(tau[c4] - 4.0)) c4 = c;
  const auto am = bayesian_transfer_acceptor_maps(tb, P_, {k}, {ta}, {rho}, BayesianAcceptorConstruction::exact);
  const std::size_t K = tb.basis.K, nt = tau.size();
  std::vector<double> dec(n, 0.0);
  double tot = 0.0;
  for (std::size_t i = 0; i < n; ++i) { for (std::size_t q = 0; q < K; ++q) dec[i] += tb.basis.B[i * K + q] * am.sensitised[q * nt + c4]; tot += dec[i]; }
  const double eff = k / (1.0 / tau[c4] + k);
  std::printf("{\"plain\": %.6e, \"degenerate\": %.6e, \"degenerate_has_t_exp\": %d, \"rotated\": %.6e, \"prototype\": %.6e,"
              " \"map_total_minus_efficiency\": %.6e, \"plain_total\": %.15g, \"efficiency\": %.15g}\n",
              e_plain, e_deg, int(has_t_exp), e_rot, e_faithful, tot - eff, plain.second, k / (1.0 / td + k));
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    return run_driver(DRIVER)


def test_sensitised_column_is_the_brute_force(result):
    assert result["plain"] < 1e-10, result
    assert abs(result["plain_total"] - result["efficiency"]) < 1e-12, result


def test_the_degenerate_case_uses_t_exp_and_is_exact(result):
    assert result["degenerate_has_t_exp"] == 1, result
    assert result["degenerate"] < 1e-10, result


def test_the_rotational_partner_is_the_brute_force(result):
    assert result["rotated"] < 1e-10, result


def test_the_prototype_construction_is_not(result):
    assert result["prototype"] > 1e-4, result


def test_the_map_totals_the_efficiency(result):
    assert abs(result["map_total_minus_efficiency"]) < 1e-6, result
