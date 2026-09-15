"""PRD-143 #22: the TCSPC instrument stage, once (`internal/TCSPCInstrument.h`).

Scale, scatter, background pattern, uncorrelated background as fractions of the
fluorescence total, Coates pile-up and the DNL table -- the stage `TCSPCDecay`
and the Bayesian decay model share. Checked:

* one function, two spaces: applied to a convolved decay `F = B a` in channel
  space it equals `B` applied to the stage in the amplitude space of a basis
  whose columns sum to one (response and flat background as unit vectors);
* its closed-form derivatives (with respect to the curve and the four fitted
  quantities) against central differences, and a planted 1e-4 error is caught;
* ChiSurf's absolute form, `(F + scatter_abs response) n0 + background_abs`, is
  reproduced exactly by the fractions `tcspc_instrument_fractions_from_absolute`
  returns;
* ChiSurf's order with pile-up and DNL -- components, pile-up, background,
  linearisation -- against a write-out of that order by hand, and adding the
  background BEFORE pile-up must give a different curve.

Written 2026-09-15 (ucfret prompt 443).
"""

import pytest

from bayesian_cxx import run_driver

DRIVER = r"""
#include <IMP/bff/internal/TCSPCInstrument.h>
#include <cstdio>
#include <random>
using namespace IMP::bff::internal;

static double maxabs(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i])); return m;
}
static double peak(const std::vector<double>& a) { double m = 0; for (double x : a) m = std::max(m, std::fabs(x)); return m; }

int main() {
  const std::size_t n = 200, K = 9;
  std::mt19937_64 rng(5); std::uniform_real_distribution<double> U(0.0, 1.0);
  // a basis: column 0 the response, K-1 flat, the rest decays; every column of unit sum
  std::vector<double> B(n * K, 0.0), resp(n), flat(n, 1.0 / double(n));
  double rs = 0; for (std::size_t i = 0; i < n; ++i) { resp[i] = std::exp(-0.5 * std::pow((double(i) - 15.0) / 3.0, 2)); rs += resp[i]; }
  for (double& x : resp) x /= rs;
  for (std::size_t i = 0; i < n; ++i) { B[i * K] = resp[i]; B[i * K + K - 1] = flat[i]; }
  for (std::size_t c = 1; c + 1 < K; ++c) {
    double tau = 5.0 + 10.0 * c, s = 0;
    for (std::size_t i = 0; i < n; ++i) { B[i * K + c] = std::exp(-double(i) / tau) * (1.0 + 0.1 * U(rng)); s += B[i * K + c]; }
    for (std::size_t i = 0; i < n; ++i) B[i * K + c] /= s;
  }
  std::vector<double> a(K); for (double& x : a) x = U(rng);
  TCSPCInstrumentParameters<double> p; p.scale = 3.0e4; p.scatter = 0.07; p.background = 0.02;
  // channel space
  std::vector<double> F(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) for (std::size_t k = 0; k < K; ++k) F[i] += B[i * K + k] * a[k];
  TCSPCInstrumentSettings sc; sc.response = resp.data(); sc.flat = flat.data();
  std::vector<double> y = F; tcspc_instrument(y.data(), n, sc, p);
  // amplitude space, then the basis
  std::vector<double> e0(K, 0.0), eK(K, 0.0); e0[0] = 1.0; eK[K - 1] = 1.0;
  TCSPCInstrumentSettings sa; sa.response = e0.data(); sa.flat = eK.data();
  std::vector<double> a2 = a; tcspc_instrument_components(a2.data(), K, sa, p, a2.data());
  std::vector<double> y2(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) for (std::size_t k = 0; k < K; ++k) y2[i] += B[i * K + k] * a2[k];
  const double spaces = maxabs(y, y2) / peak(y);

  // derivatives: theta = (a_0..a_{K-1}); F = B a, so J_F = B
  std::vector<double> Jout(n * K), dpar(n * 4);
  tcspc_instrument_components_jacobian(F.data(), n, sc, p, B.data(), K, Jout.data(), dpar.data());
  auto stage = [&](const std::vector<double>& aa, const TCSPCInstrumentParameters<double>& pp) {
    std::vector<double> f(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) for (std::size_t k = 0; k < K; ++k) f[i] += B[i * K + k] * aa[k];
    tcspc_instrument_components(f.data(), n, sc, pp, f.data()); return f;
  };
  double jerr = 0, jpk = 0;
  for (std::size_t c = 0; c < K; ++c) {
    const double h = 1e-6; auto ap = a, am = a; ap[c] += h; am[c] -= h;
    const auto fp = stage(ap, p), fm = stage(am, p);
    for (std::size_t i = 0; i < n; ++i) { const double fd = (fp[i] - fm[i]) / (2 * h); jpk = std::max(jpk, std::fabs(fd)); jerr = std::max(jerr, std::fabs(fd - Jout[i * K + c])); }
  }
  double perr = 0, planted = 0;
  for (int q = 0; q < 4; ++q) {
    auto pp = p, pm = p; const double h = q == 0 ? 1.0 : 1e-7;
    if (q == 0) { pp.scale += h; pm.scale -= h; } if (q == 1) { pp.scatter += h; pm.scatter -= h; }
    if (q == 2) { pp.pattern += h; pm.pattern -= h; } if (q == 3) { pp.background += h; pm.background -= h; }
    const auto fp = stage(a, pp), fm = stage(a, pm);
    double pk = 0, e = 0, e_bad = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const double fd = (fp[i] - fm[i]) / (2 * h); pk = std::max(pk, std::fabs(fd));
      e = std::max(e, std::fabs(fd - dpar[i * 4 + q]));
      e_bad = std::max(e_bad, std::fabs(fd - dpar[i * 4 + q] * (1.0 + 1e-4)));
    }
    if (pk > 0) { perr = std::max(perr, e / pk); if (q == 1) planted = e_bad / pk; }
  }

  // ChiSurf's absolute form
  const double n0 = 2.5e3, scat_abs = 40.0, bg_abs = 3.0;
  std::vector<double> chisurf(n);
  for (std::size_t i = 0; i < n; ++i) chisurf[i] = (F[i] + scat_abs * resp[i]) * n0 + bg_abs;
  const auto pc = tcspc_instrument_fractions_from_absolute(n0, scat_abs, bg_abs, tcspc_instrument_total(F.data(), n), n);
  std::vector<double> yc = F; tcspc_instrument(yc.data(), n, sc, pc);
  const double absolute = maxabs(chisurf, yc) / peak(chisurf);

  // ChiSurf's order with pile-up and DNL, written out by hand
  std::vector<double> data(n), lin(n);
  for (std::size_t i = 0; i < n; ++i) { data[i] = 1e4 * std::exp(-double(i) / 40.0) + 50.0; lin[i] = 1.0 + 0.05 * std::sin(0.3 * double(i)); }
  TCSPCInstrumentSettings sp = sc; sp.pile_up_data = data.data(); sp.pile_up_n_data = int(n);
  sp.repetition_rate_mhz = 20.0; sp.dead_time_ns = 85.0; sp.measurement_time_s = 5.0; sp.linearization = lin.data();
  std::vector<double> ys = F; tcspc_instrument(ys.data(), n, sp, p);
  const double total = tcspc_instrument_total(F.data(), n);
  std::vector<double> hand(n);
  for (std::size_t i = 0; i < n; ++i) hand[i] = F[i] + p.scatter * total * resp[i];
  add_pile_up_to_model_ad<double>(hand.data(), int(n), data.data(), int(n), 20.0, 85.0, 5.0, 0, -1);
  for (std::size_t i = 0; i < n; ++i) hand[i] = (hand[i] * p.scale + p.background * p.scale * total * flat[i]) * lin[i];
  const double order = maxabs(ys, hand) / peak(hand);
  std::vector<double> wrong(n);
  for (std::size_t i = 0; i < n; ++i) wrong[i] = (F[i] + p.scatter * total * resp[i]) * p.scale + p.background * p.scale * total * flat[i];
  add_pile_up_to_model_ad<double>(wrong.data(), int(n), data.data(), int(n), 20.0, 85.0, 5.0, 0, -1);
  for (std::size_t i = 0; i < n; ++i) wrong[i] *= lin[i];
  const double wrong_order = maxabs(ys, wrong) / peak(hand);

  std::printf("{\"spaces\": %.3e, \"curve_jacobian\": %.3e, \"parameter_jacobian\": %.3e, \"planted\": %.3e, "
              "\"chisurf_absolute\": %.3e, \"order\": %.3e, \"background_before_pile_up\": %.3e}\n",
              spaces, jerr / jpk, perr, planted, absolute, order, wrong_order);
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    return run_driver(DRIVER)


def test_one_function_in_both_spaces(result):
    assert result["spaces"] < 1e-12, result


def test_derivatives(result):
    assert result["curve_jacobian"] < 1e-6 and result["parameter_jacobian"] < 1e-6, result
    assert result["planted"] > 5e-5, result


def test_chisurf_absolute_form(result):
    assert result["chisurf_absolute"] < 1e-12, result


def test_chisurf_order_with_pile_up_and_dnl(result):
    assert result["order"] < 1e-12, result
    assert result["background_before_pile_up"] > 1e-6, result
