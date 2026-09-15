"""Pile-up and DNL in the Bayesian decay model (PRD-143 #22; tpeulen, 2026-09-15:
"bayesian way better, but pileup missing").

`bayesian_decay_expected_counts` applies the shared instrument stage's detection
per histogram -- Coates pile-up on fluorescence and scatter, the uncorrelated
background added after it, the DNL table multiplying the result -- when the
experiment carries a `pile_up` block or a `linearization` array. On the synthetic
experiment:

* DNL alone multiplies the counts by the table, exactly;
* with the background switched off, pile-up is a per-channel factor on the counts
  (the factor taken from the stage on a vector of ones);
* with the background on, the difference from that is `lin (1 - f) bg`, not zero
  -- the background is NOT piled up, in ChiSurf's order;
* the count Jacobian with pile-up and DNL is the derivative on every column
  (central differences, best of four steps), and pile-up does change the counts.

Written 2026-09-15 (ucfret prompt 443).
"""

import tempfile

import pytest

from bayesian_cxx import run_driver
import bayesian_decay_synthetic as syn

DRIVER = r"""
#include <IMP/bff/BayesianDecayModel.h>
#include <cstdio>
using namespace IMP::bff;

static std::vector<double> theta_of(const std::string& dir, std::size_t dim) {
  std::vector<double> th(dim);
  std::ifstream(dir + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(th.data()), std::streamsize(dim * sizeof(double)));
  return th;
}

int main(int, char** argv) {
  BayesianDecayExperiment plain, dnl, pile;
  bayesian_decay_experiment_load(argv[1], plain);
  bayesian_decay_experiment_load(argv[2], dnl);
  bayesian_decay_experiment_load(argv[3], pile);
  const std::size_t dim = plain.dim, n = plain.n_bins, nd = plain.data_keys.size();
  const BayesianDecayTensors E(plain);
  std::vector<double> th = theta_of(argv[1], dim);
  // raise the counts so the soft floor is the identity and pile-up is visible
  for (auto& var : plain.variables) if (var.name.rfind("log_scale_", 0) == 0) th[var.offset] = 12.0;
  const auto lam_plain = bayesian_decay_expected_counts(plain, E, bayesian_decay_unpack(plain, th));
  const auto lam_dnl = bayesian_decay_expected_counts(dnl, E, bayesian_decay_unpack(dnl, th));
  const auto& lin = dnl["linearization"].d;
  double e_dnl = 0, pk = 0;
  for (std::size_t j = 0; j < lam_plain.size(); ++j) { pk = std::max(pk, lam_plain[j]); e_dnl = std::max(e_dnl, std::fabs(lam_dnl[j] - lin[j] * lam_plain[j])); }

  // pile-up factor per channel from the stage on ones (DNL included in `pile`)
  std::vector<double> gfac(nd * n);
  for (std::size_t o = 0; o < nd; ++o) {
    internal::TCSPCInstrumentSettings ds;
    ds.pile_up_data = pile["y"].d.data() + o * n; ds.pile_up_n_data = int(n);
    ds.repetition_rate_mhz = pile.repetition_rate_mhz; ds.dead_time_ns = pile.dead_time_ns; ds.measurement_time_s = pile.measurement_time_s;
    ds.linearization = lin.data() + o * n;
    std::vector<double> g(n, 1.0);
    internal::tcspc_instrument_detection(g.data(), n, ds, (const double*)nullptr);
    for (std::size_t i = 0; i < n; ++i) gfac[o * n + i] = g[i];
  }
  // background off: counts = g * plain
  std::vector<double> th0 = th;
  for (auto& var : plain.variables) if (var.name.rfind("bkg_", 0) == 0) th0[var.offset] = -60.0;
  const auto p0 = bayesian_decay_expected_counts(plain, E, bayesian_decay_unpack(plain, th0));
  const auto q0 = bayesian_decay_expected_counts(pile, E, bayesian_decay_unpack(pile, th0));
  double e_pile0 = 0, effect = 0;
  for (std::size_t j = 0; j < p0.size(); ++j) { e_pile0 = std::max(e_pile0, std::fabs(q0[j] - gfac[j] * p0[j])); effect = std::max(effect, std::fabs(q0[j] - lin[j] * p0[j])); }
  // background on: the remainder lin (1 - f) bg must be there
  const auto q1 = bayesian_decay_expected_counts(pile, E, bayesian_decay_unpack(pile, th));
  double remainder = 0;
  for (std::size_t j = 0; j < q1.size(); ++j) remainder = std::max(remainder, std::fabs(q1[j] - gfac[j] * lam_plain[j]));

  // the count Jacobian with pile-up and DNL
  const auto vp = bayesian_decay_unpack(pile, th);
  const auto a2 = bayesian_decay_amplitudes(pile, E, vp);
  const auto Ja = bayesian_decay_amplitude_jacobian(pile, E, vp);
  std::vector<double> Jc;
  const auto lam = bayesian_decay_expected_counts(pile, E, vp, &Jc, &Ja, &a2);
  double worst = 0;
  for (std::size_t c = 0; c < dim; ++c) {
    double best = 1e300, colpk = 0;
    for (double h : {1e-3, 1e-4, 1e-5, 1e-6}) {
      auto tp = th, tm = th; tp[c] += h; tm[c] -= h;
      const auto lp = bayesian_decay_expected_counts(pile, E, bayesian_decay_unpack(pile, tp));
      const auto lm = bayesian_decay_expected_counts(pile, E, bayesian_decay_unpack(pile, tm));
      double d = 0, pkc = 0;
      for (std::size_t r = 0; r < lam.size(); ++r) { const double fd = (lp[r] - lm[r]) / (2 * h); pkc = std::max(pkc, std::fabs(fd)); d = std::max(d, std::fabs(fd - Jc[r * dim + c])); }
      if (pkc > 0) best = std::min(best, d / pkc);
      colpk = std::max(colpk, pkc);
    }
    if (colpk > 0) worst = std::max(worst, best);
  }
  std::printf("{\"dnl\": %.3e, \"pile_up_without_background\": %.3e, \"pile_up_effect\": %.3e, \"background_not_piled\": %.3e, \"jacobian\": %.3e, \"peak\": %.3e}\n",
              e_dnl / pk, e_pile0 / pk, effect / pk, remainder / pk, worst, pk);
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    dirs = [tempfile.mkdtemp() for _ in range(3)]
    syn.write(dirs[0])
    syn.write(dirs[1], linearization=True)
    syn.write(dirs[2], pile_up=True, linearization=True)
    return run_driver(DRIVER, args=dirs)


def test_dnl_multiplies_the_counts(result):
    assert result["dnl"] < 1e-12, result


def test_pile_up_is_a_factor_on_fluorescence(result):
    assert result["pile_up_without_background"] < 1e-9, result
    assert result["pile_up_effect"] > 1e-4, result


def test_background_is_added_after_pile_up(result):
    assert result["background_not_piled"] > 1e-6, result


def test_count_jacobian_with_pile_up_and_dnl(result):
    assert result["jacobian"] < 1e-6, result
