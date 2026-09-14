"""PRD-142 step 3: the model of a polarised multi-sample TCSPC experiment.

`BayesianDecayModel.h` reads an experiment from a manifest and computes the
amplitudes of every physics channel, the expected counts, and both Jacobians
analytically. On a small synthetic experiment (`bayesian_decay_synthetic.py`:
donor-only, FRET and reference-dye samples, two polarised detectors) at a
generic point:

* the amplitude Jacobian and the count Jacobian are the derivatives of the
  amplitudes and the counts, on every column (central differences at the best of
  four steps), and the same comparison catches one column off by 1e-4;
* the scope table is what the model reads: handing the FRET sample the
  donor-only scope must change its amplitudes (the mis-keying that once cost a
  fit 590,000 nats), and must leave the donor-only sample's alone.

Written 2026-09-14 (ucfret prompt 432).
"""

import os
import tempfile

import pytest

from bayesian_cxx import run_driver
import bayesian_decay_synthetic as syn

DRIVER = r"""
#include <IMP/bff/BayesianDecayModel.h>
#include <cstdio>
using namespace IMP::bff;

int main(int argc, char** argv) {
  const std::string dir = argv[1];
  BayesianDecayExperiment ex;
  bayesian_decay_experiment_load(dir, ex);
  const std::size_t dim = ex.dim;
  std::vector<double> th(dim);
  std::ifstream(dir + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(th.data()), std::streamsize(dim * sizeof(double)));
  const BayesianDecayTensors E(ex);
  const BayesianDecayValues v = bayesian_decay_unpack(ex, th);
  const std::vector<double> a2 = bayesian_decay_amplitudes(ex, E, v);
  const std::vector<double> Ja = bayesian_decay_amplitude_jacobian(ex, E, v);
  std::vector<double> Jc;
  const std::vector<double> lam = bayesian_decay_expected_counts(ex, E, v, &Jc, &Ja, &a2);
  const std::size_t na = a2.size(), nc = lam.size();
  //: each column against central differences at the step that suits it: the best
  //: of four, since truncation (a lifetime prior 0.05 decades wide) wants a small
  //: step and rounding (acceptor columns a millionth of the peak) a large one; a
  //: wrong derivative disagrees at every step
  auto column_error = [&](std::size_t c, const std::vector<double>& JA, const std::vector<double>& JC, double& ea, double& ec, double& pc_out) {
    ea = ec = 1e300; pc_out = 0.0;
    for (double h : {1e-3, 1e-4, 1e-5, 1e-6}) {
      std::vector<double> tp = th, tm = th;
      tp[c] += h; tm[c] -= h;
      const auto vp = bayesian_decay_unpack(ex, tp), vm = bayesian_decay_unpack(ex, tm);
      const auto ap = bayesian_decay_amplitudes(ex, E, vp), am = bayesian_decay_amplitudes(ex, E, vm);
      const auto lp = bayesian_decay_expected_counts(ex, E, vp), lm = bayesian_decay_expected_counts(ex, E, vm);
      double pa = 0, da = 0, pc = 0, dc = 0;
      for (std::size_t r = 0; r < na; ++r) { const double fd = (ap[r] - am[r]) / (2 * h); pa = std::max(pa, std::fabs(fd)); da = std::max(da, std::fabs(fd - JA[r * dim + c])); }
      for (std::size_t r = 0; r < nc; ++r) { const double fd = (lp[r] - lm[r]) / (2 * h); pc = std::max(pc, std::fabs(fd)); dc = std::max(dc, std::fabs(fd - JC[r * dim + c])); }
      if (pa > 0) ea = std::min(ea, da / pa); else ea = std::min(ea, da);
      if (pc > 0) ec = std::min(ec, dc / pc);
      pc_out = std::max(pc_out, pc);
    }
  };
  double worst_a = 0.0, worst_c = 0.0, live = 0;
  for (std::size_t c = 0; c < dim; ++c) {
    double ea, ec, pc; column_error(c, Ja, Jc, ea, ec, pc);
    if (pc > 0) { live += 1; worst_c = std::max(worst_c, ec); }
    worst_a = std::max(worst_a, ea);
  }
  //: the procedure can see an error: one column of both Jacobians off by 1e-4 of itself
  std::vector<double> Ja_bad = Ja, Jc_bad = Jc;
  const std::size_t c_bad = ex.variable("QY_A")->offset;
  for (std::size_t r = 0; r < na; ++r) Ja_bad[r * dim + c_bad] *= 1.0 + 1e-4;
  for (std::size_t r = 0; r < nc; ++r) Jc_bad[r * dim + c_bad] *= 1.0 + 1e-4;
  double ea_bad, ec_bad, pc_bad; column_error(c_bad, Ja_bad, Jc_bad, ea_bad, ec_bad, pc_bad);
  // the scope table: the FRET sample handed the donor-only scope
  BayesianDecayExperiment bad = ex;
  for (auto& p : bad.parts) if (p.sample == "DA") p.scope = 0;
  const std::vector<double> a2_bad = bayesian_decay_amplitudes(bad, E, v);
  const std::size_t K = E.K;
  double d_da = 0, p_da = 0, d_d0 = 0;
  for (const auto& p : ex.parts) for (std::size_t k = 0; k < K; ++k) {
    const double d = std::fabs(a2_bad[p.amp_index * K + k] - a2[p.amp_index * K + k]);
    if (p.sample == "DA") { d_da = std::max(d_da, d); p_da = std::max(p_da, std::fabs(a2[p.amp_index * K + k])); }
    if (p.sample == "D0") d_d0 = std::max(d_d0, d);
  }
  std::printf("{\"dim\": %zu, \"live_count_columns\": %g, \"amplitude_jacobian\": %.6e, \"count_jacobian\": %.6e,"
              " \"planted_amplitude_error\": %.6e, \"planted_count_error\": %.6e,"
              " \"misscoped_fret_change\": %.6e, \"misscoped_donor_only_change\": %.6e}\n",
              dim, live, worst_a, worst_c, ea_bad, ec_bad, d_da / p_da, d_d0);
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    tmp = tempfile.mkdtemp()
    syn.write(tmp)
    return run_driver(DRIVER, args=[tmp])


def test_the_amplitude_jacobian_is_the_derivative(result):
    assert result["amplitude_jacobian"] < 1e-6, result


def test_the_count_jacobian_is_the_derivative_on_every_column(result):
    assert result["live_count_columns"] == result["dim"], result
    assert result["count_jacobian"] < 1e-6, result


def test_the_comparison_sees_a_wrong_column(result):
    # the negative control: a column off by 1e-4 of itself must fail the same check
    assert result["planted_amplitude_error"] > 1e-6 and result["planted_count_error"] > 1e-6, result


def test_the_scope_table_is_what_the_model_reads(result):
    assert result["misscoped_fret_change"] > 1e-2, result
    assert result["misscoped_donor_only_change"] == 0.0, result
