"""PRD-144 step 2: the decay model with a fitted response tail per detector.

The synthetic experiment with `tail_fraction_<det>` and `tail_log10_tau_<det>` variables in its
responses table: the count Jacobian equals central differences on every column (the tail
coordinates included; the same driver as `test_bayesian_decay_model.py`), and the tail changes
the expected counts (it is not silently ignored). ucfret's CBM56 gates, without a tail, are
unchanged.

Written 2026-09-15 (ucfret prompt 455).
"""

import json
import os
import tempfile

import pytest

from bayesian_cxx import run_driver
import bayesian_decay_synthetic as syn
from test_bayesian_decay_model import DRIVER

EFFECT = r"""
#include <IMP/bff/BayesianDecayModel.h>
#include <cstdio>
using namespace IMP::bff;
int main(int, char** argv) {
  BayesianDecayExperiment ex;
  bayesian_decay_experiment_load(argv[1], ex);
  std::vector<double> th(ex.dim);
  std::ifstream(std::string(argv[1]) + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(th.data()), std::streamsize(ex.dim * sizeof(double)));
  const BayesianDecayTensors E(ex);
  const auto lam1 = bayesian_decay_expected_counts(ex, E, bayesian_decay_unpack(ex, th));
  std::vector<double> t2 = th; t2[ex.variable("tail_fraction_gs")->offset] += 1.0;
  const auto lam2 = bayesian_decay_expected_counts(ex, E, bayesian_decay_unpack(ex, t2));
  double d = 0, pk = 0;
  for (std::size_t i = 0; i < lam1.size(); ++i) { d = std::max(d, std::fabs(lam2[i] - lam1[i])); pk = std::max(pk, lam1[i]); }
  std::printf("{\"tail_effect\": %.6e}\n", d / pk);
  return 0;
}
"""


@pytest.fixture(scope="module")
def tmpdir_tail():
    tmp = tempfile.mkdtemp()
    syn.write(tmp, tail=True)
    return tmp


@pytest.fixture(scope="module")
def result(tmpdir_tail):
    return run_driver(DRIVER, args=[tmpdir_tail])


def test_the_count_jacobian_includes_the_tail_columns(tmpdir_tail, result):
    m = json.load(open(os.path.join(tmpdir_tail, "manifest.json")))
    assert any(v["name"].startswith("tail_") for v in m["variables"])
    assert result["live_count_columns"] == result["dim"], result
    assert result["count_jacobian"] < 1e-6, result
    assert result["amplitude_jacobian"] < 1e-6, result


def test_the_tail_changes_the_counts(tmpdir_tail):
    r = run_driver(EFFECT, args=[tmpdir_tail])
    assert r["tail_effect"] > 1e-4, r
