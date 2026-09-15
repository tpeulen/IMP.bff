"""PRD-143 #18e: an experiment whose transfer tensors are built by the loader.

A manifest may carry a `transfer` block -- the response, the lifetime grid, the
(R0, tau_ref) pair and the distance, rotational and acceptor grids -- instead of
the tensors themselves; `bayesian_decay_experiment_load` then builds them with
`BayesianTransferTensors.h`. What is checked:

* the tensors come out in the shapes the model reads, on the grids given, and the
  model evaluates on them (finite, positive expected counts);
* they are the transfer header's own maps (the donor quenching tensor equals
  `bayesian_transfer_rate_maps` called directly, bit for bit);
* a manifest giving both the block and a tensor array is refused.

On the CBM56 measurement the fit from C++-built tensors lands on the polished
Python mode (evidence 0.001 nats, p(R/R0) 6.4e-6 of its peak, D/dof 1e-5):
ucfret `s89_cpp`, `CBM56_TENSORS=cpp`. Written 2026-09-15 (ucfret prompt 448).
"""

import json
import math
import os
import shutil
import tempfile

import numpy as np
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
  bool refused = false;
  if (argc > 2) {
    try { bayesian_decay_experiment_load(argv[2], ex); } catch (const std::runtime_error&) { refused = true; }
    ex = BayesianDecayExperiment();
  }
  const nlohmann::json m = bayesian_decay_experiment_load(dir, ex);
  std::vector<double> th(ex.dim);
  std::ifstream(dir + "/theta.bin", std::ios::binary).read(reinterpret_cast<char*>(th.data()), std::streamsize(ex.dim * sizeof(double)));
  const BayesianDecayTensors E(ex);
  const std::vector<double> lam = bayesian_decay_expected_counts(ex, E, bayesian_decay_unpack(ex, th));
  double lo = 1e300; bool finite = true;
  for (double x : lam) { lo = std::min(lo, x); finite = finite && std::isfinite(x); }
  // the same maps, called directly
  const auto& t = m["transfer"];
  BayesianTransferBasisSpec spec;
  spec.axis = ex.axis(); spec.response_sigma = t["response_sigma"]; spec.log10_tau_lo = t["log10_tau_lo"];
  spec.log10_tau_hi = t["log10_tau_hi"]; spec.per_decade = t["per_decade"];
  const auto tb = bayesian_transfer_basis(spec);
  const auto P = bayesian_transfer_projector(tb.basis);
  std::vector<double> k;
  for (double r : t["rel"].get<std::vector<double>>()) k.push_back(std::pow(1.0 / r, 6) / t["tau_ref"].get<double>());
  const auto S_R = bayesian_transfer_rate_maps(tb, P, k);
  const bool same = S_R == ex["E_S_R"].d;
  std::printf("{\"K\": %zu, \"Kint\": %zu, \"nR\": %zu, \"nA\": %zu, \"nrho\": %zu, \"nrho_a\": %zu,"
              " \"counts_finite\": %d, \"counts_min\": %.6e, \"S_R_is_the_header_map\": %d, \"both_refused\": %d}\n",
              E.K, E.Kint, E.nR, E.nA, E.nrho, E.nrho_a, int(finite), lo, int(same), int(refused));
  return 0;
}
"""

TENSORS = ("E_base", "E_S_R", "E_S_Ag", "E_S_Ag_rot_r", "E_A_dir", "E_A_dir_rot_r", "E_S_rho", "tau_c")


@pytest.fixture(scope="module")
def result():
    tmp = tempfile.mkdtemp()
    both = tempfile.mkdtemp()
    try:
        syn.write(tmp)
        with open(os.path.join(tmp, "manifest.json")) as fh:
            m = json.load(fh)
        lo = math.log10(0.3) - 0.5
        # five lifetimes from 0.3 ns, two per decade: the synthetic model's Kint
        m["transfer"] = dict(response_sigma=0.15, log10_tau_lo=lo, log10_tau_hi=lo + 3.0, per_decade=2,
                             R0=50.0, tau_ref=4.0, rel=[float(x) for x in np.linspace(0.4, 2.0, syn.NR)],
                             rho=[0.5, 2.0, 8.0][: syn.NRHO], tau_a=[2.0, 5.0][: syn.NA], rho_a=[0.5, 3.0][: syn.NRHO_A])
        shutil.copytree(tmp, both, dirs_exist_ok=True)
        with open(os.path.join(both, "manifest.json"), "w") as fh:
            json.dump(m, fh)                       # the block AND the arrays
        m["arrays"] = {k: v for k, v in m["arrays"].items() if k not in TENSORS}
        with open(os.path.join(tmp, "manifest.json"), "w") as fh:
            json.dump(m, fh)
        return run_driver(DRIVER, args=[tmp, both])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(both, ignore_errors=True)


def test_tensors_have_the_model_shapes(result):
    assert (result["K"], result["Kint"]) == (syn.KINT + 2, syn.KINT), result
    assert (result["nR"], result["nA"], result["nrho"], result["nrho_a"]) == (syn.NR, syn.NA, syn.NRHO, syn.NRHO_A), result


def test_the_model_evaluates_on_them(result):
    assert result["counts_finite"] == 1 and result["counts_min"] > 0.0, result


def test_they_are_the_transfer_header_maps(result):
    assert result["S_R_is_the_header_map"] == 1, result


def test_block_and_arrays_together_are_refused(result):
    assert result["both_refused"] == 1, result
