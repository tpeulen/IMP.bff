"""PRD-143 A4.6: the P-spline's penalty weight lambda as a free node (tpeulen: "should
be free node that is varied, afterall it is not known a priori").

With `log10_lam` a variable (logit on [lo, hi], uniform) instead of a fixed value, the
prior carries the P-spline's normaliser `(rank/2) log lambda` and its derivatives in
the new coordinate, including the cross terms with the spline coefficients. Checked:

* the prior's gradient and Hessian on every coordinate, the lambda row and column
  included, against central differences;
* the prior's dependence on lambda at fixed coefficients is `(rank/2) Delta ln lambda
  - (Delta lambda / 2) |D c|^2` plus the logit transform's Jacobian -- with `D` and `c`
  rebuilt here in numpy from the manifest; without the `(rank/2) log lambda` term the
  difference is off by `rank/2 Delta ln lambda` (the control);
* the posterior gradient on the lambda coordinate against differences;
* a manifest that fixes lambda AND frees it is refused.

Written 2026-09-15 (ucfret prompt 452).
"""

import json
import math
import os
import tempfile

import numpy as np
import pytest

from bayesian_cxx import run_driver
import bayesian_decay_synthetic as syn

DRIVER = r"""
#include <IMP/bff/BayesianDecayPosterior.h>
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
  const BayesianDecayPrior P = bayesian_decay_log_prior(ex, th, true);
  const std::size_t zl = ex.variable("log10_lam")->offset;
  double g_err = 0, g_pk = 0, H_err = 0, H_pk = 0, Hl_err = 0, Hl_pk = 0;
  for (std::size_t c = 0; c < dim; ++c) {
    const double h = 1e-5;
    std::vector<double> tp = th, tm = th; tp[c] += h; tm[c] -= h;
    const BayesianDecayPrior Pp = bayesian_decay_log_prior(ex, tp, true), Pm = bayesian_decay_log_prior(ex, tm, true);
    const double fd = (Pp.lp - Pm.lp) / (2 * h);
    g_pk = std::max(g_pk, std::fabs(P.g[c])); g_err = std::max(g_err, std::fabs(fd - P.g[c]));
    for (std::size_t r = 0; r < dim; ++r) {
      const double fh = (Pp.g[r] - Pm.g[r]) / (2 * h), e = std::fabs(fh - P.H[r * dim + c]);
      H_pk = std::max(H_pk, std::fabs(P.H[r * dim + c])); H_err = std::max(H_err, e);
      if (r == zl || c == zl) { Hl_pk = std::max(Hl_pk, std::fabs(P.H[r * dim + c])); Hl_err = std::max(Hl_err, e); }
    }
  }
  const BayesianDecayPoint pt = bayesian_decay_evaluate(ex, E, th);
  std::vector<double> sp = th, sm = th; sp[zl] += 1e-6; sm[zl] -= 1e-6;
  const double fpost = (bayesian_decay_log_posterior(ex, E, sp) - bayesian_decay_log_posterior(ex, E, sm)) / 2e-6;
  // the prior at two lambdas, same coefficients
  std::vector<double> t2 = th; t2[zl] += 0.7;
  const double lp1 = bayesian_decay_log_prior(ex, th, false).lp, lp2 = bayesian_decay_log_prior(ex, t2, false).lp;
  std::printf("{\"g_err\": %.6e, \"g_pk\": %.6e, \"H_err\": %.6e, \"H_pk\": %.6e, \"Hl_err\": %.6e, \"Hl_pk\": %.6e,"
              " \"post_lambda_grad\": %.6e, \"post_lambda_fd\": %.6e, \"z1\": %.17g, \"z2\": %.17g, \"dlp\": %.17g}\n",
              g_err, g_pk, H_err, H_pk, Hl_err, Hl_pk, pt.grad[zl], fpost, th[zl], t2[zl], lp2 - lp1);
  return 0;
}
"""


@pytest.fixture(scope="module")
def setup():
    tmp = tempfile.mkdtemp()
    syn.write(tmp, free_lambda=True)
    with open(os.path.join(tmp, "manifest.json")) as fh:
        m = json.load(fh)
    return tmp, m


@pytest.fixture(scope="module")
def result(setup):
    tmp, _ = setup
    return run_driver(DRIVER, args=[tmp])


def test_prior_gradient_and_hessian_with_lambda(result):
    assert result["g_err"] < 1e-6 * max(result["g_pk"], 1.0), result
    assert result["H_err"] < 1e-5 * max(result["H_pk"], 1.0), result
    assert result["Hl_err"] < 1e-5 * max(result["Hl_pk"], 1.0), result


def test_posterior_gradient_on_lambda(result):
    g, fd = result["post_lambda_grad"], result["post_lambda_fd"]
    assert abs(g - fd) < 1e-5 * max(abs(fd), 1.0), result


def test_lambda_dependence_is_the_pspline_normaliser(setup, result):
    tmp, m = setup
    var = {v["name"]: v for v in m["variables"]}
    th = np.fromfile(os.path.join(tmp, "theta.bin"))
    Q = np.fromfile(os.path.join(tmp, "Q_c.bin")).reshape(m["arrays"]["Q_c"]["shape"])
    vc = var["c"]
    c = Q @ th[vc["offset"]:vc["offset"] + vc["size"]]
    n = len(c)
    D = np.diff(np.eye(n), 2, axis=0)
    q = float(np.sum((D @ c) ** 2))
    lo, hi, rank = var["log10_lam"]["lo"], var["log10_lam"]["hi"], m["pspline"]["rank"]
    sig = lambda z: 1.0 / (1.0 + math.exp(-z))
    L = lambda z: math.log(10.0) * (lo + (hi - lo) * sig(z))
    jac = lambda z: math.log(sig(z)) + math.log(1.0 - sig(z))
    z1, z2 = result["z1"], result["z2"]
    expected = 0.5 * rank * (L(z2) - L(z1)) - 0.5 * (math.exp(L(z2)) - math.exp(L(z1))) * q + jac(z2) - jac(z1)
    assert abs(result["dlp"] - expected) < 1e-9 * max(1.0, abs(expected)), (result["dlp"], expected)
    # the control: without the (rank/2) log lambda normaliser the difference would not match
    assert abs(result["dlp"] - (expected - 0.5 * rank * (L(z2) - L(z1)))) > 1.0


def test_fixed_and_free_together_are_refused(setup):
    tmp, m = setup
    both = tempfile.mkdtemp()
    for f in os.listdir(tmp):
        if f.endswith(".bin"):
            os.symlink(os.path.join(tmp, f), os.path.join(both, f))
    m2 = dict(m)
    m2["fixed_values"] = dict(m["fixed_values"], log10_lam=[1.0])
    with open(os.path.join(both, "manifest.json"), "w") as fh:
        json.dump(m2, fh)
    r = run_driver(DRIVER.replace("const BayesianDecayPrior P = bayesian_decay_log_prior(ex, th, true);",
                                  "try { bayesian_decay_log_prior(ex, th, true); } catch (const std::runtime_error&) { std::printf(\"{\\\"refused\\\": 1}\\n\"); return 0; }\n  const BayesianDecayPrior P = bayesian_decay_log_prior(ex, th, true);"),
                   args=[both])
    assert r.get("refused") == 1, r
