"""PRD-145 step 1: the MCMC convergence diagnostics (`internal/McmcDiagnostics.h`, tttrlib's)
against arviz.

Rank-normalised split R-hat, bulk ESS, tail ESS (5 %/95 %), ESS of the mean and MCSE of the mean,
computed by the C++ header and by arviz_stats' array interface on the same draws: independent
chains, strongly autocorrelated AR(1) chains, chains with shifted means (non-mixing), chains with
unequal scales, few draws, and ties (an indicator-like quantity). Agreement to 1e-9 relative on
every one. The header implements the algorithm of Vehtari et al. (2021) as arviz does; this is
the check that it does.

Written 2026-09-15 (ucfret prompt 461).
"""

import json

import numpy as np
import pytest

from bayesian_cxx import run_driver

arviz_stats = pytest.importorskip("arviz_stats")
from arviz_stats.base import array_stats  # noqa: E402

DRIVER = r"""
#include <IMP/bff/internal/McmcDiagnostics.h>
#include <cstdio>
#include <iostream>
#include <string>
int main() {
  // stdin: n_sets, then per set: m n, then m*n values
  int sets; std::cin >> sets;
  std::printf("[");
  for (int k = 0; k < sets; ++k) {
    std::size_t m, n; std::cin >> m >> n;
    tttrlib::McmcChains x(m, std::vector<double>(n));
    for (auto& c : x) for (double& v : c) std::cin >> v;
    std::printf("%s{\"rhat\": %.17g, \"ess_bulk\": %.17g, \"ess_tail\": %.17g, \"ess_mean\": %.17g, \"mcse_mean\": %.17g}",
                k ? "," : "", tttrlib::rhat_rank(x), tttrlib::ess_bulk(x), tttrlib::ess_tail(x, 0.05), tttrlib::ess_mean(x), tttrlib::mcse_mean(x));
  }
  std::printf("]\n");
  return 0;
}
"""


def chain_sets():
    rng = np.random.default_rng(12)
    sets = {}
    sets["iid"] = rng.normal(size=(4, 1000))

    def ar1(m, n, rho, shift=0.0, scale=0.0):
        x = np.zeros((m, n))
        for a in range(m):
            v = rng.normal()
            for i in range(n):
                v = rho * v + np.sqrt(1 - rho ** 2) * rng.normal()
                x[a, i] = a * shift + (1 + a * scale) * v
        return x
    sets["ar1_0.95"] = ar1(4, 1500, 0.95)
    sets["shifted"] = ar1(4, 800, 0.5, shift=0.5)
    sets["scaled"] = ar1(3, 900, 0.3, scale=0.8)
    sets["short"] = rng.normal(size=(2, 9))
    # 5 draws: the split chains hold 2, where Geyer's loop must not run (it once read past the array)
    sets["five"] = rng.normal(size=(4, 5))
    sets["ties"] = (rng.normal(size=(4, 500)) > 0.3).astype(float) + np.round(rng.normal(size=(4, 500)), 1)
    return sets


@pytest.fixture(scope="module")
def result():
    sets = chain_sets()
    text = f"{len(sets)}\n" + "".join(f"{x.shape[0]} {x.shape[1]}\n" + " ".join(repr(float(v)) for v in x.ravel()) + "\n"
                                       for x in sets.values())
    return sets, run_driver(DRIVER, stdin=text)


def reference(x):
    return dict(rhat=float(array_stats.rhat(x, method="rank")), ess_bulk=float(array_stats.ess(x, method="bulk")),
                ess_tail=float(array_stats.ess(x, method="tail", prob=(0.05, 0.95))),
                ess_mean=float(array_stats.ess(x, method="mean")), mcse_mean=float(array_stats.mcse(x, method="mean")))


@pytest.mark.parametrize("name", ["iid", "ar1_0.95", "shifted", "scaled", "short", "five", "ties"])
def test_equals_arviz(result, name):
    sets, out = result
    i = list(sets).index(name)
    ref = reference(sets[name])
    for k, v in ref.items():
        assert out[i][k] == pytest.approx(v, rel=1e-9), (name, k, out[i][k], v)


def test_the_sets_span_converged_and_not(result):
    sets, out = result
    r = {name: out[i]["rhat"] for i, name in enumerate(sets)}
    assert r["iid"] < 1.01 and r["shifted"] > 1.1 and r["scaled"] > 1.01, r
