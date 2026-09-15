"""PRD-143 #13 (A4.5): the crosstalk mixing arithmetic exists once
(`internal/CrosstalkMixing.h`), templated.

`crosstalk_mix` against numpy's `M.T @ S` for a non-square matrix, and on tttrlib's
dual numbers its derivative by a matrix entry is the source that entry multiplies
(the bilinear form `crosstalk_apply_mixing_jacobian` documents). A transposed matrix
must not give the same result (the control). The Bayesian decay model's emission step
runs through it; ucfret's CBM56 fit output is byte-identical before and after.

Written 2026-09-15 (ucfret prompt 452).
"""

import numpy as np
import pytest

from bayesian_cxx import run_driver

DRIVER = r"""
#include <IMP/bff/internal/CrosstalkMixing.h>
#include <IMP/bff/internal/Dual.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace IMP::bff;
int main() {
  const std::size_t ns = 3, nd = 4, ni = 5;
  std::vector<double> M(ns * nd), S(ns * ni), out(nd * ni), outT(nd * ni);
  for (std::size_t i = 0; i < M.size(); ++i) M[i] = 0.1 + 0.37 * double((i * 7) % 11);
  for (std::size_t i = 0; i < S.size(); ++i) S[i] = 1.0 + 0.53 * double((i * 5) % 13);
  internal::crosstalk_mix(M.data(), ns, nd, S.data(), ni, out.data());
  // derivative by M[1, 2] on dual numbers: the source row 1 in detector 2
  using D = tttrlib::Dual<double>;
  std::vector<D> Md(M.begin(), M.end()), Sd(S.begin(), S.end()), od(nd * ni);
  Md[1 * nd + 2].grad = 1.0;
  internal::crosstalk_mix(Md.data(), ns, nd, Sd.data(), ni, od.data());
  double dd = 0.0;
  for (std::size_t d = 0; d < nd; ++d) for (std::size_t it = 0; it < ni; ++it)
    dd = std::max(dd, std::fabs(od[d * ni + it].grad - (d == 2 ? S[1 * ni + it] : 0.0)));
  std::printf("{\"M\": [");
  for (std::size_t i = 0; i < M.size(); ++i) std::printf("%s%.17g", i ? "," : "", M[i]);
  std::printf("], \"S\": [");
  for (std::size_t i = 0; i < S.size(); ++i) std::printf("%s%.17g", i ? "," : "", S[i]);
  std::printf("], \"out\": [");
  for (std::size_t i = 0; i < out.size(); ++i) std::printf("%s%.17g", i ? "," : "", out[i]);
  std::printf("], \"dual_derivative_error\": %.6e}\n", dd);
  return 0;
}
"""


@pytest.fixture(scope="module")
def result():
    return run_driver(DRIVER)


def test_mixing_is_the_matrix_product(result):
    M = np.asarray(result["M"]).reshape(3, 4)
    S = np.asarray(result["S"]).reshape(3, 5)
    out = np.asarray(result["out"]).reshape(4, 5)
    assert np.abs(out - M.T @ S).max() < 1e-12 * np.abs(out).max()
    # the control: rows and columns of the matrix are not interchangeable
    Msq = M[:3, :3]
    assert np.abs(Msq.T @ S[:, :3] - Msq @ S[:, :3]).max() > 1e-3


def test_dual_derivative_is_the_source(result):
    assert result["dual_derivative_error"] < 1e-14, result
