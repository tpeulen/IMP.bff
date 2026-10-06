"""Special functions, distribution functions and expm against scipy.

bff computes these with scipy's own code (xsf) or with Boost.Math under
scipy's policies, so the special functions must agree with scipy to the bit
and the Boost-backed distributions to round-off. scipy is the oracle here
only; nothing in bff imports it.
"""
from __future__ import annotations

import numpy as np
import pytest

import IMP.bff as bff

scipy = pytest.importorskip("scipy")
from scipy import linalg, special, stats  # noqa: E402

RNG = np.random.default_rng(20261006)


def _x(lo, hi, n=4000):
    return RNG.uniform(lo, hi, n)


def _close(got, want, rtol, atol=0.0):
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.shape == want.shape
    same_nan = np.isnan(got) == np.isnan(want)
    assert same_nan.all(), "NaN pattern differs"
    ok = ~np.isnan(want)
    np.testing.assert_allclose(got[ok], want[ok], rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "name, ours, theirs, lo, hi",
    [
        ("gammaln", bff.gammaln_array, special.gammaln, -50.0, 200.0),
        ("erf", bff.erf_array, special.erf, -8.0, 8.0),
        ("erfc", bff.erfc_array, special.erfc, -8.0, 30.0),
        ("digamma", bff.digamma_array, special.digamma, -20.0, 200.0),
        ("i0e", bff.i0e_array, special.i0e, -800.0, 800.0),
        ("j0", bff.j0_array, special.j0, -200.0, 200.0),
        ("j1", bff.j1_array, special.j1, -200.0, 200.0),
        ("ndtr", bff.ndtr_array, special.ndtr, -40.0, 40.0),
    ],
)
def test_unary_matches_scipy_to_the_bit(name, ours, theirs, lo, hi):
    x = np.concatenate([_x(lo, hi), [0.0, 1.0, -1.0, 0.5, np.inf, -np.inf, np.nan]])
    _close(ours(x), theirs(x), rtol=0.0)


@pytest.mark.parametrize(
    "ours, theirs",
    [(bff.gammainc_array, special.gammainc), (bff.gammaincc_array, special.gammaincc)],
)
def test_incomplete_gamma(ours, theirs):
    a = _x(1e-3, 80.0)
    x = _x(0.0, 120.0)
    _close(ours(a, x), theirs(a, x), rtol=0.0)


def test_hurwitz_zeta_and_polygamma():
    s = RNG.integers(2, 8, 2000).astype(float)
    q = _x(0.05, 50.0, 2000)
    _close(bff.zeta_array(s, q), special.zeta(s, q), rtol=0.0)


def test_fresnel_returns_s_then_c():
    x = np.concatenate([_x(-50.0, 50.0), [0.0, 1.6, 40000.0, np.inf]])
    s, c = bff.fresnel_array(x)
    ss, cc = special.fresnel(x)
    _close(s, ss, rtol=0.0)
    _close(c, cc, rtol=0.0)


def test_f_and_chi2_tails_match_scipy():
    dfn = RNG.integers(1, 40, 3000).astype(float)
    dfd = RNG.integers(1, 200, 3000).astype(float)
    x = _x(0.0, 20.0, 3000)
    p = _x(1e-6, 1 - 1e-6, 3000)
    # The F distribution is Boost's in scipy 1.18 (f_cdf_wrap and friends),
    # and scipy vendors its own Boost: the two agree to ~1e-13, not the bit.
    _close(bff.fdtr_array(dfn, dfd, x), stats.f.cdf(x, dfn, dfd), rtol=1e-12)
    _close(bff.fdtrc_array(dfn, dfd, x), stats.f.sf(x, dfn, dfd), rtol=1e-12)
    _close(bff.fdtri_array(dfn, dfd, p), stats.f.ppf(p, dfn, dfd), rtol=1e-12)
    df = RNG.integers(1, 60, 3000).astype(float)
    _close(bff.chdtrc_array(df, x * 5), stats.chi2.sf(x * 5, df), rtol=0.0)
    _close(bff.chdtri_array(df, p), stats.chi2.isf(p, df), rtol=0.0)


def test_boost_backed_distributions():
    a = _x(0.05, 30.0, 3000)
    b = _x(0.05, 30.0, 3000)
    x = _x(0.0, 1.0, 3000)
    p = _x(0.0, 1.0, 3000)
    _close(bff.betainc_array(a, b, x), special.betainc(a, b, x), rtol=1e-13)
    _close(bff.betaincinv_array(a, b, p), stats.beta.ppf(p, a, b), rtol=1e-12)
    df = _x(0.5, 200.0, 3000)
    t = _x(-30.0, 30.0, 3000)
    _close(bff.stdtr_array(df, t), stats.t.cdf(t, df), rtol=1e-13)
    n = RNG.integers(0, 400, 3000).astype(float)
    k = np.floor(RNG.uniform(0, 1, 3000) * (n + 1))
    pp = _x(0.0, 1.0, 3000)
    _close(bff.binom_pmf_array(k, n, pp), stats.binom.pmf(k, n, pp), rtol=1e-13, atol=1e-300)
    xx = _x(0.0, 400.0, 3000)
    nc = _x(0.0, 300.0, 3000)
    _close(bff.ncx2_pdf_array(xx, np.full(3000, 3.0), nc), stats.ncx2.pdf(xx, 3.0, nc),
           rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 12])
def test_expm_matches_scipy(n):
    for scale in (1e-3, 1.0, 30.0):
        a = RNG.normal(size=(n, n)) * scale
        _close(bff.expm(a), linalg.expm(a), rtol=1e-11, atol=1e-13 * np.abs(linalg.expm(a)).max())


def test_expm_of_a_rate_matrix_is_a_stochastic_matrix():
    k = RNG.uniform(0.0, 5.0, (4, 4))
    np.fill_diagonal(k, 0.0)
    np.fill_diagonal(k, -k.sum(axis=0))
    p = bff.expm(k * 0.7)
    np.testing.assert_allclose(p.sum(axis=0), 1.0, rtol=0, atol=1e-13)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        bff.gammainc_array(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        bff.expm(np.ones((2, 3)))
