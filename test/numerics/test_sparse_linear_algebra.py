"""SparseLU / sparse_solve against scipy.sparse.linalg.spsolve (the oracle)."""
import numpy as np
import pytest

import IMP.bff as bff

sp = pytest.importorskip("scipy.sparse")
spl = pytest.importorskip("scipy.sparse.linalg")


def _arrays(m):
    return m.indptr.astype(np.int32), m.indices.astype(np.int32), m.data.astype(float)


def _fem_laplacian(nx, ny, dt=0.1):
    """I - dt * L for a 5-point Laplacian on an nx x ny grid (CSR)."""
    n = nx * ny
    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            rows.append(k), cols.append(k), vals.append(1.0 + 4.0 * dt)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny:
                    rows.append(k), cols.append(a * ny + b), vals.append(-dt)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _matrices():
    rng = np.random.default_rng(11)
    for seed, n, density in ((1, 50, 0.1), (2, 300, 0.02), (3, 1000, 0.005)):
        a = sp.random(n, n, density=density, random_state=seed, format="csr")
        a = a + sp.diags(np.abs(a).sum(axis=1).A1 + 1.0)  # diagonally dominant
        yield a.tocsr(), rng.normal(size=n)
    for nx, ny in ((10, 10), (40, 25)):
        a = _fem_laplacian(nx, ny)
        yield a, rng.normal(size=a.shape[0])
    # nonsymmetric advection-diffusion-like
    a = _fem_laplacian(20, 20).tolil()
    for k in range(399):
        a[k, k + 1] += 0.05
    yield a.tocsr(), rng.normal(size=400)


@pytest.mark.parametrize("case", range(6))
def test_matches_spsolve_csr_and_csc(case):
    a, b = list(_matrices())[case]
    want = spl.spsolve(a, b)
    n = a.shape[0]
    x = bff.sparse_solve(n, *_arrays(a.tocsr()), b, True)
    np.testing.assert_allclose(x, want, rtol=1e-10, atol=1e-12 * np.abs(want).max())
    lu = bff.SparseLU()
    lu.factorize(n, *_arrays(a.tocsc()), False)
    for _ in range(3):  # factorized semantics: reuse the factors
        np.testing.assert_allclose(lu.solve(b), want, rtol=1e-10, atol=1e-12 * np.abs(want).max())
    assert lu.get_size() == n


def test_duplicates_are_summed_like_scipy():
    rows = np.array([0, 0, 1, 2, 2, 2])
    cols = np.array([0, 0, 1, 2, 0, 2])
    vals = np.array([1.0, 2.0, 4.0, 1.0, 1.0, 4.0])
    m = sp.csr_matrix((vals, (rows, cols)), shape=(3, 3))
    # Raw arrays with the duplicate kept: indptr/indices built by hand.
    indptr = np.array([0, 2, 3, 6], np.int32)
    indices = np.array([0, 0, 1, 2, 0, 2], np.int32)
    b = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        bff.sparse_solve(3, indptr, indices, vals, b, True), spl.spsolve(m, b), rtol=1e-14
    )


def test_singular_raises_value_error():
    z = sp.csr_matrix(np.array([[1.0, 2.0], [2.0, 4.0]]))
    with pytest.raises(ValueError, match="singular"):
        bff.sparse_solve(2, *_arrays(z), np.ones(2), True)


def test_malformed_arrays_raise():
    with pytest.raises(ValueError):
        bff.sparse_solve(3, np.array([0, 1], np.int32), np.array([0], np.int32), np.ones(1), np.ones(3), True)
    lu = bff.SparseLU()
    with pytest.raises(ValueError):
        lu.solve(np.ones(3))
