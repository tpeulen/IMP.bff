"""EmbeddingIndex: product-quantised nearest neighbours of unit vectors."""
import numpy as np
import pytest

import IMP.bff as bff


def families(n_families=40, per=25, d=64, spread=0.15, seed=1):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_families, d))
    x = np.repeat(centres, per, axis=0) + spread * rng.normal(size=(n_families * per, d)) * np.sqrt(d) / 4
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32), np.repeat(np.arange(n_families), per)


def test_neighbours_are_family(tmp_path):
    x, fam = families()
    np.save(tmp_path / "a.npy", x[:600])
    np.save(tmp_path / "b.npy", x[600:].astype(np.float16))      # float16 too
    out = str(tmp_path / "index.pto")
    n = bff.create_embedding_index([str(tmp_path / "a.npy"), str(tmp_path / "b.npy")], out,
                                   "test", 16, 1000, 10)
    assert n == len(x)
    index = bff.EmbeddingIndex(out)
    assert index.get_number_of_vectors() == len(x)
    assert index.get_dimension() == 64
    assert index.get_number_of_subspaces() == 16
    assert index.get_model() == "test"
    hits_self, same = 0, []
    for q in range(0, len(x), 37):
        rows = index.get_nearest(list(map(float, x[q])), 25)
        assert len(rows) == 25 and len(set(rows)) == 25
        hits_self += rows[0] == q or q in rows[:3]
        same.append(np.mean(fam[rows] == fam[q]))
        scores = index.get_nearest_scores(list(map(float, x[q])), 25)
        assert all(a >= b for a, b in zip(scores, scores[1:]))
        assert abs(scores[0] - float(x[q] @ x[rows[0]])) < 0.1
    assert hits_self >= 0.9 * len(same)
    assert np.mean(same) > 0.95


def test_rows_map_reports_a_row_once(tmp_path):
    x, _ = families(n_families=10, per=10)
    # every vector twice: the second copy maps to the same row
    np.save(tmp_path / "v.npy", np.concatenate([x, x]))
    (tmp_path / "rows.txt").write_text("\n".join(str(i % len(x)) for i in range(2 * len(x))))
    out = str(tmp_path / "index.pto")
    bff.create_embedding_index([str(tmp_path / "v.npy")], out, "", 8, 500, 5,
                               str(tmp_path / "rows.txt"))
    index = bff.EmbeddingIndex(out)
    assert index.get_number_of_vectors() == 2 * len(x)
    rows = index.get_nearest(list(map(float, x[3])), 20)
    assert len(rows) == len(set(rows)) == 20
    assert max(rows) < len(x)


def test_bad_subspaces(tmp_path):
    np.save(tmp_path / "v.npy", np.ones((10, 10), dtype=np.float32))
    with pytest.raises(bff.ValueException):
        bff.create_embedding_index([str(tmp_path / "v.npy")], str(tmp_path / "i.pto"), "", 3)
