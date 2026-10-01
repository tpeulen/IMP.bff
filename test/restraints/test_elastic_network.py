"""Elastic network modes and the dynamics benefit of the probe-network selection."""
import math

import numpy as np
import pytest

import IMP.bff as bff


def hinge(seed=3, n=30):
    """Two compact blobs 30 A apart, bridged by a short linker: the softest
    motion is the blobs moving relative to each other."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 3)) * 4.0
    b = rng.normal(size=(n, 3)) * 4.0 + [30.0, 0.0, 0.0]
    # a jittered tube, not a line: springs along a straight line do not resist bending
    link = np.array([[x, 0.0, 0.0] for x in np.linspace(6.0, 24.0, 12)]) + rng.normal(size=(12, 3)) * 1.5
    return np.concatenate([a, link, b]), n


def reference_anm(xyz, cutoff):
    n = len(xyz)
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            d = xyz[j] - xyz[i]
            r2 = d @ d
            if r2 >= cutoff ** 2:
                continue
            blk = -np.outer(d, d) / r2
            H[3*i:3*i+3, 3*j:3*j+3] = blk
            H[3*j:3*j+3, 3*i:3*i+3] = blk
            H[3*i:3*i+3, 3*i:3*i+3] -= blk
            H[3*j:3*j+3, 3*j:3*j+3] -= blk
    return np.linalg.eigvalsh(H)


def test_modes_match_a_reference_and_drop_rigid_motion():
    xyz, _ = hinge()
    m = bff.ElasticNetworkModes(xyz, 12.0)
    ref = reference_anm(xyz, 12.0)
    assert np.allclose(ref[:6], 0.0, atol=1e-8)
    w = np.array(m.get_eigenvalues())
    assert m.get_number_of_modes() == 3 * len(xyz) - 6
    assert np.allclose(w, ref[6:], rtol=1e-8, atol=1e-10)
    assert np.all(np.diff(w) >= -1e-12)
    u = m.get_mode(0)
    assert u.shape == (len(xyz), 3) and abs(np.sum(u * u) - 1.0) < 1e-10


def test_the_hinge_moves_pairs_across_it():
    xyz, n = hinge()
    m = bff.ElasticNetworkModes(xyz, 12.0)
    across = np.array([[i, len(xyz) - 1 - i] for i in range(10)], dtype=np.int32)
    within = np.array([[i, i + 10] for i in range(10)], dtype=np.int32)
    f_across = bff.get_pair_distance_fluctuations(m, across, 1)
    f_within = bff.get_pair_distance_fluctuations(m, within, 1)
    assert min(f_across) > 10 * max(f_within)
    p_across = bff.get_pair_change_probabilities(m, across)
    p_within = bff.get_pair_change_probabilities(m, within)
    assert all(0.0 <= p <= 1.0 for p in list(p_across) + list(p_within))
    assert min(p_across) > max(p_within)


def test_invalid_pairs_and_inputs():
    xyz, _ = hinge()
    m = bff.ElasticNetworkModes(xyz, 12.0)
    bad = np.array([[0, 0], [-1, 3], [0, len(xyz)]], dtype=np.int32)
    assert all(math.isnan(v) for v in bff.get_pair_distance_fluctuations(m, bad))
    assert all(math.isnan(v) for v in bff.get_pair_change_probabilities(m, bad))
    with pytest.raises(bff.ValueException):
        bff.get_pair_distance_fluctuations(m, np.array([[0, 1]], dtype=np.int32), 0)
    with pytest.raises(bff.ValueException):
        bff.ElasticNetworkModes(np.zeros((2, 3)))
    with pytest.raises(bff.ValueException):                 # two blobs, no bridge at 5 A
        bff.ElasticNetworkModes(np.concatenate([xyz[:30], xyz[42:]]), 5.0)


def test_benefit_term_loss_is_the_chance_that_nothing_changes():
    term = bff.ProbePairBenefitTerm([0.5, 0.2, float("nan"), 1.0])
    assert term.get_loss() == 1.0
    assert term.get_is_eligible_pair(0) and not term.get_is_eligible_pair(2)
    assert abs(term.get_loss_with([0, 1], []) - 0.5 * 0.8) < 1e-12
    assert term.get_loss_with([3], []) < 1e-12           # a certain change
    term.commit([0], [])
    assert term.get_loss() == pytest.approx(0.5)
    assert term.get_loss_with([1], []) == pytest.approx(0.5 * 0.8)
    term.reset()
    assert term.get_loss() == 1.0
    with pytest.raises(bff.ValueException):
        bff.ProbePairBenefitTerm([0.1, 1.5])


def test_selection_by_benefit_prefers_pairs_that_change():
    d = [0.05, 0.9, 0.1, 0.7, 0.3]
    sel = bff.ProbeNetworkSelection(len(d))
    sel.add_term(bff.ProbePairBenefitTerm(d), 1.0)
    units, _ = sel.select(2)
    assert sorted(int(u) for u in units) == [1, 3]
