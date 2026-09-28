"""Mean-field DCA: A/B against the reference script's output, and toys."""

import os

import numpy as np
import pytest

import IMP.bff as bff

DATA = os.path.join(os.path.dirname(__file__), "..", "input", "sequence")


def _msa(seqs):
    return bff.SequenceMSA([f"s{k}" for k in range(len(seqs))], seqs)


def test_matches_the_reference_script():
    msa = bff.read_sequence_msa(os.path.join(DATA, "rbp_60x120.fasta"))
    co = bff.compute_sequence_coevolution(msa, 0.2, 0.5)
    ref = np.loadtxt(os.path.join(DATA, "dca_reference.dat"))
    i, j = ref[:, 0].astype(int) - 1, ref[:, 1].astype(int) - 1
    di = np.asarray(co.get_direct_information_matrix())
    mi = np.asarray(co.get_mutual_information_matrix())
    np.testing.assert_allclose(mi[i, j], ref[:, 2], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(di[i, j], ref[:, 3], rtol=1e-8, atol=1e-12)
    assert di.shape == (120, 120) and np.allclose(di, di.T)
    # the script prints Meff truncated to an integer: 28
    assert 28.0 <= co.get_n_effective() < 29.0


def test_copied_columns_couple_and_independent_ones_do_not():
    rng = np.random.default_rng(0)
    alphabet = bff.get_sequence_alphabet()
    seqs = []
    for _ in range(200):
        a = alphabet[rng.integers(0, 20)]
        seqs.append(a + a + alphabet[rng.integers(0, 20)] + alphabet[rng.integers(0, 20)])
    co = bff.compute_sequence_coevolution(_msa(seqs), 0.0)
    coupled = co.get_direct_information(0, 1)
    assert coupled > 5 * max(co.get_direct_information(0, 2), co.get_direct_information(2, 3))


def test_probe_pairs_map_structure_residues_to_columns():
    # reference: A C - D E; residues 1..4 -> columns 0, 1, 3, 4
    seqs = ["AC-DE", "AC-DE", "WCMDE", "ACMKE", "WYMDQ"]
    co = bff.compute_sequence_coevolution(_msa(seqs), 0.0)
    pairs = np.array([[11, 12], [11, 14], [13, 99]], dtype=np.int32)
    di = bff.probe_pair_coevolution(co, pairs, residue_offset=10)
    assert di[0] == pytest.approx(co.get_direct_information(0, 1))
    assert di[1] == pytest.approx(co.get_direct_information(0, 4))
    assert np.isnan(di[2])


def test_bad_inputs():
    with pytest.raises(bff.ValueException):
        bff.compute_sequence_coevolution(_msa(["A", "C"]))
    with pytest.raises(bff.ValueException):
        bff.compute_sequence_coevolution(_msa(["AC", "CA"]), 0.2, 1.5)
