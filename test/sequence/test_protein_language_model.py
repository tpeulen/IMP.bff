"""ProteinLanguageModel: ESM-2 from GGUF, forward pass in C++.

The fixture is a tiny random ESM-2 (2 layers, width 16, 4 heads, a random
projection head) whose reference embeddings were computed by Hugging Face
transformers in FP32 (prototypes/esm_retrieval/make_tiny.py). The real 35M
model is checked too when IMP_BFF_ESM_GGUF names it.
"""
import math
import os

import pytest

import IMP.bff as bff

HERE = os.path.dirname(__file__)
TINY = os.path.join(HERE, "..", "input", "sequence", "esm2_tiny.gguf")
TINY_REF = os.path.join(HERE, "..", "input", "sequence", "esm2_tiny_ref.tsv")


def references(path):
    """(sequence, pooled, projected, contacts n x n row by row) per line; a
    column may be empty (no projection head; contacts only for short ones)."""
    out = []
    for line in open(path):
        fields = line.rstrip("\n").split("\t") + [""]
        nums = [[float(v) for v in f.split(",")] if f else [] for f in fields[1:4]]
        out.append((fields[0], *nums))
    return out


def test_tiny_model_metadata():
    m = bff.ProteinLanguageModel(TINY)
    assert m.get_embedding_length() == 16
    assert m.get_number_of_layers() == 2
    assert m.get_projection_length() == 8
    assert m.get_max_length() == 1022


@pytest.mark.parametrize("row", range(3))
def test_tiny_model_matches_transformers(row):
    m = bff.ProteinLanguageModel(TINY)
    seq, pooled, projected, _ = references(TINY_REF)[row]
    got = m.get_embedding(seq)
    assert max(abs(a - b) for a, b in zip(got, pooled)) < 1e-4
    got = m.get_projected_embedding(seq)
    assert max(abs(a - b) for a, b in zip(got, projected)) < 1e-4
    assert math.isclose(sum(v * v for v in got), 1.0, rel_tol=1e-5)


def test_residue_embeddings_average_to_the_pooled_one():
    m = bff.ProteinLanguageModel(TINY)
    seq = "MKTAYIAKQRQISFVKSHFSRQ"
    per = m.get_residue_embeddings(seq)
    d = m.get_embedding_length()
    assert len(per) == len(seq) * d
    mean = [sum(per[i * d + j] for i in range(len(seq))) / len(seq) for j in range(d)]
    assert max(abs(a - b) for a, b in zip(mean, m.get_embedding(seq))) < 1e-5


def test_lower_case_and_truncation():
    m = bff.ProteinLanguageModel(TINY)
    assert m.get_embedding("mktay") == pytest.approx(m.get_embedding("MKTAY"))
    long = "ACDEFGHIKLMNPQRSTVWY" * 60              # 1200 residues
    assert m.get_embedding(long) == pytest.approx(m.get_embedding(long[:1022]))


def test_not_a_model(tmp_path):
    bad = tmp_path / "x.gguf"
    bad.write_bytes(b"nope")
    with pytest.raises(bff.IOException):
        bff.ProteinLanguageModel(str(bad))


@pytest.mark.skipif(not os.environ.get("IMP_BFF_ESM_GGUF"), reason="IMP_BFF_ESM_GGUF not set")
def test_esm2_35m():
    path = os.environ["IMP_BFF_ESM_GGUF"]
    m = bff.ProteinLanguageModel(path)
    for seq, pooled, projected, _ in references(path.replace("_head.gguf", "_ref.tsv")):
        got = m.get_embedding(seq)
        # F16 weights: the reference ran on FP32 ones
        assert max(abs(a - b) for a, b in zip(got, pooled)) < 2e-2
        cos = sum(a * b for a, b in zip(m.get_projected_embedding(seq), projected))
        assert cos > 0.999


def test_tiny_contacts_match_transformers():
    import numpy as np
    m = bff.ProteinLanguageModel(TINY)
    assert m.get_has_contact_head()
    checked = 0
    for seq, _, _, contacts in references(TINY_REF):
        if not contacts:
            continue
        c = m.get_contacts(seq)
        n = len(seq)
        assert c.shape == (n, n)
        ref = np.array(contacts).reshape(n, n)
        assert np.max(np.abs(c - ref)) < 1e-4
        assert np.allclose(c, c.T)
        checked += 1
    assert checked >= 2


def test_probe_pair_contacts_reads_the_map_by_structure_numbering():
    import numpy as np
    m = bff.ProteinLanguageModel(TINY)
    seq = "ACDEFGHIKLMNPQRSTVWY"
    c = m.get_contacts(seq)
    # structure numbering starts at 101: residue 101 is sequence position 1
    pairs = np.array([[101, 105], [110, 120], [101, 101], [99, 105], [101, 121]], dtype=np.int32)
    v = bff.probe_pair_contacts(m, seq, pairs, 100)
    assert abs(v[0] - c[0, 4]) < 1e-12 and abs(v[1] - c[9, 19]) < 1e-12
    assert all(np.isnan(x) for x in v[2:])          # same residue, outside the sequence
    with pytest.raises(bff.ValueException):
        bff.probe_pair_contacts(m, seq, np.array([[1, 2, 3]], dtype=np.int32))


def test_a_model_without_contact_head_says_so():
    if not os.environ.get("IMP_BFF_ESM_GGUF"):
        pytest.skip("IMP_BFF_ESM_GGUF not set")
    m = bff.ProteinLanguageModel(os.environ["IMP_BFF_ESM_GGUF"])
    if m.get_has_contact_head():
        pytest.skip("this model file carries a contact head")
    with pytest.raises(bff.ValueException):
        m.get_contacts("ACDEFGHIKLMNPQRSTVWY")


@pytest.mark.skipif(not os.environ.get("IMP_BFF_ESM_CONTACT_GGUF"), reason="IMP_BFF_ESM_CONTACT_GGUF not set")
def test_esm2_contact_model():
    import numpy as np
    path = os.environ["IMP_BFF_ESM_CONTACT_GGUF"]
    m = bff.ProteinLanguageModel(path)
    for seq, _, _, contacts in references(path.replace(".gguf", "_ref.tsv")):
        if not contacts:
            continue
        n = len(seq)
        # F16 weights against an FP32 reference
        assert np.max(np.abs(m.get_contacts(seq) - np.array(contacts).reshape(n, n))) < 2e-2
