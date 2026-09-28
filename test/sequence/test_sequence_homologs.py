"""select_sequence_homologs: ConSurf's rules for which hits become homologues."""

import pytest

import IMP.bff as bff

Q = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"   # 30 residues


def _hit(name, evalue, identity, aligned, target=None, query=0):
    h = bff.SequenceSearchHit()
    h.identifier = name
    h.query = query
    h.evalue = evalue
    h.identity = identity
    h.aligned = aligned
    h.query_length = len(Q)
    n = sum(c != "-" for c in aligned)
    h.query_start, h.query_end = 0, len(Q)
    h.target_start, h.target_end = 0, n if target is None else target
    h.target_length = n
    return h


def _names(hits):
    return [h.identifier for h in hits]


def _variant(k, n_diff):
    """Q with n_diff substitutions at positions k, k+1, ..."""
    s = list(Q)
    for j in range(n_diff):
        p = (k + 3 * j) % len(s)
        s[p] = "W" if s[p] != "W" else "A"
    return "".join(s)


def test_each_rule_excludes_what_it_should():
    hits = [
        _hit("good", 1e-20, 0.6, _variant(0, 12)),
        _hit("weak", 1e-3, 0.6, _variant(1, 12)),                       # E > 1e-4
        _hit("distant", 1e-10, 0.30, _variant(2, 12)),                  # identity < 35 %
        _hit("twin", 1e-30, 0.97, _variant(3, 1)),                      # identity >= 95 %
        _hit("short", 1e-10, 0.6, "-" * 15 + _variant(4, 6)[15:]),      # 15 of 30 < 60 %
        _hit("other", 1e-10, 0.5, _variant(5, 14), query=1),            # another query's
    ]
    assert _names(bff.select_sequence_homologs(hits)) == ["good"]
    standalone = bff.SequenceHomologOptions.consurf_standalone()
    assert standalone.max_evalue == 1e-3 and standalone.min_identity == 0
    assert _names(bff.select_sequence_homologs(hits, standalone)) == ["good", "distant", "weak"]


def test_redundancy_keeps_the_longest_of_near_identical_homologues():
    a = _variant(0, 12)
    near = a[:-1] + "-"                    # the same residues, one fewer: 29/29 identical
    hits = [_hit("shorter", 1e-30, 0.6, near), _hit("longer", 1e-20, 0.6, a)]
    assert _names(bff.select_sequence_homologs(hits)) == ["longer"]
    o = bff.SequenceHomologOptions()
    o.redundancy = 1.01                    # nothing is redundant
    assert _names(bff.select_sequence_homologs(hits, o)) == ["shorter", "longer"]


def test_the_cap_samples_evenly_or_takes_the_best():
    hits = [_hit(f"h{k}", 10.0 ** -(40 - k), 0.5, _variant(k, 12 + k % 3)) for k in range(10)]
    o = bff.SequenceHomologOptions()
    o.redundancy = 1.01
    o.max_homologs = 4
    assert _names(bff.select_sequence_homologs(hits, o)) == ["h0", "h3", "h6", "h9"]
    o.sampling = "best"
    assert _names(bff.select_sequence_homologs(hits, o)) == ["h0", "h1", "h2", "h3"]
    o.sampling = "random"
    with pytest.raises(bff.ValueException):
        bff.select_sequence_homologs(hits, o)
