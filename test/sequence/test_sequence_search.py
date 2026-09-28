"""SequenceSearch: BLOSUM62, affine Smith-Waterman, the database search, the
query-anchored alignment."""

import os
import random

import pytest

import IMP.bff as bff

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "input", "sequence", "rbp_60x120.fasta")
AA = "ACDEFGHIKLMNPQRSTVWY"


def test_blosum62_is_the_published_matrix():
    matrices = pytest.importorskip("Bio.Align.substitution_matrices")
    ref = matrices.load("BLOSUM62")
    for a in AA:
        for b in AA:
            assert bff.get_blosum62(a, b) == ref[a][b], (a, b)
    assert bff.get_blosum62("X", "A") == -1 and bff.get_blosum62("w", "W") == 11


def _gotoh(q, t, go=11, ge=1):
    """Brute-force affine local alignment score: a gap of length L costs go + L*ge."""
    neg = -10 ** 9
    m, n = len(q), len(t)
    H = [[0] * (n + 1) for _ in range(m + 1)]
    E = [[neg] * (n + 1) for _ in range(m + 1)]
    F = [[neg] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            E[i][j] = max(E[i][j - 1] - ge, H[i][j - 1] - go - ge)
            F[i][j] = max(F[i - 1][j] - ge, H[i - 1][j] - go - ge)
            H[i][j] = max(0, H[i - 1][j - 1] + bff.get_blosum62(q[i - 1], t[j - 1]), E[i][j], F[i][j])
            best = max(best, H[i][j])
    return best


def _mutate(rng, s, rate):
    out = []
    for c in s:
        r = rng.random()
        if r < rate / 3:
            continue                        # deletion
        if r < 2 * rate / 3:
            out.append(rng.choice(AA))      # substitution
        elif r < rate:
            out.append(c + rng.choice(AA))  # insertion
        else:
            out.append(c)
    return "".join(out)


def test_smith_waterman_scores_match_brute_force():
    rng = random.Random(3)
    for _ in range(25):
        q = "".join(rng.choice(AA) for _ in range(rng.randint(5, 40)))
        t = _mutate(rng, q, 0.4) + "".join(rng.choice(AA) for _ in range(rng.randint(0, 10)))
        hit = bff.align_sequences(q, t)
        assert hit.score == _gotoh(q, t)


def test_an_alignment_reads_back_as_itself():
    q = "MKTAYIAKQRQISFVKSHFSRQ"
    hit = bff.align_sequences(q, "GGG" + q + "PP")
    assert hit.identity == 1.0 and hit.aligned == q
    assert (hit.query_start, hit.query_end, hit.target_start, hit.target_end) == (0, len(q), 3, 3 + len(q))
    assert hit.get_query_coverage() == 1.0
    # a deletion in the target shows as '-' on the query's columns; an insertion is dropped
    hit = bff.align_sequences(q, q[:10] + q[12:])
    assert hit.aligned == q[:10] + "--" + q[12:] or hit.aligned.count("-") == 2
    hit = bff.align_sequences(q, q[:10] + "WWW" + q[10:])
    assert len(hit.aligned) == len(q) and "W" * 3 not in hit.aligned


def _family():
    msa = bff.read_sequence_msa(FIXTURE, match_columns_only=False)
    return [(msa.get_names()[k].split()[0], msa.get_sequence(k).replace("-", ""))
            for k in range(msa.get_n_sequences())]


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    d = tmp_path_factory.mktemp("search")
    rng = random.Random(11)
    family = _family()
    fasta = d / "db.fasta"
    with open(fasta, "w") as fh:
        for k in range(3000):     # decoys: random sequences, no homology
            fh.write(f">decoy{k}\n" + "".join(rng.choice(AA) for _ in range(rng.randint(50, 400))) + "\n")
            if k % 50 == 0 and k // 50 < len(family):
                name, seq = family[k // 50]
                fh.write(f">{name} family\n{seq}\n")
    out = str(d / "db.pto")
    bff.create_sequence_database(str(fasta), out, "sequences", 1)
    return bff.SequenceDatabase(out), family


def test_search_finds_the_family_and_no_decoy(database):
    db, family = database
    query = family[0][1]
    hits = bff.search_sequence_database([query], db)
    names = [h.identifier for h in hits]
    assert hits[0].identity == 1.0
    assert family[0][0] in [h.identifier for h in hits if h.identity == 1.0]
    assert not any(n.startswith("decoy") for n in names)
    # the family shares a fold, not necessarily detectable sequence similarity:
    # most members are found, and every hit is significant
    assert len(hits) >= 0.5 * len(family)
    assert all(h.evalue <= 1e-3 for h in hits)
    assert [h.evalue for h in hits] == sorted(h.evalue for h in hits)
    # the same search on one thread finds the same
    one = bff.search_sequence_database([query], db, bff.SequenceSearchOptions())
    assert [h.identifier for h in one] == names


def test_several_queries_in_one_pass(database):
    db, family = database
    queries = [family[0][1], family[7][1], "MKTAYIAKQRQISFVKSHFSRQ"]
    hits = bff.search_sequence_database(queries, db)
    by_query = {q: [h for h in hits if h.query == q] for q in range(3)}
    # the fixture repeats some sequences: the query is among the identical hits
    for q, (name, _) in ((0, family[0]), (1, family[7])):
        assert by_query[q][0].identity == 1.0
        assert name in [h.identifier for h in by_query[q] if h.identity == 1.0]
    assert by_query[2] == []
    alone = bff.search_sequence_database([family[7][1]], db)
    assert [h.identifier for h in alone] == [h.identifier for h in by_query[1]]


def test_hits_make_a_query_anchored_alignment(database):
    db, family = database
    query = family[0][1]
    hits = bff.search_sequence_database([query], db)
    msa = bff.get_query_msa(query, hits)
    assert msa.get_n_sequences() == len(hits) + 1
    assert msa.get_n_columns() == len(query)
    assert msa.get_sequence(0) == query
    assert list(msa.get_reference_positions()) == list(range(1, len(query) + 1))
