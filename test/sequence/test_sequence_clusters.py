"""Clustered search: membership from a mapping table, then representatives
first and only the members of the clusters found."""

import gzip
import os
import random

import pytest

import IMP.bff as bff

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "input", "sequence", "rbp_60x120.fasta")
AA = "ACDEFGHIKLMNPQRSTVWY"


def _row(acc, u90, u50):
    # UniProt's idmapping_selected layout: UniRef90 in column 9, UniRef50 in 10
    cols = [acc, acc + "_HUMAN", "", "", "", "", "", "UniRef100_" + acc, u90, u50, "UPI0"]
    return "\t".join(cols) + "\n"


@pytest.fixture(scope="module")
def clustered(tmp_path_factory):
    d = tmp_path_factory.mktemp("clusters")
    msa = bff.read_sequence_msa(FIXTURE, match_columns_only=False)
    family = [msa.get_sequence(k).replace("-", "") for k in range(msa.get_n_sequences())]
    rng = random.Random(9)
    decoys = ["".join(rng.choice(AA) for _ in range(rng.randint(60, 300))) for _ in range(1500)]
    members, reps, mapping = [], [], []
    # the family: one UniRef50 cluster, its members spread over the table,
    # all but the last placed by the mapping (the last is an orphan)
    for k, s in enumerate(family):
        members.append((f"UniRef90_F{k}", s))
        if k < len(family) - 1:
            mapping.append(_row(f"F{k}", f"UniRef90_F{k}", "UniRef50_F0"))
    reps.append(("UniRef50_F0", family[0]))
    for k, s in enumerate(decoys):
        members.append((f"UniRef90_D{k}", s))
        mapping.append(_row(f"D{k}", f"UniRef90_D{k}", f"UniRef50_D{k}"))
        mapping.append(_row(f"D{k}b", f"UniRef90_D{k}", f"UniRef50_D{k}"))   # two accessions
        reps.append((f"UniRef50_D{k}", s))
    rng.shuffle(members)
    for name, records in (("members", members), ("reps", reps)):
        with open(d / f"{name}.fasta", "w") as fh:
            for i, s in records:
                fh.write(f">{i} n=1 Tax=x\n{s}\n")
        bff.create_sequence_database(str(d / f"{name}.fasta"), str(d / f"{name}.pto"), "sequences", 1)
    with gzip.open(d / "idmapping.tab.gz", "wt") as fh:
        fh.writelines(mapping)
    placed = bff.create_sequence_clusters(str(d / "members.pto"), str(d / "reps.pto"),
                                          str(d / "idmapping.tab.gz"))
    assert placed == len(members) - 1
    return d, family


def test_membership_is_recorded(clustered):
    d, family = clustered
    members = bff.SequenceDatabase(str(d / "members.pto"))
    reps = bff.SequenceDatabase(str(d / "reps.pto"))
    clusters = bff.SequenceClusters(str(d / "reps.pto"))
    assert clusters.get_number_of_clusters() == reps.get_number_of_sequences()
    fam = sorted(members.get_identifier(r) for r in clusters.get_members(0))
    assert fam == sorted(f"UniRef90_F{k}" for k in range(len(family) - 1))
    assert [members.get_identifier(r) for r in clusters.get_orphans()] == [f"UniRef90_F{len(family) - 1}"]
    assert [members.get_identifier(r) for r in clusters.get_members(5)] == [
        reps.get_identifier(5).replace("UniRef50", "UniRef90")]
    # the representatives' sequences still read
    assert reps.get_sequence(0) == family[0]


def test_two_stages_find_what_one_does(clustered):
    d, family = clustered
    members = bff.SequenceDatabase(str(d / "members.pto"))
    reps = bff.SequenceDatabase(str(d / "reps.pto"))
    clusters = bff.SequenceClusters(str(d / "reps.pto"))
    queries = [family[3], family[20]]
    full = bff.search_sequence_database(queries, members)
    two = bff.search_clustered_sequence_database(queries, reps, clusters, members)
    key = lambda hits: sorted((h.query, h.identifier, h.score) for h in hits)
    assert key(two) == key(full)
    assert [h.evalue for h in two] == pytest.approx([h.evalue for h in
                                                     sorted(full, key=lambda h: (h.query, h.evalue, h.target))])
    # without the orphans, the one unplaced member is missed
    o = bff.SequenceClusterSearchOptions()
    o.search_orphans = False
    some = bff.search_clustered_sequence_database(queries, reps, clusters, members,
                                                  bff.SequenceSearchOptions(), o)
    orphan = f"UniRef90_F{len(family) - 1}"
    assert orphan in {h.identifier for h in full}
    assert orphan not in {h.identifier for h in some}


def test_a_membership_is_checked_against_its_representatives(clustered):
    d, family = clustered
    members = bff.SequenceDatabase(str(d / "members.pto"))
    clusters = bff.SequenceClusters(str(d / "reps.pto"))
    with pytest.raises(bff.ValueException):
        bff.search_clustered_sequence_database([family[0]], members, clusters, members)
    with pytest.raises(bff.IOException):
        bff.SequenceClusters(str(d / "members.pto"))


def test_consurf_searches_in_two_stages_when_the_settings_say_so(clustered, tmp_path, monkeypatch):
    import json
    d, family = clustered
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"sequence_search": {
        "databases": {"u90": str(d / "members.pto")}, "default_database": "u90",
        "clusters": {"u90": str(d / "reps.pto")}}}))
    monkeypatch.setenv("IMP_BFF_SETTINGS", str(settings))
    assert bff.get_sequence_search_settings().get_representatives("u90") == str(d / "reps.pto")
    options = bff.ConsurfOptions()
    options.homologs = bff.SequenceHomologOptions.consurf_standalone()
    two = bff.compute_consurf([family[3]], options)[0]
    options.database = str(d / "members.pto")          # a path: no clusters named for it
    one = bff.compute_consurf([family[3]], options)[0]
    assert two.get_is_ok() and [h.identifier for h in two.homologs] == [h.identifier for h in one.homologs]
    assert list(two.grades) == list(one.grades)


def test_repacking_keeps_the_membership(clustered, tmp_path):
    d, family = clustered
    out = str(tmp_path / "reps_bytes.pto")
    bff.repack_sequence_database(str(d / "reps.pto"), out, False)
    a, b = bff.SequenceClusters(str(d / "reps.pto")), bff.SequenceClusters(out)
    assert b.get_number_of_clusters() == a.get_number_of_clusters()
    assert list(b.get_members(0)) == list(a.get_members(0))
    assert list(b.get_orphans()) == list(a.get_orphans())
    assert not bff.SequenceDatabase(out).get_is_packed()


def test_uniref_xml_places_what_the_table_cannot(clustered, tmp_path):
    d, family = clustered
    members = bff.SequenceDatabase(str(d / "members.pto"))
    reps = bff.SequenceDatabase(str(d / "reps.pto"))
    # the same clusters as XML, with the orphan (a UniParc-only member) placed
    ids = [members.get_identifier(i) for i in range(members.get_number_of_sequences())]
    lines = ['<?xml version="1.0"?>', '<UniRef50 xmlns="http://uniprot.org/uniref">']
    def entry(rep, member_ids):
        lines.append(f'<entry id="{rep}" updated="2026-09-02">')
        lines.append('<name>Cluster: x</name>')
        for m in member_ids:
            lines.append('<member><dbReference type="UniParc ID" id="UPI0">')
            lines.append(f'<property type="UniRef90 ID" value="{m}"/>')
            lines.append('</dbReference></member>')
        lines.append('<sequence length="3">ACD</sequence></entry>')
    entry("UniRef50_F0", [f"UniRef90_F{k}" for k in range(len(family))])
    for k in range(reps.get_number_of_sequences() - 1):
        entry(f"UniRef50_D{k}", [f"UniRef90_D{k}"])
    lines.append("</UniRef50>")
    xml = tmp_path / "uniref50.xml.gz"
    with gzip.open(xml, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    out = str(tmp_path / "reps.pto")
    bff.repack_sequence_database(str(d / "reps.pto"), out)
    placed = bff.create_sequence_clusters(str(d / "members.pto"), out, str(xml))
    assert placed == members.get_number_of_sequences()
    c = bff.SequenceClusters(out)
    assert c.get_number_of_orphans() == 0
    assert sorted(members.get_identifier(r) for r in c.get_members(0)) == sorted(
        f"UniRef90_F{k}" for k in range(len(family)))


def test_a_clustered_database_searches_like_the_member_database(clustered, tmp_path):
    d, family = clustered
    out = str(tmp_path / "clustered.pto")
    n = bff.create_clustered_sequence_database(str(d / "members.pto"), str(d / "reps.pto"), out)
    members = bff.SequenceDatabase(str(d / "members.pto"))
    assert n == members.get_number_of_sequences()
    db = bff.SequenceDatabase(out)
    assert db.get_has_members() and not bff.SequenceDatabase(str(d / "reps.pto")).get_has_members()
    m = db.get_members_database()
    assert m.get_number_of_sequences() == members.get_number_of_sequences()
    assert m.get_number_of_residues() == members.get_number_of_residues()
    # the family's cluster holds its placed members, contiguously; the orphan is last
    first, end = db.get_member_range(0)
    assert sorted(m.get_identifier(r) for r in range(first, end)) == sorted(
        f"UniRef90_F{k}" for k in range(len(family) - 1))
    assert db.get_identifier(db.get_number_of_sequences() - 1) == "orphans"
    last = db.get_member_range(db.get_number_of_sequences() - 1)
    assert [m.get_identifier(r) for r in range(*last)] == [f"UniRef90_F{len(family) - 1}"]
    # every member once, sequences intact
    ids = {members.get_identifier(i): i for i in range(members.get_number_of_sequences())}
    for r in range(0, m.get_number_of_sequences(), 7):
        assert m.get_sequence(r) == members.get_sequence(ids[m.get_identifier(r)])
    queries = [family[3], family[20]]
    full = bff.search_sequence_database(queries, members)
    two = bff.search_clustered_sequence_database(queries, db)
    key = lambda hits: sorted((h.query, h.identifier, h.score) for h in hits)
    assert key(two) == key(full)


def test_clusters_of_close_members_chain_through_siblings(tmp_path):
    # members close to each other and far from their representative: stored
    # against siblings, chains limited, several clusters in a row
    rng = random.Random(21)
    reps, members, mapping = [], [], []
    for k in range(3):
        rep = "".join(rng.choice(AA) for _ in range(300))
        reps.append((f"UniRef50_R{k}", rep))
        centre = "".join(c if rng.random() < 0.4 else rng.choice(AA) for c in rep)
        for j in range(30):
            m = "".join(c if rng.random() > 0.02 else rng.choice(AA) for c in centre)
            members.append((f"UniRef90_M{k}_{j}", m))
            mapping.append(_row(f"M{k}_{j}", f"UniRef90_M{k}_{j}", f"UniRef50_R{k}"))
    for name, records in (("members", members), ("reps", reps)):
        with open(tmp_path / f"{name}.fasta", "w") as fh:
            for i, s in records:
                fh.write(f">{i}\n{s}\n")
        bff.create_sequence_database(str(tmp_path / f"{name}.fasta"), str(tmp_path / f"{name}.pto"))
    with gzip.open(tmp_path / "map.tab.gz", "wt") as fh:
        fh.writelines(mapping)
    bff.create_sequence_clusters(str(tmp_path / "members.pto"), str(tmp_path / "reps.pto"),
                                 str(tmp_path / "map.tab.gz"))
    out = str(tmp_path / "clustered.pto")
    assert bff.create_clustered_sequence_database(str(tmp_path / "members.pto"),
                                                  str(tmp_path / "reps.pto"), out) == 90
    m = bff.SequenceDatabase(out).get_members_database()
    want = dict(members)
    for r in range(m.get_number_of_sequences()):
        assert m.get_sequence(r) == want[m.get_identifier(r)]
    # far smaller than the members stored alone
    assert os.path.getsize(out) < os.path.getsize(tmp_path / "members.pto")


def test_early_stop_takes_the_best_clusters_first(clustered, tmp_path):
    d, family = clustered
    out = str(tmp_path / "clustered.pto")
    bff.create_clustered_sequence_database(str(d / "members.pto"), str(d / "reps.pto"), out)
    db = bff.SequenceDatabase(out)
    queries = [family[3]]
    full = bff.search_clustered_sequence_database(queries, db)
    o = bff.SequenceClusterSearchOptions()
    o.min_homologues = 1
    some = bff.search_clustered_sequence_database(queries, db, bff.SequenceSearchOptions(), o)
    full_ids = [h.identifier for h in full]
    assert set(h.identifier for h in some) <= set(full_ids)
    good = [h for h in some if h.evalue <= 1e-4 and h.identity >= 0.35]
    assert len(good) >= 1
    assert some[0].identifier == full_ids[0]


def _embedding_prefilter(d, reps_path, tmp_path):
    """The tiny test model's projections of every representative, indexed."""
    import numpy as np
    model_path = os.path.join(os.path.dirname(__file__), "..", "input", "sequence", "esm2_tiny.gguf")
    model = bff.ProteinLanguageModel(model_path)
    reps = bff.SequenceDatabase(reps_path)
    x = np.array([model.get_projected_embedding(reps.get_sequence(i))
                  for i in range(reps.get_number_of_sequences())], dtype=np.float32)
    np.save(tmp_path / "reps.npy", x)
    index = str(tmp_path / "reps_esm.pto")
    bff.create_embedding_index([str(tmp_path / "reps.npy")], index, "tiny", 4, 2000, 5)
    return model_path, index


def test_embedding_prefilter_aligning_every_representative_is_the_kmer_search(clustered, tmp_path):
    d, family = clustered
    members = bff.SequenceDatabase(str(d / "members.pto"))
    reps = bff.SequenceDatabase(str(d / "reps.pto"))
    clusters = bff.SequenceClusters(str(d / "reps.pto"))
    model, index = _embedding_prefilter(d, str(d / "reps.pto"), tmp_path)
    queries = [family[3], family[20]]
    kmer = bff.search_clustered_sequence_database(queries, reps, clusters, members)
    o = bff.SequenceClusterSearchOptions()
    o.embedding_model, o.embedding_index = model, index
    o.embedding_candidates = reps.get_number_of_sequences()
    embedded = bff.search_clustered_sequence_database(queries, reps, clusters, members,
                                                      bff.SequenceSearchOptions(), o)
    key = lambda hits: sorted((h.query, h.identifier, h.score, round(h.evalue, 12)) for h in hits)
    assert key(embedded) == key(kmer)
    # fewer candidates: a subset of the same hits
    o.embedding_candidates = 50
    few = bff.search_clustered_sequence_database(queries, reps, clusters, members,
                                                 bff.SequenceSearchOptions(), o)
    assert set(key(few)) <= set(key(kmer))


def test_embedding_prefilter_needs_both_paths(clustered):
    d, family = clustered
    o = bff.SequenceClusterSearchOptions()
    o.embedding_model = "model.gguf"
    with pytest.raises(bff.ValueException):
        bff.search_clustered_sequence_database(
            [family[0]], bff.SequenceDatabase(str(d / "reps.pto")),
            bff.SequenceClusters(str(d / "reps.pto")), bff.SequenceDatabase(str(d / "members.pto")),
            bff.SequenceSearchOptions(), o)
