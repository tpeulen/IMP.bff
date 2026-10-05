"""The pair screen says which dye model produced each distance.

A pair is scored from the two sites' accessible volumes. When one of them
comes back empty the screen measures between the two attachment points
instead -- a reasonable fallback, and a different measurement -- and the row
used to be labelled as a volume-based distance either way, so nothing
downstream could separate them. On BmrA's closed state that was 1422 of 7750
rows resting on a site with no volume while presented as volume-based.
"""

import os

import IMP.bff as bff

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "input", "labelizer")


def _path(name):
    return os.path.join(DATA, name)


def _build_volume_the_way_the_screen_does(pdb, chain, seq_id, atom, options):
    """The screen builds its volumes through the installed AV door.

    An IMP build installs its own door (IMP::atom's reader and radii) over the
    core one, and the two do not agree site for site -- on this structure one
    finds 46 sites with no volume and the other 42, agreeing on 39. A check
    that rebuilt the volume with the wrong builder would be testing the
    difference between the two readers, not the thing this file is about.
    """
    builder = (bff.get_av_from_structure
               if bff.labelizer_get_av_door_name() == "imp"
               else bff.get_av_from_pdb)
    return builder(pdb, chain, seq_id, atom, options.linker_length,
                   options.linker_width, options.r1, options.r2, options.r3,
                   options.grid_resolution)


def test_a_pair_measured_between_attachment_points_says_so():
    pdb = _path("1DDB-39.pdb")
    structure = bff.labelizer_read_structure(pdb)

    # A slice of the structure: the screen is quadratic in the site count and
    # this test rebuilds every volume it checks.
    residues = [r for r in structure.residues if 20 <= r.seq_id <= 44]
    scores = {bff.labelizer_residue_key(r.chain, r.seq_id): 2.0 for r in residues}

    options = bff.LabelizerFRETOptions()
    options.probe_model = bff.PROBE_MODEL_ACCESSIBLE_VOLUME
    options.n_refine = 0
    pairs = list(bff.labelizer_fret_pair_scores(pdb, scores, options))
    assert pairs

    has_volume = {}
    for r in residues:
        atom = "CB" if r.cb >= 0 else "CA"
        volume = _build_volume_the_way_the_screen_does(
                pdb, r.chain, r.seq_id, atom, options)
        has_volume[(r.chain, r.seq_id)] = volume.get_n_points() > 0

    # The slice has to contain both kinds of site, or the test proves nothing.
    assert any(has_volume.values()) and not all(has_volume.values())

    for pair in pairs:
        both = (has_volume[(pair.asym_id_1, pair.seq_id_1)]
                and has_volume[(pair.asym_id_2, pair.seq_id_2)])
        claimed_a_volume = pair.probe_model == bff.PROBE_MODEL_ACCESSIBLE_VOLUME
        assert claimed_a_volume == both, (
            f"{pair.asym_id_1}{pair.seq_id_1}-{pair.asym_id_2}{pair.seq_id_2} "
            f"claims {'a volume' if claimed_a_volume else 'the attachment point'}"
            f" but {'both sites have' if both else 'a site lacks'} a volume"
        )
