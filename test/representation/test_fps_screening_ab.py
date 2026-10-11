"""End-to-end: FPS's published screening numbers for T4L (PRD-121).

`prototypes/fast_label_score/cache/anisotropy/zenodo3376527_FRET_screening/`
holds what the real FPS produced for 421 T4L structures -- 33 donor-acceptor
pairs of <R_DA> and chi2 for three states (Sanabria *et al.*, *Nat. Commun.*
**11**, 1231, 2020, Zenodo 3376527). One of those 421 is **3GUN**, which this
module already ships, so the whole pipeline can be measured against FPS's own
answer without shipping a single new structure.

This is the complement to `test_fps_av_parity.py`. That one pins a single
volume against a single cloud, where a disagreement can be localised; this one
pins the whole path -- volumes, <R_DA>, the asymmetric chi2 -- against 33
numbers, where a systematic bias cannot hide. Both are needed, and this one is
the one that found a bias.

There are **two** references here and they are not the same yardstick.

*Against FPS itself* -- its own routine run on the same structure with the same
file (`prototypes/fps_oracle/`, pinned in
`references/fps_screening_oracle_pins.json`) -- the two agree to **+0.08 A,
rmsd 0.87 A, r = 0.9955** over the 24 pairs without site 132 and to +0.70 A /
1.70 A / 0.980 over all 33, and that comparison must be made with the contact
volume **off**, because FPS has none. (Until 2026-10-11 site 132 had no volume
at all and the 24 agreed to -0.31 A / 1.20 A / 0.9926; giving the linker's first
step FPS's reach out of the anchor resolved it and tightened the rest.)

Every number here depends on which van der Waals radii the volumes inflate
their obstacles by, so every one of them is labelled with a radii source. The
default is **IMP's** -- the radii on the particles, which is what a docking
score's clash term reads (`ProbeAccessibleVolumeDecorator::set_radii_source`). Olga's table is selectable
and reproduces the published numbers better; what that trade costs is measured
here rather than argued about.

*Against the published table*, neither reproduced it while the contact volume
the file asks for was inert: this module by +2.05 A, FPS's own routine by
+2.37 A. Honouring it (2026-09-01, PRD-121 G9) closes that: **+0.58 A bias,
0.94 A rmsd, r = 0.9954** over all 33 pairs, and **+0.15 A / 0.90 A / 0.9923**
if Olga's radii are selected too (before the linker's first step reached out of
the anchor, 2026-10-11: +0.22 A / 0.91 A over the 24 that resolved, and
-0.02 A / 0.71 A / 0.9954 over 33 with Olga's radii -- the one comparison the
change made worse). The table was not produced by
FPS, which has no contact volume, but by an ACV-capable program reading this
file -- so the file *is* the parameter set it was computed with, which is the
opposite of what this test recorded on the first pass.

What is established, and what is not, is in
`okf/validation/fps_screening_ab.md`.
"""
import csv
import json
import math
import os
import tempfile

import numpy as np
import pytest

import IMP
import IMP.atom
import IMP.bff
import IMP.core

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ZENODO = os.path.join(
    REPO, "prototypes", "fast_label_score", "cache", "anisotropy",
    "zenodo3376527_FRET_screening", "FRET_screening_of_PDB_structures")
TABLE = os.path.join(ZENODO, "FRET_screening_of_PDB_structures.dat")
FPS_JSON = os.path.join(ZENODO, "FRET_screening.fps.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(TABLE),
    reason="the Zenodo FPS screening reference is not in this checkout")


def _reference_row(pdb_id):
    with open(TABLE) as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    header = rows[1]
    for row in rows[2:]:
        if row and row[0].strip() == pdb_id:
            return header, dict(zip(header, row))
    raise AssertionError("%s is not in the reference table" % pdb_id)


def _scored(derive_clearance, contact_volume=False, radii_source=None):
    """3GUN through the whole path, with FPS's own labelling file.

    ``allowed_sphere_radius: 2`` in that file is an ``IMP.bff`` field somebody
    chose, not an FPS parameter -- FPS seeds its search from
    ``LinkerInitialSphere * linker_width`` and has no such key. Dropping it
    lets the clearance derive, which is what the file would mean if it had
    never carried the field.

    ``contact_volume_thickness``/``_trapped_fraction`` are the same kind of
    key, and are dropped for the same reason: **FPS has no accessible-contact
    volume at all**, so a run that honours them is not the run FPS made. They
    used to be safe to leave in because they did nothing (PRD-121 G9); since
    they were wired up they are not, and leaving them in silently turned this
    A/B into a comparison of two different models. ``contact_volume=True``
    keeps them, which is how the size of that difference is measured.

    ``radii_source`` is left alone by default, so the file gets the module
    default -- ``"imp"``, the radii the particles carry, chosen so that a
    volume and the clash term beside it size an atom the same way.
    ``"olga"`` selects the name-keyed table this file's fitted trapped
    fractions and the published table were actually produced with. Both are
    exercised here, always side by side: the difference between them is a real
    quantity and every claim in this file names which one it is about.
    """
    document = json.load(open(FPS_JSON))
    for position in document["Positions"].values():
        if derive_clearance:
            position.pop("allowed_sphere_radius", None)
        if not contact_volume:
            position.pop("contact_volume_thickness", None)
            position.pop("contact_volume_trapped_fraction", None)
        if radii_source is not None:
            position["radii_source"] = radii_source
    path = os.path.join(tempfile.mkdtemp(), "screening.fps.json")
    with open(path, "w") as fh:
        json.dump(document, fh)
    return IMP.bff.score_structures(
        [IMP.bff.get_example_path("structure/T4L/3GUN.pdb")],
        path, "chi2_C1_33p")


def _paired(result, reference, header):
    """FPS's <R_DA> against this module's, for the pairs that have both."""
    mine = {}
    unresolved = []
    for pair in result.pairs:
        key = pair.name.replace("_C1", "").replace("-", "_")
        if math.isfinite(pair.distance_model):
            mine[key] = pair.distance_model
        else:
            unresolved.append(pair.name)
    columns = [c for c in header if "_" in c and not c.startswith("χ")]
    theirs = np.array([float(reference[c]) for c in columns if c in mine])
    ours = np.array([mine[c] for c in columns if c in mine])
    return theirs, ours, unresolved, len(columns)


def test_the_reference_row_is_there():
    header, row = _reference_row("3GUN")
    assert float(row["χ²_C1"]) == pytest.approx(57.7)
    assert float(row["19_119"]) == pytest.approx(46.0)


ORACLE_PINS = os.path.join(HERE, "..", "references",
                           "fps_screening_oracle_pins.json")


def test_rda_agrees_with_fps_itself():
    """The real A/B: this module against FPS's own routine, pair by pair.

    Under the **default radii** -- IMP's, the ones on the particles -- all 33
    pairs resolve. Over the 24 without site 132 they agree to bias **+0.08 A**,
    rmsd **0.87 A**, r = **0.9955**; the nine with site 132 are the hard ones
    (+2.37 A / 2.92 A), and over all 33 it is +0.70 A / 1.70 A / 0.980.

    Until 2026-10-11 those nine had no model value: site 132's linker was
    walled in by united-atom radii, because the search left the anchor one
    stencil step at a time where FPS hops three voxels. With the first step
    reaching one linker width plus one grid step (FRETraj's rule,
    PathMap::set_source_hop_radius) the site resolves, and the 24 that already
    did moved *towards* FPS: -0.31 A / 1.20 A / 0.9926 before.

    **Olga's** radii, which have to be asked for, also resolve all 33, and
    over the same 24 they now agree with FPS *less* well than the default
    (-0.29 A / 1.03 A / 0.9928). Before the change the order was the other way
    round. Both halves are pinned so neither can be quoted without the other.
    """
    with open(ORACLE_PINS) as fh:
        oracle = json.load(fh)["rda_mean"]

    def against_fps(result):
        names = [p.name for p in result.pairs
                 if p.name in oracle and math.isfinite(p.distance_model)]
        mine = np.array([p.distance_model for p in result.pairs if p.name in names])
        theirs = np.array([oracle[n] for n in names])
        easy = np.array(["132" not in n for n in names])
        return mine, theirs, easy

    # --- the default: IMP's radii ------------------------------------------
    mine, theirs, easy = against_fps(_scored(derive_clearance=True))
    assert len(mine) == 33, "the set of resolvable pairs moved"
    assert easy.sum() == 24
    assert abs((mine - theirs).mean()) < 1.0
    assert np.sqrt(((mine - theirs) ** 2).mean()) < 2.0
    assert np.corrcoef(theirs, mine)[0, 1] > 0.975
    d = mine[easy] - theirs[easy]
    assert abs(d.mean()) < 0.3
    assert np.sqrt((d ** 2).mean()) < 1.0
    assert np.corrcoef(theirs[easy], mine[easy])[0, 1] > 0.995

    # --- and Olga's: all 33 as well, and further from FPS on the shared 24 --
    olga, olga_theirs, olga_easy = against_fps(
        _scored(derive_clearance=True, radii_source="olga"))
    assert len(olga) == 33, "Olga's radii no longer open every site"
    d_olga = olga[olga_easy] - olga_theirs[olga_easy]
    assert np.sqrt((d ** 2).mean()) < np.sqrt((d_olga ** 2).mean())


def test_the_published_table_is_not_reproduced_by_fps_either():
    """FPS does not reproduce the published table, pinned so it stays known.

    Without a contact volume both implementations sit ~2 A above the published
    <R_DA>, FPS's own by *more* than this module's (+2.37 A against +2.05 A).
    Neither R_mp (-1.40 A) nor <R_DA>_E (+3.11 A) matches it either, so it is
    not a mislabelled distance type, and FPS's own AV3 radii from `linker.txt`
    make it worse (+6.36 A).

    This test was first read as "the file is not the parameter set the table
    was computed with". It is not: what the file asks for and FPS cannot do is
    the **accessible-contact volume**, and honouring it brings this module to
    +0.22 A of the table -- see
    :func:`test_the_contact_volume_explains_the_offset_against_the_published_table`.
    What survives is the narrower statement this test actually makes, and it is
    worth keeping: *FPS* does not reproduce that table, so the table cannot pin
    an FPS port. The pin for that is the oracle.
    """
    header, reference = _reference_row("3GUN")
    theirs, ours, _, _ = _paired(
        _scored(derive_clearance=True), reference, header)
    with open(ORACLE_PINS) as fh:
        oracle = json.load(fh)["rda_mean"]
    assert 1.0 < (ours - theirs).mean() < 3.0        # this module
    # and FPS's own, over the same pairs
    result = _scored(derive_clearance=True)
    fps_bias = []
    for pair in result.pairs:
        if pair.name not in oracle or not math.isfinite(pair.distance_model):
            continue
        column = "%s_%s" % (pair.position1[:-1], pair.position2[:-1])
        if column not in reference:
            column = "%s_%s" % (pair.position2[:-1], pair.position1[:-1])
        if column in reference:
            fps_bias.append(oracle[pair.name] - float(reference[column]))
    assert np.mean(fps_bias) > 1.5, (
        "FPS itself no longer misses the published table; if that changed, "
        "the file may finally match and this test should be rewritten")


def test_the_authored_clearance_no_longer_buries_the_structure():
    """The authored clearance, the derived one, either radii set: all 33 resolve.

    Until 2026-10-11 this file pinned two findings. Under the default radii the
    file's ``allowed_sphere_radius: 2`` emptied eight of its seventeen sites
    (**nine** of 33 pairs resolved, **24** with the clearance derived), and
    Olga's smaller radii resolved all 33 either way -- so the radii, not the
    clearance convention, walled the sources in.

    Both were statements about the linker's first step. The clearance is a
    sphere carved out of an obstacle map already inflated by half the linker
    width; FPS instead seeds from ``LinkerInitialSphere * linker_width`` and
    hops three voxels from there. With the first step reaching one width plus
    one grid step (PathMap::set_source_hop_radius) the linker gets out of the
    anchor the way FPS's does, the clearance stops deciding whether a volume
    exists, and every combination resolves every pair.
    """
    header, reference = _reference_row("3GUN")
    counts = [
        len(_paired(_scored(derive_clearance=derive, radii_source=radii),
                    reference, header)[1])
        for derive in (False, True) for radii in (None, "olga")
    ]
    assert counts == [33, 33, 33, 33]


def test_the_contact_volume_moves_the_cloud():
    """An ACV request changes the cloud it asks to reweight (PRD-121 G9).

    This was a strict ``xfail`` until 2026-09-01: ``contact_volume_thickness``
    and ``contact_volume_trapped_fraction`` were read into the decorator and
    never used, so the volume, the per-voxel weights and the mean position came
    back bit-identical with the weighting on and off.

    The cloud is deliberately *not* expected to change shape -- an ACV is a
    weighting, the same voxels with different weights -- so what is asserted is
    that the points are the same, the weights are not, and the mean moves
    towards the protein. Measured on T4L 132/CB at these parameters: two weight
    levels, 11766 contact voxels of 33174 carrying 0.55 of the total, and the
    mean 1.24 A away from the unweighted one.
    """
    IMP.set_log_level(IMP.SILENT)
    model = IMP.Model()
    hier = IMP.atom.read_pdb(
        IMP.bff.get_example_path("structure/T4L/3GUN.pdb"), model)

    def cloud(thickness, trapped):
        selection = IMP.atom.Selection(hier)
        selection.set_atom_type(IMP.atom.AtomType("CB"))
        selection.set_residue_index(132)
        particle = IMP.Particle(model)
        IMP.bff.ProbeAccessibleVolumeDecorator.do_setup_particle(
            model, particle, selection.get_selected_particles()[0],
            linker_length=22.0, radii=(3.5, 0.0, 0.0), linker_width=1.5,
            simulation_grid_resolution=0.7,
            contact_volume_thickness=thickness,
            contact_volume_trapped_fraction=trapped)
        av = IMP.bff.ProbeAccessibleVolumeDecorator(model, particle)
        av.resample()
        return np.asarray(av.get_map().get_xyz_density(), dtype=float)

    plain = cloud(0.0, -1.0)
    contact = cloud(3.0, 0.55)
    assert len(plain) > 0
    assert not np.array_equal(plain, contact)
    # the same voxels: the ACV re-weights the cloud, it does not carve it
    assert np.array_equal(plain[:, :3], contact[:, :3])

    weight = contact[:, 3]
    levels = np.unique(weight)
    assert len(levels) == 2, "uniform base weights give exactly two ACV levels"
    trapped = weight[weight == levels[1]].sum() / weight.sum()
    assert trapped == pytest.approx(0.55, abs=1e-6)

    def mean(cloud_):
        return (cloud_[:, :3] * cloud_[:, 3:4]).sum(0) / cloud_[:, 3].sum()

    assert np.linalg.norm(mean(contact) - mean(plain)) > 0.5


def _against(reference, contact_volume, by_name):
    """This module's <R_DA> against a reference dict, over the shared pairs."""
    result = _scored(derive_clearance=True, contact_volume=contact_volume)
    mine, theirs = [], []
    for pair in result.pairs:
        if not math.isfinite(pair.distance_model):
            continue
        key = by_name(pair)
        if key in reference:
            mine.append(pair.distance_model)
            theirs.append(float(reference[key]))
    return np.array(mine), np.array(theirs)


def test_the_contact_volume_is_not_part_of_the_fps_comparison():
    """Honouring the ACV moves this module away from FPS, as it must.

    `FRET_screening.fps.json` asks for a contact volume at every site and FPS
    has none, so a run that honours the request is not the run FPS made. While
    the fields were inert this could be ignored; it cannot be since they were
    wired up (PRD-121 G9), which is why :func:`_scored` drops them. Pinned so
    the A/B cannot quietly become a comparison of two different models.

    Measured 2026-10-11 over all 33 pairs, under the default (IMP) radii:
    **+0.70 A** against FPS with the ACV off, **-1.45 A** with it on, every one
    of the 33 <R_DA> shorter, by 2.15 A on average (2026-09-01, over the 24
    that then resolved: -0.31 A and -2.15 A).
    """
    with open(ORACLE_PINS) as fh:
        oracle = json.load(fh)["rda_mean"]
    by_name = lambda pair: pair.name
    off, fps = _against(oracle, False, by_name)
    on, _ = _against(oracle, True, by_name)
    assert len(off) == len(on) == len(fps) == 33
    assert abs((off - fps).mean()) < 1.0            # the A/B, as pinned above
    assert (on - fps).mean() < -1.0                 # the ACV, on top of it
    assert (on - off).mean() < -1.5
    assert (on < off).all()


def test_the_contact_volume_explains_the_offset_against_the_published_table():
    """What the ACV was hiding: the table *was* computed with one.

    `okf/validation/fps_screening_ab.md` recorded a +2.05 A offset against
    Zenodo 3376527's published <R_DA> that neither the distance type, nor the
    AV1/AV3 choice, nor FPS's own routine could account for -- FPS misses that
    table by +2.37 A, *more* than this module did -- and concluded the file was
    not the parameter set the table was computed with.

    Wiring up the contact volume that file asks for at every site closes it:
    bias **+2.73 -> +0.58 A**, rmsd **3.03 -> 0.94 A**, r 0.9888 -> 0.9954 over
    all 33 pairs (2026-09-01, over the 24 that then resolved: +2.06 -> +0.22 A,
    2.54 -> 0.91 A). The file *is* the parameter set; what was
    missing was the feature it names. The table was therefore not produced by
    FPS, which has no contact volume, but by an ACV-capable program reading
    this file.

    That program was Olga, and asking for Olga's radii (`ProbeAccessibleVolumeDecorator::set_radii_source`)
    improves it again: **+0.15 A bias, 0.90 A rmsd, r = 0.9923 over all 33
    published pairs**. Before the linker's first step reached out of the anchor
    (2026-10-11) this was -0.02 A / 0.71 A / 0.9954 -- the published table, which
    Olga computed, is the one reference that change moved this module away
    from, while it moved it towards FPS's own routine.

    That is *not* the default, and the reason is in `ProbeAccessibleVolumeDecorator::set_radii_source`:
    the clash term of a docking score reads the particles' radii, so the volume
    must too. What the default costs against this reference is exactly the gap
    asserted below, and `okf/validation/fps_screening_ab.md` states it as a
    price rather than hiding it. Anyone reproducing Olga-era numbers should
    select `radii_source = "olga"`.

    Kept as a test rather than a note because it is the strongest external
    check the module has: 33 published numbers, reproduced to within 0.72 A
    rmsd by a path with no fitted parameter in it (0.90 A since 2026-10-11).
    """
    header, reference = _reference_row("3GUN")
    by_name = lambda pair: pair.name.replace("_C1", "").replace("-", "_")
    off, published = _against(reference, False, by_name)
    on, _ = _against(reference, True, by_name)
    assert len(on) == 33

    assert (off - published).mean() > 1.5           # the offset, as recorded
    assert abs((on - published).mean()) < 0.75      # and what closes it
    assert np.sqrt(((on - published) ** 2).mean()) < 1.5
    assert (np.sqrt(((on - published) ** 2).mean())
            < 0.5 * np.sqrt(((off - published) ** 2).mean()))
    assert np.corrcoef(published, on)[0, 1] > 0.995

    # ...and what the default costs against this reference, measured. Olga's
    # radii fit the 33 published pairs better than the default does. Asserted so the price cannot drift out of
    # the page that quotes it.
    olga_result = _scored(derive_clearance=True, contact_volume=True,
                          radii_source="olga")
    olga_mine, olga_theirs = [], []
    for pair in olga_result.pairs:
        if not math.isfinite(pair.distance_model):
            continue
        key = by_name(pair)
        if key in reference:
            olga_mine.append(pair.distance_model)
            olga_theirs.append(float(reference[key]))
    olga_mine = np.array(olga_mine)
    olga_theirs = np.array(olga_theirs)
    assert len(olga_mine) == 33 and len(on) == 33
    assert abs((olga_mine - olga_theirs).mean()) < abs((on - published).mean())
    assert (np.sqrt(((olga_mine - olga_theirs) ** 2).mean())
            < np.sqrt(((on - published) ** 2).mean()))
