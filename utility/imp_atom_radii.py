"""Regenerate the table behind ``IMP::bff::imp_atom_radius`` from IMP itself.

IMP gives every atom ``IMP::atom::read_pdb`` reads a radius through the
all-atom CHARMM parameters (``internal::add_pdb_radii``: topology, default
patches, atom types, ``ForceFieldParameters::add_radii``); an atom whose type
is unknown gets IMP's default, 1.7 A. The table is that mapping, keyed by
(residue name, PDB atom name), so the IMP-free core reads **the same radii**
without linking IMP. Two sources, unioned:

1. every standard residue built from IMP's CHARMM topology, once inside a
   GLY-X-GLY (or DA-X-DA / A-X-A) segment and once alone (both termini
   patched), read back under the names IMP writes;
2. every ``*.pdb`` under the given roots read with ``read_pdb`` and a
   ``NonWaterPDBSelector`` -- the spellings real files use (``C5*``, ``O1P``,
   ``1HG2``), which IMP maps onto the same CHARMM types.

The topology build wins where they disagree. Rows equal to IMP's default are
dropped, because the lookup returns the default for a missing key anyway.

Run in an environment with the full IMP (the ``arm64`` env here)::

    python utility/imp_atom_radii.py ROOT [ROOT ...] > table.inc

and paste the rows into ``imp_atom_radii_table()`` in ``src/structure/VdwRadii.cpp``.
``test/representation/test_imp_atom_radii.py`` checks the result against
``read_pdb`` on the shipped fixtures.
"""

import collections
import glob
import os
import sys

import IMP
import IMP.atom
import IMP.core

IMP.set_log_level(IMP.SILENT)
DEFAULT = 1.7
AMINO = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL"


def collect(hierarchy, out):
    for a in IMP.atom.get_by_type(hierarchy, IMP.atom.ATOM_TYPE):
        parent = a.get_parent()
        if not IMP.atom.Residue.get_is_setup(parent):
            continue
        res = IMP.atom.Residue(parent).get_residue_type().get_string()
        name = IMP.atom.Atom(a).get_atom_type().get_string()
        out[(res, name)][round(IMP.core.XYZR(a).get_radius(), 6)] += 1


def topology_radii():
    ff = IMP.atom.get_all_atom_CHARMM_parameters()
    out = collections.defaultdict(collections.Counter)

    def build(seq):
        m = IMP.Model()
        top = IMP.atom.CHARMMTopology(ff)
        seg = IMP.atom.CHARMMSegmentTopology()
        for n in seq:
            rt = ff.get_residue_topology(IMP.atom.ResidueType(n))
            seg.add_residue(IMP.atom.CHARMMResidueTopology(rt))
        top.add_segment(seg)
        top.apply_default_patches()
        h = top.create_hierarchy(m)
        top.add_atom_types(h)
        ff.add_radii(h)
        collect(h, out)

    for family in (AMINO.split(), ["DA", "DC", "DG", "DT"], ["A", "C", "G", "U"]):
        for x in family:
            build([family[0], x, family[0]])
            build([x])
    return out


def file_radii(roots):
    out = collections.defaultdict(collections.Counter)
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "*.pdb"), recursive=True)):
            try:
                m = IMP.Model()
                collect(IMP.atom.read_pdb(path, m, IMP.atom.NonWaterPDBSelector()), out)
            except Exception:
                continue
    return out


def main(roots):
    table = {}
    for key, counts in file_radii(roots).items():
        if len(counts) == 1:
            table[key] = next(iter(counts))
    for key, counts in topology_radii().items():
        table[key] = counts.most_common(1)[0][0]
    rows = sorted((k, r) for k, r in table.items() if r != DEFAULT)
    for (res, name), r in rows:
        print('    {{{{"{}", "{}"}}, {:.4f}}},'.format(res, name, r))


if __name__ == "__main__":
    main(sys.argv[1:])
