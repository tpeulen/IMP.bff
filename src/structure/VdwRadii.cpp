/**
 *  \file src/VdwRadii.cpp
 *  \brief Olga's van der Waals table, vendored, and IMP's radii by residue and atom.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */

#include <IMP/bff/VdwRadii.h>

#include <IMP/bff/IMPCompatibility.h>

#include <algorithm>
#include <cstdio>
#include <set>
#include <sstream>
#include <utility>

IMPBFF_BEGIN_NAMESPACE

/* Olga's radii, converted from the upstream nanometres to Angstrom **once,
   here**, so that no call site can forget the factor. The conversion is not a
   guess about magnitudes: `Olga/src/AV/Position.cpp:118-124` scales the
   coordinate and the radius by the same 10.0f on one line when it builds the
   (x, y, z, vdW) array `calculateAV()` consumes, because pteros stores
   coordinates in nm and the AV kernel works in Angstrom. See VdwRadii.h.

   Six distinct values over 128 names; the upstream file has 131 lines, of
   which "H1", "H2" and "H3" appear twice with identical values (JSON keeps the
   last, so the duplicates are inert). File-local helpers in this translation
   unit are prefixed `vdwradii_`: src/*.cpp is compiled as one unit here, so an
   anonymous namespace isolates nothing. */
static const std::map<std::string, double> &vdwradii_olga_table() {
    static const std::map<std::string, double> table = {
    // hydrogen (0.100 nm). Olga rasterises explicit hydrogens; a structure
    // read with a heavy-atom selector simply has none of these, which is the
    // single largest difference between this set and IMP's united-atom one.
    {"H", 1.0000}, {"H1", 1.0000}, {"H1'", 1.0000}, {"H2", 1.0000},
    {"H2'", 1.0000}, {"H21", 1.0000}, {"H22", 1.0000}, {"H3", 1.0000},
    {"H3'", 1.0000}, {"H4'", 1.0000}, {"H41", 1.0000}, {"H42", 1.0000},
    {"H5", 1.0000}, {"H5'", 1.0000}, {"H5''", 1.0000}, {"H6", 1.0000},
    {"H61", 1.0000}, {"H62", 1.0000}, {"H8", 1.0000}, {"HA", 1.0000},
    {"HA2", 1.0000}, {"HA3", 1.0000}, {"HB", 1.0000}, {"HB1", 1.0000},
    {"HB2", 1.0000}, {"HB3", 1.0000}, {"HD1", 1.0000}, {"HD11", 1.0000},
    {"HD12", 1.0000}, {"HD13", 1.0000}, {"HD2", 1.0000}, {"HD21", 1.0000},
    {"HD22", 1.0000}, {"HD23", 1.0000}, {"HD3", 1.0000}, {"HE", 1.0000},
    {"HE1", 1.0000}, {"HE2", 1.0000}, {"HE21", 1.0000}, {"HE22", 1.0000},
    {"HE3", 1.0000}, {"HG", 1.0000}, {"HG1", 1.0000}, {"HG11", 1.0000},
    {"HG12", 1.0000}, {"HG13", 1.0000}, {"HG2", 1.0000}, {"HG21", 1.0000},
    {"HG22", 1.0000}, {"HG23", 1.0000}, {"HG3", 1.0000}, {"HH", 1.0000},
    {"HH11", 1.0000}, {"HH12", 1.0000}, {"HH2", 1.0000}, {"HH21", 1.0000},
    {"HH22", 1.0000}, {"HO2'", 1.0000}, {"HO3'", 1.0000}, {"HO5'", 1.0000},
    {"HZ", 1.0000}, {"HZ1", 1.0000}, {"HZ2", 1.0000}, {"HZ3", 1.0000},
    // oxygen (0.1490 nm)
    {"O", 1.4900}, {"O2", 1.4900}, {"O2'", 1.4900}, {"O3'", 1.4900},
    {"O4", 1.4900}, {"O4'", 1.4900}, {"O5'", 1.4900}, {"O6", 1.4900},
    {"OD1", 1.4900}, {"OD2", 1.4900}, {"OE1", 1.4900}, {"OE2", 1.4900},
    {"OG", 1.4900}, {"OG1", 1.4900}, {"OH", 1.4900}, {"OP1", 1.4900},
    {"OP2", 1.4900}, {"OXT", 1.4900},
    // nitrogen (0.1625 nm)
    {"N", 1.6250}, {"N1", 1.6250}, {"N2", 1.6250}, {"N3", 1.6250},
    {"N4", 1.6250}, {"N6", 1.6250}, {"N7", 1.6250}, {"N9", 1.6250},
    {"ND1", 1.6250}, {"ND2", 1.6250}, {"NE", 1.6250}, {"NE1", 1.6250},
    {"NE2", 1.6250}, {"NH1", 1.6250}, {"NH2", 1.6250}, {"NZ", 1.6250},
    // carbon (0.1700 nm) = Bondi
    {"C", 1.7000}, {"C1'", 1.7000}, {"C2", 1.7000}, {"C2'", 1.7000},
    {"C3'", 1.7000}, {"C4", 1.7000}, {"C4'", 1.7000}, {"C5", 1.7000},
    {"C5'", 1.7000}, {"C6", 1.7000}, {"C8", 1.7000}, {"CA", 1.7000},
    {"CB", 1.7000}, {"CD", 1.7000}, {"CD1", 1.7000}, {"CD2", 1.7000},
    {"CE", 1.7000}, {"CE1", 1.7000}, {"CE2", 1.7000}, {"CE3", 1.7000},
    {"CG", 1.7000}, {"CG1", 1.7000}, {"CG2", 1.7000}, {"CH2", 1.7000},
    {"CZ", 1.7000}, {"CZ2", 1.7000}, {"CZ3", 1.7000},
    // sulphur (0.1782 nm)
    {"SD", 1.7820}, {"SG", 1.7820},
    // phosphorus (0.1860 nm). Worth noticing: Olga's *built-in* fallback map
    // in loadvdWRadii() carries P = 0.1 nm (= 1.0 A, a hydrogen), and the
    // shipped JSON overrides it with 0.186 nm. The JSON is what a real run
    // loads, so 1.86 A is Olga's phosphorus and 1.0 A is a bug in the
    // fallback nobody reaches.
    {"P", 1.8600},
    };
    return table;
}

/* IMP's radii, keyed by (residue, PDB atom name): the mapping
   `IMP::atom::read_pdb` applies through the all-atom CHARMM parameters,
   regenerated from IMP by prototypes/imp_atom_radii/generate.py (do not edit
   rows by hand). Rows equal to IMP's 1.7 A default are left out -- the lookup
   returns the default for a missing key, as IMP does for an untyped atom. */
static const std::map<std::pair<std::string, std::string>, double> &
vdwradii_imp_table() {
    static const std::map<std::pair<std::string, std::string>, double> table = {
    {{"A", "C1'"}, 2.2750},
    {{"A", "C2"}, 1.8000},
    {{"A", "C2'"}, 2.2750},
    {{"A", "C3'"}, 2.2750},
    {{"A", "C4"}, 1.8000},
    {{"A", "C4'"}, 2.2750},
    {{"A", "C5"}, 1.8000},
    {{"A", "C5'"}, 2.1750},
    {{"A", "C6"}, 1.8000},
    {{"A", "C8"}, 1.8000},
    {{"A", "H1'"}, 1.3200},
    {{"A", "H2"}, 1.3200},
    {{"A", "H2'"}, 1.3200},
    {{"A", "H3'"}, 1.3200},
    {{"A", "H4'"}, 1.3200},
    {{"A", "H5'"}, 1.3200},
    {{"A", "H5''"}, 1.3200},
    {{"A", "H61"}, 0.2245},
    {{"A", "H62"}, 0.2245},
    {{"A", "H8"}, 1.3200},
    {{"A", "HO2'"}, 0.2245},
    {{"A", "N1"}, 1.8500},
    {{"A", "N3"}, 1.8500},
    {{"A", "N6"}, 1.8500},
    {{"A", "N7"}, 1.8500},
    {{"A", "N9"}, 1.8500},
    {{"A", "O2'"}, 1.7700},
    {{"A", "O3'"}, 1.7700},
    {{"A", "O4'"}, 1.7700},
    {{"A", "O5'"}, 1.7700},
    {{"A", "P"}, 2.1500},
    {{"ADP", "HET: C1'"}, 2.2750},
    {{"ADP", "HET: C2 "}, 1.8000},
    {{"ADP", "HET: C2'"}, 2.2750},
    {{"ADP", "HET: C3'"}, 2.2750},
    {{"ADP", "HET: C4 "}, 1.8000},
    {{"ADP", "HET: C4'"}, 2.2750},
    {{"ADP", "HET: C5 "}, 1.8000},
    {{"ADP", "HET: C5'"}, 2.1750},
    {{"ADP", "HET: C6 "}, 1.8000},
    {{"ADP", "HET: C8 "}, 1.8000},
    {{"ADP", "HET: N1 "}, 1.8500},
    {{"ADP", "HET: N3 "}, 1.8500},
    {{"ADP", "HET: N6 "}, 1.8500},
    {{"ADP", "HET: N7 "}, 1.8500},
    {{"ADP", "HET: N9 "}, 1.8500},
    {{"ADP", "HET: O2'"}, 1.7700},
    {{"ADP", "HET: O3'"}, 1.7700},
    {{"ADP", "HET: O3A"}, 1.7700},
    {{"ADP", "HET: O4'"}, 1.7700},
    {{"ADP", "HET: O5'"}, 1.7700},
    {{"ADP", "HET: PA "}, 2.1500},
    {{"ADP", "HET: PB "}, 2.1500},
    {{"ALA", "C"}, 2.0000},
    {{"ALA", "CA"}, 2.2750},
    {{"ALA", "CB"}, 2.0600},
    {{"ALA", "H"}, 0.2245},
    {{"ALA", "H2"}, 0.2245},
    {{"ALA", "H3"}, 0.2245},
    {{"ALA", "HA"}, 1.3200},
    {{"ALA", "HB1"}, 1.3200},
    {{"ALA", "HB2"}, 1.3200},
    {{"ALA", "HB3"}, 1.3200},
    {{"ALA", "N"}, 1.8500},
    {{"ARG", "C"}, 2.0000},
    {{"ARG", "CA"}, 2.2750},
    {{"ARG", "CB"}, 2.1750},
    {{"ARG", "CD"}, 2.1750},
    {{"ARG", "CG"}, 2.1750},
    {{"ARG", "CZ"}, 2.0000},
    {{"ARG", "H"}, 0.2245},
    {{"ARG", "H2"}, 0.2245},
    {{"ARG", "H3"}, 0.2245},
    {{"ARG", "HA"}, 1.3200},
    {{"ARG", "HB2"}, 1.3200},
    {{"ARG", "HB3"}, 1.3200},
    {{"ARG", "HD2"}, 1.3200},
    {{"ARG", "HD3"}, 1.3200},
    {{"ARG", "HE"}, 0.2245},
    {{"ARG", "HG2"}, 1.3200},
    {{"ARG", "HG3"}, 1.3200},
    {{"ARG", "HH11"}, 0.2245},
    {{"ARG", "HH12"}, 0.2245},
    {{"ARG", "HH21"}, 0.2245},
    {{"ARG", "HH22"}, 0.2245},
    {{"ARG", "N"}, 1.8500},
    {{"ARG", "NE"}, 1.8500},
    {{"ARG", "NH1"}, 1.8500},
    {{"ARG", "NH2"}, 1.8500},
    {{"ASN", "C"}, 2.0000},
    {{"ASN", "CA"}, 2.2750},
    {{"ASN", "CB"}, 2.1750},
    {{"ASN", "CG"}, 2.0000},
    {{"ASN", "H"}, 0.2245},
    {{"ASN", "H2"}, 0.2245},
    {{"ASN", "H3"}, 0.2245},
    {{"ASN", "HA"}, 1.3200},
    {{"ASN", "HB2"}, 1.3200},
    {{"ASN", "HB3"}, 1.3200},
    {{"ASN", "HD21"}, 0.2245},
    {{"ASN", "HD22"}, 0.2245},
    {{"ASN", "N"}, 1.8500},
    {{"ASN", "ND2"}, 1.8500},
    {{"ASP", "C"}, 2.0000},
    {{"ASP", "CA"}, 2.2750},
    {{"ASP", "CB"}, 2.1750},
    {{"ASP", "CG"}, 2.0000},
    {{"ASP", "H"}, 0.2245},
    {{"ASP", "H2"}, 0.2245},
    {{"ASP", "H3"}, 0.2245},
    {{"ASP", "HA"}, 1.3200},
    {{"ASP", "HB2"}, 1.3200},
    {{"ASP", "HB3"}, 1.3200},
    {{"ASP", "N"}, 1.8500},
    {{"C", "C1'"}, 2.2750},
    {{"C", "C2"}, 1.8000},
    {{"C", "C2'"}, 2.2750},
    {{"C", "C3'"}, 2.2750},
    {{"C", "C4"}, 1.8000},
    {{"C", "C4'"}, 2.2750},
    {{"C", "C5"}, 1.8000},
    {{"C", "C5'"}, 2.1750},
    {{"C", "C6"}, 1.8000},
    {{"C", "H1'"}, 1.3200},
    {{"C", "H2'"}, 1.3200},
    {{"C", "H3'"}, 1.3200},
    {{"C", "H4'"}, 1.3200},
    {{"C", "H41"}, 0.2245},
    {{"C", "H42"}, 0.2245},
    {{"C", "H5"}, 1.3200},
    {{"C", "H5'"}, 1.3200},
    {{"C", "H5''"}, 1.3200},
    {{"C", "H6"}, 1.3200},
    {{"C", "HO2'"}, 0.2245},
    {{"C", "N1"}, 1.8500},
    {{"C", "N3"}, 1.8500},
    {{"C", "N4"}, 1.8500},
    {{"C", "O2'"}, 1.7700},
    {{"C", "O3'"}, 1.7700},
    {{"C", "O4'"}, 1.7700},
    {{"C", "O5'"}, 1.7700},
    {{"C", "P"}, 2.1500},
    {{"CYS", "C"}, 2.0000},
    {{"CYS", "CA"}, 2.2750},
    {{"CYS", "CB"}, 2.1750},
    {{"CYS", "H"}, 0.2245},
    {{"CYS", "H2"}, 0.2245},
    {{"CYS", "H3"}, 0.2245},
    {{"CYS", "HA"}, 1.3200},
    {{"CYS", "HB2"}, 1.3200},
    {{"CYS", "HB3"}, 1.3200},
    {{"CYS", "HG"}, 0.4500},
    {{"CYS", "N"}, 1.8500},
    {{"CYS", "SG"}, 2.0000},
    {{"DA", "C1'"}, 2.2750},
    {{"DA", "C2"}, 1.8000},
    {{"DA", "C2'"}, 2.1750},
    {{"DA", "C3'"}, 2.2750},
    {{"DA", "C4"}, 1.8000},
    {{"DA", "C4'"}, 2.2750},
    {{"DA", "C5"}, 1.8000},
    {{"DA", "C5'"}, 2.1750},
    {{"DA", "C6"}, 1.8000},
    {{"DA", "C8"}, 1.8000},
    {{"DA", "H1'"}, 1.3200},
    {{"DA", "H2"}, 1.3200},
    {{"DA", "H2'"}, 1.3200},
    {{"DA", "H2''"}, 1.3200},
    {{"DA", "H3'"}, 1.3200},
    {{"DA", "H4'"}, 1.3200},
    {{"DA", "H5'"}, 1.3200},
    {{"DA", "H5''"}, 1.3200},
    {{"DA", "H61"}, 0.2245},
    {{"DA", "H62"}, 0.2245},
    {{"DA", "H8"}, 1.3200},
    {{"DA", "N1"}, 1.8500},
    {{"DA", "N3"}, 1.8500},
    {{"DA", "N6"}, 1.8500},
    {{"DA", "N7"}, 1.8500},
    {{"DA", "N9"}, 1.8500},
    {{"DA", "O3'"}, 1.7700},
    {{"DA", "O4'"}, 1.7700},
    {{"DA", "O5'"}, 1.7700},
    {{"DA", "P"}, 2.1500},
    {{"DC", "C1'"}, 2.2750},
    {{"DC", "C2"}, 1.8000},
    {{"DC", "C2'"}, 2.1750},
    {{"DC", "C3'"}, 2.2750},
    {{"DC", "C4"}, 1.8000},
    {{"DC", "C4'"}, 2.2750},
    {{"DC", "C5"}, 1.8000},
    {{"DC", "C5'"}, 2.1750},
    {{"DC", "C6"}, 1.8000},
    {{"DC", "H1'"}, 1.3200},
    {{"DC", "H2'"}, 1.3200},
    {{"DC", "H2''"}, 1.3200},
    {{"DC", "H3'"}, 1.3200},
    {{"DC", "H4'"}, 1.3200},
    {{"DC", "H41"}, 0.2245},
    {{"DC", "H42"}, 0.2245},
    {{"DC", "H5"}, 1.3200},
    {{"DC", "H5'"}, 1.3200},
    {{"DC", "H5''"}, 1.3200},
    {{"DC", "H6"}, 1.3200},
    {{"DC", "N1"}, 1.8500},
    {{"DC", "N3"}, 1.8500},
    {{"DC", "N4"}, 1.8500},
    {{"DC", "O3'"}, 1.7700},
    {{"DC", "O4'"}, 1.7700},
    {{"DC", "O5'"}, 1.7700},
    {{"DC", "P"}, 2.1500},
    {{"DG", "C1'"}, 2.2750},
    {{"DG", "C2"}, 1.8000},
    {{"DG", "C2'"}, 2.1750},
    {{"DG", "C3'"}, 2.2750},
    {{"DG", "C4"}, 1.8000},
    {{"DG", "C4'"}, 2.2750},
    {{"DG", "C5"}, 1.8000},
    {{"DG", "C5'"}, 2.1750},
    {{"DG", "C6"}, 1.8000},
    {{"DG", "C8"}, 1.8000},
    {{"DG", "H1"}, 0.2245},
    {{"DG", "H1'"}, 1.3200},
    {{"DG", "H2'"}, 1.3200},
    {{"DG", "H2''"}, 1.3200},
    {{"DG", "H21"}, 0.2245},
    {{"DG", "H22"}, 0.2245},
    {{"DG", "H3'"}, 1.3200},
    {{"DG", "H4'"}, 1.3200},
    {{"DG", "H5'"}, 1.3200},
    {{"DG", "H5''"}, 1.3200},
    {{"DG", "H8"}, 1.3200},
    {{"DG", "N1"}, 1.8500},
    {{"DG", "N2"}, 1.8500},
    {{"DG", "N3"}, 1.8500},
    {{"DG", "N7"}, 1.8500},
    {{"DG", "N9"}, 1.8500},
    {{"DG", "O3'"}, 1.7700},
    {{"DG", "O4'"}, 1.7700},
    {{"DG", "O5'"}, 1.7700},
    {{"DG", "P"}, 2.1500},
    {{"DT", "C1'"}, 2.2750},
    {{"DT", "C2"}, 1.8000},
    {{"DT", "C2'"}, 2.1750},
    {{"DT", "C3'"}, 2.2750},
    {{"DT", "C4"}, 1.8000},
    {{"DT", "C4'"}, 2.2750},
    {{"DT", "C5"}, 1.8000},
    {{"DT", "C5'"}, 2.1750},
    {{"DT", "C6"}, 1.8000},
    {{"DT", "C7"}, 2.0600},
    {{"DT", "H1'"}, 1.3200},
    {{"DT", "H2'"}, 1.3200},
    {{"DT", "H2''"}, 1.3200},
    {{"DT", "H3"}, 0.2245},
    {{"DT", "H3'"}, 1.3200},
    {{"DT", "H4'"}, 1.3200},
    {{"DT", "H5'"}, 1.3200},
    {{"DT", "H5''"}, 1.3200},
    {{"DT", "H6"}, 1.3200},
    {{"DT", "H71"}, 1.3200},
    {{"DT", "H72"}, 1.3200},
    {{"DT", "H73"}, 1.3200},
    {{"DT", "N1"}, 1.8500},
    {{"DT", "N3"}, 1.8500},
    {{"DT", "O3'"}, 1.7700},
    {{"DT", "O4'"}, 1.7700},
    {{"DT", "O5'"}, 1.7700},
    {{"DT", "P"}, 2.1500},
    {{"G", "C1'"}, 2.2750},
    {{"G", "C2"}, 1.8000},
    {{"G", "C2'"}, 2.2750},
    {{"G", "C3'"}, 2.2750},
    {{"G", "C4"}, 1.8000},
    {{"G", "C4'"}, 2.2750},
    {{"G", "C5"}, 1.8000},
    {{"G", "C5'"}, 2.1750},
    {{"G", "C6"}, 1.8000},
    {{"G", "C8"}, 1.8000},
    {{"G", "H1"}, 0.2245},
    {{"G", "H1'"}, 1.3200},
    {{"G", "H2'"}, 1.3200},
    {{"G", "H21"}, 0.2245},
    {{"G", "H22"}, 0.2245},
    {{"G", "H3'"}, 1.3200},
    {{"G", "H4'"}, 1.3200},
    {{"G", "H5'"}, 1.3200},
    {{"G", "H5''"}, 1.3200},
    {{"G", "H8"}, 1.3200},
    {{"G", "HO2'"}, 0.2245},
    {{"G", "N1"}, 1.8500},
    {{"G", "N2"}, 1.8500},
    {{"G", "N3"}, 1.8500},
    {{"G", "N7"}, 1.8500},
    {{"G", "N9"}, 1.8500},
    {{"G", "O2'"}, 1.7700},
    {{"G", "O3'"}, 1.7700},
    {{"G", "O4'"}, 1.7700},
    {{"G", "O5'"}, 1.7700},
    {{"G", "P"}, 2.1500},
    {{"GLN", "C"}, 2.0000},
    {{"GLN", "CA"}, 2.2750},
    {{"GLN", "CB"}, 2.1750},
    {{"GLN", "CD"}, 2.0000},
    {{"GLN", "CG"}, 2.1750},
    {{"GLN", "H"}, 0.2245},
    {{"GLN", "H2"}, 0.2245},
    {{"GLN", "H3"}, 0.2245},
    {{"GLN", "HA"}, 1.3200},
    {{"GLN", "HB2"}, 1.3200},
    {{"GLN", "HB3"}, 1.3200},
    {{"GLN", "HE21"}, 0.2245},
    {{"GLN", "HE22"}, 0.2245},
    {{"GLN", "HG2"}, 1.3200},
    {{"GLN", "HG3"}, 1.3200},
    {{"GLN", "N"}, 1.8500},
    {{"GLN", "NE2"}, 1.8500},
    {{"GLU", "C"}, 2.0000},
    {{"GLU", "CA"}, 2.2750},
    {{"GLU", "CB"}, 2.1750},
    {{"GLU", "CD"}, 2.0000},
    {{"GLU", "CG"}, 2.1750},
    {{"GLU", "H"}, 0.2245},
    {{"GLU", "H2"}, 0.2245},
    {{"GLU", "H3"}, 0.2245},
    {{"GLU", "HA"}, 1.3200},
    {{"GLU", "HB2"}, 1.3200},
    {{"GLU", "HB3"}, 1.3200},
    {{"GLU", "HG2"}, 1.3200},
    {{"GLU", "HG3"}, 1.3200},
    {{"GLU", "N"}, 1.8500},
    {{"GLY", "C"}, 2.0000},
    {{"GLY", "CA"}, 2.1750},
    {{"GLY", "H"}, 0.2245},
    {{"GLY", "H2"}, 0.2245},
    {{"GLY", "H3"}, 0.2245},
    {{"GLY", "HA2"}, 1.3200},
    {{"GLY", "HA3"}, 1.3200},
    {{"GLY", "N"}, 1.8500},
    {{"HEME", "HET: C1A"}, 1.8000},
    {{"HEME", "HET: C1B"}, 1.8000},
    {{"HEME", "HET: C1C"}, 1.8000},
    {{"HEME", "HET: C1D"}, 1.8000},
    {{"HEME", "HET: C2A"}, 1.8000},
    {{"HEME", "HET: C2B"}, 1.8000},
    {{"HEME", "HET: C2C"}, 1.8000},
    {{"HEME", "HET: C2D"}, 1.8000},
    {{"HEME", "HET: C3A"}, 1.8000},
    {{"HEME", "HET: C3B"}, 1.8000},
    {{"HEME", "HET: C3C"}, 1.8000},
    {{"HEME", "HET: C3D"}, 1.8000},
    {{"HEME", "HET: C4A"}, 1.8000},
    {{"HEME", "HET: C4B"}, 1.8000},
    {{"HEME", "HET: C4C"}, 1.8000},
    {{"HEME", "HET: C4D"}, 1.8000},
    {{"HEME", "HET: CAA"}, 2.1750},
    {{"HEME", "HET: CAB"}, 2.0000},
    {{"HEME", "HET: CAC"}, 2.0000},
    {{"HEME", "HET: CAD"}, 2.1750},
    {{"HEME", "HET: CBA"}, 2.1750},
    {{"HEME", "HET: CBB"}, 2.0000},
    {{"HEME", "HET: CBC"}, 2.0000},
    {{"HEME", "HET: CBD"}, 2.1750},
    {{"HEME", "HET: CGA"}, 2.0000},
    {{"HEME", "HET: CGD"}, 2.0000},
    {{"HEME", "HET: CHA"}, 1.8000},
    {{"HEME", "HET: CHB"}, 1.8000},
    {{"HEME", "HET: CHC"}, 1.8000},
    {{"HEME", "HET: CHD"}, 1.8000},
    {{"HEME", "HET: CMA"}, 2.0600},
    {{"HEME", "HET: CMB"}, 2.0600},
    {{"HEME", "HET: CMC"}, 2.0600},
    {{"HEME", "HET: CMD"}, 2.0600},
    {{"HEME", "HET: HAB"}, 1.3200},
    {{"HEME", "HET: HAC"}, 1.3200},
    {{"HEME", "HET: HHA"}, 1.3200},
    {{"HEME", "HET: HHB"}, 1.3200},
    {{"HEME", "HET: HHC"}, 1.3200},
    {{"HEME", "HET: HHD"}, 1.3200},
    {{"HEME", "HET: NA "}, 1.8500},
    {{"HEME", "HET: NC "}, 1.8500},
    {{"HEME", "HET:FE  "}, 0.6500},
    {{"HEME", "HET:HAA1"}, 1.3200},
    {{"HEME", "HET:HAA2"}, 1.3200},
    {{"HEME", "HET:HAD1"}, 1.3200},
    {{"HEME", "HET:HAD2"}, 1.3200},
    {{"HEME", "HET:HBA1"}, 1.3200},
    {{"HEME", "HET:HBA2"}, 1.3200},
    {{"HEME", "HET:HBB1"}, 1.3200},
    {{"HEME", "HET:HBB2"}, 1.3200},
    {{"HEME", "HET:HBC1"}, 1.3200},
    {{"HEME", "HET:HBC2"}, 1.3200},
    {{"HEME", "HET:HBD1"}, 1.3200},
    {{"HEME", "HET:HBD2"}, 1.3200},
    {{"HEME", "HET:HMA1"}, 1.3200},
    {{"HEME", "HET:HMA2"}, 1.3200},
    {{"HEME", "HET:HMA3"}, 1.3200},
    {{"HEME", "HET:HMB1"}, 1.3200},
    {{"HEME", "HET:HMB2"}, 1.3200},
    {{"HEME", "HET:HMB3"}, 1.3200},
    {{"HEME", "HET:HMC1"}, 1.3200},
    {{"HEME", "HET:HMC2"}, 1.3200},
    {{"HEME", "HET:HMC3"}, 1.3200},
    {{"HEME", "HET:HMD1"}, 1.3200},
    {{"HEME", "HET:HMD2"}, 1.3200},
    {{"HEME", "HET:HMD3"}, 1.3200},
    {{"HIS", "C"}, 2.0000},
    {{"HIS", "CA"}, 2.2750},
    {{"HIS", "CB"}, 2.1750},
    {{"HIS", "CD2"}, 1.8000},
    {{"HIS", "CE1"}, 1.8000},
    {{"HIS", "CG"}, 1.8000},
    {{"HIS", "H"}, 0.2245},
    {{"HIS", "H2"}, 0.2245},
    {{"HIS", "H3"}, 0.2245},
    {{"HIS", "HA"}, 1.3200},
    {{"HIS", "HB2"}, 1.3200},
    {{"HIS", "HB3"}, 1.3200},
    {{"HIS", "HD1"}, 0.2245},
    {{"HIS", "HD2"}, 1.4680},
    {{"HIS", "HE1"}, 0.9000},
    {{"HIS", "N"}, 1.8500},
    {{"HIS", "ND1"}, 1.8500},
    {{"HIS", "NE2"}, 1.8500},
    {{"ILE", "C"}, 2.0000},
    {{"ILE", "CA"}, 2.2750},
    {{"ILE", "CB"}, 2.2750},
    {{"ILE", "CD1"}, 2.0600},
    {{"ILE", "CG1"}, 2.1750},
    {{"ILE", "CG2"}, 2.0600},
    {{"ILE", "H"}, 0.2245},
    {{"ILE", "H2"}, 0.2245},
    {{"ILE", "H3"}, 0.2245},
    {{"ILE", "HA"}, 1.3200},
    {{"ILE", "HB"}, 1.3200},
    {{"ILE", "HD11"}, 1.3200},
    {{"ILE", "HD12"}, 1.3200},
    {{"ILE", "HD13"}, 1.3200},
    {{"ILE", "HG12"}, 1.3200},
    {{"ILE", "HG13"}, 1.3200},
    {{"ILE", "HG21"}, 1.3200},
    {{"ILE", "HG22"}, 1.3200},
    {{"ILE", "HG23"}, 1.3200},
    {{"ILE", "N"}, 1.8500},
    {{"LEU", "C"}, 2.0000},
    {{"LEU", "CA"}, 2.2750},
    {{"LEU", "CB"}, 2.1750},
    {{"LEU", "CD1"}, 2.0600},
    {{"LEU", "CD2"}, 2.0600},
    {{"LEU", "CG"}, 2.2750},
    {{"LEU", "H"}, 0.2245},
    {{"LEU", "H2"}, 0.2245},
    {{"LEU", "H3"}, 0.2245},
    {{"LEU", "HA"}, 1.3200},
    {{"LEU", "HB2"}, 1.3200},
    {{"LEU", "HB3"}, 1.3200},
    {{"LEU", "HD11"}, 1.3200},
    {{"LEU", "HD12"}, 1.3200},
    {{"LEU", "HD13"}, 1.3200},
    {{"LEU", "HD21"}, 1.3200},
    {{"LEU", "HD22"}, 1.3200},
    {{"LEU", "HD23"}, 1.3200},
    {{"LEU", "HG"}, 1.3200},
    {{"LEU", "N"}, 1.8500},
    {{"LYS", "C"}, 2.0000},
    {{"LYS", "CA"}, 2.2750},
    {{"LYS", "CB"}, 2.1750},
    {{"LYS", "CD"}, 2.1750},
    {{"LYS", "CE"}, 2.1750},
    {{"LYS", "CG"}, 2.1750},
    {{"LYS", "H"}, 0.2245},
    {{"LYS", "H2"}, 0.2245},
    {{"LYS", "H3"}, 0.2245},
    {{"LYS", "HA"}, 1.3200},
    {{"LYS", "HB2"}, 1.3200},
    {{"LYS", "HB3"}, 1.3200},
    {{"LYS", "HD2"}, 1.3200},
    {{"LYS", "HD3"}, 1.3200},
    {{"LYS", "HE2"}, 1.3200},
    {{"LYS", "HE3"}, 1.3200},
    {{"LYS", "HET: CA "}, 2.2750},
    {{"LYS", "HET: N  "}, 1.8500},
    {{"LYS", "HG2"}, 1.3200},
    {{"LYS", "HG3"}, 1.3200},
    {{"LYS", "HZ1"}, 0.2245},
    {{"LYS", "HZ2"}, 0.2245},
    {{"LYS", "HZ3"}, 0.2245},
    {{"LYS", "N"}, 1.8500},
    {{"LYS", "NZ"}, 1.8500},
    {{"MET", "C"}, 2.0000},
    {{"MET", "CA"}, 2.2750},
    {{"MET", "CB"}, 2.1750},
    {{"MET", "CE"}, 2.0600},
    {{"MET", "CG"}, 2.1750},
    {{"MET", "H"}, 0.2245},
    {{"MET", "H2"}, 0.2245},
    {{"MET", "H3"}, 0.2245},
    {{"MET", "HA"}, 1.3200},
    {{"MET", "HB2"}, 1.3200},
    {{"MET", "HB3"}, 1.3200},
    {{"MET", "HE1"}, 1.3200},
    {{"MET", "HE2"}, 1.3200},
    {{"MET", "HE3"}, 1.3200},
    {{"MET", "HG2"}, 1.3200},
    {{"MET", "HG3"}, 1.3200},
    {{"MET", "N"}, 1.8500},
    {{"MET", "SD"}, 2.0000},
    {{"MG", "HET:MG  "}, 1.6500},
    {{"PHE", "C"}, 2.0000},
    {{"PHE", "CA"}, 2.2750},
    {{"PHE", "CB"}, 2.1750},
    {{"PHE", "CD1"}, 1.9924},
    {{"PHE", "CD2"}, 1.9924},
    {{"PHE", "CE1"}, 1.9924},
    {{"PHE", "CE2"}, 1.9924},
    {{"PHE", "CG"}, 1.9924},
    {{"PHE", "CZ"}, 1.9924},
    {{"PHE", "H"}, 0.2245},
    {{"PHE", "H2"}, 0.2245},
    {{"PHE", "H3"}, 0.2245},
    {{"PHE", "HA"}, 1.3200},
    {{"PHE", "HB2"}, 1.3200},
    {{"PHE", "HB3"}, 1.3200},
    {{"PHE", "HD1"}, 1.3582},
    {{"PHE", "HD2"}, 1.3582},
    {{"PHE", "HE1"}, 1.3582},
    {{"PHE", "HE2"}, 1.3582},
    {{"PHE", "HZ"}, 1.3582},
    {{"PHE", "N"}, 1.8500},
    {{"PRO", "C"}, 2.0000},
    {{"PRO", "CA"}, 2.2750},
    {{"PRO", "CB"}, 2.1750},
    {{"PRO", "CD"}, 2.1750},
    {{"PRO", "CG"}, 2.1750},
    {{"PRO", "H1"}, 0.2245},
    {{"PRO", "H2"}, 0.2245},
    {{"PRO", "HA"}, 1.3200},
    {{"PRO", "HB2"}, 1.3200},
    {{"PRO", "HB3"}, 1.3200},
    {{"PRO", "HD2"}, 1.3200},
    {{"PRO", "HD3"}, 1.3200},
    {{"PRO", "HG2"}, 1.3200},
    {{"PRO", "HG3"}, 1.3200},
    {{"PRO", "N"}, 1.8500},
    {{"SER", "C"}, 2.0000},
    {{"SER", "CA"}, 2.2750},
    {{"SER", "CB"}, 2.1750},
    {{"SER", "H"}, 0.2245},
    {{"SER", "H2"}, 0.2245},
    {{"SER", "H3"}, 0.2245},
    {{"SER", "HA"}, 1.3200},
    {{"SER", "HB2"}, 1.3200},
    {{"SER", "HB3"}, 1.3200},
    {{"SER", "HG"}, 0.2245},
    {{"SER", "N"}, 1.8500},
    {{"SER", "OG"}, 1.7700},
    {{"T", "P"}, 2.1500},
    {{"THR", "C"}, 2.0000},
    {{"THR", "CA"}, 2.2750},
    {{"THR", "CB"}, 2.2750},
    {{"THR", "CG2"}, 2.0600},
    {{"THR", "H"}, 0.2245},
    {{"THR", "H2"}, 0.2245},
    {{"THR", "H3"}, 0.2245},
    {{"THR", "HA"}, 1.3200},
    {{"THR", "HB"}, 1.3200},
    {{"THR", "HG1"}, 0.2245},
    {{"THR", "HG21"}, 1.3200},
    {{"THR", "HG22"}, 1.3200},
    {{"THR", "HG23"}, 1.3200},
    {{"THR", "N"}, 1.8500},
    {{"THR", "OG1"}, 1.7700},
    {{"TRP", "C"}, 2.0000},
    {{"TRP", "CA"}, 2.2750},
    {{"TRP", "CB"}, 2.1750},
    {{"TRP", "CD1"}, 1.9924},
    {{"TRP", "CD2"}, 1.8000},
    {{"TRP", "CE2"}, 1.8000},
    {{"TRP", "CE3"}, 1.9924},
    {{"TRP", "CG"}, 1.9924},
    {{"TRP", "CH2"}, 1.9924},
    {{"TRP", "CZ2"}, 1.9924},
    {{"TRP", "CZ3"}, 1.9924},
    {{"TRP", "H"}, 0.2245},
    {{"TRP", "H2"}, 0.2245},
    {{"TRP", "H3"}, 0.2245},
    {{"TRP", "HA"}, 1.3200},
    {{"TRP", "HB2"}, 1.3200},
    {{"TRP", "HB3"}, 1.3200},
    {{"TRP", "HD1"}, 1.3582},
    {{"TRP", "HE1"}, 0.2245},
    {{"TRP", "HE3"}, 1.3582},
    {{"TRP", "HH2"}, 1.3582},
    {{"TRP", "HZ2"}, 1.3582},
    {{"TRP", "HZ3"}, 1.3582},
    {{"TRP", "N"}, 1.8500},
    {{"TRP", "NE1"}, 1.8500},
    {{"TYR", "C"}, 2.0000},
    {{"TYR", "CA"}, 2.2750},
    {{"TYR", "CB"}, 2.1750},
    {{"TYR", "CD1"}, 1.9924},
    {{"TYR", "CD2"}, 1.9924},
    {{"TYR", "CE1"}, 1.9924},
    {{"TYR", "CE2"}, 1.9924},
    {{"TYR", "CG"}, 1.9924},
    {{"TYR", "CZ"}, 1.9924},
    {{"TYR", "H"}, 0.2245},
    {{"TYR", "H2"}, 0.2245},
    {{"TYR", "H3"}, 0.2245},
    {{"TYR", "HA"}, 1.3200},
    {{"TYR", "HB2"}, 1.3200},
    {{"TYR", "HB3"}, 1.3200},
    {{"TYR", "HD1"}, 1.3582},
    {{"TYR", "HD2"}, 1.3582},
    {{"TYR", "HE1"}, 1.3582},
    {{"TYR", "HE2"}, 1.3582},
    {{"TYR", "HH"}, 0.2245},
    {{"TYR", "N"}, 1.8500},
    {{"TYR", "OH"}, 1.7700},
    {{"U", "C1'"}, 2.2750},
    {{"U", "C2"}, 1.8000},
    {{"U", "C2'"}, 2.2750},
    {{"U", "C3'"}, 2.2750},
    {{"U", "C4"}, 1.8000},
    {{"U", "C4'"}, 2.2750},
    {{"U", "C5"}, 1.8000},
    {{"U", "C5'"}, 2.1750},
    {{"U", "C6"}, 1.8000},
    {{"U", "H1'"}, 1.3200},
    {{"U", "H2'"}, 1.3200},
    {{"U", "H3"}, 0.2245},
    {{"U", "H3'"}, 1.3200},
    {{"U", "H4'"}, 1.3200},
    {{"U", "H5"}, 1.3200},
    {{"U", "H5'"}, 1.3200},
    {{"U", "H5''"}, 1.3200},
    {{"U", "H6"}, 1.3200},
    {{"U", "HO2'"}, 0.2245},
    {{"U", "N1"}, 1.8500},
    {{"U", "N3"}, 1.8500},
    {{"U", "O2'"}, 1.7700},
    {{"U", "O3'"}, 1.7700},
    {{"U", "O4'"}, 1.7700},
    {{"U", "O5'"}, 1.7700},
    {{"U", "P"}, 2.1500},
    {{"VAL", "C"}, 2.0000},
    {{"VAL", "CA"}, 2.2750},
    {{"VAL", "CB"}, 2.2750},
    {{"VAL", "CG1"}, 2.0600},
    {{"VAL", "CG2"}, 2.0600},
    {{"VAL", "H"}, 0.2245},
    {{"VAL", "H2"}, 0.2245},
    {{"VAL", "H3"}, 0.2245},
    {{"VAL", "HA"}, 1.3200},
    {{"VAL", "HB"}, 1.3200},
    {{"VAL", "HG11"}, 1.3200},
    {{"VAL", "HG12"}, 1.3200},
    {{"VAL", "HG13"}, 1.3200},
    {{"VAL", "HG21"}, 1.3200},
    {{"VAL", "HG22"}, 1.3200},
    {{"VAL", "HG23"}, 1.3200},
    {{"VAL", "N"}, 1.8500},
    };
    return table;
}

double get_imp_atom_default_radius() { return 1.7; }

double imp_atom_radius(const std::string &res_name, const std::string &atom_name) {
    const std::map<std::pair<std::string, std::string>, double> &t =
            vdwradii_imp_table();
    std::map<std::pair<std::string, std::string>, double>::const_iterator it =
            t.find(std::make_pair(res_name, atom_name));
    if (it == t.end()) return get_imp_atom_default_radius();
    return it->second;
}

unsigned int get_number_of_imp_atom_radii() {
    return (unsigned int) vdwradii_imp_table().size();
}

const std::map<std::string, double> &get_olga_vdw_radii() {
    return vdwradii_olga_table();
}

double get_olga_vdw_fallback_radius() {
    /* `vdWRMap.value(QString::fromStdString(system.atom(i).name), 0.15)`
       -- Olga/src/AV/Position.cpp:110, in nanometres, so 1.5 A. It is a
       property of Olga's *code*, not of its table, which is why it is a
       constant here and not a row in data/olga_vdw_radii.csv. */
    return 1.5;
}

double olga_vdw_radius(const std::string &atom_name) {
    const std::map<std::string, double> &t = vdwradii_olga_table();
    std::map<std::string, double>::const_iterator it = t.find(atom_name);
    if (it == t.end()) return get_olga_vdw_fallback_radius();
    return it->second;
}

unsigned int get_number_of_olga_vdw_radii() {
    return (unsigned int) vdwradii_olga_table().size();
}

std::vector<std::string> get_olga_vdw_atom_names() {
    std::vector<std::string> out;
    out.reserve(vdwradii_olga_table().size());
    for (const auto &kv : vdwradii_olga_table()) out.push_back(kv.first);
    return out;
}

AVRadiiSource av_radii_source_from_string(const std::string &s) {
    if (s == "imp") return AV_RADII_IMP;
    if (s == "olga") return AV_RADII_OLGA;
    /* "model" was this value's spelling under schema 1.5, for one day. It is
       rejected rather than aliased so that a file carrying it is read as the
       1.5 file it is, loudly, instead of being silently reinterpreted. */
    IMP_THROW("radii source must be \"imp\" or \"olga\", not \"" << s << "\"",
              IMP::ValueException);
}

std::string av_radii_source_to_string(AVRadiiSource s) {
    return s == AV_RADII_OLGA ? "olga" : "imp";
}

std::string olga_vdw_radii_csv() {
    /* data/olga_vdw_radii.csv is *derived* from the table above, the way
       data/fps_json_schema.json is derived from FPS.h's schema tables: the C++
       is the definition, the data file is the shipped, human-readable record
       of provenance, and a test regenerates it and fails on drift. */
    static const char *header = R"CSVHDR(# Olga's van der Waals radii, keyed by PDB atom name.
#
# Provenance
# ----------
#   upstream file : Olga/src/vdWRadii.json  (128 entries; 131 lines, of which
#                   "H1", "H2" and "H3" appear twice with identical values)
#   upstream repo : Fluorescence-Tools/Olga -- the FRET-restrained structural
#                   modelling program by Mykola Dimura et al., the program that
#                   produced the Zenodo 3376527 T4L screening table
#                   (Sanabria et al., Nat. Commun. 11, 1231 (2020)).
#   local checkout: ../ucfret/thirdparty/olga/src/vdWRadii.json
#   loader        : Olga/src/AV/Position.cpp:79-110, loadvdWRadii/pterosVDW.
#   licence       : Olga is GPL-3.0. Only the numeric table is vendored here,
#                   as data; no Olga code is copied into this module.
#
# Unit -- nanometre upstream, Angstrom here
# -----------------------------------------
# The upstream JSON is in NANOMETRES: pteros stores coordinates in nm, and
# Olga converts both coordinate and radius on the same line when it builds the
# (x, y, z, vdW) array the AV kernel consumes:
#
#     Olga/src/AV/Position.cpp:118-124, coordsVdW()
#         xyzw.emplace_back(frame.coord.at(i)[0] * 10.0f,
#                           frame.coord.at(i)[1] * 10.0f,
#                           frame.coord.at(i)[2] * 10.0f,
#                           pterosVDW(system, i) * 10.0f);
#
# so calculateAV() -- and every linker length, linker width and dye radius
# beside it -- works in ANGSTROM. IMP.bff works in Angstrom too, so the table
# is stored here already multiplied by 10: carbon 0.17 nm -> 1.70 A, which is
# the Bondi carbon radius and the same number FPS's own vdW.txt carries.
#
# Unknown atom names
# ------------------
# Olga does NOT fall back to an element lookup. pterosVDW is
#     vdWRMap.value(name, 0.15)
# (Position.cpp:110), i.e. a flat 0.15 nm = 1.50 A for any atom name not in
# this table -- smaller than every heavy atom in it. That fallback is Olga's
# code, not its table, so it is not a row here; see
# IMP::bff::get_olga_vdw_fallback_radius().
#
# This file is DERIVED from the C++ table in src/VdwRadii.cpp by
# IMP::bff::olga_vdw_radii_csv(); test/representation/test_olga_vdw_radii.py
# regenerates it and fails on drift. Edit the C++ table, not this file.
#
atom_name,radius_angstrom
)CSVHDR";
    std::ostringstream out;
    out << header;
    char buf[64];
    for (const auto &kv : vdwradii_olga_table()) {
        std::snprintf(buf, sizeof(buf), "%.4f", kv.second);
        out << kv.first << "," << buf << "\n";
    }
    return out.str();
}

IMPBFF_END_NAMESPACE
