/** \file IMP/bff/structure/ProteinSidechainRotamerLibrary.h
 * \brief The backbone-dependent protein side-chain rotamer library.
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_PROTEINSIDECHAINROTAMERLIBRARY_H
#define IMPBFF_PROTEINSIDECHAINROTAMERLIBRARY_H
#include <IMP/bff/util/IMPCompatibility.h>
#include <IMP/bff/spectroscopy/PhotophysicsQuenching.h>
#include <map>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

// --------------------------------------------------------------------------
// The backbone-dependent side-chain library
// --------------------------------------------------------------------------

/*! The second payload the container carries, and a different kind of thing
    from everything above. A probe or a spin label is an **ensemble**: a list
    of conformers with a weight each, which is what the `.drot` grid stores. A
    side-chain library is a **distribution over backbone conformation** -- a
    table indexed by amino acid and by the (phi, psi) bin the residue sits in,
    holding for each rotamer a probability, its chi angles and their standard
    deviations. Same envelope, different payload, which is what `PtoKind` is
    for: `rot.bbdep.records` sits beside `drot.grid` in one grammar.

    The table is the Dunbrack-2010 backbone-dependent library as FASPR ships
    it (`dun2010bbdep.bin`; Shapovalov & Dunbrack, Structure 2011;19:844-858,
    free for academic use -- the notice travels inside the container, in
    `bbdep.json`). **The conversion keeps FASPR's own 20-byte records byte for
    byte**: a record is `float32 probability` then eight `int16` in tenths of a
    degree (chi1..chi4, then their sigmas), and one amino acid's records are
    1296 (phi, psi) bins x its rotamer count, contiguous. Nothing is
    re-derived, so #IMP::bff::write_protein_sidechain_dunbrack_bin reproduces the input file
    exactly -- which is both the correctness proof and what lets the vendored
    FASPR engine, which seeks in that file by byte offset, run against a
    shipped container.

    \note Querying one is also #IMP::rotamer::RotamerLibrary's job, and that is
    what #IMP::bff::get_anchor_cb_position uses. The two read different files:
    IMP's reads the Dunbrack *text* library, this reads the binary FASPR ships
    and the container that carries it. Neither can read the other's, which is
    the whole reason both are here. */

//! The rotamers of one residue type at one backbone conformation.
struct IMPBFFEXPORT ProteinSidechainDunbrackRotamers {
    //! `n_rotamers * n_chi` chi angles in degrees, rotamer-major.
    std::vector<double> chi;
    //! The same shape: each chi's standard deviation, degrees.
    std::vector<double> sigma;
    //! One per rotamer, as the library stores it (they sum to ~1 per bin).
    std::vector<double> probability;
    int n_rotamers, n_chi;

    ProteinSidechainDunbrackRotamers() : n_rotamers(0), n_chi(0) {}

    void get_chi(double** out_view, int* n_out_view) const;
    void get_sigma(double** out_view, int* n_out_view) const;
    void get_probability(double** out_view, int* n_out_view) const;

    IMP_SHOWABLE_INLINE(ProteinSidechainDunbrackRotamers,
                        out << "DunbrackRotamers(" << n_rotamers << " x "
                            << n_chi << " chi)");
};
IMP_VALUES(ProteinSidechainDunbrackRotamers, ProteinSidechainDunbrackRotamersList);

//! Convert FASPR's `dun2010bbdep.bin` into a `.drot.pto` container.
/*!
    Writes one `rot.bbdep.records` object per amino acid -- that residue's
    records exactly as they lie in the binary -- plus a `rot.bbdep.header`
    describing the bin scheme, the record layout and the library's
    provenance, and a `drot.catalog` naming the residues. Each object is
    compressed independently with Zstd level 3. Readers also accept existing
    Brotli-compressed containers.

    \param[in] bin_path FASPR's `dun2010bbdep.bin`
    \param[in] path the container to write
    \throw IOException when the binary is missing or is not the expected size
*/
IMPBFFEXPORT void write_protein_sidechain_dunbrack_library(const std::string& bin_path,
                                         const std::string& path);

//! Write the library back out as `dun2010bbdep.bin`.
/*!
    The inverse of `write_protein_sidechain_dunbrack_library`, byte for byte. Two things need
    it: proving the conversion lost nothing, and handing the vendored FASPR
    engine the file it insists on opening by name.
*/
IMPBFFEXPORT void write_protein_sidechain_dunbrack_bin(const std::string& path,
                                     const std::string& bin_path);

//! The rotamers of `residue` at backbone angles (`phi`, `psi`).
/*!
    Reads that residue's object and no other. The two cuts are FASPR's own
    defaults, applied in FASPR's order: rotamers come sorted by probability,
    the first one below \p probability_min ends the list, and the list also
    ends once the accumulated probability passes \p probability_accumulated.
    Pass 0 and 1 to take the bin whole.

    \param[in] path a container written by `write_protein_sidechain_dunbrack_library`
    \param[in] residue one-letter code; `A` and `G` have no rotamers
    \param[in] phi,psi degrees
    \throw ValueException for a residue this library does not carry
*/
IMPBFFEXPORT ProteinSidechainDunbrackRotamers read_protein_sidechain_dunbrack_rotamers(
        const std::string& path, char residue, double phi, double psi,
        double probability_min = 0.01,
        double probability_accumulated = 0.97);

// --------------------------------------------------------------------------
// Side-chain packing over that library
// --------------------------------------------------------------------------

//! Repack a protein's side chains to the global-minimum-energy conformation.
/*!
    Reads \p pdb_in (backbone N, CA, C, O required per residue; any side chains
    present are ignored), repacks every rotatable side chain, and writes the
    result to \p pdb_out with the residue numbering and ordering preserved.
    The search is deterministic: two calls on the same input and library give
    identical coordinates.

    \param[in] pdb_in the input backbone PDB
    \param[in] pdb_out where the repacked PDB is written
    \param[in] rotamer_library the Dunbrack-2010 library -- either FASPR's
               binary `dun2010bbdep.bin` or a `.drot.pto` container carrying
               it, which is unpacked to a scratch copy because the engine
               seeks in that file by byte offset
    \param[in] verbose echo the packing log to stdout

    \throw IOException when the input or the library cannot be read
    \throw std::runtime_error on a fatal packing error -- an incomplete
           backbone, an unknown library layout

    The algorithm is FASPR (Huang, Pearce & Zhang, *Bioinformatics*
    2020;36:3758-3765, MIT, github.com/tommyhuangthu/FASPR): CHARMM19 van der
    Waals, hydrogen bonds, disulfides and a rotamer prior, minimised by
    DEE-Goldstein and DEE-split elimination and then tree decomposition. It is
    vendored in `src/internal/ProteinSidechainFaspr*.{h,cpp}`, which carry the original notice, and
    pinned against the reference executable by
    `test/faspr/test_faspr_port.py`.
*/
IMPBFFEXPORT void pack_protein_sidechains(const std::string& pdb_in,
                             const std::string& pdb_out,
                             const std::string& rotamer_library,
                             bool verbose = false);

// --------------------------------------------------------------------------
// Where a quencher's redox centre can be: its side-chain rotamers
// --------------------------------------------------------------------------

//! The positions a residue's quenching centre takes over its rotamers.
/*!
    One residue of a structure, its crystal centre (the centroid of its
    quenching atoms as deposited) and the centres of the Dunbrack rotamers
    that fit, with their library probabilities renormalised over the ones
    that fit. When no rotamer fits, or the residue has no rotatable side chain
    this library can place, the ensemble is the crystal centre with
    probability one and #from_library is false.
*/
struct IMPBFFEXPORT QuencherRotamerEnsemble {
    std::string chain;
    int res_id;
    std::string res_name;
    //! The backbone angles the library was read at, degrees (-60, -45 at a
    //! chain end or a break; NaN when the library was not read).
    double phi, psi;
    //! Three doubles: the crystal quenching centre.
    std::vector<double> crystal;
    //! Flat, three per rotamer kept.
    std::vector<double> centres;
    //! One per rotamer kept, summing to one.
    std::vector<double> probabilities;
    //! How many rotamers the library offered before the clash screen.
    int n_library;
    //! False when the ensemble is the crystal fallback.
    bool from_library;

    QuencherRotamerEnsemble()
        : res_id(0), phi(0.0), psi(0.0), n_library(0), from_library(false) {}

    unsigned int size() const {
        return static_cast<unsigned int>(probabilities.size());
    }
    void get_crystal(double** out_view, int* n_out_view) const;
    //! `(n, 3)` centres in Angstrom.
    void get_centres(double** out_matrix, int* n_out_rows,
                     int* n_out_cols) const;
    void get_probabilities(double** out_view, int* n_out_view) const;

    IMP_SHOWABLE_INLINE(QuencherRotamerEnsemble,
                        out << "QuencherRotamerEnsemble(" << res_name << " "
                            << chain << res_id << ", " << size()
                            << (from_library ? " rotamers)" : " crystal)"));
};
IMP_VALUES(QuencherRotamerEnsemble, QuencherRotamerEnsembles);

//! The quenching centres one residue reaches over its side-chain rotamers.
/*!
    The backbone angles phi and psi are measured on the structure (a residue
    is bonded to its neighbour in the same chain when the C-N distance is
    below 1.8 A; a chain end or a break takes -60 and -45 degrees), the
    library is read at them (#read_protein_sidechain_dunbrack_rotamers, its
    default cuts), and every rotamer is built from the crystal side chain by
    turning chi1 about CA-CB and chi2 about CB-CG (higher chi keep their
    crystal values). A rotamer is dropped when any of its side-chain atoms
    beyond CB comes within \p clash of an atom of another residue (heavy
    atoms are what the caller passes). The kept rotamers' probabilities are
    renormalised; when none is kept the crystal centre is the ensemble.

    Placed: CYS SER THR VAL (chi1), MET TRP TYR PHE HIS LEU ILE ASP ASN GLU
    GLN LYS ARG (chi1, chi2). Anything else (PRO, ALA, GLY, a ligand) is its
    crystal centre.

    \param[in] chains,res_ids,res_names,atom_names one per atom
    \param[in] coords flat, three per atom
    \param[in] chain,res_id the residue
    \param[in] library_path a container written by write_protein_sidechain_dunbrack_library
               (IMP.bff ships one, `get_data_path("rotamer_library")/sidechains.drot.pto`)
    \param[in] quench_atoms the atom names whose centroid is the centre
    \param[in] clash Angstrom
    \throw ValueException when the residue is not in the structure or has
           none of \p quench_atoms
*/
IMPBFFEXPORT QuencherRotamerEnsemble quencher_rotamer_ensemble(
        const std::vector<std::string>& chains,
        const std::vector<int>& res_ids,
        const std::vector<std::string>& res_names,
        const std::vector<std::string>& atom_names,
        const std::vector<double>& coords,
        const std::string& chain, int res_id,
        const std::string& library_path,
        const std::vector<std::string>& quench_atoms,
        double clash = 2.8);

//! quencher_rotamer_ensemble() for every residue of the named types.
/*!
    Residues in structure order. A residue's quenching atoms are its
    \p table entry's `quench_atoms` (\p table empty: #amino_acid_quenching_defaults);
    a residue with none of them present -- a labelled site cut back to its
    CB, say -- cannot quench and is skipped.
*/
IMPBFFEXPORT QuencherRotamerEnsembles quencher_rotamer_ensembles(
        const std::vector<std::string>& chains,
        const std::vector<int>& res_ids,
        const std::vector<std::string>& res_names,
        const std::vector<std::string>& atom_names,
        const std::vector<double>& coords,
        const std::vector<std::string>& residue_types,
        const std::string& library_path,
        const std::map<std::string, ResidueQuenching>& table =
                std::map<std::string, ResidueQuenching>(),
        double clash = 2.8);

IMPBFF_END_NAMESPACE


#endif //IMPBFF_PROTEINSIDECHAINROTAMERLIBRARY_H
