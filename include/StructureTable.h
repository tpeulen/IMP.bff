/**
 *  \file IMP/bff/StructureTable.h
 *  \brief A structure as flat columns: the record every reader and writer
 *         here agrees on.
 *
 * One structure, one table, whatever produced it. #IMP::bff::read_structure_table
 * (#IMP/bff/StructureReader.h) fills it from a PDB or an mmCIF through IMP's
 * readers; #IMP::bff::RmfStructureWriter writes it as a trajectory. It is a
 * plain record -- no IMP type, no RMF type -- which is why it can sit between
 * the two and be wrapped without either one's SWIG interfaces.
 *
 * The columns are the ones an application puts in a record array. Filling
 * them is done in C++ on purpose: walking a hierarchy from Python costs a
 * measured ~100 us per atom in SWIG traffic (twelve decorator constructions
 * and a `Vector3D` handed to numpy, per atom), 1.3 s of the 1.4 s it took to
 * read a 9315-atom structure.
 *
 * \authors Thomas-Otavio Peulen
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_STRUCTURETABLE_H
#define IMPBFF_STRUCTURETABLE_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A coordinate file as parallel columns, one entry per atom.
/*!
    Every column has the same length, #get_n_atoms, except \c xyz which has
    three entries per atom. An application builds its record array by
    assigning the columns; nothing here is IMP-shaped, so nothing has to be
    walked one atom at a time on the Python side.
*/
struct IMPBFFEXPORT StructureTable {
    //! Flat, three per atom, Angstrom.
    std::vector<double> xyz;
    //! IMP's radius for the atom, Angstrom -- see #radius_no_interaction.
    std::vector<double> radius;
    //! Atomic mass, Dalton.
    std::vector<double> mass;
    //! The PDB temperature factor.
    std::vector<double> bfactor;
    //! The atom's index within its structure, as IMP reports it.
    std::vector<int> atom_id;
    //! The residue sequence number.
    std::vector<int> res_id;
    std::vector<std::string> chain, res_name, atom_name, element;
    //! The residue's insertion code, one character or empty when it has none.
    /*! A residue is `(chain, res_id, insertion_code)`: `res_id` alone names
        two residues wherever a file inserts one (`60`, `60A`). */
    std::vector<std::string> insertion_code;

    StructureTable() {}

    //! The coordinate column is the one that counts.
    /*! A table under construction has its columns assigned one at a time, so
        any other choice makes the count depend on assignment order. */
    int get_n_atoms() const { return static_cast<int>(xyz.size() / 3); }

    void get_xyz(double** out_view, int* n_out_view) const;
    void get_radius(double** out_view, int* n_out_view) const;
    void get_mass(double** out_view, int* n_out_view) const;
    void get_bfactor(double** out_view, int* n_out_view) const;
    void get_atom_id(int** out_view_i, int* n_out_view_i) const;
    void get_res_id(int** out_view_i, int* n_out_view_i) const;

    // A table is built as well as read: a caller with coordinates of its own
    // -- a trajectory frame, a generated chain, an application's record array
    // -- assembles one column by column and hands it to a writer.
    void set_xyz(const std::vector<double>& v) { xyz = v; }
    void set_radius(const std::vector<double>& v) { radius = v; }
    void set_mass(const std::vector<double>& v) { mass = v; }
    void set_bfactor(const std::vector<double>& v) { bfactor = v; }
    void set_atom_id(const std::vector<int>& v) { atom_id = v; }
    void set_res_id(const std::vector<int>& v) { res_id = v; }

    IMP_SHOWABLE_INLINE(StructureTable,
                        out << "StructureTable(" << get_n_atoms()
                            << " atoms)");
};
IMP_VALUES(StructureTable, StructureTables);

//! Is this a standard amino-acid or nucleotide residue name?
/*! The test #read_structure_table applies when \c only_standard_residues is
    set, exposed because a caller that filters its own table wants the same
    answer rather than a second list that drifts from this one. */
IMPBFFEXPORT bool is_standard_residue(const std::string& res_name);

//! Read a PDB or mmCIF into a #IMP::bff::StructureTable without IMP.
/*!
    The core's reader for the table #read_structure_table fills through
    `IMP::atom` -- the same atoms, in the same order, with the same numbers,
    so a caller that has only the core reads a structure the way the IMP
    module build does. What it reproduces, rule by rule:

    - **the atom set** of IMP's default selector: every ATOM and HETATM
      record, hydrogens and ligands included, except water (`HOH`, `DOD`)
      unless \p keep_water, and except alternate locations other than blank
      or `A`. Only the first model: a PDB is read up to its second `MODEL`
      record, an mmCIF's rows of any other `pdbx_PDB_model_num` are skipped.
    - **the order**: chains in file order and residues sorted by number
      within a chain. A PDB chain is a run of one chain identifier, so a
      chain that comes back after another (ligands after `TER`) is a chain
      of its own; an mmCIF chain is one author chain identifier however its
      rows are spread. The sort is stable, so residues sharing a number
      (insertion codes) keep their file order.
    - **the numbers**: coordinates and B-factors at single precision, as IMP
      stores them; the residue number of a PDB as a 16-bit integer; an mmCIF
      residue's number and insertion code from `auth_seq_id` (`12B` is 12,
      `B`), falling back to `label_seq_id`.
    - **the element** from the element column, else from the atom name by
      IMP's rule (the first letter of an ATOM name, the first two of a
      HETATM name); `Un`, mass 0, when neither names one. The mass is IMP's
      element table.
    - **the radius** #IMP::bff::imp_atom_radius by residue and atom name --
      the radius `IMP::atom::read_pdb` gives the particle -- rounded to
      single precision as IMP's CHARMM table stores it. An atom CHARMM
      leaves untyped (a ligand's) IMP gives exactly 1.7 where this gives
      1.7 at single precision, 5e-8 A apart.
    - **the atom name** without IMP's `HET:` prefix, as #read_structure_table
      reports it.

    `atom_id` is the record's serial number (`_atom_site.id`).

    \param[in] path the coordinate file; `.pdb`/`.ent` or `.cif`/`.mmcif`
    \param[in] keep_water keep `HOH`/`DOD`
    \param[in] only_standard_residues drop residues that are not
               #is_standard_residue
    \param[in] radius_no_interaction report \f$2^{-1/6} R_{min}\f$ rather
               than \f$R_{min}\f$, as #read_structure_table does
    \throw ValueException for an extension this does not read, including a
           compressed one
    \throw IOException when the file cannot be opened or parsed
*/
IMPBFFEXPORT StructureTable parse_structure_table(
        const std::string& path, bool keep_water = false,
        bool only_standard_residues = true,
        bool radius_no_interaction = true);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_STRUCTURETABLE_H
