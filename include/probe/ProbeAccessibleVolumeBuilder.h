/**
 *  \file IMP/bff/probe/ProbeAccessibleVolumeBuilder.h
 *  \brief Building an accessible volume without an IMP::Model: the array
 *         door, the lattice search behind it, and the PDB read.
 *
 * #IMP::bff::ProbeAccessibleVolume is the result; this is the core's way to one.
 * #IMP::bff::get_av takes **raw arrays** -- atoms with radii and an
 * attachment coordinate; no file, no fps position, no Model -- and
 * #IMP::bff::get_av_lattice is the search it runs: the decorator's lattice
 * path (window, two occupancy rasters, attachment-atom subtraction, bounded
 * search, dye-radius carve, contact weighting, read-out) as one call over
 * spheres. The PDB reader, the van der Waals table and the attachment-point
 * lookup are the way from a file to those arrays.
 *
 * The doors that go through an `IMP::Model` -- #IMP::bff::resample_av, which
 * decorates a particle as an #IMP::bff::ProbeAccessibleVolumeDecorator, and #IMP::bff::get_av_from_structure
 * and #IMP::bff::get_avs_for_structure, which read a PDB with `IMP::atom` --
 * are the connection layer's, in ProbeAccessibleVolumeDecorator.h (PRD-137 step 5). Both roads compute
 * the same volume, and test/test_density_grid.py pins that with records
 * taken from each: the array door used to build a Model of its own to reach
 * the decorator, and the lessons that road taught -- IMP numbers voxels with
 * *x* fastest, so a C-order reshape into `(nx, ny, nz)` returns the volume
 * **transposed** (PRD-113 stage 3a) -- are kept in the read-out here.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_PROBEACCESSIBLEVOLUMEBUILDER_H
#define IMPBFF_PROBEACCESSIBLEVOLUMEBUILDER_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/probe/ProbeAccessibleVolume.h>
#include <IMP/bff/probe/ProbeAccessibleVolumeDefaults.h>
#include <IMP/bff/probe/DensityGrid.h>
#include <IMP/algebra/Vector3D.h>
#include <IMP/algebra/VectorD.h>


#include <IMP/bff/util/IMPCompatibility.h>


#include <map>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Van der Waals radius per atomic number, Angstrom; 1.70 for anything else.
/*! Bondi's element radii, for the readers that want an element's size
    (#PDBAtomRecord::vdw_radius, the Labelizer's surface areas). **Not** an
    accessible volume's obstacle set: that is IMP's radii times
    #IMP::bff::AV_DEFAULT_RADIUS_SCALE at every door
    (#IMP::bff::imp_atom_radius, #load_structure_with_vdw). */
IMPBFFEXPORT std::map<int, double> vdw_radii();

//! The van der Waals radius of an element symbol, A.
/*! The same table #vdw_radii holds, addressed the way a caller with an atom
    name has it. Bondi's values; an unknown symbol gets carbon's 1.7, which is
    what every reader in this module already falls back to.

    There were two of these -- this one by atomic number and a second by
    symbol in the dye sampler -- and they disagreed about hydrogen (1.20
    against 1.10, Bondi against Rowland & Taylor). One table now. */
IMPBFFEXPORT double vdw_radius(const std::string& element);

//! The element symbol of a PDB ATOM/HETATM record.
/*!
    Columns 77-78 carry it in a modern PDB and are used verbatim when present.
    Older files — the shipped FPS screening structures are 66-character records
    — leave them empty, so the element comes from the atom-name field, where it
    is *right-justified in columns 13-14*: a blank or numeric column 13 means a
    one-letter element, so `" CA "` is an alpha-carbon while `"CA  "` is
    calcium. A two-letter reading is additionally rejected when columns 15-16
    contain a digit, which is how a four-character hydrogen name such as
    `"HE21"` is written — helium would otherwise win.
*/
IMPBFFEXPORT std::string element_symbol_from_pdb_line(const std::string& line);

//! One ATOM/HETATM record: where it is, how big, and what names it.
/*! Every field a reader in this package wanted of one. There were **two** such
    records until 2026-08-20 — this one, and a serial/element-carrying twin the
    MOL2 writer used — with two parsers behind them, only one of them cached. */
struct IMPBFFEXPORT PDBAtomRecord {
    //! The atom serial, columns 7-11. What `CONECT` records refer to.
    int serial;
    std::string chain;
    int resseq;
    std::string res_name, atom_name;
    double x, y, z;
    double vdw_radius;
    //! Temperature factor, columns 61-66; zero when absent or unparseable.
    double bfactor;
    //! The element symbol — see #IMP::bff::element_symbol_from_pdb_line.
    std::string element;
    //! A HETATM record (an ATOM one when false).
    bool hetatm;
    //! The alternate-location indicator, column 17; empty when blank.
    std::string alt_loc;
    //! How many `MODEL` records precede the atom: 0 or 1 is the first model.
    int model;
    PDBAtomRecord()
        : serial(0), resseq(0), x(0), y(0), z(0), vdw_radius(1.70), bfactor(0),
          hetatm(false), model(0) {}
    IMP_SHOWABLE_INLINE(PDBAtomRecord,
                        out << "PDBAtomRecord(" << chain << resseq << ":"
                            << atom_name << ")");
};
IMP_VALUES(PDBAtomRecord, PDBAtomRecords);

//! Every ATOM/HETATM record of a PDB, with a van der Waals radius each.
/*!
    Parsed directly rather than through `IMP::atom::read_pdb`, which warns per
    unsupported HETATM residue and there are thousands in a labelled structure.

    **Cached on (path, mtime, size)**: an fps.json with a dozen positions reads
    the same file a dozen times, and the strip below reads it again per site.

    \throw IOException when the file has no coordinates at all
*/
IMPBFFEXPORT std::vector<PDBAtomRecord> read_pdb_records(
        const std::string& pdb_path);

//! `(N, 4)` of x, y, z, radius — the obstacle array the solver takes.
/*! Every ATOM/HETATM record (#read_pdb_records), its radius **IMP's**
    (#IMP::bff::imp_atom_radius, by residue and atom name) times
    \p radius_scale -- the one atom-radius set an accessible volume uses at
    every door (owner, 2026-10-12), so this file door and the structure door
    (#get_av_from_structure) inflate the same atoms by the same numbers. It
    used to be Bondi's element table (#vdw_radii), FPS's set, a second radii
    set the scale was fitted to replace.

    \param[in] radius_scale the factor on IMP's radii,
               #IMP::bff::AV_DEFAULT_RADIUS_SCALE (fitted to FPS); 1.0 gives
               IMP's radii as they are */
IMPBFFEXPORT void load_structure_with_vdw(const std::string& pdb_path,
                                          double** out_view, int* n_out_view,
                                          double radius_scale =
                                                  IMP::bff::AV_DEFAULT_RADIUS_SCALE);

//! The coordinates of an attachment atom, resolved by identity.
/*!
    A miss stays a miss: an unresolvable site returns an empty view rather than
    a positional guess, because a dye attached to an unrelated atom yields a
    plausible and entirely wrong accessible volume.

    \param[in] pdb_path the structure the site names
    \param[in] chain empty matches any chain
    \param[in] resseq,atom_name the site
    \param[out] out_view,n_out_view three coordinates, or zero-length
*/
IMPBFFEXPORT void get_attachment_point(const std::string& pdb_path,
                                        const std::string& chain, int resseq,
                                        const std::string& atom_name,
                                        double** out_view, int* n_out_view);


//! The accessible-volume search over spheres, without an IMP::Model.
/*! The decorator's lattice path (ProbeAccessibleVolumeDecorator::resample_lattice_*) as one call over a
    list of (x, y, z, radius) spheres: the window, the two occupancy rasters,
    the attachment-atom subtraction, the bounded search, the dye-radius carve
    and the contact weighting, then the read-out. `source_radius` is the
    radius of the sphere at the attachment site (0 when none is there), the
    one the raster subtracts around the source so the linker is not blocked
    by the atom it is tied to. `search_stencil` is 342 (the default, FPS's
    link search: the 3-voxel cube hop from every node), 74 (the seed-only
    "fast" search, the reference metric) or 26; `allowed_sphere_radius` defaults to #IMP::bff::AV_DEFAULT_ALLOWED_SPHERE
    and < 0 derives the clearance as the decorator does. get_av() is this over a caller's `(N, 4)` array. */
IMPBFFEXPORT ProbeAccessibleVolume get_av_lattice(
        const std::vector<IMP::algebra::Vector4D>& spheres,
        const IMP::algebra::Vector3D& source, double source_radius,
        double linker_length, double linker_width,
        double r1, double r2, double r3, double grid_resolution,
        double allowed_sphere_radius = IMP::bff::AV_DEFAULT_ALLOWED_SPHERE, int search_stencil = IMP::bff::AV_DEFAULT_SEARCH_STENCIL,
        double contact_volume_thickness = 0.0,
        double contact_volume_trapped_fraction = -1.0);

//! An accessible volume from raw atomic coordinates.
/*!
    The **array** front door: no PDB file and no fps position definition, which
    is what a restraint has.

    \param[in] atoms_xyzr,n_atoms,n_cols obstacles, `(N, 4)` of x, y, z, radius
    \param[in] source_xyz where the linker is tied, three coordinates
    \param[in] linker_length,linker_width the linker
    \param[in] r1,r2,r3 dye radii; `(3.5, 0, 0)` is the AV1 single-sphere model
    \param[in] grid_resolution voxel spacing, A
    \param[in] allowed_sphere_radius the allowed sphere, A, on by default
               (#IMP::bff::AV_DEFAULT_ALLOWED_SPHERE); < 0 derives it from the linker width
               (#IMP::bff::av_derived_allowed_sphere_radius)
    \param[in] search_stencil see #IMP::bff::resample_av
*/
IMPBFFEXPORT ProbeAccessibleVolume get_av(
        double* atoms_xyzr, int n_atoms, int n_cols,
        const std::vector<double>& source_xyz,
        double linker_length = IMP::bff::AV_DEFAULT_LINKER_LENGTH,
        double linker_width = IMP::bff::AV_DEFAULT_LINKER_WIDTH,
        double r1 = IMP::bff::AV_DEFAULT_RADIUS1, double r2 = 0.0, double r3 = 0.0,
        double grid_resolution = IMP::bff::AV_DEFAULT_SIMULATION_GRID_RESOLUTION,
        double allowed_sphere_radius = IMP::bff::AV_DEFAULT_ALLOWED_SPHERE, int search_stencil = 0);




//! An accessible volume from a PDB file and an attachment atom, with no IMP.
/*!
    The **file** front door of the core, one call the way labellib's
    `dyeDensityAV3` is one call: the structure's atoms with IMP's radii times
    \p radius_scale (#load_structure_with_vdw), the attachment atom's position
    (#get_attachment_point), then #get_av. This is the road the Labelizer's
    AV-mode sites take under the standalone build.

    The IMP module build also has #get_av_from_structure (ProbeAccessibleVolumeDecorator.h), which reads
    the structure through IMP::atom. Both roads take IMP's radii times the
    same scale (owner, 2026-10-12: one atom-radius set); they still differ in
    which atoms they read -- this door reads every record, waters and all,
    while the structure door drops waters and strips the attachment residue.

    \param[in] pdb_path the structure
    \param[in] chain,resseq,atom_name the attachment atom
    \param[in] linker_length,linker_width the linker
    \param[in] r1,r2,r3 dye radii; `(3.5, 0, 0)` is the AV1 single-sphere model
    \param[in] grid_resolution voxel spacing, A
    \param[in] allowed_sphere_radius the allowed sphere, A, on by default
               (#IMP::bff::AV_DEFAULT_ALLOWED_SPHERE); < 0 derives it from the linker width
    \param[in] search_stencil see #IMP::bff::resample_av
    \param[in] radius_scale the factor on IMP's radii as the obstacle set
               (#IMP::bff::AV_DEFAULT_RADIUS_SCALE)
    \throw ValueException when the attachment atom is not in the structure
*/
IMPBFFEXPORT ProbeAccessibleVolume get_av_from_pdb(
        const std::string& pdb_path, const std::string& chain, int resseq,
        const std::string& atom_name,
        double linker_length = IMP::bff::AV_DEFAULT_LINKER_LENGTH,
        double linker_width = IMP::bff::AV_DEFAULT_LINKER_WIDTH,
        double r1 = IMP::bff::AV_DEFAULT_RADIUS1, double r2 = 0.0, double r3 = 0.0,
        double grid_resolution = IMP::bff::AV_DEFAULT_SIMULATION_GRID_RESOLUTION,
        double allowed_sphere_radius = IMP::bff::AV_DEFAULT_ALLOWED_SPHERE, int search_stencil = 0,
        double radius_scale = IMP::bff::AV_DEFAULT_RADIUS_SCALE);

//! An accessible volume from a structure as IMP reads it: the **structure** front door.
/*!
    The attachment residue is stripped (#IMP::bff::stripped_pdb_for, with
    \p strip_mask), then the atoms are read the way `IMP::atom::read_pdb`
    with a `NonWaterPDBSelector` reads them -- the first model only, the
    alternate location blank or `A`, no `HOH`/`DOD` -- each inflated by IMP's
    radius (#IMP::bff::imp_atom_radius) times \p radius_scale, and the
    lattice search (#get_av_lattice) runs from the attachment atom. The point
    weights are forced to one except for an accessible *contact* volume
    (\p contact_volume_thickness > 0 and a trapped fraction in [0, 1)), whose
    weights are the answer.

    One implementation for both builds: the IMP module build used to read the
    structure into an IMP::Model and resample a decorated particle, which made
    contact volumes unreachable from the standalone build; that route computes
    the same volume (`test/test_av_structure_door.py` holds them equal).

    \param[in] disc_step the voxel spacing, A
    \param[in] allowed_sphere_radius the allowed sphere; < 0 derives it from
               the linker width (#IMP::bff::av_derived_allowed_sphere_radius)
    \throw ValueException when the attachment atom is not in the structure
*/
IMPBFFEXPORT ProbeAccessibleVolume get_av_from_structure(
        const std::string& pdb_path, const std::string& chain, int resseq,
        const std::string& atom_name, double linker_length, double linker_width,
        double r1, double r2, double r3,
        double disc_step = IMP::bff::AV_DEFAULT_SIMULATION_GRID_RESOLUTION,
        const std::string& strip_mask = "",
        double allowed_sphere_radius = IMP::bff::AV_DEFAULT_ALLOWED_SPHERE,
        double contact_volume_thickness = 0.0,
        double contact_volume_trapped_fraction = -1.0,
        double radius_scale = IMP::bff::AV_DEFAULT_RADIUS_SCALE);

//! The same front door, from an fps.json `Positions` entry.
/*!
    A position is data, so this takes it as data (JSON text), not as separate
    keywords. The entry's fields -- chain_identifier, residue_seq_number,
    atom_name, linker_length, linker_width, radius1..radius3 (an AV3 position's
    absent radii are FPS's three, #av_position_radii), strip_mask,
    allowed_sphere_radius, contact_volume_*, radius_scale and
    simulation_grid_resolution -- are read the way the fps dictionary states
    them, and a declared `simulation_grid_resolution` that disagrees with
    \p disc_step raises: that field would otherwise be silently overridden.

    \param[in] disc_step the voxel spacing, A; `<= 0` derives it from the
               entry's `simulation_grid_resolution` and then refuses a declared
               one that disagrees
    \throw ValueException on a disagreeing resolution or a missing attachment site
*/
IMPBFFEXPORT ProbeAccessibleVolume get_av_from_structure(
        const std::string& pdb_path, const std::string& position_json,
        double disc_step = -1.0);

//! A position's dye radii (radius1..3), the absent ones from its model's defaults.
/*! An AV3 position that leaves its radii out gets FPS's three (5, 4.5, 1.5),
    not the AV1 radius and two zeros: zero is this library's "unused"
    sentinel, so an AV3 read with AV1 defaults would silently become an AV1.
    \param[in] position_json one fps.json `Positions` entry, as JSON text */
IMPBFFEXPORT IMP::algebra::Vector3D av_position_radii(const std::string& position_json);

//! The shortest linker path to every voxel, LabelLib's `minLinkerLength`.
/*!
    The same lattice search #get_av_lattice runs, read out as path lengths
    instead of as an accessible volume: each voxel carries the length of the
    shortest obstacle-free linker path from the attachment atom to it, and a
    voxel the linker cannot reach carries a negative value. The grid is the
    volume's grid -- same origin, same spacing, same (x, y, z) order.

    \param[in] spheres obstacles, (x, y, z, radius)
    \param[in] source,source_radius the attachment atom and its own radius
    \param[in] linker_length,linker_width the linker
    \param[in] dye_radius the probe radius the path must clear
    \param[in] h voxel spacing, A
    \param[in] allowed_sphere_radius the allowed sphere, A, on by default
               (#IMP::bff::AV_DEFAULT_ALLOWED_SPHERE); < 0 derives it from the linker width
    \param[in] search_stencil see #IMP::bff::resample_av
    \return a new #DensityGrid the caller owns
*/
IMPBFFEXPORT DensityGrid* get_linker_path_lengths(
        const std::vector<IMP::algebra::Vector4D>& spheres,
        const IMP::algebra::Vector3D& source, double source_radius,
        double linker_length, double linker_width, double dye_radius, double h,
        double allowed_sphere_radius = IMP::bff::AV_DEFAULT_ALLOWED_SPHERE, int search_stencil = IMP::bff::AV_DEFAULT_SEARCH_STENCIL);

//! The same, over a caller's `(N, 4)` array -- the twin of #get_av.
/*! Reads the obstacles and the attachment site exactly as #get_av does, so
    the path lengths and the volume are the same search read out twice. */
IMPBFFEXPORT DensityGrid* get_linker_path_lengths(
        double* atoms_xyzr, int n_atoms, int n_cols,
        const std::vector<double>& source_xyz,
        double linker_length = IMP::bff::AV_DEFAULT_LINKER_LENGTH,
        double linker_width = IMP::bff::AV_DEFAULT_LINKER_WIDTH,
        double dye_radius = IMP::bff::AV_DEFAULT_RADIUS1,
        double grid_resolution = IMP::bff::AV_DEFAULT_SIMULATION_GRID_RESOLUTION,
        double allowed_sphere_radius = IMP::bff::AV_DEFAULT_ALLOWED_SPHERE, int search_stencil = 0);

IMPBFF_END_NAMESPACE

#endif //IMPBFF_PROBEACCESSIBLEVOLUMEBUILDER_H
