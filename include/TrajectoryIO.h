/**
 *  \file IMP/bff/TrajectoryIO.h
 *  \brief Rotamer-library trajectories: BinaryCIF (read and write), DCD
 *         and XTC (read).
 *
 * BinaryCIF is this package's trajectory format as of 2026-08-19, replacing
 * DCD and XTC. It is smaller than either -- 1.27 bytes per coordinate against
 * DCD's 4.31 and XTC's 1.60, on a 0.1 A grid -- and it is decoded by the C
 * implementation of `ihm` that IMP already vendors, so reading it costs no new
 * dependency and no build change.
 *
 * The measurements, the precision that buys the size, and the two encoding
 * traps are in `okf/validation/bcif_for_trajectories.md`. The encoder is
 * #write_bcif_trajectory (it was the Python program `imp_bff_traj2bcif`,
 * which existed because python-ihm's writer implements neither FixedPoint nor
 * IntegerPacking).
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_TRAJECTORYIO_H
#define IMPBFF_TRAJECTORYIO_H

#include <IMP/bff/bff_config.h>

#include <IMP/bff/IMPCompatibility.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Read coordinates from a BinaryCIF trajectory.
/*!
    The file stores one row per (atom, frame) with `x`, `y`, `z` columns, laid
    out atom-major -- every frame of atom 0, then atom 1 -- so that the delta
    encoding runs along an atom's own series.

    \param[in] path the `.bcif` file
    \param[in] n_atoms atoms per frame; the frame count follows from the row
               count, and a row count that is not a multiple of this is an error
               rather than a silent truncation
    \param[in] category the CIF category to read, `_rotamer_coord` by default
    \param[out] out_view,n_out_view flat `(n_frames, n_atoms, 3)` in Angstrom
    \throws IMP::IOException if the file cannot be read or does not parse
    \throws IMP::ValueException if the row count is not a multiple of \p n_atoms
*/
IMPBFFEXPORT void read_bcif_trajectory(
        const std::string& path, int n_atoms,
        const std::string& category, double** out_view, int* n_out_view);

//! How many rows a BinaryCIF trajectory holds, without decoding it all.
/*!
    Cheap enough to call first: it still parses, but it discards the
    coordinates instead of accumulating them.
*/
IMPBFFEXPORT int bcif_trajectory_rows(const std::string& path,
                                      const std::string& category);

//! What a DCD header says, without loading any coordinates.
struct IMPBFFEXPORT DCDHeader {
    int n_frames, n_atoms, first_step, step_stride, charmm_version;
    double time_step;
    bool has_unit_cell;
    //! `"<"` for little-endian, `">"` for big — the struct prefix Python used.
    std::string endianness;
    //! Byte offset at which frame data begins.
    int offset;

    DCDHeader()
        : n_frames(0), n_atoms(0), first_step(0), step_stride(0),
          charmm_version(0), time_step(0.0), has_unit_cell(false),
          endianness("<"), offset(0) {}

    IMP_SHOWABLE_INLINE(DCDHeader, out << "DCDHeader(" << n_frames
                                       << " frames, " << n_atoms << " atoms)");
};
IMP_VALUES(DCDHeader, DCDHeaders);

//! Read a DCD header without loading coordinates.
/*!
    DCD carries no magic number for endianness; the convention is to read the
    leading Fortran record length and see which byte order makes it the expected
    84.

    Only what the rotamer libraries need is implemented — fixed atom counts, no
    velocity blocks, no four-dimensional trajectories. Anything outside that
    throws rather than guessing, because a trajectory silently read as the wrong
    shape is worse than one that refuses.

    \throw ValueException when the file is not a DCD this reader handles
    \throw IOException when it cannot be read
*/
IMPBFFEXPORT DCDHeader read_dcd_header(const std::string& path);

//! Read coordinates from a DCD trajectory.
/*! \param[in] max_frames stop after this many; negative reads all of them
    \param[out] out_view,n_out_view `n_frames * n_atoms * 3`, in the file's own
                units (Angstrom for the bundled libraries) */
IMPBFFEXPORT void read_dcd(const std::string& path, int max_frames = -1,
                           double** out_view = NULL, int* n_out_view = NULL);

//! Coordinates from a trajectory, whatever format it is in.
/*!
    **BinaryCIF is the format this package stores.** The rotamer libraries were
    re-encoded on 2026-08-19: 44.78 MB of DCD and XTC became 17.99 MB of
    `.bcif`, verified exact on a 0.1 A grid, and read through the C parser IMP
    already vendors. `.dcd` still reads, because the format is not gone from the
    world — a user's own library may be one. It is simply not what is shipped.

    \param[in] n_atoms required for BinaryCIF, which stores one row per
               (atom, frame) and cannot infer the split; callers have it from
               the companion PDB. Ignored for DCD, which carries its own.
    \throw ValueException for any other suffix, or for a BinaryCIF with no
           \p n_atoms
*/
IMPBFFEXPORT void read_trajectory(const std::string& path, int n_atoms,
                                  int max_frames, double** out_view,
                                  int* n_out_view);

//! Write coordinates as a BinaryCIF trajectory.
/*!
    The encoder `bin/imp_bff_traj2bcif` was, moved into the library
    (`imp_bff traj2bcif` is its command line). One category, one row per
    (atom, frame), columns `x`, `y`, `z` laid out **atom-major** -- every frame
    of atom 0, then atom 1 -- which #read_bcif_trajectory reads back.

    **Lossless is the default** (\p grid_a <= 0): float32 in, float32 out,
    `ByteArray` type 32, bit-exact and 4.00 bytes per coordinate. Quantising
    is the option, not the default: a 0.1 A grid moves transition-dipole
    directions by 1.6 degrees and broke the FRETpredict pins; at 0.001 A they
    hold. With a grid each column is `FixedPoint` and then whichever chain is
    smaller -- plain int16/int32, or `Delta` followed by `IntegerPacking` into
    int8 with escape runs. Delta encoding assumes the next value resembles the
    last one, which holds for an MD trajectory and not for a rotamer library,
    whose consecutive frames are independent conformers; the sizes are
    arithmetic, so choosing costs nothing.

    Two traps, both caught by the reader a long way from the cause: the int8
    sentinels 127 and -128 are also legitimate values, so a delta of exactly
    +-127 is written as an escape plus a remainder; and the `encoding` list is
    stored in **encode** order, because `ihm_format.c` prepends each entry as
    it parses.

    The msgpack document is byte-for-byte what the Python encoder wrote with
    `msgpack.packb(doc, use_bin_type=True)`, including its `encoder` string.

    \param[in] path the `.bcif` to write
    \param[in] in_xyz,n_frames,n_atoms,n_dim `(n_frames, n_atoms, 3)` in
               Angstrom
    \param[in] grid_a quantisation grid in Angstrom; <= 0 is lossless float32
    \param[in] category,block the CIF category and data block names
    \return the number of bytes written
    \throw ValueException when \p n_dim is not 3 or a quantised column is
           empty
    \throw IOException when \p path cannot be written
*/
IMPBFFEXPORT long write_bcif_trajectory(
        const std::string& path, double* in_xyz, int n_frames, int n_atoms,
        int n_dim, double grid_a = -1.0,
        const std::string& category = "_rotamer_coord",
        const std::string& block = "rotamers");

//! The atom count of an XTC trajectory, from its first frame header.
/*! \throw IOException when the file cannot be read or is not an XTC */
IMPBFFEXPORT int read_xtc_n_atoms(const std::string& path);

//! Read coordinates from a GROMACS XTC trajectory, in **Angstrom**.
/*!
    The compressed-coordinate format of GROMACS' `xdrfile` (big-endian XDR,
    frame magic 1995), decoded here with no library: frames of up to nine atoms
    are plain floats, larger ones the integer run-length scheme with its
    "swap the first two atoms of a run" water trick. Coordinates are decoded to
    float32 nanometres exactly as `xdrfile` does and then multiplied by ten,
    which is what `mdtraj.load(...).xyz * 10` gave the Python programs.

    \param[in] max_frames stop after this many; negative reads all of them
    \param[out] out_view,n_out_view `n_frames * n_atoms * 3`, Angstrom
    \throw IOException when the file cannot be read, is truncated, or a frame
           does not parse
*/
IMPBFFEXPORT void read_xtc(const std::string& path, int max_frames = -1,
                           double** out_view = NULL, int* n_out_view = NULL);

IMPBFF_END_NAMESPACE

#endif //IMPBFF_TRAJECTORYIO_H
