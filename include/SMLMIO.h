/** \file IMP/bff/SMLMIO.h
 *  \brief Native, strict 3D localization CSV loading and spatial selection.
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SMLMIO_H
#define IMPBFF_SMLMIO_H

#include <IMP/bff/SMLM.h>
#include <string>

IMPBFF_BEGIN_NAMESPACE

//! Explicit CSV opt-ins; defaults require measured 3D and retain every row.
struct IMPBFFEXPORT SMLMCSVOptions {
  bool allow_missing_z = false;
  //! Source-unit z used only if allow_missing_z and the z column is absent.
  double default_z = 0.0;
  bool allow_missing_sigmas = false;
  //! Three positive source-unit sigmas used ONLY for absent precision columns.
  std::vector<double> default_sigmas;
  //! Empty means unit weights, even when photon/intensity columns are present.
  std::string weight_column;
  //! Empty disables selection; otherwise names one numeric CSV field.
  std::string selection_column;
  //! Two finite inclusive source-field bounds [min, max], required for selection.
  //! Must be empty when selection_column is empty.
  std::vector<double> selection_range;
  //! SMAP/LocMoFit uses 0 for unassigned. Set false for zero-based source IDs.
  bool zero_particle_id_is_unassigned = true;
  int leaf_size = 16;
  IMP_SHOWABLE_INLINE(SMLMCSVOptions,
      out << "SMLMCSVOptions(strict measured 3D by default)");
};

//! Read one CSV into an owning native index; no iterative Python marshalling.
/*! Coordinate and sigma units are chosen explicitly by the caller: output
    coordinates = coordinate_scale * (source coordinates - origin), output
    sigmas = coordinate_scale * source sigmas. origin has three entries in
    SOURCE coordinate units; an omitted/empty origin means zero. Scale must be
    finite and positive. No pixel/nm
    inference or automatic unit conversion is performed.

    Header matching is case-insensitive with non-alphanumeric characters
    removed (e.g. x_nm and quoted x [nm] both match xnm). Per-axis positional
    uncertainty wins over scalar xy uncertainty. PSF widths and photon errors
    are not positional sigmas. Positive sitenumbers/particle IDs are preserved;
    negative source IDs become -1. Zero becomes -1 by default (SMAP); set
    zero_particle_id_is_unassigned=false for zero-based groups. Missing IDs
    become -1. Multiple aliases of the same field are rejected. Malformed
    records, nonfinite numbers and nonpositive sigmas always raise ValueException
    with source record context. Every row is validated, including rows outside
    an explicitly requested selection. By default all rows are retained;
    selection_column opts into retaining only rows whose numeric field lies
    within selection_range = [min, max], inclusively. The column uses the same
    canonical header matching, must exist, and is read as a finite number on
    every row. Bounds must be finite with min <= max, and refer to the source
    field without coordinate scaling or origin subtraction. A range without a
    column is rejected. No protein/channel interpretation or implicit filtering
    is performed. Selected rows keep their source order, coordinates, sigmas,
    weights and IDs (with the explicit unit/ID conventions above); no matches
    produces an empty owning index. Rows are never silently dropped. */
IMPBFFEXPORT SMLMIndex read_smlm_csv(
    const std::string& path, double coordinate_scale = 1.0,
    const std::vector<double>& origin = std::vector<double>(),
    const SMLMCSVOptions& options = SMLMCSVOptions());

//! Copy rows inside inclusive lower/upper bounds into a new owning native index.
/*! Bounds use the index coordinate unit. Coordinates, sigmas, weights and IDs
    are preserved. Negative new_particle_id leaves source IDs untouched; a
    nonnegative value explicitly assigns every selected row to that ID.
    Selection is a single native O(N) pass, outside scoring/optimizer loops. */
IMPBFFEXPORT SMLMIndex select_smlm_region(
    const SMLMIndex& index, const std::vector<double>& lower,
    const std::vector<double>& upper, int new_particle_id = -1);

//! Select saved source particle IDs without exposing the full table to Python.
/*! IDs must be nonempty, unique, nonnegative and present in the source index.
    Copies selected measured coordinates, sigmas, weights and IDs in source row
    order. A single native selection pass; no alignment or quality filtering. */
IMPBFFEXPORT SMLMIndex select_smlm_particles(
    const SMLMIndex& index, const std::vector<int>& particle_ids);

IMPBFF_END_NAMESPACE
#endif  // IMPBFF_SMLMIO_H
