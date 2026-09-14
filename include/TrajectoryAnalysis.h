/**
 *  \file IMP/bff/TrajectoryAnalysis.h
 *  \brief Axes, profiles and histograms read off the frames of a trajectory.
 *
 *  The array half of `imp_bff analyze-trajectories`: principal axes of a
 *  point cloud and the CSV tables the analysis writes. The IMP half -- RMF
 *  frames, particles, region atoms, density maps -- is
 *  #IMP::bff::analyze_probe_trajectories in ProbeTrajectoryDensity.h.
 *
 *  The arithmetic is numpy's where the Python program used numpy, so a table
 *  written here is the table the program wrote (internal/NumpyCompat.h says
 *  where the two would otherwise differ: column means, BLAS dots, and the
 *  eigenvector signs of LAPACK's `dsyevd`).
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_TRAJECTORYANALYSIS_H
#define IMPBFF_TRAJECTORYANALYSIS_H

#include <IMP/bff/bff_config.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The centre and a principal axis of a point cloud.
/*!
    The centre is the unweighted mean; the axis is the eigenvector of the
    covariance \f$C^T C / n\f$ with the smallest (or largest) eigenvalue,
    normalised, with the sign `numpy.linalg.eigh` gives it. Callers that need a
    stable sign across frames flip it against a reference themselves.

    \param[in] coords flat, three per point
    \param[in] smallest_variance the minimal-variance direction (a plane's
               normal) when true, the maximal one (a rod's long axis) when false
    \return six values: centre x, y, z, then axis x, y, z; empty when there are
            no points
*/
IMPBFFEXPORT std::vector<double> point_cloud_principal_axis(
        const std::vector<double>& coords, bool smallest_variance = true);

//! A binned profile per region, as CSV.
/*!
    Columns `bin_start_A,bin_end_A,center_A`, then `<region>_count` for every
    region, then `<region>_density` (count over total count times the bin
    width). The bins start at the smallest value over all regions, or at zero
    with \p from_zero -- a distance from an axis has an origin, a position along
    it does not. With no values at all, only the header is written.

    \param[in] path the CSV to write
    \param[in] regions the region names, in column order
    \param[in] values one list per region, same order
    \param[in] bin_width A
    \param[in] from_zero start the first bin at zero
    \throw IOException when \p path cannot be written
*/
IMPBFFEXPORT void write_binned_profile(const std::string& path,
                                       const std::vector<std::string>& regions,
                                       const std::vector<std::vector<double> >& values,
                                       double bin_width, bool from_zero = false);

//! A histogram of distances from zero, as CSV `bin_start_A,bin_end_A,count`.
/*!
    \param[in] distances A, non-negative
    \param[in] path the CSV to write
    \param[in] bin_width A
    \throw IOException when \p path cannot be written
*/
IMPBFFEXPORT void write_radial_histogram(const std::vector<double>& distances,
                                         const std::string& path, double bin_width);

//! The per-frame angle between two axes, as CSV
//! `frame_index,angle_deg,abs_cos_theta,cos_theta`.
/*! \throw IOException when \p path cannot be written */
IMPBFFEXPORT void write_orientation_profile(const std::string& path,
                                            const std::vector<int>& frame_index,
                                            const std::vector<double>& angle_deg,
                                            const std::vector<double>& abs_cos_theta,
                                            const std::vector<double>& cos_theta);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_TRAJECTORYANALYSIS_H
