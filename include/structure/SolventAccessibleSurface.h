/**
 *  \file IMP/bff/structure/SolventAccessibleSurface.h
 *  \brief Shrake-Rupley solvent-accessible surface of selected atoms.
 *
 * Not `IMP::saxs::SolventAccessibleSurface`, which answers a different
 * question: it returns each atom's accessibility as a *fraction* of its
 * isolated surface, for a form factor. These return **area in square
 * Angstrom**, per atom or summed, which is what a quenching or burial term
 * integrates -- and they take flat arrays, so a caller with coordinates from
 * anywhere does not have to build particles first.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_SOLVENTACCESSIBLESURFACE_H
#define IMPBFF_SOLVENTACCESSIBLESURFACE_H

#include <IMP/bff/bff_config.h>

#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! \p n roughly equidistant points on the unit sphere, by the golden spiral.
/*!
    Returned flat, three doubles per point.

    Not `IMP::algebra::get_uniform_surface_cover`, which covers a sphere by a
    different construction: every sampled quantity in this module -- the
    surface areas below, the coarse-grained potentials' residue ASA -- is
    pinned against *this* point set, so the two are not interchangeable at the
    precision those pins assert.

    \param[in] n number of points
*/
IMPBFFEXPORT std::vector<double> sphere_points(int n);

//! Solvent-accessible surface area of selected atoms (Shrake-Rupley).
/*!
    Samples each selected atom's surface at \p radius from its centre and counts
    the samples no other atom occludes within \f$radius + probe\f$.

    **The neighbour cutoff is \f$(2\,radius + probe)^2\f$**, which is the tight
    criterion for this kernel: a sample sits \p radius from its own centre and
    is occluded by a neighbour within \f$radius + probe\f$ of it. The kernel
    this was ported from tested a *squared* distance against an *unsquared* sum,
    and against the van der Waals radius rather than the sampling radius --
    a 2.45 A cutoff where 6.0 A was meant, so nearly every occluding neighbour
    was missed and the area came back far too large. Fixed in PRD-109; nothing
    downstream moved, because nothing consumed the result.

    \param[in] xyz atom coordinates, flat, three per atom
    \param[in] vdw per-atom van der Waals radii; the area is scaled by its square
    \param[in] probe_atom_indices which atoms to compute the area for
    \param[in] points unit-sphere samples, flat, from sphere_points()
    \param[in] probe probe-sphere radius in Angstrom
    \param[in] radius radius at which the samples are placed
    \return one area per entry of \p probe_atom_indices
*/
IMPBFFEXPORT std::vector<double> solvent_accessible_surface_area(
        const std::vector<double>& xyz,
        const std::vector<double>& vdw,
        const std::vector<int>& probe_atom_indices,
        const std::vector<double>& points,
        double probe = 1.4,
        double radius = 2.5
);

//! Solvent-accessible surface area with a per-atom radius (Shrake-Rupley).
/*!
    The textbook form, and the one a *relative* accessibility needs: atom \p i
    is sampled on a sphere of radius \f$r_i + probe\f$ and a sample is occluded
    when it falls within \f$r_j + probe\f$ of another atom's centre, so each
    atom is occluded by its neighbour's real size. The area returned is the
    accessible surface \f$4\pi (r_i + probe)^2 f_i\f$.

    solvent_accessible_surface_area() above differs deliberately and is not a
    worse version of this: it samples every atom at one radius and scales the
    count by \f$r_i^2\f$, which is what the contact and quenching estimates
    want (a probe of fixed size rolling over a uniform obstacle set) and is
    cheaper. The two answer different questions; a caller comparing an area
    against a tabulated maximum wants this one.

    \param[in] xyz atom coordinates, flat, three per atom
    \param[in] vdw per-atom van der Waals radii
    \param[in] probe_atom_indices which atoms to compute the area for
    \param[in] points unit-sphere samples, flat, from sphere_points()
    \param[in] probe probe-sphere radius in Angstrom
    \return one accessible area per entry of \p probe_atom_indices
*/
IMPBFFEXPORT std::vector<double> solvent_accessible_surface_area_per_atom(
        const std::vector<double>& xyz,
        const std::vector<double>& vdw,
        const std::vector<int>& probe_atom_indices,
        const std::vector<double>& points,
        double probe = 1.4
);

//! Total quenching rate at each frame of a trajectory.
/*!
    Sums the rate constants of the quenchers a walker is in contact with, frame
    by frame. Rates add because the channels are parallel.

    \param[in] collided flat row-major \f$(n\_frames \times n\_atoms)\f$ contact flags
    \param[in] n_frames rows
    \param[in] k_quench per-atom rate constant, length \f$n\_atoms\f$
*/
IMPBFFEXPORT std::vector<double> quenching_rate_per_frame(
        const std::vector<int>& collided,
        int n_frames,
        const std::vector<double>& k_quench
);

//! \p n points on the unit sphere, by the Fibonacci lattice.
/*!
    Point \f$k\f$ sits at polar angle \f$\arccos(1 - 2(k + \tfrac12)/n)\f$ and
    azimuth \f$\pi(1 + \sqrt5)(k + \tfrac12)\f$, so the points are
    equal-area and none sits on a pole. Returned flat, three doubles per
    point, as (x, y, z) with z along the polar axis.

    A second golden spiral beside sphere_points(), and deliberately so: this
    is the point set probe_solvent_exposure() samples, and the quenching
    analysis it was ported from (Shrake-Rupley water exposure of a dye at
    every accessible-volume voxel) pinned its exposures against this one. The
    two sets differ by a rotation, so a covered/uncovered count is not the
    same for both.

    \param[in] n number of points
*/
IMPBFFEXPORT std::vector<double> fibonacci_sphere_points(int n);

//! Solvent exposure of a spherical probe at each of many centres.
/*!
    The Shrake-Rupley test for a probe that is *not* one of the atoms: the
    probe (a dye, radius \p probe_radius) sits at a centre, a water molecule
    (radius \p water_radius) touches it where the water's centre lies on the
    sphere of radius \f$probe\_radius + water\_radius\f$ around that centre,
    and a sample point there is **covered** when it is closer than
    \f$r_j + water\_radius\f$ to some atom \f$j\f$ (strict). The exposure is
    the uncovered fraction of the \p n_points samples of
    fibonacci_sphere_points(): 1 in open water, 0 buried.

    The probe itself, its linker and any other probe do not cover anything;
    only the listed atoms do. Neighbours come from a cell list, so the cost is
    linear in the number of centres.

    \param[in] centres flat, three per centre, Angstrom
    \param[in] atom_xyz flat, three per atom, Angstrom
    \param[in] atom_radii one van der Waals radius per atom, Angstrom
    \param[out] out_view,n_out_view one exposure in [0, 1] per centre
    \param[in] probe_radius the probe sphere, Angstrom
    \param[in] water_radius the solvent sphere, Angstrom
    \param[in] n_points samples on the contact sphere
    \throw ValueException when the radii do not match the atoms
*/
IMPBFFEXPORT void probe_solvent_exposure(
        const std::vector<double>& centres,
        const std::vector<double>& atom_xyz,
        const std::vector<double>& atom_radii,
        double** out_view, int* n_out_view,
        double probe_radius = 3.5,
        double water_radius = 1.4,
        int n_points = 64
);

//! The default number of unit-sphere samples per atom.
extern IMPBFFEXPORT const int DEFAULT_SPHERE_POINTS;

//! Accessible surface area of the selected atoms, with the default sampling.
/*!
    solvent_accessible_surface_area() with sphere_points(DEFAULT_SPHERE_POINTS)
    when \p points is empty -- which is how every caller wants it.
*/
IMPBFFEXPORT std::vector<double> solvent_accessible_surface(
        const std::vector<double>& xyz,
        const std::vector<double>& vdw,
        const std::vector<int>& probe_atom_indices,
        const std::vector<double>& points = std::vector<double>(),
        double probe = 1.0,
        double radius = 2.5
);

IMPBFF_END_NAMESPACE

#endif //IMPBFF_SOLVENTACCESSIBLESURFACE_H
