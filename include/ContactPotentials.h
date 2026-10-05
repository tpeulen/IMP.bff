/**
 *  \file IMP/bff/ContactPotentials.h
 *  \brief Coordinate-array kernels for coarse-grained contact potentials.
 *
 * Sums over pairs of *sites* -- one representative atom per residue or bead,
 * picked out of a coordinate array by an index (-1 when the site has no such
 * atom) -- and over pairs of atoms found by a cell list. Every function takes
 * flat arrays and returns plain numbers, so a Monte Carlo move can call them
 * once per step without building particles.
 *
 * The tabulated terms are generic: a type-pair contact table (a
 * Miyazawa-Jernigan matrix is one), a type-pair distance-binned table (an
 * UNRES centroid potential is one), a four-channel backbone hydrogen-bond
 * table, a soft-sphere overlap, a matrix-parametrised truncated 6-12 well
 * (a Go model is one), a Generalized-Born pair sum, and a gated
 * Shrake-Rupley site area.
 *
 * **These are the array door to the same terms the IMP objects score.** The
 * IMP side -- #IMP::bff::MiyazawaJerniganPairScore,
 * #IMP::bff::UNRESCentroidPairScore, #IMP::bff::SoftSphereOverlapPairScore,
 * #IMP::bff::HydrogenBondRestraint, #IMP::bff::GoRestraint,
 * #IMP::bff::GeneralizedBornRestraint, #IMP::bff::SiteAccessibleAreaRestraint
 * in ProbePotentialRestraints.h -- and these functions call one kernel per
 * term (`internal/ContactKernels.h`); a restraint gathers its particles'
 * coordinates and passes a gate that measures them, an array caller passes a
 * gate matrix. A Monte Carlo move holding numpy calls these once per step
 * without building particles; a model calls the restraints.
 *
 * Several functions take a *gate*: an \f$(n \times n)\f$ matrix over sites and
 * a threshold; a site pair is skipped when `gate[i, j] > gate_cutoff`. It is
 * how a caller that already keeps a C-alpha distance matrix prunes pairs
 * before any coordinate is read.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_CONTACTPOTENTIALS_H
#define IMPBFF_CONTACTPOTENTIALS_H

#include <IMP/bff/bff_config.h>

#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Tabulated type-pair contact energy between sites closer than a cutoff.
/*!
    \f$E = \sum_{i<j} T[t_i, t_j]\f$ over site pairs passing the gate whose
    atoms are closer than \p cutoff (strictly).

    \return `[n_contacts, energy]`
*/
IMPBFFEXPORT std::vector<double> get_site_contact_energy(
        const double* in_xyz, int n_atoms, int n_xyz_dim,
        const int* in_site_atom, int n_sites,
        const int* in_site_type, int n_site_types,
        const double* in_gate, int n_gate_rows, int n_gate_cols,
        double gate_cutoff,
        const double* in_table, int n_table_rows, int n_table_cols,
        double cutoff);

//! Tabulated type-pair energy binned in the site-site distance.
/*!
    For each site pair passing the gate: below \p min_distance the pair adds
    \p repulsion; otherwise it adds `table[t_i, t_j, b]` with
    \f$b = \lfloor \min(d, d_{max}) / w \rfloor\f$. The table is passed flat,
    row-major \f$(n_{types}, n_{types}, n_{bins})\f$.

    \return `[n_pairs, energy]` -- every gated pair counts, close or not
    \throw ValueException when a distance bins past the end of the table
*/
IMPBFFEXPORT std::vector<double> get_site_binned_pair_energy(
        const double* in_xyz, int n_atoms, int n_xyz_dim,
        const int* in_site_atom, int n_sites,
        const int* in_site_type, int n_site_types,
        const double* in_gate, int n_gate_rows, int n_gate_cols,
        double gate_cutoff,
        const double* in_values, int n_values,
        int n_types, int n_bins,
        double min_distance, double max_distance, double bin_width,
        double repulsion);

//! Four-channel tabulated backbone hydrogen-bond energy.
/*!
    A residue donates through its H (with its N) and accepts through its O
    (with its C). For each gated residue pair both directions are tried; a
    direction d -> a counts when the donor's H index is positive, the
    acceptor's H index is non-zero, and \f$|H_d - O_a|^2 < \f$ \p cutoff_h2.
    Each counted bond adds `table[c, floor(r / bin_width)]` (skipped past the
    table end) for the channels c = 0: C_a-H_d, 1: N_d-O_a, 2: H_d-O_a,
    3: N_d-C_a. The table is row-major \f$(4, n_{bins})\f$. A direction whose
    residues lack the N, C or O atom (index < 0) is not a bond.

    \return `[n_bonds, energy]`
*/
IMPBFFEXPORT std::vector<double> get_backbone_hbond_energy(
        const double* in_xyz, int n_atoms, int n_xyz_dim,
        const int* in_n_atom, int n_n,
        const int* in_c_atom, int n_c,
        const int* in_o_atom, int n_o,
        const int* in_h_atom, int n_h,
        const double* in_gate, int n_gate_rows, int n_gate_cols,
        double gate_cutoff,
        const double* in_hbond_table, int n_channels, int n_hbond_bins,
        double cutoff_h2, double bin_width);

//! Soft-sphere overlap energy over all atom pairs (cell list).
/*!
    \f$E = \sum_{i<j} ((r_i + r_j - d_{ij}) / s)^2\f$ for
    \f$d_{min} < d_{ij} < r_i + r_j\f$; pairs at or below \p min_distance
    (bonded neighbours) are left out.
*/
IMPBFFEXPORT double get_soft_sphere_overlap_energy(
        const double* in_xyz, int n_atoms, int n_xyz_dim,
        const double* in_radii, int n_radii,
        double scale, double min_distance);

//! Truncated 6-12 well with per-pair depth and minimum (Go-type model).
/*!
    Over the upper triangle of the site distance matrix \p in_distance:
    for \f$r_m < r < 2.5\,r_m\f$ the pair adds
    \f$\epsilon(-s + s^2/4 + 0.00818)\f$ with \f$s = 2 (r_m/r)^6\f$, and
    \f$-\epsilon\f$ otherwise.
*/
IMPBFFEXPORT double get_matrix_lj_well_energy(
        const double* in_distance, int n_dist_rows, int n_dist_cols,
        const double* in_epsilon, int n_eps_rows, int n_eps_cols,
        const double* in_r_min, int n_rm_rows, int n_rm_cols);

//! Generalized-Born pair sum with a shifted reference term (cell list).
/*!
    \f$E = \frac{1}{8\pi}\left(\frac{1}{\epsilon_{out}} - \frac{1}{\epsilon_{in}}\right)
    \sum_{i \le j,\, d_{ij} \le r_c} q_i q_j \left(f(d_{ij}^2) - f(\rho)\right)\f$,
    \f$f(x) = 1/\sqrt{x + a_i a_j e^{-d_{ij}^2 / 4 a_i a_j}}\f$, self pairs
    included, uncharged atoms skipped.

    \p reference_distance2 is \f$\rho\f$, the squared distance the reference
    term is taken at. The cutoff squared is the consistent choice; the
    coarse-grained kernel this replaces passed the *unsquared* cutoff, and a
    caller reproducing it passes that.
*/
IMPBFFEXPORT double get_generalized_born_energy(
        const double* in_xyz, int n_atoms, int n_xyz_dim,
        const double* in_radii, int n_radii,
        const double* in_charges, int n_charges,
        double epsilon_in, double epsilon_out, double cutoff,
        double reference_distance2);

//! Shrake-Rupley accessible area of sites, with occluders chosen by a gate.
/*!
    Each present site is sampled at \p radius on the unit-sphere \p points
    (flat, three per point); a sample is buried when it lies within
    \f$radius + probe\f$ of the atom of another site \f$j\f$ with
    `gate[i, j] < gate_cutoff` (strict). Returns
    \f$\sum_i 4\pi\, radius^2\, n_i^{acc} / n_{points}\f$.

    Not solvent_accessible_surface_area(): there every atom occludes, here only
    the gated sites do, which is the residue-level burial term a
    coarse-grained model scores.
*/
IMPBFFEXPORT double get_site_accessible_area(
        const double* in_xyz, int n_atoms, int n_xyz_dim,
        const int* in_site_atom, int n_sites,
        const double* in_gate, int n_gate_rows, int n_gate_cols,
        double gate_cutoff,
        const double* in_points, int n_points, int n_point_dim,
        double probe, double radius);

IMPBFF_END_NAMESPACE

#endif // IMPBFF_CONTACTPOTENTIALS_H
