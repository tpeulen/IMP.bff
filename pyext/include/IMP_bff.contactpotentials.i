/*
 * Coordinate-array kernels for coarse-grained contact potentials: site
 * contact and distance-binned type-pair tables, backbone hydrogen bonds,
 * soft-sphere overlap, matrix 6-12 well, Generalized Born, gated site area.
 *
 * Every array comes in zero-copy (IN_ARRAY: C-contiguous; the caller hands
 * float64 / int32), because a Monte Carlo move calls these once per step and
 * a std::vector conversion would walk an n x n gate matrix in Python first.
 */
%apply(double* IN_ARRAY2, int DIM1, int DIM2) {
    (const double* in_xyz, int n_atoms, int n_xyz_dim),
    (const double* in_gate, int n_gate_rows, int n_gate_cols),
    (const double* in_table, int n_table_rows, int n_table_cols),
    (const double* in_hbond_table, int n_channels, int n_hbond_bins),
    (const double* in_distance, int n_dist_rows, int n_dist_cols),
    (const double* in_epsilon, int n_eps_rows, int n_eps_cols),
    (const double* in_r_min, int n_rm_rows, int n_rm_cols),
    (const double* in_points, int n_points, int n_point_dim)
};
%apply(double* IN_ARRAY1, int DIM1) {
    (const double* in_values, int n_values),
    (const double* in_radii, int n_radii),
    (const double* in_charges, int n_charges)
};
%apply(int* IN_ARRAY1, int DIM1) {
    (const int* in_site_atom, int n_sites),
    (const int* in_site_type, int n_site_types),
    (const int* in_n_atom, int n_n),
    (const int* in_c_atom, int n_c),
    (const int* in_o_atom, int n_o),
    (const int* in_h_atom, int n_h)
};
%include "IMP/bff/ContactPotentials.h"
