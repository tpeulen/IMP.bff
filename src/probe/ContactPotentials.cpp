/**
 * \file ContactPotentials.cpp
 * \brief Coordinate-array kernels for coarse-grained contact potentials.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/ContactPotentials.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/internal/ContactKernels.h>

#include <algorithm>
#include <cmath>

IMPBFF_BEGIN_NAMESPACE

// Named, not anonymous: the IMP build compiles every source of the module
// into one translation unit (bff_all.cpp), where anonymous helpers collide.
namespace contact_api {

void check_xyz(int n_atoms, int n_dim, const char* who) {
  if (n_atoms < 0 || (n_atoms > 0 && n_dim != 3)) {
    IMP_THROW(who << ": coordinates must be an (n, 3) array", ValueException);
  }
}

void check_square(int rows, int cols, int n, const char* what, const char* who) {
  if (rows != n || cols != n) {
    IMP_THROW(who << ": " << what << " must be (" << n << ", " << n << "), got ("
                  << rows << ", " << cols << ")",
              ValueException);
  }
}

//! Every index must address an atom; -1 (absent) is allowed.
void check_indices(const int* idx, int n, int n_atoms, const char* who) {
  for (int k = 0; k < n; ++k) {
    if (idx[k] >= n_atoms || idx[k] < -1) {
      IMP_THROW(who << ": atom index " << idx[k] << " out of range for "
                    << n_atoms << " atoms",
                ValueException);
    }
  }
}


}  // namespace contact_api

std::vector<double> get_site_contact_energy(
        const double* in_xyz, int n_atoms, int n_xyz_dim,
        const int* in_site_atom, int n_sites,
        const int* in_site_type, int n_site_types,
        const double* in_gate, int n_gate_rows, int n_gate_cols,
        double gate_cutoff,
        const double* in_table, int n_table_rows, int n_table_cols,
        double cutoff) {
  using namespace contact_api;
  const char* who = "get_site_contact_energy";
  check_xyz(n_atoms, n_xyz_dim, who);
  if (n_site_types != n_sites) {
    IMP_THROW(who << ": one type per site is required", ValueException);
  }
  check_square(n_gate_rows, n_gate_cols, n_sites, "the gate", who);
  check_indices(in_site_atom, n_sites, n_atoms, who);
  long n = 0;
  const double e = internal::contact::site_contact_sum(
          in_xyz, in_site_atom, in_site_type, n_sites,
          internal::contact::MatrixGate{in_gate, n_sites, gate_cutoff}, in_table, n_table_rows,
          n_table_cols, cutoff, &n);
  if (std::isnan(e)) {
    IMP_THROW(who << ": a site type is outside the table", ValueException);
  }
  return std::vector<double>{static_cast<double>(n), e};
}

std::vector<double> get_site_binned_pair_energy(
        const double* in_xyz, int n_atoms, int n_xyz_dim,
        const int* in_site_atom, int n_sites,
        const int* in_site_type, int n_site_types,
        const double* in_gate, int n_gate_rows, int n_gate_cols,
        double gate_cutoff,
        const double* in_values, int n_values,
        int n_types, int n_bins,
        double min_distance, double max_distance, double bin_width,
        double repulsion) {
  using namespace contact_api;
  const char* who = "get_site_binned_pair_energy";
  check_xyz(n_atoms, n_xyz_dim, who);
  if (n_site_types != n_sites) {
    IMP_THROW(who << ": one type per site is required", ValueException);
  }
  check_square(n_gate_rows, n_gate_cols, n_sites, "the gate", who);
  check_indices(in_site_atom, n_sites, n_atoms, who);
  if (n_types < 0 || n_bins < 0 ||
      static_cast<long>(n_types) * n_types * n_bins != n_values) {
    IMP_THROW(who << ": the table holds " << n_values << " values, not "
                  << n_types << " x " << n_types << " x " << n_bins,
              ValueException);
  }
  long n = 0;
  const double e = internal::contact::site_binned_sum(
          in_xyz, in_site_atom, in_site_type, n_sites,
          internal::contact::MatrixGate{in_gate, n_sites, gate_cutoff}, in_values, n_types,
          n_bins, min_distance, max_distance, bin_width, repulsion, &n);
  if (std::isnan(e)) {
    IMP_THROW(who << ": a site type or distance bin is past the table end",
              ValueException);
  }
  return std::vector<double>{static_cast<double>(n), e};
}

std::vector<double> get_backbone_hbond_energy(
        const double* in_xyz, int n_atoms, int n_xyz_dim,
        const int* in_n_atom, int n_n,
        const int* in_c_atom, int n_c,
        const int* in_o_atom, int n_o,
        const int* in_h_atom, int n_h,
        const double* in_gate, int n_gate_rows, int n_gate_cols,
        double gate_cutoff,
        const double* in_hbond_table, int n_channels, int n_hbond_bins,
        double cutoff_h2, double bin_width) {
  using namespace contact_api;
  const char* who = "get_backbone_hbond_energy";
  check_xyz(n_atoms, n_xyz_dim, who);
  const int n_res = n_n;
  if (n_c != n_res || n_o != n_res || n_h != n_res) {
    IMP_THROW(who << ": N, C, O and H need one index per residue", ValueException);
  }
  check_square(n_gate_rows, n_gate_cols, n_res, "the gate", who);
  if (n_channels != 4) {
    IMP_THROW(who << ": the table needs 4 channels, got " << n_channels,
              ValueException);
  }
  check_indices(in_n_atom, n_res, n_atoms, who);
  check_indices(in_c_atom, n_res, n_atoms, who);
  check_indices(in_o_atom, n_res, n_atoms, who);
  check_indices(in_h_atom, n_res, n_atoms, who);
  const bool channels[4] = {true, true, true, true};
  long n = 0;
  const double e = internal::contact::backbone_hbond_sum(
          in_xyz, in_n_atom, in_c_atom, in_o_atom, in_h_atom, n_res,
          internal::contact::MatrixGate{in_gate, n_res, gate_cutoff}, in_hbond_table,
          n_hbond_bins, cutoff_h2, bin_width, channels, &n);
  return std::vector<double>{static_cast<double>(n), e};
}

double get_soft_sphere_overlap_energy(
        const double* in_xyz, int n_atoms, int n_xyz_dim,
        const double* in_radii, int n_radii,
        double scale, double min_distance) {
  using namespace contact_api;
  const char* who = "get_soft_sphere_overlap_energy";
  check_xyz(n_atoms, n_xyz_dim, who);
  if (n_radii != n_atoms) {
    IMP_THROW(who << ": one radius per atom is required", ValueException);
  }
  if (scale == 0.0) {
    IMP_THROW(who << ": the scale divides the overlap and cannot be zero",
              ValueException);
  }
  return internal::contact::soft_sphere_overlap_sum(in_xyz, in_radii, n_atoms,
                                                    scale, min_distance);
}

double get_matrix_lj_well_energy(
        const double* in_distance, int n_dist_rows, int n_dist_cols,
        const double* in_epsilon, int n_eps_rows, int n_eps_cols,
        const double* in_r_min, int n_rm_rows, int n_rm_cols) {
  using namespace contact_api;
  const char* who = "get_matrix_lj_well_energy";
  const int n = n_dist_rows;
  check_square(n_dist_rows, n_dist_cols, n, "the distance matrix", who);
  check_square(n_eps_rows, n_eps_cols, n, "the depth matrix", who);
  check_square(n_rm_rows, n_rm_cols, n, "the minimum matrix", who);
  double e = 0.0;
  for (int i = 0; i < n; ++i) {
    const long row = static_cast<long>(i) * n;
    for (int j = i + 1; j < n; ++j) {
      e += internal::contact::lj_well(in_epsilon[row + j], in_r_min[row + j],
                                      in_distance[row + j]);
    }
  }
  return e;
}

double get_generalized_born_energy(
        const double* in_xyz, int n_atoms, int n_xyz_dim,
        const double* in_radii, int n_radii,
        const double* in_charges, int n_charges,
        double epsilon_in, double epsilon_out, double cutoff,
        double reference_distance2) {
  using namespace contact_api;
  const char* who = "get_generalized_born_energy";
  check_xyz(n_atoms, n_xyz_dim, who);
  if (n_radii != n_atoms || n_charges != n_atoms) {
    IMP_THROW(who << ": one radius and one charge per atom are required",
              ValueException);
  }
  return internal::contact::generalized_born_sum(
          in_xyz, in_radii, in_charges, n_atoms, epsilon_in, epsilon_out,
          cutoff, reference_distance2);
}

double get_site_accessible_area(
        const double* in_xyz, int n_atoms, int n_xyz_dim,
        const int* in_site_atom, int n_sites,
        const double* in_gate, int n_gate_rows, int n_gate_cols,
        double gate_cutoff,
        const double* in_points, int n_points, int n_point_dim,
        double probe, double radius) {
  using namespace contact_api;
  const char* who = "get_site_accessible_area";
  check_xyz(n_atoms, n_xyz_dim, who);
  if (n_points > 0 && n_point_dim != 3) {
    IMP_THROW(who << ": sphere points must be an (n, 3) array", ValueException);
  }
  check_square(n_gate_rows, n_gate_cols, n_sites, "the gate", who);
  check_indices(in_site_atom, n_sites, n_atoms, who);
  // The area's occluder test is strict (`gate < cut`), unlike the pair
  // terms' `gate <= cut`: the kernel this replaces read it that way.
  auto gate = [&](int i, int j) {
    return in_gate[static_cast<long>(i) * n_sites + j] < gate_cutoff;
  };
  return internal::contact::site_area_sum(in_xyz, in_site_atom, n_sites, gate,
                                          in_points, n_points, probe, radius);
}

IMPBFF_END_NAMESPACE
