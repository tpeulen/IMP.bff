/**
 *  \file IMP/bff/internal/ContactKernels.h
 *  \brief The one implementation of the coarse-grained contact terms.
 *
 * Every coarse-grained contact term is computed here and nowhere else. The
 * array functions in ContactPotentials.h and the IMP scores and restraints in
 * ProbePotentialRestraints.h both call these: an IMP restraint gathers its
 * particles' coordinates into a flat array and passes a gate that reads them,
 * an array caller passes its own gate matrix. A term that differs between the
 * two paths therefore can only differ in its gate, never in its arithmetic.
 *
 * A *gate* is a callable `bool(int i, int j)` over site indices: true when the
 * pair is examined at all.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_INTERNAL_CONTACTKERNELS_H
#define IMPBFF_INTERNAL_CONTACTKERNELS_H

#include <IMP/bff/bff_config.h>

#include <algorithm>
#include <cmath>
#include <vector>

IMPBFF_BEGIN_INTERNAL_NAMESPACE

namespace contact {

inline double dist2(const double* xyz, int a, int b) {
  const double dx = xyz[3 * a] - xyz[3 * b];
  const double dy = xyz[3 * a + 1] - xyz[3 * b + 1];
  const double dz = xyz[3 * a + 2] - xyz[3 * b + 2];
  return dx * dx + dy * dy + dz * dz;
}

//! Visit every unordered point pair closer than \p cutoff, by a cell list.
/*! Cells have edge \p cutoff, so only the 27 surrounding cells can hold a
    partner. Each pair is visited once as \p f(i, j, d2) with i < j (and, with
    \p include_self, once as f(i, i, 0)). All pairs are tested directly when
    the box is too small, or too sparse, for the cells to prune anything. */
template <class F>
void for_close_pairs(const double* xyz, int n, double cutoff,
                     bool include_self, F&& f) {
  const double c2 = cutoff * cutoff;
  if (include_self) {
    for (int i = 0; i < n; ++i) f(i, i, 0.0);
  }
  if (n < 2) return;
  double lo[3] = {xyz[0], xyz[1], xyz[2]};
  double hi[3] = {xyz[0], xyz[1], xyz[2]};
  for (int i = 1; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], xyz[3 * i + k]);
      hi[k] = std::max(hi[k], xyz[3 * i + k]);
    }
  }
  const bool finite = std::isfinite(cutoff) && cutoff > 0.0;
  long dims[3] = {1, 1, 1};
  long n_cells = 1;
  if (finite) {
    for (int k = 0; k < 3; ++k) {
      dims[k] = static_cast<long>(std::floor((hi[k] - lo[k]) / cutoff)) + 1;
      n_cells *= dims[k];
    }
  }
  if (!finite || n_cells <= 27 || n_cells > 64L * n) {
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const double d2 = dist2(xyz, i, j);
        if (d2 <= c2) f(i, j, d2);
      }
    }
    return;
  }
  std::vector<int> cell_of(n);
  std::vector<int> start(n_cells + 1, 0);
  for (int i = 0; i < n; ++i) {
    long c[3];
    for (int k = 0; k < 3; ++k) {
      c[k] = static_cast<long>(std::floor((xyz[3 * i + k] - lo[k]) / cutoff));
      c[k] = std::min(std::max(c[k], 0L), dims[k] - 1);
    }
    cell_of[i] = static_cast<int>((c[0] * dims[1] + c[1]) * dims[2] + c[2]);
    ++start[cell_of[i] + 1];
  }
  for (long c = 0; c < n_cells; ++c) start[c + 1] += start[c];
  std::vector<int> order(n);
  {
    std::vector<int> fill(start.begin(), start.end() - 1);
    for (int i = 0; i < n; ++i) order[fill[cell_of[i]]++] = i;
  }
  for (int i = 0; i < n; ++i) {
    const long ci = cell_of[i];
    const long cx = ci / (dims[1] * dims[2]);
    const long cy = (ci / dims[2]) % dims[1];
    const long cz = ci % dims[2];
    for (long ax = std::max(cx - 1, 0L); ax <= std::min(cx + 1, dims[0] - 1); ++ax) {
      for (long ay = std::max(cy - 1, 0L); ay <= std::min(cy + 1, dims[1] - 1); ++ay) {
        for (long az = std::max(cz - 1, 0L); az <= std::min(cz + 1, dims[2] - 1); ++az) {
          const long cj = (ax * dims[1] + ay) * dims[2] + az;
          for (int p = start[cj]; p < start[cj + 1]; ++p) {
            const int j = order[p];
            if (j <= i) continue;
            const double d2 = dist2(xyz, i, j);
            if (d2 <= c2) f(i, j, d2);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

//! A gate read from a caller's row-major \f$n \times n\f$ matrix: a pair
//! passes when `g[i, j] <= cut` (NaN never passes).
struct MatrixGate {
  const double* g;
  int n;
  double cut;
  bool operator()(int i, int j) const {
    return g[static_cast<long>(i) * n + j] <= cut;
  }
};

//! Call \p f(i, j) for every pair i < j < n the gate passes.
template <class Gate, class F>
void for_gated_upper_pairs(int n, const Gate& gate, F&& f) {
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (gate(i, j)) f(i, j);
    }
  }
}

//! The same for a matrix gate, by a branchless scan of each row.
/*! A coarse-grained gate passes a few percent of the pairs, so the scan is
    the cost; compacting a row's passing columns first keeps it free of
    mispredicted branches, which is what a scalar test-and-continue pays. */
template <class F>
void for_gated_upper_pairs(int n, const MatrixGate& gate, F&& f) {
  std::vector<int> cand(n > 0 ? n : 1);
  for (int i = 0; i < n; ++i) {
    const double* row = gate.g + static_cast<long>(i) * gate.n;
    int k = 0;
    for (int j = i + 1; j < n; ++j) {
      cand[k] = j;
      k += row[j] <= gate.cut;
    }
    for (int t = 0; t < k; ++t) f(i, cand[t]);
  }
}

// ---------------------------------------------------------------------------
// Pair terms
// ---------------------------------------------------------------------------

//! Soft-sphere overlap of one pair: \f$((\sigma - r)/s)^2\f$ on
//! \f$d_{min} < r < \sigma\f$, else 0; \p dedr receives \f$dE/dr\f$.
inline double soft_sphere_overlap(double r, double sigma, double scale,
                                  double min_distance, double* dedr) {
  if (min_distance < r && r < sigma) {
    const double o = (sigma - r) / scale;
    if (dedr) *dedr = -2.0 * o / scale;
    return o * o;
  }
  if (dedr) *dedr = 0.0;
  return 0.0;
}

//! Truncated (6,12) well of depth \p eps and minimum \p rm at distance \p r:
//! \f$\epsilon(-s + s^2/4 + 0.00818)\f$, \f$s = 2(r_m/r)^6\f$, on
//! \f$r_m < r < 2.5 r_m\f$, and \f$-\epsilon\f$ otherwise.
/*! The 0.00818 shift puts the truncated well at zero at 2.5 rm. */
inline double lj_well(double eps, double rm, double r) {
  if (r > rm && r < rm * 2.5 && r > 0.0) {
    const double q = rm / r;
    const double q3 = q * q * q;
    const double sr = 2.0 * q3 * q3;
    return eps * (-sr + sr * sr / 4.0 + 0.00818);
  }
  return -eps;
}

// ---------------------------------------------------------------------------
// Sums
// ---------------------------------------------------------------------------

//! Soft-sphere overlap over every atom pair, by a cell list.
inline double soft_sphere_overlap_sum(const double* xyz, const double* radii,
                                      int n, double scale, double min_distance) {
  if (n < 2) return 0.0;
  const double reach = 2.0 * *std::max_element(radii, radii + n);
  double e = 0.0;
  for_close_pairs(xyz, n, reach, false, [&](int i, int j, double d2) {
    e += soft_sphere_overlap(std::sqrt(d2), radii[i] + radii[j], scale,
                             min_distance, nullptr);
  });
  return e;
}

//! Generalized-Born pair sum, self pairs included, by a cell list.
/*! Atoms without charge or radius take no part. \p reference_distance2 is the
    squared distance the reference term is taken at (see
    get_generalized_born_energy()). */
inline double generalized_born_sum(const double* xyz, const double* radii,
                                   const double* charges, int n,
                                   double epsilon_in, double epsilon_out,
                                   double cutoff, double reference_distance2) {
  std::vector<int> keep;
  keep.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (charges[i] != 0.0 && radii[i] != 0.0) keep.push_back(i);
  }
  const int m = static_cast<int>(keep.size());
  if (m == 0) return 0.0;
  std::vector<double> x(3L * m), a(m), q(m);
  for (int k = 0; k < m; ++k) {
    const int i = keep[k];
    x[3 * k] = xyz[3 * i];
    x[3 * k + 1] = xyz[3 * i + 1];
    x[3 * k + 2] = xyz[3 * i + 2];
    a[k] = radii[i];
    q[k] = charges[i];
  }
  const double pre = 1.0 / (8.0 * M_PI) * (1.0 / epsilon_out - 1.0 / epsilon_in);
  double e = 0.0;
  for_close_pairs(x.data(), m, cutoff, true, [&](int i, int j, double rij2) {
    const double aij2 = a[i] * a[j];
    const double qij = q[i] * q[j];
    const double ex = std::exp(-rij2 / (4.0 * aij2));
    e += qij / std::sqrt(rij2 + aij2 * ex) -
         qij / std::sqrt(reference_distance2 + aij2 * ex);
  });
  return e * pre;
}

//! Type-pair contact sum over gated site pairs: `table[t_i * n_cols + t_j]`
//! for every pair of present sites closer than \p cutoff (strict).
/*! \return the energy; \p n_contacts receives the count. A type outside the
    table returns NaN energy, which the callers turn into an error. */
template <class Gate>
double site_contact_sum(const double* xyz, const int* site_atom,
                        const int* site_type, int n_sites, const Gate& gate,
                        const double* table, int n_rows, int n_cols,
                        double cutoff, long* n_contacts) {
  const double cut2 = cutoff * cutoff;
  double e = 0.0;
  long n = 0;
  bool bad = false;
  for_gated_upper_pairs(n_sites, gate, [&](int i, int j) {
    const int ai = site_atom[i], aj = site_atom[j];
    if (ai < 0 || aj < 0) return;
    if (dist2(xyz, ai, aj) < cut2) {
      const int ti = site_type[i], tj = site_type[j];
      if (ti < 0 || tj < 0 || ti >= n_rows || tj >= n_cols) {
        bad = true;
        return;
      }
      e += table[static_cast<long>(ti) * n_cols + tj];
      ++n;
    }
  });
  *n_contacts = n;
  return bad ? std::nan("") : e;
}

//! Type-pair distance-binned sum over gated site pairs.
/*! Below \p min_distance a pair adds \p repulsion; otherwise
    `values[(t_i * n_types + t_j) * n_bins + floor(min(d, max_distance) / w)]`.
    \p n_pairs receives the number of gated pairs of present sites. NaN when a
    type or bin falls outside the table. */
template <class Gate>
double site_binned_sum(const double* xyz, const int* site_atom,
                       const int* site_type, int n_sites, const Gate& gate,
                       const double* values, int n_types, int n_bins,
                       double min_distance, double max_distance,
                       double bin_width, double repulsion, long* n_pairs) {
  double e = 0.0;
  long n_close = 0, n = 0;
  bool bad = false;
  for_gated_upper_pairs(n_sites, gate, [&](int i, int j) {
    const int ai = site_atom[i], aj = site_atom[j];
    if (ai < 0 || aj < 0) return;
    ++n;
    const double d = std::sqrt(dist2(xyz, ai, aj));
    if (d < min_distance) {
      ++n_close;
      return;
    }
    const int ti = site_type[i], tj = site_type[j];
    const long b = static_cast<long>(std::min(d, max_distance) / bin_width);
    if (ti < 0 || tj < 0 || ti >= n_types || tj >= n_types || b < 0 ||
        b >= n_bins) {
      bad = true;
      return;
    }
    e += values[(static_cast<long>(ti) * n_types + tj) * n_bins + b];
  });
  *n_pairs = n;
  return bad ? std::nan("") : e + repulsion * static_cast<double>(n_close);
}

//! Four-channel backbone hydrogen-bond sum over gated residue pairs.
/*! Residue d donates to residue a when `h[d] > 0`, `h[a] != 0`, the donor's N
    and the acceptor's C and O exist (index >= 0) and \f$|H_d - O_a|^2 <\f$
    \p cutoff_h2. (An H index of 0 is neither present nor missing -- the
    convention of the residue lookups this serves, where atom 0 is always a
    backbone N.) Each bond adds `table[c * n_bins + floor(r / w)]`, skipped
    past the end, for the channels switched on in \p channels:
    0 C_a-H_d, 1 N_d-O_a, 2 H_d-O_a, 3 N_d-C_a. \p n_bonds receives the count. */
template <class Gate>
double backbone_hbond_sum(const double* xyz, const int* n_atom,
                          const int* c_atom, const int* o_atom,
                          const int* h_atom, int n_res, const Gate& gate,
                          const double* table, int n_bins, double cutoff_h2,
                          double bin_width, const bool channels[4],
                          long* n_bonds) {
  // Per-channel sums in registers; the channel switches are applied once at
  // the end, so the hot loop carries no flag tests and no division.
  double e0 = 0.0, e1 = 0.0, e2 = 0.0, e3 = 0.0;
  long nb = 0;
  const double inv_w = 1.0 / bin_width;
  const double* t0 = table;
  const double* t1 = table + n_bins;
  const double* t2 = table + 2L * n_bins;
  const double* t3 = table + 3L * n_bins;
  auto bin = [&](const double* t, double r) -> double {
    const long b = static_cast<long>(r * inv_w);
    return (b >= 0 && b < n_bins) ? t[b] : 0.0;
  };
  auto bond = [&](int d, int a) {
    const int h = h_atom[d], o = o_atom[a];
    if (o < 0) return;
    const double doh = dist2(xyz, h, o);
    if (!(doh < cutoff_h2)) return;
    const int n = n_atom[d], c = c_atom[a];
    if (n < 0 || c < 0) return;
    ++nb;
    e2 += bin(t2, std::sqrt(doh));
    e1 += bin(t1, std::sqrt(dist2(xyz, n, o)));
    e0 += bin(t0, std::sqrt(dist2(xyz, c, h)));
    e3 += bin(t3, std::sqrt(dist2(xyz, n, c)));
  };
  for_gated_upper_pairs(n_res, gate, [&](int i, int j) {
    const int hi = h_atom[i], hj = h_atom[j];
    if (hj > 0 && hi != 0) bond(j, i);
    if (hi > 0 && hj != 0) bond(i, j);
  });
  double e = 0.0;
  if (channels[2]) e += e2;
  if (channels[1]) e += e1;
  if (channels[0]) e += e0;
  if (channels[3]) e += e3;
  *n_bonds = nb;
  return e;
}

//! Shrake-Rupley area of present sites; site j occludes site i when
//! `gate(i, j)`. \f$\sum_i 4\pi r^2 n_i^{acc} / n_{points}\f$.
template <class Gate>
double site_area_sum(const double* xyz, const int* site_atom, int n_sites,
                     const Gate& gate, const double* points, int n_points,
                     double probe, double radius) {
  if (n_points == 0) return 0.0;
  const double contact2 = (radius + probe) * (radius + probe);
  const double c = 4.0 * M_PI / static_cast<double>(n_points);
  double area = 0.0;
  std::vector<int> occluders;
  occluders.reserve(n_sites);
  for (int i = 0; i < n_sites; ++i) {
    const int ai = site_atom[i];
    if (ai < 0) continue;
    occluders.clear();
    for (int j = 0; j < n_sites; ++j) {
      if (j == i || site_atom[j] < 0 || !gate(i, j)) continue;
      occluders.push_back(site_atom[j]);
    }
    const double cx = xyz[3 * ai], cy = xyz[3 * ai + 1], cz = xyz[3 * ai + 2];
    long n_accessible = 0;
    for (int p = 0; p < n_points; ++p) {
      const double px = points[3 * p] * radius + cx;
      const double py = points[3 * p + 1] * radius + cy;
      const double pz = points[3 * p + 2] * radius + cz;
      bool accessible = true;
      for (int b : occluders) {
        const double dx = px - xyz[3 * b];
        const double dy = py - xyz[3 * b + 1];
        const double dz = pz - xyz[3 * b + 2];
        if (dx * dx + dy * dy + dz * dz < contact2) {
          accessible = false;
          break;
        }
      }
      if (accessible) ++n_accessible;
    }
    area += c * static_cast<double>(n_accessible) * radius * radius;
  }
  return area;
}

}  // namespace contact

IMPBFF_END_INTERNAL_NAMESPACE

#endif  // IMPBFF_INTERNAL_CONTACTKERNELS_H
