/**
 * \file SolventAccessibleSurface.cpp
 * \brief Shrake-Rupley solvent-accessible surface, and per-frame quenching rates.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/structure/SolventAccessibleSurface.h>
#include <IMP/bff/internal/OutputView.h>
#include <IMP/bff/util/IMPCompatibility.h>

#include <algorithm>
#include <cmath>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

std::vector<double> sphere_points(int n) {
    std::vector<double> out;
    if (n <= 0) return out;
    out.resize(static_cast<std::size_t>(n) * 3);
    const double inc = M_PI * (3.0 - std::sqrt(5.0));
    const double offset = 2.0 / static_cast<double>(n);
    for (int k = 0; k < n; ++k) {
        const double y = k * offset - 1.0 + offset / 2.0;
        const double rd = std::sqrt(std::max(0.0, 1.0 - y * y));
        const double phi = k * inc;
        out[3 * k + 0] = std::cos(phi) * rd;
        out[3 * k + 1] = y;
        out[3 * k + 2] = std::sin(phi) * rd;
    }
    return out;
}

std::vector<double> solvent_accessible_surface_area(
        const std::vector<double>& xyz, const std::vector<double>& vdw,
        const std::vector<int>& probe_atom_indices,
        const std::vector<double>& points, double probe, double radius) {
    const std::size_t n_probe = probe_atom_indices.size();
    const std::size_t n_sphere = points.size() / 3;
    const std::size_t n_atoms = xyz.size() / 3;
    std::vector<double> asa(n_probe, 0.0);
    if (n_probe == 0 || n_sphere == 0 || n_atoms == 0) return asa;

    const double c = 4.0 * M_PI / static_cast<double>(n_sphere);
    const double neighbour_cutoff2 = (2.0 * radius + probe) * (2.0 * radius + probe);
    const double occlusion2 = (radius + probe) * (radius + probe);
    std::vector<std::size_t> neighbours(n_atoms);

    for (std::size_t i = 0; i < n_probe; ++i) {
        const std::size_t p = static_cast<std::size_t>(probe_atom_indices[i]);
        const double ax = xyz[3 * p + 0], ay = xyz[3 * p + 1], az = xyz[3 * p + 2];
        std::size_t n_neighbour = 0;
        for (std::size_t a = 0; a < n_atoms; ++a) {
            if (a == p) continue;
            const double dx = ax - xyz[3 * a + 0];
            const double dy = ay - xyz[3 * a + 1];
            const double dz = az - xyz[3 * a + 2];
            if (dx * dx + dy * dy + dz * dz < neighbour_cutoff2) {
                neighbours[n_neighbour++] = a;
            }
        }
        std::size_t n_accessible = 0;
        for (std::size_t j = 0; j < n_sphere; ++j) {
            const double sx = points[3 * j + 0] * radius + ax;
            const double sy = points[3 * j + 1] * radius + ay;
            const double sz = points[3 * j + 2] * radius + az;
            bool accessible = true;
            for (std::size_t k = 0; k < n_neighbour; ++k) {
                const std::size_t b = neighbours[k];
                const double dx = sx - xyz[3 * b + 0];
                const double dy = sy - xyz[3 * b + 1];
                const double dz = sz - xyz[3 * b + 2];
                if (dx * dx + dy * dy + dz * dz < occlusion2) {
                    accessible = false;
                    break;
                }
            }
            if (accessible) ++n_accessible;
        }
        asa[i] = c * static_cast<double>(n_accessible) * vdw[p] * vdw[p];
    }
    return asa;
}

std::vector<double> solvent_accessible_surface_area_per_atom(
        const std::vector<double>& xyz, const std::vector<double>& vdw,
        const std::vector<int>& probe_atom_indices,
        const std::vector<double>& points, double probe) {
    const std::size_t n_probe = probe_atom_indices.size();
    const std::size_t n_sphere = points.size() / 3;
    const std::size_t n_atoms = xyz.size() / 3;
    std::vector<double> asa(n_probe, 0.0);
    if (n_probe == 0 || n_sphere == 0 || n_atoms == 0) return asa;

    // The largest sphere any atom is sampled on, so one cutoff serves all.
    double r_max = 0.0;
    for (std::size_t a = 0; a < n_atoms; ++a) r_max = std::max(r_max, vdw[a]);

    const double c = 4.0 * M_PI / static_cast<double>(n_sphere);
    std::vector<std::size_t> neighbours(n_atoms);

    for (std::size_t i = 0; i < n_probe; ++i) {
        const std::size_t p = static_cast<std::size_t>(probe_atom_indices[i]);
        const double rp = vdw[p] + probe;
        const double ax = xyz[3 * p + 0], ay = xyz[3 * p + 1], az = xyz[3 * p + 2];
        // A sample sits rp from p and is occluded within (r_b + probe) of b,
        // so only atoms closer than rp + r_max + probe can occlude anything.
        const double cut = rp + r_max + probe;
        const double cut2 = cut * cut;
        std::size_t n_neighbour = 0;
        for (std::size_t a = 0; a < n_atoms; ++a) {
            if (a == p) continue;
            const double dx = ax - xyz[3 * a + 0];
            const double dy = ay - xyz[3 * a + 1];
            const double dz = az - xyz[3 * a + 2];
            if (dx * dx + dy * dy + dz * dz < cut2) neighbours[n_neighbour++] = a;
        }
        std::size_t n_accessible = 0;
        for (std::size_t j = 0; j < n_sphere; ++j) {
            const double sx = points[3 * j + 0] * rp + ax;
            const double sy = points[3 * j + 1] * rp + ay;
            const double sz = points[3 * j + 2] * rp + az;
            bool accessible = true;
            for (std::size_t k = 0; k < n_neighbour; ++k) {
                const std::size_t b = neighbours[k];
                const double rb = vdw[b] + probe;
                const double dx = sx - xyz[3 * b + 0];
                const double dy = sy - xyz[3 * b + 1];
                const double dz = sz - xyz[3 * b + 2];
                if (dx * dx + dy * dy + dz * dz < rb * rb) {
                    accessible = false;
                    break;
                }
            }
            if (accessible) ++n_accessible;
        }
        asa[i] = c * static_cast<double>(n_accessible) * rp * rp;
    }
    return asa;
}

std::vector<double> quenching_rate_per_frame(
        const std::vector<int>& collided, int n_frames,
        const std::vector<double>& k_quench) {
    std::vector<double> out(static_cast<std::size_t>(std::max(0, n_frames)), 0.0);
    if (n_frames <= 0 || k_quench.empty()) return out;
    const std::size_t n_atoms = k_quench.size();
    for (std::size_t f = 0; f < out.size(); ++f) {
        double total = 0.0;
        for (std::size_t a = 0; a < n_atoms; ++a) {
            if (collided[f * n_atoms + a]) total += k_quench[a];
        }
        out[f] = total;
    }
    return out;
}

std::vector<double> fibonacci_sphere_points(int n) {
    std::vector<double> out;
    if (n <= 0) return out;
    out.resize(static_cast<std::size_t>(n) * 3);
    const double turn = M_PI * (1.0 + std::sqrt(5.0));
    for (int k = 0; k < n; ++k) {
        const double i = k + 0.5;
        const double polar = std::acos(1.0 - 2.0 * i / static_cast<double>(n));
        const double azimuth = turn * i;
        out[3 * k + 0] = std::cos(azimuth) * std::sin(polar);
        out[3 * k + 1] = std::sin(azimuth) * std::sin(polar);
        out[3 * k + 2] = std::cos(polar);
    }
    return out;
}

void probe_solvent_exposure(const std::vector<double>& centres,
                            const std::vector<double>& atom_xyz,
                            const std::vector<double>& atom_radii,
                            double** out_view, int* n_out_view,
                            double probe_radius, double water_radius,
                            int n_points) {
    const std::size_t n_centres = centres.size() / 3;
    const std::size_t n_atoms = atom_xyz.size() / 3;
    if (atom_radii.size() != n_atoms) {
        IMP_THROW("probe_solvent_exposure: " << atom_radii.size()
                  << " radii for " << n_atoms << " atoms", ValueException);
    }
    if (n_points <= 0) {
        IMP_THROW("probe_solvent_exposure: n_points must be positive",
                  ValueException);
    }
    double* out = internal::new_double_view(n_centres, out_view, n_out_view);
    if (out == nullptr) return;
    if (n_atoms == 0) {
        for (std::size_t c = 0; c < n_centres; ++c) out[c] = 1.0;
        return;
    }

    // The contact sphere, sampled once.
    const double shell = probe_radius + water_radius;
    std::vector<double> offset = fibonacci_sphere_points(n_points);
    for (std::size_t k = 0; k < offset.size(); ++k) offset[k] *= shell;

    // Per-atom reach (r_j + water) and a cell list of edge = the largest one,
    // so a sample's covering atoms lie in its own cell or the 26 around it.
    std::vector<double> reach(n_atoms);
    double cell = 0.0;
    double lo[3] = {atom_xyz[0], atom_xyz[1], atom_xyz[2]};
    double hi[3] = {lo[0], lo[1], lo[2]};
    for (std::size_t a = 0; a < n_atoms; ++a) {
        reach[a] = atom_radii[a] + water_radius;
        cell = std::max(cell, reach[a]);
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], atom_xyz[3 * a + d]);
            hi[d] = std::max(hi[d], atom_xyz[3 * a + d]);
        }
    }
    if (!(cell > 0.0)) {
        for (std::size_t c = 0; c < n_centres; ++c) out[c] = 1.0;
        return;
    }
    int nc[3];
    for (int d = 0; d < 3; ++d) {
        nc[d] = static_cast<int>(std::floor((hi[d] - lo[d]) / cell)) + 1;
    }
    const std::size_t n_cells =
            static_cast<std::size_t>(nc[0]) * nc[1] * nc[2];
    std::vector<int> head(n_cells, -1), next(n_atoms, -1);
    for (std::size_t a = 0; a < n_atoms; ++a) {
        const int ix = static_cast<int>(std::floor((atom_xyz[3 * a + 0] - lo[0]) / cell));
        const int iy = static_cast<int>(std::floor((atom_xyz[3 * a + 1] - lo[1]) / cell));
        const int iz = static_cast<int>(std::floor((atom_xyz[3 * a + 2] - lo[2]) / cell));
        const std::size_t k = (static_cast<std::size_t>(ix) * nc[1] + iy) * nc[2] + iz;
        next[a] = head[k];
        head[k] = static_cast<int>(a);
    }

    for (std::size_t c = 0; c < n_centres; ++c) {
        int covered = 0;
        for (int p = 0; p < n_points; ++p) {
            const double sx = centres[3 * c + 0] + offset[3 * p + 0];
            const double sy = centres[3 * c + 1] + offset[3 * p + 1];
            const double sz = centres[3 * c + 2] + offset[3 * p + 2];
            const int cx = static_cast<int>(std::floor((sx - lo[0]) / cell));
            const int cy = static_cast<int>(std::floor((sy - lo[1]) / cell));
            const int cz = static_cast<int>(std::floor((sz - lo[2]) / cell));
            bool hit = false;
            for (int ix = std::max(0, cx - 1); !hit && ix <= std::min(nc[0] - 1, cx + 1); ++ix) {
                for (int iy = std::max(0, cy - 1); !hit && iy <= std::min(nc[1] - 1, cy + 1); ++iy) {
                    for (int iz = std::max(0, cz - 1); !hit && iz <= std::min(nc[2] - 1, cz + 1); ++iz) {
                        const std::size_t k =
                                (static_cast<std::size_t>(ix) * nc[1] + iy) * nc[2] + iz;
                        for (int a = head[k]; a >= 0; a = next[a]) {
                            const double dx = atom_xyz[3 * a + 0] - sx;
                            const double dy = atom_xyz[3 * a + 1] - sy;
                            const double dz = atom_xyz[3 * a + 2] - sz;
                            // The distance itself, not its square: the
                            // analysis this reproduces compared norms, and a
                            // sample on the boundary must fall the same way.
                            const double xx = dx * dx;
                            const double yy = dy * dy;
                            const double zz = dz * dz;
                            if (std::sqrt((xx + yy) + zz) < reach[a]) {
                                hit = true;
                                break;
                            }
                        }
                    }
                }
            }
            if (hit) ++covered;
        }
        out[c] = 1.0 - static_cast<double>(covered) / static_cast<double>(n_points);
    }
}

const int DEFAULT_SPHERE_POINTS = 590;

std::vector<double> solvent_accessible_surface(
        const std::vector<double>& xyz, const std::vector<double>& vdw,
        const std::vector<int>& probe_atom_indices,
        const std::vector<double>& points, double probe, double radius) {
    return solvent_accessible_surface_area(
            xyz, vdw, probe_atom_indices,
            points.empty() ? sphere_points(DEFAULT_SPHERE_POINTS) : points,
            probe, radius);
}

IMPBFF_END_NAMESPACE
