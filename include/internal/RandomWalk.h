/**
 *  \file IMP/bff/internal/RandomWalk.h
 *  \brief The rejection-sampled walk core, shared by everything that walks.
 *
 * Header-only and templated on what to do with each step, so a caller that
 * wants the trajectory and a caller that wants only a field read along it run
 * the *same* walk -- same draws, same rejections, same answer -- without either
 * of them materialising what the other needs.
 *
 * That matters because the trajectory is the expensive part. A 5 000 000-step
 * walk is 20 million doubles; a caller that only wants the quenching rate the
 * dye saw wants 5 million floats and none of the coordinates.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_INTERNAL_RANDOMWALK_H
#define IMPBFF_INTERNAL_RANDOMWALK_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/util/IMPCompatibility.h>

#include <cmath>
#include <random>
#include <vector>
#include <IMP/bff/internal/PortableRandom.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE

//! Run one rejection-sampled Brownian walk in an occupancy grid.
/*!
    \param sink called once per step as
           `sink(step, px, py, pz, accepted, voxel)` -- positions in **voxel
           units**, `voxel` the flat index of the voxel occupied after the step.
    \param start_weight,n_start_weight the start weight, flat `ng^3` -- the
           field the PDE path spells `weight_map` (for the sticky ensemble,
           `w = prod_t s_t^(-n_t)`). The starting voxel is drawn in proportion
           to it over the occupied voxels. An absent field -- a null pointer or
           any length other than `ng^3` -- leaves the walk exactly as it was,
           draws included: it starts uniformly and retries up to 1000 times
           until it lands inside the accessible region.
    \return true if a starting voxel was found
*/
template <class Sink>
inline bool run_walk(const int* occupancy, std::size_t n_occupancy,
                     const double* mobility, std::size_t n_mobility,
                     int ng, double t_step, double diffusion_coefficient,
                     double dg, int seed, int n_steps,
                     int& n_accepted, int& n_rejected,
                     const double* start_weight, std::size_t n_start_weight,
                     Sink&& sink) {
    n_accepted = 0;
    n_rejected = 0;
    if (n_steps <= 0 || ng <= 0) return false;

    std::mt19937_64 rng(seed >= 0 ? static_cast<std::uint64_t>(seed)
                                  : std::random_device{}());
    // Per Cartesian component: 2 D dt. See BrownianWalk.h on why not 6 D dt.
    const double sigma = std::sqrt(2.0 * diffusion_coefficient * t_step) / dg;
    IMP::bff::internal::portable::normal_distribution<double> gauss(0.0, sigma);
    IMP::bff::internal::portable::uniform_int_distribution<int> pick(0, ng - 1);

    const std::size_t n = static_cast<std::size_t>(ng);
    const std::size_t n_vox = n * n * n;
    const bool has_mobility = n_mobility == n_vox;
    // A field of the wrong length is ignored -- the rule mobility already
    // follows, so an optional grid is one kind of thing here, not two.
    const bool has_start_weight =
            start_weight != nullptr && n_start_weight == n_vox;
    (void)n_occupancy;

    double px = 0.0, py = 0.0, pz = 0.0;
    std::size_t voxel = 0;
    bool found = false;
    if (has_start_weight) {
        // The inverse-CDF draw: one uniform double, the grid scanned in index
        // order -- the order the weights were accumulated in, so the partial
        // sums below end exactly on `total`. A voxel with no weight is never
        // picked; a voxel outside the accessible volume never is either, the
        // walk cannot stand there. One scan per walk, against steps numbered
        // in the millions: nothing here is on the walk's hot path.
        double total = 0.0;
        for (std::size_t k = 0; k < n_vox; ++k) {
            if (occupancy[k] <= 0) continue;
            const double w = start_weight[k];
            // NaN fails `w >= 0.0`, so the PDE path's rule -- a weight field
            // is real and non-negative -- holds here too.
            if (!(w >= 0.0) || !std::isfinite(w)) {
                IMP_THROW("the start weight must be finite and non-negative "
                          "on the accessible voxels; voxel " << k << " has "
                          << w, ValueException);
            }
            total += w;
        }
        if (total > 0.0) {
            IMP::bff::internal::portable::uniform_real_distribution<double> in_cdf(0.0, total);
            const double u = in_cdf(rng);
            double accumulated = 0.0;
            for (std::size_t k = 0; k < n_vox; ++k) {
                if (occupancy[k] <= 0) continue;
                accumulated += start_weight[k];
                if (start_weight[k] > 0.0 && accumulated >= u) {
                    const std::size_t iz = k % n;
                    const std::size_t iy = (k / n) % n;
                    const std::size_t ix = k / (n * n);
                    px = static_cast<double>(ix);
                    py = static_cast<double>(iy);
                    pz = static_cast<double>(iz);
                    voxel = k;
                    found = true;
                    break;
                }
            }
        }
        // total == 0.0 leaves `found` unset: nothing to start on, the same
        // answer an empty volume gets.
    } else {
        for (int attempt = 0; attempt < 1000; ++attempt) {
            const int ix = pick(rng), iy = pick(rng), iz = pick(rng);
            const std::size_t k = (static_cast<std::size_t>(ix) * n + iy) * n + iz;
            if (occupancy[k] > 0) {
                px = ix; py = iy; pz = iz;
                voxel = k;
                found = true;
                break;
            }
        }
    }
    if (!found) return false;

    ++n_accepted;
    sink(0, px, py, pz, true, voxel);
    for (int i = 1; i < n_steps; ++i) {
        double scale = 1.0;
        if (has_mobility) scale = std::sqrt(mobility[voxel]);

        const double nx = px + gauss(rng) * scale;
        const double ny = py + gauss(rng) * scale;
        const double nz = pz + gauss(rng) * scale;

        const int ix = static_cast<int>(nx);
        const int iy = static_cast<int>(ny);
        const int iz = static_cast<int>(nz);
        bool accepted = false;
        if (ix >= 0 && ix < ng && iy >= 0 && iy < ng && iz >= 0 && iz < ng) {
            const std::size_t k = (static_cast<std::size_t>(ix) * n + iy) * n + iz;
            if (occupancy[k] > 0) {
                px = nx; py = ny; pz = nz;
                voxel = k;
                accepted = true;
            }
        }
        if (accepted) ++n_accepted; else ++n_rejected;
        sink(i, px, py, pz, accepted, voxel);
    }
    return true;
}

IMPBFF_END_INTERNAL_NAMESPACE

#endif //IMPBFF_INTERNAL_RANDOMWALK_H
