/**
 * \file PhotophysicsQuenchingModel.cpp
 * \brief A labelled site's donor decay, two ways.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/spectroscopy/PhotophysicsQuenchingModel.h>

#include <IMP/bff/probe/DiffusionSolver.h>
#include <IMP/bff/probe/ProbeSampling.h>
#include <IMP/bff/probe/GridDiffusionSolver.h>
#include <IMP/bff/spectroscopy/PhotophysicsPhotonSimulation.h>
#include <IMP/bff/spectroscopy/PhotophysicsQuenching.h>
#include <IMP/bff/fret/FRET.h>
#include <IMP/bff/internal/OutputView.h>

#include <IMP/bff/util/IMPCompatibility.h>

#include <cctype>
#include <map>

#include <algorithm>
#include <cmath>
#include <cstdlib>

IMPBFF_BEGIN_NAMESPACE

// Named, not anonymous: IMP compiles this module as one translation unit.
namespace qmodel {

//! Drain an output-view call into a vector, and free the kernel's buffer.
std::vector<double> drained(double* buffer, int n) {
    std::vector<double> out(buffer, buffer + (n > 0 ? n : 0));
    std::free(buffer);
    return out;
}

//! The walk/stamp anchor: voxel i is centred at x0 + (i - half + 0.5) dg.
/*! The frame #center_grid_indices floors into and the walk reports in. It
    used to be the attachment point, which put every stamped sphere and every
    walker up to half a voxel per axis away from the voxel the volume itself
    means (`grid_origin + i dg`): the volume's lattice is snapped to the grid,
    the attachment point is not. */
std::vector<double> anchor(const ProbeAccessibleVolume& av) {
    return av.get_walk_anchor();
}

//! The voxel-centre anchor of the #grid_axis maps: centre x0 + (i - half) dg.
std::vector<double> centre_anchor(const ProbeAccessibleVolume& av) {
    std::vector<double> x0 = av.get_walk_anchor();
    for (std::size_t d = 0; d < x0.size(); ++d) x0[d] += 0.5 * av.get_grid_step();
    return x0;
}

//! The density of a volume, as a vector.
std::vector<double> density_of(const ProbeAccessibleVolume& av) {
    double* p = NULL;
    int n = 0;
    av.get_density(&p, &n);
    return drained(p, n);
}

}  // namespace qmodel

// --------------------------------------------------------------------------
// the obstacles
// --------------------------------------------------------------------------

void ObstacleAtoms::get_coords(double** out_view, int* n_out_view) const {
    internal::copy_to_view(coords, out_view, n_out_view);
}

// --------------------------------------------------------------------------
// the field picture
// --------------------------------------------------------------------------

DynamicAccessibleVolume::DynamicAccessibleVolume(
        const ProbeAccessibleVolume& av, const ObstacleAtoms& atoms, double tau0,
        double probe_radius, double free_diffusion, double contact_distance,
        double slow_factor, const std::string& flux_form)
    : av_(av), atoms_(atoms), tau0_(tau0), probe_radius_(probe_radius),
      free_diffusion_(free_diffusion), contact_distance_(contact_distance),
      slow_factor_(slow_factor), flux_form_(flux_form) {
    if (flux_form != "smoluchowski" && flux_form != "ito") {
        IMP_THROW("flux_form must be 'smoluchowski' or 'ito', not '"
                          << flux_form << "'",
                  ValueException);
    }
}

void DynamicAccessibleVolume::get_density(double** out_view,
                                          int* n_out_view) const {
    av_.get_density(out_view, n_out_view);
}

void DynamicAccessibleVolume::get_x0(double** out_view, int* n_out_view) const {
    internal::copy_to_view(qmodel::centre_anchor(av_), out_view, n_out_view);
}

std::vector<double> DynamicAccessibleVolume::bounds() const {
    double* d = NULL;
    int n = 0;
    av_.get_density(&d, &n);
    std::vector<double> out(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) out[i] = d[i] > 0.0 ? 1.0 : 0.0;
    std::free(d);
    return out;
}

void DynamicAccessibleVolume::get_bounds(double** out_view,
                                         int* n_out_view) const {
    internal::copy_to_view(bounds(), out_view, n_out_view);
}

void DynamicAccessibleVolume::update_diffusion_map(
        const std::vector<double>& base) {
    double* d = NULL;
    int n = 0;
    av_.get_density(&d, &n);
    const std::vector<double> density = qmodel::drained(d, n);

    double* out = NULL;
    int n_out = 0;
    diffusion_coefficient_map(density, qmodel::centre_anchor(av_),
                              av_.get_grid_step(), atoms_.coords,
                              free_diffusion_, contact_distance_, slow_factor_,
                              base, &out, &n_out);
    diffusion_map_ = qmodel::drained(out, n_out);
}

void DynamicAccessibleVolume::get_diffusion_map(double** out_view,
                                                int* n_out_view) const {
    if (diffusion_map_.empty()) {
        const_cast<DynamicAccessibleVolume*>(this)->update_diffusion_map();
    }
    internal::copy_to_view(diffusion_map_, out_view, n_out_view);
}

void DynamicAccessibleVolume::update_quenching_map(
        const std::map<std::string, PETParameters>& quencher, double rC) {
    double* kQ = NULL;
    double* rC_atoms = NULL;
    int n_kQ = 0, n_rC = 0;
    atomic_quenching_parameters(atoms_.res_names, atoms_.atom_names, quencher,
                                &kQ, &n_kQ, &rC_atoms, &n_rC);
    std::vector<double> kQ_v = qmodel::drained(kQ, n_kQ);
    std::vector<double> rC_v = qmodel::drained(rC_atoms, n_rC);
    if (rC == rC) {  // not NaN: one electron-transfer length for every quencher
        for (std::size_t i = 0; i < rC_v.size(); ++i) {
            rC_v[i] = kQ_v[i] > 0.0 ? rC : 0.0;
        }
    }

    double* d = NULL;
    int n = 0;
    av_.get_density(&d, &n);
    const std::vector<double> density = qmodel::drained(d, n);

    double* out = NULL;
    int n_out = 0;
    quenching_rate_map(density, qmodel::centre_anchor(av_),
                       av_.get_grid_step(), atoms_.coords, kQ_v, rC_v, tau0_,
                       probe_radius_, &out, &n_out);
    quenching_rate_map_ = qmodel::drained(out, n_out);
}

void DynamicAccessibleVolume::get_quenching_rate_map(double** out_view,
                                                     int* n_out_view) const {
    if (quenching_rate_map_.empty()) {
        IMP_THROW("No quenching map yet -- call update_quenching_map(quencher) "
                  "with the per-atom PET parameters.",
                  ValueException);
    }
    internal::copy_to_view(quenching_rate_map_, out_view, n_out_view);
}

void DynamicAccessibleVolume::update_fret_map(
        const DynamicAccessibleVolume& acceptor, double forster_radius,
        int acceptor_step) {
    double* dd = NULL;
    double* da = NULL;
    int n_d = 0, n_a = 0;
    av_.get_density(&dd, &n_d);
    acceptor.av_.get_density(&da, &n_a);
    const std::vector<double> density_d = qmodel::drained(dd, n_d);
    const std::vector<double> density_a = qmodel::drained(da, n_a);

    double* out = NULL;
    int n_out = 0;
    fret_rate_map(density_d, density_a, qmodel::centre_anchor(av_),
                  qmodel::centre_anchor(acceptor.av_), av_.get_grid_step(),
                  acceptor.av_.get_grid_step(), forster_radius, 1.0 / tau0_,
                  acceptor_step, &out, &n_out);
    fret_rate_map_ = qmodel::drained(out, n_out);
}

void DynamicAccessibleVolume::get_fret_rate_map(double** out_view,
                                                int* n_out_view) const {
    internal::copy_to_view(fret_rate_map_, out_view, n_out_view);
}

void DynamicAccessibleVolume::get_rate_map(double** out_view,
                                           int* n_out_view) const {
    double* total = NULL;
    int n_total = 0;
    get_quenching_rate_map(&total, &n_total);
    if (!fret_rate_map_.empty()) {
        for (int i = 0; i < n_total; ++i) total[i] += fret_rate_map_[i];
    }
    *out_view = total;
    *n_out_view = n_total;
}

void DynamicAccessibleVolume::update_occupancy() {
    if (diffusion_map_.empty()) update_diffusion_map();
    const std::vector<double> b = bounds();
    std::vector<int> mask(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) mask[i] = b[i] > 0.0 ? 1 : 0;

    double* out = NULL;
    int n_out = 0;
    equilibrium_occupancy(diffusion_map_, mask, flux_form_, std::vector<double>(),
                          &out, &n_out);
    occupancy_ = qmodel::drained(out, n_out);
}

double DynamicAccessibleVolume::resolve_t_step(double t_step) const {
    if (t_step > 0.0) return t_step;
    if (diffusion_map_.empty()) {
        const_cast<DynamicAccessibleVolume*>(this)->update_diffusion_map();
    }
    const double d_max = diffusion_map_.empty()
                                 ? 0.0
                                 : *std::max_element(diffusion_map_.begin(),
                                                     diffusion_map_.end());
    // Half the explicit limit: stable with room for the map to change.
    return 0.5 * diffusion_stability_limit(d_max, av_.get_grid_step());
}

void DynamicAccessibleVolume::update_occupancy_by_iteration(double t_step,
                                                            int n_steps,
                                                            double tolerance,
                                                            int n_check) {
    if (diffusion_map_.empty()) update_diffusion_map();
    const std::vector<double> b = bounds();
    GridDiffusionSolver solver(
            diffusion_map_, b, b,
            std::vector<double>(diffusion_map_.size(), 0.0),
            av_.get_grid_step(), resolve_t_step(t_step), flux_form_, true);
    double* out = NULL;
    int n_out = 0;
    solver.equilibrium(n_steps, tolerance, n_check, &out, &n_out);
    occupancy_ = qmodel::drained(out, n_out);
}

void DynamicAccessibleVolume::get_occupancy(double** out_view,
                                            int* n_out_view) const {
    if (occupancy_.empty()) {
        const_cast<DynamicAccessibleVolume*>(this)->update_occupancy();
    }
    internal::copy_to_view(occupancy_, out_view, n_out_view);
}

GridDiffusionResult DynamicAccessibleVolume::donor_decay(
        double t_max, double t_step, int n_out) const {
    DynamicAccessibleVolume* self = const_cast<DynamicAccessibleVolume*>(this);
    if (occupancy_.empty()) self->update_occupancy();
    if (diffusion_map_.empty()) self->update_diffusion_map();

    double* rate = NULL;
    int n_rate = 0;
    get_rate_map(&rate, &n_rate);
    const std::vector<double> rate_map = qmodel::drained(rate, n_rate);

    const double step = resolve_t_step(t_step);
    GridDiffusionSolver solver(
            diffusion_map_, bounds(), occupancy_, rate_map, av_.get_grid_step(),
            step, flux_form_, true);
    const int n_steps = std::max(1, static_cast<int>(t_max / step));

    // run() builds the time axis from the resolved step, so this is the decay
    // as a GridDiffusionResult -- no lead-step buffer to split.
    return solver.run(n_steps, std::max(1, n_out));
}

// --------------------------------------------------------------------------
// the particle picture
// --------------------------------------------------------------------------

QuenchedDonorDecay::QuenchedDonorDecay(
        const ProbeAccessibleVolume& av, const ObstacleAtoms& atoms, double tau0,
        const std::map<std::string, ResidueQuenching>& quenching_table,
        double critical_distance, double slow_radius, double probe_radius,
        double diffusion_coefficient, double slow_fact, double t_step,
        double t_max, int n_photons, int n_trajectories, int random_seed)
    : av_(av), atoms_(atoms), tau0_(tau0),
      critical_distance_(critical_distance), slow_radius_(slow_radius),
      probe_radius_(probe_radius), diffusion_coefficient_(diffusion_coefficient),
      slow_fact_(std::min(1.0, std::max(0.0, slow_fact))), t_step_(t_step),
      t_max_(t_max), n_photons_(n_photons), n_trajectories_(n_trajectories),
      random_seed_(random_seed),
      table_(normalize_amino_acid_quenching(quenching_table)),
      has_sites_(false), has_grids_(false), has_walk_(false),
      has_photons_(false), n_frames_(0.0), mean_k_quench_(0.0),
      collision_fraction_(0.0) {}

std::vector<int> QuenchedDonorDecay::density() const {
    double* d = NULL;
    int n = 0;
    av_.get_density(&d, &n);
    std::vector<int> out(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) out[i] = d[i] != 0.0 ? 1 : 0;
    std::free(d);
    return out;
}

std::vector<double> QuenchedDonorDecay::x0() const {
    return qmodel::anchor(av_);
}

void QuenchedDonorDecay::get_density(double** out_view,
                                     int* n_out_view) const {
    av_.get_density(out_view, n_out_view);
}

void QuenchedDonorDecay::get_x0(double** out_view, int* n_out_view) const {
    internal::copy_to_view(x0(), out_view, n_out_view);
}

const ResidueSites& QuenchedDonorDecay::get_sites() const {
    if (!has_sites_) {
        sites_ = residue_sites(
                atoms_.chains, atoms_.res_ids, atoms_.res_names,
                atoms_.atom_names,
                atoms_.coords.empty()
                        ? NULL
                        : const_cast<double*>(&atoms_.coords[0]),
                static_cast<int>(atoms_.coords.size() / 3), 3, table_);
        has_sites_ = true;
    }
    return sites_;
}

void QuenchedDonorDecay::update_grids() {
    const ResidueSites& sites = get_sites();
    const std::vector<std::string> names = sites.get_residue_names();
    const std::vector<double> d = [&] {
        double* p = NULL;
        int n = 0;
        av_.get_density(&p, &n);
        return qmodel::drained(p, n);
    }();
    const int ng = av_.get_ng();

    double* centres = NULL;
    int n_centres = 0;
    sites.get_quench_centers(&centres, &n_centres);
    const std::vector<double> quench = qmodel::drained(centres, n_centres);

    double* out = NULL;
    int n_out = 0;
    quenching_rate_grid(d, ng, av_.get_grid_step(),
                        quench_radii_for_residues(names, table_,
                                                  critical_distance_),
                        quench, x0(),
                        quenching_rates_for_residues(names, table_), &out,
                        &n_out);
    quenching_rate_map_ = qmodel::drained(out, n_out);

    sites.get_slow_centers(&centres, &n_centres);
    const std::vector<double> slow = qmodel::drained(centres, n_centres);
    out = NULL;
    n_out = 0;
    slow_factor_grid(d, ng, av_.get_grid_step(),
                     std::vector<double>(names.size(), slow_radius_), slow,
                     x0(), slow_factors_for_residues(names, table_), &out,
                     &n_out);
    slow_factor_map_ = qmodel::drained(out, n_out);
    has_grids_ = true;
}

void QuenchedDonorDecay::ensure_grids() const {
    if (!has_grids_) const_cast<QuenchedDonorDecay*>(this)->update_grids();
}

void QuenchedDonorDecay::get_quenching_rate_map(double** out_view,
                                                int* n_out_view) const {
    ensure_grids();
    internal::copy_to_view(quenching_rate_map_, out_view, n_out_view);
}

void QuenchedDonorDecay::get_slow_factor_map(double** out_view,
                                             int* n_out_view) const {
    ensure_grids();
    internal::copy_to_view(slow_factor_map_, out_view, n_out_view);
}

void QuenchedDonorDecay::set_slow_factor_map(
        const std::vector<double>& m) {
    slow_factor_map_ = m;
    has_grids_ = true;
    has_walk_ = false;   // a changed mobility means any previous walk is stale
}

bool QuenchedDonorDecay::simulate_diffusion() {
    ensure_grids();
    walk_ = ProbeDiffusionSimulation(density(), av_.get_grid_step(), x0(),
                                   std::vector<int>(), slow_factor_map_,
                                   quenching_rate_map_);
    walk_.simulate(diffusion_coefficient_, slow_fact_, t_step_, t_max_,
              n_trajectories_, random_seed_);
    has_walk_ = true;
    has_photons_ = false;
    n_frames_ = walk_.get_n_frames();
    collision_fraction_ = walk_.get_collision_fraction();
    double* k = NULL;
    int n_k = 0;
    walk_.get_k_quench(&k, &n_k);
    // Accumulated in float, as the fused kernel's own mean is, so the two paths
    // report the same number rather than one that differs in the last digits.
    float total = 0.0f;
    for (int i = 0; i < n_k; ++i) total += static_cast<float>(k[i]);
    std::free(k);
    mean_k_quench_ = n_k > 0 ? static_cast<double>(total / static_cast<float>(n_k))
                             : 0.0;
    return walk_.get_n_frames() > 0;
}

void QuenchedDonorDecay::ensure_walk() const {
    if (!has_walk_) const_cast<QuenchedDonorDecay*>(this)->simulate_diffusion();
}

const ProbeDiffusionSimulation& QuenchedDonorDecay::get_diffusion() const {
    ensure_walk();
    return walk_;
}

void QuenchedDonorDecay::get_k_quench(double** out_view,
                                      int* n_out_view) const {
    ensure_walk();
    walk_.get_k_quench(out_view, n_out_view);
}

int QuenchedDonorDecay::get_photon_seed() const {
    if (random_seed_ < 0) return -1;
    return static_cast<int>((static_cast<long long>(random_seed_) + 7919LL) %
                            ((1LL << 31) - 1));
}

void QuenchedDonorDecay::simulate_photons(
        const std::vector<double>& k_quench) {
    std::vector<double> rates = k_quench;
    if (rates.empty()) {
        ensure_walk();
        double* k = NULL;
        int n_k = 0;
        walk_.get_k_quench(&k, &n_k);
        rates = qmodel::drained(k, n_k);
    }
    double* flat = NULL;
    int n_flat = 0;
    photon_trace(n_photons_, rates, walk_.get_t_step(), tau0_, get_photon_seed(),
                 &flat, &n_flat);
    const int n = n_flat / 2;
    delays_.assign(n, 0.0);
    emitted_.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        delays_[i] = flat[2 * i];
        emitted_[i] = flat[2 * i + 1] != 0.0 ? 1 : 0;
    }
    std::free(flat);
    has_photons_ = true;
}

void QuenchedDonorDecay::photons_fused() {
    ensure_grids();
    // NOT get_diffusion() -- that runs the split walk first, which would do the
    // whole simulation twice and make the fused path the slower one. Measured:
    // 4.9 s against 1.3 s at a million steps.
    if (!has_walk_) {
        walk_ = ProbeDiffusionSimulation(density(), av_.get_grid_step(), x0(),
                                       std::vector<int>(), slow_factor_map_,
                                       quenching_rate_map_);
    }
    walk_.set_t_step(t_step_);

    std::vector<int> occupancy = density();
    std::vector<double> mobility = slow_factor_map_;
    std::vector<double> rate_map = quenching_rate_map_;
    const std::vector<int> seeds = trajectory_seeds(
            random_seed_, resolve_trajectory_count(n_trajectories_));

    std::vector<double> stats;
    double* flat = NULL;
    int n_flat = 0;
    quenched_donor_photons(
            occupancy.empty() ? NULL : &occupancy[0],
            static_cast<int>(occupancy.size()),
            mobility.empty() ? NULL : &mobility[0],
            static_cast<int>(mobility.size()),
            rate_map.empty() ? NULL : &rate_map[0],
            static_cast<int>(rate_map.size()), av_.get_ng(),
            av_.get_grid_step(), t_max_, t_step_, diffusion_coefficient_, seeds,
            tau0_, n_photons_, get_photon_seed(), stats, &flat, &n_flat);

    const int n = n_flat / 2;
    delays_.assign(n, 0.0);
    emitted_.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        delays_[i] = flat[2 * i];
        emitted_[i] = flat[2 * i + 1] != 0.0 ? 1 : 0;
    }
    std::free(flat);
    has_photons_ = true;

    // The fused kernel never materialises a trajectory, but it does report the
    // step counts, and the walk record is where callers look for them.
    if (stats.size() == 5) {
        walk_.set_step_counts(static_cast<int>(stats[1]),
                              static_cast<int>(stats[2]));
        n_frames_ = stats[0];
        mean_k_quench_ = stats[3];
        collision_fraction_ = stats[4];
    }
    has_walk_ = true;
}

void QuenchedDonorDecay::ensure_photons() const {
    if (!has_photons_) const_cast<QuenchedDonorDecay*>(this)->simulate_photons();
}

void QuenchedDonorDecay::get_delays(double** out_view, int* n_out_view) const {
    ensure_photons();
    internal::copy_to_view(delays_, out_view, n_out_view);
}

void QuenchedDonorDecay::get_emitted(int** out_view_i,
                                     int* n_out_view_i) const {
    ensure_photons();
    int* out = internal::new_int_view(emitted_.size(), out_view_i, n_out_view_i);
    if (out == NULL) return;
    for (std::size_t i = 0; i < emitted_.size(); ++i) out[i] = emitted_[i];
}

int QuenchedDonorDecay::get_n_frames() const {
    ensure_walk();
    return static_cast<int>(n_frames_);
}

double QuenchedDonorDecay::get_mean_k_quench() const {
    ensure_walk();
    return mean_k_quench_;
}

double QuenchedDonorDecay::get_collision_fraction() const {
    ensure_walk();
    return collision_fraction_;
}

double QuenchedDonorDecay::get_quantum_yield() const {
    if (n_photons_ <= 0) return 0.0;
    ensure_photons();
    double total = 0.0;
    for (std::size_t i = 0; i < emitted_.size(); ++i) total += emitted_[i];
    return total / n_photons_;
}

double QuenchedDonorDecay::get_fluorescence_lifetime() const {
    ensure_photons();
    double total = 0.0;
    int n = 0;
    for (std::size_t i = 0; i < emitted_.size(); ++i) {
        if (emitted_[i] == 1) { total += delays_[i]; ++n; }
    }
    return n > 0 ? total / n : 0.0;
}

PhotophysicsLifetimeSpectrum QuenchedDonorDecay::lifetime_spectrum(int n_species) const {
    ensure_walk();
    double* k = NULL;
    int n_k = 0;
    walk_.get_k_quench(&k, &n_k);
    std::vector<double> rates = qmodel::drained(k, n_k);
    if (rates.empty()) {
        IMP_THROW("no quenching rate trace; run the walk first", ValueException);
    }
    const double intrinsic = tau0_ > 0.0 ? 1.0 / tau0_ : 0.0;
    for (std::size_t i = 0; i < rates.size(); ++i) rates[i] += intrinsic;
    const PhotophysicsLifetimeSpectrum spectrum =
            lifetime_spectrum_from_rates(rates, std::vector<double>(), false);
    return n_species > 0 ? spectrum.coarse_grain(n_species) : spectrum;
}

void QuenchedDonorDecay::decay_histogram(int n_bins, double t_min,
                                         double t_max, double** out_view,
                                         int* n_out_view) const {
    ensure_photons();
    if (n_bins < 1) {
        IMP_THROW("n_bins must be at least 1, not " << n_bins, ValueException);
    }
    double* out = internal::new_double_view(2 * n_bins + 1, out_view,
                                            n_out_view);
    if (out == NULL) return;
    const double width = (t_max - t_min) / n_bins;
    for (int i = 0; i <= n_bins; ++i) out[i] = t_min + i * width;
    for (int i = 0; i < n_bins; ++i) out[n_bins + 1 + i] = 0.0;
    for (std::size_t i = 0; i < delays_.size(); ++i) {
        if (emitted_[i] != 1) continue;
        const double t = delays_[i];
        if (t < t_min || t > t_max) continue;
        // The last bin is closed on the right, as numpy's histogram is.
        int bin = static_cast<int>((t - t_min) / width);
        if (bin >= n_bins) bin = n_bins - 1;
        out[n_bins + 1 + bin] += 1.0;
    }
}

void QuenchedDonorDecay::fret_rate_trace_paired(
        const QuenchedDonorDecay& acceptor, double forster_radius,
        double kappa2, double r_min, double** out_view,
        int* n_out_view) const {
    ensure_walk();
    acceptor.ensure_walk();
    double* d = NULL;
    double* a = NULL;
    int n_d = 0, n_a = 0;
    walk_.get_trajectory(&d, &n_d);
    acceptor.walk_.get_trajectory(&a, &n_a);
    std::vector<double> donor = qmodel::drained(d, n_d);
    std::vector<double> other = qmodel::drained(a, n_a);
    if (donor.empty()) {
        IMP_THROW("The donor walk produced no trajectory.", ValueException);
    }
    if (other.empty()) {
        IMP_THROW("The acceptor walk produced no trajectory.", ValueException);
    }
    const std::size_t n = std::min(donor.size(), other.size());
    donor.resize(n);
    other.resize(n);
    fret_rate_pair_trace(donor, other, forster_radius, tau0_, r_min, kappa2,
                         out_view, n_out_view);
}

void QuenchedDonorDecay::fret_rate_trace_cloud(
        const std::vector<double>& points, double forster_radius,
        double kappa2, double r_min, double** out_view,
        int* n_out_view) const {
    ensure_walk();
    double* d = NULL;
    int n_d = 0;
    walk_.get_trajectory(&d, &n_d);
    const std::vector<double> donor = qmodel::drained(d, n_d);
    if (donor.empty()) {
        IMP_THROW("The donor walk produced no trajectory.", ValueException);
    }
    fret_rate_trace(donor, points, forster_radius, tau0_, r_min, 512, kappa2,
                    out_view, n_out_view);
}

double QuenchedDonorDecay::fret_efficiency(
        const std::vector<double>& fret_rates) const {
    const double donor_yield = get_quantum_yield();
    if (!(donor_yield > 0.0)) return 0.0;

    ensure_walk();
    double* k = NULL;
    int n_k = 0;
    walk_.get_k_quench(&k, &n_k);
    std::vector<double> total = qmodel::drained(k, n_k);
    const std::size_t n = std::min(total.size(), fret_rates.size());
    total.resize(n);
    for (std::size_t i = 0; i < n; ++i) total[i] += fret_rates[i];

    double* flat = NULL;
    int n_flat = 0;
    photon_trace(n_photons_, total, walk_.get_t_step(), tau0_, get_photon_seed(),
                 &flat, &n_flat);
    double emitted = 0.0;
    for (int i = 0; i < n_flat / 2; ++i) emitted += flat[2 * i + 1];
    std::free(flat);
    const double paired_yield =
            n_photons_ > 0 ? emitted / n_photons_ : 0.0;
    return 1.0 - paired_yield / donor_yield;
}

namespace {
std::string upper_trimmed(const std::string& name) {
    std::size_t b = 0, e = name.size();
    while (b < e && std::isspace(static_cast<unsigned char>(name[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(name[e - 1]))) --e;
    std::string out = name.substr(b, e - b);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[i])));
    }
    return out;
}
}  // namespace

void residue_type_neighbour_counts(
        const ProbeAccessibleVolume& av, const ObstacleAtoms& atoms,
        const std::vector<std::string>& residue_types,
        int** out_matrix_i, int* n_out_rows, int* n_out_cols,
        double radius) {
    const std::vector<double>& density = av.get_density_vector();
    const int ng = av.get_ng();
    const double dg = av.get_grid_step();
    if (density.empty() || ng <= 0) {
        IMP_THROW("residue_type_neighbour_counts: the volume carries no grid",
                  ValueException);
    }
    const std::size_t n_types = residue_types.size();
    // Rows: the occupied voxels, ascending flat index.
    std::vector<int> row_of(density.size(), -1);
    std::size_t n_rows = 0;
    for (std::size_t k = 0; k < density.size(); ++k) {
        if (density[k] != 0.0) row_of[k] = static_cast<int>(n_rows++);
    }
    int n_flat = 0;
    int* out = internal::new_int_view(n_rows * n_types, out_matrix_i, &n_flat);
    *n_out_cols = static_cast<int>(n_types);
    *n_out_rows = 0;
    if (out == nullptr) return;
    *n_out_rows = static_cast<int>(n_rows);
    if (n_types == 0 || atoms.coords.empty()) return;

    std::map<std::string, int> column;
    for (std::size_t t = 0; t < n_types; ++t) {
        column[upper_trimmed(residue_types[t])] = static_cast<int>(t);
    }
    const ResidueSites sites = residue_sites(
            atoms.chains, atoms.res_ids, atoms.res_names, atoms.atom_names,
            const_cast<double*>(&atoms.coords[0]),
            static_cast<int>(atoms.coords.size() / 3), 3);
    const std::vector<std::string> names = sites.get_residue_names();
    double* c = NULL;
    int n_c = 0;
    sites.get_slow_centers(&c, &n_c);
    const std::vector<double> centres = qmodel::drained(c, n_c);

    std::vector<int> ix0, iy0, iz0, r_idx;
    center_grid_indices(centres, qmodel::anchor(av), dg, ng,
                        std::vector<double>(names.size(), radius), ix0, iy0,
                        iz0, r_idx);
    for (std::size_t r = 0; r < names.size(); ++r) {
        const std::map<std::string, int>::const_iterator col =
                column.find(upper_trimmed(names[r]));
        if (col == column.end()) continue;
        const int ri = r_idx[r];
        const int r2 = ri * ri;
        for (int ix = std::max(0, ix0[r] - ri); ix <= std::min(ng - 1, ix0[r] + ri); ++ix) {
            const int dx = ix - ix0[r];
            for (int iy = std::max(0, iy0[r] - ri); iy <= std::min(ng - 1, iy0[r] + ri); ++iy) {
                const int dy = iy - iy0[r];
                for (int iz = std::max(0, iz0[r] - ri); iz <= std::min(ng - 1, iz0[r] + ri); ++iz) {
                    const int dz = iz - iz0[r];
                    if (dx * dx + dy * dy + dz * dz >= r2) continue;
                    const std::size_t k =
                            (static_cast<std::size_t>(ix) * ng + iy) * ng + iz;
                    const int row = row_of[k];
                    if (row >= 0) ++out[static_cast<std::size_t>(row) * n_types + col->second];
                }
            }
        }
    }
}

IMPBFF_END_NAMESPACE

