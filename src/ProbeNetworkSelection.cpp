/**
 * \file ProbeNetworkSelection.cpp
 * \brief Greedy selection of a probe network by a weighted mix of scores.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/ProbeNetworkSelection.h>
#include <IMP/bff/internal/OutputView.h>
#include <IMP/bff/internal/ProbePairKernels.h>

#include <IMP/bff/IMPCompatibility.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

namespace {

bool contains(const std::vector<int>& v, int x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.compare(0, prefix.size(), prefix) == 0;
}

//! log det of a symmetric positive definite matrix; -inf when it is not.
double log_det_spd(const Eigen::MatrixXd& a) {
    Eigen::LLT<Eigen::MatrixXd> llt(a);
    if (llt.info() != Eigen::Success) return -std::numeric_limits<double>::infinity();
    const Eigen::MatrixXd& l = llt.matrixL();
    double s = 0.0;
    for (int i = 0; i < a.rows(); ++i) s += std::log(l(i, i));
    return 2.0 * s;
}

}  // namespace

// --- ProbeNetworkTerm -------------------------------------------------------------

void ProbeNetworkTerm::reset() {
    pairs_.clear();
    sites_.clear();
    do_reset();
}

void ProbeNetworkTerm::commit(const std::vector<int>& pairs,
                              const std::vector<int>& sites) {
    std::vector<int> new_sites;
    for (int s : sites) {
        if (!contains(sites_, s) && !contains(new_sites, s)) new_sites.push_back(s);
    }
    pairs_.insert(pairs_.end(), pairs.begin(), pairs.end());
    sites_.insert(sites_.end(), new_sites.begin(), new_sites.end());
    do_commit(pairs, new_sites);
}

// --- ProbeResolutionTerm ----------------------------------------------------------

ProbeResolutionTerm::ProbeResolutionTerm(
        double* predicted_measurements, int n_frames, int n_pairs,
        double* rmsds, int n_rmsd_rows, int n_rmsd_cols,
        double measurement_error, double diag_weight)
    : ProbeNetworkTerm("ProbeResolutionTerm"),
      n_frames_(n_frames), n_pairs_(n_pairs),
      inv_err_sq_(1.0 / (measurement_error * measurement_error)),
      diag_weight_(diag_weight) {
    if (n_rmsd_rows != n_rmsd_cols || n_rmsd_rows != n_frames) {
        IMP_THROW("rmsds must be square and match the frame count of predicted_measurements",
                  ValueException);
    }
    if (!(measurement_error > 0.0)) {
        IMP_THROW("measurement_error must be positive", ValueException);
    }
    const std::size_t n = static_cast<std::size_t>(std::max(n_frames, 0));
    const std::size_t m = static_cast<std::size_t>(std::max(n_pairs, 0));
    // Candidate-major, transposed once, as the free selectors do.
    e_t_.assign(m * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < m; ++k) e_t_[k * n + i] = predicted_measurements[i * m + k];
    }
    rmsds_.assign(rmsds, rmsds + n * n);
    chi2_.assign(n * n, 0.0);
    if (n > 0) {
        // Nothing measured: every weight is one, whatever the ndof.
        initial_ = internal::weighted_column_mean(rmsds_.data(), chi2_.data(), 1,
                                                  diag_weight_, n);
    }
}

void ProbeResolutionTerm::do_reset() {
    std::fill(chi2_.begin(), chi2_.end(), 0.0);
    n_measurements_ = 0;
}

void ProbeResolutionTerm::do_commit(const std::vector<int>& pairs,
                                    const std::vector<int>&) {
    const std::size_t n = static_cast<std::size_t>(n_frames_);
    for (int p : pairs) {
        internal::add_pair_to_chi2(chi2_, &e_t_[static_cast<std::size_t>(p) * n],
                                   inv_err_sq_, n);
        ++n_measurements_;
    }
}

double ProbeResolutionTerm::get_expected_rmsd() const {
    if (n_frames_ <= 0) return 0.0;
    return internal::weighted_column_mean(rmsds_.data(), chi2_.data(),
                                          std::max(n_measurements_, 1), diag_weight_,
                                          static_cast<std::size_t>(n_frames_));
}

double ProbeResolutionTerm::get_loss() const {
    return initial_ > 0.0 ? get_expected_rmsd() / initial_ : 0.0;
}

double ProbeResolutionTerm::get_loss_with(const std::vector<int>& pairs,
                                          const std::vector<int>&) const {
    if (n_frames_ <= 0 || !(initial_ > 0.0)) return 0.0;
    const std::size_t n = static_cast<std::size_t>(n_frames_);
    const int n_new = static_cast<int>(pairs.size());
    // The selector's clamped ndof: two below the measurement count it is
    // heading for (select_probe_pairs: step - 1; select_probe_positions:
    // n + n_new - 2).
    const int ndof = std::max(n_measurements_ + n_new - 2, 1);
    double value = 0.0;
    if (n_new == 1) {
        // One pair: the candidate scorer adds it on the fly, no copy.
        internal::score_candidates(rmsds_.data(), chi2_.data(),
                                   &e_t_[static_cast<std::size_t>(pairs[0]) * n],
                                   inv_err_sq_, ndof, diag_weight_, n, 1, &value);
    } else {
        std::vector<double> scratch(chi2_);
        for (int p : pairs) {
            internal::add_pair_to_chi2(scratch, &e_t_[static_cast<std::size_t>(p) * n],
                                       inv_err_sq_, n);
        }
        value = internal::weighted_column_mean(rmsds_.data(), scratch.data(), ndof,
                                               diag_weight_, n);
    }
    return value / initial_;
}

// --- ProbeKineticsTerm ------------------------------------------------------------

ProbeKineticsTerm::ProbeKineticsTerm(const FRETHiddenProcess& process,
                                     const FRETMeasurement& measurement,
                                     double* state_distances, int n_states, int n_pairs,
                                     const FRETSimulationOptions& options,
                                     double max_gap, int min_photons,
                                     double n_bursts_per_pair, double prior_sd)
    : ProbeNetworkTerm("ProbeKineticsTerm"), n_pairs_(n_pairs) {
    if (process.get_is_landscape()) {
        IMP_THROW("ProbeKineticsTerm needs a discrete process", ValueException);
    }
    if (n_states != process.get_n_states()) {
        IMP_THROW("state_distances must have one row per state of the process ("
                  << process.get_n_states() << "), not " << n_states,
                  ValueException);
    }
    if (!(prior_sd > 0.0)) IMP_THROW("prior_sd must be positive", ValueException);
    if (!(n_bursts_per_pair > 0.0)) {
        IMP_THROW("n_bursts_per_pair must be positive", ValueException);
    }
    prior_precision_ = 1.0 / (prior_sd * prior_sd);
    for (const std::string& name : process.get_parameter_names()) {
        rate_names_.push_back(name);
    }
    n_k_ = static_cast<int>(rate_names_.size());
    if (n_k_ == 0) IMP_THROW("the process has no rates", ValueException);

    const std::string mean_prefix = measurement.get_name() + ".mean[";
    const std::size_t nk = static_cast<std::size_t>(n_k_);
    g_.assign(static_cast<std::size_t>(std::max(n_pairs, 0)),
              std::vector<double>(nk * nk, 0.0));
    n_bursts_.assign(static_cast<std::size_t>(std::max(n_pairs, 0)), 0);

    for (int p = 0; p < n_pairs; ++p) {
        FRETMeasurement m(measurement);
        for (int h = 0; h < n_states; ++h) {
            m.set_state_distance(h, state_distances[h * n_pairs + p]);
        }
        FRETSimulationOptions o(options);
        o.seed = options.seed + static_cast<unsigned int>(p);
        FRETPhotonData data = simulate_fret_measurement(process, m, o);
        if (max_gap > 0.0) data = select_bursts(data, max_gap, min_photons);
        const int n_seg = data.get_n_segments();
        n_bursts_[static_cast<std::size_t>(p)] = n_seg;
        if (n_seg == 0) continue;   // nothing seen: no information

        FRETNetworkModel model(process);
        model.add_measurement(m, data);
        // Only the rates and this pair's state means are free: the rates
        // are what the network shares, the means are what each pair pays.
        for (const std::string& name : model.get_parameter_names()) {
            const bool want = starts_with(name, "hidden.") || starts_with(name, mean_prefix);
            if (model.get_parameter_free(name) != want) model.set_parameter_free(name, want);
        }
        const std::vector<std::string> free = model.get_free_parameter_names();
        const int n_free = static_cast<int>(free.size());
        const std::vector<double> scores = model.segment_scores(model.get_theta());
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
                s(scores.data(), n_seg, n_free);
        const Eigen::MatrixXd f = (s.transpose() * s) / static_cast<double>(n_seg);

        // The process's parameters come first (FRETNetworkModel's order).
        const int nd = n_free - n_k_;
        Eigen::MatrixXd g = f.topLeftCorner(n_k_, n_k_);
        if (nd > 0) {
            const Eigen::MatrixXd fkd = f.topRightCorner(n_k_, nd);
            const Eigen::MatrixXd fdd = f.bottomRightCorner(nd, nd);
            // A pseudo-inverse: a state no burst visits leaves its mean
            // undetermined, and that direction must drop out, not blow up.
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(fdd);
            const Eigen::VectorXd ev = es.eigenvalues();
            const double cut = 1e-12 * std::max(ev.cwiseAbs().maxCoeff(), 1e-300);
            Eigen::VectorXd inv(nd);
            for (int i = 0; i < nd; ++i) inv(i) = ev(i) > cut ? 1.0 / ev(i) : 0.0;
            const Eigen::MatrixXd fdd_pinv =
                    es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
            g -= fkd * fdd_pinv * fkd.transpose();
        }
        // Into a new matrix: `g = g + g.transpose()` aliases in Eigen and
        // overwrites the off-diagonal it is still reading.
        const Eigen::MatrixXd gs = (0.5 * n_bursts_per_pair) * (g + g.transpose()).eval();
        std::vector<double>& out = g_[static_cast<std::size_t>(p)];
        for (int i = 0; i < n_k_; ++i) {
            for (int j = 0; j < n_k_; ++j) out[static_cast<std::size_t>(i * n_k_ + j)] = gs(i, j);
        }
    }
    committed_.assign(nk * nk, 0.0);
}

std::vector<double> ProbeKineticsTerm::get_pair_information(int pair) const {
    return g_.at(static_cast<std::size_t>(pair));
}

void ProbeKineticsTerm::do_reset() { std::fill(committed_.begin(), committed_.end(), 0.0); }

void ProbeKineticsTerm::do_commit(const std::vector<int>& pairs, const std::vector<int>&) {
    for (int p : pairs) {
        const std::vector<double>& g = g_.at(static_cast<std::size_t>(p));
        for (std::size_t i = 0; i < committed_.size(); ++i) committed_[i] += g[i];
    }
}

double ProbeKineticsTerm::loss_of(const std::vector<double>& information) const {
    Eigen::MatrixXd a(n_k_, n_k_);
    for (int i = 0; i < n_k_; ++i) {
        for (int j = 0; j < n_k_; ++j) {
            a(i, j) = information[static_cast<std::size_t>(i * n_k_ + j)];
        }
        a(i, i) += prior_precision_;
    }
    const double log_det = log_det_spd(a);
    const double log_det0 = n_k_ * std::log(prior_precision_);
    // A Schur complement is positive semidefinite, so F0 + sum G is positive
    // definite; failing that, the information did not help.
    if (!std::isfinite(log_det)) return 1.0;
    return std::min(1.0, std::exp((log_det0 - log_det) / n_k_));
}

double ProbeKineticsTerm::get_loss() const { return loss_of(committed_); }

double ProbeKineticsTerm::get_loss_with(const std::vector<int>& pairs,
                                        const std::vector<int>&) const {
    std::vector<double> total(committed_);
    for (int p : pairs) {
        const std::vector<double>& g = g_.at(static_cast<std::size_t>(p));
        for (std::size_t i = 0; i < total.size(); ++i) total[i] += g[i];
    }
    return loss_of(total);
}

std::vector<double> ProbeKineticsTerm::get_rate_sigmas() const {
    Eigen::MatrixXd a(n_k_, n_k_);
    for (int i = 0; i < n_k_; ++i) {
        for (int j = 0; j < n_k_; ++j) a(i, j) = committed_[static_cast<std::size_t>(i * n_k_ + j)];
        a(i, i) += prior_precision_;
    }
    const Eigen::MatrixXd cov = a.ldlt().solve(Eigen::MatrixXd::Identity(n_k_, n_k_));
    std::vector<double> out(static_cast<std::size_t>(n_k_));
    for (int i = 0; i < n_k_; ++i) out[static_cast<std::size_t>(i)] = std::sqrt(cov(i, i));
    return out;
}

// --- ProbeLabellingTerm -----------------------------------------------------------

ProbeLabellingTerm::ProbeLabellingTerm(const std::map<std::string, double>& label_scores,
                                       const std::vector<std::string>& site_keys,
                                       double mutation_cost)
    : ProbeNetworkTerm("ProbeLabellingTerm") {
    if (!(mutation_cost >= 0.0)) IMP_THROW("mutation_cost must be >= 0", ValueException);
    cost_.reserve(site_keys.size());
    for (const std::string& key : site_keys) {
        const auto it = label_scores.find(key);
        if (it == label_scores.end() || !(it->second >= 0.0)) {
            cost_.push_back(std::numeric_limits<double>::infinity());
            continue;
        }
        // The label score is a likelihood ratio: at even prior odds a site
        // fails with 1 / (1 + LS).
        const double fail = 1.0 / (1.0 + it->second);
        cost_.push_back((mutation_cost + fail) / (mutation_cost + 1.0));
    }
    n_reference_ = std::max(1, static_cast<int>(site_keys.size()));
}

bool ProbeLabellingTerm::get_is_eligible_site(int site) const {
    return site >= 0 && site < static_cast<int>(cost_.size()) &&
           std::isfinite(cost_[static_cast<std::size_t>(site)]);
}

double ProbeLabellingTerm::get_site_cost(int site) const {
    return cost_.at(static_cast<std::size_t>(site));
}

void ProbeLabellingTerm::set_budget(int, int max_sites) {
    n_reference_ = std::max(1, max_sites);
}

void ProbeLabellingTerm::do_reset() { committed_ = 0.0; }

void ProbeLabellingTerm::do_commit(const std::vector<int>&, const std::vector<int>& new_sites) {
    for (int s : new_sites) committed_ += get_site_cost(s);
}

double ProbeLabellingTerm::get_loss() const {
    return std::min(1.0, committed_ / n_reference_);
}

double ProbeLabellingTerm::get_loss_with(const std::vector<int>&,
                                         const std::vector<int>& sites) const {
    double total = committed_;
    std::vector<int> seen;
    for (int s : sites) {
        if (contains(get_committed_sites(), s) || contains(seen, s)) continue;
        seen.push_back(s);
        total += get_site_cost(s);
    }
    return std::min(1.0, total / n_reference_);
}

// --- ProbeNetworkSelection --------------------------------------------------------

ProbeNetworkSelection::ProbeNetworkSelection(int n_pairs, std::string name)
    : IMP::Object(name), n_pairs_(n_pairs) {
    if (n_pairs < 0) IMP_THROW("n_pairs must be >= 0", ValueException);
}

void ProbeNetworkSelection::set_pair_sites(int* pair_sites, int n_site_rows, int n_site_cols) {
    if (n_site_rows != n_pairs_) {
        IMP_THROW("pair_sites must have one row per pair", ValueException);
    }
    if (n_site_cols != 2) {
        IMP_THROW("pair_sites must hold two site indices per pair", ValueException);
    }
    for (int p = 0; p < 2 * n_pairs_; ++p) {
        if (pair_sites[p] < 0) {
            IMP_THROW("pair_sites entries must be site indices >= 0", ValueException);
        }
    }
    pair_sites_.assign(pair_sites, pair_sites + 2 * n_pairs_);
}

int ProbeNetworkSelection::add_term(ProbeNetworkTerm* term, double weight) {
    if (term == nullptr) IMP_THROW("add_term: no term", ValueException);
    if (!(weight >= 0.0)) IMP_THROW("add_term: weight must be >= 0", ValueException);
    if (term->get_n_pairs() >= 0 && term->get_n_pairs() != n_pairs_) {
        IMP_THROW("add_term: " << term->get_name() << " was built for "
                  << term->get_n_pairs() << " pairs, the selector has " << n_pairs_,
                  ValueException);
    }
    terms_.push_back(term);
    weights_.push_back(weight);
    return static_cast<int>(terms_.size()) - 1;
}

void ProbeNetworkSelection::set_weight(int i, double weight) {
    if (!(weight >= 0.0)) IMP_THROW("set_weight: weight must be >= 0", ValueException);
    weights_.at(static_cast<std::size_t>(i)) = weight;
}

void ProbeNetworkSelection::select(int max_units, bool by_sites,
                                   int** out_units, int* n_out_units,
                                   double** out_losses, int* n_out_losses) {
    if (terms_.empty()) IMP_THROW("select: add a term first", ValueException);
    if (by_sites && pair_sites_.empty()) {
        IMP_THROW("select by sites needs set_pair_sites", ValueException);
    }
    const int n_terms = static_cast<int>(terms_.size());
    const bool have_sites = !pair_sites_.empty();

    // Sites a pair connects, and the pairs of each site.
    int n_sites = 0;
    for (int v : pair_sites_) n_sites = std::max(n_sites, v + 1);
    std::vector<std::vector<int> > pairs_of(static_cast<std::size_t>(n_sites));
    for (int p = 0; p < n_pairs_ && have_sites; ++p) {
        const int a = pair_sites_[2 * p], b = pair_sites_[2 * p + 1];
        pairs_of[static_cast<std::size_t>(a)].push_back(p);
        if (b != a) pairs_of[static_cast<std::size_t>(b)].push_back(p);
    }

    // Which units may be chosen at all.
    const int n_units = by_sites ? n_sites : n_pairs_;
    std::vector<char> eligible(static_cast<std::size_t>(n_units), 1);
    for (int u = 0; u < n_units; ++u) {
        const std::size_t su = static_cast<std::size_t>(u);
        if (contains(excluded_, u)) eligible[su] = 0;
        if (by_sites && pairs_of[su].empty()) eligible[su] = 0;
        for (int t = 0; t < n_terms && eligible[su]; ++t) {
            const ProbeNetworkTerm* term = terms_[static_cast<std::size_t>(t)];
            if (by_sites) {
                if (!term->get_is_eligible_site(u)) eligible[su] = 0;
            } else {
                if (!term->get_is_eligible_pair(u)) eligible[su] = 0;
                if (have_sites && (!term->get_is_eligible_site(pair_sites_[2 * u]) ||
                                   !term->get_is_eligible_site(pair_sites_[2 * u + 1]))) {
                    eligible[su] = 0;
                }
            }
        }
    }

    const int n_take = std::max(0, std::min(max_units, n_units));
    // The most distinct sites the selection can reach: the labelling term's
    // unit, so a full budget of fresh, unlabellable sites is a loss of 1.
    const int max_sites = by_sites ? n_take
                          : have_sites ? std::min(2 * n_take, n_sites) : 2 * n_take;
    for (int t = 0; t < n_terms; ++t) {
        terms_[static_cast<std::size_t>(t)]->set_budget(n_take, max_sites);
        terms_[static_cast<std::size_t>(t)]->reset();
    }

    std::vector<char> chosen(static_cast<std::size_t>(n_units), 0);
    std::vector<char> active(static_cast<std::size_t>(n_pairs_), 0);
    std::vector<char> site_chosen(static_cast<std::size_t>(n_sites), 0);
    std::vector<int> units;
    std::vector<double> losses;
    selected_pairs_.clear();
    term_losses_.clear();

    // What unit `u` adds: its pairs, and its sites.
    auto additions = [&](int u, std::vector<int>& pairs, std::vector<int>& sites) {
        pairs.clear();
        sites.clear();
        if (!by_sites) {
            pairs.push_back(u);
            if (have_sites) {
                sites.push_back(pair_sites_[2 * u]);
                if (pair_sites_[2 * u + 1] != pair_sites_[2 * u]) {
                    sites.push_back(pair_sites_[2 * u + 1]);
                }
            }
            return;
        }
        sites.push_back(u);
        for (int p : pairs_of[static_cast<std::size_t>(u)]) {
            const std::size_t sp = static_cast<std::size_t>(p);
            const int other = pair_sites_[2 * sp] == u ? pair_sites_[2 * sp + 1]
                                                       : pair_sites_[2 * sp];
            if (active[sp] || (other != u && !site_chosen[static_cast<std::size_t>(other)])) {
                continue;
            }
            pairs.push_back(p);
        }
    };

    std::vector<double> objective(static_cast<std::size_t>(n_units), 0.0);
    for (int step = 0; step < n_take; ++step) {
#pragma omp parallel for schedule(dynamic)
        for (int u = 0; u < n_units; ++u) {
            const std::size_t su = static_cast<std::size_t>(u);
            if (chosen[su] || !eligible[su]) continue;
            std::vector<int> pairs, sites;
            additions(u, pairs, sites);
            double j = 0.0;
            for (int t = 0; t < n_terms; ++t) {
                const double w = weights_[static_cast<std::size_t>(t)];
                if (w == 0.0) continue;
                j += w * terms_[static_cast<std::size_t>(t)]->get_loss_with(pairs, sites);
            }
            objective[su] = j;
        }
        int best = -1;
        for (int u = 0; u < n_units; ++u) {
            const std::size_t su = static_cast<std::size_t>(u);
            if (chosen[su] || !eligible[su]) continue;
            if (best < 0 || objective[su] < objective[static_cast<std::size_t>(best)]) best = u;
        }
        if (best < 0) break;   // every eligible unit is spent

        std::vector<int> pairs, sites;
        additions(best, pairs, sites);
        chosen[static_cast<std::size_t>(best)] = 1;
        for (int p : pairs) {
            active[static_cast<std::size_t>(p)] = 1;
            selected_pairs_.push_back(p);
        }
        for (int s : sites) site_chosen[static_cast<std::size_t>(s)] = 1;
        units.push_back(best);
        double total = 0.0;
        for (int t = 0; t < n_terms; ++t) {
            ProbeNetworkTerm* term = terms_[static_cast<std::size_t>(t)];
            term->commit(pairs, sites);
            const double l = term->get_loss();
            term_losses_.push_back(l);
            total += weights_[static_cast<std::size_t>(t)] * l;
        }
        losses.push_back(total);
    }

    const std::size_t n_sel = units.size();
    if (out_units == nullptr || n_out_units == nullptr ||
        out_losses == nullptr || n_out_losses == nullptr) {
        return;   // a C++ caller without views: results via the getters
    }
    internal::new_int_view(n_sel, out_units, n_out_units);
    if (*out_units != nullptr && n_sel > 0) {
        std::memcpy(*out_units, units.data(), n_sel * sizeof(int));
    }
    internal::new_double_view(n_sel, out_losses, n_out_losses);
    if (*out_losses != nullptr && n_sel > 0) {
        std::memcpy(*out_losses, losses.data(), n_sel * sizeof(double));
    }
}

void ProbeNetworkSelection::get_term_losses(double** out_matrix, int* n_out_rows,
                                            int* n_out_cols) const {
    if (out_matrix == nullptr || n_out_rows == nullptr || n_out_cols == nullptr) return;
    const int n_terms = static_cast<int>(terms_.size());
    const std::size_t n = term_losses_.size();
    int n_flat = 0;
    double* buffer = internal::new_double_view(n, out_matrix, &n_flat);
    if (buffer == nullptr || n_terms == 0) {
        *n_out_rows = 0;
        *n_out_cols = n_terms;
        return;
    }
    if (n > 0) std::memcpy(buffer, term_losses_.data(), n * sizeof(double));
    *n_out_rows = static_cast<int>(n) / n_terms;
    *n_out_cols = n_terms;
}

IMPBFF_END_NAMESPACE
