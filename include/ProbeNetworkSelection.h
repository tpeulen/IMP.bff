/**
 *  \file IMP/bff/ProbeNetworkSelection.h
 *  \brief Greedy selection of a probe network by a weighted mix of scores:
 *         structural resolution, dynamics, and how well the sites label.
 *
 * #IMP::bff::select_probe_pairs (Olga) picks the pairs that resolve the
 * *structure*: the expected RMSD over an ensemble. It is blind to dynamics.
 * Two pairs that separate A from B while seeing the same exchange look
 * complementary to it. This file generalises it into one selector with
 * pluggable terms, mixed by a weighted sum:
 *
 * - #IMP::bff::ProbeResolutionTerm: Olga's expected RMSD, relative to the
 *   RMSD before any measurement.
 * - #IMP::bff::ProbeKineticsTerm: how well the rates of a kinetic scheme are
 *   determined. Each pair is its own double mutant, measured on its own
 *   molecules, so pairs share the scheme's rates but not a trajectory. What a
 *   pair adds is its Fisher information on the rates after its own per-state
 *   distances are paid for (a Schur complement). That information adds up
 *   across pairs, and the loss is the geometric-mean posterior variance of the
 *   log-rates relative to their prior.
 * - #IMP::bff::ProbeLabellingTerm: the Labelizer's per-site label score and
 *   a cost per mutation, so reusing a chosen site is free.
 *
 * Every term reports a dimensionless loss in `[0, 1]`, relative to nothing
 * selected, so the weights of #IMP::bff::ProbeNetworkSelection::add_term are
 * comparable. With only a resolution term the selector is exactly
 * #IMP::bff::select_probe_pairs (pairs) or #IMP::bff::select_probe_positions
 * (sites).
 *
 * A term scores a *set*: #IMP::bff::ProbeNetworkTerm::get_loss_with receives
 * everything one candidate would add (in site mode, all pairs a new site
 * implies), and the selector assumes neither additivity nor independent pairs.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_PROBENETWORKSELECTION_H
#define IMPBFF_PROBENETWORKSELECTION_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/FRETNetwork.h>
#include <IMP/bff/FRETNetworkSimulation.h>

#include <IMP/Object.h>
#include <IMP/Pointer.h>
#include <IMP/bff/IMPCompatibility.h>

#include <cmath>
#include <map>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! One score of a probe network: a loss in `[0, 1]` over a set of pairs.
/*!
    A term keeps the set committed so far and answers, without changing it,
    what its loss would be with a group of pairs and sites added. The
    selector calls #get_loss_with concurrently from several threads, so an
    implementation must not modify state there.

    Pairs are candidate indices, the columns of the prediction matrices;
    sites are labelling-site indices, as in `pair_sites` of
    #IMP::bff::select_probe_positions.
*/
class IMPBFFEXPORT ProbeNetworkTerm : public IMP::Object {
public:
    ProbeNetworkTerm(std::string name) : IMP::Object(name) {}

    //! Number of candidate pairs the term was built for; -1 when any.
    virtual int get_n_pairs() const { return -1; }
    //! Whether the term allows pair \p pair to be selected at all.
    virtual bool get_is_eligible_pair(int pair) const { return pair >= 0; }
    //! Whether the term allows site \p site to be selected at all.
    virtual bool get_is_eligible_site(int site) const { return site >= 0; }
    //! Told by the selector before a run: the most units and distinct sites
    //! the selection can reach.
    virtual void set_budget(int max_units, int max_sites) {
        (void)max_units;
        (void)max_sites;
    }

    //! The loss of the committed set.
    virtual double get_loss() const = 0;
    //! The loss with \p pairs and \p sites added to the committed set.
    virtual double get_loss_with(const std::vector<int>& pairs,
                                 const std::vector<int>& sites) const = 0;

    //! Forget the committed set.
    void reset();
    //! Add \p pairs and \p sites to the committed set.
    void commit(const std::vector<int>& pairs, const std::vector<int>& sites);
    //! The committed pairs, in the order they were committed.
    const std::vector<int>& get_committed_pairs() const { return pairs_; }
    //! The committed sites, each once, in the order they were committed.
    const std::vector<int>& get_committed_sites() const { return sites_; }

    IMP_OBJECT_METHODS(ProbeNetworkTerm);

protected:
    //! Clear the term's own state; the committed lists are already empty.
    virtual void do_reset() = 0;
    //! Add to the term's own state; \p new_sites excludes sites committed
    //! before.
    virtual void do_commit(const std::vector<int>& pairs,
                           const std::vector<int>& new_sites) = 0;

private:
    std::vector<int> pairs_, sites_;
};
IMP_OBJECTS(ProbeNetworkTerm, ProbeNetworkTerms);

//! Structural resolution: Olga's expected RMSD.
/*!
    The loss is the expected mean RMSD between the true frame and the one the
    measurements would pick (#IMP::bff::expected_rmsd), divided by its value
    before any measurement. The degrees of freedom follow Olga: while
    choosing, `max(n + n_new - 2, 1)` for `n` measurements made and `n_new`
    added by the candidate; for the loss of what was chosen, `max(n, 1)`.
    Those are the conventions of #IMP::bff::select_probe_pairs and
    #IMP::bff::select_probe_positions, so this term alone reproduces them.
*/
class IMPBFFEXPORT ProbeResolutionTerm : public ProbeNetworkTerm {
public:
    /*!
        \param[in] predicted_measurements,n_frames,n_pairs predicted measurement per frame and pair (FRET, EPR or PRE)
        \param[in] rmsds,n_rmsd_rows,n_rmsd_cols pairwise RMSD between frames
        \param[in] measurement_error expected absolute measurement error, in the prediction units
        \param[in] diag_weight Olga's diagonal correction on the denominator
    */
    ProbeResolutionTerm(double* predicted_measurements, int n_frames, int n_pairs,
                        double* rmsds, int n_rmsd_rows, int n_rmsd_cols,
                        double measurement_error, double diag_weight = 0.99);

    int get_n_pairs() const override { return n_pairs_; }
    double get_loss() const override;
    double get_loss_with(const std::vector<int>& pairs,
                         const std::vector<int>& sites) const override;

    //! Expected mean RMSD before any measurement, the loss's unit.
    double get_initial_rmsd() const { return initial_; }
    //! Expected mean RMSD of the committed set (`get_loss() * get_initial_rmsd()`).
    double get_expected_rmsd() const;

    IMP_OBJECT_METHODS(ProbeResolutionTerm);

protected:
    void do_reset() override;
    void do_commit(const std::vector<int>& pairs,
                   const std::vector<int>& new_sites) override;

private:
    int n_frames_, n_pairs_, n_measurements_ = 0;
    double inv_err_sq_, diag_weight_, initial_ = 0.0;
    std::vector<double> e_t_;    // candidate-major predictions, n_pairs x n_frames
    std::vector<double> rmsds_;  // n_frames x n_frames
    std::vector<double> chi2_;   // n_frames x n_frames, committed
};
IMP_OBJECTS(ProbeResolutionTerm, ProbeResolutionTerms);

//! Dynamics: how well the rates of a kinetic scheme are determined.
/*!
    For each candidate pair, bursts are simulated from \p process with that
    pair's per-state distances (#IMP::bff::simulate_fret_measurement, then
    #IMP::bff::select_bursts). A #IMP::bff::FRETNetworkModel with only the
    hidden rates and the pair's state means free gives the per-burst scores
    `S`, and the Fisher information per burst `F = S^T S / n_bursts` over
    `(log k, d)`. The pair's information on the rates after its own distances
    are paid for is the Schur complement

    \f[ G_p = F_{kk} - F_{kd} F_{dd}^{-1} F_{dk}, \f]

    scaled to \p n_bursts_per_pair. Pairs are measured on separate molecules,
    so the network's information is \f$F_0 + \sum_p G_p\f$ with the prior
    precision \f$F_0 = I / \sigma_0^2\f$ on the log-rates, and the loss

    \f[ L = \left[\det F_0 / \det(F_0 + \sum_p G_p)\right]^{1/n_k} \f]

    is the geometric-mean posterior variance of the log-rates relative to the
    prior. It is 1 for a pair whose distance is the same in every state.

    The measurement template supplies dyes, instrument and Forster radius;
    each candidate replaces its state distances with a single distance per
    state. Only discrete processes are supported.
*/
class IMPBFFEXPORT ProbeKineticsTerm : public ProbeNetworkTerm {
public:
    /*!
        \param[in] process the kinetic scheme, discrete, with its rates
        \param[in] measurement template pair: dyes, instrument, Forster radius
        \param[in] state_distances,n_states,n_pairs mean donor-acceptor distance per state and candidate pair
        \param[in] options how many molecules, how long, and the seed; candidate `p` uses `seed + p`
        \param[in] max_gap,min_photons burst selection; `max_gap <= 0` keeps each molecule as one segment
        \param[in] n_bursts_per_pair the planned measurement per pair, which scales the information
        \param[in] prior_sd prior standard deviation of each log-rate (default one decade)
    */
    ProbeKineticsTerm(const FRETHiddenProcess& process,
                      const FRETMeasurement& measurement,
                      double* state_distances, int n_states, int n_pairs,
                      const FRETSimulationOptions& options = FRETSimulationOptions(),
                      double max_gap = 0.0, int min_photons = 1,
                      double n_bursts_per_pair = 1000.0,
                      double prior_sd = 2.302585092994046);

    int get_n_pairs() const override { return n_pairs_; }
    double get_loss() const override;
    double get_loss_with(const std::vector<int>& pairs,
                         const std::vector<int>& sites) const override;

    //! Names of the rates, the rows and columns of the information matrices.
    const std::vector<std::string>& get_rate_names() const { return rate_names_; }
    //! Bursts simulated for candidate \p pair.
    int get_n_bursts(int pair) const { return n_bursts_.at(pair); }
    //! Candidate \p pair's rate information \f$G_p\f$, flat `n_k x n_k`.
    std::vector<double> get_pair_information(int pair) const;
    //! Posterior standard deviation of each log-rate with the committed set.
    std::vector<double> get_rate_sigmas() const;

    IMP_OBJECT_METHODS(ProbeKineticsTerm);

protected:
    void do_reset() override;
    void do_commit(const std::vector<int>& pairs,
                   const std::vector<int>& new_sites) override;

private:
    double loss_of(const std::vector<double>& information) const;
    int n_pairs_ = 0, n_k_ = 0;
    double prior_precision_ = 1.0;
    std::vector<std::string> rate_names_;
    std::vector<int> n_bursts_;
    std::vector<std::vector<double> > g_;  // per pair, n_k x n_k
    std::vector<double> committed_;        // sum of committed G, n_k x n_k
};
IMP_OBJECTS(ProbeKineticsTerm, ProbeKineticsTerms);

//! Labelling: how well the chosen sites label, and how many mutations.
/*!
    Fed by the Labelizer's combined score per residue key,
    #IMP::bff::labelizer_combined_by_key. That score is a likelihood ratio, so
    at even prior odds a site labels with probability `LS / (1 + LS)` and
    fails with `f_s = 1 / (1 + LS)`. Each distinct site costs
    `(m + f_s) / (m + 1)` with the mutation cost `m`: a pair that reuses a
    chosen site adds nothing, and a poorly labelled site costs more than a
    good one. The loss is the summed cost over the most sites the selection
    can reach (#set_budget), capped at 1. A site whose key the scores do not
    hold (excluded, unresolved) is not eligible.
*/
class IMPBFFEXPORT ProbeLabellingTerm : public ProbeNetworkTerm {
public:
    /*!
        \param[in] label_scores combined label score by residue key, `"<chain><seq_id>"`
        \param[in] site_keys the residue key of each site index
        \param[in] mutation_cost cost of one mutation relative to a failed label
    */
    ProbeLabellingTerm(const std::map<std::string, double>& label_scores,
                       const std::vector<std::string>& site_keys,
                       double mutation_cost = 1.0);

    bool get_is_eligible_site(int site) const override;
    void set_budget(int max_units, int max_sites) override;
    double get_loss() const override;
    double get_loss_with(const std::vector<int>& pairs,
                         const std::vector<int>& sites) const override;

    //! The cost of site \p site, `(m + f_s) / (m + 1)`; infinite when not eligible.
    double get_site_cost(int site) const;
    //! The site count the loss is normalised by.
    int get_reference_sites() const { return n_reference_; }

    IMP_OBJECT_METHODS(ProbeLabellingTerm);

protected:
    void do_reset() override;
    void do_commit(const std::vector<int>& pairs,
                   const std::vector<int>& new_sites) override;

private:
    std::vector<double> cost_;  // per site; +inf when not eligible
    int n_reference_ = 1;
    double committed_ = 0.0;
};
IMP_OBJECTS(ProbeLabellingTerm, ProbeLabellingTerms);

//! Greedy selection of a probe network by a weighted sum of term losses.
/*!
    Each step adds the unit (a pair, or in site mode a labelling site with
    every pair it completes) that minimises

    \f[ J = \sum_t w_t L_t, \f]

    the weighted sum of the terms' losses. A weight of 0 turns a term off; it
    is still reported. Ties go to the lowest index, as in
    #IMP::bff::select_probe_pairs.

    Site mode needs #set_pair_sites. A pair is implied once both of its sites
    are chosen, a homotypic pair `(i, i)` as soon as `i` is (homo-oligomers,
    as #IMP::bff::select_probe_positions). In pair mode the sites, when set,
    are passed to the terms so the labelling term sees them.
*/
class IMPBFFEXPORT ProbeNetworkSelection : public IMP::Object {
public:
    //! A selector over \p n_pairs candidate pairs.
    ProbeNetworkSelection(int n_pairs, std::string name = "ProbeNetworkSelection");

    int get_n_pairs() const { return n_pairs_; }
    //! The two site indices of every candidate pair, `n_pairs x 2`.
    void set_pair_sites(int* pair_sites, int n_site_rows, int n_site_cols);
    //! Add a term with weight \p weight (>= 0). Returns its index.
    int add_term(ProbeNetworkTerm* term, double weight = 1.0);
    int get_n_terms() const { return static_cast<int>(terms_.size()); }
    ProbeNetworkTerm* get_term(int i) const { return terms_.at(i); }
    double get_weight(int i) const { return weights_.at(i); }
    void set_weight(int i, double weight);
    //! Units (pairs in pair mode, sites in site mode) never to select.
    void set_excluded(const std::vector<int>& units) { excluded_ = units; }

    //! Run the greedy selection.
    /*!
        \param[in] max_units how many units to select
        \param[in] by_sites select labelling sites (needs #set_pair_sites) instead of pairs
        \param[out] out_units,n_out_units selected units in selection order (int view)
        \param[out] out_losses,n_out_losses the weighted total loss after each (double view)
    */
    void select(int max_units, bool by_sites = false,
                int** out_units = 0, int* n_out_units = 0,
                double** out_losses = 0, int* n_out_losses = 0);

    //! Pairs measured by the last selection, in the order they joined.
    const std::vector<int>& get_selected_pairs() const { return selected_pairs_; }
    //! Each term's loss after each step of the last selection, `steps x terms`.
    void get_term_losses(double** out_matrix, int* n_out_rows, int* n_out_cols) const;

    IMP_OBJECT_METHODS(ProbeNetworkSelection);

private:
    int n_pairs_;
    std::vector<int> pair_sites_;  // n_pairs x 2, or empty
    std::vector<IMP::PointerMember<ProbeNetworkTerm> > terms_;
    std::vector<double> weights_;
    std::vector<int> excluded_;
    std::vector<int> selected_pairs_;
    std::vector<double> term_losses_;  // steps x terms
};
IMP_OBJECTS(ProbeNetworkSelection, ProbeNetworkSelections);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_PROBENETWORKSELECTION_H
