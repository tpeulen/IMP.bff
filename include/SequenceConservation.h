/**
 *  \file IMP/bff/SequenceConservation.h
 *  \brief Per-site evolutionary rates of an alignment, and ConSurf's
 *         conservation scores and grades from them.
 *
 * The method of Rate4Site (Pupko et al., Bioinformatics 18:S71, 2002;
 * Mayrose et al., MBE 21:1781, 2004), implemented independently:
 *
 * - **Model.** JTT (Jones, Taylor & Thornton, CABIOS 8:275, 1992) with its
 *   own equilibrium frequencies, scaled to one expected substitution per unit
 *   branch length; gaps and unknown residues are missing data.
 * - **Rate heterogeneity.** A discrete gamma with K equally probable
 *   categories at their means (Yang, J Mol Evol 39:306, 1994).
 * - **Tree.** Neighbour joining (Saitou & Nei 1987) on pairwise
 *   maximum-likelihood JTT distances, or a given Newick tree.
 * - **Fit.** The gamma shape and the branch lengths by maximum likelihood,
 *   branch lengths by expectation-maximisation over expected transitions.
 * - **Rates.** Per site of the reference sequence, the posterior mean rate
 *   under the gamma prior (empirical Bayes), its standard deviation and the
 *   25-75 % interval of the discrete posterior; or the maximum-likelihood rate.
 * - **Scores.** Rates normalised to mean 0, standard deviation 1: low means
 *   conserved.
 *
 * The defaults, tolerances, bounds and conventions of Rate4Site 3.0 are
 * reproduced (checked against the program in A/B tests). The default options
 * are what ConSurf runs (#SequenceConservationOptions::consurf: no branch
 * length refit), which is also the fast setting;
 * #SequenceConservationOptions::rate4site gives the program's own defaults.
 *
 * ConSurf's nine grades (Ashkenazy et al., NAR 44:W344, 2016) are computed
 * by #IMP::bff::get_consurf_grades from the normalised scores.
 *
 * Unlike Rate4Site the likelihood is scaled against underflow, so large
 * alignments stay finite.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SEQUENCECONSERVATION_H
#define IMPBFF_SEQUENCECONSERVATION_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/SequenceMSA.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! How #IMP::bff::compute_sequence_conservation fits and infers.
class IMPBFFEXPORT SequenceConservationOptions {
public:
    //! Gamma categories while the shape (and branch lengths) are fitted (`-k`).
    int n_categories = 16;
    //! Gamma categories of the posterior (`-n`).
    int n_posterior_categories = 16;
    //! Branch lengths: `"none"` (neighbour-joining lengths, `-bn`, the default),
    //! `"gamma"` (EM under the gamma model, `-bg`) or `"homogeneous"` (EM
    //! without gamma, `-bh`).
    std::string branch_lengths = "none";
    //! Joint shape/branch-length rounds under `"gamma"` (Rate4Site: 2).
    int max_rounds = 2;
    //! Rates: `"bayes"` (posterior mean, `-ib`) or `"ml"` (`-im`).
    std::string rate_inference = "bayes";
    //! A Newick tree to use instead of neighbour joining (`-t`).
    std::string tree;
    //! A fixed gamma shape (`-d`); 0 estimates it.
    double alpha = 0.0;

    //! What ConSurf runs (`-ib -zn -Mj -bn`): the default. The Labelizer's
    //! conservation table was fitted on scores made this way, and it is fast.
    static SequenceConservationOptions consurf() { return SequenceConservationOptions(); }
    //! Rate4Site's own defaults (`-bg`): branch lengths refitted under gamma,
    //! two rounds; slower, a slightly better tree.
    static SequenceConservationOptions rate4site();

    IMP_SHOWABLE_INLINE(SequenceConservationOptions,
                        out << "SequenceConservationOptions(k " << n_categories << ", n "
                            << n_posterior_categories << ", " << branch_lengths << ", "
                            << rate_inference << ")");
};
IMP_VALUES(SequenceConservationOptions, SequenceConservationOptionsList);

//! Per-site rates and conservation scores of the reference sequence.
/*!
    One entry per column in which the reference has a residue, in alignment
    order. Scores and interval bounds are normalised (mean 0, sample standard
    deviation 1 over these positions); the standard deviation is divided by
    the same scale. The `raw` getters return the rates before normalisation.
*/
class IMPBFFEXPORT SequenceConservation {
public:
    SequenceConservation() {}

    int get_n_positions() const { return static_cast<int>(score_.size()); }
    //! Alignment column of each position.
    const std::vector<int>& get_columns() const { return column_; }
    //! The reference residue at each position, one letter.
    const std::string& get_residues() const { return residues_; }
    const std::vector<double>& get_scores() const { return score_; }
    const std::vector<double>& get_lower() const { return lower_; }
    const std::vector<double>& get_upper() const { return upper_; }
    const std::vector<double>& get_std() const { return std_; }
    const std::vector<double>& get_raw_rates() const { return raw_; }
    //! Sequences with an amino acid at each position.
    const std::vector<int>& get_n_data() const { return n_data_; }
    int get_n_sequences() const { return n_seq_; }
    double get_alpha() const { return alpha_; }
    double get_log_likelihood() const { return log_likelihood_; }
    //! Mean and sample standard deviation of the raw rates.
    double get_raw_mean() const { return mean_; }
    double get_raw_sd() const { return sd_; }
    //! The fitted tree, Newick with sequence names.
    const std::string& get_tree() const { return tree_; }
    //! The table in Rate4Site's `r4s.res` layout (normalised).
    std::string get_rate4site_table() const;

    IMP_SHOWABLE_INLINE(SequenceConservation,
                        out << "SequenceConservation(" << score_.size() << " positions, alpha "
                            << alpha_ << ")");

private:
    friend class SequenceConservationBuilder;
    std::vector<int> column_, n_data_;
    std::string residues_;
    std::vector<double> score_, lower_, upper_, std_, raw_;
    int n_seq_ = 0, n_categories_ = 0;
    bool bayes_ = true;
    double alpha_ = 0.0, log_likelihood_ = 0.0, mean_ = 0.0, sd_ = 1.0;
    std::string tree_;
};
IMP_VALUES(SequenceConservation, SequenceConservations);

//! Per-site rates of the alignment's reference sequence (Rate4Site's method).
/*! \param[in] msa the alignment; build it with `match_columns_only = false`
               to use every column, as Rate4Site does
    \param[in] options fitting and inference settings */
IMPBFFEXPORT SequenceConservation compute_sequence_conservation(
        const SequenceMSA& msa,
        const SequenceConservationOptions& options = SequenceConservationOptions());

//! ConSurf's grades of every position: `n_positions x 4`, row-major.
/*!
    Per position: the grade 1 (variable) .. 9 (conserved), the grades of the
    interval's lower and upper bound, and 1 when the estimate is below
    ConSurf's confidence cut-off (the interval spans more than three grades,
    or at most five sequences have an amino acid there), else 0.

    The bins: the lowest normalised score `m` and `-m` span nine equal bins
    (width `|m| / 4.5`); a score at or above `-m` is grade 1.
*/
IMPBFFEXPORT std::vector<int> get_consurf_grades(const SequenceConservation& conservation);

//! The same, from normalised scores, interval bounds and data counts.
IMPBFFEXPORT std::vector<int> get_consurf_grades(const std::vector<double>& scores,
                                                 const std::vector<double>& lower,
                                                 const std::vector<double>& upper,
                                                 const std::vector<int>& n_data);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_SEQUENCECONSERVATION_H
