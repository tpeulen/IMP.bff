/**
 *  \file IMP/bff/SequenceCoevolution.h
 *  \brief Co-evolution of alignment columns by mean-field direct coupling
 *         analysis (DCA).
 *
 * Mean-field DCA after Morcos et al., PNAS 108:E1293 (2011): sequence
 * reweighting at an identity threshold, single- and pair-site frequencies
 * with a pseudocount, the connected correlation matrix over `q - 1` states
 * per column (the last state is the gauge), its inverse as the coupling
 * matrix, and per pair of columns the direct information of the two-site
 * model that carries only their direct coupling. Mutual information is
 * reported beside it.
 *
 * The conventions -- q = 21 with the gap as state 0 and `Y` as the dropped
 * gauge state, pseudocount weight 0.5, fixed-point tolerance 1e-4 -- are
 * those of the mean-field DCA scripts FRETNet-Designer ships, so the numbers
 * are comparable with that pipeline (pinned by an A/B test). This is an
 * independent implementation: the covariance is built from the encoded
 * alignment directly, never as an `N x N x q x q` array.
 *
 * Memory: the correlation matrix and its inverse are `N (q-1)` square, twice
 * about 235 MB at N = 271 columns and 0.8 GB at N = 500.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SEQUENCECOEVOLUTION_H
#define IMPBFF_SEQUENCECOEVOLUTION_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/SequenceMSA.h>

#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Direct and mutual information between every pair of alignment columns.
class IMPBFFEXPORT SequenceCoevolution {
public:
    SequenceCoevolution() {}
    //! From computed tables (as #IMP::bff::compute_sequence_coevolution builds them).
    SequenceCoevolution(int n_columns, double n_effective,
                        const std::vector<int>& reference_positions,
                        const std::vector<double>& direct_information,
                        const std::vector<double>& mutual_information);

    int get_n_columns() const { return n_; }
    //! Effective number of sequences after reweighting.
    double get_n_effective() const { return m_eff_; }
    //! Per column, the residue number in the ungapped reference, or -1.
    const std::vector<int>& get_reference_positions() const { return ref_pos_; }
    //! Direct information of columns \p i and \p j (0 on the diagonal).
    double get_direct_information(int i, int j) const { return di_.at(index(i, j)); }
    //! Mutual information of columns \p i and \p j (0 on the diagonal).
    double get_mutual_information(int i, int j) const { return mi_.at(index(i, j)); }
    //! All direct information, `n_columns x n_columns`.
    void get_direct_information_matrix(double** out_matrix, int* n_out_rows,
                                       int* n_out_cols) const;
    //! All mutual information, `n_columns x n_columns`.
    void get_mutual_information_matrix(double** out_matrix, int* n_out_rows,
                                       int* n_out_cols) const;

    IMP_SHOWABLE_INLINE(SequenceCoevolution,
                        out << "SequenceCoevolution(" << n_ << " columns, Meff " << m_eff_ << ")");

private:
    std::size_t index(int i, int j) const {
        return static_cast<std::size_t>(i) * n_ + j;
    }
    int n_ = 0;
    double m_eff_ = 0.0;
    std::vector<int> ref_pos_;
    std::vector<double> di_, mi_;  // n x n, symmetric
};
IMP_VALUES(SequenceCoevolution, SequenceCoevolutions);

//! Mean-field DCA of an alignment.
/*!
    \param[in] msa the encoded alignment
    \param[in] theta identity threshold of the reweighting (fraction of differing columns)
    \param[in] pseudocount pseudocount weight, 0..1
*/
IMPBFFEXPORT SequenceCoevolution compute_sequence_coevolution(const SequenceMSA& msa,
                                                              double theta = 0.2,
                                                              double pseudocount = 0.5);

//! Direct information for pairs of structure residues.
/*!
    Maps each residue to the alignment column whose reference residue is
    `residue - residue_offset`, and returns the direct information of the two
    columns; NaN when either residue has no column (a
    #IMP::bff::ProbePairCostTerm treats NaN as ineligible). The alignment's
    reference must be the structure's sequence; `residue_offset` is the
    structure's number of the residue before the first one of the reference.

    \param[in] coevolution the analysis
    \param[in] pair_residues,n_pair_rows,n_pair_cols two structure residue numbers per pair
    \param[in] residue_offset structure numbering minus reference numbering
*/
IMPBFFEXPORT std::vector<double> probe_pair_coevolution(
        const SequenceCoevolution& coevolution, int* pair_residues, int n_pair_rows,
        int n_pair_cols, int residue_offset = 0);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_SEQUENCECOEVOLUTION_H
