/**
 *  \file IMP/bff/SequenceMSA.h
 *  \brief A protein multiple sequence alignment, encoded once for the
 *         evolutionary analyses built on it: conservation
 *         (#IMP::bff::compute_sequence_conservation) and co-evolution
 *         (#IMP::bff::compute_sequence_coevolution).
 *
 * The alphabet is fixed at q = 21 states: 0 is a gap (or any letter that is
 * not one of the 20 amino acids), 1..20 are `ACDEFGHIKLMNPQRSTVWY`. Columns are
 * the match states of the reference sequence (by default the first): a column
 * is kept when the reference character there is not `.` and not lower case,
 * the A2M convention for insert states. Each kept column records which residue
 * of the reference sequence it holds (1-based, ungapped), or -1 where the
 * reference has a gap.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SEQUENCEMSA_H
#define IMPBFF_SEQUENCEMSA_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The 20 amino acids in state order 1..20; state 0 is the gap.
IMPBFFEXPORT std::string get_sequence_alphabet();

//! An encoded multiple sequence alignment.
class IMPBFFEXPORT SequenceMSA {
public:
    SequenceMSA() {}
    //! Encode aligned sequences.
    /*!
        \param[in] names one name per sequence
        \param[in] sequences aligned, all of the same length
        \param[in] reference index of the reference sequence
        \param[in] match_columns_only keep only the reference's match states
                   (not `.`, not lower case); false keeps every column
    */
    SequenceMSA(const std::vector<std::string>& names,
                const std::vector<std::string>& sequences, int reference = 0,
                bool match_columns_only = true);

    int get_n_sequences() const { return n_seq_; }
    int get_n_columns() const { return n_col_; }
    int get_reference() const { return reference_; }
    const std::vector<std::string>& get_names() const { return names_; }
    //! The state (0 gap, 1..20 amino acid) of sequence \p sequence in column \p column.
    int get_state(int sequence, int column) const {
        return data_[static_cast<std::size_t>(sequence) * n_col_ + column];
    }
    //! Sequence \p sequence over the kept columns, `-` for a gap.
    std::string get_sequence(int sequence) const;
    //! Per kept column, the residue number in the ungapped reference (1-based), or -1.
    const std::vector<int>& get_reference_positions() const { return ref_pos_; }
    //! The encoded states, row-major `n_sequences x n_columns`.
    const std::vector<signed char>& get_data() const { return data_; }

    IMP_SHOWABLE_INLINE(SequenceMSA, out << "SequenceMSA(" << n_seq_ << " sequences, "
                                         << n_col_ << " columns)");

private:
    int n_seq_ = 0, n_col_ = 0, reference_ = 0;
    std::vector<std::string> names_;
    std::vector<signed char> data_;
    std::vector<int> ref_pos_;
};
IMP_VALUES(SequenceMSA, SequenceMSAs);

//! Read an aligned FASTA or A2M file; sequence lines may be wrapped.
/*! \param[in] path the alignment
    \param[in] reference index of the reference sequence
    \param[in] match_columns_only as in the #SequenceMSA constructor
    \throw IOException when the file cannot be read, ValueException when the
           sequences differ in length */
IMPBFFEXPORT SequenceMSA read_sequence_msa(const std::string& path, int reference = 0,
                                           bool match_columns_only = true);

//! Sequence weights against redundancy.
/*!
    `w_m = 1 / (1 + #{n != m : d(m, n) < theta})`, with `d` the fraction of
    kept columns in which two sequences differ (a gap is a state). Sequences
    closer than `theta` share their weight; `theta <= 0` gives every sequence
    weight 1 (Morcos et al., PNAS 108:E1293, 2011).
*/
IMPBFFEXPORT std::vector<double> get_sequence_weights(const SequenceMSA& msa,
                                                      double theta = 0.2);

//! The amino acids present in column \p column, in alphabet order.
IMPBFFEXPORT std::string get_residue_variety(const SequenceMSA& msa, int column);

//! Sequences with an amino acid (not a gap) in column \p column.
IMPBFFEXPORT int get_n_with_data(const SequenceMSA& msa, int column);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_SEQUENCEMSA_H
