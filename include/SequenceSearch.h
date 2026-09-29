/**
 *  \file IMP/bff/SequenceSearch.h
 *  \brief Protein homology search of a #IMP::bff::SequenceDatabase, and the
 *         query-anchored alignment of its hits.
 *
 * The method follows the published description of MMseqs2 (Steinegger &
 * Soding, Nat Biotechnol 35:1026, 2017), implemented independently:
 *
 * - **Prefilter.** Each query position contributes its k-mer and every k-mer
 *   scoring at least #SequenceSearchOptions::kmer_threshold against it under
 *   BLOSUM62 (its neighbourhood). The database streams past once; a target
 *   k-mer found in a query's neighbourhood is a hit on the diagonal
 *   `target position - query position`, and two hits on one diagonal score
 *   that diagonal without gaps. The targets with the best ungapped diagonal,
 *   at most #SequenceSearchOptions::max_candidates per query, go on.
 * - **Alignment.** Smith-Waterman with affine gaps (Gotoh), BLOSUM62, gap
 *   open 11, extend 1: a gap of length L costs 11 + L.
 * - **Significance.** Karlin-Altschul E-values for that scoring, lambda 0.267
 *   and K 0.041 (Altschul et al., NAR 25:3389, 1997), over the database's
 *   residues.
 *
 * The database is read segment by segment from its memory map across
 * threads; memory is the query tables, one diagonal array per thread and the
 * candidates, whatever the database's size.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SEQUENCESEARCH_H
#define IMPBFF_SEQUENCESEARCH_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/SequenceDatabase.h>
#include <IMP/bff/SequenceMSA.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! How #IMP::bff::search_sequence_database searches.
class IMPBFFEXPORT SequenceSearchOptions {
public:
    //! Neighbourhood: k-mers scoring at least this against a query k-mer (5
    //! residues, BLOSUM62) are hits. Lower is more sensitive and slower.
    int kmer_threshold = 18;
    //! Targets aligned per query, the best by ungapped diagonal score.
    int max_candidates = 1000;
    //! Ungapped diagonal score a target needs to be aligned, in bits.
    double min_ungapped_bits = 15.0;
    //! Hits reported: E-value at most this.
    double max_evalue = 1e-3;
    int gap_open = 11;
    int gap_extend = 1;
    //! Threads; 0 uses every core.
    int threads = 0;

    IMP_SHOWABLE_INLINE(SequenceSearchOptions,
                        out << "SequenceSearchOptions(T " << kmer_threshold << ", "
                            << max_candidates << " candidates, E <= " << max_evalue << ")");
};
IMP_VALUES(SequenceSearchOptions, SequenceSearchOptionsList);

//! One aligned target.
class IMPBFFEXPORT SequenceSearchHit {
public:
    int query = 0;               //!< index of the query
    std::size_t target = 0;      //!< index in the database
    std::string identifier;      //!< the target's accession
    int score = 0;               //!< raw Smith-Waterman score
    double bits = 0.0;
    double evalue = 0.0;
    //! Identical pairs over alignment columns (pairs and gaps).
    double identity = 0.0;
    //! The aligned ranges, 0-based half-open.
    int query_start = 0, query_end = 0, target_start = 0, target_end = 0;
    int query_length = 0, target_length = 0;
    //! The target on the query's columns: one character per query residue,
    //! `-` where it has none; its insertions are dropped (A3M's match states).
    std::string aligned;

    double get_query_coverage() const {
        return query_length > 0 ? double(query_end - query_start) / query_length : 0.0;
    }
    double get_target_coverage() const {
        return target_length > 0 ? double(target_end - target_start) / target_length : 0.0;
    }

    IMP_SHOWABLE_INLINE(SequenceSearchHit,
                        out << "SequenceSearchHit(" << identifier << ", E " << evalue
                            << ", " << bits << " bits, id " << identity << ")");
};
IMP_VALUES(SequenceSearchHit, SequenceSearchHits);

//! Search \p database with each of \p queries.
/*!
    \param[in] queries protein sequences (one-letter codes)
    \param[in] database the database
    \return the hits of every query, by query and then E-value
*/
IMPBFFEXPORT SequenceSearchHits search_sequence_database(
        const Strings& queries, const SequenceDatabase& database,
        const SequenceSearchOptions& options = SequenceSearchOptions());

#ifndef SWIG
//! The search over \p rows of \p database only (sorted, distinct): the
//! second stage of a clustered search. E-values are against \p residues
//! (0: the whole database's), so they compare with a full search's.
IMPBFFEXPORT SequenceSearchHits search_sequence_database_rows(
        const Strings& queries, const SequenceDatabase& database,
        const std::vector<std::size_t>& rows,
        const SequenceSearchOptions& options = SequenceSearchOptions(), double residues = 0);
#endif

#ifndef SWIG
//! The alignment of \p member to \p reference as 'M'/'I'/'D' operations
//! (reference ends skipped as leading and trailing 'D', the member's
//! unaligned ends as 'I'), for storing a member against its cluster's
//! representative. A banded local alignment (BLOSUM62, 11/1) around the
//! diagonal the length difference suggests, \p band columns either side of
//! it; identical sequences take no alignment. Empty when nothing aligns.
IMPBFFEXPORT std::string align_to_reference(const unsigned char* reference, std::size_t n,
                                            const unsigned char* member, std::size_t m,
                                            int band = 24);
#endif

//! Align \p query to \p target: the same scoring and hit record as the search.
IMPBFFEXPORT SequenceSearchHit align_sequences(
        const std::string& query, const std::string& target,
        const SequenceSearchOptions& options = SequenceSearchOptions());

//! The query-anchored alignment of \p query and the hits of query \p query_index:
//! the query first, then each hit's #SequenceSearchHit::aligned.
IMPBFFEXPORT SequenceMSA get_query_msa(const std::string& query, const SequenceSearchHits& hits,
                                       int query_index = 0);

//! BLOSUM62 score of two one-letter codes (X or any other letter: -1).
IMPBFFEXPORT int get_blosum62(char a, char b);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_SEQUENCESEARCH_H */
