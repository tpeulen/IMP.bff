/**
 *  \file IMP/bff/SequenceHomologs.h
 *  \brief Choosing the homologues an evolutionary analysis is built on, by
 *         ConSurf's rules.
 *
 * ConSurf (Ashkenazy et al., NAR 44:W344, 2016) does not use every hit of its
 * search. It keeps a hit when it is significant, long enough, similar enough
 * to be a homologue and not so similar that it adds nothing; it removes
 * redundancy among what is left; and it caps the number, sampling across the
 * list rather than keeping only the closest. #select_sequence_homologs applies
 * the same rules to the hits of #IMP::bff::search_sequence_database.
 *
 * - **Significance**: E-value at most #SequenceHomologOptions::max_evalue.
 * - **Length**: the hit's aligned target residues at least
 *   #SequenceHomologOptions::min_length of the query's length.
 * - **Identity to the query** (identical pairs over the alignment's columns,
 *   gaps included, as BLAST and HMMER report it): at least `min_identity`,
 *   below `max_identity`.
 * - **Redundancy**: greedily, longest first (CD-HIT's order), a hit identical
 *   to a kept one in at least #SequenceHomologOptions::redundancy of the
 *   shorter one's residues is dropped. Identity between two hits is read off
 *   their alignments to the query.
 * - **Cap**: #SequenceHomologOptions::max_homologs, by E-value; `"sample"`
 *   takes them evenly spaced over the E-value-ordered list (the best and the
 *   last included), `"best"` the first ones.
 *
 * A search here aligns a target once, so ConSurf's handling of several
 * fragments of one target (kept when overlapping at most 10 %) does not arise.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SEQUENCEHOMOLOGS_H
#define IMPBFF_SEQUENCEHOMOLOGS_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/SequenceSearch.h>

#include <string>

IMPBFF_BEGIN_NAMESPACE

//! How #IMP::bff::select_sequence_homologs chooses; the defaults are ConSurf's.
class IMPBFFEXPORT SequenceHomologOptions {
public:
    double max_evalue = 1e-4;
    //! Identity to the query: at least this ...
    double min_identity = 0.35;
    //! ... and below this.
    double max_identity = 0.95;
    //! Aligned target residues, as a fraction of the query's length.
    double min_length = 0.60;
    //! Identity between two homologues at which the shorter is dropped.
    double redundancy = 0.95;
    int max_homologs = 150;
    //! Fewer than this and the alignment is too thin to analyse.
    int min_homologs = 5;
    //! `"sample"` (evenly over the E-value order) or `"best"`.
    std::string sampling = "sample";

    //! The ConSurf server's defaults (HMMER on UniRef90): the default.
    static SequenceHomologOptions consurf() { return SequenceHomologOptions(); }
    //! Stand-alone ConSurf (PSI-BLAST on Swiss-Prot): E <= 1e-3, no identity
    //! floor, the best 149 besides the query.
    static SequenceHomologOptions consurf_standalone();

    IMP_SHOWABLE_INLINE(SequenceHomologOptions,
                        out << "SequenceHomologOptions(E <= " << max_evalue << ", id "
                            << min_identity << ".." << max_identity << ", <= " << max_homologs
                            << ", " << sampling << ")");
};
IMP_VALUES(SequenceHomologOptions, SequenceHomologOptionsList);

//! The homologues of query \p query_index among \p hits, by E-value.
IMPBFFEXPORT SequenceSearchHits select_sequence_homologs(
        const SequenceSearchHits& hits,
        const SequenceHomologOptions& options = SequenceHomologOptions(),
        int query_index = 0);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_SEQUENCEHOMOLOGS_H */
