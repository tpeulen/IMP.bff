/**
 *  \file IMP/bff/Consurf.h
 *  \brief ConSurf end to end, native: homologues from a sequence database,
 *         their alignment, per-site rates and the nine conservation grades.
 *
 * What the ConSurf server does for a chain (Ashkenazy et al., NAR 44:W344,
 * 2016), in one process and without external programs:
 *
 * 1. search a sequence database (#IMP::bff::search_sequence_database;
 *    the server uses HMMER on UniRef90),
 * 2. choose the homologues by ConSurf's rules
 *    (#IMP::bff::select_sequence_homologs),
 * 3. align them to the query (the search's query-anchored alignment; the
 *    server realigns with MAFFT),
 * 4. per-site rates by Rate4Site's method with ConSurf's settings
 *    (#IMP::bff::compute_sequence_conservation), and the grades
 *    (#IMP::bff::get_consurf_grades).
 *
 * The database is #ConsurfOptions::database, a `.pto` path or a name from the
 * settings (#IMP::bff::get_sequence_search_settings), else the settings'
 * default database. Several queries share one pass over the database. With
 * no database at all, a server from the settings supplies each query's
 * alignment instead (#IMP::bff::fetch_server_msa), and the homologues are
 * chosen from its rows by the same rules.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_CONSURF_H
#define IMPBFF_CONSURF_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/SequenceConservation.h>
#include <IMP/bff/SequenceClusters.h>
#include <IMP/bff/SequenceHomologs.h>
#include <IMP/bff/SequenceMSA.h>
#include <IMP/bff/SequenceSearch.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The settings of each step of #IMP::bff::compute_consurf.
class IMPBFFEXPORT ConsurfOptions {
public:
    //! A `.pto` path or a database name from the settings; empty: the
    //! settings' default database.
    std::string database;
    //! A server name from the settings, asked when there is no database
    //! (neither given nor the settings' default); empty: the settings'
    //! fallback_server. Its alignment's rows are the hits the homologues
    //! are chosen from.
    std::string server;
    //! Cluster representatives of the database (a `.pto` or settings name):
    //! search in two stages; empty: the settings' `clusters` entry of a
    //! database named there, else one stage.
    std::string representatives;
    SequenceSearchOptions search;
    //! The two-stage search of a clustered database: reach of the first stage,
    //! and the optional early stop of the second (min_homologues).
    SequenceClusterSearchOptions clusters;
    SequenceHomologOptions homologs;
    SequenceConservationOptions conservation;

    IMP_SHOWABLE_INLINE(ConsurfOptions, out << "ConsurfOptions(" << database << ")");
};
IMP_VALUES(ConsurfOptions, ConsurfOptionsList);

//! ConSurf's result for one query.
class IMPBFFEXPORT ConsurfResult {
public:
    std::string query;
    //! `"ok"`, or why there is no result (too few homologues).
    std::string status;
    SequenceSearchHits homologs;
    //! The query first, then the homologues, on the query's columns.
    SequenceMSA msa;
    SequenceConservation conservation;
    //! #IMP::bff::get_consurf_grades of the conservation: `n x 4`.
    std::vector<int> grades;

    bool get_is_ok() const { return status == "ok"; }

    IMP_SHOWABLE_INLINE(ConsurfResult, out << "ConsurfResult(" << query.size() << " residues, "
                                           << homologs.size() << " homologues, " << status
                                           << ")");
};
IMP_VALUES(ConsurfResult, ConsurfResults);

//! ConSurf for each of \p queries (protein sequences), in one database pass.
/*! \throw IOException when neither a database nor a server is available */
IMPBFFEXPORT ConsurfResults compute_consurf(const Strings& queries,
                                            const ConsurfOptions& options = ConsurfOptions());

//! ConSurf from an alignment already made (its first sequence the query).
IMPBFFEXPORT ConsurfResult compute_consurf_from_msa(
        const SequenceMSA& msa,
        const SequenceConservationOptions& options = SequenceConservationOptions());

//! The result in ConSurf's `.grades` layout.
/*! \param[in] atom_labels per query residue, its structure label as ConSurf
               writes it (`"GLY1:A"`), or empty for a residue the structure
               lacks */
IMPBFFEXPORT std::string get_consurf_grades_text(const ConsurfResult& result,
                                                 const Strings& atom_labels = Strings());

IMPBFFEXPORT void write_consurf_grades(const ConsurfResult& result, const std::string& path,
                                       const Strings& atom_labels = Strings());

//! The alignment as aligned FASTA, the query first.
IMPBFFEXPORT void write_consurf_msa(const ConsurfResult& result, const std::string& path);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_CONSURF_H */
