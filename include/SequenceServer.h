/**
 *  \file IMP/bff/SequenceServer.h
 *  \brief Alignments from a remote MSA server, when no local sequence
 *         database is configured.
 *
 * The server and how to reach it come from the settings
 * (#IMP::bff::SequenceSearchServer): nothing about it is compiled in. The one
 * protocol spoken is `"colabfold-v1"`, the MMseqs2 API ColabFold's servers
 * offer (Mirdita et al., Nat Methods 19:679, 2022): the query is submitted as
 * a ticket, the ticket polled until its search completes, and the result
 * fetched as a gzip-compressed tar archive whose `uniref.a3m` is the
 * alignment. The archive is unpacked here (zlib, and a tar reader of a few
 * lines); the network needs libcurl at build time
 * (#get_sequence_server_available).
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SEQUENCESERVER_H
#define IMPBFF_SEQUENCESERVER_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/BffSettings.h>
#include <IMP/bff/SequenceSearch.h>

#include <string>

IMPBFF_BEGIN_NAMESPACE

//! Whether this build can reach an MSA server (libcurl and zlib).
IMPBFFEXPORT bool get_sequence_server_available();

//! The A3M alignment \p server returns for \p query.
/*! \param[in] mode the server's search mode; for colabfold-v1 `"all"`
               (UniRef) or `"env"` (UniRef and environmental sequences)
    \throw IOException when the server cannot be reached, refuses the query
           or does not finish within its timeout */
IMPBFFEXPORT std::string fetch_server_msa(const std::string& query,
                                          const SequenceSearchServer& server,
                                          const std::string& mode = "all");

//! Hit records of an A3M alignment's rows, for #IMP::bff::select_sequence_homologs.
/*! The first row is the query. For each other row: its match states as
    #SequenceSearchHit::aligned, the aligned range on the query, its identity
    (identical pairs over the alignment's columns, insertions and deletions
    included), and the E-value when the row's header carries MMseqs2's fields
    (`name score identity evalue qstart qend qlen tstart tend tlen`), else 0. */
IMPBFFEXPORT SequenceSearchHits get_a3m_hits(const std::string& a3m, int query_index = 0);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_SEQUENCESERVER_H */
