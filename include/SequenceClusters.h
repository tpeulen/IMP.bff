/**
 *  \file IMP/bff/SequenceClusters.h
 *  \brief A two-stage search: cluster representatives first, then only the
 *         members of the clusters that hit.
 *
 * UniRef is hierarchical: every UniRef90 cluster lies in one UniRef50 cluster,
 * whose representative is at least 50 % identical to its members. A homologue
 * ConSurf would keep (at least 35 % identical to the query) therefore has a
 * representative the query also finds, searched with looser cut-offs. The
 * clustered search reads the representatives (UniRef50: about a third of
 * UniRef90's residues), then only the members of the clusters found, in row
 * order, and reports hits as a search of the members would.
 *
 * #create_sequence_clusters records the membership, from UniProt's ID mapping
 * (`idmapping_selected.tab.gz`: per UniProtKB accession its UniRef90 and
 * UniRef50 cluster), as a `members` object in the representatives' `.pto`:
 * per representative the member rows, and the orphans -- members the mapping
 * does not place (clusters of UniParc sequences only), which the second
 * stage always searches. The join runs in hashed buckets on disk, so memory
 * stays bounded whatever the size.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SEQUENCECLUSTERS_H
#define IMPBFF_SEQUENCECLUSTERS_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/SequenceDatabase.h>
#include <IMP/bff/SequenceSearch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pto { class StoreReader; }

IMPBFF_BEGIN_NAMESPACE

//! Record which rows of \p members belong to each sequence of \p representatives.
/*!
    \param[in] members the member database (e.g. UniRef90), a `.pto` or a
               settings name
    \param[in] representatives the representatives' database (e.g. UniRef50);
               the `members` object is added to its container
    \param[in] mapping (`.gz` with zlib) either UniRef's XML of the
               representatives' level (`uniref50.xml.gz`: every member, UniParc
               ones included, with its id at the members' level, e.g. its
               UniRef90 ID for UniRef90 members -- a path containing `.xml`),
               or a tab-separated table, one row per accession, the member
               cluster id in column \p member_column and the representative
               cluster id in \p representative_column (1-based; UniProt's
               idmapping_selected: 9 and 10). The table lacks the clusters of
               UniParc sequences only (a fifth of UniRef90), which then become
               orphans that every search reads; the XML places them.
    \param[in] temporary a directory for the bucket files; empty: next to the
               representatives
    \return the number of members placed in a cluster
*/
IMPBFFEXPORT std::size_t create_sequence_clusters(const std::string& members,
                                                  const std::string& representatives,
                                                  const std::string& mapping,
                                                  int member_column = 9,
                                                  int representative_column = 10,
                                                  const std::string& temporary = "");

//! One database of both levels, cluster-ordered: the representatives as its
//! sequences, each followed (as its members, #SequenceDatabase::get_members_database)
//! by the member sequences of its cluster, stored together.
/*! A two-stage search of it reads the representatives, then each hit
    cluster's members as one contiguous run, instead of scattered rows of a
    separate member database; and the representatives need no file of their
    own. Built in two streaming passes: \p members into bucket files by
    cluster (in \p temporary, next to \p out by default), then cluster by
    cluster in the representatives' order. Members the membership does not
    place go under a last, empty representative named `orphans`.
    \param[in] members the member database (e.g. UniRef90)
    \param[in] representatives the representatives, with their membership
               (#create_sequence_clusters)
    \return the number of members written */
IMPBFFEXPORT std::size_t create_clustered_sequence_database(
        const std::string& members, const std::string& representatives, const std::string& out,
        const std::string& temporary = "");

//! The membership recorded by #create_sequence_clusters.
class IMPBFFEXPORT SequenceClusters {
public:
    SequenceClusters() {}
    //! Open the membership in \p representatives (a `.pto` or a settings name).
    explicit SequenceClusters(const std::string& representatives);

    std::size_t get_number_of_clusters() const { return n_clusters_; }
    std::size_t get_number_of_orphans() const { return n_orphans_; }
    //! The member rows of cluster \p k.
    Ints get_members(std::size_t k) const;
    Ints get_orphans() const;
#ifndef SWIG
    std::vector<std::size_t> get_member_rows(std::size_t k) const;
    std::vector<std::size_t> get_orphan_rows() const;
#endif

    IMP_SHOWABLE_INLINE(SequenceClusters, out << "SequenceClusters(" << n_clusters_
                                              << " clusters, " << n_orphans_ << " orphans)");

private:
    std::shared_ptr<pto::StoreReader> reader_;
    std::size_t n_clusters_ = 0, n_orphans_ = 0;
};
IMP_VALUES(SequenceClusters, SequenceClustersList);

//! How far the first stage reaches: it must find the representative of every
//! homologue the second stage would report.
class IMPBFFEXPORT SequenceClusterSearchOptions {
public:
    //! Representatives kept per query (#SequenceSearchOptions::max_candidates
    //! of the first stage).
    int max_representatives = 5000;
    //! E-value a representative needs, against the representatives' database.
    double max_representative_evalue = 10.0;
    //! Always search the orphans in the second stage.
    bool search_orphans = true;

    IMP_SHOWABLE_INLINE(SequenceClusterSearchOptions,
                        out << "SequenceClusterSearchOptions(" << max_representatives << ", E <= "
                            << max_representative_evalue << ")");
};
IMP_VALUES(SequenceClusterSearchOptions, SequenceClusterSearchOptionsList);

//! Search \p members in two stages through \p representatives.
/*! The hits are those of the members (#SequenceSearchHit::target a row of
    \p members), with E-values against all of \p members' residues. */
IMPBFFEXPORT SequenceSearchHits search_clustered_sequence_database(
        const Strings& queries, const SequenceDatabase& representatives,
        const SequenceClusters& clusters, const SequenceDatabase& members,
        const SequenceSearchOptions& options = SequenceSearchOptions(),
        const SequenceClusterSearchOptions& cluster_options = SequenceClusterSearchOptions());

//! Search a clustered database (#create_clustered_sequence_database) in two
//! stages: its representatives, then the members of the clusters found.
/*! Hits are rows of \p database's members level, E-values against all members. */
IMPBFFEXPORT SequenceSearchHits search_clustered_sequence_database(
        const Strings& queries, const SequenceDatabase& database,
        const SequenceSearchOptions& options = SequenceSearchOptions(),
        const SequenceClusterSearchOptions& cluster_options = SequenceClusterSearchOptions());

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_SEQUENCECLUSTERS_H */
