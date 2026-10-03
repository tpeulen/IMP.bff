/**
 * \file Consurf.cpp
 * \brief ConSurf end to end: search, homologues, alignment, rates, grades.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/Consurf.h>
#include <IMP/bff/BffSettings.h>
#include <IMP/bff/SequenceClusters.h>
#include <IMP/bff/SequenceDatabase.h>
#include <IMP/bff/SequenceServer.h>

#include <cstdio>
#include <fstream>
#include <algorithm>
#include <map>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

namespace {

std::string clean_query(const std::string& s) {
    std::string out;
    for (char c : s)
        if (std::isalpha(static_cast<unsigned char>(c)))
            out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    return out;
}

//! Where the hits come from: a database path, or a server; one of the two.
struct ConsurfSource {
    std::string database, representatives;
    SequenceSearchServer server;
    bool use_server = false;
};

ConsurfSource consurf_source(const ConsurfOptions& options) {
    ConsurfSource source;
    const SequenceSearchSettings settings = get_sequence_search_settings();
    const std::string named = !options.database.empty() ? options.database : settings.default_database;
    if (!named.empty()) {
        source.database = named;
        source.representatives = !options.representatives.empty()
                                         ? options.representatives
                                         : settings.get_representatives(named);
        return source;
    }
    const std::string server = options.server.empty() ? settings.fallback_server : options.server;
    if (server.empty())
        IMP_THROW("compute_consurf: no database given, and the settings ("
                  << get_settings_path() << ") name neither a default_database nor a "
                  "fallback_server", IOException);
    source.server = settings.get_server(server);
    source.use_server = true;
    return source;
}

}  // namespace

ConsurfResult compute_consurf_from_msa(const SequenceMSA& msa,
                                       const SequenceConservationOptions& options) {
    ConsurfResult r;
    r.query = msa.get_sequence(0);
    r.msa = msa;
    r.conservation = compute_sequence_conservation(msa, options);
    r.grades = get_consurf_grades(r.conservation);
    r.status = "ok";
    return r;
}

ConsurfResults compute_consurf(const Strings& queries, const ConsurfOptions& options) {
    // Identical chains are one query.
    std::vector<std::string> unique;
    std::map<std::string, int> index_of;
    std::vector<int> which;
    for (const std::string& q : queries) {
        const std::string s = clean_query(q);
        auto it = index_of.find(s);
        if (it == index_of.end()) {
            it = index_of.emplace(s, static_cast<int>(unique.size())).first;
            unique.push_back(s);
        }
        which.push_back(it->second);
    }
    ConsurfResults per_unique(unique.size());
    if (!unique.empty()) {
        const ConsurfSource source = consurf_source(options);
        SequenceSearchHits hits;
        if (source.use_server) {
            for (std::size_t k = 0; k < unique.size(); ++k) {
                const SequenceSearchHits rows = get_a3m_hits(
                        fetch_server_msa(unique[k], source.server), static_cast<int>(k));
                hits.insert(hits.end(), rows.begin(), rows.end());
            }
        } else {
            const SequenceDatabase database(source.database);
            SequenceSearchOptions search = options.search;
            search.max_evalue = std::max(search.max_evalue, options.homologs.max_evalue);
            auto run = [&](const Strings& queries, const SequenceSearchOptions& so) {
                if (database.get_has_members())   // one clustered store: representatives, then members
                    return search_clustered_sequence_database(queries, database, so, options.clusters);
                if (source.representatives.empty()) return search_sequence_database(queries, database, so);
                return search_clustered_sequence_database(
                        queries, SequenceDatabase(source.representatives),
                        SequenceClusters(source.representatives), database, so, options.clusters);
            };
            hits = run(Strings(unique.begin(), unique.end()), search);
            // A query whose candidates filled the cap and were mostly near-identical
            // (ConSurf drops those) is searched again with a larger cap.
            if (options.retry_max_candidates > search.max_candidates) {
                std::vector<int> retry;
                for (std::size_t k = 0; k < unique.size(); ++k) {
                    int n = 0, near_identical = 0;
                    for (const SequenceSearchHit& h : hits) {
                        if (h.query != static_cast<int>(k)) continue;
                        ++n;
                        if (h.identity >= options.homologs.max_identity) ++near_identical;
                    }
                    const int kept = static_cast<int>(
                            select_sequence_homologs(hits, options.homologs, static_cast<int>(k)).size());
                    if (kept < options.homologs.min_homologs && n >= search.max_candidates &&
                        2 * near_identical >= n)
                        retry.push_back(static_cast<int>(k));
                }
                if (!retry.empty()) {
                    Strings again;
                    for (int k : retry) again.push_back(unique[static_cast<std::size_t>(k)]);
                    SequenceSearchOptions wider = search;
                    wider.max_candidates = options.retry_max_candidates;
                    SequenceSearchHits more = run(again, wider);
                    SequenceSearchHits merged;
                    for (const SequenceSearchHit& h : hits)
                        if (std::find(retry.begin(), retry.end(), h.query) == retry.end()) merged.push_back(h);
                    for (SequenceSearchHit h : more) {
                        h.query = retry[static_cast<std::size_t>(h.query)];
                        merged.push_back(h);
                    }
                    hits.swap(merged);
                }
            }
        }
        for (std::size_t k = 0; k < unique.size(); ++k) {
            ConsurfResult& r = per_unique[k];
            r.query = unique[k];
            r.homologs = select_sequence_homologs(hits, options.homologs, static_cast<int>(k));
            if (static_cast<int>(r.homologs.size()) < options.homologs.min_homologs) {
                std::ostringstream why;
                why << "too few homologues: " << r.homologs.size() << " of the "
                    << options.homologs.min_homologs << " needed";
                r.status = why.str();
                continue;
            }
            r.msa = get_query_msa(unique[k], r.homologs, static_cast<int>(k));
            r.conservation = compute_sequence_conservation(r.msa, options.conservation);
            r.grades = get_consurf_grades(r.conservation);
            r.status = "ok";
        }
    }
    ConsurfResults out;
    for (int k : which) out.push_back(per_unique[static_cast<std::size_t>(k)]);
    return out;
}

std::string get_consurf_grades_text(const ConsurfResult& r, const Strings& atom_labels) {
    if (!r.get_is_ok()) IMP_THROW("get_consurf_grades_text: no result (" << r.status << ")",
                                  ValueException);
    const SequenceConservation& c = r.conservation;
    const int n = c.get_n_positions();
    if (!atom_labels.empty() && static_cast<int>(atom_labels.size()) != n)
        IMP_THROW("get_consurf_grades_text: " << atom_labels.size() << " labels for " << n
                  << " residues", ValueException);
    std::ostringstream out;
    out << "\t Amino Acid Conservation Scores\n"
           "\t===============================\n\n"
           "- POS: The position of the AA in the query sequence.\n"
           "- SEQ: The query sequence in one letter code.\n"
           "- 3LATOM: The residue in the structure, three letters, number and chain.\n"
           "- SCORE: The normalized conservation scores.\n"
           "- COLOR: The color scale representing the conservation scores "
           "(9 - conserved, 1 - variable).\n"
           "- CONFIDENCE INTERVAL: The posterior's interval of the score.\n"
           "- CONFIDENCE INTERVAL COLORS: The colors of its lower and upper bound.\n"
           "- MSA DATA: Aligned sequences with an amino acid at the position, of all.\n"
           "- RESIDUE VARIETY: The residues at the position in the alignment.\n"
           "  (computed by IMP.bff: native search, homologue selection and "
           "Rate4Site's method)\n\n"
           " POS\t SEQ\t    3LATOM\tSCORE\t\tCOLOR\tCONFIDENCE INTERVAL\t"
           "CONFIDENCE INTERVAL COLORS\tMSA DATA\tRESIDUE VARIETY\n"
           "    \t    \t        \t(normalized)\t        \t               \n";
    const std::string& seq = c.get_residues();
    char buf[256];
    for (int i = 0; i < n; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        const std::string label =
                atom_labels.empty() || atom_labels[k].empty() ? std::string("-") : atom_labels[k];
        std::string variety;
        for (char v : get_residue_variety(r.msa, c.get_columns()[k])) {
            if (!variety.empty()) variety.push_back(',');
            variety.push_back(v);
        }
        std::snprintf(buf, sizeof buf,
                      "%4d\t%4c\t%10s\t%6.3f\t\t%3d%s\t%6.3f,%6.3f\t\t\t%5d,%d\t\t\t%5d/%d\t",
                      i + 1, seq[k], label.c_str(), c.get_scores()[k], r.grades[4 * k],
                      r.grades[4 * k + 3] ? "*" : " ", c.get_lower()[k], c.get_upper()[k],
                      r.grades[4 * k + 1], r.grades[4 * k + 2], c.get_n_data()[k],
                      c.get_n_sequences());
        out << buf << variety << "\n";
    }
    out << "\n\n*Below the confidence cut-off - The calculations for this site were performed "
           "on less than 6 non-gaped homologue sequences,\nor the confidence interval for the "
           "estimated score is equal to- or larger than- 4 color grades.\n";
    return out.str();
}

void write_consurf_grades(const ConsurfResult& result, const std::string& path,
                          const Strings& atom_labels) {
    const std::string text = get_consurf_grades_text(result, atom_labels);
    std::ofstream out(path.c_str());
    if (!out) IMP_THROW("write_consurf_grades: cannot write " << path, IOException);
    out << text;
}

void write_consurf_msa(const ConsurfResult& result, const std::string& path) {
    if (!result.get_is_ok()) IMP_THROW("write_consurf_msa: no result (" << result.status << ")",
                                       ValueException);
    std::ofstream out(path.c_str());
    if (!out) IMP_THROW("write_consurf_msa: cannot write " << path, IOException);
    const SequenceMSA& msa = result.msa;
    for (int k = 0; k < msa.get_n_sequences(); ++k)
        out << ">" << msa.get_names()[static_cast<std::size_t>(k)] << "\n"
            << msa.get_sequence(k) << "\n";
}

IMPBFF_END_NAMESPACE
