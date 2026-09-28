/**
 * \file SequenceHomologs.cpp
 * \brief Choosing homologues by ConSurf's rules.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SequenceHomologs.h>

#include <algorithm>
#include <cmath>
#include <numeric>

IMPBFF_BEGIN_NAMESPACE

namespace {

int residues_on_query(const std::string& aligned) {
    return static_cast<int>(std::count_if(aligned.begin(), aligned.end(),
                                          [](char c) { return c != '-'; }));
}

//! Identity of two hits read off their alignments to the query: identical
//! residues in shared query columns over the shorter one's residues there.
double pair_identity(const std::string& a, const std::string& b, int na, int nb) {
    const int shorter = std::min(na, nb);
    if (shorter == 0) return 0.0;
    int same = 0;
    for (std::size_t k = 0; k < a.size() && k < b.size(); ++k)
        same += a[k] != '-' && a[k] == b[k];
    return double(same) / shorter;
}

}  // namespace

SequenceHomologOptions SequenceHomologOptions::consurf_standalone() {
    SequenceHomologOptions o;
    o.max_evalue = 1e-3;
    o.min_identity = 0.0;
    o.max_homologs = 149;
    o.sampling = "best";
    return o;
}

SequenceSearchHits select_sequence_homologs(const SequenceSearchHits& hits,
                                            const SequenceHomologOptions& options,
                                            int query_index) {
    if (options.sampling != "sample" && options.sampling != "best")
        IMP_THROW("select_sequence_homologs: sampling is 'sample' or 'best', not '"
                  << options.sampling << "'", ValueException);
    // The hits that pass on their own, in E-value order.
    SequenceSearchHits passing;
    for (const SequenceSearchHit& h : hits) {
        if (h.query != query_index) continue;
        if (h.evalue > options.max_evalue) continue;
        if (h.identity < options.min_identity || h.identity >= options.max_identity) continue;
        if (h.target_end - h.target_start < options.min_length * h.query_length) continue;
        passing.push_back(h);
    }
    std::stable_sort(passing.begin(), passing.end(),
                     [](const SequenceSearchHit& a, const SequenceSearchHit& b) {
                         return a.evalue < b.evalue;
                     });
    // Redundancy, CD-HIT's way: longest first, each kept hit a representative.
    std::vector<int> n_residues(passing.size());
    for (std::size_t k = 0; k < passing.size(); ++k)
        n_residues[k] = residues_on_query(passing[k].aligned);
    std::vector<std::size_t> by_length(passing.size());
    std::iota(by_length.begin(), by_length.end(), 0);
    std::stable_sort(by_length.begin(), by_length.end(), [&](std::size_t a, std::size_t b) {
        const int la = passing[a].target_end - passing[a].target_start;
        const int lb = passing[b].target_end - passing[b].target_start;
        return la > lb;
    });
    std::vector<char> kept(passing.size(), 0);
    std::vector<std::size_t> representatives;
    for (std::size_t k : by_length) {
        bool redundant = false;
        for (std::size_t r : representatives) {
            if (pair_identity(passing[k].aligned, passing[r].aligned, n_residues[k],
                              n_residues[r]) >= options.redundancy) {
                redundant = true;
                break;
            }
        }
        if (redundant) continue;
        representatives.push_back(k);
        kept[k] = 1;
    }
    SequenceSearchHits unique;
    for (std::size_t k = 0; k < passing.size(); ++k)
        if (kept[k]) unique.push_back(passing[k]);
    const std::size_t cap = static_cast<std::size_t>(std::max(options.max_homologs, 0));
    if (unique.size() <= cap) return unique;
    SequenceSearchHits out;
    if (options.sampling == "best" || cap < 2) {
        out.assign(unique.begin(), unique.begin() + static_cast<std::ptrdiff_t>(cap));
        return out;
    }
    // Evenly over the list: index k * (n - 1) / (cap - 1), rounded; distinct
    // because n > cap.
    const double step = double(unique.size() - 1) / double(cap - 1);
    for (std::size_t k = 0; k < cap; ++k)
        out.push_back(unique[static_cast<std::size_t>(std::llround(k * step))]);
    return out;
}

IMPBFF_END_NAMESPACE
