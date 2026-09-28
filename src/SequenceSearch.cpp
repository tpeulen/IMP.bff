/**
 * \file SequenceSearch.cpp
 * \brief Protein homology search of a sequence database: k-mer neighbourhood
 *        prefilter, ungapped diagonals, affine Smith-Waterman, E-values.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SequenceSearch.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

IMPBFF_BEGIN_NAMESPACE

namespace sequence_search {

// ---- scoring ----------------------------------------------------------------

//! BLOSUM62 (Henikoff & Henikoff, PNAS 89:10915, 1992) in the order below.
const char kBlosumOrder[] = "ARNDCQEGHILKMFPSTWYV";
const int kBlosum62[20][20] = {
    { 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0},
    {-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3},
    {-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3},
    {-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3},
    { 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1},
    {-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2},
    {-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2},
    { 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3},
    {-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3},
    {-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3},
    {-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1},
    {-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2},
    {-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1},
    {-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1},
    {-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2},
    { 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2},
    { 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0},
    {-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3},
    {-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1},
    { 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4}};

// Karlin-Altschul parameters of BLOSUM62: gapped at 11/1, and ungapped.
const double kLambda = 0.267, kK = 0.041;
const double kLambdaUngapped = 0.3176, kKUngapped = 0.134;
const double kLn2 = 0.69314718055994530942;

//! Substitution scores over the database codes: 0 = X, 1..20 = SequenceMSA's
//! alphabet. Anything against X scores -1.
struct Matrix {
    std::array<std::array<int, 21>, 21> s;
    Matrix() {
        const std::string alphabet = get_sequence_alphabet();
        for (auto& row : s) row.fill(-1);
        for (int a = 0; a < 20; ++a) {
            const int ia = static_cast<int>(std::string(kBlosumOrder).find(alphabet[a]));
            for (int b = 0; b < 20; ++b) {
                const int ib = static_cast<int>(std::string(kBlosumOrder).find(alphabet[b]));
                s[a + 1][b + 1] = kBlosum62[ia][ib];
            }
        }
    }
    int operator()(unsigned a, unsigned b) const { return s[a][b]; }
};

const Matrix& matrix() {
    static const Matrix m;
    return m;
}

std::vector<unsigned char> encode(const std::string& sequence) {
    const std::string alphabet = get_sequence_alphabet();
    std::vector<unsigned char> out;
    out.reserve(sequence.size());
    for (char c : sequence) {
        if (!std::isalpha(static_cast<unsigned char>(c))) continue;
        const char u = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        const std::string::size_type k = alphabet.find(u);
        out.push_back(k == std::string::npos ? 0 : static_cast<unsigned char>(k + 1));
    }
    return out;
}

const char kSearchLetters[] = "XACDEFGHIKLMNPQRSTVWY";

// ---- k-mers -----------------------------------------------------------------

const int kK_mer = 5;
const std::uint32_t kKmerSpace = 20 * 20 * 20 * 20 * 20;

//! Where a query k-mer (or a neighbour of it) sits: which query, which position.
struct Seed {
    std::uint16_t query;
    std::uint16_t position;
};

//! The neighbourhoods of all queries, by k-mer: a bitset to reject a target
//! k-mer in one load, and the seeds of the k-mers that pass.
struct SeedTable {
    std::vector<std::uint64_t> present;       // kKmerSpace bits
    std::vector<std::uint32_t> start;         // CSR offsets, kKmerSpace + 1
    std::vector<Seed> seeds;

    bool has(std::uint32_t kmer) const { return (present[kmer >> 6] >> (kmer & 63)) & 1u; }
};

SeedTable build_seeds(const std::vector<std::vector<unsigned char>>& queries, int threshold) {
    const Matrix& m = matrix();
    // Per residue code, the 20 amino acids by descending score against it, to
    // stop a branch once the best remaining letters cannot reach the threshold.
    std::array<std::array<unsigned char, 20>, 21> by_score;
    std::array<int, 21> best;
    for (unsigned a = 0; a <= 20; ++a) {
        for (unsigned b = 0; b < 20; ++b) by_score[a][b] = static_cast<unsigned char>(b + 1);
        std::sort(by_score[a].begin(), by_score[a].end(),
                  [&](unsigned char x, unsigned char y) { return m(a, x) > m(a, y); });
        best[a] = m(a, by_score[a][0]);
    }
    std::vector<std::pair<std::uint32_t, Seed>> all;
    for (std::size_t q = 0; q < queries.size(); ++q) {
        const std::vector<unsigned char>& s = queries[q];
        for (std::size_t p = 0; p + kK_mer <= s.size(); ++p) {
            bool clean = true;
            for (int r = 0; r < kK_mer; ++r) clean = clean && s[p + r] != 0;
            if (!clean) continue;
            int rest[kK_mer + 1];
            rest[kK_mer] = 0;
            for (int r = kK_mer - 1; r >= 0; --r) rest[r] = rest[r + 1] + best[s[p + r]];
            const Seed seed{static_cast<std::uint16_t>(q), static_cast<std::uint16_t>(p)};
            std::uint32_t exact = 0;
            for (int r = 0; r < kK_mer; ++r) exact = exact * 20 + (s[p + r] - 1u);
            // Depth-first over the neighbourhood; the exact k-mer always seeds.
            bool exact_seen = false;
            std::function<void(int, int, std::uint32_t)> walk = [&](int r, int score, std::uint32_t code) {
                if (r == kK_mer) {
                    all.push_back({code, seed});
                    if (code == exact) exact_seen = true;
                    return;
                }
                const unsigned a = s[p + r];
                for (unsigned char b : by_score[a]) {
                    const int next = score + m(a, b);
                    if (next + rest[r + 1] < threshold) break;
                    walk(r + 1, next, code * 20 + (b - 1u));
                }
            };
            walk(0, 0, 0);
            if (!exact_seen) all.push_back({exact, seed});
        }
    }
    SeedTable t;
    t.present.assign(kKmerSpace / 64 + 1, 0);
    t.start.assign(kKmerSpace + 1, 0);
    for (const auto& e : all) {
        t.present[e.first >> 6] |= std::uint64_t(1) << (e.first & 63);
        ++t.start[e.first + 1];
    }
    for (std::uint32_t k = 0; k < kKmerSpace; ++k) t.start[k + 1] += t.start[k];
    t.seeds.resize(all.size());
    std::vector<std::uint32_t> fill(t.start.begin(), t.start.end() - 1);
    for (const auto& e : all) t.seeds[fill[e.first]++] = e.second;
    return t;
}

// ---- prefilter ----------------------------------------------------------------

//! The best ungapped segment on diagonal `offset` (target position minus
//! query position): the maximum-scoring run, Kadane's.
int ungapped_diagonal(const std::vector<unsigned char>& q, const unsigned char* t,
                      std::uint64_t n, long offset) {
    const Matrix& m = matrix();
    const long first = std::max(0L, -offset);
    const long last = std::min(static_cast<long>(q.size()), static_cast<long>(n) - offset);
    int best = 0, run = 0;
    for (long i = first; i < last; ++i) {
        run += m(q[static_cast<std::size_t>(i)], t[i + offset]);
        if (run < 0) run = 0;
        if (run > best) best = run;
    }
    return best;
}

struct Candidate {
    int score;
    std::size_t target;
    bool operator>(const Candidate& o) const {
        return score != o.score ? score > o.score : target < o.target;
    }
};
//! A min-heap: its top is the weakest candidate kept.
using CandidateHeap = std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>>;

void keep(CandidateHeap& heap, const Candidate& c, std::size_t capacity) {
    if (heap.size() < capacity) heap.push(c);
    else if (c > heap.top()) { heap.pop(); heap.push(c); }
}

//! One thread's diagonal bookkeeping for one query: the last hit per diagonal,
//! stamped with the target it belongs to so nothing is cleared between targets.
struct Diagonals {
    std::vector<std::uint32_t> stamp;
    std::vector<std::int32_t> last;   // target position of the last hit; < 0: scored
    void fit(std::size_t n) {
        if (stamp.size() < n) { stamp.resize(n, 0xFFFFFFFFu); last.resize(n, 0); }
    }
};

// ---- alignment ----------------------------------------------------------------

//! Gotoh's affine Smith-Waterman with traceback; the hit's aligned fields.
SequenceSearchHit smith_waterman_hit(const std::vector<unsigned char>& q,
                                     const unsigned char* t, int n, int gap_open,
                                     int gap_extend) {
    const Matrix& sm = matrix();
    const int m = static_cast<int>(q.size());
    const int open = gap_open + gap_extend, ext = gap_extend;
    const int kNeg = -(1 << 28);
    // Traceback, one byte a cell: bits 0-1 where H came from (0 start,
    // 1 diagonal, 2 E, 3 F); bit 2 E extended E; bit 3 F extended F.
    std::vector<unsigned char> tb(static_cast<std::size_t>(m + 1) * (n + 1), 0);
    std::vector<int> h_prev(n + 1, 0), h_cur(n + 1, 0), f(n + 1, kNeg);
    int best = 0, bi = 0, bj = 0;
    for (int i = 1; i <= m; ++i) {
        const unsigned qa = q[i - 1];
        int e = kNeg;
        h_cur[0] = 0;
        unsigned char* row = &tb[static_cast<std::size_t>(i) * (n + 1)];
        for (int j = 1; j <= n; ++j) {
            unsigned char dir = 0;
            // E: a gap in the query, moving along the target.
            const int e_open = h_cur[j - 1] - open, e_ext = e - ext;
            if (e_ext >= e_open) { e = e_ext; dir |= 4; } else e = e_open;
            // F: a gap in the target, moving along the query.
            const int f_open = h_prev[j] - open, f_ext = f[j] - ext;
            if (f_ext >= f_open) { f[j] = f_ext; dir |= 8; } else f[j] = f_open;
            int h = h_prev[j - 1] + sm(qa, t[j - 1]);
            int from = 1;
            if (e > h) { h = e; from = 2; }
            if (f[j] > h) { h = f[j]; from = 3; }
            if (h <= 0) { h = 0; from = 0; }
            h_cur[j] = h;
            row[j] = static_cast<unsigned char>(dir | from);
            if (h > best) { best = h; bi = i; bj = j; }
        }
        std::swap(h_prev, h_cur);
    }
    SequenceSearchHit hit;
    hit.score = best;
    hit.query_length = m;
    hit.target_length = n;
    hit.aligned.assign(static_cast<std::size_t>(m), '-');
    if (best == 0) return hit;
    int i = bi, j = bj, state = 0;   // 0 H, 2 E, 3 F
    int identities = 0, columns = 0;
    while (i > 0 && j > 0) {
        const unsigned char cell = tb[static_cast<std::size_t>(i) * (n + 1) + j];
        if (state == 0) {
            const int from = cell & 3;
            if (from == 0) break;
            if (from == 1) {
                hit.aligned[static_cast<std::size_t>(i - 1)] = kSearchLetters[t[j - 1]];
                if (q[i - 1] == t[j - 1] && q[i - 1] != 0) ++identities;
                ++columns;
                --i; --j;
                continue;
            }
            state = from;
            continue;
        }
        if (state == 2) {   // target residue against a query gap: an insertion, dropped
            ++columns;
            const bool extended = (cell & 4) != 0;
            --j;
            if (!extended) state = 0;
            continue;
        }
        // state 3: query residue against a target gap
        ++columns;
        const bool extended = (cell & 8) != 0;
        --i;
        if (!extended) state = 0;
    }
    hit.query_start = i;
    hit.query_end = bi;
    hit.target_start = j;
    hit.target_end = bj;
    hit.identity = columns > 0 ? double(identities) / columns : 0.0;
    hit.bits = (kLambda * best - std::log(kK)) / kLn2;
    return hit;
}

double evalue(int score, std::size_t query_length, double database_residues) {
    return kK * double(query_length) * database_residues * std::exp(-kLambda * score);
}

}  // namespace sequence_search

int get_blosum62(char a, char b) {
    using namespace sequence_search;
    const std::string alphabet = get_sequence_alphabet();
    const auto code = [&](char c) {
        const std::string::size_type k =
                alphabet.find(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
        return k == std::string::npos ? 0u : static_cast<unsigned>(k + 1);
    };
    return matrix()(code(a), code(b));
}

SequenceSearchHit align_sequences(const std::string& query, const std::string& target,
                                  const SequenceSearchOptions& options) {
    using namespace sequence_search;
    const std::vector<unsigned char> q = encode(query), t = encode(target);
    SequenceSearchHit hit = smith_waterman_hit(q, t.data(), static_cast<int>(t.size()),
                                               options.gap_open, options.gap_extend);
    hit.evalue = evalue(hit.score, q.size(), double(t.size()));
    return hit;
}

namespace sequence_search {

//! The search, over every sequence of \p database or over \p rows of it
//! (sorted); E-values against \p residues.
SequenceSearchHits search(const Strings& queries, const SequenceDatabase& database,
                          const std::vector<std::size_t>* rows, double residues,
                          const SequenceSearchOptions& options) {
    if (queries.empty()) return SequenceSearchHits();
    if (queries.size() > 65535)
        IMP_THROW("search_sequence_database: at most 65535 queries at once", ValueException);
    std::vector<std::vector<unsigned char>> q;
    std::size_t longest_query = 0;
    for (const std::string& s : queries) {
        q.push_back(encode(s));
        if (q.back().size() > 65535)
            IMP_THROW("search_sequence_database: a query longer than 65535 residues",
                      ValueException);
        longest_query = std::max(longest_query, q.back().size());
    }
    const SeedTable seeds = build_seeds(q, options.kmer_threshold);
    const std::size_t capacity = static_cast<std::size_t>(std::max(options.max_candidates, 1));
    // The ungapped threshold in raw score units.
    const int min_ungapped = static_cast<int>(std::ceil(
            (options.min_ungapped_bits * kLn2 + std::log(kKUngapped)) / kLambdaUngapped));

    unsigned n_threads = options.threads > 0 ? static_cast<unsigned>(options.threads)
                                             : std::thread::hardware_concurrency();
    n_threads = std::max(1u, n_threads);
    // Work units: whole segments of the database, or blocks of the rows.
    const std::size_t kRowBlock = 4096;
    const std::size_t n_units = rows ? (rows->size() + kRowBlock - 1) / kRowBlock
                                     : database.get_number_of_segments();
    if (!rows) database.advise_sequential();

    // ---- prefilter ----
    std::vector<std::vector<CandidateHeap>> heaps(n_threads, std::vector<CandidateHeap>(q.size()));
    std::atomic<std::size_t> next_unit{0};
    std::vector<std::exception_ptr> errors(n_threads);
    const std::uint32_t kTop = 20 * 20 * 20 * 20;   // the weight of a k-mer's first residue
    auto prefilter = [&](unsigned thread) {
        try {
            std::vector<Diagonals> diagonals(q.size());
            std::vector<std::uint64_t> offsets;
            std::vector<int> best(q.size());
            // One target: its diagonals, and its best ungapped score per query.
            auto scan = [&](const unsigned char* t, std::uint64_t n, std::size_t row) {
                if (n < static_cast<std::uint64_t>(kK_mer)) return;
                const std::uint32_t stamp = static_cast<std::uint32_t>(row);
                std::fill(best.begin(), best.end(), 0);
                std::uint32_t kmer = 0;
                int valid = 0;
                for (std::uint64_t j = 0; j < n; ++j) {
                    const unsigned char c = t[j];
                    if (c == 0) { valid = 0; kmer = 0; continue; }
                    if (valid == kK_mer) kmer -= (t[j - kK_mer] - 1u) * kTop;
                    else ++valid;
                    kmer = kmer * 20 + (c - 1u);
                    if (valid < kK_mer || !seeds.has(kmer)) continue;
                    const std::int32_t at = static_cast<std::int32_t>(j + 1 - kK_mer);
                    for (std::uint32_t e = seeds.start[kmer]; e < seeds.start[kmer + 1]; ++e) {
                        const Seed s = seeds.seeds[e];
                        Diagonals& dg = diagonals[s.query];
                        const std::size_t qn = q[s.query].size();
                        dg.fit(qn + static_cast<std::size_t>(n) + 1);
                        const std::size_t d = static_cast<std::size_t>(at) + qn - s.position;
                        if (dg.stamp[d] != stamp) {
                            dg.stamp[d] = stamp;
                            dg.last[d] = at;
                            continue;
                        }
                        if (dg.last[d] < 0 || dg.last[d] == at) continue;   // scored, or the same hit
                        dg.last[d] = -1;
                        const int score = ungapped_diagonal(
                                q[s.query], t, n, static_cast<long>(at) - s.position);
                        if (score > best[s.query]) best[s.query] = score;
                    }
                }
                for (std::size_t qi = 0; qi < q.size(); ++qi)
                    if (best[qi] >= min_ungapped)
                        keep(heaps[thread][qi], Candidate{best[qi], row}, capacity);
            };
            for (std::size_t k; (k = next_unit++) < n_units;) {
                if (rows) {
                    const std::size_t end = std::min(rows->size(), (k + 1) * kRowBlock);
                    for (std::size_t i = k * kRowBlock; i < end; ++i) {
                        std::uint64_t n = 0;
                        const unsigned char* t = database.get_codes((*rows)[i], &n);
                        scan(t, n, (*rows)[i]);
                    }
                    continue;
                }
                std::size_t first = 0, count = 0;
                std::uint64_t n_codes = 0;
                const unsigned char* codes = database.get_segment(k, &first, &count, &n_codes);
                offsets.resize(count + 1);
                database.get_offsets(first, count, offsets.data());
                const std::uint64_t base = offsets[0];
                for (std::size_t r = 0; r < count; ++r)
                    scan(codes + (offsets[r] - base), offsets[r + 1] - offsets[r], first + r);
            }
        } catch (...) {
            errors[thread] = std::current_exception();
        }
    };
    {
        std::vector<std::thread> pool;
        for (unsigned t = 1; t < n_threads; ++t) pool.emplace_back(prefilter, t);
        prefilter(0);
        for (std::thread& t : pool) t.join();
    }
    for (const std::exception_ptr& e : errors) if (e) std::rethrow_exception(e);

    // ---- merge the candidates of every thread, per query ----
    struct Job { int query; std::size_t target; };
    std::vector<Job> jobs;
    for (std::size_t qi = 0; qi < q.size(); ++qi) {
        CandidateHeap merged;
        for (unsigned t = 0; t < n_threads; ++t) {
            CandidateHeap& h = heaps[t][qi];
            while (!h.empty()) { keep(merged, h.top(), capacity); h.pop(); }
        }
        while (!merged.empty()) {
            jobs.push_back(Job{static_cast<int>(qi), merged.top().target});
            merged.pop();
        }
    }

    // ---- align the candidates ----
    std::vector<SequenceSearchHit> aligned(jobs.size());
    std::vector<char> accepted(jobs.size(), 0);
    std::atomic<std::size_t> next_job{0};
    auto align = [&](unsigned thread) {
        try {
            for (std::size_t k; (k = next_job++) < jobs.size();) {
                const Job& job = jobs[k];
                std::uint64_t n = 0;
                const unsigned char* t = database.get_codes(job.target, &n);
                SequenceSearchHit hit = smith_waterman_hit(q[job.query], t, static_cast<int>(n),
                                                           options.gap_open, options.gap_extend);
                hit.evalue = evalue(hit.score, q[job.query].size(), residues);
                if (hit.score == 0 || hit.evalue > options.max_evalue) continue;
                hit.query = job.query;
                hit.target = job.target;
                aligned[k] = hit;
                accepted[k] = 1;
            }
        } catch (...) {
            errors[thread] = std::current_exception();
        }
    };
    {
        std::vector<std::thread> pool;
        for (unsigned t = 1; t < n_threads; ++t) pool.emplace_back(align, t);
        align(0);
        for (std::thread& t : pool) t.join();
    }
    for (const std::exception_ptr& e : errors) if (e) std::rethrow_exception(e);

    SequenceSearchHits out;
    for (std::size_t k = 0; k < jobs.size(); ++k) {
        if (!accepted[k]) continue;
        aligned[k].identifier = database.get_identifier(aligned[k].target);
        out.push_back(aligned[k]);
    }
    std::sort(out.begin(), out.end(), [](const SequenceSearchHit& a, const SequenceSearchHit& b) {
        if (a.query != b.query) return a.query < b.query;
        if (a.evalue != b.evalue) return a.evalue < b.evalue;
        return a.target < b.target;
    });
    return out;
}

}  // namespace sequence_search

SequenceSearchHits search_sequence_database(const Strings& queries,
                                            const SequenceDatabase& database,
                                            const SequenceSearchOptions& options) {
    return sequence_search::search(queries, database, nullptr,
                                   double(database.get_number_of_residues()), options);
}

SequenceSearchHits search_sequence_database_rows(const Strings& queries,
                                                 const SequenceDatabase& database,
                                                 const std::vector<std::size_t>& rows,
                                                 const SequenceSearchOptions& options,
                                                 double residues) {
    for (std::size_t i = 1; i < rows.size(); ++i)
        if (rows[i] <= rows[i - 1])
            IMP_THROW("search_sequence_database_rows: rows must be sorted and distinct",
                      ValueException);
    if (!rows.empty() && rows.back() >= database.get_number_of_sequences())
        IMP_THROW("search_sequence_database_rows: no row " << rows.back(), IndexException);
    return sequence_search::search(
            queries, database, &rows,
            residues > 0 ? residues : double(database.get_number_of_residues()), options);
}

SequenceMSA get_query_msa(const std::string& query, const SequenceSearchHits& hits,
                          int query_index) {
    std::string q;
    for (char c : query)
        if (std::isalpha(static_cast<unsigned char>(c)))
            q.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    std::vector<std::string> names{"query"}, sequences{q};
    for (const SequenceSearchHit& h : hits) {
        if (h.query != query_index) continue;
        if (h.aligned.size() != q.size())
            IMP_THROW("get_query_msa: hit " << h.identifier << " is not aligned to a query of length "
                      << q.size(), ValueException);
        names.push_back(h.identifier);
        sequences.push_back(h.aligned);
    }
    return SequenceMSA(names, sequences, 0, false);
}

IMPBFF_END_NAMESPACE
