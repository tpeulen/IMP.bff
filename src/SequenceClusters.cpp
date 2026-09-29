/**
 * \file SequenceClusters.cpp
 * \brief Cluster membership (a bucketed join on disk) and the two-stage search.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SequenceClusters.h>
#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/ptolib.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#ifdef IMP_BFF_HAS_ZLIB
#include <zlib.h>
#endif

IMPBFF_BEGIN_NAMESPACE

namespace {

const char kClusterObject[] = "members";
const int kClusterBuckets = 128;   // files open at once stay under common limits

//! A 64-bit hash of an identifier: FNV-1a, then a finalising mix. Two
//! hundred million identifiers collide with probability ~1e-3.
std::uint64_t cluster_hash(const char* s, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<unsigned char>(s[i]);
        h *= 1099511628211ULL;
    }
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

struct Pair {
    std::uint64_t key, value;
    bool operator<(const Pair& o) const { return key != o.key ? key < o.key : value < o.value; }
};

//! Pairs written to bucket files by a caller-chosen bucket, buffered.
class PairBuckets {
public:
    PairBuckets(const std::string& dir, const std::string& stem) {
        for (int b = 0; b < kClusterBuckets; ++b) {
            paths_.push_back(dir + "/" + stem + "_" + std::to_string(b));
            FILE* f = std::fopen(paths_.back().c_str(), "wb");
            if (f == nullptr)
                IMP_THROW("create_sequence_clusters: cannot write " << paths_.back(), IOException);
            files_.push_back(f);
            buffers_.emplace_back();
            buffers_.back().reserve(kBuffer);
        }
    }
    ~PairBuckets() { close(); }
    void add(int bucket, std::uint64_t key, std::uint64_t value) {
        std::vector<Pair>& buf = buffers_[static_cast<std::size_t>(bucket)];
        buf.push_back(Pair{key, value});
        if (buf.size() == kBuffer) flush(bucket);
    }
    void close() {
        for (int b = 0; b < static_cast<int>(files_.size()); ++b) {
            if (files_[static_cast<std::size_t>(b)] == nullptr) continue;
            flush(b);
            std::fclose(files_[static_cast<std::size_t>(b)]);
            files_[static_cast<std::size_t>(b)] = nullptr;
        }
    }
    const std::string& path(int b) const { return paths_[static_cast<std::size_t>(b)]; }

private:
    static const std::size_t kBuffer = 16384;
    void flush(int b) {
        std::vector<Pair>& buf = buffers_[static_cast<std::size_t>(b)];
        if (buf.empty()) return;
        if (std::fwrite(buf.data(), sizeof(Pair), buf.size(), files_[static_cast<std::size_t>(b)]) !=
            buf.size())
            IMP_THROW("create_sequence_clusters: cannot write " << paths_[static_cast<std::size_t>(b)]
                      << " (disk full?)", IOException);
        buf.clear();
    }
    std::vector<std::string> paths_;
    std::vector<FILE*> files_;
    std::vector<std::vector<Pair> > buffers_;
};

int hash_bucket(std::uint64_t h) {
    return static_cast<int>(h >> 57);   // 128 buckets
}

std::vector<Pair> read_pairs(const std::string& path) {
    std::vector<Pair> out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) IMP_THROW("create_sequence_clusters: cannot read " << path, IOException);
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(static_cast<std::size_t>(bytes) / sizeof(Pair));
    const std::size_t got = std::fread(out.data(), sizeof(Pair), out.size(), f);
    std::fclose(f);
    if (got != out.size()) IMP_THROW("create_sequence_clusters: short read of " << path, IOException);
    std::remove(path.c_str());
    return out;
}

//! Lines of a text file, plain or gzip-compressed, in blocks.
class LineSource {
public:
    explicit LineSource(const std::string& path) : path_(path) {
#ifdef IMP_BFF_HAS_ZLIB
        gz_ = gzopen(path.c_str(), "rb");
        if (gz_ == nullptr) IMP_THROW("create_sequence_clusters: cannot read " << path, IOException);
        gzbuffer(gz_, 1u << 20);
#else
        if (path.size() > 3 && path.compare(path.size() - 3, 3, ".gz") == 0)
            IMP_THROW("create_sequence_clusters: " << path << " is gzip-compressed and this "
                      "imp.bff was built without zlib", IOException);
        f_ = std::fopen(path.c_str(), "rb");
        if (f_ == nullptr) IMP_THROW("create_sequence_clusters: cannot read " << path, IOException);
#endif
        block_.resize(1u << 20);
    }
    ~LineSource() {
#ifdef IMP_BFF_HAS_ZLIB
        if (gz_) gzclose(gz_);
#else
        if (f_) std::fclose(f_);
#endif
    }
    //! The next line without its end; false at the end of the file.
    bool next(std::string& line) {
        line.clear();
        for (;;) {
            if (at_ == size_) {
#ifdef IMP_BFF_HAS_ZLIB
                const int got = gzread(gz_, block_.data(), static_cast<unsigned>(block_.size()));
                if (got < 0) IMP_THROW("create_sequence_clusters: error reading " << path_, IOException);
                size_ = static_cast<std::size_t>(got);
#else
                size_ = std::fread(block_.data(), 1, block_.size(), f_);
#endif
                at_ = 0;
                if (size_ == 0) return !line.empty();
            }
            const char* start = block_.data() + at_;
            const void* nl = std::memchr(start, '\n', size_ - at_);
            if (nl == nullptr) {
                line.append(start, size_ - at_);
                at_ = size_;
                continue;
            }
            const std::size_t n = static_cast<std::size_t>(static_cast<const char*>(nl) - start);
            line.append(start, n);
            at_ += n + 1;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return true;
        }
    }

private:
    std::string path_;
    std::vector<char> block_;
    std::size_t at_ = 0, size_ = 0;
#ifdef IMP_BFF_HAS_ZLIB
    gzFile gz_ = nullptr;
#else
    FILE* f_ = nullptr;
#endif
};

//! Field \p column (1-based) of a tab-separated line, as a pointer and length.
bool tab_field(const std::string& line, int column, const char** at, std::size_t* n) {
    std::size_t start = 0;
    for (int c = 1; c < column; ++c) {
        start = line.find('\t', start);
        if (start == std::string::npos) return false;
        ++start;
    }
    std::size_t end = line.find('\t', start);
    if (end == std::string::npos) end = line.size();
    *at = line.data() + start;
    *n = end - start;
    return true;
}

}  // namespace

std::size_t create_sequence_clusters(const std::string& members,
                                     const std::string& representatives,
                                     const std::string& mapping, int member_column,
                                     int representative_column, const std::string& temporary) {
    const SequenceDatabase member_db(members);
    const std::string rep_path = SequenceDatabase(representatives).get_path();
    const std::size_t n_reps = SequenceDatabase(rep_path).get_number_of_sequences();
    const std::string dir = temporary.empty() ? rep_path + ".clusters-tmp" : temporary;
    std::filesystem::create_directories(dir);

    // 1. the mapping: member cluster -> representative cluster, by hash
    {
        PairBuckets a(dir, "map");
        LineSource in(mapping);
        std::string line;
        const bool xml = mapping.find(".xml") != std::string::npos;
        if (xml) {
            // UniRef's XML: <entry id="UniRef50_X"> ... per member a
            // <property type="UniRef90 ID" value="UniRef90_Y"/>; the member
            // level is the one the member database's identifiers name.
            const std::string entry = "<entry id=\"";
            std::string level = "UniRef90";
            if (member_db.get_number_of_sequences() > 0) {
                const std::string first = member_db.get_identifier(0);
                level = first.substr(0, first.find('_'));
            }
            const std::string member_key = "type=\"" + level + " ID\" value=\"";
            std::uint64_t current = 0;
            bool have = false;
            while (in.next(line)) {
                std::string::size_type at = line.find(entry);
                if (at != std::string::npos) {
                    at += entry.size();
                    const std::string::size_type end = line.find('"', at);
                    if (end == std::string::npos) continue;
                    current = cluster_hash(line.data() + at, end - at);
                    have = true;
                    continue;
                }
                if (!have) continue;
                at = line.find(member_key);
                if (at == std::string::npos) continue;
                at += member_key.size();
                const std::string::size_type end = line.find('"', at);
                if (end == std::string::npos) continue;
                const std::uint64_t hm = cluster_hash(line.data() + at, end - at);
                a.add(hash_bucket(hm), hm, current);
            }
        } else {
            const char* p = nullptr;
            const char* q = nullptr;
            std::size_t np = 0, nq = 0;
            while (in.next(line)) {
                if (!tab_field(line, member_column, &p, &np) || np == 0) continue;
                if (!tab_field(line, representative_column, &q, &nq) || nq == 0) continue;
                const std::uint64_t hm = cluster_hash(p, np);
                a.add(hash_bucket(hm), hm, cluster_hash(q, nq));
            }
        }
    }
    // 2. the members' identifiers -> rows
    {
        PairBuckets b(dir, "member");
        member_db.scan_identifiers([&](std::size_t row, const std::string& id) {
            const std::uint64_t h = cluster_hash(id.data(), id.size());
            b.add(hash_bucket(h), h, row);
        });
    }
    // 3. join: member row -> representative hash; unplaced rows are orphans
    std::vector<std::uint64_t> orphans;
    {
        PairBuckets c(dir, "placed");
        for (int k = 0; k < kClusterBuckets; ++k) {
            std::vector<Pair> map = read_pairs(dir + "/map_" + std::to_string(k));
            std::vector<Pair> rows = read_pairs(dir + "/member_" + std::to_string(k));
            std::sort(map.begin(), map.end());
            map.erase(std::unique(map.begin(), map.end(),
                                  [](const Pair& x, const Pair& y) { return x.key == y.key; }),
                      map.end());
            std::sort(rows.begin(), rows.end());
            std::size_t i = 0;
            for (const Pair& r : rows) {
                while (i < map.size() && map[i].key < r.key) ++i;
                if (i < map.size() && map[i].key == r.key)
                    c.add(hash_bucket(map[i].value), map[i].value, r.value);
                else
                    orphans.push_back(r.value);
            }
        }
    }
    // 4. the representatives' identifiers -> rows
    {
        PairBuckets d(dir, "rep");
        SequenceDatabase(rep_path).scan_identifiers([&](std::size_t row, const std::string& id) {
            const std::uint64_t h = cluster_hash(id.data(), id.size());
            d.add(hash_bucket(h), h, row);
        });
    }
    // 5. join: (representative row, member row), bucketed by representative row
    std::size_t placed = 0;
    {
        PairBuckets e(dir, "cluster");
        for (int k = 0; k < kClusterBuckets; ++k) {
            std::vector<Pair> reps = read_pairs(dir + "/rep_" + std::to_string(k));
            std::vector<Pair> rows = read_pairs(dir + "/placed_" + std::to_string(k));
            std::sort(reps.begin(), reps.end());
            std::sort(rows.begin(), rows.end());
            std::size_t i = 0;
            for (const Pair& r : rows) {
                while (i < reps.size() && reps[i].key < r.key) ++i;
                if (i < reps.size() && reps[i].key == r.key) {
                    const int bucket = static_cast<int>(reps[i].value * kClusterBuckets /
                                                        std::max<std::size_t>(n_reps, 1));
                    e.add(bucket, reps[i].value, r.value);
                    ++placed;
                } else {
                    orphans.push_back(r.value);   // a cluster the representatives lack
                }
            }
        }
    }
    // 6. write the membership into the representatives' container
    {
        pto::File container;
        if (!container.open(rep_path, true))
            IMP_THROW("create_sequence_clusters: " << container.error(), IOException);
        const std::uint64_t old = container.find(kClusterObject);
        if (old != 0) container.remove(old);
        pto::StoreWriter store(container, "table", kClusterObject);
        const int col_members = store.add_column("members", pto::ColumnType::UInt32, true, "none");
        const int col_orphans = store.add_column("orphans", pto::ColumnType::UInt32, false, "none");
        nlohmann::json meta;
        meta["members"] = members;
        meta["mapping"] = mapping;
        store.set_metadata(col_members, meta.dump());
        std::size_t next = 0;
        std::vector<std::uint32_t> row;
        for (int k = 0; k < kClusterBuckets; ++k) {
            std::vector<Pair> pairs = read_pairs(dir + "/cluster_" + std::to_string(k));
            std::sort(pairs.begin(), pairs.end());
            std::size_t i = 0;
            // the rows r with r * buckets / n_reps == k end at ceil((k + 1) * n_reps / buckets)
            const std::size_t end = ((static_cast<std::size_t>(k) + 1) * n_reps + kClusterBuckets - 1) /
                                    kClusterBuckets;
            for (; next < end; ++next) {
                row.clear();
                while (i < pairs.size() && pairs[i].key == next)
                    row.push_back(static_cast<std::uint32_t>(pairs[i++].value));
                store.append_row(col_members, row.data(), row.size());
            }
        }
        std::sort(orphans.begin(), orphans.end());
        std::vector<std::uint32_t> o(orphans.begin(), orphans.end());
        if (!o.empty()) store.append(col_orphans, o.data(), o.size());
        if (!store.close() || !container.commit())
            IMP_THROW("create_sequence_clusters: could not write " << rep_path, IOException);
        container.close();
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return placed;
}

SequenceClusters::SequenceClusters(const std::string& representatives) {
    const std::string path = SequenceDatabase(representatives).get_path();
    pto::File container;
    if (!container.open(path))
        IMP_THROW("SequenceClusters: " << path << ": " << container.error(), IOException);
    const std::uint64_t uid = container.find(kClusterObject);
    if (uid == 0)
        IMP_THROW("SequenceClusters: " << path << " has no cluster membership; "
                  "run create_sequence_clusters", IOException);
    reader_ = std::make_shared<pto::StoreReader>(container, uid);
    n_clusters_ = static_cast<std::size_t>(reader_->info("members").n_rows);
    n_orphans_ = static_cast<std::size_t>(reader_->info("orphans").n_rows);
}

Ints SequenceClusters::get_members(std::size_t k) const {
    const std::vector<std::size_t> m = get_member_rows(k);
    return Ints(m.begin(), m.end());
}

Ints SequenceClusters::get_orphans() const {
    const std::vector<std::size_t> o = get_orphan_rows();
    return Ints(o.begin(), o.end());
}

std::vector<std::size_t> SequenceClusters::get_member_rows(std::size_t k) const {
    if (!reader_ || k >= n_clusters_) IMP_THROW("SequenceClusters: no cluster " << k, IndexException);
    std::uint64_t n = 0;
    const std::uint32_t* p = static_cast<const std::uint32_t*>(reader_->row_view("members", k, &n));
    return std::vector<std::size_t>(p, p + n);
}

std::vector<std::size_t> SequenceClusters::get_orphan_rows() const {
    if (!reader_ || n_orphans_ == 0) return std::vector<std::size_t>();
    std::vector<std::uint32_t> o(n_orphans_);
    reader_->read("orphans", 0, n_orphans_, o.data());
    return std::vector<std::size_t>(o.begin(), o.end());
}

SequenceSearchHits search_clustered_sequence_database(
        const Strings& queries, const SequenceDatabase& representatives,
        const SequenceClusters& clusters, const SequenceDatabase& members,
        const SequenceSearchOptions& options, const SequenceClusterSearchOptions& cluster_options) {
    if (clusters.get_number_of_clusters() != representatives.get_number_of_sequences())
        IMP_THROW("search_clustered_sequence_database: " << clusters.get_number_of_clusters()
                  << " clusters for " << representatives.get_number_of_sequences()
                  << " representatives: not this database's membership", ValueException);
    // Stage 1: the representatives, reaching further than the final cut-off.
    SequenceSearchOptions first = options;
    first.max_candidates = std::max(cluster_options.max_representatives, options.max_candidates);
    first.max_evalue = std::max(cluster_options.max_representative_evalue, options.max_evalue);
    const SequenceSearchHits found = search_sequence_database(queries, representatives, first);
    // Stage 2: the members of every cluster found (and the orphans), in row order.
    std::vector<std::size_t> rows;
    std::vector<char> seen(clusters.get_number_of_clusters(), 0);
    for (const SequenceSearchHit& h : found) {
        if (seen[h.target]) continue;
        seen[h.target] = 1;
        const std::vector<std::size_t> m = clusters.get_member_rows(h.target);
        rows.insert(rows.end(), m.begin(), m.end());
    }
    if (cluster_options.search_orphans) {
        const std::vector<std::size_t> o = clusters.get_orphan_rows();
        rows.insert(rows.end(), o.begin(), o.end());
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return search_sequence_database_rows(queries, members, rows, options);
}

IMPBFF_END_NAMESPACE
