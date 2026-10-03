/**
 * \file SequenceClusters.cpp
 * \brief Cluster membership (a bucketed join on disk) and the two-stage search.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SequenceClusters.h>
#include <IMP/bff/EmbeddingIndex.h>
#include <IMP/bff/ProteinLanguageModel.h>
#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/ptolib.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <atomic>
#include <thread>
#ifdef _WIN32
#include <cstdio>   // _setmaxstdio
#else
#include <sys/resource.h>
#endif

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

namespace {

//! One member record in a bucket file: cluster, row, then codes and header.
struct MemberRecordHead {
    std::uint32_t cluster, row, n_codes, n_header;
};

//! Residue codes (0..20) in 5 bits each, for the bucket files.
std::size_t packed_bytes(std::size_t n) { return (n * 5 + 7) / 8; }

void pack5(const unsigned char* in, std::size_t n, std::vector<char>& out) {
    const std::size_t at = out.size();
    out.resize(at + packed_bytes(n), 0);
    unsigned char* p = reinterpret_cast<unsigned char*>(out.data() + at);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t bit = i * 5;
        const unsigned v = in[i] & 31u;
        p[bit / 8] |= static_cast<unsigned char>(v << (bit % 8));
        if (bit % 8 > 3) p[bit / 8 + 1] |= static_cast<unsigned char>(v >> (8 - bit % 8));
    }
}

void unpack5(const char* packed, std::size_t n, std::vector<unsigned char>& out) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(packed);
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t bit = i * 5;
        unsigned v = p[bit / 8] >> (bit % 8);
        if (bit % 8 > 3) v |= static_cast<unsigned>(p[bit / 8 + 1]) << (8 - bit % 8);
        out[i] = static_cast<unsigned char>(v & 31u);
    }
}

//! Which of the 8000 3-mers over the 20 amino acids a sequence contains.
struct KmerSet {
    std::vector<std::uint64_t> bits = std::vector<std::uint64_t>(125, 0);
    unsigned count = 0;
    explicit KmerSet(const std::vector<unsigned char>& s) {
        for (std::size_t i = 2; i < s.size(); ++i) {
            if (!s[i] || !s[i - 1] || !s[i - 2]) continue;
            const unsigned k = (s[i - 2] - 1u) * 400u + (s[i - 1] - 1u) * 20u + (s[i] - 1u);
            if (!((bits[k / 64] >> (k % 64)) & 1u)) { bits[k / 64] |= std::uint64_t(1) << (k % 64); ++count; }
        }
    }
    double similarity(const KmerSet& o) const {
        const unsigned m = std::min(count, o.count);
        if (m == 0) return 0.0;
        unsigned shared = 0;
        for (std::size_t w = 0; w < bits.size(); ++w)
            shared += static_cast<unsigned>(__builtin_popcountll(bits[w] & o.bits[w]));
        return double(shared) / m;
    }
};

}  // namespace

std::size_t create_clustered_sequence_database(const std::string& members,
                                               const std::string& representatives,
                                               const std::string& out,
                                               const std::string& temporary) {
    const SequenceDatabase member_db(members);
    const SequenceDatabase rep_db(representatives);
    const SequenceClusters clusters(representatives);
    const std::size_t n_reps = rep_db.get_number_of_sequences();
    const std::size_t n_members = member_db.get_number_of_sequences();
    if (clusters.get_number_of_clusters() != n_reps)
        IMP_THROW("create_clustered_sequence_database: the membership is not these "
                  "representatives'", ValueException);
    if (n_members >= 0xFFFFFFFFu || n_reps >= 0xFFFFFFFFu)
        IMP_THROW("create_clustered_sequence_database: too many sequences", ValueException);
    // 1. member row -> cluster (orphans: n_reps)
    std::vector<std::uint32_t> cluster_of(n_members, static_cast<std::uint32_t>(n_reps));
    for (std::size_t k = 0; k < n_reps; ++k)
        for (std::size_t m : clusters.get_member_rows(k))
            cluster_of[m] = static_cast<std::uint32_t>(k);
    // 2. members into bucket files by cluster range, in one streaming pass
    const std::string dir = temporary.empty() ? out + ".clustered-tmp" : temporary;
    std::filesystem::create_directories(dir);
    // 512 bucket files open at once: raise the open-file limit (the CRT's stdio
    // limit on Windows, 512 by default)
#ifdef _WIN32
    if (_getmaxstdio() < 2048) _setmaxstdio(2048);
#else
    struct rlimit lim;
    if (getrlimit(RLIMIT_NOFILE, &lim) == 0 && lim.rlim_cur < 2048) {
        lim.rlim_cur = std::min<rlim_t>(lim.rlim_max, 2048);
        setrlimit(RLIMIT_NOFILE, &lim);
    }
#endif
    const std::size_t n_buckets = 512;
    const std::size_t groups = n_reps + 1;   // with the orphans' group
    auto bucket_of = [&](std::size_t k) { return k * n_buckets / groups; };
    std::vector<std::string> paths(n_buckets);
    {
        std::vector<FILE*> files(n_buckets, nullptr);
        std::vector<std::vector<char> > buffers(n_buckets);
        for (std::size_t b = 0; b < n_buckets; ++b) {
            paths[b] = dir + "/bucket_" + std::to_string(b);
            files[b] = std::fopen(paths[b].c_str(), "wb");
            if (files[b] == nullptr)
                IMP_THROW("create_clustered_sequence_database: cannot write " << paths[b]
                          << " (open-file limit?)", IOException);
        }
        auto flush = [&](std::size_t b) {
            if (buffers[b].empty()) return;
            if (std::fwrite(buffers[b].data(), 1, buffers[b].size(), files[b]) != buffers[b].size())
                IMP_THROW("create_clustered_sequence_database: cannot write " << paths[b]
                          << " (disk full?)", IOException);
            buffers[b].clear();
        };
        member_db.scan_records([&](std::size_t row, const unsigned char* codes, std::uint64_t n,
                                   const std::string& header) {
            const std::uint32_t k = cluster_of[row];
            const std::size_t b = bucket_of(k);
            MemberRecordHead h{k, static_cast<std::uint32_t>(row), static_cast<std::uint32_t>(n),
                               static_cast<std::uint32_t>(header.size())};
            std::vector<char>& buf = buffers[b];
            const char* hp = reinterpret_cast<const char*>(&h);
            buf.insert(buf.end(), hp, hp + sizeof h);
            pack5(codes, static_cast<std::size_t>(n), buf);
            buf.insert(buf.end(), header.begin(), header.end());
            if (buf.size() >= (std::size_t(1) << 17)) flush(b);
        });
        for (std::size_t b = 0; b < n_buckets; ++b) {
            flush(b);
            std::fclose(files[b]);
        }
    }
    std::vector<std::uint32_t>().swap(cluster_of);
    // 3. the store: each representative, then its members, bucket by bucket
    pto::File container;
    if (!container.create(out, "clustered sequence database"))
        IMP_THROW("create_clustered_sequence_database: " << container.error(), IOException);
    container.set_writing_app("imp.bff create_clustered_sequence_database");
    std::size_t written = 0;
    {
        pto::StoreWriter store(container, "table", "sequences");
        const std::string header_codec = pto::can_compress("zstd") ? "zstd" : "none";
        auto sequence_columns = [&](int level, int* residues, int* headers) {
            *residues = store.add_column("residues", pto::ColumnType::UInt8, true, "none", 0, level);
            store.set_bit_width(*residues, 5);
            store.set_huffman(*residues);
            *headers = store.add_column("headers", pto::ColumnType::UInt8, true, header_codec,
                                        std::size_t(16) << 10, level);
            if (header_codec == "zstd") store.set_dictionary(*headers);
        };
        int rep_residues, rep_headers, mem_residues, mem_headers;
        sequence_columns(0, &rep_residues, &rep_headers);
        const int level = store.add_level("members");
        mem_residues = store.add_column("residues", pto::ColumnType::UInt8, true, "none", 0, level);
        mem_headers = store.add_column("headers", pto::ColumnType::UInt8, true, header_codec,
                                       std::size_t(16) << 10, level);
        if (header_codec == "zstd") store.set_dictionary(mem_headers);
        // members as edits against their representative or a closer sibling;
        // literals Huffman-coded by the representatives' composition
        store.set_reference(mem_residues, rep_residues, 5);
        {
            std::vector<std::uint64_t> freq(256, 0);
            for (int c = 0; c <= 20; ++c) freq[static_cast<std::size_t>(c)] = 1;
            std::vector<unsigned char> scratch;
            for (std::size_t k = 0; k < std::min<std::size_t>(n_reps, 20000); ++k) {
                std::uint64_t rn = 0;
                const unsigned char* rc = rep_db.get_codes(k * (n_reps / std::min<std::size_t>(n_reps, 20000)), &rn, scratch);
                for (std::uint64_t i = 0; i < rn; ++i) ++freq[rc[i]];
            }
            store.set_literal_frequencies(mem_residues, freq.data());
        }
        store.set_metadata(rep_residues, rep_db.get_metadata());

        std::vector<char> data;
        struct Ref { std::uint32_t cluster, row; std::size_t at; };
        std::vector<Ref> refs;
        std::vector<std::string> ops;          // per ref: its alignment to its reference
        std::vector<std::uint32_t> sibling;    // per ref: 0 the representative, else rows back
        std::size_t loaded = n_buckets, next = 0;   // refs[next..] are the clusters to come
        auto load = [&](std::size_t b) {
            data.clear();
            refs.clear();
            FILE* f = std::fopen(paths[b].c_str(), "rb");
            if (f == nullptr) IMP_THROW("create_clustered_sequence_database: cannot read " << paths[b], IOException);
            char chunk[1 << 16];
            for (std::size_t got; (got = std::fread(chunk, 1, sizeof chunk, f)) > 0;)
                data.insert(data.end(), chunk, chunk + got);
            std::fclose(f);
            std::remove(paths[b].c_str());
            for (std::size_t at = 0; at + sizeof(MemberRecordHead) <= data.size();) {
                MemberRecordHead h;
                std::memcpy(&h, data.data() + at, sizeof h);
                refs.push_back(Ref{h.cluster, h.row, at});
                at += sizeof h + packed_bytes(h.n_codes) + h.n_header;
            }
            std::sort(refs.begin(), refs.end(), [](const Ref& a, const Ref& c) {
                return a.cluster != c.cluster ? a.cluster < c.cluster : a.row < c.row;
            });
            // Each member's reference: the representative, or an earlier member
            // of its cluster that shares more 3-mers (by at least 0.05), up to 16
            // back and 7 deep; then its alignment to it. Clusters across threads.
            ops.assign(refs.size(), std::string());
            sibling.assign(refs.size(), 0);
            std::vector<std::pair<std::size_t, std::size_t> > groups;   // [first, end) of a cluster
            for (std::size_t q = 0; q < refs.size();) {
                std::size_t e = q;
                while (e < refs.size() && refs[e].cluster == refs[q].cluster) ++e;
                groups.push_back(std::make_pair(q, e));
                q = e;
            }
            std::atomic<std::size_t> next_group{0};
            auto align = [&]() {
                std::vector<unsigned char> scratch;
                std::vector<std::vector<unsigned char> > seqs;
                std::vector<KmerSet> sets;
                std::vector<unsigned> depth;
                for (std::size_t g; (g = next_group++) < groups.size();) {
                    const std::size_t first = groups[g].first, end = groups[g].second;
                    const std::uint32_t k = refs[first].cluster;
                    if (k >= n_reps) continue;   // orphans: literal
                    std::uint64_t rn = 0;
                    const unsigned char* rc = rep_db.get_codes(k, &rn, scratch);
                    const std::vector<unsigned char> rep(rc, rc + rn);
                    const KmerSet rep_set(rep);
                    seqs.clear();   // the last 17 members only: a window, not the cluster
                    sets.clear();
                    depth.clear();
                    const std::size_t window = 17;
                    for (std::size_t q = first; q < end; ++q) {
                        MemberRecordHead h;
                        std::memcpy(&h, data.data() + refs[q].at, sizeof h);
                        std::vector<unsigned char> m;
                        unpack5(data.data() + refs[q].at + sizeof h, h.n_codes, m);
                        const KmerSet set(m);
                        double best = set.similarity(rep_set) + 0.05;
                        std::size_t pick = 0;   // 0: the representative, else a distance back
                        const std::size_t j = q - first;
                        for (std::size_t back = 1; back <= std::min<std::size_t>(j, 16); ++back) {
                            if (depth[j - back] >= 7) continue;
                            const double sim = set.similarity(sets[(j - back) % window]);
                            if (sim > best) { best = sim; pick = back; }
                        }
                        const std::vector<unsigned char>& ref = pick ? seqs[(j - pick) % window] : rep;
                        ops[q] = align_to_reference(ref.data(), ref.size(), m.data(), m.size());
                        sibling[q] = ops[q].empty() ? 0 : static_cast<std::uint32_t>(pick);
                        depth.push_back(ops[q].empty() ? 0u : (pick ? depth[j - pick] + 1u : 1u));
                        if (seqs.size() < window) { seqs.push_back(std::move(m)); sets.push_back(set); }
                        else { seqs[j % window] = std::move(m); sets[j % window] = set; }
                    }
                }
            };
            const unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> pool;
            for (unsigned t = 1; t < n_threads; ++t) pool.emplace_back(align);
            align();
            for (std::thread& t : pool) t.join();
            loaded = b;
            next = 0;
        };
        auto emit_members = [&](std::size_t k, const unsigned char* rep, std::uint64_t rep_n) {
            if (loaded != bucket_of(k)) load(bucket_of(k));
            // the parent row owns its members before they are written: the
            // writer tracks siblings and their depth per parent
            std::uint64_t count = 0;
            while (next + count < refs.size() && refs[next + count].cluster == k) ++count;
            store.append_children(level, count);
            std::vector<std::vector<unsigned char> > written_rows;   // this cluster's members
            for (std::uint64_t written_here = 0; written_here < count; ++next, ++written_here) {
                MemberRecordHead h;
                std::memcpy(&h, data.data() + refs[next].at, sizeof h);
                const char* p = data.data() + refs[next].at + sizeof h;
                std::vector<unsigned char> m;
                unpack5(p, h.n_codes, m);
                const std::uint32_t back = sibling[next];
                const std::vector<unsigned char>* ref = back ? &written_rows[written_rows.size() - back] : nullptr;
                store.append_row_against(mem_residues, m.data(), m.size(),
                                         ref ? ref->data() : rep, ref ? ref->size() : rep_n,
                                         ops[next], back);
                store.append_row(mem_headers, p + packed_bytes(h.n_codes), h.n_header);
                written_rows.push_back(std::move(m));
                if (written_rows.size() > 17) written_rows.erase(written_rows.begin());
            }
            written += count;
        };
        rep_db.scan_records([&](std::size_t k, const unsigned char* codes, std::uint64_t n,
                                const std::string& header) {
            store.append_row(rep_residues, codes, n);
            store.append_row(rep_headers, header.data(), header.size());
            emit_members(k, codes, n);
        });
        // the orphans, if any, under an empty last representative
        if (loaded != bucket_of(n_reps)) load(bucket_of(n_reps));
        if (next < refs.size()) {
            const std::string name = "orphans";
            store.append_row(rep_residues, nullptr, 0);
            store.append_row(rep_headers, name.data(), name.size());
            emit_members(n_reps, nullptr, 0);
        }
        // the members' residues are scripts: record how many residues they hold
        {
            nlohmann::json meta = nlohmann::json::object();
            try { meta = nlohmann::json::parse(member_db.get_metadata()); } catch (...) {}
            meta["residues"] = member_db.get_number_of_residues();
            store.set_metadata(mem_residues, meta.dump());
        }
        if (!store.close())
            IMP_THROW("create_clustered_sequence_database: could not write " << out, IOException);
    }
    if (!container.commit())
        IMP_THROW("create_clustered_sequence_database: " << container.error(), IOException);
    container.close();
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return written;
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

namespace {

//! Stage 2: the members of the clusters \p found, all at once or best first in
//! growing batches until every query has enough good hits.
SequenceSearchHits search_members(const Strings& queries, const SequenceSearchHits& found,
                                  const std::function<void(std::size_t, std::vector<std::size_t>&)>& rows_of,
                                  const std::vector<std::size_t>& always, const SequenceDatabase& members,
                                  const SequenceSearchOptions& options,
                                  const SequenceClusterSearchOptions& co) {
    // clusters by their best representative E-value over all queries
    std::map<std::size_t, double> best;
    for (const SequenceSearchHit& h : found) {
        auto it = best.find(h.target);
        if (it == best.end() || h.evalue < it->second) best[h.target] = h.evalue;
    }
    std::vector<std::pair<double, std::size_t> > order;
    for (const auto& kv : best) order.push_back(std::make_pair(kv.second, kv.first));
    std::sort(order.begin(), order.end());
    auto rows_for = [&](std::size_t first, std::size_t end, bool with_always) {
        std::vector<std::size_t> rows;
        for (std::size_t i = first; i < end; ++i) rows_of(order[i].second, rows);
        if (with_always) rows.insert(rows.end(), always.begin(), always.end());
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        return rows;
    };
    if (co.min_homologues <= 0)
        return search_sequence_database_rows(queries, members, rows_for(0, order.size(), true), options);
    SequenceSearchHits out;
    std::vector<int> good(queries.size(), 0);
    std::size_t done = 0, batch = 64;
    bool first = true;
    while (done < order.size() || first) {
        const std::size_t end = std::min(order.size(), done + batch);
        const SequenceSearchHits hits =
                search_sequence_database_rows(queries, members, rows_for(done, end, first), options);
        for (const SequenceSearchHit& h : hits) {
            out.push_back(h);
            if (h.evalue <= co.stop_evalue && h.identity >= co.stop_min_identity)
                ++good[static_cast<std::size_t>(h.query)];
        }
        done = end;
        first = false;
        batch *= 2;
        if (std::all_of(good.begin(), good.end(), [&](int g) { return g >= co.min_homologues; })) break;
    }
    std::sort(out.begin(), out.end(), [](const SequenceSearchHit& a, const SequenceSearchHit& b) {
        if (a.query != b.query) return a.query < b.query;
        if (a.evalue != b.evalue) return a.evalue < b.evalue;
        return a.target < b.target;
    });
    return out;
}

SequenceSearchOptions stage_one(const SequenceSearchOptions& options,
                                const SequenceClusterSearchOptions& co) {
    SequenceSearchOptions first = options;
    first.max_candidates = std::max(co.max_representatives, options.max_candidates);
    first.max_evalue = std::max(co.max_representative_evalue, options.max_evalue);
    return first;
}

//! Stage 1: every representative scanned for k-mers, or, with an embedding
//! prefilter, only the ones nearest each query's embedding.
SequenceSearchHits search_representatives(const Strings& queries, const SequenceDatabase& representatives,
                                          const SequenceSearchOptions& options,
                                          const SequenceClusterSearchOptions& co) {
    std::string model_path = co.embedding_model, index_path = co.embedding_index;
    if (model_path.empty() && index_path.empty() && co.use_database_prefilter &&
        get_has_embedding_prefilter(representatives.get_path()))
        model_path = index_path = representatives.get_path();
    if (model_path.empty() && index_path.empty())
        return search_sequence_database(queries, representatives, stage_one(options, co));
    if (model_path.empty() || index_path.empty())
        IMP_THROW("search_clustered_sequence_database: the embedding prefilter needs both "
                  "embedding_model and embedding_index", ValueException);
    const ProteinLanguageModel model(model_path);
    const EmbeddingIndex index(index_path);
    if (model.get_projection_length() != index.get_dimension())
        IMP_THROW("search_clustered_sequence_database: " << model_path << " embeds in "
                  << model.get_projection_length() << " dimensions, " << index_path
                  << " holds " << index.get_dimension(), ValueException);
    const std::vector<float> embedded =
            model.embed(std::vector<std::string>(queries.begin(), queries.end()), true);
    const std::size_t k = static_cast<std::size_t>(std::max(co.embedding_candidates, 1));
    std::vector<std::size_t> rows;
    for (const auto& hits : index.nearest(embedded, k))
        for (const auto& h : hits)
            if (h.second < representatives.get_number_of_sequences()) rows.push_back(h.second);
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return search_sequence_database_rows(queries, representatives, rows, stage_one(options, co));
}

}  // namespace

SequenceSearchHits search_clustered_sequence_database(
        const Strings& queries, const SequenceDatabase& representatives,
        const SequenceClusters& clusters, const SequenceDatabase& members,
        const SequenceSearchOptions& options, const SequenceClusterSearchOptions& cluster_options) {
    if (clusters.get_number_of_clusters() != representatives.get_number_of_sequences())
        IMP_THROW("search_clustered_sequence_database: " << clusters.get_number_of_clusters()
                  << " clusters for " << representatives.get_number_of_sequences()
                  << " representatives: not this database's membership", ValueException);
    // Stage 1: the representatives, reaching further than the final cut-off.
    const SequenceSearchHits found =
            search_representatives(queries, representatives, options, cluster_options);
    // Stage 2: the members of the clusters found (and the orphans).
    std::vector<std::size_t> always;
    if (cluster_options.search_orphans) always = clusters.get_orphan_rows();
    return search_members(queries, found,
                          [&](std::size_t k, std::vector<std::size_t>& rows) {
                              const std::vector<std::size_t> m = clusters.get_member_rows(k);
                              rows.insert(rows.end(), m.begin(), m.end());
                          },
                          always, members, options, cluster_options);
}

SequenceSearchHits search_clustered_sequence_database(
        const Strings& queries, const SequenceDatabase& database,
        const SequenceSearchOptions& options, const SequenceClusterSearchOptions& cluster_options) {
    const SequenceDatabase members = database.get_members_database();
    const SequenceSearchHits found = search_representatives(queries, database, options, cluster_options);
    // an orphans' group is the last representative, named so: always searched
    std::vector<std::size_t> always;
    if (cluster_options.search_orphans && database.get_number_of_sequences() > 0) {
        const std::size_t last = database.get_number_of_sequences() - 1;
        if (database.get_identifier(last) == "orphans") {
            const Ints range = database.get_member_range(last);
            for (int r = range[0]; r < range[1]; ++r) always.push_back(static_cast<std::size_t>(r));
        }
    }
    // each cluster's members: one contiguous run
    return search_members(queries, found,
                          [&](std::size_t k, std::vector<std::size_t>& rows) {
                              const Ints range = database.get_member_range(k);
                              for (int r = range[0]; r < range[1]; ++r) rows.push_back(static_cast<std::size_t>(r));
                          },
                          always, members, options, cluster_options);
}

bool get_has_embedding_prefilter(const std::string& database) {
    pto::File container;
    if (!container.open(database)) return false;
    return container.find(ProteinLanguageModel::get_object_name()) != 0 &&
           container.find(EmbeddingIndex::get_object_name()) != 0;
}

void add_embedding_prefilter(const std::string& database, const std::string& model_path,
                             const std::string& index_path) {
    const SequenceDatabase db(database);
    const ProteinLanguageModel model(model_path);
    const EmbeddingIndex index(index_path);
    if (model.get_projection_length() != index.get_dimension())
        IMP_THROW("add_embedding_prefilter: " << model_path << " embeds in "
                  << model.get_projection_length() << " dimensions, " << index_path << " holds "
                  << index.get_dimension(), ValueException);
    // the orphans' group, a last empty representative, is searched in any case
    std::size_t covered = db.get_number_of_sequences();
    if (covered > 0 && db.get_identifier(covered - 1) == "orphans") --covered;
    if (index.get_number_of_vectors() < covered)
        IMP_THROW("add_embedding_prefilter: " << index_path << " holds "
                  << index.get_number_of_vectors() << " vectors for " << covered
                  << " representatives in " << database, ValueException);
    // the model's GGUF file: from a GGUF file directly, else out of a container
    const std::string temporary = db.get_path() + ".prefilter-tmp";
    std::string gguf = model_path;
    {
        std::ifstream in(model_path, std::ios::binary);
        char magic[4] = {0, 0, 0, 0};
        in.read(magic, 4);
        if (!in || std::memcmp(magic, "GGUF", 4) != 0) {
            pto::File from;
            if (!from.open(model_path) || !from.extract(from.find(ProteinLanguageModel::get_object_name()),
                                                        temporary + ".gguf"))
                IMP_THROW("add_embedding_prefilter: cannot read the model out of " << model_path,
                          IOException);
            gguf = temporary + ".gguf";
        }
    }
    pto::File container;
    if (!container.open(db.get_path(), true))
        IMP_THROW("add_embedding_prefilter: " << container.error(), IOException);
    for (const char* name : {ProteinLanguageModel::get_object_name(), EmbeddingIndex::get_object_name()})
        for (std::uint64_t old : container.find_all(name)) container.remove(old);
    bool ok = container.add_file("attachment", "gguf", ProteinLanguageModel::get_object_name(), gguf) != 0;
    {
        pto::File from;
        std::uint64_t uid = 0;
        ok = ok && from.open(index.get_path()) &&
             (uid = from.find(EmbeddingIndex::get_object_name())) != 0 &&
             from.extract(uid, temporary);
        if (ok) {
            const pto::PtoObject o = from.object(uid);   // kind and encoding as they were
            ok = container.add_file(o.kind, o.encoding, o.name, temporary) != 0;
        }
    }
    std::remove(temporary.c_str());
    if (gguf != model_path) std::remove(gguf.c_str());
    if (!ok || !container.commit())
        IMP_THROW("add_embedding_prefilter: could not write " << db.get_path() << ": "
                  << container.error(), IOException);
    container.close();
}

IMPBFF_END_NAMESPACE
