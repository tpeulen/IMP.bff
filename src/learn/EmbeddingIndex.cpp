/**
 *  \file EmbeddingIndex.cpp
 *  \brief Product-quantised embeddings: build (k-means per subspace) and
 *         lookup-table search.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/EmbeddingIndex.h>

#include <ptolib/ptolib.h>
#include <IMP/bff/internal/json.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>

IMPBFF_BEGIN_NAMESPACE

namespace embedding_index {

const char* const kObject = EmbeddingIndex::get_object_name();
const int kCentroids = 256;

float half_to_float(std::uint16_t h) {
    std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ffu;
    std::uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; --exp; }
            bits = sign | (exp << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

//! A NumPy `.npy` file of `n x d` float16 or float32, C order, read in blocks.
class Npy {
public:
    explicit Npy(const std::string& path) : path_(path), in_(path, std::ios::binary) {
        char magic[8];
        if (!in_.read(magic, 8) || std::memcmp(magic, "\x93NUMPY", 6) != 0) fail("not a .npy file");
        std::uint32_t header_length = 0;
        if (magic[6] == 1) {
            std::uint16_t h;
            in_.read(reinterpret_cast<char*>(&h), 2);
            header_length = h;
        } else {
            in_.read(reinterpret_cast<char*>(&header_length), 4);
        }
        std::string header(header_length, '\0');
        in_.read(&header[0], header_length);
        if (!in_) fail("truncated header");
        if (header.find("'fortran_order': False") == std::string::npos) fail("not C order");
        if (header.find("'<f2'") != std::string::npos) width_ = 2;
        else if (header.find("'<f4'") != std::string::npos) width_ = 4;
        else fail("neither float16 nor float32");
        std::size_t s = header.find("'shape': (");
        if (s == std::string::npos) fail("no shape");
        s += 10;
        rows_ = std::stoull(header.substr(s));
        std::size_t comma = header.find(',', s);
        cols_ = std::stoull(header.substr(comma + 1));
        data_ = static_cast<std::size_t>(in_.tellg());
    }
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    //! Rows [first, first + n) as floats into out.
    void read(std::size_t first, std::size_t n, float* out) {
        in_.seekg(static_cast<std::streamoff>(data_ + first * cols_ * width_));
        buffer_.resize(n * cols_ * width_);
        if (!in_.read(buffer_.data(), static_cast<std::streamsize>(buffer_.size()))) fail("truncated data");
        if (width_ == 4) {
            std::memcpy(out, buffer_.data(), buffer_.size());
        } else {
            for (std::size_t k = 0; k < n * cols_; ++k) {
                std::uint16_t h;
                std::memcpy(&h, buffer_.data() + 2 * k, 2);
                out[k] = half_to_float(h);
            }
        }
    }

private:
    [[noreturn]] void fail(const std::string& why) const {
        IMP_THROW("create_embedding_index: " << path_ << ": " << why, IOException);
    }
    std::string path_;
    std::ifstream in_;
    std::size_t rows_ = 0, cols_ = 0, data_ = 0, width_ = 4;
    std::vector<char> buffer_;
};

unsigned threads() {
    unsigned t = std::thread::hardware_concurrency();
    return t == 0 ? 4 : t;
}

template <class F> void parallel(std::size_t n, F f) {
    std::atomic<std::size_t> next(0);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < std::min<std::size_t>(threads(), n); ++t)
        pool.emplace_back([&]() {
            for (std::size_t i; (i = next++) < n;) f(i);
        });
    for (std::thread& t : pool) t.join();
}

//! k-means (Lloyd) of \p n points of \p dsub dimensions into kCentroids;
//! centroids start at distinct sample points.
std::vector<float> kmeans(const std::vector<float>& points, std::size_t n, int dsub, int iterations,
                          unsigned seed) {
    std::vector<float> c(std::size_t(kCentroids) * dsub);
    std::mt19937 rng(seed);
    std::vector<std::size_t> pick(n);
    for (std::size_t i = 0; i < n; ++i) pick[i] = i;
    std::shuffle(pick.begin(), pick.end(), rng);
    for (int k = 0; k < kCentroids; ++k)
        std::copy_n(&points[pick[k % n] * dsub], dsub, &c[std::size_t(k) * dsub]);
    std::vector<int> assign(n);
    for (int it = 0; it < iterations; ++it) {
        for (std::size_t i = 0; i < n; ++i) {
            const float* p = &points[i * dsub];
            float best = std::numeric_limits<float>::max();
            int arg = 0;
            for (int k = 0; k < kCentroids; ++k) {
                const float* q = &c[std::size_t(k) * dsub];
                float dist = 0;
                for (int j = 0; j < dsub; ++j) dist += (p[j] - q[j]) * (p[j] - q[j]);
                if (dist < best) { best = dist; arg = k; }
            }
            assign[i] = arg;
        }
        std::vector<double> sum(c.size(), 0.0);
        std::vector<std::size_t> count(kCentroids, 0);
        for (std::size_t i = 0; i < n; ++i) {
            ++count[assign[i]];
            for (int j = 0; j < dsub; ++j) sum[std::size_t(assign[i]) * dsub + j] += points[i * dsub + j];
        }
        for (int k = 0; k < kCentroids; ++k) {
            if (count[k] == 0) {                     // empty: restart at a random point
                std::copy_n(&points[rng() % n * dsub], dsub, &c[std::size_t(k) * dsub]);
                continue;
            }
            for (int j = 0; j < dsub; ++j)
                c[std::size_t(k) * dsub + j] = float(sum[std::size_t(k) * dsub + j] / double(count[k]));
        }
    }
    return c;
}

}  // namespace embedding_index

std::size_t create_embedding_index(const Strings& vectors, const std::string& out,
                                   const std::string& model, int subspaces, int sample,
                                   int iterations, const std::string& rows_file) {
    using namespace embedding_index;
    if (vectors.empty()) IMP_THROW("create_embedding_index: no vector files", ValueException);
    std::vector<std::unique_ptr<Npy>> files;
    std::size_t n = 0, d = 0;
    for (const std::string& path : vectors) {
        files.emplace_back(new Npy(path));
        if (d == 0) d = files.back()->cols();
        if (files.back()->cols() != d)
            IMP_THROW("create_embedding_index: " << path << " has " << files.back()->cols()
                                                 << " columns, the first file " << d, IOException);
        n += files.back()->rows();
    }
    if (subspaces < 1 || d % std::size_t(subspaces) != 0)
        IMP_THROW("create_embedding_index: " << subspaces << " subspaces do not divide " << d
                                             << " dimensions", ValueException);
    std::vector<std::uint32_t> rows;
    if (!rows_file.empty()) {
        std::ifstream in(rows_file);
        if (!in) IMP_THROW("create_embedding_index: cannot read " << rows_file, IOException);
        for (std::uint64_t r; in >> r;) rows.push_back(static_cast<std::uint32_t>(r));
        if (rows.size() != n)
            IMP_THROW("create_embedding_index: " << rows_file << " has " << rows.size()
                                                 << " rows for " << n << " vectors", IOException);
    }
    const int m = subspaces, dsub = int(d) / subspaces;

    // 1. an even sample, by subspace: sample[s] is `ns x dsub`
    const std::size_t ns = std::min<std::size_t>(std::max(sample, kCentroids), n);
    std::vector<std::vector<float>> pieces(m, std::vector<float>(ns * dsub));
    {
        std::vector<float> v(d);
        std::size_t file = 0, base = 0;
        for (std::size_t i = 0; i < ns; ++i) {
            std::size_t g = i * n / ns;
            while (g >= base + files[file]->rows()) base += files[file++]->rows();
            files[file]->read(g - base, 1, v.data());
            for (int s = 0; s < m; ++s) std::copy_n(&v[std::size_t(s) * dsub], dsub, &pieces[s][i * dsub]);
        }
    }
    // 2. centroids per subspace
    std::vector<float> centroids(std::size_t(m) * kCentroids * dsub);
    parallel(m, [&](std::size_t s) {
        std::vector<float> c = kmeans(pieces[s], ns, dsub, iterations, unsigned(17 + s));
        std::copy(c.begin(), c.end(), centroids.begin() + s * kCentroids * dsub);
    });
    pieces.clear();

    // 3. encode, block by block, and write
    pto::File container;
    if (!container.create(out, "embedding index"))
        IMP_THROW("create_embedding_index: " << container.error(), IOException);
    container.set_writing_app("imp.bff create_embedding_index");
    {
        pto::StoreWriter store(container, "table", kObject);
        const int col_centroids = store.add_column("centroids", pto::ColumnType::Float32, false, "none");
        // segments of whole rows: 512 Ki vectors each, scanned in parallel
        const int col_codes = store.add_column("codes", pto::ColumnType::UInt8, false, "none",
                                               std::size_t(m) << 19);
        nlohmann::json meta;
        meta["model"] = model;
        meta["dimension"] = d;
        meta["subspaces"] = m;
        meta["centroids"] = kCentroids;
        meta["vectors"] = n;
        meta["metric"] = "inner product";
        store.set_metadata(col_codes, meta.dump());
        store.append(col_centroids, centroids.data(), centroids.size());
        if (!rows.empty()) {
            const int col_rows = store.add_column("rows", pto::ColumnType::UInt32, false, "none");
            store.append(col_rows, rows.data(), rows.size());
        }
        const std::size_t block = 65536;
        std::vector<float> v(block * d);
        std::vector<unsigned char> codes(block * m);
        for (auto& f : files) {
            for (std::size_t first = 0; first < f->rows(); first += block) {
                const std::size_t count = std::min(block, f->rows() - first);
                f->read(first, count, v.data());
                parallel((count + 255) / 256, [&](std::size_t b) {
                    for (std::size_t i = b * 256; i < std::min(count, (b + 1) * 256); ++i)
                        for (int s = 0; s < m; ++s) {
                            const float* p = &v[i * d + std::size_t(s) * dsub];
                            const float* c = &centroids[std::size_t(s) * kCentroids * dsub];
                            float best = std::numeric_limits<float>::max();
                            int arg = 0;
                            for (int k = 0; k < kCentroids; ++k) {
                                float dist = 0;
                                for (int j = 0; j < dsub; ++j) {
                                    float e = p[j] - c[k * dsub + j];
                                    dist += e * e;
                                }
                                if (dist < best) { best = dist; arg = k; }
                            }
                            codes[i * m + s] = static_cast<unsigned char>(arg);
                        }
                });
                store.append(col_codes, codes.data(), count * m);
            }
        }
        if (!store.close() || !container.commit())
            IMP_THROW("create_embedding_index: could not write " << out, IOException);
    }
    container.close();
    return n;
}

EmbeddingIndex::EmbeddingIndex(const std::string& path) : path_(path) {
    using namespace embedding_index;
    pto::File container;
    if (!container.open(path))
        IMP_THROW("EmbeddingIndex: " << path << ": " << container.error(), IOException);
    const std::uint64_t uid = container.find(kObject);
    if (uid == 0) IMP_THROW("EmbeddingIndex: " << path << " holds no embedding index", IOException);
    reader_ = std::make_shared<pto::StoreReader>(container, uid);
    nlohmann::json meta = nlohmann::json::parse(reader_->metadata("codes"));
    model_ = meta.value("model", std::string());
    d_ = meta["dimension"].get<int>();
    m_ = meta["subspaces"].get<int>();
    n_ = meta["vectors"].get<std::size_t>();
    centroids_.resize(std::size_t(m_) * kCentroids * (d_ / m_));
    reader_->read("centroids", 0, centroids_.size(), centroids_.data());
    if (reader_->has_column("rows")) {
        rows_.resize(n_);
        reader_->read("rows", 0, n_, rows_.data());
    }
}

std::vector<std::vector<std::pair<float, std::size_t>>> EmbeddingIndex::nearest(
        const std::vector<float>& queries, std::size_t k) const {
    using namespace embedding_index;
    typedef std::pair<float, std::size_t> Hit;
    if (!reader_) IMP_THROW("EmbeddingIndex: no index open", ValueException);
    if (queries.size() % std::size_t(d_) != 0)
        IMP_THROW("EmbeddingIndex: queries are not rows of " << d_ << " values", ValueException);
    const std::size_t nq = queries.size() / d_;
    const int dsub = d_ / m_;
    // lookup tables: query x subspace x centroid
    std::vector<float> table(nq * m_ * kCentroids);
    for (std::size_t q = 0; q < nq; ++q)
        for (int s = 0; s < m_; ++s)
            for (int c = 0; c < kCentroids; ++c) {
                const float* a = &queries[q * d_ + std::size_t(s) * dsub];
                const float* b = &centroids_[(std::size_t(s) * kCentroids + c) * dsub];
                float dot = 0;
                for (int j = 0; j < dsub; ++j) dot += a[j] * b[j];
                table[(q * m_ + s) * kCentroids + c] = dot;
            }
    // vectors per row may repeat: keep extra so k distinct rows survive
    const std::size_t keep = rows_.empty() ? k : k * 2;
    typedef std::priority_queue<Hit, std::vector<Hit>, std::greater<Hit>> Heap;   // min-heap
    std::mutex lock;
    const std::size_t segments = reader_->n_segments("codes");
    std::vector<Heap> merged(nq);
    std::atomic<std::size_t> next(0);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < std::min<std::size_t>(threads(), segments); ++t)
        pool.emplace_back([&]() {
            std::vector<Heap> local(nq);
            std::vector<unsigned char> scratch;
            for (std::size_t g; (g = next++) < segments;) {
                const pto::SegmentInfo info = reader_->segment("codes", g);
                const unsigned char* codes =
                        static_cast<const unsigned char*>(reader_->segment_data("codes", g, scratch));
                const std::size_t first = static_cast<std::size_t>(info.first_value) / m_,
                                  count = static_cast<std::size_t>(info.n_values) / m_;
                for (std::size_t q = 0; q < nq; ++q) {
                    const float* tq = &table[q * m_ * kCentroids];
                    Heap& h = local[q];
                    for (std::size_t i = 0; i < count; ++i) {
                        const unsigned char* c = codes + i * m_;
                        float score = 0;
                        for (int s = 0; s < m_; ++s) score += tq[s * kCentroids + c[s]];
                        if (h.size() < keep) h.emplace(score, first + i);
                        else if (score > h.top().first) { h.pop(); h.emplace(score, first + i); }
                    }
                }
            }
            std::lock_guard<std::mutex> guard(lock);
            for (std::size_t q = 0; q < nq; ++q)
                for (; !local[q].empty(); local[q].pop()) {
                    Heap& h = merged[q];
                    if (h.size() < keep) h.push(local[q].top());
                    else if (local[q].top().first > h.top().first) { h.pop(); h.push(local[q].top()); }
                }
        });
    for (std::thread& t : pool) t.join();
    std::vector<std::vector<Hit>> out(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        std::vector<Hit> all;
        for (; !merged[q].empty(); merged[q].pop()) all.push_back(merged[q].top());
        std::sort(all.begin(), all.end(), std::greater<Hit>());
        std::unordered_map<std::size_t, bool> seen;
        for (const Hit& h : all) {
            std::size_t row = rows_.empty() ? h.second : rows_[h.second];
            if (seen.emplace(row, true).second) out[q].emplace_back(h.first, row);
            if (out[q].size() == k) break;
        }
    }
    return out;
}

Ints EmbeddingIndex::get_nearest(const Floats& query, int k) const {
    std::vector<float> q(query.begin(), query.end());
    // the result must outlive the loop: a range over an element of a temporary dangles
    const auto hits = nearest(q, std::size_t(std::max(k, 0)));
    Ints out;
    for (const auto& h : hits[0]) out.push_back(int(h.second));
    return out;
}

Floats EmbeddingIndex::get_nearest_scores(const Floats& query, int k) const {
    std::vector<float> q(query.begin(), query.end());
    const auto hits = nearest(q, std::size_t(std::max(k, 0)));
    Floats out;
    for (const auto& h : hits[0]) out.push_back(h.first);
    return out;
}

IMPBFF_END_NAMESPACE
