/**
 *  \file ProteinLanguageModel.cpp
 *  \brief ESM-2 from GGUF, forward pass on the CPU (Eigen).
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/ProteinLanguageModel.h>

#include <IMP/bff/internal/OutputView.h>
#include <IMP/bff/internal/ptolib.h>

#include <Eigen/Dense>
#ifdef IMP_BFF_HAS_ACCELERATE
#include <vecLib/cblas.h>
#endif

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

namespace language_model {

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> Row;

//! C = A B^T for row-major A (m x k, leading dimension lda), B (n x k, ldb)
//! and C (m x n, ldc): Accelerate's BLAS when built with it, else Eigen.
void gemm_nt(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
#ifdef IMP_BFF_HAS_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a, lda, b, ldb, 0.0f, c, ldc);
#else
    typedef Eigen::Map<const Matrix, 0, Eigen::OuterStride<>> In;
    Eigen::Map<Matrix, 0, Eigen::OuterStride<>>(c, m, n, Eigen::OuterStride<>(ldc)).noalias() =
            In(a, m, k, Eigen::OuterStride<>(lda)) * In(b, n, k, Eigen::OuterStride<>(ldb)).transpose();
#endif
}

//! C = A B for row-major A (m x k, lda), B (k x n, ldb), C (m x n, ldc).
void gemm_nn(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc) {
#ifdef IMP_BFF_HAS_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, lda, b, ldb, 0.0f, c, ldc);
#else
    typedef Eigen::Map<const Matrix, 0, Eigen::OuterStride<>> In;
    Eigen::Map<Matrix, 0, Eigen::OuterStride<>>(c, m, n, Eigen::OuterStride<>(ldc)).noalias() =
            In(a, m, k, Eigen::OuterStride<>(lda)) * In(b, k, n, Eigen::OuterStride<>(ldb));
#endif
}

float half_to_float(std::uint16_t h) {
    std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ffu;
    std::uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {                                       // subnormal: normalise
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

// GGUF version 2/3: header, key-value metadata, tensor directory, aligned data.
class Gguf {
public:
    explicit Gguf(const std::string& path) : path_(path) {
        load(path);
        if (bytes_.size() < 24 || std::memcmp(bytes_.data(), "GGUF", 4) != 0) fail("not a GGUF file");
        pos_ = 4;
        std::uint32_t version = get<std::uint32_t>();
        if (version < 2 || version > 3) fail("GGUF version " + std::to_string(version) + " unsupported");
        std::uint64_t n_tensors = get<std::uint64_t>(), n_kv = get<std::uint64_t>();
        for (std::uint64_t i = 0; i < n_kv; ++i) {
            std::string key = string();
            read_value(key, get<std::uint32_t>());
        }
        std::uint64_t alignment = number("general.alignment", 32);
        struct Info { std::vector<std::uint64_t> dims; std::uint32_t type; std::uint64_t offset; };
        std::map<std::string, Info> infos;
        for (std::uint64_t i = 0; i < n_tensors; ++i) {
            std::string name = string();
            Info t;
            t.dims.resize(get<std::uint32_t>());
            for (std::uint64_t& d : t.dims) d = get<std::uint64_t>();
            t.type = get<std::uint32_t>();
            t.offset = get<std::uint64_t>();
            infos[name] = t;
        }
        std::size_t data = (pos_ + alignment - 1) / alignment * alignment;
        for (const auto& it : infos) {
            const Info& t = it.second;
            if (t.type != 0 && t.type != 1)
                fail("tensor " + it.first + ": only F32 and F16 are read");
            std::size_t n = 1;
            for (std::uint64_t d : t.dims) n *= d;
            std::size_t start = data + t.offset, size = n * (t.type == 0 ? 4 : 2);
            if (start + size > bytes_.size()) fail("tensor " + it.first + " runs past the end");
            Tensor& out = tensors_[it.first];
            out.dims = t.dims;
            out.values.resize(n);
            const char* p = bytes_.data() + start;
            if (t.type == 0) {
                std::memcpy(out.values.data(), p, size);
            } else {
                for (std::size_t k = 0; k < n; ++k) {
                    std::uint16_t h;
                    std::memcpy(&h, p + 2 * k, 2);
                    out.values[k] = half_to_float(h);
                }
            }
        }
        bytes_.clear();
        bytes_.shrink_to_fit();
    }

    double number(const std::string& key, double otherwise) const {
        auto it = numbers_.find(key);
        return it == numbers_.end() ? otherwise : it->second;
    }
    double number(const std::string& key) const {
        auto it = numbers_.find(key);
        if (it == numbers_.end()) fail("no " + key);
        return it->second;
    }
    std::string text(const std::string& key) const {
        auto it = strings_.find(key);
        return it == strings_.end() ? std::string() : it->second;
    }
    const std::vector<std::string>& strings(const std::string& key) const {
        auto it = arrays_.find(key);
        if (it == arrays_.end()) fail("no " + key);
        return it->second;
    }
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    std::size_t length(const std::string& name) const { return tensor(name).values.size(); }
    //! A PyTorch `(rows, cols)` weight (GGUF lists dimensions innermost first).
    Matrix matrix(const std::string& name, int rows, int cols) const {
        const Tensor& t = tensor(name);
        if (t.dims.size() != 2 || int(t.dims[1]) != rows || int(t.dims[0]) != cols)
            fail("tensor " + name + " is not " + std::to_string(rows) + " x " + std::to_string(cols));
        return Eigen::Map<const Matrix>(t.values.data(), rows, cols);
    }
    Row vector(const std::string& name, int n) const {
        const Tensor& t = tensor(name);
        if (t.values.size() != std::size_t(n)) fail("tensor " + name + " is not of length " + std::to_string(n));
        return Eigen::Map<const Row>(t.values.data(), n);
    }

private:
    struct Tensor { std::vector<std::uint64_t> dims; std::vector<float> values; };

    [[noreturn]] void fail(const std::string& why) const {
        IMP_THROW("ProteinLanguageModel: " << path_ << ": " << why, IOException);
    }
    //! The GGUF bytes: the file itself, or the model object of a `.pto`
    //! container (a database carrying its own prefilter).
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) fail("cannot read the file");
        char magic[4] = {0, 0, 0, 0};
        in.read(magic, 4);
        if (in && std::memcmp(magic, "GGUF", 4) == 0) {
            in.seekg(0, std::ios::end);
            bytes_.resize(static_cast<std::size_t>(in.tellg()));
            in.seekg(0);
            in.read(bytes_.data(), static_cast<std::streamsize>(bytes_.size()));
            if (!in) fail("cannot read the file");
            return;
        }
        in.close();
        pto::File container;
        if (!container.open(path)) fail("neither a GGUF file nor a .pto container");
        const std::uint64_t uid = container.find(ProteinLanguageModel::get_object_name());
        if (uid == 0)
            fail(std::string("a .pto container without a '") +
                 ProteinLanguageModel::get_object_name() + "' object");
        const std::vector<unsigned char> data = container.read(uid);
        bytes_.assign(data.begin(), data.end());
    }
    const Tensor& tensor(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) fail("no tensor " + name);
        return it->second;
    }
    template <class T> T get() {
        if (pos_ + sizeof(T) > bytes_.size()) fail("truncated");
        T v;
        std::memcpy(&v, bytes_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string string() {
        std::uint64_t n = get<std::uint64_t>();
        if (pos_ + n > bytes_.size()) fail("truncated");
        std::string s(bytes_.data() + pos_, n);
        pos_ += n;
        return s;
    }
    double scalar(std::uint32_t type) {
        switch (type) {
            case 0: return get<std::uint8_t>();
            case 1: return get<std::int8_t>();
            case 2: return get<std::uint16_t>();
            case 3: return get<std::int16_t>();
            case 4: return get<std::uint32_t>();
            case 5: return get<std::int32_t>();
            case 6: return get<float>();
            case 7: return get<std::uint8_t>();
            case 10: return double(get<std::uint64_t>());
            case 11: return double(get<std::int64_t>());
            case 12: return get<double>();
        }
        fail("metadata type " + std::to_string(type) + " unknown");
    }
    void read_value(const std::string& key, std::uint32_t type) {
        if (type == 8) {
            strings_[key] = string();
        } else if (type == 9) {
            std::uint32_t element = get<std::uint32_t>();
            std::uint64_t n = get<std::uint64_t>();
            std::vector<std::string> values;
            for (std::uint64_t i = 0; i < n; ++i) {
                if (element == 8) values.push_back(string());
                else if (element == 9) fail(key + ": nested arrays are not read");
                else scalar(element);                    // numeric arrays: skipped
            }
            if (element == 8) arrays_[key] = values;
        } else {
            numbers_[key] = scalar(type);
        }
    }

    std::string path_;
    std::vector<char> bytes_;
    std::size_t pos_ = 0;
    std::map<std::string, double> numbers_;
    std::map<std::string, std::string> strings_;
    std::map<std::string, std::vector<std::string>> arrays_;
    std::map<std::string, Tensor> tensors_;
};

struct Linear {
    Matrix w;   // (out, in)
    Row b;
    void load(const Gguf& g, const std::string& name, int out, int in, bool bias = true) {
        w = g.matrix(name + ".weight", out, in);
        b = bias ? g.vector(name + ".bias", out) : Row::Zero(out);
    }
    Matrix operator()(const Matrix& x) const {
        Matrix y(x.rows(), w.rows());
        gemm_nt(int(x.rows()), int(w.rows()), int(w.cols()), x.data(), int(x.cols()), w.data(),
                int(w.cols()), y.data(), int(y.cols()));
        y.rowwise() += b;
        return y;
    }
};

struct Norm {
    Row g, b;
    void load(const Gguf& gg, const std::string& name, int d) {
        g = gg.vector(name + ".weight", d);
        b = gg.vector(name + ".bias", d);
    }
    Matrix operator()(const Matrix& x, float eps) const {
        Matrix y(x.rows(), x.cols());
        for (Eigen::Index r = 0; r < x.rows(); ++r) {
            float mean = x.row(r).mean();
            Row c = x.row(r).array() - mean;
            float var = c.squaredNorm() / float(x.cols());
            y.row(r) = (c / std::sqrt(var + eps)).cwiseProduct(g) + b;
        }
        return y;
    }
};

void gelu(Matrix& x) {
    x = x.unaryExpr([](float v) { return 0.5f * v * (1.0f + std::erf(v * float(M_SQRT1_2))); });
}

}  // namespace language_model

struct ProteinLanguageModel::Weights {
    std::string name;
    int d = 0, heads = 0, ffn = 0, layers = 0, max_length = 1022, projection = 0;
    float eps = 1e-5f, theta = 10000.f, embedding_scale = 1.f;
    int cls = 0, eos = 2, unk = 3;
    int token[256];
    language_model::Matrix embeddings;   // (vocabulary, d)
    struct Layer { language_model::Norm attention_norm, ffn_norm; language_model::Linear q, k, v, o, up, down; };
    std::vector<Layer> blocks;
    language_model::Norm final_norm;
    language_model::Row mean, std;
    language_model::Linear head0, head1, head2, skip;
    // contact head: logistic regression over layers x heads attention maps
    language_model::Row contact_w;   // empty: no contact head
    float contact_b = 0.f;
    // language-model head: dense, GELU, LayerNorm, then the token embeddings (tied) and a bias
    language_model::Linear lm_dense;
    language_model::Norm lm_norm;
    language_model::Row lm_bias;     // empty: no language-model head
    int aa_token[20];

    //! The last layer for the residues; with \p contacts, also the contact
    //! logits' accumulator (`n x n`, before the bias and the sigmoid).
    language_model::Matrix forward(const std::string& sequence,
                                   language_model::Matrix* contacts = nullptr) const;
    std::vector<float> project(const language_model::Row& pooled) const;
};

language_model::Matrix ProteinLanguageModel::Weights::forward(const std::string& sequence,
                                                              language_model::Matrix* contacts) const {
    const int n = std::min<int>(int(sequence.size()), max_length), L = n + 2, hd = d / heads;
    language_model::Matrix x(L, d);
    x.row(0) = embeddings.row(cls);
    for (int i = 0; i < n; ++i) x.row(i + 1) = embeddings.row(token[static_cast<unsigned char>(sequence[i])]);
    x.row(L - 1) = embeddings.row(eos);
    x *= embedding_scale;
    // rotary angles: position p, pair i -> p * theta^(-2i/hd)
    language_model::Matrix cosine(L, hd / 2), sine(L, hd / 2);
    for (int p = 0; p < L; ++p)
        for (int i = 0; i < hd / 2; ++i) {
            float inv = 1.0f / std::pow(theta, float(2 * i) / float(hd));
            float a = float(p) * inv;
            cosine(p, i) = std::cos(a);
            sine(p, i) = std::sin(a);
        }
    auto rotate = [&](language_model::Matrix& m) {
        for (int p = 0; p < L; ++p)
            for (int h = 0; h < heads; ++h) {
                float* v = m.row(p).data() + h * hd;
                for (int i = 0; i < hd / 2; ++i) {
                    float a = v[i], b = v[i + hd / 2], c = cosine(p, i), s = sine(p, i);
                    v[i] = a * c - b * s;
                    v[i + hd / 2] = b * c + a * s;
                }
            }
    };
    const float scale = 1.0f / std::sqrt(float(hd));
    language_model::Matrix attended(L, d), scores(L, L), symmetric(n, n);
    if (contacts) *contacts = language_model::Matrix::Zero(n, n);
    int layer_index = 0;
    for (const Layer& layer : blocks) {
        language_model::Matrix h = layer.attention_norm(x, eps);
        language_model::Matrix q = layer.q(h), k = layer.k(h), v = layer.v(h);
        q *= scale;
        rotate(q);
        rotate(k);
        for (int a = 0; a < heads; ++a) {
            language_model::gemm_nt(L, L, hd, q.data() + a * hd, d, k.data() + a * hd, d, scores.data(), L);
            for (int r = 0; r < L; ++r) {
                float top = scores.row(r).maxCoeff();
                scores.row(r) = (scores.row(r).array() - top).exp();
                scores.row(r) /= scores.row(r).sum();
            }
            if (contacts && n > 0) {
                // ESM's contact head: the residues' block (no <cls>, no <eos>),
                // symmetrised, average-product corrected, weighted per head
                symmetric = scores.block(1, 1, n, n) + scores.block(1, 1, n, n).transpose();
                const Eigen::VectorXf rows = symmetric.rowwise().sum();
                const language_model::Row cols = symmetric.colwise().sum();
                const float total = rows.sum();
                const float w = contact_w[layer_index * heads + a];
                if (total != 0.f) *contacts += w * (symmetric - rows * cols / total);
                else *contacts += w * symmetric;
            }
            language_model::gemm_nn(L, hd, L, scores.data(), L, v.data() + a * hd, d,
                                    attended.data() + a * hd, d);
        }
        x += layer.o(attended);
        language_model::Matrix f = layer.up(layer.ffn_norm(x, eps));
        language_model::gelu(f);
        x += layer.down(f);
        ++layer_index;
    }
    return final_norm(x, eps).middleRows(1, n);
}

std::vector<float> ProteinLanguageModel::Weights::project(const language_model::Row& pooled) const {
    language_model::Matrix x = ((pooled - mean).array() / std.array()).matrix();
    language_model::Matrix y = head0(x);
    language_model::gelu(y);
    y = head1(y);
    language_model::gelu(y);
    y = head2(y) + skip(x);
    y /= std::max(y.norm(), 1e-12f);
    return std::vector<float>(y.data(), y.data() + y.size());
}

ProteinLanguageModel::ProteinLanguageModel(const std::string& path) : path_(path) {
    language_model::Gguf g(path);
    if (g.text("general.architecture") != "esm2")
        IMP_THROW("ProteinLanguageModel: " << path << " is not an ESM-2 model (general.architecture '"
                                           << g.text("general.architecture") << "')",
                  IOException);
    auto w = std::make_shared<Weights>();
    w->name = g.text("general.name");
    w->layers = int(g.number("esm2.block_count"));
    w->d = int(g.number("esm2.embedding_length"));
    w->heads = int(g.number("esm2.head_count"));
    w->ffn = int(g.number("esm2.feed_forward_length"));
    w->eps = float(g.number("esm2.layer_norm_epsilon", 1e-5));
    w->theta = float(g.number("esm2.rope_theta", 10000.0));
    w->max_length = int(g.number("esm2.max_length", 1022));
    // training masked 15 % of the tokens, 80 % of those by <mask>: without
    // masks the embeddings are scaled by 1 - 0.12
    w->embedding_scale = g.number("esm2.token_dropout", 0) != 0 ? 1.0f - 0.15f * 0.8f : 1.0f;
    if (w->d % w->heads != 0 || (w->d / w->heads) % 2 != 0)
        IMP_THROW("ProteinLanguageModel: " << path << ": embedding_length " << w->d
                                           << " does not split into " << w->heads << " even heads",
                  IOException);
    const std::vector<std::string>& vocabulary = g.strings("tokenizer.ggml.tokens");
    auto index = [&](const std::string& t) {
        for (std::size_t i = 0; i < vocabulary.size(); ++i)
            if (vocabulary[i] == t) return int(i);
        IMP_THROW("ProteinLanguageModel: " << path << ": no token " << t, IOException);
    };
    w->cls = index("<cls>");
    w->eos = index("<eos>");
    w->unk = index("<unk>");
    for (int c = 0; c < 256; ++c) w->token[c] = w->unk;
    for (std::size_t i = 0; i < vocabulary.size(); ++i)
        if (vocabulary[i].size() == 1 && std::isalpha(static_cast<unsigned char>(vocabulary[i][0]))) {
            w->token[static_cast<unsigned char>(vocabulary[i][0])] = int(i);
            w->token[std::tolower(static_cast<unsigned char>(vocabulary[i][0]))] = int(i);
        }
    w->embeddings = g.matrix("embeddings.word_embeddings.weight", int(vocabulary.size()), w->d);
    w->blocks.resize(w->layers);
    for (int i = 0; i < w->layers; ++i) {
        std::string p = "encoder.layer." + std::to_string(i) + ".";
        Weights::Layer& l = w->blocks[i];
        l.attention_norm.load(g, p + "attention.LayerNorm", w->d);
        l.q.load(g, p + "attention.self.query", w->d, w->d);
        l.k.load(g, p + "attention.self.key", w->d, w->d);
        l.v.load(g, p + "attention.self.value", w->d, w->d);
        l.o.load(g, p + "attention.output.dense", w->d, w->d);
        l.ffn_norm.load(g, p + "LayerNorm", w->d);
        l.up.load(g, p + "intermediate.dense", w->ffn, w->d);
        l.down.load(g, p + "output.dense", w->d, w->ffn);
    }
    w->final_norm.load(g, "encoder.emb_layer_norm_after", w->d);
    if (g.has("head.skip.weight")) {
        w->projection = int(g.number("head.output_length"));
        int hidden = int(g.length("head.mlp.0.bias"));
        w->mean = g.vector("head.mean", w->d);
        w->std = g.vector("head.std", w->d);
        w->head0.load(g, "head.mlp.0", hidden, w->d);
        w->head1.load(g, "head.mlp.2", hidden, hidden);
        w->head2.load(g, "head.mlp.4", w->projection, hidden);
        w->skip.load(g, "head.skip", w->projection, w->d, false);
    }
    if (g.has("lm_head.bias")) {
        w->lm_dense.load(g, "lm_head.dense", w->d, w->d);
        w->lm_norm.load(g, "lm_head.layer_norm", w->d);
        w->lm_bias = g.vector("lm_head.bias", int(vocabulary.size()));
        static const char* const aa = "ACDEFGHIKLMNPQRSTVWY";
        for (int a = 0; a < 20; ++a) w->aa_token[a] = index(std::string(1, aa[a]));
    }
    if (g.has("contact_head.regression.weight")) {
        w->contact_w = g.matrix("contact_head.regression.weight", 1, w->layers * w->heads).row(0);
        w->contact_b = g.has("contact_head.regression.bias")
                               ? g.vector("contact_head.regression.bias", 1)[0] : 0.f;
    }
    w_ = w;
}

std::string ProteinLanguageModel::get_name() const { return w_ ? w_->name : std::string(); }
int ProteinLanguageModel::get_embedding_length() const { return w_ ? w_->d : 0; }
int ProteinLanguageModel::get_number_of_layers() const { return w_ ? w_->layers : 0; }
int ProteinLanguageModel::get_projection_length() const { return w_ ? w_->projection : 0; }
int ProteinLanguageModel::get_max_length() const { return w_ ? w_->max_length : 0; }
bool ProteinLanguageModel::get_has_contact_head() const { return w_ && w_->contact_w.size() > 0; }
bool ProteinLanguageModel::get_has_language_model_head() const { return w_ && w_->lm_bias.size() > 0; }

std::vector<double> ProteinLanguageModel::log_probabilities(const std::string& sequence, int* n) const {
    if (!w_) IMP_THROW("ProteinLanguageModel: no model loaded", ValueException);
    if (w_->lm_bias.size() == 0)
        IMP_THROW("ProteinLanguageModel: " << path_ << " has no language-model head", ValueException);
    language_model::Matrix h = w_->forward(sequence);                 // n x d, final LayerNorm applied
    language_model::Matrix x = w_->lm_dense(h);
    language_model::gelu(x);
    x = w_->lm_norm(x, w_->eps);
    *n = int(x.rows());
    std::vector<double> out(std::size_t(*n) * 20);
    for (int i = 0; i < *n; ++i) {
        // logits over the whole vocabulary (log-softmax normalises over all tokens, as ESM does)
        Eigen::VectorXf logits = w_->embeddings * x.row(i).transpose();
        logits += w_->lm_bias.transpose();
        const float top = logits.maxCoeff();
        const double lse = double(top) + std::log((logits.array() - top).exp().sum());
        for (int a = 0; a < 20; ++a) out[std::size_t(i) * 20 + a] = double(logits[w_->aa_token[a]]) - lse;
    }
    return out;
}

void ProteinLanguageModel::get_log_probabilities(const std::string& sequence, double** out_matrix,
                                                 int* n_out_rows, int* n_out_cols) const {
    int n = 0;
    const std::vector<double> lp = log_probabilities(sequence, &n);
    if (out_matrix == nullptr || n_out_rows == nullptr || n_out_cols == nullptr) return;
    int n_flat = 0;
    double* buffer = internal::new_double_view(lp.size(), out_matrix, &n_flat);
    *n_out_rows = 0;
    *n_out_cols = 20;
    if (buffer == nullptr) return;
    if (!lp.empty()) std::memcpy(buffer, lp.data(), lp.size() * sizeof(double));
    *n_out_rows = n;
}

Floats ProteinLanguageModel::get_site_tolerance(const std::string& sequence, const std::string& residue) const {
    static const std::string alphabet = "ACDEFGHIKLMNPQRSTVWY";
    int target = -1;
    if (!residue.empty()) {
        if (residue.size() != 1 || alphabet.find(char(std::toupper(residue[0]))) == std::string::npos)
            IMP_THROW("get_site_tolerance: residue must be one amino-acid letter or empty", ValueException);
        target = int(alphabet.find(char(std::toupper(residue[0]))));
    }
    int n = 0;
    const std::vector<double> lp = log_probabilities(sequence, &n);
    Floats out(std::size_t(n), std::numeric_limits<double>::quiet_NaN());
    for (int i = 0; i < n; ++i) {
        const std::size_t wt = alphabet.find(char(std::toupper(sequence[std::size_t(i)])));
        if (wt == std::string::npos) continue;                          // X, B, Z ...: no wild-type term
        const double* row = &lp[std::size_t(i) * 20];
        if (target >= 0) {
            out[std::size_t(i)] = row[target] - row[wt];
        } else {
            double mean = 0;
            for (int a = 0; a < 20; ++a) mean += row[a];
            out[std::size_t(i)] = mean / 20.0 - row[wt];
        }
    }
    return out;
}

std::vector<double> ProteinLanguageModel::contacts(const std::string& sequence, int* n) const {
    if (!w_) IMP_THROW("ProteinLanguageModel: no model loaded", ValueException);
    if (w_->contact_w.size() == 0)
        IMP_THROW("ProteinLanguageModel: " << path_ << " has no contact head", ValueException);
    language_model::Matrix logits;
    w_->forward(sequence, &logits);
    *n = int(logits.rows());
    std::vector<double> out(std::size_t(*n) * std::size_t(*n));
    for (int i = 0; i < *n; ++i)
        for (int j = 0; j < *n; ++j)
            out[std::size_t(i) * *n + j] = 1.0 / (1.0 + std::exp(-double(logits(i, j) + w_->contact_b)));
    return out;
}

void ProteinLanguageModel::get_contacts(const std::string& sequence, double** out_matrix,
                                        int* n_out_rows, int* n_out_cols) const {
    int n = 0;
    const std::vector<double> c = contacts(sequence, &n);
    if (out_matrix == nullptr || n_out_rows == nullptr || n_out_cols == nullptr) return;
    int n_flat = 0;
    double* buffer = internal::new_double_view(c.size(), out_matrix, &n_flat);
    *n_out_rows = 0;
    *n_out_cols = n;
    if (buffer == nullptr) return;
    if (!c.empty()) std::memcpy(buffer, c.data(), c.size() * sizeof(double));
    *n_out_rows = n;
}

std::vector<double> probe_pair_contacts(const ProteinLanguageModel& model, const std::string& sequence,
                                        int* pair_residues, int n_pair_rows, int n_pair_cols,
                                        int residue_offset) {
    if (n_pair_cols != 2) IMP_THROW("probe_pair_contacts: two residues per pair", ValueException);
    int n = 0;
    const std::vector<double> c = model.contacts(sequence, &n);
    std::vector<double> out(static_cast<std::size_t>(n_pair_rows),
                            std::numeric_limits<double>::quiet_NaN());
    for (int p = 0; p < n_pair_rows; ++p) {
        const int a = pair_residues[2 * p] - residue_offset - 1;     // 0-based in the sequence
        const int b = pair_residues[2 * p + 1] - residue_offset - 1;
        if (a < 0 || b < 0 || a >= n || b >= n || a == b) continue;
        out[static_cast<std::size_t>(p)] = c[std::size_t(a) * n + b];
    }
    return out;
}

int ProteinLanguageModel::compute(const std::string& sequence, std::vector<float>& out) const {
    if (!w_) IMP_THROW("ProteinLanguageModel: no model loaded", ValueException);
    language_model::Matrix h = w_->forward(sequence);
    out.assign(h.data(), h.data() + h.size());
    return int(h.rows());
}

Floats ProteinLanguageModel::get_residue_embeddings(const std::string& sequence) const {
    std::vector<float> out;
    compute(sequence, out);
    return Floats(out.begin(), out.end());
}

Floats ProteinLanguageModel::get_embedding(const std::string& sequence) const {
    std::vector<float> v = embed({sequence}, false);
    return Floats(v.begin(), v.end());
}

Floats ProteinLanguageModel::get_projected_embedding(const std::string& sequence) const {
    std::vector<float> v = embed({sequence}, true);
    return Floats(v.begin(), v.end());
}

std::vector<float> ProteinLanguageModel::embed(const std::vector<std::string>& sequences,
                                               bool project) const {
    if (!w_) IMP_THROW("ProteinLanguageModel: no model loaded", ValueException);
    if (project && w_->projection == 0)
        IMP_THROW("ProteinLanguageModel: " << path_ << " has no projection head", ValueException);
    const int width = project ? w_->projection : w_->d;
    std::vector<float> out(sequences.size() * std::size_t(width), 0.f);
    const int n = int(sequences.size());
#pragma omp parallel for schedule(dynamic) if (n > 1)
    for (int i = 0; i < n; ++i) {
        if (sequences[i].empty()) continue;
        language_model::Row pooled = w_->forward(sequences[i]).colwise().mean();
        if (project) {
            std::vector<float> z = w_->project(pooled);
            std::copy(z.begin(), z.end(), out.begin() + std::size_t(i) * width);
        } else {
            std::copy(pooled.data(), pooled.data() + width, out.begin() + std::size_t(i) * width);
        }
    }
    return out;
}

IMPBFF_END_NAMESPACE
