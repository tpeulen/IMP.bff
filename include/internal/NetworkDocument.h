/**
 *  \file IMP/bff/internal/NetworkDocument.h
 *  \brief The one door between a network and its msgpack bytes.
 *
 * msgpack is bff's native format for every neural-network document: the
 * `bff.neural_net` network itself, the action policy a model search reads
 * (the same document plus a `"temperature"`), and the documents that nest a
 * network as a map (`bff.hmm_surrogate`). float64 weights survive exactly,
 * the bytes are a third the size of the text, and there is no second
 * reader: every producer and consumer in bff goes through the functions
 * here. nlohmann::json is only the in-memory tree they decode into;
 * `mlpcore::model_from_json` / `model_to_json` map that tree to an MlpModel.
 *
 * `bff.quantized_neural_net` (QuantizedNeuralNet) is the same idea for a
 * quantised network: `"quantization"` (int8 | fp4 | mxfp4 | nvfp4),
 * `"quantize_activations"`, the scalers as in `bff.neural_net`, and per
 * layer `n_in`, `n_out`, `activation`, `bias` (float64 array) and the
 * weights as msgpack bin fields -- int8: `"weight"` (n_out * n_in int8
 * bytes, two's complement) and `"weight_scale"`; FP4: `"codes"` (n_out rows
 * of padded_cols / 2 bytes, element 2i in the low nibble), `"scales"` (fp4:
 * one little-endian float32 a row; mxfp4: one E8M0 byte a block; nvfp4: one
 * E4M3 byte a block) and, for nvfp4, `"tensor_scale"` (a float32 value);
 * a layer that FP4 training kept in high precision has `"precision":
 * "float64"` and a float64 `"weight"` array instead. Ternary
 * (`"ternary"`, `"ternary_row"`, `"ternary_tq1"`, `"ternary_tq1_row"`;
 * `"quantize_activations"` always true, W1.58A8): `"codes"` (the packed
 * trits, MlpTernary.h: 2 bits a weight, or five trits a byte for `_tq1`)
 * and `"weight_scale"` (the absmean, a float64 value) or, for `_row`,
 * `"weight_scales"` (n_out little-endian float64 in a bin); kept layers as
 * for FP4.
 * Every field round-trips bit-exactly.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_NETWORK_DOCUMENT_H
#define IMPBFF_INTERNAL_NETWORK_DOCUMENT_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/NeuralNet.h>
#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/MlpCore.h>
#include <IMP/bff/internal/MlpFp4.h>
#include <IMP/bff/internal/MlpQuant.h>
#include <IMP/bff/internal/MlpTernary.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE
namespace internal {

//! Decode msgpack bytes to a map whose `"format"` is `format`.
/*! `who` prefixes the messages. Strict: trailing bytes are an error.
    \throws IMP::ValueException on empty, malformed or non-map input, or a
            missing or different `"format"`. */
inline nlohmann::json document_from_msgpack(const MsgpackBytes& bytes,
                                            const std::string& format,
                                            const std::string& who) {
  if (bytes.empty())
    IMP_THROW(who << ": the document is empty (expected msgpack bytes of a '"
                  << format << "' map)",
              IMP::ValueException);
  const nlohmann::json doc = nlohmann::json::from_msgpack(bytes, true, false);
  if (doc.is_discarded())
    IMP_THROW(who << ": the document is not valid msgpack (expected a '"
                  << format << "' map)",
              IMP::ValueException);
  if (!doc.is_object())
    IMP_THROW(who << ": the msgpack document is not a map (expected a '"
                  << format << "' map)",
              IMP::ValueException);
  const auto found = doc.find("format");
  if (found == doc.end() || !found->is_string())
    IMP_THROW(who << ": the document has no \"format\" (expected '" << format
                  << "')",
              IMP::ValueException);
  const std::string got = found->get<std::string>();
  if (got != format)
    IMP_THROW(who << ": unexpected format '" << got << "', expected '" << format
                  << "'",
              IMP::ValueException);
  return doc;
}

//! Encode a document tree as msgpack bytes.
inline MsgpackBytes document_to_msgpack(const nlohmann::json& doc) {
  const std::vector<std::uint8_t> bytes = nlohmann::json::to_msgpack(doc);
  return MsgpackBytes(bytes.begin(), bytes.end());
}

//! A `bff.neural_net` tree (already decoded) as a validated model.
inline MlpModel model_from_document(const nlohmann::json& doc,
                                    const std::string& who = "NeuralNet") {
  if (!doc.is_object() || !doc.contains("format") || !doc.at("format").is_string() ||
      doc.at("format").get<std::string>() != "bff.neural_net")
    IMP_THROW(who << ": the network is not a 'bff.neural_net' map", IMP::ValueException);
  try {
    MlpModel model = mlpcore::model_from_json(doc);
    model.validate();
    return model;
  } catch (const std::exception& e) {
    IMP_THROW(who << ": malformed 'bff.neural_net' document: " << e.what(),
              IMP::ValueException);
  }
}

//! Parse a `bff.neural_net` msgpack document; validates before returning.
inline MlpModel model_from_msgpack(const MsgpackBytes& bytes,
                                   const std::string& who = "NeuralNet") {
  return model_from_document(document_from_msgpack(bytes, "bff.neural_net", who), who);
}

//! Serialise a model as a `bff.neural_net` msgpack document.
inline MsgpackBytes model_to_msgpack(const MlpModel& model) {
  return document_to_msgpack(mlpcore::model_to_json<nlohmann::json>(model));
}

/*! A minimal msgpack tree with `bin` support, for `bff.quantized_neural_net`.
    The vendored nlohmann::json (3.6.1) predates its binary type and can
    neither write nor read msgpack `bin`, which carries the packed codes; so
    this document is written and read here, strictly (every msgpack type bff
    writes; ext is refused; trailing bytes are an error). */
namespace mpk {

struct Value {
  enum Kind { Nil, Bool, Int, Float, Str, Bin, Array, Map } kind = Nil;
  bool b = false;
  std::int64_t i = 0;
  double f = 0.0;
  std::string s;  // Str and Bin
  std::vector<Value> a;
  std::vector<std::pair<std::string, Value>> m;

  const Value* find(const std::string& key) const {
    for (const auto& kv : m)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  const Value& at(const std::string& key) const {
    const Value* v = kind == Map ? find(key) : nullptr;
    if (v == nullptr) throw std::runtime_error("missing field '" + key + "'");
    return *v;
  }
  double number() const {
    if (kind == Float) return f;
    if (kind == Int) return static_cast<double>(i);
    throw std::runtime_error("expected a number");
  }
  std::int64_t integer() const {
    if (kind != Int) throw std::runtime_error("expected an integer");
    return i;
  }
  const std::string& str() const {
    if (kind != Str) throw std::runtime_error("expected a string");
    return s;
  }
  const std::string& bin() const {
    if (kind != Bin) throw std::runtime_error("expected msgpack bin");
    return s;
  }
  bool boolean() const {
    if (kind != Bool) throw std::runtime_error("expected a boolean");
    return b;
  }
  std::vector<double> doubles() const {
    if (kind != Array) throw std::runtime_error("expected an array");
    std::vector<double> out;
    for (const Value& v : a) out.push_back(v.number());
    return out;
  }
};

inline void put_be(std::string& o, std::uint64_t v, int n) {
  for (int k = n - 1; k >= 0; --k) o.push_back(static_cast<char>((v >> (8 * k)) & 0xFF));
}
inline void put_nil(std::string& o) { o.push_back(static_cast<char>(0xC0)); }
inline void put_bool(std::string& o, bool v) { o.push_back(static_cast<char>(v ? 0xC3 : 0xC2)); }
inline void put_int(std::string& o, std::int64_t v) {
  if (v >= 0 && v < 128) { o.push_back(static_cast<char>(v)); return; }
  if (v < 0 && v >= -32) { o.push_back(static_cast<char>(v)); return; }
  o.push_back(static_cast<char>(0xD3));  // int64
  put_be(o, static_cast<std::uint64_t>(v), 8);
}
inline void put_double(std::string& o, double v) {
  std::uint64_t u;
  std::memcpy(&u, &v, 8);
  o.push_back(static_cast<char>(0xCB));
  put_be(o, u, 8);
}
inline void put_len(std::string& o, std::size_t n, int fix_tag, int fix_max, int t8, int t16, int t32) {
  if (fix_tag >= 0 && n <= static_cast<std::size_t>(fix_max)) {
    o.push_back(static_cast<char>(fix_tag | static_cast<int>(n)));
  } else if (t8 >= 0 && n <= 0xFF) {
    o.push_back(static_cast<char>(t8));
    put_be(o, n, 1);
  } else if (n <= 0xFFFF) {
    o.push_back(static_cast<char>(t16));
    put_be(o, n, 2);
  } else {
    o.push_back(static_cast<char>(t32));
    put_be(o, n, 4);
  }
}
inline void put_str(std::string& o, const std::string& v) {
  put_len(o, v.size(), 0xA0, 31, 0xD9, 0xDA, 0xDB);
  o += v;
}
inline void put_bin(std::string& o, const std::uint8_t* p, std::size_t n) {
  put_len(o, n, -1, 0, 0xC4, 0xC5, 0xC6);
  o.append(reinterpret_cast<const char*>(p), n);
}
inline void put_array(std::string& o, std::size_t n) { put_len(o, n, 0x90, 15, -1, 0xDC, 0xDD); }
inline void put_map(std::string& o, std::size_t n) { put_len(o, n, 0x80, 15, -1, 0xDE, 0xDF); }
inline void put_doubles(std::string& o, const std::vector<double>& v) {
  put_array(o, v.size());
  for (double x : v) put_double(o, x);
}

struct Reader {
  const std::string& b;
  std::size_t p = 0;
  explicit Reader(const std::string& bytes) : b(bytes) {}
  std::uint64_t be(int n) {
    if (p + static_cast<std::size_t>(n) > b.size()) throw std::runtime_error("truncated msgpack");
    std::uint64_t v = 0;
    for (int k = 0; k < n; ++k) v = (v << 8) | static_cast<std::uint8_t>(b[p++]);
    return v;
  }
  std::string raw(std::size_t n) {
    if (n > b.size() - p) throw std::runtime_error("truncated msgpack");
    std::string out = b.substr(p, n);
    p += n;
    return out;
  }
  Value value(int depth = 0) {
    if (depth > 64) throw std::runtime_error("msgpack nested too deeply");
    const std::uint8_t t = static_cast<std::uint8_t>(be(1));
    Value v;
    auto array = [&](std::size_t n) {
      v.kind = Value::Array;
      for (std::size_t k = 0; k < n; ++k) v.a.push_back(value(depth + 1));
    };
    auto map = [&](std::size_t n) {
      v.kind = Value::Map;
      for (std::size_t k = 0; k < n; ++k) {
        Value key = value(depth + 1);
        if (key.kind != Value::Str) throw std::runtime_error("msgpack map key is not a string");
        v.m.emplace_back(key.s, value(depth + 1));
      }
    };
    auto text = [&](Value::Kind kind, std::size_t n) {
      v.kind = kind;
      v.s = raw(n);
    };
    if (t <= 0x7F) { v.kind = Value::Int; v.i = t; }
    else if (t >= 0xE0) { v.kind = Value::Int; v.i = static_cast<std::int8_t>(t); }
    else if ((t & 0xF0) == 0x80) map(t & 0x0F);
    else if ((t & 0xF0) == 0x90) array(t & 0x0F);
    else if ((t & 0xE0) == 0xA0) text(Value::Str, t & 0x1F);
    else switch (t) {
      case 0xC0: v.kind = Value::Nil; break;
      case 0xC2: v.kind = Value::Bool; v.b = false; break;
      case 0xC3: v.kind = Value::Bool; v.b = true; break;
      case 0xC4: text(Value::Bin, be(1)); break;
      case 0xC5: text(Value::Bin, be(2)); break;
      case 0xC6: text(Value::Bin, be(4)); break;
      case 0xCA: {
        const std::uint32_t u = static_cast<std::uint32_t>(be(4));
        float x;
        std::memcpy(&x, &u, 4);
        v.kind = Value::Float;
        v.f = x;
        break;
      }
      case 0xCB: {
        const std::uint64_t u = be(8);
        std::memcpy(&v.f, &u, 8);
        v.kind = Value::Float;
        break;
      }
      case 0xCC: v.kind = Value::Int; v.i = static_cast<std::int64_t>(be(1)); break;
      case 0xCD: v.kind = Value::Int; v.i = static_cast<std::int64_t>(be(2)); break;
      case 0xCE: v.kind = Value::Int; v.i = static_cast<std::int64_t>(be(4)); break;
      case 0xCF: v.kind = Value::Int; v.i = static_cast<std::int64_t>(be(8)); break;
      case 0xD0: v.kind = Value::Int; v.i = static_cast<std::int8_t>(be(1)); break;
      case 0xD1: v.kind = Value::Int; v.i = static_cast<std::int16_t>(be(2)); break;
      case 0xD2: v.kind = Value::Int; v.i = static_cast<std::int32_t>(be(4)); break;
      case 0xD3: v.kind = Value::Int; v.i = static_cast<std::int64_t>(be(8)); break;
      case 0xD9: text(Value::Str, be(1)); break;
      case 0xDA: text(Value::Str, be(2)); break;
      case 0xDB: text(Value::Str, be(4)); break;
      case 0xDC: array(be(2)); break;
      case 0xDD: array(be(4)); break;
      case 0xDE: map(be(2)); break;
      case 0xDF: map(be(4)); break;
      default: throw std::runtime_error("unsupported msgpack type");
    }
    return v;
  }
};

inline void put_scaler(std::string& o, const StandardScaler& s) {
  if (!s.active()) {
    put_map(o, 0);
    return;
  }
  put_map(o, 2);
  put_str(o, "mean");
  put_doubles(o, s.mean);
  put_str(o, "scale");
  put_doubles(o, s.scale);
}
inline StandardScaler get_scaler(const Value* v) {
  StandardScaler s;
  if (v == nullptr || v->kind == Value::Nil || (v->kind == Value::Map && v->m.empty())) return s;
  s.mean = v->at("mean").doubles();
  s.scale = v->at("scale").doubles();
  if (s.mean.size() != s.scale.size()) throw std::runtime_error("scaler mean/scale length mismatch");
  for (auto& x : s.scale)
    if (x == 0.0) x = 1.0;
  return s;
}

}  // namespace mpk

//! Serialise a quantised network as a `bff.quantized_neural_net` document.
/*! `format` is "int8" (uses `q8`) or an FP4 format (uses `f4`). */
inline MsgpackBytes quantized_to_msgpack(const std::string& format, bool quantize_activations,
                                         const mlpquant::QuantModel& q8,
                                         const mlpfp4::Fp4Model& f4) {
  std::string o;
  const bool is_int8 = format == "int8";
  mpk::put_map(o, 7);
  mpk::put_str(o, "format");
  mpk::put_str(o, "bff.quantized_neural_net");
  mpk::put_str(o, "version");
  mpk::put_int(o, 1);
  mpk::put_str(o, "quantization");
  mpk::put_str(o, format);
  mpk::put_str(o, "quantize_activations");
  mpk::put_bool(o, quantize_activations);
  mpk::put_str(o, "x_scaler");
  mpk::put_scaler(o, is_int8 ? q8.x_scaler : f4.x_scaler);
  mpk::put_str(o, "y_scaler");
  mpk::put_scaler(o, is_int8 ? q8.y_scaler : f4.y_scaler);
  mpk::put_str(o, "layers");
  const std::size_t n = is_int8 ? q8.layers.size() : f4.layers.size();
  mpk::put_array(o, n);
  for (std::size_t i = 0; i < n; ++i) {
    if (is_int8) {
      const mlpquant::QuantLayer& l = q8.layers[i];
      mpk::put_map(o, 6);
      mpk::put_str(o, "n_in"); mpk::put_int(o, l.n_in);
      mpk::put_str(o, "n_out"); mpk::put_int(o, l.n_out);
      mpk::put_str(o, "activation"); mpk::put_str(o, activation_to_string(l.activation));
      mpk::put_str(o, "bias"); mpk::put_doubles(o, l.bias);
      mpk::put_str(o, "weight");
      mpk::put_bin(o, reinterpret_cast<const std::uint8_t*>(l.weight.data()), l.weight.size());
      mpk::put_str(o, "weight_scale"); mpk::put_double(o, l.weight_scale);
    } else {
      const mlpfp4::Fp4Layer& l = f4.layers[i];
      if (l.full_precision) {  // a layer the FP4 training kept in float64
        mpk::put_map(o, 6);
        mpk::put_str(o, "n_in"); mpk::put_int(o, l.n_in);
        mpk::put_str(o, "n_out"); mpk::put_int(o, l.n_out);
        mpk::put_str(o, "activation"); mpk::put_str(o, activation_to_string(l.activation));
        mpk::put_str(o, "bias"); mpk::put_doubles(o, l.bias);
        mpk::put_str(o, "precision"); mpk::put_str(o, "float64");
        mpk::put_str(o, "weight"); mpk::put_doubles(o, l.weight_f64);
        continue;
      }
      const bool nv = l.weight.format == mlpfp4::Format::NVFP4;
      mpk::put_map(o, nv ? 8 : 7);
      mpk::put_str(o, "n_in"); mpk::put_int(o, l.n_in);
      mpk::put_str(o, "n_out"); mpk::put_int(o, l.n_out);
      mpk::put_str(o, "activation"); mpk::put_str(o, activation_to_string(l.activation));
      mpk::put_str(o, "bias"); mpk::put_doubles(o, l.bias);
      mpk::put_str(o, "block"); mpk::put_int(o, l.weight.block);
      mpk::put_str(o, "codes"); mpk::put_bin(o, l.weight.codes.data(), l.weight.codes.size());
      mpk::put_str(o, "scales"); mpk::put_bin(o, l.weight.scales.data(), l.weight.scales.size());
      if (nv) {
        mpk::put_str(o, "tensor_scale");
        mpk::put_double(o, static_cast<double>(l.weight.tensor_scale));
      }
    }
  }
  return MsgpackBytes(o);
}

//! Serialise a ternary network as a `bff.quantized_neural_net` document.
inline MsgpackBytes ternary_to_msgpack(const mlpternary::TModel& t) {
  std::string o;
  mpk::put_map(o, 7);
  mpk::put_str(o, "format");
  mpk::put_str(o, "bff.quantized_neural_net");
  mpk::put_str(o, "version");
  mpk::put_int(o, 1);
  mpk::put_str(o, "quantization");
  mpk::put_str(o, mlpternary::format_name(t.scale, t.storage));
  mpk::put_str(o, "quantize_activations");
  mpk::put_bool(o, true);
  mpk::put_str(o, "x_scaler");
  mpk::put_scaler(o, t.x_scaler);
  mpk::put_str(o, "y_scaler");
  mpk::put_scaler(o, t.y_scaler);
  mpk::put_str(o, "layers");
  mpk::put_array(o, t.layers.size());
  for (const mlpternary::TLayer& l : t.layers) {
    mpk::put_map(o, 6);
    mpk::put_str(o, "n_in"); mpk::put_int(o, l.n_in);
    mpk::put_str(o, "n_out"); mpk::put_int(o, l.n_out);
    mpk::put_str(o, "activation"); mpk::put_str(o, activation_to_string(l.activation));
    mpk::put_str(o, "bias"); mpk::put_doubles(o, l.bias);
    if (l.full_precision) {
      mpk::put_str(o, "precision"); mpk::put_str(o, "float64");
      mpk::put_str(o, "weight"); mpk::put_doubles(o, l.weight_f64);
      continue;
    }
    const std::vector<std::uint8_t> codes = mlpternary::pack(l.weight, t.storage);
    mpk::put_str(o, "codes"); mpk::put_bin(o, codes.data(), codes.size());
    if (t.scale == mlpternary::Scale::Tensor) {
      mpk::put_str(o, "weight_scale"); mpk::put_double(o, l.weight.gamma.at(0));
    } else {
      std::vector<std::uint8_t> le(8 * l.weight.gamma.size());
      for (std::size_t i = 0; i < l.weight.gamma.size(); ++i) {
        std::uint64_t u;
        std::memcpy(&u, &l.weight.gamma[i], 8);
        for (int b = 0; b < 8; ++b) le[8 * i + static_cast<std::size_t>(b)] = static_cast<std::uint8_t>(u >> (8 * b));
      }
      mpk::put_str(o, "weight_scales"); mpk::put_bin(o, le.data(), le.size());
    }
  }
  return MsgpackBytes(o);
}

//! Parse a `bff.quantized_neural_net` document into `q8`, `f4` or `tm`;
//! returns the format and sets `quantize_activations`.
/*! \throws IMP::ValueException on a malformed document */
inline std::string quantized_from_msgpack(const MsgpackBytes& bytes, bool& quantize_activations,
                                          mlpquant::QuantModel& q8, mlpfp4::Fp4Model& f4,
                                          mlpternary::TModel& tm,
                                          const std::string& who = "QuantizedNeuralNet") {
  if (bytes.empty())
    IMP_THROW(who << ": the document is empty (expected msgpack bytes of a "
                     "'bff.quantized_neural_net' map)",
              IMP::ValueException);
  std::string format;
  try {
    mpk::Reader rd(bytes);
    const mpk::Value doc = rd.value();
    if (rd.p != bytes.size()) throw std::runtime_error("trailing bytes after the document");
    if (doc.kind != mpk::Value::Map) throw std::runtime_error("not a map");
    const mpk::Value* fmt = doc.find("format");
    if (fmt == nullptr || fmt->kind != mpk::Value::Str)
      throw std::runtime_error("no \"format\" (expected 'bff.quantized_neural_net')");
    if (fmt->s != "bff.quantized_neural_net")
      throw std::runtime_error("unexpected format '" + fmt->s + "', expected 'bff.quantized_neural_net'");
    format = doc.at("quantization").str();
    quantize_activations = doc.at("quantize_activations").boolean();
    const bool is_int8 = format == "int8";
    mlpternary::Scale tsc = mlpternary::Scale::Tensor;
    mlpternary::Storage tst = mlpternary::Storage::TQ2;
    const bool is_ter = mlpternary::parse_format(format, tsc, tst);
    if (is_ter && !quantize_activations) throw std::runtime_error("ternary: quantize_activations must be true");
    mlpfp4::Format ff = mlpfp4::Format::FP4;
    if (!is_int8 && !is_ter) ff = mlpfp4::format_from_string(format);
    const StandardScaler xs = mpk::get_scaler(doc.find("x_scaler"));
    const StandardScaler ys = mpk::get_scaler(doc.find("y_scaler"));
    q8 = mlpquant::QuantModel();
    f4 = mlpfp4::Fp4Model();
    tm = mlpternary::TModel();
    tm.scale = tsc;
    tm.storage = tst;
    f4.format = ff;
    f4.quantize_activations = quantize_activations;
    const mpk::Value& layers = doc.at("layers");
    if (layers.kind != mpk::Value::Array || layers.a.empty()) throw std::runtime_error("no layers");
    int prev_out = -1;
    for (const mpk::Value& lj : layers.a) {
      const std::int64_t n_in = lj.at("n_in").integer(), n_out = lj.at("n_out").integer();
      if (n_in <= 0 || n_out <= 0 || n_in > (1 << 24) || n_out > (1 << 24))
        throw std::runtime_error("layer sizes out of range");
      if (prev_out >= 0 && n_in != prev_out) throw std::runtime_error("layer sizes do not chain");
      prev_out = static_cast<int>(n_out);
      const Activation act = activation_from_string(lj.at("activation").str());
      std::vector<double> bias = lj.at("bias").doubles();
      if (bias.size() != static_cast<std::size_t>(n_out)) throw std::runtime_error("bias length");
      const std::size_t nw = static_cast<std::size_t>(n_in) * static_cast<std::size_t>(n_out);
      if (is_ter) {
        mlpternary::TLayer l;
        l.n_in = static_cast<int>(n_in);
        l.n_out = static_cast<int>(n_out);
        l.activation = act;
        l.bias = std::move(bias);
        const mpk::Value* prec = lj.find("precision");
        if (prec != nullptr) {
          if (prec->str() != "float64") throw std::runtime_error("unknown layer precision");
          l.full_precision = true;
          l.weight_f64 = lj.at("weight").doubles();
          if (l.weight_f64.size() != nw) throw std::runtime_error("float64 weight length");
          tm.layers.push_back(std::move(l));
          continue;
        }
        l.weight.rows = l.n_out;
        l.weight.cols = l.n_in;
        l.weight.scale = tsc;
        const std::string& c = lj.at("codes").bin();
        mlpternary::unpack(reinterpret_cast<const std::uint8_t*>(c.data()), c.size(), tst, l.weight);
        if (tsc == mlpternary::Scale::Tensor) {
          l.weight.gamma.assign(1, lj.at("weight_scale").number());
        } else {
          const std::string& g = lj.at("weight_scales").bin();
          if (g.size() != 8 * static_cast<std::size_t>(n_out)) throw std::runtime_error("weight_scales length");
          for (std::int64_t o = 0; o < n_out; ++o) {
            std::uint64_t u = 0;
            for (int b = 0; b < 8; ++b)
              u |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(g[static_cast<std::size_t>(8 * o + b)])) << (8 * b);
            double v;
            std::memcpy(&v, &u, 8);
            l.weight.gamma.push_back(v);
          }
        }
        for (double v : l.weight.gamma)
          if (!(v > 0.0) || !std::isfinite(v)) throw std::runtime_error("ternary weight scale not positive");
        tm.layers.push_back(std::move(l));
        continue;
      }
      if (is_int8) {
        mlpquant::QuantLayer l;
        l.n_in = static_cast<int>(n_in);
        l.n_out = static_cast<int>(n_out);
        l.activation = act;
        l.bias = std::move(bias);
        const std::string& w = lj.at("weight").bin();
        if (w.size() != nw) throw std::runtime_error("int8 weight length");
        l.weight.resize(nw);
        std::memcpy(l.weight.data(), w.data(), nw);
        l.weight_scale = lj.at("weight_scale").number();
        q8.layers.push_back(std::move(l));
      } else {
        mlpfp4::Fp4Layer l;
        l.n_in = static_cast<int>(n_in);
        l.n_out = static_cast<int>(n_out);
        l.activation = act;
        l.bias = std::move(bias);
        const mpk::Value* prec = lj.find("precision");
        if (prec != nullptr) {
          if (prec->str() != "float64") throw std::runtime_error("unknown layer precision");
          l.full_precision = true;
          l.weight_f64 = lj.at("weight").doubles();
          if (l.weight_f64.size() != nw) throw std::runtime_error("float64 weight length");
          f4.layers.push_back(std::move(l));
          continue;
        }
        l.weight = mlpfp4::make_tensor(l.n_out, l.n_in, ff);
        if (lj.at("block").integer() != l.weight.block) throw std::runtime_error("block size");
        const std::string& c = lj.at("codes").bin();
        const std::string& sc = lj.at("scales").bin();
        l.weight.codes.assign(c.begin(), c.end());
        l.weight.scales.assign(sc.begin(), sc.end());
        if (ff == mlpfp4::Format::NVFP4)
          l.weight.tensor_scale = static_cast<float>(lj.at("tensor_scale").number());
        if (!l.weight.consistent()) throw std::runtime_error("codes / scales length");
        f4.layers.push_back(std::move(l));
      }
    }
    const int n_in0 = is_ter ? tm.n_inputs() : is_int8 ? q8.n_inputs() : f4.n_inputs();
    const int n_out0 = is_ter ? tm.n_outputs() : is_int8 ? q8.n_outputs() : f4.n_outputs();
    if ((xs.active() && xs.size() != n_in0) || (ys.active() && ys.size() != n_out0))
      throw std::runtime_error("scaler length");
    q8.x_scaler = f4.x_scaler = tm.x_scaler = xs;
    q8.y_scaler = f4.y_scaler = tm.y_scaler = ys;
  } catch (const std::exception& e) {
    IMP_THROW(who << ": malformed 'bff.quantized_neural_net' document: " << e.what(),
              IMP::ValueException);
  }
  return format;
}

}  // namespace internal
IMPBFF_END_NAMESPACE

#endif  // IMPBFF_INTERNAL_NETWORK_DOCUMENT_H
