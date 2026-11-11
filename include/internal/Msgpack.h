/**
 *  \file IMP/bff/internal/Msgpack.h
 *  \brief A minimal, strict msgpack tree with `bin` support.
 *
 *  Moved out of internal/NetworkDocument.h on 2026-11-11 (PRD-156 step 3, claude-code/Opus 5.5
 *  for tpeulen) with no change, so the factor graph's documents and the network documents share
 *  one msgpack path. The vendored nlohmann::json (3.6.1) predates its binary type and can
 *  neither write nor read msgpack `bin`, which carries packed arrays; so documents with `bin`
 *  are written and read here, strictly (every msgpack type bff writes; ext is refused; a caller
 *  checks `Reader::p` against the size, so trailing bytes are an error).
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_INTERNAL_MSGPACK_H
#define IMPBFF_INTERNAL_MSGPACK_H

#include <IMP/bff/bff_config.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

IMPBFF_BEGIN_NAMESPACE
namespace internal {
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

//! Decode a whole document: one value and nothing after it.
inline Value parse(const std::string& bytes) {
  Reader rd(bytes);
  Value v = rd.value();
  if (rd.p != bytes.size()) throw std::runtime_error("trailing bytes after the msgpack document");
  return v;
}

//! Encode a tree (the inverse of Reader::value; maps keep their order).
inline void put_value(std::string& o, const Value& v) {
  switch (v.kind) {
    case Value::Nil: put_nil(o); break;
    case Value::Bool: put_bool(o, v.b); break;
    case Value::Int: put_int(o, v.i); break;
    case Value::Float: put_double(o, v.f); break;
    case Value::Str: put_str(o, v.s); break;
    case Value::Bin: put_bin(o, reinterpret_cast<const std::uint8_t*>(v.s.data()), v.s.size()); break;
    case Value::Array:
      put_array(o, v.a.size());
      for (const Value& x : v.a) put_value(o, x);
      break;
    case Value::Map:
      put_map(o, v.m.size());
      for (const auto& kv : v.m) { put_str(o, kv.first); put_value(o, kv.second); }
      break;
  }
}

//! float64 values as a little-endian `bin` (the byte order every platform bff builds for uses).
inline Value bin_of_doubles(const std::vector<double>& d) {
  Value v;
  v.kind = Value::Bin;
  v.s.assign(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(double));
  return v;
}

//! The inverse of bin_of_doubles; the size must be a whole number of doubles.
inline std::vector<double> doubles_of_bin(const Value& v) {
  const std::string& b = v.bin();
  if (b.size() % sizeof(double) != 0) throw std::runtime_error("a float64 bin whose size is not a multiple of 8");
  std::vector<double> d(b.size() / sizeof(double));
  if (!d.empty()) std::memcpy(d.data(), b.data(), b.size());
  return d;
}

}  // namespace mpk
}  // namespace internal
IMPBFF_END_NAMESPACE

#endif /* IMPBFF_INTERNAL_MSGPACK_H */
