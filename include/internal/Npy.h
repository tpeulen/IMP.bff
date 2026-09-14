/**
 *  \file IMP/bff/internal/Npy.h
 *  \brief Read a NumPy `.npy` file into doubles.
 *
 *  The format is a magic string, a version, a header length and a Python
 *  dict literal -- `{'descr': '<f8', 'fortran_order': False, 'shape': (4, 800), }`
 *  -- then the raw array (numpy.org/neps/nep-0001-npy-format). Versions 1.0,
 *  2.0 and 3.0 are read. Only what this module's converters meet is decoded:
 *  little-endian (or byte-order-free) float32/64, int8/16/32/64,
 *  uint8/16/32/64 and bool, C order. Anything else is refused by name rather
 *  than misread.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_NPY_H
#define IMPBFF_INTERNAL_NPY_H

#include <IMP/bff/bff_config.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

IMPBFF_BEGIN_INTERNAL_NAMESPACE

//! An array read from a `.npy` file: its shape and its values, C order.
struct NpyArray {
  std::vector<long> shape;
  std::vector<double> values;
  std::string descr;  //!< as the file wrote it, e.g. `<f4`

  std::size_t size() const {
    std::size_t n = 1;
    for (std::size_t i = 0; i < shape.size(); ++i) n *= static_cast<std::size_t>(shape[i]);
    return n;
  }
};

namespace npy_detail {

inline std::string quoted_value(const std::string& header, const std::string& key) {
  const std::size_t k = header.find("'" + key + "'");
  if (k == std::string::npos) return std::string();
  const std::size_t colon = header.find(':', k);
  if (colon == std::string::npos) return std::string();
  const std::size_t q = header.find_first_of("'\"", colon + 1);
  if (q == std::string::npos) return std::string();
  const std::size_t e = header.find(header[q], q + 1);
  return e == std::string::npos ? std::string() : header.substr(q + 1, e - q - 1);
}

inline std::string bare_value(const std::string& header, const std::string& key) {
  const std::size_t k = header.find("'" + key + "'");
  if (k == std::string::npos) return std::string();
  std::size_t b = header.find(':', k);
  if (b == std::string::npos) return std::string();
  ++b;
  while (b < header.size() && header[b] == ' ') ++b;
  if (b < header.size() && header[b] == '(') {
    const std::size_t e = header.find(')', b);
    return e == std::string::npos ? std::string() : header.substr(b, e - b + 1);
  }
  std::size_t e = b;
  while (e < header.size() && header[e] != ',' && header[e] != '}') ++e;
  return header.substr(b, e - b);
}

template <class T>
inline void append_as_double(const char* data, std::size_t n, std::vector<double>& out) {
  out.reserve(out.size() + n);
  for (std::size_t i = 0; i < n; ++i) {
    T v;
    std::memcpy(&v, data + i * sizeof(T), sizeof(T));
    out.push_back(static_cast<double>(v));
  }
}

}  // namespace npy_detail

//! Read \p path; std::runtime_error names the problem when it cannot.
inline NpyArray read_npy(const std::string& path) {
  std::ifstream in(path.c_str(), std::ios::binary);
  if (!in) throw std::runtime_error(path + ": cannot read");
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string bytes = ss.str();
  if (bytes.size() < 10 || bytes.compare(0, 6, "\x93NUMPY") != 0) {
    throw std::runtime_error(path + ": not a .npy file");
  }
  const unsigned char major = static_cast<unsigned char>(bytes[6]);
  std::size_t header_len = 0, offset = 0;
  if (major == 1) {
    header_len = static_cast<std::size_t>(static_cast<unsigned char>(bytes[8])) |
                 (static_cast<std::size_t>(static_cast<unsigned char>(bytes[9])) << 8);
    offset = 10;
  } else if (major == 2 || major == 3) {
    if (bytes.size() < 12) throw std::runtime_error(path + ": truncated .npy header");
    for (int i = 0; i < 4; ++i) {
      header_len |= static_cast<std::size_t>(static_cast<unsigned char>(bytes[8 + i])) << (8 * i);
    }
    offset = 12;
  } else {
    throw std::runtime_error(path + ": unsupported .npy version");
  }
  if (bytes.size() < offset + header_len) {
    throw std::runtime_error(path + ": truncated .npy header");
  }
  const std::string header = bytes.substr(offset, header_len);
  const std::size_t data_start = offset + header_len;

  NpyArray out;
  out.descr = npy_detail::quoted_value(header, "descr");
  if (npy_detail::bare_value(header, "fortran_order").find("True") != std::string::npos) {
    throw std::runtime_error(path + ": fortran_order arrays are not read");
  }
  const std::string shape = npy_detail::bare_value(header, "shape");
  for (std::size_t i = 0; i < shape.size();) {
    if (shape[i] >= '0' && shape[i] <= '9') {
      char* end = nullptr;
      out.shape.push_back(std::strtol(shape.c_str() + i, &end, 10));
      i = static_cast<std::size_t>(end - shape.c_str());
    } else {
      ++i;
    }
  }
  const std::string& d = out.descr;
  if (d.size() < 3) throw std::runtime_error(path + ": unreadable dtype '" + d + "'");
  if (d[0] == '>') throw std::runtime_error(path + ": big-endian dtype '" + d + "' is not read");
  const char kind = d[1];
  const int width = std::atoi(d.c_str() + 2);
  const std::size_t n = out.size();
  if (width <= 0 || bytes.size() < data_start + n * static_cast<std::size_t>(width)) {
    throw std::runtime_error(path + ": holds fewer values than its shape says");
  }
  const char* data = bytes.data() + data_start;
  using npy_detail::append_as_double;
  if (kind == 'f' && width == 8) append_as_double<double>(data, n, out.values);
  else if (kind == 'f' && width == 4) append_as_double<float>(data, n, out.values);
  else if (kind == 'i' && width == 8) append_as_double<std::int64_t>(data, n, out.values);
  else if (kind == 'i' && width == 4) append_as_double<std::int32_t>(data, n, out.values);
  else if (kind == 'i' && width == 2) append_as_double<std::int16_t>(data, n, out.values);
  else if (kind == 'i' && width == 1) append_as_double<std::int8_t>(data, n, out.values);
  else if (kind == 'u' && width == 8) append_as_double<std::uint64_t>(data, n, out.values);
  else if (kind == 'u' && width == 4) append_as_double<std::uint32_t>(data, n, out.values);
  else if (kind == 'u' && width == 2) append_as_double<std::uint16_t>(data, n, out.values);
  else if ((kind == 'u' || kind == 'b') && width == 1) append_as_double<std::uint8_t>(data, n, out.values);
  else throw std::runtime_error(path + ": dtype '" + d + "' is not read");
  return out;
}

IMPBFF_END_INTERNAL_NAMESPACE

#endif  // IMPBFF_INTERNAL_NPY_H
