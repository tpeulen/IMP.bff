/**
 *  \file IMP/bff/internal/NumpyCompat.h
 *  \brief numpy's and CPython's arithmetic, where a port has to agree with a
 *         Python program digit for digit.
 *
 *  A program ported from Python is checked by diffing what it writes against
 *  what the Python wrote. Most arithmetic agrees by construction; these are
 *  the places where it does not, each measured against the arm64 conda env
 *  (numpy 2.4, OpenBLAS 0.3.34 with reference LAPACK, CPython 3.12):
 *
 *  - `np.sum` of a 1-D array (contiguous or strided) is numpy's *pairwise*
 *    summation (`pairwise_sum`); `arr.mean(axis=0)` of an `(n, 3)` array is
 *    sequential per column (`sequential_sum`).
 *  - `np.dot`/`np.linalg.norm` of 3-vectors go through OpenBLAS `ddot`, which
 *    accumulates with fused multiply-add (`dot_fma`).
 *  - `np.linalg.eigh` of a 3x3 matrix is LAPACK `dsyevd('V', 'L')`; its
 *    eigenvector *signs* come from the Householder and Givens sequence, which
 *    `eigh3` reproduces (reference `dsytd2` + `dsteqr` + `dorm2r`, with the
 *    LAPACK >= 3.10 `dlartg`). Signs agree on 99.99 % of 200 000 random
 *    covariances -- the rest are exactly rank-deficient -- and values to
 *    1e-12; Eigen's solver agrees on values and picks signs its own way.
 *  - Python's `sum()` of floats is Neumaier-compensated since 3.12
 *    (`python_sum`), and `x // y` on floats is not `floor(x / y)`
 *    (`python_floor_div`).
 *  - `repr(float)` and `json.dumps` spell floats as the shortest string that
 *    reads back (`python_repr`), and `json.dumps(obj, indent=k)` has a layout
 *    `PyJson` reproduces, key order included.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_NUMPYCOMPAT_H
#define IMPBFF_INTERNAL_NUMPYCOMPAT_H

#include <IMP/bff/bff_config.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace numpy_compat {

//! numpy's pairwise summation (`DOUBLE_pairwise_sum`) over `n` values at `stride`.
inline double pairwise_sum(const double* a, std::size_t n, std::size_t stride = 1) {
  if (n < 8) {
    double res = 0.0;
    for (std::size_t i = 0; i < n; ++i) res += a[i * stride];
    return res;
  }
  if (n <= 128) {
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j * stride];
    std::size_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (int j = 0; j < 8; ++j) r[j] += a[(i + j) * stride];
    }
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i * stride];
    return res;
  }
  std::size_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2, stride) + pairwise_sum(a + n2 * stride, n - n2, stride);
}

inline double pairwise_sum(const std::vector<double>& a) {
  return a.empty() ? 0.0 : pairwise_sum(&a[0], a.size());
}

//! A left-to-right sum starting from 0.0, as `arr.mean(axis=0)` sums each column.
inline double sequential_sum(const double* a, std::size_t n, std::size_t stride = 1) {
  double res = 0.0;
  for (std::size_t i = 0; i < n; ++i) res += a[i * stride];
  return res;
}

//! OpenBLAS `ddot` on arm64: an FMA accumulation.
inline double dot_fma(const double* x, const double* y, std::size_t n) {
  double r = 0.0;
  for (std::size_t i = 0; i < n; ++i) r = std::fma(x[i], y[i], r);
  return r;
}

//! One entry of `A.T @ A` for an `(n, k)` array, as OpenBLAS computes it on
//! arm64: an FMA accumulation for n <= 512 and, above, two halves (the
//! second from `ceil(n / 2)`) added -- the split the threaded kernel makes.
//! Measured exact to n = 700; beyond that the partition depends on the thread
//! count, so the last bits do too, in numpy itself.
inline double gram_entry(const double* x, const double* y, std::size_t n) {
  if (n <= 512) return dot_fma(x, y, n);
  const std::size_t half = (n + 1) / 2;
  const double first = dot_fma(x, y, half);
  const double second = dot_fma(x + half, y + half, n - half);
  return first + second;
}

//! `np.linalg.norm` of a 3-vector.
inline double norm3(const double* x) { return std::sqrt(dot_fma(x, x, 3)); }

//! CPython 3.12 `sum()` of floats, from a start of 0: Neumaier summation.
inline double python_sum(const std::vector<double>& v) {
  double f = 0.0, c = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    const double x = v[i];
    const double t = f + x;
    if (std::fabs(f) >= std::fabs(x)) {
      c += (f - t) + x;
    } else {
      c += (x - t) + f;
    }
    f = t;
  }
  if (c != 0.0 && std::isfinite(c)) f += c;
  return f;
}

//! CPython's `float.__floordiv__` (`float_floor_div`).
inline double python_floor_div(double vx, double wx) {
  double mod = std::fmod(vx, wx);
  double div = (vx - mod) / wx;
  if (mod != 0.0) {
    if ((wx < 0) != (mod < 0)) {
      mod += wx;
      div -= 1.0;
    }
  }
  double floordiv;
  if (div != 0.0) {
    floordiv = std::floor(div);
    if (div - floordiv > 0.5) floordiv += 1.0;
  } else {
    floordiv = std::copysign(0.0, vx / wx);
  }
  return floordiv;
}

//! `repr(float)`: shortest round-trip digits, positional for exponents -4..15.
inline std::string python_repr(double v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  const std::string sign = std::signbit(v) ? "-" : "";
  const double a = std::fabs(v);
  if (a == 0.0) return sign + "0.0";
  char buf[48];
  for (int p = 1; p <= 17; ++p) {
    std::snprintf(buf, sizeof(buf), "%.*e", p - 1, a);
    if (std::strtod(buf, nullptr) == a) break;
  }
  const std::string sci = buf;
  const std::size_t e = sci.find('e');
  std::string digits = sci.substr(0, e);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
  while (digits.size() > 1 && digits[digits.size() - 1] == '0') digits.erase(digits.size() - 1);
  const int exponent = std::atoi(sci.c_str() + e + 1);
  const int n = static_cast<int>(digits.size());
  if (exponent >= -4 && exponent < 16) {
    const int point = exponent + 1;
    if (point <= 0) return sign + "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
    if (point >= n) return sign + digits + std::string(static_cast<std::size_t>(point - n), '0') + ".0";
    return sign + digits.substr(0, point) + "." + digits.substr(point);
  }
  const std::string mantissa = n == 1 ? digits : digits.substr(0, 1) + "." + digits.substr(1);
  char exp[16];
  std::snprintf(exp, sizeof(exp), "e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
  return sign + mantissa + exp;
}

//! A float as `json.dumps` writes it.
inline std::string json_repr(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  return python_repr(v);
}

//! A JSON value built in insertion order and printed as `json.dumps` prints it.
class PyJson {
 public:
  enum Kind { NUL, SCALAR, STRING, ARRAY, OBJECT };

  PyJson() : kind_(NUL) {}
  static PyJson number(double v) { return scalar(json_repr(v)); }
  static PyJson integer(long long v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld", v);
    return scalar(buf);
  }
  static PyJson boolean(bool v) { return scalar(v ? "true" : "false"); }
  //! Already-spelled JSON text, placed as it is (it must be laid out for its depth).
  static PyJson raw(const std::string& text) { return scalar(text); }
  static PyJson string(const std::string& s) {
    PyJson j;
    j.kind_ = STRING;
    j.text_ = s;
    return j;
  }
  static PyJson array() {
    PyJson j;
    j.kind_ = ARRAY;
    return j;
  }
  static PyJson object() {
    PyJson j;
    j.kind_ = OBJECT;
    return j;
  }

  //! Set a key; an existing key keeps its place, as a dict does.
  PyJson& set(const std::string& key, const PyJson& value) {
    for (std::size_t i = 0; i < items_.size(); ++i) {
      if (items_[i].first == key) {
        items_[i].second = value;
        return *this;
      }
    }
    items_.push_back(std::make_pair(key, value));
    return *this;
  }
  PyJson& push(const PyJson& value) {
    elements_.push_back(value);
    return *this;
  }

  //! `json.dumps(self, indent=indent)`; a negative indent is the compact form.
  std::string dumps(int indent = -1, int depth = 0) const {
    const bool pretty = indent >= 0;
    const std::string pad = pretty ? "\n" + std::string(static_cast<std::size_t>(indent * (depth + 1)), ' ') : "";
    const std::string close = pretty ? "\n" + std::string(static_cast<std::size_t>(indent * depth), ' ') : "";
    const std::string sep = pretty ? "," : ", ";
    switch (kind_) {
      case NUL:
        return "null";
      case SCALAR:
        return text_;
      case STRING:
        return quote(text_);
      case ARRAY: {
        if (elements_.empty()) return "[]";
        std::string out = "[";
        for (std::size_t i = 0; i < elements_.size(); ++i) {
          if (i) out += sep;
          out += pad + elements_[i].dumps(indent, depth + 1);
        }
        return out + close + "]";
      }
      case OBJECT: {
        if (items_.empty()) return "{}";
        std::string out = "{";
        for (std::size_t i = 0; i < items_.size(); ++i) {
          if (i) out += sep;
          out += pad + quote(items_[i].first) + ": " + items_[i].second.dumps(indent, depth + 1);
        }
        return out + close + "}";
      }
    }
    return "null";
  }

  //! A string as `json.dumps` quotes it (`ensure_ascii`).
  static std::string quote(const std::string& text) {
    std::string out = "\"";
    char buf[16];
    for (std::size_t i = 0; i < text.size();) {
      const unsigned char c = static_cast<unsigned char>(text[i]);
      if (c < 0x80) {
        switch (c) {
          case '"': out += "\\\""; break;
          case '\\': out += "\\\\"; break;
          case '\n': out += "\\n"; break;
          case '\r': out += "\\r"; break;
          case '\t': out += "\\t"; break;
          case '\b': out += "\\b"; break;
          case '\f': out += "\\f"; break;
          default:
            if (c < 0x20) {
              std::snprintf(buf, sizeof(buf), "\\u%04x", c);
              out += buf;
            } else {
              out += static_cast<char>(c);
            }
        }
        ++i;
        continue;
      }
      unsigned long cp = 0;
      int extra = 0;
      if ((c & 0xE0) == 0xC0) {
        cp = c & 0x1F;
        extra = 1;
      } else if ((c & 0xF0) == 0xE0) {
        cp = c & 0x0F;
        extra = 2;
      } else {
        cp = c & 0x07;
        extra = 3;
      }
      ++i;
      for (int k = 0; k < extra && i < text.size(); ++k, ++i) {
        cp = (cp << 6) | (static_cast<unsigned char>(text[i]) & 0x3F);
      }
      if (cp >= 0x10000) {
        cp -= 0x10000;
        std::snprintf(buf, sizeof(buf), "\\u%04lx\\u%04lx", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF));
      } else {
        std::snprintf(buf, sizeof(buf), "\\u%04lx", cp);
      }
      out += buf;
    }
    return out + "\"";
  }

 private:
  static PyJson scalar(const std::string& text) {
    PyJson j;
    j.kind_ = SCALAR;
    j.text_ = text;
    return j;
  }
  Kind kind_;
  std::string text_;
  std::vector<PyJson> elements_;
  std::vector<std::pair<std::string, PyJson> > items_;
};

// ---- LAPACK dsyevd('V', 'L') for a 3x3 symmetric matrix -------------------

namespace lapack {

inline double sign(double a, double b) { return std::signbit(b) ? -std::fabs(a) : std::fabs(a); }

inline double dlapy2(double x, double y) {
  if (std::isnan(x)) return x;
  if (std::isnan(y)) return y;
  const double xa = std::fabs(x), ya = std::fabs(y);
  const double w = (std::max)(xa, ya), z = (std::min)(xa, ya);
  if (z == 0.0 || w > std::numeric_limits<double>::max()) return w;
  const double q = z / w;
  return w * std::sqrt(1.0 + q * q);
}

//! LAPACK >= 3.10 `dlartg` (la_xlartg.f90).
inline void dlartg(double f, double g, double& c, double& s, double& r) {
  const double safmin = std::numeric_limits<double>::min();
  const double safmax = 1.0 / safmin;
  const double rtmin = std::sqrt(safmin);
  const double rtmax = std::sqrt(safmax / 2);
  const double f1 = std::fabs(f), g1 = std::fabs(g);
  if (g == 0.0) {
    c = 1.0;
    s = 0.0;
    r = f;
  } else if (f == 0.0) {
    c = 0.0;
    s = sign(1.0, g);
    r = g1;
  } else if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
    const double d = std::sqrt(f * f + g * g);
    c = f1 / d;
    r = sign(d, f);
    s = g / r;
  } else {
    const double u = (std::min)(safmax, (std::max)(safmin, (std::max)(f1, g1)));
    const double fs = f / u, gs = g / u;
    const double d = std::sqrt(fs * fs + gs * gs);
    c = std::fabs(fs) / d;
    r = sign(d, f);
    s = gs / r;
    r = r * u;
  }
}

inline void dlaev2(double a, double b, double c, double& rt1, double& rt2, double& cs1,
                   double& sn1) {
  const double sm = a + c, df = a - c, adf = std::fabs(df), tb = b + b, ab = std::fabs(tb);
  double acmx, acmn;
  if (std::fabs(a) > std::fabs(c)) {
    acmx = a;
    acmn = c;
  } else {
    acmx = c;
    acmn = a;
  }
  double rt;
  if (adf > ab) {
    const double q = ab / adf;
    rt = adf * std::sqrt(1.0 + q * q);
  } else if (adf < ab) {
    const double q = adf / ab;
    rt = ab * std::sqrt(1.0 + q * q);
  } else {
    rt = ab * std::sqrt(2.0);
  }
  int sgn1, sgn2;
  if (sm < 0.0) {
    rt1 = 0.5 * (sm - rt);
    sgn1 = -1;
    rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
  } else if (sm > 0.0) {
    rt1 = 0.5 * (sm + rt);
    sgn1 = 1;
    rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
  } else {
    rt1 = 0.5 * rt;
    rt2 = -0.5 * rt;
    sgn1 = 1;
  }
  double cs;
  if (df >= 0.0) {
    cs = df + rt;
    sgn2 = 1;
  } else {
    cs = df - rt;
    sgn2 = -1;
  }
  if (std::fabs(cs) > ab) {
    const double ct = -tb / cs;
    sn1 = 1.0 / std::sqrt(1.0 + ct * ct);
    cs1 = ct * sn1;
  } else if (ab == 0.0) {
    cs1 = 1.0;
    sn1 = 0.0;
  } else {
    const double tn = -cs / tb;
    cs1 = 1.0 / std::sqrt(1.0 + tn * tn);
    sn1 = tn * cs1;
  }
  if (sgn1 == sgn2) {
    const double tn = cs1;
    cs1 = -sn1;
    sn1 = tn;
  }
}

//! A 3x3 column-major matrix with 1-based `(row, col)` access.
struct M3 {
  double v[3][3];
  double& operator()(int i, int j) { return v[j - 1][i - 1]; }
};

//! `dlasr('R', 'V', F|B, 3, nn, c, s, z(1, col0))`
inline void dlasr_rv(bool forward, int nn, const double* c, const double* s, M3& z, int col0) {
  for (int step = 0; step < nn - 1; ++step) {
    const int j = forward ? step + 1 : nn - 1 - step;
    const double ct = c[j - 1], st = s[j - 1];
    if (ct == 1.0 && st == 0.0) continue;
    for (int i = 1; i <= 3; ++i) {
      const double temp = z(i, col0 + j);
      z(i, col0 + j) = ct * temp - st * z(i, col0 + j - 1);
      z(i, col0 + j - 1) = st * temp + ct * z(i, col0 + j - 1);
    }
  }
}

//! `dsteqr('I', 3, d, e, z)`
inline void dsteqr(double* d, double* e, M3& z) {
  const int n = 3;
  const double eps = std::numeric_limits<double>::epsilon() * 0.5;  // dlamch('E')
  const double eps2 = eps * eps;
  const double safmin = std::numeric_limits<double>::min();
  const double safmax = 1.0 / safmin;
  const double ssfmax = std::sqrt(safmax) / 3.0;
  const double ssfmin = std::sqrt(safmin) / eps2;
  for (int i = 1; i <= 3; ++i)
    for (int j = 1; j <= 3; ++j) z(i, j) = i == j ? 1.0 : 0.0;
  double work[6];
#define IMPBFF_D(k) d[(k) - 1]
#define IMPBFF_E(k) e[(k) - 1]
#define IMPBFF_W(k) work[(k) - 1]
  const int nmaxit = n * 30;
  int jtot = 0, l1 = 1;
  const int nm1 = n - 1;
  while (l1 <= n) {
    if (l1 > 1) IMPBFF_E(l1 - 1) = 0.0;
    int m = l1;
    for (; m <= nm1; ++m) {
      const double tst = std::fabs(IMPBFF_E(m));
      if (tst == 0.0) break;
      if (tst <= (std::sqrt(std::fabs(IMPBFF_D(m))) * std::sqrt(std::fabs(IMPBFF_D(m + 1)))) * eps) {
        IMPBFF_E(m) = 0.0;
        break;
      }
    }
    if (m > nm1) m = n;
    int l = l1;
    const int lsv = l;
    int lend = m;
    const int lendsv = lend;
    l1 = m + 1;
    if (lend == l) continue;
    double anorm = 0.0;
    for (int k = l; k <= lend; ++k) anorm = (std::max)(anorm, std::fabs(IMPBFF_D(k)));
    for (int k = l; k < lend; ++k) anorm = (std::max)(anorm, std::fabs(IMPBFF_E(k)));
    if (anorm == 0.0) continue;
    int iscale = 0;
    if (anorm > ssfmax) {
      iscale = 1;
      for (int k = l; k <= lend; ++k) IMPBFF_D(k) *= ssfmax / anorm;
      for (int k = l; k < lend; ++k) IMPBFF_E(k) *= ssfmax / anorm;
    }
    if (anorm < ssfmin) {
      iscale = 2;
      for (int k = l; k <= lend; ++k) IMPBFF_D(k) *= ssfmin / anorm;
      for (int k = l; k < lend; ++k) IMPBFF_E(k) *= ssfmin / anorm;
    }
    if (std::fabs(IMPBFF_D(lend)) < std::fabs(IMPBFF_D(l))) {
      lend = lsv;
      l = lendsv;
    }
    if (lend > l) {
      // QL iteration
      while (true) {
        m = l;
        if (l != lend) {
          for (; m <= lend - 1; ++m) {
            const double tst = std::fabs(IMPBFF_E(m)) * std::fabs(IMPBFF_E(m));
            if (tst <= (eps2 * std::fabs(IMPBFF_D(m))) * std::fabs(IMPBFF_D(m + 1)) + safmin) break;
          }
          if (m > lend - 1) m = lend;
        } else {
          m = lend;
        }
        if (m < lend) IMPBFF_E(m) = 0.0;
        double p = IMPBFF_D(l);
        if (m == l) {
          IMPBFF_D(l) = p;
          ++l;
          if (l <= lend) continue;
          break;
        }
        if (m == l + 1) {
          double rt1, rt2, c, s;
          dlaev2(IMPBFF_D(l), IMPBFF_E(l), IMPBFF_D(l + 1), rt1, rt2, c, s);
          IMPBFF_W(l) = c;
          IMPBFF_W(n - 1 + l) = s;
          dlasr_rv(false, 2, &IMPBFF_W(l), &IMPBFF_W(n - 1 + l), z, l);
          IMPBFF_D(l) = rt1;
          IMPBFF_D(l + 1) = rt2;
          IMPBFF_E(l) = 0.0;
          l += 2;
          if (l <= lend) continue;
          break;
        }
        if (jtot == nmaxit) break;
        ++jtot;
        double g = (IMPBFF_D(l + 1) - p) / (2.0 * IMPBFF_E(l));
        double r = dlapy2(g, 1.0);
        g = IMPBFF_D(m) - p + (IMPBFF_E(l) / (g + sign(r, g)));
        double s = 1.0, c = 1.0;
        p = 0.0;
        for (int i = m - 1; i >= l; --i) {
          const double f = s * IMPBFF_E(i), b = c * IMPBFF_E(i);
          dlartg(g, f, c, s, r);
          if (i != m - 1) IMPBFF_E(i + 1) = r;
          g = IMPBFF_D(i + 1) - p;
          r = (IMPBFF_D(i) - g) * s + 2.0 * c * b;
          p = s * r;
          IMPBFF_D(i + 1) = g + p;
          g = c * r - b;
          IMPBFF_W(i) = c;
          IMPBFF_W(n - 1 + i) = -s;
        }
        dlasr_rv(false, m - l + 1, &IMPBFF_W(l), &IMPBFF_W(n - 1 + l), z, l);
        IMPBFF_D(l) = IMPBFF_D(l) - p;
        IMPBFF_E(l) = g;
      }
    } else {
      // QR iteration
      while (true) {
        m = l;
        if (l != lend) {
          for (; m >= lend + 1; --m) {
            const double tst = std::fabs(IMPBFF_E(m - 1)) * std::fabs(IMPBFF_E(m - 1));
            if (tst <= (eps2 * std::fabs(IMPBFF_D(m))) * std::fabs(IMPBFF_D(m - 1)) + safmin) break;
          }
          if (m < lend + 1) m = lend;
        } else {
          m = lend;
        }
        if (m > lend) IMPBFF_E(m - 1) = 0.0;
        double p = IMPBFF_D(l);
        if (m == l) {
          IMPBFF_D(l) = p;
          --l;
          if (l >= lend) continue;
          break;
        }
        if (m == l - 1) {
          double rt1, rt2, c, s;
          dlaev2(IMPBFF_D(l - 1), IMPBFF_E(l - 1), IMPBFF_D(l), rt1, rt2, c, s);
          IMPBFF_W(m) = c;
          IMPBFF_W(n - 1 + m) = s;
          dlasr_rv(true, 2, &IMPBFF_W(m), &IMPBFF_W(n - 1 + m), z, l - 1);
          IMPBFF_D(l - 1) = rt1;
          IMPBFF_D(l) = rt2;
          IMPBFF_E(l - 1) = 0.0;
          l -= 2;
          if (l >= lend) continue;
          break;
        }
        if (jtot == nmaxit) break;
        ++jtot;
        double g = (IMPBFF_D(l - 1) - p) / (2.0 * IMPBFF_E(l - 1));
        double r = dlapy2(g, 1.0);
        g = IMPBFF_D(m) - p + (IMPBFF_E(l - 1) / (g + sign(r, g)));
        double s = 1.0, c = 1.0;
        p = 0.0;
        for (int i = m; i <= l - 1; ++i) {
          const double f = s * IMPBFF_E(i), b = c * IMPBFF_E(i);
          dlartg(g, f, c, s, r);
          if (i != m) IMPBFF_E(i - 1) = r;
          g = IMPBFF_D(i) - p;
          r = (IMPBFF_D(i + 1) - g) * s + 2.0 * c * b;
          p = s * r;
          IMPBFF_D(i) = g + p;
          g = c * r - b;
          IMPBFF_W(i) = c;
          IMPBFF_W(n - 1 + i) = s;
        }
        dlasr_rv(true, l - m + 1, &IMPBFF_W(m), &IMPBFF_W(n - 1 + m), z, m);
        IMPBFF_D(l) = IMPBFF_D(l) - p;
        IMPBFF_E(l - 1) = g;
      }
    }
    if (iscale == 1) {
      for (int k = lsv; k <= lendsv; ++k) IMPBFF_D(k) *= anorm / ssfmax;
      for (int k = lsv; k < lendsv; ++k) IMPBFF_E(k) *= anorm / ssfmax;
    } else if (iscale == 2) {
      for (int k = lsv; k <= lendsv; ++k) IMPBFF_D(k) *= anorm / ssfmin;
      for (int k = lsv; k < lendsv; ++k) IMPBFF_E(k) *= anorm / ssfmin;
    }
    if (jtot >= nmaxit) break;
  }
  // selection sort, swapping the vectors with the values
  for (int ii = 2; ii <= n; ++ii) {
    const int i = ii - 1;
    int k = i;
    double p = IMPBFF_D(i);
    for (int j = ii; j <= n; ++j) {
      if (IMPBFF_D(j) < p) {
        k = j;
        p = IMPBFF_D(j);
      }
    }
    if (k != i) {
      IMPBFF_D(k) = IMPBFF_D(i);
      IMPBFF_D(i) = p;
      for (int r = 1; r <= 3; ++r) std::swap(z(r, i), z(r, k));
    }
  }
#undef IMPBFF_D
#undef IMPBFF_E
#undef IMPBFF_W
}

}  // namespace lapack

//! `np.linalg.eigh` of a symmetric 3x3 matrix.
/*! \param[in] m row-major, `m[3 * i + j]`
    \param[out] w eigenvalues, ascending
    \param[out] vectors `vectors[3 * k + i]` is component `i` of eigenvector
                `k`, i.e. numpy's `v[i, k]` */
inline void eigh3(const double m[9], double w[3], double vectors[9]) {
  using lapack::M3;
  M3 a;
  for (int i = 1; i <= 3; ++i)
    for (int j = 1; j <= 3; ++j) a(i, j) = m[3 * (i - 1) + (j - 1)];
  double d[3], e[2], tau[2];
  // dsytd2, lower
  for (int i = 1; i <= 2; ++i) {
    double taui = 0.0;
    double& alpha = a(i + 1, i);
    if (3 - i > 1) {
      double ss = 0.0;
      for (int k = i + 2; k <= 3; ++k) ss += a(k, i) * a(k, i);
      const double xnorm = std::sqrt(ss);
      if (xnorm != 0.0) {
        const double beta = -lapack::sign(lapack::dlapy2(alpha, xnorm), alpha);
        taui = (beta - alpha) / beta;
        const double scale = 1.0 / (alpha - beta);
        for (int k = i + 2; k <= 3; ++k) a(k, i) = a(k, i) * scale;
        alpha = beta;
      }
    }
    e[i - 1] = a(i + 1, i);
    if (taui != 0.0) {
      a(i + 1, i) = 1.0;
      const int mm = 3 - i;
      double v[3], y[3] = {0.0, 0.0, 0.0};
      for (int k = 0; k < mm; ++k) v[k] = a(i + 1 + k, i);
      for (int j = 0; j < mm; ++j) {  // dsymv, lower
        const double temp1 = taui * v[j];
        double temp2 = 0.0;
        y[j] += temp1 * a(i + 1 + j, i + 1 + j);
        for (int r = j + 1; r < mm; ++r) {
          y[r] += temp1 * a(i + 1 + r, i + 1 + j);
          temp2 += a(i + 1 + r, i + 1 + j) * v[r];
        }
        y[j] += taui * temp2;
      }
      double dot = 0.0;
      for (int k = 0; k < mm; ++k) dot += y[k] * v[k];
      const double alph = -0.5 * taui * dot;
      for (int k = 0; k < mm; ++k) y[k] += alph * v[k];
      for (int j = 0; j < mm; ++j) {  // dsyr2, lower, alpha = -1
        if (v[j] == 0.0 && y[j] == 0.0) continue;
        const double t1 = -y[j], t2 = -v[j];
        for (int r = j; r < mm; ++r) {
          a(i + 1 + r, i + 1 + j) = a(i + 1 + r, i + 1 + j) + v[r] * t1 + y[r] * t2;
        }
      }
      a(i + 1, i) = e[i - 1];
    }
    d[i - 1] = a(i, i);
    tau[i - 1] = taui;
  }
  d[2] = a(3, 3);
  M3 z;
  lapack::dsteqr(d, e, z);
  // dormtr('L', 'L', 'N') = dorm2r on Z(2:3, :) with the reflectors of A(2:3, 1:2)
  for (int i = 2; i >= 1; --i) {
    const double t = tau[i - 1];
    if (t == 0.0) continue;
    const int mi = 3 - i;
    double v[2];
    v[0] = 1.0;
    for (int k = 1; k < mi; ++k) v[k] = a(i + 1 + k, i);
    int lastv = mi;
    while (lastv > 0 && v[lastv - 1] == 0.0) --lastv;
    if (lastv == 0) continue;
    double wv[3];
    for (int col = 1; col <= 3; ++col) {
      double s = 0.0;
      for (int r = 0; r < lastv; ++r) s += z(i + 1 + r, col) * v[r];
      wv[col - 1] = s;
    }
    for (int col = 1; col <= 3; ++col)
      for (int r = 0; r < lastv; ++r) z(i + 1 + r, col) += (-t * wv[col - 1]) * v[r];
  }
  for (int k = 0; k < 3; ++k) {
    w[k] = d[k];
    for (int r = 0; r < 3; ++r) vectors[3 * k + r] = z(r + 1, k + 1);
  }
}

}  // namespace numpy_compat
IMPBFF_END_INTERNAL_NAMESPACE

#endif  // IMPBFF_INTERNAL_NUMPYCOMPAT_H
