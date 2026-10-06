/**
 * \file Numerics.cpp
 * \brief Special functions, distribution functions and expm over arrays.
 *
 * The special functions are SciPy's own (xsf, thirdparty/xsf), the
 * distribution functions are Boost.Math under scipy's policies, each with
 * the edge-case handling of the scipy wrapper it replaces
 * (scipy/special/boost_special_functions.h) -- so a caller moving off scipy
 * gets scipy's numbers, not merely numbers close to them.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/Numerics.h>
#include <IMP/bff/internal/OutputView.h>

#include <IMP/bff/IMPCompatibility.h>

// SciPy's special-function library. Private: only this file includes it.
#include <xsf/cephes/fresnl.h>
#include <xsf/cephes/i0.h>
#include <xsf/cephes/j0.h>
#include <xsf/cephes/j1.h>
#include <xsf/digamma.h>
#include <xsf/erf.h>
#include <xsf/gamma.h>
#include <xsf/stats.h>
#include <xsf/zeta.h>

#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/special_functions/beta.hpp>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <cmath>
#include <limits>

IMPBFF_BEGIN_NAMESPACE

namespace {

const double kNumericsNaN = std::numeric_limits<double>::quiet_NaN();
const double kNumericsInf = std::numeric_limits<double>::infinity();

// scipy's NumericsSpecialPolicy (boost_special_functions.h): compute in double, no
// promotion, generous root iterations. Errors throw and are mapped below the
// way scipy's wrappers map them.
typedef boost::math::policies::policy<
    boost::math::policies::promote_float<false>,
    boost::math::policies::promote_double<false>,
    boost::math::policies::max_root_iterations<400>,
    boost::math::policies::discrete_quantile<boost::math::policies::real> >
    NumericsSpecialPolicy;

// scipy's NumericsStatsPolicy, with its two user_error handlers (warn, return the
// best guess) as ignore_error (return the best guess): same value, no
// warning channel to warn on.
typedef boost::math::policies::policy<
    boost::math::policies::domain_error<boost::math::policies::ignore_error>,
    boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::evaluation_error<boost::math::policies::ignore_error>,
    boost::math::policies::promote_float<false>,
    boost::math::policies::promote_double<false>,
    boost::math::policies::discrete_quantile<
        boost::math::policies::integer_round_up> >
    NumericsStatsPolicy;

void numerics_check_lengths(int n_reference, int n_other, const char* who) {
  if (n_other != n_reference) {
    IMP_THROW(who << ": the arguments must have the same length ("
                  << n_reference << " and " << n_other
                  << "); broadcast them first",
              ValueException);
  }
}

template <class F>
void numerics_unary(const double* x, int n, double** out_view, int* n_out_view, F f) {
  double* out = internal::new_double_view(static_cast<std::size_t>(n), out_view,
                                          n_out_view);
  if (out == nullptr) return;
  for (int i = 0; i < n; ++i) out[i] = f(x[i]);
}

template <class F>
void numerics_binary(const double* a, int n_a, const double* b, int n_b,
            double** out_view, int* n_out_view, const char* who, F f) {
  numerics_check_lengths(n_a, n_b, who);
  double* out = internal::new_double_view(static_cast<std::size_t>(n_a), out_view,
                                          n_out_view);
  if (out == nullptr) return;
  for (int i = 0; i < n_a; ++i) out[i] = f(a[i], b[i]);
}

template <class F>
void numerics_ternary(const double* a, int n_a, const double* b, int n_b,
             const double* c, int n_c, double** out_view, int* n_out_view,
             const char* who, F f) {
  numerics_check_lengths(n_a, n_b, who);
  numerics_check_lengths(n_a, n_c, who);
  double* out = internal::new_double_view(static_cast<std::size_t>(n_a), out_view,
                                          n_out_view);
  if (out == nullptr) return;
  for (int i = 0; i < n_a; ++i) out[i] = f(a[i], b[i], c[i]);
}

// scipy's ibeta_wrap.
double numerics_betainc_one(double a, double b, double x) {
  if (std::isnan(a) || std::isnan(b) || std::isnan(x)) return kNumericsNaN;
  if (a < 0 || b < 0 || x < 0 || x > 1) return kNumericsNaN;
  if ((a == 0 && b == 0) || (std::isinf(a) && std::isinf(b))) return kNumericsNaN;
  if (a == 0 || std::isinf(b)) return x > 0 ? 1.0 : 0.0;
  if (b == 0 || std::isinf(a)) return x < 1 ? 0.0 : 1.0;
  try {
    return boost::math::ibeta(a, b, x, NumericsSpecialPolicy());
  } catch (const std::overflow_error&) {
    return kNumericsInf;
  } catch (const std::underflow_error&) {
    return 0.0;
  } catch (...) {
    return kNumericsNaN;
  }
}

// scipy's ibeta_inv_wrap.
double numerics_betaincinv_one(double a, double b, double p) {
  if (std::isnan(a) || std::isnan(b) || std::isnan(p)) return kNumericsNaN;
  if (a <= 0 || b <= 0 || p < 0 || p > 1) return kNumericsNaN;
  try {
    return boost::math::ibeta_inv(a, b, p, NumericsSpecialPolicy());
  } catch (const std::overflow_error&) {
    return kNumericsInf;
  } catch (const std::underflow_error&) {
    return 0.0;
  } catch (...) {
    return kNumericsNaN;
  }
}

// scipy's t_cdf_wrap.
double numerics_stdtr_one(double v, double x) {
  if (std::isnan(x) || std::isnan(v)) return kNumericsNaN;
  if (v <= 0) return kNumericsNaN;
  if (std::isinf(x)) return x > 0 ? 1.0 : 0.0;
  double y;
  try {
    y = boost::math::cdf(
        boost::math::students_t_distribution<double, NumericsSpecialPolicy>(v), x);
  } catch (...) {
    y = kNumericsNaN;
  }
  if (y < 0 || y > 1) y = kNumericsNaN;
  return y;
}

// scipy's binom_pmf_wrap.
double numerics_binom_pmf_one(double k, double n, double p) {
  if (!std::isfinite(k)) return kNumericsNaN;
  try {
    return boost::math::pdf(
        boost::math::binomial_distribution<double, NumericsStatsPolicy>(n, p), k);
  } catch (...) {
    return kNumericsNaN;
  }
}

// scipy's ncx2_pdf_wrap.
double numerics_ncx2_pdf_one(double x, double k, double l) {
  if (!std::isfinite(x)) return kNumericsNaN;
  try {
    return boost::math::pdf(
        boost::math::non_central_chi_squared_distribution<double, NumericsSpecialPolicy>(k, l),
        x);
  } catch (const std::overflow_error&) {
    return kNumericsInf;
  } catch (const std::underflow_error&) {
    return 0.0;
  } catch (...) {
    return kNumericsNaN;
  }
}

// scipy's f_cdf_wrap / f_sf_wrap / f_ppf_wrap: scipy 1.18 computes fdtr,
// fdtrc and fdtri with Boost's Fisher F, not with the Cephes functions of
// the same names that xsf also carries (they differ at ~1e-14).
double numerics_fdtr_one(double dfn, double dfd, double x) {
  if (std::isnan(x) || std::isnan(dfn) || std::isnan(dfd)) return kNumericsNaN;
  if (dfn <= 0 || dfd <= 0 || x < 0) return kNumericsNaN;
  if (std::isinf(x)) return 1.0;
  double y;
  try {
    y = boost::math::cdf(
        boost::math::fisher_f_distribution<double, NumericsSpecialPolicy>(dfn, dfd), x);
  } catch (...) {
    y = kNumericsNaN;
  }
  if (y < 0 || y > 1) y = kNumericsNaN;
  return y;
}

double numerics_fdtrc_one(double dfn, double dfd, double x) {
  if (std::isnan(x) || std::isnan(dfn) || std::isnan(dfd)) return kNumericsNaN;
  if (dfn <= 0 || dfd <= 0 || x < 0) return kNumericsNaN;
  if (std::isinf(x)) return 0.0;
  double y;
  try {
    y = boost::math::cdf(boost::math::complement(
        boost::math::fisher_f_distribution<double, NumericsSpecialPolicy>(dfn, dfd), x));
  } catch (...) {
    y = kNumericsNaN;
  }
  if (y < 0 || y > 1) y = kNumericsNaN;
  return y;
}

double numerics_fdtri_one(double dfn, double dfd, double p) {
  if (std::isnan(p) || std::isnan(dfn) || std::isnan(dfd)) return kNumericsNaN;
  if (dfn <= 0 || dfd <= 0 || p < 0 || p > 1) return kNumericsNaN;
  double y;
  try {
    y = boost::math::quantile(
        boost::math::fisher_f_distribution<double, NumericsSpecialPolicy>(dfn, dfd), p);
  } catch (const std::overflow_error&) {
    y = kNumericsInf;
  } catch (const std::underflow_error&) {
    y = 0.0;
  } catch (...) {
    y = kNumericsNaN;
  }
  if (y < 0) y = kNumericsNaN;
  return y;
}

}  // namespace

void gammaln_array(const double* in_x, int n_x, double** out_view, int* n_out_view) {
  numerics_unary(in_x, n_x, out_view, n_out_view, [](double x) { return xsf::gammaln(x); });
}

void erf_array(const double* in_x, int n_x, double** out_view, int* n_out_view) {
  numerics_unary(in_x, n_x, out_view, n_out_view, [](double x) { return xsf::erf(x); });
}

void erfc_array(const double* in_x, int n_x, double** out_view, int* n_out_view) {
  numerics_unary(in_x, n_x, out_view, n_out_view, [](double x) { return xsf::erfc(x); });
}

void digamma_array(const double* in_x, int n_x, double** out_view, int* n_out_view) {
  numerics_unary(in_x, n_x, out_view, n_out_view, [](double x) { return xsf::digamma(x); });
}

void i0e_array(const double* in_x, int n_x, double** out_view, int* n_out_view) {
  numerics_unary(in_x, n_x, out_view, n_out_view, [](double x) { return xsf::cephes::i0e(x); });
}

void j0_array(const double* in_x, int n_x, double** out_view, int* n_out_view) {
  numerics_unary(in_x, n_x, out_view, n_out_view, [](double x) { return xsf::cephes::j0(x); });
}

void j1_array(const double* in_x, int n_x, double** out_view, int* n_out_view) {
  numerics_unary(in_x, n_x, out_view, n_out_view, [](double x) { return xsf::cephes::j1(x); });
}

void ndtr_array(const double* in_x, int n_x, double** out_view, int* n_out_view) {
  numerics_unary(in_x, n_x, out_view, n_out_view, [](double x) { return xsf::ndtr(x); });
}

void gammainc_array(const double* in_a, int n_a, const double* in_x, int n_x,
                    double** out_view, int* n_out_view) {
  numerics_binary(in_a, n_a, in_x, n_x, out_view, n_out_view, "gammainc",
         [](double a, double x) { return xsf::gammainc(a, x); });
}

void gammaincc_array(const double* in_a, int n_a, const double* in_x, int n_x,
                     double** out_view, int* n_out_view) {
  numerics_binary(in_a, n_a, in_x, n_x, out_view, n_out_view, "gammaincc",
         [](double a, double x) { return xsf::gammaincc(a, x); });
}

void zeta_array(const double* in_s, int n_s, const double* in_q, int n_q,
                double** out_view, int* n_out_view) {
  numerics_binary(in_s, n_s, in_q, n_q, out_view, n_out_view, "zeta",
         [](double s, double q) { return xsf::zeta(s, q); });
}

void fresnel_array(const double* in_x, int n_x, double** out_view, int* n_out_view,
                   double** out_second, int* n_out_second) {
  const std::size_t n = static_cast<std::size_t>(n_x);
  double* s = internal::new_double_view(n, out_view, n_out_view);
  double* c = internal::new_double_view(n, out_second, n_out_second);
  if (s == nullptr || c == nullptr) return;
  for (std::size_t i = 0; i < n; ++i) xsf::cephes::fresnl(in_x[i], &s[i], &c[i]);
}

void betainc_array(const double* in_a, int n_a, const double* in_b, int n_b,
                   const double* in_x, int n_x, double** out_view, int* n_out_view) {
  numerics_ternary(in_a, n_a, in_b, n_b, in_x, n_x, out_view, n_out_view, "betainc",
          numerics_betainc_one);
}

void betaincinv_array(const double* in_a, int n_a, const double* in_b, int n_b,
                      const double* in_p, int n_p, double** out_view,
                      int* n_out_view) {
  numerics_ternary(in_a, n_a, in_b, n_b, in_p, n_p, out_view, n_out_view, "betaincinv",
          numerics_betaincinv_one);
}

void fdtr_array(const double* in_dfn, int n_dfn, const double* in_dfd, int n_dfd,
                const double* in_x, int n_x, double** out_view, int* n_out_view) {
  numerics_ternary(in_dfn, n_dfn, in_dfd, n_dfd, in_x, n_x, out_view, n_out_view, "fdtr",
          numerics_fdtr_one);
}

void fdtrc_array(const double* in_dfn, int n_dfn, const double* in_dfd, int n_dfd,
                 const double* in_x, int n_x, double** out_view, int* n_out_view) {
  numerics_ternary(in_dfn, n_dfn, in_dfd, n_dfd, in_x, n_x, out_view, n_out_view, "fdtrc",
          numerics_fdtrc_one);
}

void fdtri_array(const double* in_dfn, int n_dfn, const double* in_dfd, int n_dfd,
                 const double* in_p, int n_p, double** out_view, int* n_out_view) {
  numerics_ternary(in_dfn, n_dfn, in_dfd, n_dfd, in_p, n_p, out_view, n_out_view, "fdtri",
          numerics_fdtri_one);
}

void chdtrc_array(const double* in_df, int n_df, const double* in_x, int n_x,
                  double** out_view, int* n_out_view) {
  numerics_binary(in_df, n_df, in_x, n_x, out_view, n_out_view, "chdtrc",
         [](double df, double x) { return xsf::chdtrc(df, x); });
}

void chdtri_array(const double* in_df, int n_df, const double* in_q, int n_q,
                  double** out_view, int* n_out_view) {
  numerics_binary(in_df, n_df, in_q, n_q, out_view, n_out_view, "chdtri",
         [](double df, double q) { return xsf::chdtri(df, q); });
}

void stdtr_array(const double* in_df, int n_df, const double* in_t, int n_t,
                 double** out_view, int* n_out_view) {
  numerics_binary(in_df, n_df, in_t, n_t, out_view, n_out_view, "stdtr", numerics_stdtr_one);
}

void binom_pmf_array(const double* in_k, int n_k, const double* in_n, int n_n,
                     const double* in_p, int n_p, double** out_view,
                     int* n_out_view) {
  numerics_ternary(in_k, n_k, in_n, n_n, in_p, n_p, out_view, n_out_view, "binom_pmf",
          numerics_binom_pmf_one);
}

void ncx2_pdf_array(const double* in_x, int n_x, const double* in_df, int n_df,
                    const double* in_nc, int n_nc, double** out_view,
                    int* n_out_view) {
  numerics_ternary(in_x, n_x, in_df, n_df, in_nc, n_nc, out_view, n_out_view, "ncx2_pdf",
          numerics_ncx2_pdf_one);
}

void expm(const double* in_matrix, int n_rows, int n_cols, double** out_matrix,
          int* n_out_rows, int* n_out_cols) {
  if (n_rows != n_cols) {
    IMP_THROW("expm: the matrix must be square, got " << n_rows << " x "
                                                      << n_cols,
              ValueException);
  }
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      RowMatrix;
  const std::size_t n2 = static_cast<std::size_t>(n_rows) * n_cols;
  double* out = internal::new_double_view(n2, out_matrix, n_out_rows);
  if (n_out_cols != nullptr) *n_out_cols = n_cols;
  if (out == nullptr) {
    if (n_out_cols != nullptr) *n_out_cols = 0;
    return;
  }
  if (n_out_rows != nullptr) *n_out_rows = n_rows;
  if (n_rows == 0) return;
  Eigen::Map<const RowMatrix> a(in_matrix, n_rows, n_cols);
  Eigen::Map<RowMatrix> e(out, n_rows, n_cols);
  e = a.exp();
}

IMPBFF_END_NAMESPACE
