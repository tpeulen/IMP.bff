/**
 *  \file IMP/bff/InferenceCanonicalForm.h
 *  \brief A Gaussian factor in canonical (information) form, where
 *         conditioning and marginalisation are closed-form.
 *
 *  A Gaussian factor over a named scope is written
 *
 *      phi(x) = exp(g + h^T x - x^T K x / 2)
 *
 *  with K the precision and h the information vector. The operations exact
 *  inference needs are linear algebra in this form, and none of them needs K
 *  to be invertible:
 *
 *  - product: add K, h and g on the union of the scopes;
 *  - reduce / condition on x_B = v: slice K to the rest, h_A - K_AB v, and
 *    g + h_B v - v K_BB v / 2;
 *  - marginalise x_B out: the Schur complement K_AA - K_AB K_BB^-1 K_BA, with
 *    the matching h and g -- which needs K_BB positive definite, because the
 *    integral over a direction nothing constrains diverges.
 *
 *  Converting back to a mean and a covariance needs K positive definite. A
 *  factor over a parameter the data do not constrain has a singular K, and
 *  that is the informative case, not an error: the form still multiplies,
 *  conditions and marginalises, and get_null_space() names the unconstrained
 *  directions. Only get_mean()/get_covariance() refuse it.
 *
 *  Ported from aGrUM's pyAgrum CLG package (`clg/canonicalForm.py`, dual
 *  LGPL-3.0-or-later OR MIT, Wuillemin and Gonzales), whose semantics this
 *  mirrors -- product, division, augment, reduce ignoring out-of-scope
 *  evidence, marginalize, fromCLG -- and from ChiSurf's former
 *  `chisurf.core.fitting.canonical`, which added what aGrUM lacks: building a
 *  form from moments with a given total mass, the log normaliser, the log
 *  density, strict conditioning, a marginal that keeps a named order, and a
 *  refusal of a repeated name. The scope is addressed by name, not by a
 *  sorted id; a variable may hold several numbers (its size). See
 *  okf/prds/prd-151.md.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#ifndef IMPBFF_INFERENCECANONICALFORM_H
#define IMPBFF_INFERENCECANONICALFORM_H

#include <IMP/bff/bff_config.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A Gaussian factor exp(g + h.x - x.K.x/2) over a named scope.
/*!
    Matrices cross as flat row-major arrays; `K` over a scope of total
    dimension d comes back from #get_precision with `d*d` entries. A variable
    holds `size` numbers (1 unless given), and a variable's numbers are
    contiguous in the order of the scope.

    Every operation returns a new form; a form is a value.
*/
class IMPBFFEXPORT InferenceCanonicalForm {
 public:
  //! The empty form: no variables, g = 0 (the constant 1).
  InferenceCanonicalForm();

  //! A form from its parameters.
  /*!
      \param[in] names variable names, unique
      \param[in] precision K, flat row-major, d*d; symmetrised on entry
      \param[in] information h, d
      \param[in] log_constant g
      \param[in] sizes numbers each variable holds; empty means all 1
      \throw ValueException on a repeated name or a size mismatch
  */
  InferenceCanonicalForm(const std::vector<std::string>& names,
                         const std::vector<double>& precision,
                         const std::vector<double>& information,
                         double log_constant = 0.0,
                         const std::vector<int>& sizes = std::vector<int>());

  //! A form from a mean and a covariance, integrating to exp(log_mass).
  /*!
      g is chosen so the integral of the form is `exp(log_mass)`: zero gives
      a normalised density, and a log evidence passed here is carried through
      marginalisation unchanged (#get_log_normalizer).
      \throw ValueException when the covariance is not positive definite
  */
  static InferenceCanonicalForm from_moments(
      const std::vector<std::string>& names, const std::vector<double>& mean,
      const std::vector<double>& covariance, double log_mass = 0.0,
      const std::vector<int>& sizes = std::vector<int>());

  //! The factor of a linear-Gaussian conditional density.
  /*!
      `p(x | parents) = N(mu + sum_i weights_i parents_i, sigma^2)`, over the
      scope `[variable] + parents` -- aGrUM's `CanonicalForm.fromCLG`
      (Lauritzen 1992). \p sigma is a standard deviation.
  */
  static InferenceCanonicalForm from_linear_gaussian(
      const std::string& variable, const std::vector<std::string>& parents,
      double mu, double sigma, const std::vector<double>& weights);

  //! Variable names, in the order of K and h.
  std::vector<std::string> get_names() const { return names_; }
  //! Numbers each variable holds, parallel to get_names().
  std::vector<int> get_sizes() const { return sizes_; }
  //! Number of variables.
  unsigned int get_number_of_variables() const {
    return static_cast<unsigned int>(names_.size());
  }
  //! Total dimension d, the sum of the sizes.
  int get_dimension() const { return static_cast<int>(h_.size()); }
  //! Whether a name is in the scope.
  bool has_variable(const std::string& name) const;
  //! The offset of a variable's first number in K and h; -1 if absent.
  int get_offset(const std::string& name) const;

  //! K, flat row-major d*d.
  void get_precision(double** out_view, int* n_out_view) const;
  //! h, d numbers.
  void get_information(double** out_view, int* n_out_view) const;
  //! g.
  double get_log_constant() const { return g_; }

  //! The same factor over a wider scope, zero in the new variables.
  /*!
      aGrUM's `augment`. Names already in scope are left alone; new ones are
      appended in the order given.
  */
  InferenceCanonicalForm extend(
      const std::vector<std::string>& names,
      const std::vector<int>& sizes = std::vector<int>()) const;

  //! The product, on the union of the scopes (this scope first).
  /*! \throw ValueException when a shared variable's size differs */
  InferenceCanonicalForm product(const InferenceCanonicalForm& other) const;

  //! The quotient: K, h and g subtracted. aGrUM's `__truediv__`.
  InferenceCanonicalForm divide(const InferenceCanonicalForm& other) const;

  //! Enter evidence, ignoring any variable not in the scope.
  /*!
      aGrUM's `reduce`: a factor only sees its own evidence, which is what
      variable elimination hands every factor. \p values concatenates each
      named variable's numbers, in the order of \p names; \p sizes says how
      many each holds (empty: one each), which is how numbers of a variable
      not in this scope are skipped.
      \throw ValueException when a size disagrees with the scope or the
             values do not add up
  */
  InferenceCanonicalForm reduce(
      const std::vector<std::string>& names, const std::vector<double>& values,
      const std::vector<int>& sizes = std::vector<int>()) const;

  //! Hold variables at values; every name must be in the scope.
  /*!
      The strict form of #reduce. For a Gaussian the result *is* what fixing
      those variables and re-optimising the rest gives: the constrained
      minimum of a quadratic is its conditional mode.
      \throw ValueException on a name not in scope
  */
  InferenceCanonicalForm condition(const std::vector<std::string>& names,
                                   const std::vector<double>& values) const;

  //! Integrate the named variables out. aGrUM's `marginalize`.
  /*!
      \throw ValueException on a name not in scope, or when the eliminated
             block of K is not positive definite (the integral diverges)
  */
  InferenceCanonicalForm marginalize(
      const std::vector<std::string>& names) const;

  //! Integrate out everything not in \p keep; the result is ordered as \p keep.
  /*! \throw ValueException as #marginalize, or on a repeated name */
  InferenceCanonicalForm marginal(const std::vector<std::string>& keep) const;

  //! Whether K is positive definite, i.e. the form is a proper Gaussian.
  bool get_is_proper() const;
  //! Rank of K: eigenvalues above \p relative_tolerance times the largest.
  int get_rank(double relative_tolerance = 1e-10) const;
  //! The unconstrained directions: an orthonormal basis of K's null space.
  /*!
      Flat row-major `d * k` for k = d - #get_rank: column j is the j-th
      direction, in the coordinates of K and h. Empty for a proper form.
  */
  void get_null_space(double** out_view, int* n_out_view,
                      double relative_tolerance = 1e-10) const;

  //! K^-1 h. aGrUM's `toGaussian`.
  /*! \throw ValueException when K is not positive definite; the message
             names the rank, since that is the answer the caller wanted */
  void get_mean(double** out_view, int* n_out_view) const;
  //! K^-1, flat row-major d*d.
  /*! \throw ValueException when K is not positive definite */
  void get_covariance(double** out_view, int* n_out_view) const;

  //! log of the integral of the form.
  /*!
      `g + (d log 2 pi - log det K + h K^-1 h) / 2`. For a form built by
      #from_moments this is its log mass, and marginalisation leaves it
      unchanged. +infinity when K is not positive definite: the integral
      diverges. For the empty form it is g.
  */
  double get_log_normalizer() const;

  //! `g + h.x - x.K.x/2` at a point of the scope.
  double get_log_density(const std::vector<double>& x) const;

  //! A one-line description of the scope.
  std::string get_description() const;

 private:
  std::vector<std::string> names_;
  std::vector<int> sizes_;
  std::vector<int> offsets_;
  std::vector<double> K_;  // row-major d*d
  std::vector<double> h_;
  double g_ = 0.0;

  void finish();  // offsets, name uniqueness, symmetry
  //! K, h and g of other added with \p sign on the union scope.
  InferenceCanonicalForm combine(const InferenceCanonicalForm& other,
                                 double sign) const;
  //! The positions in K/h of the named variables, or throw.
  std::vector<int> indices_of(const std::vector<std::string>& names) const;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_INFERENCECANONICALFORM_H
