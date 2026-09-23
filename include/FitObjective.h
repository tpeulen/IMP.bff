/**
 * \file IMP/bff/FitObjective.h
 * \brief A graph node whose evaluation is a weighted residual vector.
 *
 * What a least-squares fit, a model search and a joint fit all need from the
 * end of a graph is the same thing: after `update()`, the residuals. This is
 * that contract as a type, so they take an objective rather than a node plus
 * the name of the port its residuals happen to sit on. `FitChiSquared` and
 * `FitJointChiSquared` are objectives; a model this library cannot represent
 * is one too, as a Python subclass that calls #set_residuals from
 * `evaluate()`.
 *
 * The residuals live on the output port named #residuals_port_name, created
 * on first use (a port needs its node to be owned by a `std::shared_ptr`, which
 * a constructor cannot rely on). A description that declares that port supplies
 * it instead.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_FITOBJECTIVE_H
#define IMPBFF_FITOBJECTIVE_H

#include <IMP/bff/bff_config.h>

#include <memory>
#include <string>
#include <vector>

#include <IMP/bff/GraphNode.h>
#include <IMP/bff/GraphPort.h>

IMPBFF_BEGIN_NAMESPACE

//! A node that evaluates to weighted residuals; what fits and searches consume.
class IMPBFFEXPORT FitObjective : public GraphNode {
 public:
  //! The output port every objective carries its residuals on.
  static const char* const residuals_port_name;

  explicit FitObjective(const std::string& name = "objective");
  virtual ~FitObjective();

  //! The port carrying the residuals, created on first use.
  std::shared_ptr<GraphPort> get_residuals_port();

  //! The weighted residuals of the last evaluation; empty before one.
  const std::vector<double>& get_residuals() const;

  //! Store this evaluation's residuals; what `evaluate()` ends with.
  void set_residuals(const std::vector<double>& residuals);

  //! Sum of squared residuals of the last evaluation; infinite if any is NaN.
  double get_chi2() const;

  //! `chi2 / (n_residuals - n_free - 1)`.
  double get_chi2r(int n_free) const;

  unsigned int get_number_of_residuals() const {
    return static_cast<unsigned int>(get_residuals().size());
  }
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_FITOBJECTIVE_H
