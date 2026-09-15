/**
 *  \file IMP/bff/TabulatedDistances.h
 *  \brief Given distance distributions on a common axis, mixed by fractions.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_TABULATEDDISTANCES_H
#define IMPBFF_TABULATEDDISTANCES_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/GraphNode.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A distance distribution mixed from K distributions that are given, not modelled.
/*!
    The producer for distributions computed elsewhere and fixed during a fit:
    one per structure of an ensemble (accessible-volume clouds, `histogram_rda`),
    or a single measured one. Input ports: `axis` (the distances, Å) and, per
    distribution k, the vector `distribution<k>` on that axis and the scalar
    `fraction<k>`. The output, keyed by the node's name, is the interleaved
    `(p, r)` distribution `FRETSpectrumNode` reads:
    `p_j = sum_k |x_k| q_kj / sum_k,j |x_k| q_kj`, where `q_k` is distribution k
    normalised to unit sum, so a fraction is a share of the ensemble whatever
    scale a distribution came in. Weights below `threshold` times the largest
    are then set to zero (ChiSurf's structure model trims at 1e-3).

    Settings (#configure): `number_of_distributions`, `threshold` (default 0).
*/
class IMPBFFEXPORT TabulatedDistances : public GraphNode {
 public:
  explicit TabulatedDistances(const std::string& name = "distances");

  void set_number_of_distributions(int n);
  int get_number_of_distributions() const { return n_distributions_; }
  void set_threshold(double relative);

  //! d p_j / d fraction_k of the output weights, row-major `n_axis x K`.
  /*! Analytic; `|fraction|` carries `sign(x)`, undefined at 0, and a weight
      set to zero by the threshold has zero derivative. */
  std::vector<double> get_weights_jacobian() const;
  std::vector<std::string> get_parameter_names() const;

  void evaluate() override;
  std::string get_node_type() const override;
  void configure(const std::string& json_text) override;

 private:
  //! The unit-sum distributions, their fractions and the axis.
  void read_inputs(std::vector<double>* axis, std::vector<std::vector<double> >* q,
                   std::vector<double>* fractions) const;
  int n_distributions_ = 0;
  double threshold_ = 0.0;
  std::vector<double> out_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_TABULATEDDISTANCES_H
