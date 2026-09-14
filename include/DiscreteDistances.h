/**
 *  \file IMP/bff/DiscreteDistances.h
 *  \brief A distance distribution of a few discrete distances, as a node.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_DISCRETEDISTANCES_H
#define IMPBFF_DISCRETEDISTANCES_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/GraphNode.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A distance distribution that is a set of discrete distances.
/*!
    Scalar input ports `distance0`, `amplitude0`, `distance1`, ... hold each
    distance and its weight. The output, keyed by the node's own name, is the
    interleaved `(p0, r0, p1, r1, ...)` distribution `FRETSpectrumNode` reads,
    with `p_i = |amplitude_i| / sum_j |amplitude_j|` and `r_i = |distance_i|`
    -- the magnitudes ChiSurf's discrete FRET model uses, so a sign an
    optimiser wanders through is not a different model.

    Settings (#configure): `number_of_distances`.
*/
class IMPBFFEXPORT DiscreteDistances : public GraphNode {
 public:
  explicit DiscreteDistances(const std::string& name = "distances");

  void set_number_of_distances(int n);
  int get_number_of_distances() const { return n_distances_; }

  void evaluate() override;
  std::string get_node_type() const override;
  void configure(const std::string& json_text) override;

 private:
  int n_distances_ = 0;
  std::vector<double> out_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_DISCRETEDISTANCES_H
