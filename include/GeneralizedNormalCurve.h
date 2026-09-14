/**
 * \file IMP/bff/GeneralizedNormalCurve.h
 * \brief A sampled generalized-normal peak, as a node.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_GENERALIZEDNORMALCURVE_H
#define IMPBFF_GENERALIZEDNORMALCURVE_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/GraphNode.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A generalized-normal peak sampled on an even axis, scaled to an area.
/*!
    `area * p(x) / sum p`, with `p` the skewed normal density of
    `generalized_normal_distribution` (a normal for `shape == 0`) at
    `x_i = i * dt`, and values under `floor` set to zero. Scalar input ports:
    `loc`, `scale` (its magnitude is used, floored at machine epsilon) and
    `shape`. Settings (#configure): `n_points`, `dt`, `area`, `floor`
    (default 0), `normalize` (default true). Writes to the output keyed by
    its own name.

    What it is for is a response nobody measured: ChiSurf models a missing
    IRF as exactly this peak at the decay's rise, and its width and shape can
    then be fitted like any other parameter.
*/
class IMPBFFEXPORT GeneralizedNormalCurve : public GraphNode {
 public:
  explicit GeneralizedNormalCurve(const std::string& name = "generalized_normal");

  void set_sampling(int n_points, double dt);
  void set_area(double area);
  void set_floor(double floor);
  void set_normalize(bool v);

  void evaluate() override;
  std::string get_node_type() const override;
  void configure(const std::string& json_text) override;

 private:
  double scalar_input(const char* key, double fallback) const;
  int n_points_ = 0;
  double dt_ = 1.0;
  double area_ = 1.0;
  double floor_ = 0.0;
  bool normalize_ = true;
  std::vector<double> out_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_GENERALIZEDNORMALCURVE_H
