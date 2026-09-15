/**
 *  \file IMP/bff/AcceptorDensityDecay.h
 *  \brief The donor decay quenched by an acceptor density, as a graph node.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_ACCEPTORDENSITYDECAY_H
#define IMPBFF_ACCEPTORDENSITYDECAY_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/GraphNode.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! #acceptor_density_decay (FRETAcceptorDensity.h) on a sampled time axis.
/*!
    Input ports: `lifetime_spectrum` (the donor-only interleaved spectrum),
    scalars `c_over_c0` and `tau0`. Settings (#configure): `dimension` (1, 2, 3),
    `n_points`, `dt`, `period` (0 for single excitation), `folded_periods`
    (default 8). Writes the unconvolved decay to the output keyed by its name;
    a Convolution node reconvolves it and a TCSPCDecay with `curve_from_port`
    adds the instrument. Scalar outputs named `efficiency` (the transfer
    efficiency the density implies) and `acceptor_density` (C/C0 times C0 for
    the scalar input `forster_radius`, in Å^-d) are written where they exist.
*/
class IMPBFFEXPORT AcceptorDensityDecay : public GraphNode {
 public:
  explicit AcceptorDensityDecay(const std::string& name = "quenched");

  void evaluate() override;
  std::string get_node_type() const override;
  void configure(const std::string& json_text) override;

 private:
  int dimension_ = 3;
  int n_points_ = 0;
  double dt_ = 1.0;
  double period_ = 0.0;
  int folded_periods_ = 8;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_ACCEPTORDENSITYDECAY_H
