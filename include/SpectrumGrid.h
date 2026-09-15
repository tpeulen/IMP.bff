/**
 *  \file IMP/bff/SpectrumGrid.h
 *  \brief A fixed grid as a spectrum of unit amplitudes.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_SPECTRUMGRID_H
#define IMPBFF_SPECTRUMGRID_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/GraphNode.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! `bins` points from `from` to `to`, as the interleaved pairs `(1, x)`.
/*!
    What an inversion distributes amplitude over: lifetimes for TCSPCDecay's
    basis, or distances for FRETSpectrumNode. Inputs: scalars `from`, `to`,
    `bins` (rounded) -- ports, so a grid follows its parameters without a
    rebuild. Setting (#configure): `spacing`, "linear" (the default) or "log".
    The output is keyed by the node's name.
*/
class IMPBFFEXPORT SpectrumGrid : public GraphNode {
 public:
  explicit SpectrumGrid(const std::string& name = "grid");

  void set_logarithmic(bool v);
  bool get_logarithmic() const { return logarithmic_; }

  //! The grid points for these bounds.
  static std::vector<double> points(double from, double to, int bins, bool logarithmic);

  void evaluate() override;
  std::string get_node_type() const override;
  void configure(const std::string& json_text) override;

 private:
  bool logarithmic_ = false;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_SPECTRUMGRID_H
