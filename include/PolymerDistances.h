/**\file IMP/bff/PolymerDistances.h
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_POLYMERDISTANCES_H
#define IMPBFF_POLYMERDISTANCES_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/GraphNode.h>
#include <IMP/bff/GraphPort.h>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A polymer-chain distance distribution as a producer node.
/**
 * The closed-form polymer models of chisurf's `FRETModel` subclasses, each a
 * distance distribution and nothing else (board `T-20260901-08`): the
 * worm-like chain (with or without dye-linker broadening), the SAW-nu
 * des Cloizeaux form, and the Ising two-state Gaussian chain. One node with
 * a mode rather than one node per model, because after the kernel ports of
 * `T-20260902-01/-14` every mode is a thin dispatch into `PolymerChain.h`
 * -- the cost lives in the kernel, and the kernels are shared with the
 * Python forwarders in `chisurf/core/math/functions/rdf.py`, so the graph
 * path and the numpy path cannot drift apart.
 *
 * Ports, created by set_mode():
 *
 * | mode | ports |
 * |---|---|
 * | `worm_like_chain` | `chain_length`, `persistence_length` |
 * | `worm_like_chain_linker` | the same plus `sigma_linker` |
 * | `saw_nu` | `r_rms`, `nu` |
 * | `ising_chain` | `n_residues`, `b_structured`, `b_unstructured`, `coupling`, `field` |
 *
 * The worm-like chain takes the *contour* and *persistence* lengths as its
 * ports and derives the kernel's dimensionless `kappa = lp / l` itself, the
 * same derivation chisurf's model property makes; `n_residues` is rounded to
 * the nearest integer as chisurf rounds it. The output is interleaved
 * `(p0, r0, p1, r1, ...)`, the layout `FRETSpectrumNode` reads, with the
 * kernel's own normalisation -- the same array the Python property returns.
 */
class IMPBFFEXPORT PolymerDistances : public GraphNode {
 public:
  explicit PolymerDistances(const std::string& name = "distances");

  //! Select the distribution and build its input ports (once).
  void set_mode(const std::string& mode);
  const std::string& get_mode() const { return mode_; }

  //! The distance axis the distribution is evaluated on, in Angstrom.
  void set_axis(const std::vector<double>& axis);
  const std::vector<double>& get_axis() const { return axis_; }
  //! Numpy in, for the axis a caller holds as an array.
  void set_axis_array(double* in_axis, int n_axis);

  //! k points of the Ising inverse transform (set-once configuration).
  void set_n_k(int n_k) { n_k_ = n_k; set_valid(false); }
  int get_n_k() const { return n_k_; }

  //! Normalise the weights to unit sum on the axis (default false).
  /*! Off, the output keeps each kernel's own normalisation, which is what
      ChiSurf's models were fitted against; on, the weights sum to 1 on any
      caller's grid, and #get_weights_jacobian differentiates that. */
  void set_normalize_weights(bool v) { normalize_weights_ = v; set_valid(false); }
  bool get_normalize_weights() const { return normalize_weights_; }

  //! The derivative of the output weights by the mode's parameters.
  /*! Row-major, `n_axis x n_params`: entry `[j * n_params + c]` is
      d p_j / d theta_c in #get_parameter_names order, of exactly what the
      node outputs (the #set_normalize_weights normalisation included), at the
      ports' current values. By central differences with the step
      `relative_step * max(|theta_c|, 1)`: a mode's two to five parameters make
      that cheap, and with the default 1e-5 the difference to half the step is
      below 1e-8 on ChiSurf's distance axis (pinned by the tests). Columns
      whose parameter the weights do not depend on smoothly are 0: the Ising
      residue count (rounded to an integer) and the linker width without the
      linker. */
  std::vector<double> get_weights_jacobian(double relative_step = 1e-5) const;
  //! The parameter names of #get_weights_jacobian's columns, in order.
  std::vector<std::string> get_parameter_names() const;

  //! The interleaved `(p, r)` distribution from the last evaluation.
  const std::vector<double>& get_distribution() const { return spectrum_; }

  void evaluate() override;

  std::string get_node_type() const override;
  //! Settings: `mode`, `n_k`, `normalize_weights`, and the axis as `axis` or `axis_range`
  //! `[min, max, n]` with `axis_scale`.
  void configure(const std::string& json_text) override;

 private:
  std::vector<double> weights_at(const std::vector<double>& parameters) const;
  std::vector<double> current_parameters() const;
  bool normalize_weights_ = false;
  std::string mode_;
  std::vector<double> axis_;
  std::vector<double> spectrum_;
  std::vector<GraphPort*> parameter_ports_;
  int n_k_ = 2000;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_POLYMERDISTANCES_H
