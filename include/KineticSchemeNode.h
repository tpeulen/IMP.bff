/**
 * \file IMP/bff/KineticSchemeNode.h
 * \brief A chain of interconverting states, seen by TCSPC and by FCS at once.
 *
 * One kinetic scheme, two measurements. States `0 ... n-1` interconvert along
 * a chain: `i-1 -> i` at `kf{i}` and back at `kb{i}`, for `i = 1 ... n-1`,
 * each rate named after the state it leads into; state `i` fluoresces with
 * lifetime `t{i}`. What each measurement sees of it:
 *
 * - **TCSPC, the populations.** The stationary distribution `p` weights each
 *   state's decay, published as the interleaved lifetime spectrum
 *   `(p0, t0, p1, t1, ...)` a `TCSPCDecay` reads from a port.
 * - **FCS, the relaxations.** A molecule's brightness follows its quantum
 *   yield, so `q_i` is proportional to `t{i}`. Interconversion then modulates
 *   the correlation by `X(tau) = sum_ij q_i p_i P_ij(tau) q_j / (q.p)^2`,
 *   published on the lag axis `x`; `X` tends to 1 at long lags and carries
 *   one relaxation per nonzero eigenvalue of the generator.
 *
 * A chain satisfies detailed balance, so the generator is similar to a
 * symmetric matrix and `X` is one symmetric eigen-decomposition:
 * `X(tau) = sum_k w_k^2 exp(lambda_k tau) / (q.p)^2`. Rates are in the inverse
 * unit of the lag axis. Fitting the two measurements against one objective is
 * what makes the scheme identifiable: TCSPC fixes the populations and
 * lifetimes, FCS the rates.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_KINETICSCHEMENODE_H
#define IMPBFF_KINETICSCHEMENODE_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/GraphNode.h>
#include <IMP/bff/GraphPort.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Populations for TCSPC and a correlation factor for FCS, from one chain.
class IMPBFFEXPORT KineticSchemeNode : public GraphNode {
 public:
  explicit KineticSchemeNode(const std::string& name = "kinetics");

  //! Declare the states; adds inputs `t{i}`, `kf{i}`, `kb{i}` and `x`.
  void set_number_of_states(int n);
  int get_number_of_states() const { return n_states_; }

  //! The stationary populations of the last evaluation.
  const std::vector<double>& get_populations() const { return populations_; }
  //! The relaxation rates (minus the nonzero eigenvalues), slowest first.
  std::vector<double> get_relaxation_rates() const;

  void evaluate() override;
  std::string get_node_type() const override;
  void configure(const std::string& json_text) override;

 private:
  int n_states_ = 0;
  std::vector<double> populations_;
  std::vector<double> eigenvalues_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_KINETICSCHEMENODE_H
