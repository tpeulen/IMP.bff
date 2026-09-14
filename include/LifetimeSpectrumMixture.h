/**
 *  \file IMP/bff/LifetimeSpectrumMixture.h
 *  \brief Lifetime spectra of several species, weighted by fractions and joined.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_LIFETIMESPECTRUMMIXTURE_H
#define IMPBFF_LIFETIMESPECTRUMMIXTURE_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/GraphNode.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A mixture of lifetime spectra, as a node.
/*!
    A lifetime spectrum here is the interleaved `(a0, tau0, a1, tau1, ...)`
    vector TCSPCDecay reads -- not an excitation or emission spectrum. Input
    ports `s0 ... s{K-1}` carry the lifetime spectra of K species and scalar
    ports `x0 ... x{K-1}` their fractions. The output, keyed by the node's own
    name, is the lifetime spectra joined in order, with the amplitudes of
    species `k` multiplied by `|x_k| / sum_j |x_j|`.

    Settings (#configure): `number_of_species` (K), `normalize_amplitudes`
    (default true: each species' amplitudes are first divided by the sum of
    their magnitudes, so a fraction is the share of that species whatever
    scale its source writes the amplitudes in).

    What it is for is a mixture of models: a sample that is several species,
    each described by a model of its own. What gets mixed is their lifetimes,
    weighted, and the one decay built from the joined lifetime spectrum is the
    mixture's.
*/
class IMPBFFEXPORT LifetimeSpectrumMixture : public GraphNode {
 public:
  explicit LifetimeSpectrumMixture(const std::string& name = "lifetime_spectrum_mixture");

  void set_number_of_species(int n);
  int get_number_of_species() const { return n_species_; }
  void set_normalize_amplitudes(bool v);
  bool get_normalize_amplitudes() const { return normalize_amplitudes_; }

  void evaluate() override;
  std::string get_node_type() const override;
  void configure(const std::string& json_text) override;

 private:
  int n_species_ = 0;
  bool normalize_amplitudes_ = true;
  std::vector<double> out_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_LIFETIMESPECTRUMMIXTURE_H
