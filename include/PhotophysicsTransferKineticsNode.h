/**
 *  \file IMP/bff/PhotophysicsTransferKineticsNode.h
 *  \brief Transfer kinetics between two chromophores, as a graph node.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_PHOTOPHYSICSTRANSFERKINETICSNODE_H
#define IMPBFF_PHOTOPHYSICSTRANSFERKINETICSNODE_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/GraphNode.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The lifetime spectrum one pulse gives in one channel, under transfer kinetics.
/*!
    Evaluates #transfer_kinetics_spectrum (PhotophysicsTransferKinetics.h) --
    the kinetics is not written here. Chromophores are labelled `A` and `B`.

    Settings (#configure): `pulses` and `channels` (label lists; default one
    each, `"pulse"` and `"channel"`), `pulse` and `channel` (which one this node
    outputs), `mode` (`"exact"` or `"chisurf"`, see #TransferKineticsMode),
    `eps` (the degeneracy threshold).

    Input ports: `spectrum_a`, `spectrum_b` (interleaved lifetime spectra),
    `rates` (interleaved `(w, k)`, e.g. FRETSpectrumNode's `fret_rates`),
    scalars `f_ab`, `f_ba` (the factors on k for each direction), `pure_a`,
    `pure_b` (pure fractions, turned into populations by
    #transfer_populations_from_pure_fractions), and the light path as scalars
    `excitation_<pulse>_<A|B>` and `emission_<A|B>_<channel>` -- the entries of
    the excitation [pulse x chromophore] and emission [chromophore x channel]
    PhotophysicsCrosstalkMatrix the node assembles. The output, keyed by the
    node's name, is the interleaved `(amplitude, lifetime)` spectrum.
*/
class IMPBFFEXPORT PhotophysicsTransferKineticsNode : public GraphNode {
 public:
  explicit PhotophysicsTransferKineticsNode(const std::string& name = "transfer");

  //! Build the ports for the current pulses and channels. Cannot be done in
  //! the constructor; #configure calls it again for its labels.
  void build_ports();

  void evaluate() override;
  std::string get_node_type() const override;
  void configure(const std::string& json_text) override;

 private:
  std::vector<std::string> pulses_{"pulse"};
  std::vector<std::string> channels_{"channel"};
  std::string pulse_ = "pulse";
  std::string channel_ = "channel";
  int mode_ = 0;
  double eps_ = 1e-9;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_PHOTOPHYSICSTRANSFERKINETICSNODE_H
