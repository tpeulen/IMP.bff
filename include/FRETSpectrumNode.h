/**\file IMP/bff/FRETSpectrumNode.h
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_FRETSPECTRUMNODE_H
#define IMPBFF_FRETSPECTRUMNODE_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/GraphNode.h>
#include <IMP/bff/GraphPort.h>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The lifetime spectrum of a donor quenched by FRET over a distance
//! distribution.
/**
 * The piece that makes a FRET decay a graph. It takes two spectra and
 * returns one:
 *
 * * the **donor** lifetime spectrum -- what the donor would do with no
 *   acceptor present -- from upstream;
 * * a **distance distribution**, weight and distance interleaved, from
 *   whichever producer the model uses;
 *
 * and returns the lifetime spectrum of the mixture, ready for the
 * instrument node.
 *
 * \par The arithmetic, and why it is a spectrum transform
 * A distance becomes a transfer rate,
 * $k_{FRET} = rac{3}{2}\kappa^2 	au_0^{-1} (R_0/r)^6$, and a
 * quenched species decays at the *sum* of its own rate and the transfer
 * rate. Rates add, so the quenched spectrum is the Cartesian product of the
 * donor's rates and the transfer rates with the amplitudes multiplied --
 * which is exactly why this can be done on spectra rather than on curves.
 *
 * \par Donor-only is a mixture, not a term
 * A sample never labels perfectly. The fraction `x_donly` that carries no
 * acceptor decays at the *donor's* rates, so it enters as its own set of
 * components appended to the quenched ones, scaled by `x_donly` while the
 * quenched ones are scaled by `1 - x_donly`. They are appended
 * unconditionally, including at `x_donly = 0` where they carry zero
 * amplitude: the length of a spectrum is observable, and a node that
 * sometimes returns a shorter one is a second code path for no gain.
 *
 * Ports:
 *
 * | port | what it is |
 * |---|---|
 * | `donor_lifetime_spectrum` | interleaved `(a, tau)`, from upstream |
 * | `distance_distribution` | interleaved `(p, r)`, from upstream |
 * | `x_donly` | fraction of the sample carrying no acceptor |
 * | `forster_radius` | $R_0$, in the units of the distances |
 * | `tau0` | the donor lifetime $R_0$ was determined at |
 * | `kappa2` | the orientation factor |
 *
 * \par R0 and tau0 are a pair
 * The transfer rate uses the node's own `tau0` -- the donor lifetime at which
 * `forster_radius` was determined -- for every donor component, never each
 * component's lifetime: R0^6 scales with the donor quantum yield and Q_D/tau_D
 * is the radiative rate, so one (R0, tau0) pair fixes k_FRET at a distance.
 * Pass the two together. PhotophysicsTransferKinetics.h takes these rates.
 *
 * \par Orientation
 * `kappa2` is one orientation factor for every molecule -- dipoles that
 * reorient fast against the donor's lifetime (`orientation: "dynamic"`, the
 * default). For dipoles that stay put (`"static"`), each molecule has its own
 * kappa^2 from a distribution, and a distance and an orientation factor make
 * one transfer rate together, `k = 3/2 kappa^2 / tau0 (R0/r)^6`. The
 * distribution arrives on the port `kappa2_distribution` as interleaved
 * `(w, kappa^2)`, or is the isotropic one (`"static_isotropic"`, the density on
 * `kappa2_points` points over `[0.01, 4]`, 128 as ChiSurf samples it). With
 * `kappa2_bins` 0 every pair is its own species; otherwise the pairs are
 * histogrammed by apparent distance `r (<kappa^2>/kappa^2)^(1/6)` into that
 * many linear bins, whose rates use `<kappa^2>` -- the same rate for a pair at
 * a bin centre, and far fewer species for the instrument to reconvolve.
 *
 * Besides the lifetime spectrum (the output keyed by the node's name), an
 * output named `fret_rates`, where one exists, receives the interleaved
 * `(p, k_FRET)` transfer rates per distance, before any donor is combined.
 */
class IMPBFFEXPORT FRETSpectrumNode : public GraphNode {
 public:
  explicit FRETSpectrumNode(const std::string& name = "fret");

  //! Build the node's ports. Cannot be done in the constructor.
  void build_ports();

  static const char* donor_port_key() { return "donor_lifetime_spectrum"; }
  static const char* distance_port_key() { return "distance_distribution"; }

  //! The interleaved `(a, tau)` spectrum from the last evaluation.
  const std::vector<double>& get_spectrum() const { return spectrum_; }

  void evaluate() override;

  std::string describe() const;

  std::string get_node_type() const override;
  //! Settings: `orientation` (`dynamic`, `static`, `static_isotropic`),
  //! `kappa2_bins` (0 exact), `kappa2_points`.
  void configure(const std::string& json_text) override;

  void set_orientation(const std::string& orientation);
  const std::string& get_orientation() const { return orientation_; }
  void set_kappa2_bins(int n);
  int get_kappa2_bins() const { return kappa2_bins_; }

 private:
  //! The transfer rates `(p, k)` of the distances under the orientation.
  std::vector<double> transfer_rates(const std::vector<double>& distances,
                                     double forster_radius, double tau0,
                                     double kappa2) const;
  std::string orientation_ = "dynamic";
  int kappa2_bins_ = 0;
  int kappa2_points_ = 128;
  std::vector<double> spectrum_;
  GraphPort* donor_port_ = nullptr;
  GraphPort* distance_port_ = nullptr;
  GraphPort* x_donly_port_ = nullptr;
  GraphPort* forster_radius_port_ = nullptr;
  GraphPort* tau0_port_ = nullptr;
  GraphPort* kappa2_port_ = nullptr;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_FRETSPECTRUMNODE_H
