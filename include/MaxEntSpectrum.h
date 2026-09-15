/**
 *  \file IMP/bff/MaxEntSpectrum.h
 *  \brief Maximum-entropy amplitudes over a decay basis, as a graph node.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_MAXENTSPECTRUM_H
#define IMPBFF_MAXENTSPECTRUM_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/FitDataset.h>
#include <IMP/bff/GraphNode.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! A distribution recovered by maximum entropy from a measured decay.
/*!
    The model part of a MaxEnt TCSPC analysis. Nothing a decay is made of is
    computed here: the columns are a TCSPCDecay's basis (its convolution, its
    response preparation), a distance grid's transfer rates are
    FRETSpectrumNode's, and the entropy-regularised quadratic programme is
    tttrlib's Skilling-Bryan engine (vendored as internal/MaxEntQp.h). What this
    node does is assemble the programme and turn its answer back into a
    spectrum.

    With `A` the design (rows the active channels, columns the grid), `sigma`
    the data's standard deviations and `b` the background and scatter the
    instrument adds (`background + scatter r`, `r` the prepared response),
    the programme is `chi^2(p) = sum ((y - b - A p) / sigma)^2 / M` against the
    Skilling-Bryan entropy relative to a uniform prior, weighted by `nu`.

    Inputs: `basis` (a TCSPCDecay's, bins x species row-major), `spectrum`
    (the spectrum that basis was built over), `grid` (the grid, `(1, x)`
    pairs, for the distribution's axis), `response` (TCSPCDecay's
    `prepared_response`), scalars `scatter`, `background`, `log10_nu`, and
    optionally `prior` (one weight per grid point; uniform when absent or a
    single value). Bound:
    `data` (its mask is the fit window, its variance at the data the weights).

    Settings (#configure): `grouping` -- "lifetime" (one amplitude per
    species) or "fret" (the spectrum is FRETSpectrumNode's: for each distance
    `donor_components` quenched species, then the donor-only ones; one
    amplitude per distance, the donor-only part following their total);
    `donor_components`; `max_iter`, `tol`; `target_chisq` (> 0: find `nu` so
    the reduced chi-square lands there, historic MaxEnt; `log10_nu` then
    seeds the search).

    Outputs: keyed by the node's name, the spectrum the amplitudes make (for a
    TCSPCDecay with n0 = 1 and the same scatter and background); where
    present, `distribution` (`(p, x)` pairs over the grid, p fractions),
    `amplitudes` (the same in counts per unit basis column), `chisq` (the
    reduced chi-square the programme minimised), `chisq_pearson` (the same
    solution weighted by the model instead of the data: never optimised
    against, so it can disagree), `entropy`, `nu` (the one used) and
    `converged` (1 when the stopping test was met).
*/
class IMPBFFEXPORT MaxEntSpectrum : public GraphNode {
 public:
  explicit MaxEntSpectrum(const std::string& name = "maxent");

  void set_grouping(const std::string& grouping, int donor_components = 1);
  std::string get_grouping() const { return fret_ ? "fret" : "lifetime"; }

  void bind_dataset(const std::string& role, const FitDataset& dataset) override;
  void evaluate() override;
  std::string get_node_type() const override;
  void configure(const std::string& json_text) override;

  //! The amplitudes of the last evaluation, one per grid point.
  std::vector<double> get_amplitudes() const { return amplitudes_; }

 private:
  bool fret_ = false;
  int donor_components_ = 1;
  int max_iter_ = 200;
  double tol_ = 1e-4;
  double target_chisq_ = -1.0;
  std::vector<double> data_;
  std::vector<double> sigma_;
  std::vector<bool> active_;
  std::vector<double> amplitudes_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_MAXENTSPECTRUM_H
