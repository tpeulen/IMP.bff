/**
 *  \file IMP/bff/internal/FRETNetworkFilter.h
 *  \brief The operator and segment pass of a FRET network measurement,
 *         shared by the filter and the fit (src/FRETNetworkFilter.cpp).
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_INTERNAL_FRETNETWORKFILTER_H
#define IMPBFF_INTERNAL_FRETNETWORKFILTER_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/FRETNetwork.h>

#include <vector>

IMPBFF_BEGIN_NAMESPACE

namespace fret_network_detail {

struct NetOp {
  int n = 0, C = 0, nb = 1;
  double q = 1.0;
  // B = I + A/q as CSR by target (for B v) and by source (for B^T v)
  std::vector<int> fp, fi, bp, bi;
  std::vector<double> fv, bv;
  std::vector<double> av;  // A itself on the pattern (aligned with fv)
  // nonzero pattern of A, [t, s] pairs in CSR-by-target order (matches fv)
  std::vector<double> factor;  // [(c*nb + b)*n + s]
  std::vector<double> ltot, start, pi, emission;
  bool conditional = false, detection = true;
};

//! What one segment pass collects beyond the log-likelihood.
struct SegmentOut {
  bool want_adjoint = false, want_occupancy = false, want_posteriors = false;
  // adjoint accumulators
  std::vector<double> gA;       // per nnz of B, dlogL/dA_ts
  std::vector<double> gfactor;  // [(c*nb+b)*n + s]
  std::vector<int> rows;        // the (c*nb+b) rows gfactor touches
  std::vector<double> gstart;   // n
  // outputs
  std::vector<double> occupancy;   // n, time fractions
  std::vector<double> posteriors;  // N x n
};

NetOp build_operator(const FRETHiddenProcess& process, const FRETMeasurement& m,
                     bool conditional, bool detection, bool joint_start);
double segment_pass(const NetOp& op, const FRETPhotonData& data, int seg, SegmentOut* out);

}  // namespace fret_network_detail

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_INTERNAL_FRETNETWORKFILTER_H
