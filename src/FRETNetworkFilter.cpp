// SPDX-License-Identifier: BSD-3-Clause
/**
 * The photon-by-photon filter of a FRET network measurement: segments,
 * arrival models, uniformization between photons, per-segment likelihoods,
 * posterior occupancies and (for the gradient) the adjoint. See
 * FRETNetwork.h.
 *
 * Between photons the forward vector is propagated by uniformization,
 *
 *     exp(A tau) v = sum_k Pois(k; q tau) B^k v,   B = I + A/q,
 *
 * with `q` the largest exit-plus-killing rate, so `B` is nonnegative and no
 * term cancels. The adjoint of one chunk with left vector `beta` is exact on
 * the sparsity pattern of `A`:
 *
 *     d(beta^T exp(A tau) v)/dA_ts = (1/q) sum_j c_j[t] u_j[s],
 *     u_j = B^j v,   c_j = sum_{m>=0} Pois(j+1+m; q tau) (B^T)^m beta,
 *
 * and the time integral of the posterior over a chunk is
 * `(1/q) sum_m Pois(m+1; q tau) sum_{j+k=m} u_j o (B^T)^k beta`.
 */
#include <IMP/bff/FRETNetwork.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

IMPBFF_BEGIN_NAMESPACE

// --- photon data ----------------------------------------------------------------------

FRETPhotonData::FRETPhotonData(const std::vector<double>& macrotimes,
                               const std::vector<int>& channels,
                               const std::vector<int>& microtimes,
                               const std::vector<int>& segment_starts,
                               const std::vector<int>& segment_stops)
    : t_(macrotimes), c_(channels), b_(microtimes), start_(segment_starts),
      stop_(segment_stops) {
  const int n = static_cast<int>(t_.size());
  if (static_cast<int>(c_.size()) != n || (!b_.empty() && static_cast<int>(b_.size()) != n))
    IMP_THROW("FRETPhotonData: macrotimes, channels and microtimes differ in length",
              IMP::ValueException);
  if (start_.size() != stop_.size())
    IMP_THROW("FRETPhotonData: segment starts and stops differ in length", IMP::ValueException);
  for (std::size_t k = 0; k < start_.size(); ++k) {
    if (start_[k] < 0 || stop_[k] >= n || stop_[k] < start_[k])
      IMP_THROW("FRETPhotonData: segment " << k << " is not within the photons",
                IMP::ValueException);
    for (int i = start_[k] + 1; i <= stop_[k]; ++i)
      if (!(t_[i] >= t_[i - 1]))
        IMP_THROW("FRETPhotonData: macrotimes decrease in segment " << k, IMP::ValueException);
  }
}

double FRETPhotonData::get_max_gap() const {
  double g = 0.0;
  for (std::size_t k = 0; k < start_.size(); ++k)
    for (int i = start_[k] + 1; i <= stop_[k]; ++i) g = std::max(g, t_[i] - t_[i - 1]);
  return g;
}

// --- the operator of one measurement ---------------------------------------------------------

namespace fret_network_detail {

struct NetOp {
  int n = 0, C = 0, nb = 1;
  double q = 1.0;
  // B = I + A/q as CSR by target (for B v) and by source (for B^T v)
  std::vector<int> fp, fi, bp, bi;
  std::vector<double> fv, bv;
  // nonzero pattern of A, [t, s] pairs in CSR-by-target order (matches fv)
  std::vector<double> factor;  // [(c*nb + b)*n + s]
  std::vector<double> ltot, start, pi, emission;
  bool conditional = false, detection = true;
};

NetOp build_operator(const FRETHiddenProcess& process, const FRETMeasurement& m, bool conditional,
                     bool detection, bool joint_start) {
  NetOp op;
  const std::vector<double> Q = m.get_generator(process);
  op.n = m.get_n_states(process);
  const int n = op.n;
  op.emission = m.get_emission(process);
  op.C = m.get_instrument().get_n_channels();
  op.nb = m.get_instrument().get_n_bins();
  op.ltot.assign(n, 0.0);
  for (int c = 0; c < op.C; ++c)
    for (int b = 0; b < op.nb; ++b)
      for (int s = 0; s < n; ++s)
        op.ltot[s] += op.emission[(static_cast<std::size_t>(c) * op.nb + b) * n + s];
  op.conditional = conditional;
  op.detection = detection;
  // A = Q - diag(Lambda) (full) or Q (conditional)
  std::vector<double> diag(n);
  double q = 0.0;
  for (int s = 0; s < n; ++s) {
    diag[s] = Q[static_cast<std::size_t>(s) * n + s] - (conditional ? 0.0 : op.ltot[s]);
    q = std::max(q, -diag[s]);
  }
  op.q = q > 0.0 ? q : 1.0;
  op.fp.assign(n + 1, 0);
  for (int t = 0; t < n; ++t) {
    for (int s = 0; s < n; ++s) {
      const double a = t == s ? diag[s] : Q[static_cast<std::size_t>(t) * n + s];
      const double b = (t == s ? 1.0 : 0.0) + a / op.q;
      if (t != s && a == 0.0) continue;
      op.fi.push_back(s);
      op.fv.push_back(b);
    }
    op.fp[t + 1] = static_cast<int>(op.fi.size());
  }
  // transpose
  op.bp.assign(n + 1, 0);
  for (int k : op.fi) ++op.bp[k + 1];
  for (int s = 0; s < n; ++s) op.bp[s + 1] += op.bp[s];
  op.bi.resize(op.fi.size());
  op.bv.resize(op.fv.size());
  std::vector<int> pos(op.bp.begin(), op.bp.end() - 1);
  for (int t = 0; t < n; ++t)
    for (int k = op.fp[t]; k < op.fp[t + 1]; ++k) {
      const int s = op.fi[k];
      op.bi[pos[s]] = t;
      op.bv[pos[s]++] = op.fv[k];
    }
  // photon factors
  op.factor = op.emission;
  if (conditional)
    for (std::size_t i = 0; i < op.factor.size(); ++i) {
      const int s = static_cast<int>(i % n);
      op.factor[i] = op.ltot[s] > 0.0 ? op.factor[i] / op.ltot[s] : 0.0;
    }
  // start
  op.pi = joint_start ? m.get_joint_stationary(process) : m.get_start(process);
  op.start = op.pi;
  if (detection) {
    double z = 0.0;
    for (int s = 0; s < n; ++s) z += op.pi[s] * op.ltot[s];
    if (!(z > 0.0))
      IMP_THROW("FRETNetworkModel: no state of the start distribution emits", IMP::ValueException);
    // conditional: pi Lambda / <pi, Lambda>; full: pi / <pi, Lambda> (the
    // photon factor then carries Lambda)
    for (int s = 0; s < n; ++s) op.start[s] = op.pi[s] * (conditional ? op.ltot[s] : 1.0) / z;
  }
  return op;
}

inline void apply_b(const NetOp& op, const double* v, double* out) {
  for (int t = 0; t < op.n; ++t) {
    double x = 0.0;
    for (int k = op.fp[t]; k < op.fp[t + 1]; ++k) x += op.fv[k] * v[op.fi[k]];
    out[t] = x;
  }
}

inline void apply_bt(const NetOp& op, const double* v, double* out) {
  for (int s = 0; s < op.n; ++s) {
    double x = 0.0;
    for (int k = op.bp[s]; k < op.bp[s + 1]; ++k) x += op.bv[k] * v[op.bi[k]];
    out[s] = x;
  }
}

//! Poisson weights of mean mu up to the 1e-14 tail.
void poisson_weights(double mu, std::vector<double>& p) {
  p.clear();
  double w = std::exp(-mu), cum = w;
  p.push_back(w);
  for (int k = 1; k < 100000; ++k) {
    w *= mu / k;
    p.push_back(w);
    cum += w;
    if (k > mu && (1.0 - cum < 1e-14 || w < 1e-17)) break;
  }
  p.push_back(0.0);  // p[K+1] for the adjoint recursion's upper end
}

const double kChunk = 64.0;

inline int n_chunks(const NetOp& op, double tau) {
  return std::max(1, static_cast<int>(std::ceil(op.q * tau / kChunk)));
}

//! v <- exp(A tau_c) v for one chunk; returns the log of the renormalisation.
double propagate_chunk(const NetOp& op, double mu, std::vector<double>& v,
                       std::vector<double>& w, std::vector<double>& tmp,
                       std::vector<double>& p) {
  poisson_weights(mu, p);
  const int n = op.n;
  std::vector<double> acc(n);
  w = v;
  for (int s = 0; s < n; ++s) acc[s] = p[0] * w[s];
  for (std::size_t k = 1; k + 1 < p.size(); ++k) {
    apply_b(op, w.data(), tmp.data());
    std::swap(w, tmp);
    for (int s = 0; s < n; ++s) acc[s] += p[k] * w[s];
  }
  double z = 0.0;
  for (double x : acc) z += x;
  if (!(z > 0.0)) {
    v = acc;
    return -std::numeric_limits<double>::infinity();
  }
  for (int s = 0; s < n; ++s) v[s] = acc[s] / z;
  return std::log(z);
}

//! What one segment pass collects beyond the log-likelihood.
struct SegmentOut {
  bool want_adjoint = false, want_occupancy = false, want_posteriors = false;
  // adjoint accumulators
  std::vector<double> gA;       // per nnz of B, dlogL/dA_ts
  std::vector<double> gfactor;  // [(c*nb+b)*n + s]
  std::vector<double> gstart;   // n
  // outputs
  std::vector<double> occupancy;   // n, time fractions
  std::vector<double> posteriors;  // N x n
};

//! Log-likelihood of one segment; fills `out` as asked.
double segment_pass(const NetOp& op, const FRETPhotonData& data, int seg, SegmentOut* out) {
  const int n = op.n, a = data.get_segment_starts()[seg], b = data.get_segment_stops()[seg];
  const int N = b - a + 1;
  const std::vector<double>& T = data.get_macrotimes();
  const std::vector<int>& CH = data.get_channels();
  const std::vector<int>& MB = data.get_microtimes();
  auto fac = [&](int i) {
    const int c = CH[i];
    const int bin = op.nb > 1 ? (MB.empty() ? 0 : MB[i]) : 0;
    if (c < 0 || c >= op.C || bin < 0 || bin >= op.nb)
      IMP_THROW("FRETNetworkModel: photon " << i << " has channel " << c << ", bin " << bin
                                            << " outside the instrument",
                IMP::ValueException);
    return op.factor.data() + (static_cast<std::size_t>(c) * op.nb + bin) * n;
  };
  const bool back = out && (out->want_adjoint || out->want_occupancy || out->want_posteriors);
  std::vector<std::vector<double> > fwd;  // normalised alpha after each photon
  if (back) fwd.resize(N);
  std::vector<double> v(n), w(n), tmp(n), p;
  double logl = 0.0;
  {
    const double* f = fac(a);
    double z = 0.0;
    for (int s = 0; s < n; ++s) z += (v[s] = op.start[s] * f[s]);
    if (!(z > 0.0)) return -std::numeric_limits<double>::infinity();
    logl += std::log(z);
    for (double& x : v) x /= z;
    if (back) fwd[0] = v;
  }
  for (int i = 1; i < N; ++i) {
    const double tau = T[a + i] - T[a + i - 1];
    if (tau > 0.0) {
      const int nc = n_chunks(op, tau);
      const double mu = op.q * tau / nc;
      for (int c = 0; c < nc; ++c) logl += propagate_chunk(op, mu, v, w, tmp, p);
    }
    const double* f = fac(a + i);
    double z = 0.0;
    for (int s = 0; s < n; ++s) z += (v[s] *= f[s]);
    if (!(z > 0.0) || !std::isfinite(logl)) return -std::numeric_limits<double>::infinity();
    logl += std::log(z);
    for (double& x : v) x /= z;
    if (back) fwd[i] = v;
  }
  if (!back) return logl;

  // --- backward sweep -----------------------------------------------------------------
  if (out->want_adjoint) {
    out->gA.assign(op.fv.size(), 0.0);
    out->gfactor.assign(op.factor.size(), 0.0);
    out->gstart.assign(n, 0.0);
  }
  if (out->want_occupancy) out->occupancy.assign(n, 0.0);
  if (out->want_posteriors) out->posteriors.assign(static_cast<std::size_t>(N) * n, 0.0);
  std::vector<double> beta(n, 1.0);  // left vector after photon i
  std::vector<std::vector<double> > U, Wb;
  std::vector<double> pre(n);
  for (int i = N - 1; i >= 0; --i) {
    const double* f = fac(a + i);
    // posterior at photon i
    if (out->want_posteriors || (out->want_occupancy && N == 1)) {
      double z = 0.0;
      for (int s = 0; s < n; ++s) z += fwd[i][s] * beta[s];
      for (int s = 0; s < n; ++s) {
        const double g = fwd[i][s] * beta[s] / z;
        if (out->want_posteriors) out->posteriors[static_cast<std::size_t>(i) * n + s] = g;
        if (N == 1 && out->want_occupancy) out->occupancy[s] = g;
      }
    }
    // the forward vector just before photon i's factor
    std::vector<std::vector<double> > starts;  // chunk starts within the gap
    double tau = 0.0;
    int nc = 0;
    double mu = 0.0;
    if (i == 0) {
      pre = op.start;
    } else {
      tau = T[a + i] - T[a + i - 1];
      pre = fwd[i - 1];
      if (tau > 0.0) {
        nc = n_chunks(op, tau);
        mu = op.q * tau / nc;
        for (int c = 0; c < nc; ++c) {
          starts.push_back(pre);
          propagate_chunk(op, mu, pre, w, tmp, p);
        }
      }
    }
    // photon factor: dlogL/dfactor = pre o beta / <pre o f, beta>
    double z = 0.0;
    for (int s = 0; s < n; ++s) z += pre[s] * f[s] * beta[s];
    if (out->want_adjoint) {
      const int c = CH[a + i];
      const int bin = op.nb > 1 ? (MB.empty() ? 0 : MB[a + i]) : 0;
      double* gf = out->gfactor.data() + (static_cast<std::size_t>(c) * op.nb + bin) * n;
      for (int s = 0; s < n; ++s) gf[s] += pre[s] * beta[s] / z;
      if (i == 0)
        for (int s = 0; s < n; ++s) out->gstart[s] += f[s] * beta[s] / z;
    }
    // left vector before the factor, then back through the chunks
    std::vector<double> bl(n);
    double bn = 0.0;
    for (int s = 0; s < n; ++s) bn = std::max(bn, bl[s] = f[s] * beta[s]);
    for (double& x : bl) x /= (bn > 0.0 ? bn : 1.0);
    for (int c = nc - 1; c >= 0; --c) {
      // chunk c: start alpha = starts[c], end beta = bl
      poisson_weights(mu, p);
      const int K = static_cast<int>(p.size()) - 2;  // p[0..K], p[K+1] = 0
      U.assign(K + 1, std::vector<double>(n));
      U[0] = starts[c];
      for (int k = 1; k <= K; ++k) apply_b(op, U[k - 1].data(), U[k].data());
      double zc = 0.0;
      for (int k = 0; k <= K; ++k)
        for (int s = 0; s < n; ++s) zc += p[k] * U[k][s] * bl[s];
      if (out->want_adjoint) {
        // c_j = p[j+1] beta + B^T c_{j+1}, from j = K down
        std::vector<double> cj(n, 0.0), nxt(n);
        for (int j = K; j >= 0; --j) {
          if (j < K) {
            apply_bt(op, cj.data(), nxt.data());
            cj.swap(nxt);
          }
          for (int s = 0; s < n; ++s) cj[s] += p[j + 1] * bl[s];
          // gA_ts += c_j[t] u_j[s] / (q zc) over the pattern
          for (int t = 0; t < n; ++t) {
            const double ct = cj[t];
            if (ct == 0.0) continue;
            for (int k = op.fp[t]; k < op.fp[t + 1]; ++k)
              out->gA[k] += ct * U[j][op.fi[k]] / (op.q * zc);
          }
        }
      }
      if (out->want_occupancy) {
        Wb.assign(K + 1, std::vector<double>(n));
        Wb[0] = bl;
        for (int k = 1; k <= K; ++k) apply_bt(op, Wb[k - 1].data(), Wb[k].data());
        for (int m = 0; m <= K; ++m) {
          const double pm = p[m + 1];
          if (pm == 0.0) continue;
          for (int j = 0; j <= m; ++j)
            for (int s = 0; s < n; ++s)
              out->occupancy[s] += pm * U[j][s] * Wb[m - j][s] / (op.q * zc);
        }
      }
      // bl <- exp(A tau_c)^T bl, renormalised
      std::vector<double> acc(n, 0.0), cur(bl);
      for (int s = 0; s < n; ++s) acc[s] = p[0] * cur[s];
      for (int k = 1; k <= K; ++k) {
        apply_bt(op, cur.data(), tmp.data());
        cur.swap(tmp);
        for (int s = 0; s < n; ++s) acc[s] += p[k] * cur[s];
      }
      double mx = 0.0;
      for (double x : acc) mx = std::max(mx, x);
      for (int s = 0; s < n; ++s) bl[s] = acc[s] / (mx > 0.0 ? mx : 1.0);
    }
    beta = bl;
  }
  if (out->want_occupancy && N > 1) {
    const double dur = T[b] - T[a];
    if (dur > 0.0)
      for (double& x : out->occupancy) x /= dur;
  }
  return logl;
}

}  // namespace fret_network_detail

// --- network model ------------------------------------------------------------------------

FRETNetworkModel::FRETNetworkModel(const FRETHiddenProcess& process) : process_(process) {}

int FRETNetworkModel::add_measurement(const FRETMeasurement& measurement,
                                      const FRETPhotonData& data) {
  measurement.get_n_states(process_);
  measurement.get_generator(process_);  // validates the pair against the process
  meas_.push_back(measurement);
  data_.push_back(data);
  arrival_.push_back(FRET_ARRIVAL_CONDITIONAL);
  detection_start_.push_back(1);
  joint_start_.push_back(0);
  return static_cast<int>(meas_.size()) - 1;
}

void FRETNetworkModel::set_arrival_model(int i, int model) {
  if (model != FRET_ARRIVAL_FULL && model != FRET_ARRIVAL_CONDITIONAL)
    IMP_THROW("set_arrival_model: FRET_ARRIVAL_FULL or FRET_ARRIVAL_CONDITIONAL",
              IMP::ValueException);
  arrival_.at(i) = model;
}


std::vector<double> FRETNetworkModel::segment_log_likelihoods(int i) const {
  const fret_network_detail::NetOp op = fret_network_detail::build_operator(
      process_, meas_.at(i), arrival_[i] == FRET_ARRIVAL_CONDITIONAL, detection_start_[i],
      joint_start_[i]);
  const FRETPhotonData& d = data_[i];
  std::vector<double> out(d.get_n_segments());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int k = 0; k < d.get_n_segments(); ++k)
    out[k] = fret_network_detail::segment_pass(op, d, k, nullptr);
  return out;
}

double FRETNetworkModel::log_likelihood() const {
  double total = 0.0;
  for (int i = 0; i < get_n_measurements(); ++i)
    for (double v : segment_log_likelihoods(i)) total += v;
  return total;
}

std::vector<double> FRETNetworkModel::segment_occupancies(int i, const std::string& factor) const {
  const fret_network_detail::NetOp op = fret_network_detail::build_operator(
      process_, meas_.at(i), arrival_[i] == FRET_ARRIVAL_CONDITIONAL, detection_start_[i],
      joint_start_[i]);
  const FRETMeasurement& m = meas_[i];
  const int nh = process_.get_n_states(), nd = m.get_donor().get_n_states(),
            na = m.get_acceptor().get_n_states();
  int nf, which;
  if (factor == "hidden") { nf = nh; which = 0; }
  else if (factor == "donor") { nf = nd; which = 1; }
  else if (factor == "acceptor") { nf = na; which = 2; }
  else if (factor == "joint") { nf = op.n; which = 3; }
  else IMP_THROW("segment_occupancies: factor is hidden, donor, acceptor or joint",
                 IMP::ValueException);
  const FRETPhotonData& d = data_[i];
  const int ns = d.get_n_segments();
  std::vector<double> out(static_cast<std::size_t>(ns) * nf, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int k = 0; k < ns; ++k) {
    fret_network_detail::SegmentOut so;
    so.want_occupancy = true;
    fret_network_detail::segment_pass(op, d, k, &so);
    for (int s = 0; s < op.n; ++s) {
      const int h = s / (nd * na), dd = (s / na) % nd, aa = s % na;
      const int col = which == 0 ? h : which == 1 ? dd : which == 2 ? aa : s;
      out[static_cast<std::size_t>(k) * nf + col] += so.occupancy.empty() ? 0.0 : so.occupancy[s];
    }
  }
  return out;
}

std::vector<double> FRETNetworkModel::photon_posteriors(int i, int segment) const {
  const fret_network_detail::NetOp op = fret_network_detail::build_operator(
      process_, meas_.at(i), arrival_[i] == FRET_ARRIVAL_CONDITIONAL, detection_start_[i],
      joint_start_[i]);
  if (segment < 0 || segment >= data_[i].get_n_segments())
    IMP_THROW("photon_posteriors: no segment " << segment, IMP::ValueException);
  fret_network_detail::SegmentOut so;
  so.want_posteriors = true;
  fret_network_detail::segment_pass(op, data_[i], segment, &so);
  return so.posteriors;
}

IMPBFF_END_NAMESPACE
