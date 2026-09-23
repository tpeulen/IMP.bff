// SPDX-License-Identifier: BSD-3-Clause
/**
 * The photon-by-photon landscape likelihood and its exact gradient
 * (Dingeldein & Covino, arXiv:2608.21061). See FRETLandscape.h.
 *
 * Notation of the kernel. Grid vectors (length M) are in the symmetric
 * coordinates `v = Pi^-1/2 alpha`; eigen-coordinates are `w = Psi^T v`.
 * `s = pi^1/2` is both the start vector (`alpha_0 = pi`) and the end
 * functional (`<1, alpha> = <s, v>`). One photon step is
 *
 *     y = Psi (e o w),  e_k = exp(nu_k tau)     (dark gap, grid)
 *     v = lambda_c o y, Z = <s, v>, v /= Z      (photon, normalise)
 *     w = Psi^T v
 *
 * and `log L = sum log Z`. The backward sweep carries the grid vector `rho`
 * to the left of each photon operator (`rho = s` after the last photon).
 */
#include <IMP/bff/FRETLandscape.h>
#include <IMP/bff/States.h>

#include <Eigen/Dense>

#if IMP_BFF_HAS_TTTRLIB
// The optimiser is tttrlib's header-only L-BFGS, not a copy (owner, 2026-09-23).
#include <tttrlib/i_lbfgs.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

IMPBFF_BEGIN_NAMESPACE

namespace fret_landscape_detail {

using Eigen::MatrixXd;
using Eigen::VectorXd;

//! Everything that depends on theta but not on the photons.
struct Operator {
  int M = 0, C = 0;
  double D = 0.0, o = 0.0;
  VectorXd u, pi, s, ltot, cnb, nu;
  MatrixXd lam;   // M x C
  MatrixXd shape; // M x C, d_c (1-E) + e_c E
  MatrixXd Psi;   // M x M, eigenvectors in columns
  MatrixXd R;     // 1/(nu_k - nu_l), 0 on the diagonal and for close pairs
  std::vector<std::pair<int, int> > close;  // near-degenerate ordered pairs
  std::vector<double> amp, bg;
};

//! Gradient accumulators of one batch (summed over batches).
struct Accum {
  MatrixXd S;          // sum (lhat o e) w^T - lhat (e o w)^T
  VectorXd Sdiag;      // diagonal of W: sum tau lhat_k e_k w_k
  std::vector<double> Sclose;
  MatrixXd glam;       // M x C, photon part of dlogL/dlambda_c
  VectorXd gs;         // dlogL/ds + dlogL/dp0 (grid)
  void init(int M, int C, std::size_t nclose) {
    S = MatrixXd::Zero(M, M);
    Sdiag = VectorXd::Zero(M);
    Sclose.assign(nclose, 0.0);
    glam = MatrixXd::Zero(M, C);
    gs = VectorXd::Zero(M);
  }
  void add(const Accum& o) {
    S += o.S;
    Sdiag += o.Sdiag;
    for (std::size_t i = 0; i < Sclose.size(); ++i) Sclose[i] += o.Sclose[i];
    glam += o.glam;
    gs += o.gs;
  }
};

//! G_kl for a close pair: tau exp((nu_k+nu_l) tau/2) sinh(d)/d, d = (nu_k-nu_l) tau/2.
inline double close_pair_factor(double nuk, double nul, double tau) {
  const double d = 0.5 * (nuk - nul) * tau;
  return tau * std::exp(0.5 * (nuk + nul) * tau) * (1.0 + d * d / 6.0);
}


Operator fret_landscape_operator(const std::vector<double>& theta, int M, int K, int C,
                                 double h, const std::vector<double>& phi,
                                 const std::vector<double>& eff,
                                 const std::vector<double>& dfrac,
                                 const std::vector<double>& afrac, double tau_max) {
  Operator op;
  op.M = M;
  op.C = C;
  op.u = VectorXd::Zero(M);
  for (int i = 0; i < M; ++i) {
    double v = 0.0;
    for (int k = 0; k < K; ++k) v += phi[static_cast<std::size_t>(i) * K + k] * theta[k];
    op.u[i] = v;
  }
  op.D = std::exp(theta[K]);
  op.o = op.D / (h * h);
  op.amp.resize(C);
  op.bg.resize(C);
  for (int c = 0; c < C; ++c) {
    op.amp[c] = std::exp(theta[K + 1 + c]);
    op.bg[c] = std::exp(theta[K + 1 + C + c]);
  }
  const double umin = op.u.minCoeff();
  op.pi = (-(op.u.array() - umin)).exp().matrix();
  op.pi /= op.pi.sum();
  op.s = op.pi.array().sqrt().matrix();
  op.shape.resize(M, C);
  op.lam.resize(M, C);
  for (int c = 0; c < C; ++c)
    for (int i = 0; i < M; ++i) {
      op.shape(i, c) = dfrac[c] * (1.0 - eff[i]) + afrac[c] * eff[i];
      op.lam(i, c) = op.amp[c] * op.shape(i, c) + op.bg[c];
    }
  op.ltot = op.lam.rowwise().sum();
  op.cnb = VectorXd::Zero(M);
  std::vector<double> diag(M), off(std::max(0, M - 1), op.o);
  for (int i = 0; i < M; ++i) {
    double c = 0.0;
    if (i + 1 < M) c += std::exp(-0.5 * (op.u[i + 1] - op.u[i]));
    if (i > 0) c += std::exp(-0.5 * (op.u[i - 1] - op.u[i]));
    op.cnb[i] = c;
    diag[i] = -op.o * c - op.ltot[i];
  }
  const TridiagonalEigenSystem es = symmetric_tridiagonal_eigen(diag, off);
  op.nu = Eigen::Map<const VectorXd>(es.get_values().data(), M);
  op.Psi.resize(M, M);
  const std::vector<double>& vec = es.get_vectors();
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < M; ++k) op.Psi(i, k) = vec[static_cast<std::size_t>(i) * M + k];
  // Pairs whose divided difference would cancel badly get the series form.
  op.R = MatrixXd::Zero(M, M);
  const double tmax = std::max(tau_max, 1e-300);
  for (int k = 0; k < M; ++k)
    for (int l = 0; l < M; ++l) {
      if (k == l) continue;
      const double dn = op.nu[k] - op.nu[l];
      if (std::fabs(dn) * tmax < 1e-5) op.close.push_back(std::make_pair(k, l));
      else op.R(k, l) = 1.0 / dn;
    }
  return op;
}

struct PhotonView {
  const double* t;
  const int* ch;
  const int* off;
};

//! Forward filter (and, with `acc`, the adjoint sweep) over a batch of traces.
/*! Returns the summed log-likelihood; `per_trace` (if given) receives each
    trace's value at its batch index. */
double fret_landscape_batch(const Operator& op, const PhotonView& ph,
                            const int* traces, int B, Accum* acc, double* per_trace) {
  const int M = op.M;
  std::vector<int> len(B), start(B);
  int N = 0;
  for (int b = 0; b < B; ++b) {
    start[b] = ph.off[traces[b]];
    len[b] = ph.off[traces[b] + 1] - start[b];
    N = std::max(N, len[b]);
  }
  std::vector<double> logl(B, 0.0);
  if (N == 0) {
    if (per_trace) std::fill(per_trace, per_trace + B, 0.0);
    return 0.0;
  }
  const bool grad = acc != nullptr;
  MatrixXd W(M, B), E(M, B), X(M, B), Y(M, B), V(M, B), Wn(M, B);
  std::vector<MatrixXd> store;
  MatrixXd vlast;
  if (grad) {
    store.resize(N);
    vlast = MatrixXd::Zero(M, B);
  }
  auto tau_of = [&](int n, int b) { return ph.t[start[b] + n] - ph.t[start[b] + n - 1]; };
  bool bad = false;

  // first photon: no dark gap before it (paper Eq. 14)
  for (int b = 0; b < B; ++b) {
    if (len[b] == 0) { V.col(b) = op.s; continue; }
    const int c = ph.ch[start[b]];
    V.col(b) = op.lam.col(c).cwiseProduct(op.s);
    const double z = op.s.dot(V.col(b));
    if (!(z > 0.0)) bad = true;
    logl[b] += std::log(z);
    V.col(b) /= z;
    if (grad && len[b] == 1) vlast.col(b) = V.col(b);
  }
  W.noalias() = op.Psi.transpose() * V;
  if (grad) store[0] = W;

  for (int n = 1; n < N; ++n) {
    for (int b = 0; b < B; ++b) {
      if (n < len[b]) E.col(b) = (op.nu * tau_of(n, b)).array().exp().matrix();
      else E.col(b).setOnes();
    }
    X = E.cwiseProduct(W);
    Y.noalias() = op.Psi * X;
    for (int b = 0; b < B; ++b) {
      if (n >= len[b]) { V.col(b).setZero(); continue; }
      const int c = ph.ch[start[b] + n];
      V.col(b) = op.lam.col(c).cwiseProduct(Y.col(b));
      const double z = op.s.dot(V.col(b));
      if (!(z > 0.0)) { bad = true; continue; }
      logl[b] += std::log(z);
      V.col(b) /= z;
      if (grad && n == len[b] - 1) vlast.col(b) = V.col(b);
    }
    Wn.noalias() = op.Psi.transpose() * V;
    for (int b = 0; b < B; ++b)
      if (n < len[b]) W.col(b) = Wn.col(b);
    if (grad) store[n] = W;
  }
  double total = 0.0;
  for (int b = 0; b < B; ++b) {
    if (bad) logl[b] = -std::numeric_limits<double>::infinity();
    total += logl[b];
    if (per_trace) per_trace[b] = logl[b];
  }
  if (!grad || bad) return total;

  // --- adjoint sweep ----------------------------------------------------------
  Accum& a = *acc;
  MatrixXd rho = MatrixXd::Zero(M, B), Sig(M, B), L(M, B), P(M, B);
  MatrixXd Lcat(M, 2 * B), Rcat(M, 2 * B);
  std::vector<double> tau(B, 0.0);
  for (int n = N - 1; n >= 1; --n) {
    for (int b = 0; b < B; ++b) {
      if (n == len[b] - 1) {
        rho.col(b) = op.s;
        a.gs += vlast.col(b);
      }
      if (n < len[b]) {
        tau[b] = tau_of(n, b);
        E.col(b) = (op.nu * tau[b]).array().exp().matrix();
      } else {
        tau[b] = 0.0;
        E.col(b).setOnes();
      }
    }
    const MatrixXd& Wp = store[n - 1];
    X = E.cwiseProduct(Wp);
    Y.noalias() = op.Psi * X;  // the forward vector just before photon n
    for (int b = 0; b < B; ++b) {
      if (n >= len[b]) { Sig.col(b).setZero(); continue; }
      const int c = ph.ch[start[b] + n];
      const double norm = rho.col(b).cwiseProduct(op.lam.col(c)).dot(Y.col(b));
      a.glam.col(c) += rho.col(b).cwiseProduct(Y.col(b)) / norm;
      Sig.col(b) = op.lam.col(c).cwiseProduct(rho.col(b)) / norm;
    }
    L.noalias() = op.Psi.transpose() * Sig;  // lhat: left vector of the gap, eigen coords
    Lcat.leftCols(B) = L.cwiseProduct(E);
    Lcat.rightCols(B) = L;
    Rcat.leftCols(B) = Wp;
    Rcat.rightCols(B) = -X;
    a.S.noalias() += Lcat * Rcat.transpose();
    for (int b = 0; b < B; ++b)
      if (tau[b] > 0.0) a.Sdiag += tau[b] * L.col(b).cwiseProduct(X.col(b));
    for (std::size_t p = 0; p < op.close.size(); ++p) {
      const int k = op.close[p].first, l = op.close[p].second;
      double v = 0.0;
      for (int b = 0; b < B; ++b)
        if (tau[b] > 0.0) v += L(k, b) * Wp(l, b) * close_pair_factor(op.nu[k], op.nu[l], tau[b]);
      a.Sclose[p] += v;
    }
    // rho for photon n-1: Psi (e o l), rescaled (the norms are scale-free)
    P.noalias() = op.Psi * E.cwiseProduct(L);
    for (int b = 0; b < B; ++b) {
      if (n >= len[b]) continue;
      const double mx = P.col(b).cwiseAbs().maxCoeff();
      rho.col(b) = P.col(b) / (mx > 0.0 ? mx : 1.0);
    }
  }
  // first photon of every trace
  for (int b = 0; b < B; ++b) {
    if (len[b] == 0) continue;
    if (len[b] == 1) {
      rho.col(b) = op.s;
      a.gs += vlast.col(b);
    }
    const int c = ph.ch[start[b]];
    const double norm = rho.col(b).cwiseProduct(op.lam.col(c)).dot(op.s);
    a.glam.col(c) += rho.col(b).cwiseProduct(op.s) / norm;
    a.gs += op.lam.col(c).cwiseProduct(rho.col(b)) / norm;
  }
  return total;
}

//! dlogL/dtheta from the accumulators.
std::vector<double> fret_landscape_chain(const Operator& op, const Accum& a, int K,
                                         const std::vector<double>& phi) {
  const int M = op.M, C = op.C;
  MatrixXd Wf = MatrixXd::Zero(M, M);
  for (int k = 0; k < M; ++k)
    for (int l = 0; l < M; ++l)
      if (k != l) Wf(k, l) = op.R(k, l) * a.S(k, l);
  for (std::size_t p = 0; p < op.close.size(); ++p)
    Wf(op.close[p].first, op.close[p].second) = a.Sclose[p];
  Wf.diagonal() = a.Sdiag;
  const MatrixXd T = op.Psi * Wf;  // dlogL/dA = T Psi^T; only three diagonals needed
  VectorXd dA(M);
  double doff = 0.0;
  for (int i = 0; i < M; ++i) {
    dA[i] = T.row(i).dot(op.Psi.row(i));
    if (i + 1 < M) doff += T.row(i).dot(op.Psi.row(i + 1)) + T.row(i + 1).dot(op.Psi.row(i));
  }
  // lambda: photon weights minus the killing on the diagonal of A
  MatrixXd glam = a.glam;
  for (int c = 0; c < C; ++c) glam.col(c) -= dA;
  // u: through the diagonal of A and through s = pi^1/2
  VectorXd du = VectorXd::Zero(M);
  for (int i = 0; i < M; ++i) {
    du[i] += dA[i] * (-0.5 * op.o * op.cnb[i]);
    if (i + 1 < M) du[i + 1] += dA[i] * 0.5 * op.o * std::exp(-0.5 * (op.u[i + 1] - op.u[i]));
    if (i > 0) du[i - 1] += dA[i] * 0.5 * op.o * std::exp(-0.5 * (op.u[i - 1] - op.u[i]));
  }
  const double gss = a.gs.dot(op.s);
  for (int j = 0; j < M; ++j) du[j] += -0.5 * (a.gs[j] * op.s[j] - op.pi[j] * gss);
  std::vector<double> g(K + 1 + 2 * C, 0.0);
  for (int k = 0; k < K; ++k) {
    double v = 0.0;
    for (int i = 0; i < M; ++i) v += phi[static_cast<std::size_t>(i) * K + k] * du[i];
    g[k] = v;
  }
  g[K] = op.o * (doff - dA.dot(op.cnb));
  for (int c = 0; c < C; ++c) {
    g[K + 1 + c] = op.amp[c] * glam.col(c).dot(op.shape.col(c));
    g[K + 1 + C + c] = op.bg[c] * glam.col(c).sum();
  }
  return g;
}

}  // namespace fret_landscape_detail

// --- FRETLandscapeModel -------------------------------------------------------

FRETLandscapeModel::FRETLandscapeModel(double x_min, double x_max, int n_grid, int n_knots,
                                       double forster_radius,
                                       const std::vector<double>& donor_fraction,
                                       const std::vector<double>& acceptor_fraction)
    : xmin_(x_min), xmax_(x_max), h_(0.0), r0_(forster_radius), m_(n_grid), k_(n_knots),
      c_(0), spline_(x_min, x_max, n_knots < 2 ? 2 : n_knots) {
  if (n_grid < 2) IMP_THROW("FRETLandscapeModel: n_grid must be >= 2", IMP::ValueException);
  if (n_knots < 2) IMP_THROW("FRETLandscapeModel: n_knots must be >= 2", IMP::ValueException);
  if (!(x_max > x_min)) IMP_THROW("FRETLandscapeModel: x_max must exceed x_min", IMP::ValueException);
  if (!(forster_radius > 0.0))
    IMP_THROW("FRETLandscapeModel: forster_radius must be > 0", IMP::ValueException);
  dfrac_ = donor_fraction.empty() ? std::vector<double>{0.97, 0.03} : donor_fraction;
  afrac_ = acceptor_fraction.empty() ? std::vector<double>{0.08, 0.92} : acceptor_fraction;
  if (dfrac_.size() != afrac_.size())
    IMP_THROW("FRETLandscapeModel: donor_fraction and acceptor_fraction need one entry per channel",
              IMP::ValueException);
  c_ = static_cast<int>(dfrac_.size());
  if (c_ < 1) IMP_THROW("FRETLandscapeModel: at least one channel", IMP::ValueException);
  h_ = (x_max - x_min) / (n_grid - 1);
  x_.resize(m_);
  eff_.resize(m_);
  for (int i = 0; i < m_; ++i) {
    x_[i] = x_min + h_ * i;
    eff_[i] = fret_efficiency(x_[i], r0_);
  }
  phi_ = spline_.get_basis(x_);
}

void FRETLandscapeModel::check_theta(const std::vector<double>& theta) const {
  if (static_cast<int>(theta.size()) != get_n_parameters())
    IMP_THROW("FRETLandscapeModel: theta has " << theta.size() << " entries, expected "
                                               << get_n_parameters(),
              IMP::ValueException);
}

std::vector<double> FRETLandscapeModel::pack_parameters(
    const std::vector<double>& knot_heights, double diffusion,
    const std::vector<double>& amplitudes, const std::vector<double>& backgrounds) const {
  if (static_cast<int>(knot_heights.size()) != k_ ||
      static_cast<int>(amplitudes.size()) != c_ || static_cast<int>(backgrounds.size()) != c_)
    IMP_THROW("pack_parameters: expected " << k_ << " knot heights and " << c_
                                           << " amplitudes and backgrounds",
              IMP::ValueException);
  if (!(diffusion > 0.0)) IMP_THROW("pack_parameters: diffusion must be > 0", IMP::ValueException);
  std::vector<double> t(knot_heights);
  t.push_back(std::log(diffusion));
  for (double a : amplitudes) {
    if (!(a > 0.0)) IMP_THROW("pack_parameters: amplitudes must be > 0", IMP::ValueException);
    t.push_back(std::log(a));
  }
  for (double b : backgrounds) {
    if (!(b > 0.0)) IMP_THROW("pack_parameters: backgrounds must be > 0", IMP::ValueException);
    t.push_back(std::log(b));
  }
  return t;
}

std::vector<double> FRETLandscapeModel::get_landscape(const std::vector<double>& theta) const {
  check_theta(theta);
  std::vector<double> u(m_, 0.0);
  for (int i = 0; i < m_; ++i)
    for (int k = 0; k < k_; ++k) u[i] += phi_[static_cast<std::size_t>(i) * k_ + k] * theta[k];
  return u;
}

double FRETLandscapeModel::get_diffusion(const std::vector<double>& theta) const {
  check_theta(theta);
  return std::exp(theta[k_]);
}

std::vector<double> FRETLandscapeModel::get_amplitudes(const std::vector<double>& theta) const {
  check_theta(theta);
  std::vector<double> a(c_);
  for (int c = 0; c < c_; ++c) a[c] = std::exp(theta[k_ + 1 + c]);
  return a;
}

std::vector<double> FRETLandscapeModel::get_backgrounds(const std::vector<double>& theta) const {
  check_theta(theta);
  std::vector<double> b(c_);
  for (int c = 0; c < c_; ++c) b[c] = std::exp(theta[k_ + 1 + c_ + c]);
  return b;
}

std::vector<double> FRETLandscapeModel::get_rates(const std::vector<double>& theta) const {
  const std::vector<double> a = get_amplitudes(theta), b = get_backgrounds(theta);
  std::vector<double> r(static_cast<std::size_t>(c_) * m_);
  for (int c = 0; c < c_; ++c)
    for (int i = 0; i < m_; ++i)
      r[static_cast<std::size_t>(c) * m_ + i] =
          a[c] * (dfrac_[c] * (1.0 - eff_[i]) + afrac_[c] * eff_[i]) + b[c];
  return r;
}

void FRETLandscapeModel::set_photons(const std::vector<double>& times,
                                     const std::vector<int>& channels,
                                     const std::vector<int>& offsets) {
  if (times.size() != channels.size())
    IMP_THROW("set_photons: times and channels differ in length", IMP::ValueException);
  if (offsets.size() < 2 || offsets.front() != 0 ||
      offsets.back() != static_cast<int>(times.size()))
    IMP_THROW("set_photons: offsets must run from 0 to the number of photons",
              IMP::ValueException);
  double tmax = 0.0;
  for (std::size_t m = 0; m + 1 < offsets.size(); ++m) {
    if (offsets[m + 1] < offsets[m])
      IMP_THROW("set_photons: offsets must be non-decreasing", IMP::ValueException);
    for (int n = offsets[m] + 1; n < offsets[m + 1]; ++n) {
      const double d = times[n] - times[n - 1];
      if (!(d >= 0.0))
        IMP_THROW("set_photons: times must be non-decreasing within a trace (trace "
                      << m << ")",
                  IMP::ValueException);
      tmax = std::max(tmax, d);
    }
  }
  for (int c : channels)
    if (c < 0 || c >= c_)
      IMP_THROW("set_photons: channel " << c << " outside 0.." << c_ - 1, IMP::ValueException);
  times_ = times;
  channels_ = channels;
  offsets_ = offsets;
  tau_max_ = tmax;
}

void FRETLandscapeModel::set_batch_size(int b) {
  if (b < 1) IMP_THROW("set_batch_size: must be >= 1", IMP::ValueException);
  batch_ = b;
}

double FRETLandscapeModel::evaluate(const std::vector<double>& theta,
                                    const std::vector<int>& traces,
                                    std::vector<double>* gradient,
                                    std::vector<double>* per_trace) const {
  check_theta(theta);
  if (offsets_.empty()) IMP_THROW("FRETLandscapeModel: no photons (set_photons)", IMP::ValueException);
  namespace fl = fret_landscape_detail;
  const fl::Operator op = fl::fret_landscape_operator(theta, m_, k_, c_, h_, phi_, eff_, dfrac_,
                                              afrac_, tau_max_);
  // longest first, so a batch holds traces of similar length
  std::vector<int> order(traces);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return offsets_[a + 1] - offsets_[a] > offsets_[b + 1] - offsets_[b];
  });
  const int nb = (static_cast<int>(order.size()) + batch_ - 1) / batch_;
  const fl::PhotonView ph{times_.data(), channels_.data(), offsets_.data()};
  std::vector<double> batch_logl(nb, 0.0);
  std::vector<double> sorted_trace(order.size(), 0.0);
  std::vector<fl::Accum> accs(gradient ? nb : 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int ib = 0; ib < nb; ++ib) {
    const int b0 = ib * batch_;
    const int B = std::min(batch_, static_cast<int>(order.size()) - b0);
    fl::Accum* acc = nullptr;
    if (gradient) {
      accs[ib].init(m_, c_, op.close.size());
      acc = &accs[ib];
    }
    batch_logl[ib] = fl::fret_landscape_batch(op, ph, order.data() + b0, B, acc,
                                          sorted_trace.data() + b0);
  }
  double total = 0.0;
  for (double v : batch_logl) total += v;
  if (per_trace) {
    per_trace->assign(traces.size(), 0.0);
    std::vector<int> pos(get_n_traces(), -1);
    for (std::size_t i = 0; i < traces.size(); ++i) pos[traces[i]] = static_cast<int>(i);
    for (std::size_t i = 0; i < order.size(); ++i) (*per_trace)[pos[order[i]]] = sorted_trace[i];
  }
  if (gradient) {
    if (!std::isfinite(total)) {
      gradient->assign(get_n_parameters(), 0.0);
      return total;
    }
    fl::Accum a;
    a.init(m_, c_, op.close.size());
    for (const fl::Accum& x : accs) a.add(x);
    *gradient = fl::fret_landscape_chain(op, a, k_, phi_);
  }
  return total;
}

namespace {
std::vector<int> fret_landscape_all_traces(int n) {
  std::vector<int> t(n);
  std::iota(t.begin(), t.end(), 0);
  return t;
}
}  // namespace

double FRETLandscapeModel::log_likelihood(const std::vector<double>& theta) const {
  return evaluate(theta, fret_landscape_all_traces(get_n_traces()), nullptr, nullptr);
}

std::vector<double> FRETLandscapeModel::trace_log_likelihoods(
    const std::vector<double>& theta) const {
  std::vector<double> out;
  evaluate(theta, fret_landscape_all_traces(get_n_traces()), nullptr, &out);
  return out;
}

std::vector<double> FRETLandscapeModel::log_likelihood_gradient(
    const std::vector<double>& theta) const {
  std::vector<double> g;
  evaluate(theta, fret_landscape_all_traces(get_n_traces()), &g, nullptr);
  return g;
}

std::vector<double> FRETLandscapeModel::log_likelihood_and_gradient(
    const std::vector<double>& theta) const {
  std::vector<double> g;
  const double f = evaluate(theta, fret_landscape_all_traces(get_n_traces()), &g, nullptr);
  g.insert(g.begin(), f);
  return g;
}


// --- priors -------------------------------------------------------------------

void FRETLandscapeModel::set_roughness_weight(double omega) {
  if (!(omega >= 0.0)) IMP_THROW("set_roughness_weight: must be >= 0", IMP::ValueException);
  omega_ = omega;
}

void FRETLandscapeModel::set_anchor_sigma(double sigma) { anchor_ = sigma; }

void FRETLandscapeModel::set_background_prior(const std::vector<double>& modes,
                                              const std::vector<double>& sds) {
  if (modes.empty() && sds.empty()) {
    bg_mode_.clear();
    bg_sd_.clear();
    return;
  }
  if (static_cast<int>(modes.size()) != c_ || static_cast<int>(sds.size()) != c_)
    IMP_THROW("set_background_prior: one mode and one width per channel", IMP::ValueException);
  for (int c = 0; c < c_; ++c)
    if (sds[c] > 0.0 && !(modes[c] > 0.0))
      IMP_THROW("set_background_prior: modes must be > 0", IMP::ValueException);
  bg_mode_ = modes;
  bg_sd_ = sds;
}

namespace {
//! -log p, its gradient and (optionally) its Hessian, in theta.
double fret_landscape_neg_log_prior(const std::vector<double>& theta, int K, int C, double hs,
                                    double omega, double anchor,
                                    const std::vector<double>& bg_mode,
                                    const std::vector<double>& bg_sd,
                                    std::vector<double>* grad, std::vector<double>* hess) {
  const int P = K + 1 + 2 * C;
  if (grad) grad->assign(P, 0.0);
  if (hess) hess->assign(static_cast<std::size_t>(P) * P, 0.0);
  double f = 0.0;
  if (omega > 0.0 && K >= 3) {
    const double w = omega / (hs * hs * hs * hs);
    for (int k = 1; k + 1 < K; ++k) {
      const double d = theta[k + 1] - 2.0 * theta[k] + theta[k - 1];
      f += w * d * d;
      const int idx[3] = {k - 1, k, k + 1};
      const double cf[3] = {1.0, -2.0, 1.0};
      for (int a = 0; a < 3; ++a) {
        if (grad) (*grad)[idx[a]] += 2.0 * w * d * cf[a];
        if (hess)
          for (int b = 0; b < 3; ++b)
            (*hess)[static_cast<std::size_t>(idx[a]) * P + idx[b]] += 2.0 * w * cf[a] * cf[b];
      }
    }
  }
  if (anchor > 0.0) {
    double mean = 0.0;
    for (int k = 0; k < K; ++k) mean += theta[k];
    mean /= K;
    const double s2 = anchor * anchor;
    f += 0.5 * mean * mean / s2;
    for (int k = 0; k < K; ++k) {
      if (grad) (*grad)[k] += mean / (s2 * K);
      if (hess)
        for (int l = 0; l < K; ++l)
          (*hess)[static_cast<std::size_t>(k) * P + l] += 1.0 / (s2 * K * K);
    }
  }
  for (int c = 0; c < static_cast<int>(bg_mode.size()); ++c) {
    if (!(bg_sd[c] > 0.0)) continue;
    const int i = K + 1 + C + c;
    const double kk = bg_mode[c] * bg_mode[c] / (bg_sd[c] * bg_sd[c]);
    const double r = std::exp(theta[i]) / bg_mode[c];
    f += kk * (r - std::log(r));
    if (grad) (*grad)[i] += kk * (r - 1.0);
    if (hess) (*hess)[static_cast<std::size_t>(i) * P + i] += kk * r;
  }
  return f;
}
}  // namespace

double FRETLandscapeModel::log_prior(const std::vector<double>& theta) const {
  check_theta(theta);
  return -fret_landscape_neg_log_prior(theta, k_, c_, spline_.get_knot_spacing(), omega_,
                                       anchor_, bg_mode_, bg_sd_, nullptr, nullptr);
}

std::vector<double> FRETLandscapeModel::log_prior_gradient(
    const std::vector<double>& theta) const {
  check_theta(theta);
  std::vector<double> g;
  fret_landscape_neg_log_prior(theta, k_, c_, spline_.get_knot_spacing(), omega_, anchor_,
                               bg_mode_, bg_sd_, &g, nullptr);
  for (double& v : g) v = -v;
  return g;
}

std::vector<double> FRETLandscapeModel::prior_precision(const std::vector<double>& theta) const {
  check_theta(theta);
  std::vector<double> h;
  fret_landscape_neg_log_prior(theta, k_, c_, spline_.get_knot_spacing(), omega_, anchor_,
                               bg_mode_, bg_sd_, nullptr, &h);
  return h;
}

double FRETLandscapeModel::log_posterior(const std::vector<double>& theta) const {
  return log_likelihood(theta) + log_prior(theta);
}

std::vector<double> FRETLandscapeModel::log_posterior_gradient(
    const std::vector<double>& theta) const {
  std::vector<double> g = log_likelihood_gradient(theta);
  const std::vector<double> gp = log_prior_gradient(theta);
  for (std::size_t i = 0; i < g.size(); ++i) g[i] += gp[i];
  return g;
}

// --- fit ------------------------------------------------------------------------

std::vector<double> FRETLandscapeModel::trace_scores(const std::vector<double>& theta) const {
  check_theta(theta);
  if (offsets_.empty()) IMP_THROW("FRETLandscapeModel: no photons (set_photons)", IMP::ValueException);
  namespace fl = fret_landscape_detail;
  const fl::Operator op = fl::fret_landscape_operator(theta, m_, k_, c_, h_, phi_, eff_, dfrac_,
                                                      afrac_, tau_max_);
  const int nt = get_n_traces(), P = get_n_parameters();
  const fl::PhotonView ph{times_.data(), channels_.data(), offsets_.data()};
  std::vector<double> out(static_cast<std::size_t>(nt) * P, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int m = 0; m < nt; ++m) {
    fl::Accum acc;
    acc.init(m_, c_, op.close.size());
    double ll;
    const double v = fl::fret_landscape_batch(op, ph, &m, 1, &acc, &ll);
    if (!std::isfinite(v)) continue;
    const std::vector<double> g = fl::fret_landscape_chain(op, acc, k_, phi_);
    std::copy(g.begin(), g.end(), out.begin() + static_cast<std::size_t>(m) * P);
  }
  return out;
}

// --- fit ------------------------------------------------------------------------

#if IMP_BFF_HAS_TTTRLIB
namespace {
//! The optimiser sees z, with theta = theta0 + T z on the free indices.
struct FRETLandscapeFitContext {
  const FRETLandscapeModel* model;
  std::vector<double> theta0;
  std::vector<int> free;
  Eigen::MatrixXd T;
  std::vector<double> theta_of(const double* z) const {
    std::vector<double> th(theta0);
    const int n = static_cast<int>(free.size());
    for (int i = 0; i < n; ++i) {
      double v = 0.0;
      for (int j = 0; j < n; ++j) v += T(i, j) * z[j];
      th[free[i]] += v;
    }
    return th;
  }
};

double fret_landscape_fit_target(double* z, void* p) {
  const FRETLandscapeFitContext* ctx = static_cast<const FRETLandscapeFitContext*>(p);
  const double v = -ctx->model->log_posterior(ctx->theta_of(z));
  return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
}

double fret_landscape_fit_gradient(double* z, double* g, void* p) {
  const FRETLandscapeFitContext* ctx = static_cast<const FRETLandscapeFitContext*>(p);
  const std::vector<double> th = ctx->theta_of(z);
  const std::vector<double> fg = ctx->model->log_likelihood_and_gradient(th);
  const std::vector<double> gp = ctx->model->log_prior_gradient(th);
  const int n = static_cast<int>(ctx->free.size());
  for (int j = 0; j < n; ++j) {
    double v = 0.0;
    for (int i = 0; i < n; ++i) v -= ctx->T(i, j) * (fg[ctx->free[i] + 1] + gp[ctx->free[i]]);
    g[j] = v;
  }
  const double v = -(fg[0] + ctx->model->log_prior(th));
  return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
}
}  // namespace
#endif

FRETLandscapeFit FRETLandscapeModel::fit(const std::vector<double>& theta0,
                                         const FRETLandscapeFitOptions& options) const {
  check_theta(theta0);
#if !IMP_BFF_HAS_TTTRLIB
  IMP_THROW("FRETLandscapeModel::fit needs tttrlib's L-BFGS (tttrlib/i_lbfgs.h); this "
            "IMP.bff was built without tttrlib",
            IMP::ValueException);
#else
  if (options.patience < 1 || options.max_iterations < 1)
    IMP_THROW("fit: patience and max_iterations must be >= 1", IMP::ValueException);
  const int P = get_n_parameters();
  std::vector<char> is_fixed(P, 0);
  for (int i : options.fixed) {
    if (i < 0 || i >= P) IMP_THROW("fit: fixed index " << i << " out of range", IMP::ValueException);
    is_fixed[i] = 1;
  }
  FRETLandscapeFitContext ctx;
  ctx.model = this;
  for (int i = 0; i < P; ++i)
    if (!is_fixed[i]) ctx.free.push_back(i);
  const int n = static_cast<int>(ctx.free.size());
  FRETLandscapeFit out;
  std::vector<double> x(theta0);
  double best = log_posterior(x);
  out.history_.push_back(best);
  if (!std::isfinite(best) || n == 0) {
    out.theta_ = x;
    out.log_posterior_ = best;
    out.log_likelihood_ = log_likelihood(x);
    out.status_ = std::isfinite(best) ? "converged" : "failed";
    return out;
  }
  out.status_ = "max_iterations";
  int done = 0;
  while (done < options.max_iterations) {
    // Preconditioner: theta = x + T z with T T^T = H^-1, H the empirical
    // Fisher information plus the prior precision (the Laplace precision of
    // Sec. III E) at the block's start point. Identity when not requested.
    ctx.theta0 = x;
    ctx.T = Eigen::MatrixXd::Identity(n, n);
    if (options.precondition) {
      const std::vector<double> sc = trace_scores(x), pp = prior_precision(x);
      Eigen::MatrixXd H(n, n);
      const int nt = get_n_traces();
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
          double v = pp[static_cast<std::size_t>(ctx.free[i]) * P + ctx.free[j]];
          for (int m = 0; m < nt; ++m)
            v += sc[static_cast<std::size_t>(m) * P + ctx.free[i]] *
                 sc[static_cast<std::size_t>(m) * P + ctx.free[j]];
          H(i, j) = v;
        }
      // Damping: directions the data barely see (a knot on a steep, empty
      // wall) have almost no curvature, and a Newton step along them is huge
      // and wrong -- the landscape there is far from quadratic. The floor
      // caps such a step at about |gradient| / damping.
      const double ridge = std::max(options.precondition_damping,
                                    1e-8 * std::max(1.0, H.diagonal().maxCoeff()));
      for (int i = 0; i < n; ++i) H(i, i) += ridge;
      Eigen::LLT<Eigen::MatrixXd> llt(H);
      if (llt.info() == Eigen::Success)
        ctx.T = llt.matrixU().solve(Eigen::MatrixXd::Identity(n, n));  // U^-1: (U^-1)(U^-1)^T = H^-1
    }
    const int iters = std::min(options.patience, options.max_iterations - done);
    bfgs opt(fret_landscape_fit_target, n);
    opt.set_gradient(fret_landscape_fit_gradient);
    opt.maxiter = iters;
    std::vector<double> z(n, 0.0);
    const int info = opt.minimize(z.data(), &ctx);
    done += iters;
    const std::vector<double> trial = ctx.theta_of(z.data());
    const double v = log_posterior(trial);
    const double gain = std::isfinite(v) ? v - best : -1.0;
    if (std::isfinite(v) && v > best) {
      x = trial;
      best = v;
    }
    out.history_.push_back(best);
    if (gain < options.min_delta) {
      out.status_ = (info == 1 || info == 2 || info == 4) ? "converged" : "patience";
      break;
    }
  }
  out.n_iterations_ = done;
  out.theta_ = x;
  out.log_posterior_ = best;
  out.log_likelihood_ = log_likelihood(x);
  return out;
#endif
}

IMPBFF_END_NAMESPACE
