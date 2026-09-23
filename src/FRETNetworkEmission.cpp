// SPDX-License-Identifier: BSD-3-Clause
/**
 * Per-state emission of a FRET network measurement: detection rates and
 * microtime densities from the excited-state kinetics
 * (PhotophysicsTransferKinetics.h), the light path
 * (PhotophysicsCrosstalkMatrix.h) and the instrument response, integrated
 * over each hidden state's distance distribution. See FRETNetwork.h.
 */
#include <IMP/bff/FRETNetwork.h>
#include <IMP/bff/PhotophysicsTransferKinetics.h>

#include <algorithm>
#include <cmath>

IMPBFF_BEGIN_NAMESPACE

// --- instrument -------------------------------------------------------------------

FRETInstrument::FRETInstrument()
    : FRETInstrument(PhotophysicsCrosstalkMatrix(std::vector<std::string>{"donor_pulse"},
                                                 std::vector<std::string>{"donor", "acceptor"},
                                                 std::vector<double>{1.0, 0.0}),
                     PhotophysicsCrosstalkMatrix(std::vector<std::string>{"donor", "acceptor"},
                                                 std::vector<std::string>{"green", "red"},
                                                 std::vector<double>{1.0, 0.0, 0.0, 1.0}),
                     1, 0.1) {}

FRETInstrument::FRETInstrument(const PhotophysicsCrosstalkMatrix& excitation,
                               const PhotophysicsCrosstalkMatrix& emission, int n_bins,
                               double bin_width)
    : n_bins_(n_bins), bin_width_(bin_width) {
  if (n_bins < 1 || !(bin_width > 0.0))
    IMP_THROW("FRETInstrument: n_bins >= 1 and bin_width > 0", IMP::ValueException);
  if (excitation.get_n_rows() < 1 || emission.get_n_columns() < 1)
    IMP_THROW("FRETInstrument: need at least one pulse and one channel", IMP::ValueException);
  pulses_ = excitation.get_rows();
  channels_ = emission.get_columns();
  const std::vector<std::string> chrom = {"donor", "acceptor"};
  const PhotophysicsCrosstalkMatrix ex = excitation.select(pulses_, chrom);
  const PhotophysicsCrosstalkMatrix em = emission.select(chrom, channels_);
  exc_ = ex.get_values();
  em_ = em.get_values();
  for (double v : exc_)
    if (!(v >= 0.0)) IMP_THROW("FRETInstrument: excitation must be >= 0", IMP::ValueException);
  for (double v : em_)
    if (!(v >= 0.0)) IMP_THROW("FRETInstrument: emission must be >= 0", IMP::ValueException);
  const int C = get_n_channels();
  std::vector<double> delta(n_bins_, 0.0);
  delta[0] = 1.0;
  irf_.assign(static_cast<std::size_t>(get_n_pulses()) * C, delta);
  gain_.assign(C, 1.0);
  bg_.assign(C, 0.0);
  bg_density_.assign(C, std::vector<double>());
}

double FRETInstrument::get_excitation(int pulse, int chromophore) const {
  return exc_.at(static_cast<std::size_t>(pulse) * 2 + chromophore);
}

double FRETInstrument::get_emission(int chromophore, int channel) const {
  return em_.at(static_cast<std::size_t>(chromophore) * get_n_channels() + channel);
}

void FRETInstrument::set_irf(int pulse, int channel, const std::vector<double>& irf) {
  if (pulse < 0 || pulse >= get_n_pulses() || channel < 0 || channel >= get_n_channels())
    IMP_THROW("set_irf: no pulse " << pulse << " / channel " << channel, IMP::ValueException);
  if (static_cast<int>(irf.size()) != n_bins_)
    IMP_THROW("set_irf: need " << n_bins_ << " values", IMP::ValueException);
  double s = 0.0;
  for (double v : irf) {
    if (!(v >= 0.0)) IMP_THROW("set_irf: values must be >= 0", IMP::ValueException);
    s += v;
  }
  if (!(s > 0.0)) IMP_THROW("set_irf: all zero", IMP::ValueException);
  std::vector<double> n(irf);
  for (double& v : n) v /= s;
  irf_[static_cast<std::size_t>(pulse) * get_n_channels() + channel] = n;
}

std::vector<double> FRETInstrument::get_irf(int pulse, int channel) const {
  return irf_.at(static_cast<std::size_t>(pulse) * get_n_channels() + channel);
}

void FRETInstrument::set_gain(int channel, double gain) {
  if (!(gain > 0.0)) IMP_THROW("set_gain: must be > 0", IMP::ValueException);
  gain_.at(channel) = gain;
}

void FRETInstrument::set_background(int channel, double rate,
                                    const std::vector<double>& density) {
  if (!(rate >= 0.0)) IMP_THROW("set_background: rate must be >= 0", IMP::ValueException);
  if (!density.empty() && static_cast<int>(density.size()) != n_bins_)
    IMP_THROW("set_background: density needs " << n_bins_ << " values", IMP::ValueException);
  bg_.at(channel) = rate;
  std::vector<double> d(density);
  double s = 0.0;
  for (double v : d) s += v;
  if (!d.empty()) {
    if (!(s > 0.0)) IMP_THROW("set_background: density sums to zero", IMP::ValueException);
    for (double& v : d) v /= s;
  }
  bg_density_.at(channel) = d;
}

std::vector<double> FRETInstrument::get_background_density(int channel) const {
  const std::vector<double>& d = bg_density_.at(channel);
  if (!d.empty()) return d;
  return std::vector<double>(n_bins_, 1.0 / n_bins_);
}

std::vector<std::string> FRETInstrument::get_parameter_names() const {
  std::vector<std::string> n;
  const char* chrom[2] = {"donor", "acceptor"};
  for (const std::string& p : pulses_)
    for (int k = 0; k < 2; ++k) n.push_back("excitation[" + p + "," + chrom[k] + "]");
  for (int k = 0; k < 2; ++k)
    for (const std::string& c : channels_) n.push_back("emission[" + std::string(chrom[k]) + "," + c + "]");
  for (const std::string& c : channels_) n.push_back("gain[" + c + "]");
  for (const std::string& c : channels_) n.push_back("background[" + c + "]");
  return n;
}

std::vector<double> FRETInstrument::get_parameter_values() const {
  std::vector<double> v(exc_);
  v.insert(v.end(), em_.begin(), em_.end());
  v.insert(v.end(), gain_.begin(), gain_.end());
  v.insert(v.end(), bg_.begin(), bg_.end());
  return v;
}

void FRETInstrument::set_parameter_values(const std::vector<double>& v) {
  if (v.size() != exc_.size() + em_.size() + gain_.size() + bg_.size())
    IMP_THROW("FRETInstrument: wrong number of parameter values", IMP::ValueException);
  std::size_t j = 0;
  for (double& x : exc_) x = v[j++];
  for (double& x : em_) x = v[j++];
  for (double& x : gain_) x = v[j++];
  for (double& x : bg_) x = v[j++];
}

// --- emission ------------------------------------------------------------------------

namespace fret_network_detail {

//! kD - (1 - exp(-kD)), accurate for small kD.
inline double exp_excess(double kd) {
  if (kd < 1e-3) return kd * kd * (0.5 - kd * (1.0 / 6.0 - kd / 24.0));
  return kd + std::expm1(-kd);
}

//! Adds the bin integrals of `A exp(-k t)` convolved with a box-per-bin IRF,
//! periodic over the n bins, to `out`.
void add_periodic_exponential(double A, double k, const std::vector<double>& irf, double dt,
                              double* out) {
  const int n = static_cast<int>(irf.size());
  if (A == 0.0) return;
  const double kd = k * dt;
  const double x = std::exp(-kd), omx = -std::expm1(-kd);
  const double c0 = A * exp_excess(kd) / (k * kd);
  const double c1 = A * omx * omx / (k * kd);
  // z_j = sum_{m>=1} irf[(j-m) mod n] x^(m-1), periodic
  double z = 0.0, xp = 1.0;
  for (int m = 1; m <= n; ++m) {
    z += irf[n - m] * xp;
    xp *= x;
  }
  const double xn = std::exp(-kd * n);
  z /= (1.0 - xn);
  for (int j = 0; j < n; ++j) {
    if (j > 0) z = irf[j - 1] + x * z;
    out[j] += irf[j] * c0 + c1 * z;
  }
}

//! Photon counts and microtime shapes of one (state, distance) in every
//! channel: adds `weight * E_c(bin)` into `table[c * n_bins + bin]`.
void add_state_emission(const FRETInstrument& ins, double r0, double tau0, double r,
                        double tauD, double qyD, double excD, double tauA, double qyA,
                        double excA, double accA, double weight, double* table) {
  const int C = ins.get_n_channels(), P = ins.get_n_pulses(), nb = ins.get_n_bins();
  const double kt = r > 0.0 ? accA * std::pow(r0 / r, 6) / tau0 : 0.0;
  const bool micro = nb > 1;
  for (int p = 0; p < P; ++p) {
    const double pD = ins.get_excitation(p, 0) * excD, pA = ins.get_excitation(p, 1) * excA;
    if (pD == 0.0 && pA == 0.0) continue;
    for (int c = 0; c < C; ++c) {
      const double g = ins.get_gain(c);
      const double mD = ins.get_emission(0, c) * g * qyD / tauD;
      const double mA = ins.get_emission(1, c) * g * qyA / tauA;
      if (mD == 0.0 && mA == 0.0) continue;
      const std::vector<double> comp =
          transfer_pair_components(tauD, tauA, kt, 0.0, pD, pA, mD, mA);
      double* out = table + static_cast<std::size_t>(c) * nb;
      if (!micro) {
        double cnt = 0.0;
        for (int i = 0; i < 2; ++i) {
          const double amp = comp[3 * i], rate = comp[3 * i + 1];
          cnt += comp[3 * i + 2] == TRANSFER_T_EXPONENTIAL ? amp / (rate * rate) : amp / rate;
        }
        out[0] += weight * cnt;
        continue;
      }
      const std::vector<double> irf = ins.get_irf(p, c);
      for (int i = 0; i < 2; ++i) {
        const double amp = weight * comp[3 * i], rate = comp[3 * i + 1];
        if (comp[3 * i + 2] == TRANSFER_T_EXPONENTIAL) {
          // A t e^{-kt} = lim (A/2d)(e^{-(k-d)t} - e^{-(k+d)t}); O(d^2) error
          const double d = 1e-4 * rate;
          add_periodic_exponential(amp / (2 * d), rate - d, irf, ins.get_bin_width(), out);
          add_periodic_exponential(-amp / (2 * d), rate + d, irf, ins.get_bin_width(), out);
        } else {
          add_periodic_exponential(amp, rate, irf, ins.get_bin_width(), out);
        }
      }
    }
  }
}

}  // namespace fret_network_detail

std::vector<double> FRETMeasurement::get_emission(const FRETHiddenProcess& process) const {
  check_process(process);
  const int nh = process.get_n_states(), nd = donor_.get_n_states(),
            na = acceptor_.get_n_states();
  const int n = nh * nd * na, C = instrument_.get_n_channels(), nb = instrument_.get_n_bins();
  const double tau0 = get_reference_lifetime();
  std::vector<double> out(static_cast<std::size_t>(C) * nb * n, 0.0);
  std::vector<double> table(static_cast<std::size_t>(C) * nb);
  const std::vector<double>& dt = donor_.get_lifetimes();
  const std::vector<double>& dq = donor_.get_quantum_yields();
  const std::vector<double>& de = donor_.get_excitations();
  const std::vector<double>& at = acceptor_.get_lifetimes();
  const std::vector<double>& aq = acceptor_.get_quantum_yields();
  const std::vector<double>& ae = acceptor_.get_excitations();
  const std::vector<double>& ac = acceptor_.get_accepts();
  std::vector<std::vector<double> > bgd(C);
  for (int c = 0; c < C; ++c) bgd[c] = instrument_.get_background_density(c);
  for (int h = 0; h < nh; ++h) {
    const std::vector<double> rw = get_state_distances(process, h);
    for (int d = 0; d < nd; ++d)
      for (int a = 0; a < na; ++a) {
        std::fill(table.begin(), table.end(), 0.0);
        for (std::size_t j = 0; j + 1 < rw.size(); j += 2)
          fret_network_detail::add_state_emission(instrument_, r0_, tau0, rw[j], dt[d], dq[d],
                                                  de[d], at[a], aq[a], ae[a], ac[a], rw[j + 1],
                                                  table.data());
        const int s = (h * nd + d) * na + a;
        for (int c = 0; c < C; ++c)
          for (int b = 0; b < nb; ++b)
            out[(static_cast<std::size_t>(c) * nb + b) * n + s] =
                table[static_cast<std::size_t>(c) * nb + b] + instrument_.get_background(c) * bgd[c][b];
      }
  }
  return out;
}

std::vector<double> FRETMeasurement::get_detection_rates(const FRETHiddenProcess& process) const {
  const std::vector<double> e = get_emission(process);
  const int C = instrument_.get_n_channels(), nb = instrument_.get_n_bins();
  const int n = static_cast<int>(e.size() / (static_cast<std::size_t>(C) * nb));
  std::vector<double> out(static_cast<std::size_t>(C) * n, 0.0);
  for (int c = 0; c < C; ++c)
    for (int b = 0; b < nb; ++b)
      for (int s = 0; s < n; ++s)
        out[static_cast<std::size_t>(c) * n + s] += e[(static_cast<std::size_t>(c) * nb + b) * n + s];
  return out;
}

IMPBFF_END_NAMESPACE
