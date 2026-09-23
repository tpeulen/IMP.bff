/**
 *  \file IMP/bff/FRETNetwork.h
 *  \brief Burstwise, microtime-resolved photon-by-photon inference of a
 *         hidden conformational process observed by a network of FRET pairs.
 *
 * Generalises FRETLandscapeModel (FRETLandscape.h, arXiv:2608.21061) in three
 * directions.
 *
 * **The state is a product of factors.** The joint state of a measured
 * molecule is `s = (h, d, a)`: a hidden conformational state `h`, a donor
 * photophysical state `d` and an acceptor photophysical state `a`. The
 * hidden process (FRETHiddenProcess) is either `K` discrete conformers with a
 * rate matrix, or a free-energy landscape over a path coordinate `q`,
 * discretised by SqRA. Each dye (FRETDye) is declared as data: named states,
 * rates between them (optionally scaled by excitation power, optionally
 * conditioned on the hidden state), and per state an excitability, a quantum
 * yield, a lifetime and whether it accepts energy (an acceptor that absorbs
 * but does not emit is `accepts = 1`, `quantum_yield = 0`; a bleached one is
 * `accepts = 0`, `excitation = 0`). The joint generator is the amalgamation
 * of the factors' conditional intensity matrices (KineticNetwork.h), never
 * hand-written Kronecker arithmetic. A "FRET state" is whatever function of
 * the joint state a caller reads off; the hidden and photophysical states are
 * different things.
 *
 * **The hidden process is shared; FRET pairs are observations of it.** A
 * FRETMeasurement is one label pair with its own dyes, instrument and bursts,
 * and a map from hidden state to donor-acceptor distance (a distribution per
 * state). Several measurements of the same molecule observe the same hidden
 * states with the same kinetics; FRETNetworkModel sums their likelihoods.
 * With a landscape, the map `r_p(q)` is a spline that may be non-monotonic,
 * which is what one pair alone cannot tell from extra intermediates.
 *
 * Joint state index: C-order over `(hidden, donor, acceptor)`, hidden most
 * significant -- the KineticNetwork convention. Matrices are
 * `K[target, source]`, row-major, columns summing to zero.
 *
 * Units: macrotimes and rates in one unit (e.g. ms and 1/ms), distances in
 * the unit of the Forster radius, lifetimes and microtimes in ns.
 *
 * \authors Thomas-Otavio Peulen
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_FRETNETWORK_H
#define IMPBFF_FRETNETWORK_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/KineticNetwork.h>
#include <IMP/bff/PhotophysicsCrosstalkMatrix.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! How a parameter enters the model: which builders a change invalidates.
enum FRETParameterKind {
  FRET_PARAMETER_GENERATOR = 1,  //!< rates of the joint generator
  FRET_PARAMETER_START = 2,      //!< the start distribution
  FRET_PARAMETER_EMISSION = 4    //!< detection rates and microtime densities
};

//! How a parameter is transformed to the unconstrained scale the fit uses.
enum FRETParameterTransform {
  FRET_TRANSFORM_IDENTITY = 0,
  FRET_TRANSFORM_LOG = 1,
  FRET_TRANSFORM_LOGIT = 2
};

//! The hidden conformational process shared by every measurement.
/*! Either `K` discrete conformers with a rate matrix, or a free-energy
    landscape `u(q)` (kT, natural cubic spline on uniform knots) with
    diffusion coefficient `D` over a path coordinate `q`, discretised on a
    uniform grid by SqRA (FRETLandscapeGrid.h). Start distribution: the
    stationary distribution (Boltzmann on the grid for a landscape). */
class IMPBFFEXPORT FRETHiddenProcess {
 public:
  //! `n_states` discrete conformers; every rate starts at zero.
  explicit FRETHiddenProcess(int n_states = 2);
  //! A landscape over `q` in `[q_min, q_max]`: `n_grid` SqRA points,
  //! `n_knots` spline knots, flat, `D = 1`.
  FRETHiddenProcess(double q_min, double q_max, int n_grid, int n_knots);

  bool get_is_landscape() const { return landscape_; }
  //! Discrete conformers, or grid points of a landscape.
  int get_n_states() const { return n_; }
  //! The grid of `q`, or `0 .. K-1` for discrete conformers.
  std::vector<double> get_coordinates() const;

  //! Discrete: the rate from conformer `source` to `target` (> 0 makes it a
  //! parameter, `log k`).
  void set_rate(int source, int target, double rate);
  double get_rate(int source, int target) const;
  //! Landscape: knot heights (kT) and `D`.
  void set_landscape(const std::vector<double>& knot_heights);
  const std::vector<double>& get_knot_heights() const { return mu_; }
  std::vector<double> get_knots() const;
  void set_diffusion(double diffusion);
  double get_diffusion() const { return diffusion_; }
  //! `u` on the grid (landscape only).
  std::vector<double> get_landscape() const;

  //! The generator `K[target, source]`, row-major `n x n`.
  std::vector<double> get_generator() const;
  //! Its stationary distribution.
  std::vector<double> get_stationary() const;

  //! Parameters, natural scale: discrete `rate[i->j]` for every nonzero
  //! rate; landscape `mu[k]` then `D`.
  std::vector<std::string> get_parameter_names() const;
  std::vector<double> get_parameter_values() const;
  void set_parameter_values(const std::vector<double>& values);
  std::vector<int> get_parameter_transforms() const;
  std::vector<int> get_parameter_kinds() const;

  IMP_SHOWABLE_INLINE(FRETHiddenProcess,
                      out << "FRETHiddenProcess(" << (landscape_ ? "landscape, " : "discrete, ")
                          << n_ << " states)");

 private:
  bool landscape_ = false;
  int n_ = 2;
  // discrete
  std::vector<double> rates_;  // [target * n + source]
  std::vector<std::pair<int, int> > rate_index_;
  // landscape
  double q_min_ = 0.0, q_max_ = 1.0, diffusion_ = 1.0;
  int n_knots_ = 0;
  std::vector<double> mu_;
  void rebuild_rate_index();
};
IMP_VALUES(FRETHiddenProcess, FRETHiddenProcesses);

//! One dye's photophysics, declared as data.
/*! Named states with, per state, a relative excitability (absorption at the
    excitation wavelengths, 0 for a bleached or dark-absorbing state), a
    fluorescence quantum yield, a fluorescence lifetime (ns), and whether it
    accepts energy by FRET (0..1, scales `R0^6`; acceptor only). Transitions
    between states are rates; a `light` rate is multiplied by the
    measurement's excitation power (triplet formation, photobleaching,
    photo-isomerisation). A rate may be conditioned on the hidden state
    (PET quenching only in some conformers), which makes the hidden process a
    parent of the dye in the KineticNetwork. */
class IMPBFFEXPORT FRETDye {
 public:
  explicit FRETDye(const std::string& name = "dye");
  const std::string& get_name() const { return name_; }

  //! Add a state; returns its index.
  int add_state(const std::string& name, double excitation = 1.0,
                double quantum_yield = 1.0, double lifetime = 4.0, double accepts = 1.0);
  int get_n_states() const { return static_cast<int>(names_.size()); }
  const std::vector<std::string>& get_state_names() const { return names_; }
  int get_state_index(const std::string& name) const;
  const std::vector<double>& get_excitations() const { return excitation_; }
  const std::vector<double>& get_quantum_yields() const { return qy_; }
  const std::vector<double>& get_lifetimes() const { return lifetime_; }
  const std::vector<double>& get_accepts() const { return accepts_; }

  //! A transition rate. `light`: scaled by the excitation power.
  //! `hidden_state >= 0`: only in that (discrete) hidden state; a rate set
  //! for all hidden states and one set for a specific one add.
  void set_rate(const std::string& source, const std::string& target, double rate,
                bool light = false, int hidden_state = -1);
  //! True when a rate is conditioned on the hidden state.
  bool get_is_conditioned() const;

  //! Start populations at the beginning of a burst (normalised internally).
  /*! By default the stationary distribution of the dye's own rates in each
      hidden state; with an absorbing state (bleaching) that is not what a
      burst starts from, so give the weights explicitly. */
  void set_initial(const std::vector<double>& weights);
  void set_initial_stationary() { initial_.clear(); }
  bool get_initial_is_stationary() const { return initial_.empty(); }
  const std::vector<double>& get_initial() const { return initial_; }

  //! The dye's generator `K[target, source]` in `hidden_state` (-1: the
  //! unconditioned rates) at excitation `power`.
  std::vector<double> get_generator(double power = 1.0, int hidden_state = -1) const;
  //! Its start distribution in `hidden_state`.
  std::vector<double> get_start(double power = 1.0, int hidden_state = -1) const;

  //! Parameters, natural scale, prefixed with the dye's name:
  //! `lifetime[s]`, `quantum_yield[s]`, `excitation[s]`, `accepts[s]`,
  //! `rate[a->b]` (per declared rate, with `|light` and `@h` suffixes),
  //! `initial[s]` (explicit start weights only).
  std::vector<std::string> get_parameter_names() const;
  std::vector<double> get_parameter_values() const;
  void set_parameter_values(const std::vector<double>& values);
  std::vector<int> get_parameter_transforms() const;
  std::vector<int> get_parameter_kinds() const;

  IMP_SHOWABLE_INLINE(FRETDye, out << "FRETDye(" << name_ << ", " << names_.size()
                                   << " states)");

 private:
  struct Rate {
    int source, target, hidden;
    bool light;
    double value;
  };
  std::string name_;
  std::vector<std::string> names_;
  std::vector<double> excitation_, qy_, lifetime_, accepts_, initial_;
  std::vector<Rate> rates_;
};
IMP_VALUES(FRETDye, FRETDyes);

//! The light path of one measurement: pulses, channels, microtime axis.
/*! Built from the two crosstalk matrices (PhotophysicsCrosstalkMatrix.h):
    the **excitation** matrix, rows = pulses, columns `donor`/`acceptor`,
    the rate (per macrotime unit) at which a pulse train excites a fully
    excitable chromophore -- a PIE/ALEX acceptor pulse is a row with an
    `acceptor` entry; and the **emission** matrix, rows `donor`/`acceptor`,
    columns = detection channels, the fraction of emitted photons detected in
    each channel (spectral crosstalk included; quantum yields are per dye
    state, not here). Per channel a gain (a parameter) multiplies its column.

    Microtimes: `n_bins` bins of `bin_width` ns spanning the excitation
    period; `n_bins = 1` switches microtimes off (only channel and macrotime
    are used). Each (pulse, channel) has an instrument response on the bin
    axis, placed where that pulse arrives (default: all in bin 0); it is
    taken as uniform within each bin, and decays are integrated exactly over
    each bin with periodic wrap-around of the tails. Background per channel:
    a rate and a microtime density (uniform by default).

    Not modelled: pile-up, detector dead time, afterpulsing. The molecule's
    state is taken as constant within one excitation cycle (ns against
    us-ms); faster exchange is FRETExchange's domain. */
class IMPBFFEXPORT FRETInstrument {
 public:
  //! One donor pulse (rate 1), channels `green`/`red` with no crosstalk, no
  //! microtimes.
  FRETInstrument();
  FRETInstrument(const PhotophysicsCrosstalkMatrix& excitation,
                 const PhotophysicsCrosstalkMatrix& emission, int n_bins = 1,
                 double bin_width = 0.1);
  int get_n_pulses() const { return static_cast<int>(pulses_.size()); }
  int get_n_channels() const { return static_cast<int>(channels_.size()); }
  const std::vector<std::string>& get_pulses() const { return pulses_; }
  const std::vector<std::string>& get_channels() const { return channels_; }
  int get_n_bins() const { return n_bins_; }
  double get_bin_width() const { return bin_width_; }
  bool get_use_microtime() const { return n_bins_ > 1; }
  //! `excitation[pulse][chromophore]`, chromophore 0 donor, 1 acceptor.
  double get_excitation(int pulse, int chromophore) const;
  //! `emission[chromophore][channel]`.
  double get_emission(int chromophore, int channel) const;

  //! Instrument response of `pulse` in `channel` (n_bins values, >= 0,
  //! normalised internally).
  void set_irf(int pulse, int channel, const std::vector<double>& irf);
  std::vector<double> get_irf(int pulse, int channel) const;
  void set_gain(int channel, double gain);
  double get_gain(int channel) const { return gain_.at(channel); }
  //! Background rate of `channel` and its microtime density (empty: uniform).
  void set_background(int channel, double rate,
                      const std::vector<double>& density = std::vector<double>());
  double get_background(int channel) const { return bg_.at(channel); }
  std::vector<double> get_background_density(int channel) const;

  //! Parameters, natural scale: `excitation[pulse,chromophore]`,
  //! `emission[chromophore,channel]`, `gain[channel]`, `background[channel]`.
  std::vector<std::string> get_parameter_names() const;
  std::vector<double> get_parameter_values() const;
  void set_parameter_values(const std::vector<double>& values);

  IMP_SHOWABLE_INLINE(FRETInstrument,
                      out << "FRETInstrument(" << pulses_.size() << " pulses, "
                          << channels_.size() << " channels, " << n_bins_ << " bins)");

 private:
  std::vector<std::string> pulses_, channels_;
  std::vector<double> exc_, em_;  // [pulse*2 + chrom], [chrom*C + channel]
  int n_bins_ = 1;
  double bin_width_ = 0.1;
  std::vector<std::vector<double> > irf_;  // [pulse*C + channel]
  std::vector<double> gain_, bg_;
  std::vector<std::vector<double> > bg_density_;
};
IMP_VALUES(FRETInstrument, FRETInstruments);

//! One FRET label pair measured on the molecule: its dyes, how its distance
//! follows the hidden state, and (later) its instrument and bursts.
/*! The map from hidden state to donor-acceptor distance:
    - discrete hidden process: per conformer a mean distance (a parameter)
      and a fixed distribution of offsets around it (the dye-linker spread,
      e.g. from two accessible-volume clouds);
    - landscape: a natural cubic spline `r(q)` on `n_map_knots` uniform knots
      over the process's `q` range (knot values are parameters; the curve may
      be non-monotonic), and a Gaussian linker spread `sigma` (a parameter,
      fixed by default), integrated by Gauss-Hermite quadrature. */
class IMPBFFEXPORT FRETMeasurement {
 public:
  FRETMeasurement(const std::string& name = "pair", const FRETDye& donor = FRETDye("donor"),
                  const FRETDye& acceptor = FRETDye("acceptor"),
                  double forster_radius = 52.0);
  const std::string& get_name() const { return name_; }
  const FRETDye& get_donor() const { return donor_; }
  const FRETDye& get_acceptor() const { return acceptor_; }
  void set_donor(const FRETDye& d) { donor_ = d; }
  void set_acceptor(const FRETDye& a) { acceptor_ = a; }
  double get_forster_radius() const { return r0_; }
  void set_forster_radius(double r0);
  //! The donor lifetime `R0` refers to (`k_FRET = (R0/r)^6 / tau0`); by
  //! default (<= 0) the lifetime of the donor's first state.
  void set_reference_lifetime(double tau0) { tau0_ = tau0; }
  double get_reference_lifetime() const;
  void set_instrument(const FRETInstrument& instrument) { instrument_ = instrument; }
  const FRETInstrument& get_instrument() const { return instrument_; }
  //! Excitation power multiplying the dyes' light-driven rates (default 1).
  void set_power(double power);
  double get_power() const { return power_; }

  //! Discrete map: conformer `state` at `mean`, with `offsets`/`weights`
  //! around it (empty: a single distance).
  void set_state_distance(int state, double mean,
                          const std::vector<double>& offsets = std::vector<double>(),
                          const std::vector<double>& weights = std::vector<double>());
  //! Landscape map: knot values of `r(q)` and the linker spread.
  void set_distance_map(const std::vector<double>& knot_values, double spread = 0.0);
  const std::vector<double>& get_state_means() const { return means_; }
  const std::vector<double>& get_map_values() const { return map_; }
  double get_spread() const { return spread_; }
  //! Distance quadrature of hidden state `h`: `[r_0, w_0, r_1, w_1, ...]`.
  std::vector<double> get_state_distances(const FRETHiddenProcess& process, int h) const;

  //! The KineticNetwork of `(hidden, donor, acceptor)` for this pair.
  KineticNetwork get_network(const FRETHiddenProcess& process) const;
  //! Its joint generator `K[target, source]`, `n x n` row-major.
  std::vector<double> get_generator(const FRETHiddenProcess& process) const;
  //! The start distribution before detection weighting: the hidden
  //! stationary distribution times each dye's start in that hidden state.
  std::vector<double> get_start(const FRETHiddenProcess& process) const;
  //! The stationary distribution of the joint generator. Without bleaching
  //! and with dyes that do not depend on the hidden state it equals
  //! get_start(); with bleaching it is the all-bleached absorbing state,
  //! which is why a burst does not start from it.
  std::vector<double> get_joint_stationary(const FRETHiddenProcess& process) const;
  //! Detection rates `lambda_c(s)` per channel and joint state, background
  //! included, row-major `C x n`: the distance distribution of each hidden
  //! state is integrated as a mixture over its quadrature.
  std::vector<double> get_detection_rates(const FRETHiddenProcess& process) const;
  //! The photon factor `sum_j w_j lambda_c(r_j) f_c(t | r_j) + beta_c b_c(t)`
  //! per channel, microtime bin and joint state, `C x n_bins x n`: expected
  //! photons per macrotime unit in that channel and bin (the mixture over
  //! the distance distribution, not a product of averages). Summed over the
  //! bins it is get_detection_rates().
  std::vector<double> get_emission(const FRETHiddenProcess& process) const;
  //! Number of joint states `n_hidden * n_donor * n_acceptor`.
  int get_n_states(const FRETHiddenProcess& process) const;
  //! `(hidden, donor, acceptor)` of joint state `s`.
  std::vector<int> get_state(const FRETHiddenProcess& process, int s) const;

  //! Map, dye and instrument parameters, natural scale, prefixed
  //! `<pair>.`. The map is `mean[h]` (discrete) or `map[k]`, `spread`.
  std::vector<std::string> get_parameter_names(const FRETHiddenProcess& process) const;
  std::vector<double> get_parameter_values(const FRETHiddenProcess& process) const;
  void set_parameter_values(const FRETHiddenProcess& process, const std::vector<double>& v);
  std::vector<int> get_parameter_transforms(const FRETHiddenProcess& process) const;
  std::vector<int> get_parameter_kinds(const FRETHiddenProcess& process) const;

  IMP_SHOWABLE_INLINE(FRETMeasurement, out << "FRETMeasurement(" << name_ << ")");

 private:
  void check_process(const FRETHiddenProcess& process) const;
  std::string name_;
  FRETDye donor_, acceptor_;
  FRETInstrument instrument_;
  double r0_ = 52.0, tau0_ = -1.0, power_ = 1.0;
  // discrete map
  std::vector<double> means_;
  std::vector<std::vector<double> > offsets_, weights_;
  // landscape map
  std::vector<double> map_;
  double spread_ = 0.0;
};
IMP_VALUES(FRETMeasurement, FRETMeasurements);

//! How photon arrival times enter a measurement's likelihood.
enum FRETArrivalModel {
  //! Killing between photons with `Lambda_tot(s)`: the count rate is
  //! information. Valid when each state's brightness is constant
  //! (immobilised molecules, the paper's traces).
  FRET_ARRIVAL_FULL = 0,
  //! Arrival times taken as given (Gopich-Szabo / H2MM): between photons the
  //! generator alone, each photon the normalised probability of its channel
  //! and microtime, `E_c(t|s) / Lambda_tot(s)`. Insensitive to the brightness
  //! modulation of a molecule crossing the focus; the default for bursts.
  FRET_ARRIVAL_CONDITIONAL = 1
};

//! Photons of one measurement, cut into segments.
/*! Arrays: macrotime (the rate unit's time), channel, microtime bin (may be
    empty when the instrument has no microtimes), and segments as inclusive
    first/last photon indices. A burst selection is one segmentation; a whole
    immobilised trace, or the whole stream, are others. Each segment is scored
    independently from the start distribution. */
class IMPBFFEXPORT FRETPhotonData {
 public:
  FRETPhotonData() {}
  FRETPhotonData(const std::vector<double>& macrotimes, const std::vector<int>& channels,
                 const std::vector<int>& microtimes, const std::vector<int>& segment_starts,
                 const std::vector<int>& segment_stops);
  const std::vector<double>& get_macrotimes() const { return t_; }
  const std::vector<int>& get_channels() const { return c_; }
  const std::vector<int>& get_microtimes() const { return b_; }
  const std::vector<int>& get_segment_starts() const { return start_; }
  const std::vector<int>& get_segment_stops() const { return stop_; }
  int get_n_segments() const { return static_cast<int>(start_.size()); }
  int get_n_photons() const { return static_cast<int>(t_.size()); }
  //! Longest gap between consecutive photons of one segment.
  double get_max_gap() const;

  IMP_SHOWABLE_INLINE(FRETPhotonData, out << "FRETPhotonData(" << t_.size() << " photons, "
                                          << start_.size() << " segments)");

 private:
  std::vector<double> t_;
  std::vector<int> c_, b_, start_, stop_;
};
IMP_VALUES(FRETPhotonData, FRETPhotonDataList);

//! One hidden process, observed by several FRET measurements.
/*! The log-likelihood is the sum over measurements and their segments. Each
    measurement has its own arrival model (default conditional) and start:
    by default the product start (FRETMeasurement::get_start) weighted by
    detection, `pi * Lambda_tot` normalised, because a data-selected segment
    begins with a detection; the plain start is for a segment whose start
    time the data did not choose. Detection weighting corrects only for the
    first photon: the burst search's selection of bright, long stretches is a
    bias the likelihood does not model (it would need the selection rule).

    Between photons the state is propagated by uniformization of the sparse
    joint generator (adaptive Poisson truncation at 1e-14 tail mass, gaps
    split into chunks of at most `q tau = 64`), which stays exact for
    non-symmetric, block-triangular (bleaching) generators. */
class IMPBFFEXPORT FRETNetworkModel {
 public:
  explicit FRETNetworkModel(const FRETHiddenProcess& process = FRETHiddenProcess(2));
  const FRETHiddenProcess& get_process() const { return process_; }
  void set_process(const FRETHiddenProcess& process) { process_ = process; }

  //! Add a measurement with its photons; returns its index.
  int add_measurement(const FRETMeasurement& measurement, const FRETPhotonData& data);
  int get_n_measurements() const { return static_cast<int>(meas_.size()); }
  const FRETMeasurement& get_measurement(int i) const { return meas_.at(i); }
  void set_measurement(int i, const FRETMeasurement& m) { meas_.at(i) = m; }
  const FRETPhotonData& get_data(int i) const { return data_.at(i); }
  void set_arrival_model(int i, int model);
  int get_arrival_model(int i) const { return arrival_.at(i); }
  void set_detection_weighted_start(int i, bool on) { detection_start_.at(i) = on; }
  bool get_detection_weighted_start(int i) const { return detection_start_.at(i); }
  //! Start from the joint generator's stationary distribution instead of the
  //! product start (no bleaching; see FRETMeasurement::get_joint_stationary).
  void set_joint_stationary_start(int i, bool on) { joint_start_.at(i) = on; }

  //! Total log-likelihood at the current parameter values.
  double log_likelihood() const;
  //! Log-likelihood of each segment of measurement `i`.
  std::vector<double> segment_log_likelihoods(int i) const;
  //! Posterior time fractions of `factor` (`hidden`, `donor`, `acceptor`, or
  //! `joint`) over each segment of measurement `i`, between its first and
  //! last photon: row-major `n_segments x n_factor_states`, one row per
  //! segment -- columns a burst companion can hold.
  std::vector<double> segment_occupancies(int i, const std::string& factor) const;
  //! Posterior probabilities of the joint state at each photon of segment
  //! `segment`: row-major `n_photons x n_states`.
  std::vector<double> photon_posteriors(int i, int segment) const;

  IMP_SHOWABLE_INLINE(FRETNetworkModel, out << "FRETNetworkModel(" << meas_.size()
                                            << " measurements)");

 private:
  FRETHiddenProcess process_;
  std::vector<FRETMeasurement> meas_;
  std::vector<FRETPhotonData> data_;
  std::vector<int> arrival_;
  std::vector<char> detection_start_, joint_start_;
};
IMP_VALUES(FRETNetworkModel, FRETNetworkModels);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_FRETNETWORK_H
