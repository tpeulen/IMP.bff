/** \example bff/counterfactual/network_site_bias.cpp

    Can a correct analysis of a FRET network give the wrong structure?

    Six labelling sites move along a one-parameter conformational path (40
    candidate structures); all 15 distances are measured with 2 A error bars. The dye at site 3 misbehaves
    and adds +6 A to every pair it is in. The usual chi^2 analysis picks a
    wrong structure with a reduced chi^2 that raises no alarm.
    CounterfactualDistanceNetwork makes the hidden assumption explicit: each
    site gets a bias with a prior, the biases are abducted from the network's
    redundancy, and three counterfactual questions are answered:
      - do(b = 0): which structure would ideal dyes have given?
      - do(b_s = 0) per site: which dye does the conclusion hinge on?
      - replay: had the structure been f', would this data set have shown it?
*/

#include <IMP/bff/CounterfactualDistanceNetwork.h>
#include <IMP/flags.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
int argmax(const std::vector<double>& v) {
  int best = 0;
  for (int i = 1; i < static_cast<int>(v.size()); ++i)
    if (v[i] > v[best]) best = i;
  return best;
}
}  // namespace

int main(int argc, char* argv[]) {
  IMP::setup_from_argv(argc, argv, "Site-bias abduction, hinge and replay for a distance network.");
  const int n_sites = 6, n_cand = 40, f_true = 25, bad = 3;
  // Site positions (A) at the start of the path and their displacement per
  // unit of the path coordinate; site 0 barely moves.
  const double start[6][3] = {{8.9693, 37.7669, -44.6583}, {42.1653, -1.1829, -19.9995},
                              {-20.0739, -27.0704, -5.5911}, {20.8471, 14.6016, 15.9572},
                              {-42.3702, -39.2741, 38.8451}, {24.2221, 54.5804, 30.2454}};
  const double dir[6][3] = {{-0.2049, 0.2571, 0.1257}, {0.2148, -0.8199, 0.0028},
                            {-0.1470, 0.8925, 0.1245}, {1.5847, -0.7745, 0.4711},
                            {0.4988, -0.8643, -1.1813}, {1.0136, 0.5174, -0.1625}};
  const double noise[15] = {0.3456, 0.8216, 0.3304, -1.3032, 0.9054, 0.4464, -0.5370, 0.5811,
                            0.3646, 0.2941, 0.0284, 0.5467, -0.7365, -0.1629, -0.4821};
  std::vector<int> pair_sites;
  for (int i = 0; i < n_sites; ++i)
    for (int j = i + 1; j < n_sites; ++j) pair_sites.insert(pair_sites.end(), {i, j});
  const int n_pairs = static_cast<int>(pair_sites.size() / 2);
  std::vector<double> model;
  for (int f = 0; f < n_cand; ++f) {
    const double t = 12.0 * f / (n_cand - 1);
    for (int p = 0; p < n_pairs; ++p) {
      double d2 = 0.0;
      for (int k = 0; k < 3; ++k) {
        const int i = pair_sites[2 * p], j = pair_sites[2 * p + 1];
        const double d = (start[i][k] + t * dir[i][k]) - (start[j][k] + t * dir[j][k]);
        d2 += d * d;
      }
      model.push_back(std::sqrt(d2));
    }
  }
  // The measurement: truth f_true, +6 A at site `bad`, 1 A noise.
  std::vector<double> y(n_pairs), sigma(n_pairs, 2.0);
  for (int p = 0; p < n_pairs; ++p) {
    const int i = pair_sites[2 * p], j = pair_sites[2 * p + 1];
    y[p] = model[f_true * n_pairs + p] + 6.0 * ((i == bad) + (j == bad)) + noise[p];
  }

  IMP::bff::CounterfactualDistanceNetwork net(model, n_cand, pair_sites, n_sites);
  net.set_measurement(y, sigma);
  net.set_site_bias_prior(std::vector<double>(n_sites, 3.0));

  const std::vector<double> standard = net.get_candidate_posterior(false);
  const std::vector<double> aware = net.get_candidate_posterior(true);
  const std::vector<double> chi2 = net.get_chi2();
  const int f_std = argmax(standard), f_aware = argmax(aware);
  std::printf("truth                : structure %d, +6 A at site %d\n", f_true, bad);
  std::printf("usual chi2 analysis  : structure %d, P(truth) = %.3g, reduced chi2 = %.2f\n", f_std,
              standard[f_true], chi2[f_std] / (n_pairs - 1));
  std::printf("biases abducted      : structure %d, P(truth) = %.2f\n", f_aware, aware[f_true]);
  const std::vector<double> b = net.get_abducted_site_biases(), sd = net.get_abducted_site_bias_sd();
  for (int s = 0; s < n_sites; ++s) std::printf("  site %d bias %+5.2f +- %.2f A\n", s, b[s], sd[s]);

  const std::vector<double> ideal = net.get_ideal_measurement();
  std::printf("do(b = 0)            : structure %d\n", argmax(net.get_standard_posterior(ideal)));
  const std::vector<double> hinge = net.get_hinge_posteriors();
  for (int s = 0; s < n_sites; ++s) {
    const std::vector<double> row(hinge.begin() + s * n_cand, hinge.begin() + (s + 1) * n_cand);
    std::printf("  do(b_%d = 0)       : structure %d\n", s, argmax(row));
  }
  const std::vector<int> replay = net.get_counterfactual_replay();
  const std::vector<int> fresh = net.get_fresh_noise_replay(5);
  int blind = 0, blind_fresh = 0;
  for (int f = 0; f < n_cand; ++f) {
    blind += std::abs(replay[f] - f) > 2;
    blind_fresh += std::abs(fresh[f] - f) > 2;
  }
  std::printf("replay               : %d of %d structures misread with this experiment's biases, "
              "%d with fresh noise\n", blind, n_cand, blind_fresh);
  return std::abs(f_aware - f_true) <= 1 ? 0 : 1;
}
