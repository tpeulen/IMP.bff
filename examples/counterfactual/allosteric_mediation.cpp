/** \example bff/counterfactual/allosteric_mediation.cpp

    Allostery as a causal question: through which path does an effector act?

    An effector L binds a regulatory site. FRET pair 1 reads a hinge distance M,
    pair 2 the active-site distance Y. L acts on Y through the hinge and
    directly, and a hidden fluctuation W moves hinge and active site together.
    The natural indirect effect -- the active site with the effector bound but
    the hinge where it would have been without it, Y(1, M(0)) -- is a
    counterfactual: it mixes two worlds of the same molecule. With the model
    written down, IMP.bff computes it exactly, for the population and for one
    observed molecule, and also shows why the obvious experiment (both pairs
    on the same molecule, regress Y on M) gets the pathway wrong while a
    hinge-locking intervention, do(M), does not.
*/

#include <IMP/bff/CausalLinearGaussian.h>
#include <IMP/flags.h>

#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char* argv[]) {
  IMP::setup_from_argv(argc, argv, "Natural direct and indirect effects of an allosteric effector.");

  // Distances in Angstrom; weights are the effect of one unit of the parent.
  IMP::bff::CausalLinearGaussian scm;
  scm.add_variable("L", 0.0, 0.5);  // effector occupancy
  scm.add_variable("W", 0.0, 1.0);  // hidden fluctuation (exogenous)
  scm.add_variable("M", 0.0, 0.5);  // hinge distance, FRET pair 1
  scm.add_variable("Y", 0.0, 0.5);  // active-site distance, FRET pair 2
  scm.add_edge("L", "M", 1.0);
  scm.add_edge("M", "Y", 0.8);
  scm.add_edge("L", "Y", 0.3);
  scm.add_edge("W", "M", 1.0);
  scm.add_edge("W", "Y", -1.2);

  // 1. The population: total = direct + indirect.
  const std::vector<double> eff = scm.get_natural_effects("L", {"M"}, "Y", 0.0, 1.0);
  std::printf("population : total %.3f A = direct %.3f + through the hinge %.3f\n", eff[0], eff[1],
              eff[2]);

  // 2. One molecule, both distances measured with the effector bound.
  const std::vector<std::string> obs = {"L", "M", "Y"};
  const std::vector<double> val = {1.0, 2.4, 0.9};
  IMP::bff::CausalGaussianWorld noise = scm.get_abducted_noise(obs, val);
  std::printf("abduction  : this molecule's hidden fluctuation W = %.2f +- %.2f\n",
              noise.get_mean("U[W]"), noise.get_sd("U[W]"));
  IMP::bff::CausalGaussianWorld cf = scm.get_counterfactual(obs, val, {"L"}, {0.0});
  std::printf("counterfact: had the effector been absent, hinge %.2f +- %.2f, active site %.2f +- %.2f\n",
              cf.get_mean("M"), cf.get_sd("M"), cf.get_mean("Y"), cf.get_sd("Y"));

  // 3. Which experiment identifies the hinge pathway? Sample the two designs.
  const unsigned int n = 200000;
  const std::vector<double> x = scm.get_samples(n, 11);  // columns L, W, M, Y
  // Design B: both pairs on the same molecules, regress Y on (1, L, M).
  double s[3][3] = {{0}}, r[3] = {0};
  for (unsigned int i = 0; i < n; ++i) {
    const double f[3] = {1.0, x[4 * i + 0], x[4 * i + 2]};
    for (int a = 0; a < 3; ++a) {
      r[a] += f[a] * x[4 * i + 3];
      for (int b = 0; b < 3; ++b) s[a][b] += f[a] * f[b];
    }
  }
  // Solve the 3x3 normal equations by Cramer's rule.
  auto det3 = [](double m[3][3]) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  };
  double m2[3][3];
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) m2[a][b] = b == 2 ? r[a] : s[a][b];
  const double slope_m = det3(m2) / det3(s);
  // Design C: lock the hinge, do(M = m), and read the slope of Y on m.
  const double y_lo = scm.get_interventional({"M"}, {0.0}).get_mean("Y");
  const double y_hi = scm.get_interventional({"M"}, {1.0}).get_mean("Y");
  // Effector -> hinge, the first factor of the indirect effect (unconfounded: L has no parents).
  const double a_hat = scm.get_interventional({"L"}, {1.0}).get_mean("M") -
                       scm.get_interventional({"L"}, {0.0}).get_mean("M");
  std::printf("design B   : same-molecule regression -> indirect effect %.2f (wrong: W confounds)\n",
              a_hat * slope_m);
  std::printf("design C   : hinge locked, do(M)      -> indirect effect %.2f (truth %.2f)\n",
              a_hat * (y_hi - y_lo), eff[2]);
  return std::abs(eff[2] - 0.8) < 1e-9 ? 0 : 1;
}
