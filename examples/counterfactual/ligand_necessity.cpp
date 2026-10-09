/** \example bff/counterfactual/ligand_necessity.cpp

    Conformational selection or induced fit, decided per binding event.

    A surface-immobilised molecule is watched in 0.2 ms frames: open (0) or
    closed (1). A labelled ligand binds at frame 0 and the molecule is closed
    2 ms later. Would it have closed without the ligand? Written as a
    Gumbel-max causal model, the frames determine the noise of every step;
    replaying that noise with the ligand-free transition matrix gives the
    probability of necessity. Under conformational selection (the ligand binds
    molecules that are already closed) it is small; under induced fit (the
    ligand drives closure) it is large.
*/

#include <IMP/bff/CounterfactualMarkovChain.h>
#include <IMP/flags.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
// Frame propagator of a two-state chain with rates open->closed and closed->open (1/ms).
std::vector<double> propagator(double oc, double co, double dt = 0.2) {
  const double k = oc + co, e = std::exp(-k * dt), po = co / k, pc = oc / k;
  return {po + (1 - po) * e, pc - pc * e, po - po * e, pc + (1 - pc) * e};
}
}  // namespace

int main(int argc, char* argv[]) {
  IMP::setup_from_argv(argc, argv, "Probability of necessity of a ligand, per binding event.");
  const std::vector<double> apo = propagator(0.1, 0.1);
  const std::vector<double> selection = propagator(0.1, 0.02);  // ligand stabilises closed
  const std::vector<double> induced = propagator(1.0, 0.02);    // ligand drives closure

  // Two observed events, each ending closed 2 ms after binding.
  const std::vector<int> event_cs = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};  // bound while closed
  const std::vector<int> event_if = {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};  // bound open, closed next

  IMP::bff::CounterfactualMarkovChain cs(selection, 2), fit(induced, 2);
  const double pn_cs = cs.get_counterfactual_probability(event_cs, apo, 0, 4000, 1);
  const double pn_if = fit.get_counterfactual_probability(event_if, apo, 0, 4000, 2);
  std::printf("conformational selection: P(open at 2 ms had the ligand not bound) = %.2f\n", pn_cs);
  std::printf("induced fit             : P(open at 2 ms had the ligand not bound) = %.2f\n", pn_if);

  // The counterfactual occupancy along the induced-fit event.
  const std::vector<double> occ = fit.get_counterfactual_occupancy(event_if, apo, 2000, 3);
  std::printf("induced fit, frame by frame: P(open | no ligand) =");
  for (std::size_t t = 0; t < event_if.size(); ++t) std::printf(" %.2f", occ[2 * t]);
  std::printf("\n");
  return pn_if > pn_cs ? 0 : 1;
}
