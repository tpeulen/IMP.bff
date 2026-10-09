/** \example bff/counterfactual/twin_factor_graph.cpp

    Direction on the factor graph: mechanisms, do() and the twin network.

    The allosteric model as an InferenceFactorGraph: two mechanism factors
    (hinge from effector and fluctuation, active site from all three), two
    FRET likelihoods. Declaring each mechanism's child leaves every structural
    answer (cliques, treewidth, blocks) unchanged and adds the causal ones:
    parents, the mutilated graph of a hinge lock, and the twin network in
    which a counterfactual world shares the hidden fluctuation W with the
    factual one.
*/

#include <IMP/bff/InferenceFactorGraph.h>
#include <IMP/flags.h>

#include <cstdio>
#include <iostream>

int main(int argc, char* argv[]) {
  IMP::setup_from_argv(argc, argv, "Directed factors, interventions and twin networks.");
  IMP::bff::InferenceFactorGraph g;
  g.add_variable("L", "effector", 0, -1, 1, "intervention");
  g.add_variable("W", "fluctuation", 1, -1, 1, "exogenous");
  g.add_variable("M", "hinge", 2, -1, 1, "hinge");
  g.add_variable("Y", "active site", 3, -1, 1, "active_site");
  g.add_factor("f_M", IMP::bff::INFERENCE_FACTOR_PRIOR, {"L", "W", "M"});
  g.add_factor("f_Y", IMP::bff::INFERENCE_FACTOR_PRIOR, {"L", "W", "M", "Y"});
  g.add_factor("pair1", IMP::bff::INFERENCE_FACTOR_LIKELIHOOD, {"M"}, 0, 100);
  g.add_factor("pair2", IMP::bff::INFERENCE_FACTOR_LIKELIHOOD, {"Y"}, 1, 100);
  const int width_undirected = g.get_treewidth();
  g.set_factor_children("f_M", {"M"});
  g.set_factor_children("f_Y", {"Y"});

  std::cout << "== the model ==\n" << g.describe();
  std::cout << "parents of Y : ";
  for (const auto& p : g.get_parents("Y")) std::cout << p << " ";
  std::cout << "\n";

  std::cout << "\n== do(M): hinge locked ==\n" << g.get_intervened({"M"}).describe();

  IMP::bff::InferenceFactorGraph twin = g.get_twin({"L"});
  std::cout << "\n== twin network for do(L) ==\n" << twin.describe();
  std::cout << "parents of Y@cf : ";
  for (const auto& p : twin.get_parents("Y@cf")) std::cout << p << " ";
  std::cout << "\n";
  return (g.get_treewidth() == width_undirected && twin.get_is_acyclic()) ? 0 : 1;
}
