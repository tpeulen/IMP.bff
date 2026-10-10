/**
 * \file GraphNodeRegistry.cpp
 * \brief Node types by name, so a graph can be described rather than built.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/GraphNodeRegistry.h>
#include <IMP/bff/Registry.h>

#include <IMP/bff/Convolution.h>
#include <IMP/bff/FCS.h>
#include <IMP/bff/AcceptorDensityDecay.h>
#include <IMP/bff/DiscreteDistances.h>
#include <IMP/bff/FRETSpectrumNode.h>
#include <IMP/bff/FitChiSquared.h>
#include <IMP/bff/FitJointChiSquared.h>
#include <IMP/bff/GaussianDistances.h>
#include <IMP/bff/GeneralizedNormalCurve.h>
#include <IMP/bff/LifetimeSpectrumMixture.h>
#include <IMP/bff/MaxEntSpectrum.h>
#include <IMP/bff/GraphExpression.h>
#include <IMP/bff/PhotophysicsAnisotropySpectrumNode.h>
#include <IMP/bff/KineticSchemeNode.h>
#include <IMP/bff/PhotophysicsLifetimeSpectrumNode.h>
#include <IMP/bff/PhotophysicsTransferKineticsNode.h>
#include <IMP/bff/PolymerDistances.h>
#include <IMP/bff/SpectrumGrid.h>
#include <IMP/bff/TabulatedDistances.h>
#include <IMP/bff/TCSPCDecay.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

namespace {

typedef std::map<std::string, GraphNodeRegistry::Factory> FactoryMap;

//! A node that needs no post-construction port building.
template <typename T>
std::shared_ptr<GraphNode> plain(const std::string& name) {
  return std::make_shared<T>(name);
}

//! A node whose ports can only be built once a shared_ptr owns it.
template <typename T>
std::shared_ptr<GraphNode> with_ports(const std::string& name) {
  std::shared_ptr<T> node = std::make_shared<T>(name);
  node->build_ports();
  return node;
}

void install_builtin_types(FactoryMap& factories) {
  factories["GraphNode"] = [](const std::string& name) {
    return std::make_shared<GraphNode>(name);
  };
  factories["GraphExpression"] = plain<GraphExpression>;
  factories["Convolution"] = plain<Convolution>;
  factories["GeneralizedNormalCurve"] = plain<GeneralizedNormalCurve>;
  factories["LifetimeSpectrumMixture"] = plain<LifetimeSpectrumMixture>;
  factories["TCSPCDecay"] = plain<TCSPCDecay>;
  factories["FitChiSquared"] = plain<FitChiSquared>;
  factories["FitJointChiSquared"] = plain<FitJointChiSquared>;
  factories["GaussianDistances"] = plain<GaussianDistances>;
  factories["DiscreteDistances"] = plain<DiscreteDistances>;
  factories["AcceptorDensityDecay"] = plain<AcceptorDensityDecay>;
  factories["PolymerDistances"] = plain<PolymerDistances>;
  factories["TabulatedDistances"] = plain<TabulatedDistances>;
  factories["SpectrumGrid"] = plain<SpectrumGrid>;
  factories["MaxEntSpectrum"] = plain<MaxEntSpectrum>;
  factories["PhotophysicsLifetimeSpectrumNode"] =
      plain<PhotophysicsLifetimeSpectrumNode>;
  factories["KineticSchemeNode"] = plain<KineticSchemeNode>;
  factories["PhotophysicsTransferKineticsNode"] =
      with_ports<PhotophysicsTransferKineticsNode>;
  factories["PhotophysicsAnisotropySpectrumNode"] =
      plain<PhotophysicsAnisotropySpectrumNode>;
  factories["FCSMdfCurve"] = with_ports<FCSMdfCurve>;
  factories["FCSSaturationCurve"] = with_ports<FCSSaturationCurve>;
  factories["FRETSpectrumNode"] = with_ports<FRETSpectrumNode>;
}

// The node types, as registry entries (category `graph_node`), beside the factories that build them.
// registry_json() serves them; GraphNodeRegistry::get_registered_types() and create() stay the
// factory side. A test holds the two sets equal (test/test_registry.py).
const int registered_graph_nodes = [] {
  const char* const entries[][2] = {
    {"GraphNode", R"JSON({"label": "A plain node", "summary": "The base node: named input and output ports and no computation of its own.", "description": "", "params_schema": {}, "api": ["GraphNode"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"GraphExpression", R"JSON({"label": "Arithmetic expression", "summary": "A vectorised arithmetic expression over input ports, compiled once and evaluated as a node.", "description": "", "params_schema": {}, "api": ["GraphExpression"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"Convolution", R"JSON({"label": "Convolution with a response", "summary": "A sampled curve convolved with a measured instrument response.", "description": "", "params_schema": {}, "api": ["Convolution"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"GeneralizedNormalCurve", R"JSON({"label": "Generalized-normal peak", "summary": "A sampled generalized-normal peak.", "description": "", "params_schema": {}, "api": ["GeneralizedNormalCurve"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"LifetimeSpectrumMixture", R"JSON({"label": "Lifetime spectrum mixture", "summary": "Lifetime spectra of several species, weighted by fractions and joined.", "description": "", "params_schema": {}, "api": ["LifetimeSpectrumMixture"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"TCSPCDecay", R"JSON({"label": "TCSPC decay", "summary": "A time-correlated single-photon-counting decay: a lifetime spectrum convolved with the instrument response, with scatter and background.", "description": "", "params_schema": {}, "api": ["TCSPCDecay"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"FitChiSquared", R"JSON({"label": "Data misfit", "summary": "The misfit of a model curve against one dataset.", "description": "", "params_schema": {}, "api": ["FitChiSquared"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"FitJointChiSquared", R"JSON({"label": "Joint data misfit", "summary": "One misfit over several datasets.", "description": "", "params_schema": {}, "api": ["FitJointChiSquared"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"GaussianDistances", R"JSON({"label": "Gaussian distance distribution", "summary": "Distance components as (generalized) Gaussians, output as interleaved (weight, distance) pairs.", "description": "", "params_schema": {}, "api": ["GaussianDistances"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"DiscreteDistances", R"JSON({"label": "Discrete distances", "summary": "A distance distribution of a few discrete distances.", "description": "", "params_schema": {}, "api": ["DiscreteDistances"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"AcceptorDensityDecay", R"JSON({"label": "Acceptor-density quenched decay", "summary": "The donor decay quenched by a density of acceptors.", "description": "", "params_schema": {}, "api": ["AcceptorDensityDecay"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"PolymerDistances", R"JSON({"label": "Polymer distance distribution", "summary": "End-to-end distance distributions of polymer models (worm-like chain and others) from contour and persistence lengths.", "description": "", "params_schema": {}, "api": ["PolymerDistances"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"TabulatedDistances", R"JSON({"label": "Tabulated distance distributions", "summary": "Given distance distributions on a common axis, mixed by fractions.", "description": "", "params_schema": {}, "api": ["TabulatedDistances"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"SpectrumGrid", R"JSON({"label": "Fixed spectrum grid", "summary": "A fixed grid as a spectrum of unit amplitudes.", "description": "", "params_schema": {}, "api": ["SpectrumGrid"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"MaxEntSpectrum", R"JSON({"label": "Maximum-entropy spectrum", "summary": "Maximum-entropy amplitudes over a decay basis.", "description": "", "params_schema": {}, "api": ["MaxEntSpectrum"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"KineticSchemeNode", R"JSON({"label": "Kinetic scheme", "summary": "A chain of interconverting states: stationary populations as a lifetime spectrum for TCSPC, and the kinetic correlation factor for FCS.", "description": "", "params_schema": {}, "api": ["KineticSchemeNode"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"PhotophysicsLifetimeSpectrumNode", R"JSON({"label": "Lifetime spectrum", "summary": "Amplitudes and lifetimes of each species as an interleaved lifetime spectrum.", "description": "", "params_schema": {}, "api": ["PhotophysicsLifetimeSpectrumNode"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"PhotophysicsTransferKineticsNode", R"JSON({"label": "Transfer kinetics", "summary": "Energy-transfer kinetics between two chromophores.", "description": "", "params_schema": {}, "api": ["PhotophysicsTransferKineticsNode"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"PhotophysicsAnisotropySpectrumNode", R"JSON({"label": "Anisotropy spectrum", "summary": "A lifetime spectrum split into parallel and perpendicular detection with r0, g, l1, l2 and rotational correlation times.", "description": "", "params_schema": {}, "api": ["PhotophysicsAnisotropySpectrumNode"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"FCSMdfCurve", R"JSON({"label": "FCS curve, MDF shape", "summary": "An FCS correlation curve for a molecule detection function beyond the closed forms.", "description": "", "params_schema": {}, "api": ["FCSMdfCurve"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"FCSSaturationCurve", R"JSON({"label": "FCS curve, saturated excitation", "summary": "An FCS correlation curve under saturated excitation.", "description": "", "params_schema": {}, "api": ["FCSSaturationCurve"], "factory": "GraphNodeRegistry.create"})JSON"},
    {"FRETSpectrumNode", R"JSON({"label": "FRET lifetime spectrum", "summary": "A donor lifetime spectrum quenched by FRET over a distance distribution, with kappa-squared handling.", "description": "", "params_schema": {}, "api": ["FRETSpectrumNode"], "factory": "GraphNodeRegistry.create"})JSON"},
  };
  for (const auto& e : entries) register_algorithm_json("graph_node", e[0], e[1]);
  return 0;
}();

//! The one registry, installed on first use rather than at static init.
FactoryMap& factories() {
  static FactoryMap* the_factories = [] {
    FactoryMap* created = new FactoryMap();
    install_builtin_types(*created);
    return created;
  }();
  return *the_factories;
}

}  // namespace

void GraphNodeRegistry::register_type(const std::string& type,
                                      Factory factory) {
  if (type.empty()) {
    throw std::domain_error("a graph node type needs a name");
  }
  if (!factory) {
    throw std::domain_error("graph node type '" + type +
                            "' needs a factory that builds it");
  }
  factories()[type] = factory;
  // A type registered at run time is in the registry as well; a key already taken keeps its entry.
  register_algorithm_json("graph_node", type,
                          "{\"label\": \"" + type + "\", \"summary\": \"A node type registered at run time.\", \"provider\": \"runtime\", \"factory\": \"GraphNodeRegistry.create\"}");
}

bool GraphNodeRegistry::has_type(const std::string& type) {
  return factories().find(type) != factories().end();
}

std::vector<std::string> GraphNodeRegistry::get_registered_types() {
  std::vector<std::string> names;
  const FactoryMap& known = factories();
  for (FactoryMap::const_iterator it = known.begin(); it != known.end(); ++it) {
    names.push_back(it->first);
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::shared_ptr<GraphNode> GraphNodeRegistry::create(const std::string& type,
                                                     const std::string& name) {
  const FactoryMap& known = factories();
  FactoryMap::const_iterator found = known.find(type);
  if (found == known.end()) {
    std::ostringstream message;
    message << "unknown graph node type '" << type << "'; registered types are";
    const std::vector<std::string> names = get_registered_types();
    for (std::size_t i = 0; i < names.size(); ++i) {
      message << (i == 0 ? " " : ", ") << names[i];
    }
    throw std::domain_error(message.str());
  }
  return found->second(name.empty() ? type : name);
}

IMPBFF_END_NAMESPACE
