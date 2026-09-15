/**
 * \file GraphNodeRegistry.cpp
 * \brief Node types by name, so a graph can be described rather than built.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/GraphNodeRegistry.h>

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
  factories["PhotophysicsTransferKineticsNode"] =
      with_ports<PhotophysicsTransferKineticsNode>;
  factories["PhotophysicsAnisotropySpectrumNode"] =
      plain<PhotophysicsAnisotropySpectrumNode>;
  factories["FCSMdfCurve"] = with_ports<FCSMdfCurve>;
  factories["FCSSaturationCurve"] = with_ports<FCSSaturationCurve>;
  factories["FRETSpectrumNode"] = with_ports<FRETSpectrumNode>;
}

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
