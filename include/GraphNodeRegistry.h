/**
 *  \file IMP/bff/GraphNodeRegistry.h
 *  \brief Node types by name, so a graph can be described rather than built.
 *
 *  A `GraphNode` subclass is a numerical kernel: it knows how to turn its
 *  input ports into its output ports and nothing else. *Which* kernels a
 *  model is made of, how they are wired, and which of their ports follow a
 *  fitted parameter are not properties of any kernel -- they are the model,
 *  and a model is a thing this library should be able to read rather than
 *  be recompiled for.
 *
 *  This registry is the one piece that cannot live in data: something has to
 *  map the name `"TCSPCDecay"` onto the constructor. It is deliberately the
 *  *only* such piece. Everything else a description needs -- the expression
 *  text, the quadrature resolution, the number of lifetimes, which port
 *  follows which canonical parameter -- reaches the node through
 *  GraphNode::configure(), which each node implements beside its own
 *  setters. Adding a model family is then a file; adding a numerical kernel
 *  is a class plus one line here.
 *
 *  Registration is lazy and explicit rather than static-initialiser driven:
 *  the built-in types are installed on first use, so there is no order to
 *  get wrong across translation units.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_GRAPHNODEREGISTRY_H
#define IMPBFF_GRAPHNODEREGISTRY_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/GraphNode.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Constructs node types by their registered name.
class IMPBFFEXPORT GraphNodeRegistry {
 public:
  //! Builds one ready node: constructed, owned, and with its ports built.
  /*! Ports attach to their node through a `weak_ptr`, so a node that owns
      ports can only build them once a `shared_ptr` holds it. A factory
      returning an owned node is therefore the only correct shape; see
      FRETSpectrumNode::build_ports(). */
  typedef std::function<std::shared_ptr<GraphNode>(const std::string& name)>
      Factory;

  //! Install a type, replacing any factory already under that name.
  static void register_type(const std::string& type, Factory factory);

  //! Whether a type of this name can be created.
  static bool has_type(const std::string& type);

  //! Every registered type name, sorted.
  static std::vector<std::string> get_registered_types();

  //! Build one node of a registered type.
  /*! \throws std::domain_error naming the unknown type and listing what is
      registered. A description that misspells a type must fail loudly: the
      alternative is a node that evaluates to nothing and fits anything. */
  static std::shared_ptr<GraphNode> create(const std::string& type,
                                           const std::string& name = "");
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_GRAPHNODEREGISTRY_H
