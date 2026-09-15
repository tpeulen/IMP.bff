/**\file PhotophysicsAnisotropySpectrumNode.cpp
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/PhotophysicsAnisotropySpectrumNode.h>
#include <IMP/bff/internal/NodeConfig.h>
#include "internal/SpectrumNodeHelpers.h"
#include <IMP/bff/PhotophysicsLifetimeSpectrum.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

// ------------------------------------------------------- PhotophysicsAnisotropySpectrumNode

PhotophysicsAnisotropySpectrumNode::PhotophysicsAnisotropySpectrumNode(const std::string& name) : GraphNode(name) {}

void PhotophysicsAnisotropySpectrumNode::set_number_of_rotations(int n) {
  if (n < 0) {
    throw std::domain_error(
        "PhotophysicsAnisotropySpectrumNode::set_number_of_rotations: a negative count");
  }
  if (spectrum_port_ == nullptr) {
    std::shared_ptr<GraphPort> incoming(new GraphPort(std::vector<double>(2, 1.0)));
    incoming->set_sanitize(false);
    add_input_port(spectrum_port_key(), incoming);
    spectrum_port_ = incoming.get();

    spectrum_node_detail::add_scalar_port(this, "r0", 0.38, &r0_port_);
    spectrum_node_detail::add_scalar_port(this, "g", 1.0, &g_port_);
    spectrum_node_detail::add_scalar_port(this, "l1", 0.0, &l1_port_);
    spectrum_node_detail::add_scalar_port(this, "l2", 0.0, &l2_port_);
  }
  rotation_ports_.clear();
  rotation_ports_.reserve(static_cast<std::size_t>(2 * n));
  for (int i = 0; i < n; ++i) {
    std::ostringstream b, rho;
    b << "b" << i;
    rho << "rho" << i;
    GraphPort* amplitude = nullptr;
    GraphPort* time = nullptr;
    if (get_input_port(b.str())) {
      amplitude = get_input_port(b.str()).get();
      time = get_input_port(rho.str()).get();
    } else {
      spectrum_node_detail::add_scalar_port(this, b.str(), 1.0, &amplitude);
      spectrum_node_detail::add_scalar_port(this, rho.str(), 1.0, &time);
    }
    rotation_ports_.push_back(amplitude);
    rotation_ports_.push_back(time);
  }
  n_rotations_ = n;
  set_valid(false);
}

void PhotophysicsAnisotropySpectrumNode::set_polarization(Polarization p) {
  polarization_ = p;
  set_valid(false);
}

void PhotophysicsAnisotropySpectrumNode::set_polarization_name(const std::string& name) {
  std::string lowered;
  lowered.reserve(name.size());
  for (char c : name) {
    lowered.push_back(static_cast<char>(
        std::tolower(static_cast<unsigned char>(c))));
  }
  if (lowered == "vm") {
    set_polarization(VM);
  } else if (lowered == "vv") {
    set_polarization(VV);
  } else if (lowered == "vh") {
    set_polarization(VH);
  } else if (lowered == "vv/vh" || lowered == "vv_vh") {
    set_polarization(VV_VH);
  } else {
    // Not defaulted to VM: that would return the spectrum unchanged, which
    // is a model with no anisotropy at all rather than an error anyone sees.
    throw std::domain_error(
        "PhotophysicsAnisotropySpectrumNode::set_polarization_name: '" + name +
        "' is not one of vm, vv, vh, vv/vh");
  }
}

std::string PhotophysicsAnisotropySpectrumNode::get_polarization_name() const {
  switch (polarization_) {
    case VV:
      return "vv";
    case VH:
      return "vh";
    case VV_VH:
      return "vv/vh";
    default:
      return "vm";
  }
}

void PhotophysicsAnisotropySpectrumNode::build_rotation_spectrum() {
  rotation_.resize(static_cast<std::size_t>(2 * n_rotations_));
  double sum = 0.0;
  for (int i = 0; i < n_rotations_; ++i) {
    const std::size_t k = static_cast<std::size_t>(2 * i);
    const double amplitude =
        std::fabs(rotation_ports_[k]->get_value());
    rotation_[k] = amplitude;
    sum += amplitude;
    rotation_[k + 1] = std::fabs(rotation_ports_[k + 1]->get_value());
  }
  // b_i <- |b_i| / sum|b| * r0, so `r0` alone sets r(0) and the b_i only
  // divide it up. ChiSurf normalises in the getter and writes the result
  // back to the parameters; the write-back is the application's business,
  // the normalisation is the model's.
  const double r0 = r0_port_->get_value();
  for (int i = 0; i < n_rotations_; ++i) {
    rotation_[static_cast<std::size_t>(2 * i)] *= r0 / sum;
  }
}

namespace {

//! The component kinds the spectrum arrived with (`spectrum_kinds`, optional),
//! one per component: zeros when absent.
std::vector<double> incoming_kinds(GraphNode* node, std::size_t n) {
  std::vector<double> kinds(n, 0.0);
  if (const std::shared_ptr<GraphPort> in = node->get_input_port("spectrum_kinds")) {
    const std::vector<double>& given = in->get_values_ref();
    const bool any = std::any_of(given.begin(), given.end(), [](double v) { return v != 0.0; });
    if (given.size() == n) {
      kinds.assign(given.begin(), given.end());
    } else if (any) {
      throw std::domain_error("PhotophysicsAnisotropySpectrumNode '" + node->get_name() + "': " +
                              std::to_string(given.size()) + " component kinds for " +
                              std::to_string(n) + " components");
    }
  }
  return kinds;
}

//! Publish the kinds of the output spectrum, when a `kinds` output was asked for.
void publish_kinds(GraphNode* node, const std::vector<double>& kinds) {
  if (const std::shared_ptr<GraphPort> out = node->get_output_port("kinds")) out->set_value_vector(kinds);
}

}  // namespace

void PhotophysicsAnisotropySpectrumNode::evaluate() {
  if (spectrum_port_ == nullptr) {
    throw std::domain_error("PhotophysicsAnisotropySpectrumNode '" + get_name() +
                            "' has no ports; call set_number_of_rotations() "
                            "first, which is what builds them");
  }
  // Resolved by key rather than through the cached pointer: a caller who
  // hands in their *own* port for this key -- which is how the misfit node
  // downstream is wired, so it is the shape a builder copies -- replaces the
  // one built here, and the cached raw pointer would then be reading a port
  // nothing owns any more. The scalars keep their pointers; only this one is
  // ever replaced, and one map lookup per evaluation is not a cost a fit can
  // measure.
  const std::shared_ptr<GraphPort> in = get_input_port(spectrum_port_key());
  if (!in) {
    throw std::domain_error("PhotophysicsAnisotropySpectrumNode '" + get_name() +
                            "' has no '" + spectrum_port_key() + "' port");
  }
  const std::vector<double>& incoming = in->get_values_ref();
  spectrum_node_detail::check_interleaved("PhotophysicsAnisotropySpectrumNode '" + get_name() + "'", incoming);

  if (polarization_ == VM || n_rotations_ <= 0) {
    // Magic angle has no anisotropy contribution, and building the rotation
    // spectrum first would be waste -- not cheap waste, since the caller's
    // normalisation writes back per component on every evaluation.
    spectrum_ = incoming;
    rotation_.clear();
    spectrum_node_detail::publish(this, spectrum_);
    publish_kinds(this, incoming_kinds(this, incoming.size() / 2));
    set_valid(true);
    return;
  }

  build_rotation_spectrum();

  // The anisotropy decay times the fluorescence decay, as a spectrum. The
  // rotation spectrum is the *first* argument, so the product runs
  // rotation-major -- the order ChiSurf's `elte2(a, f)` produces, and the
  // order is visible to anyone who reads the spectrum back.
  const std::vector<double> product = interleaved_product(rotation_, incoming);

  const double g = g_port_->get_value();
  const double l1 = l1_port_->get_value();
  const double l2 = l2_port_->get_value();

  //  f_VV = f (1 + 2 r),  f_VH = f (1 - r) / G, as spectra: the unmodified
  //  spectrum followed by the product scaled by +2 (resp. -1), the whole of
  //  VH then divided by G.
  std::vector<double> vv(incoming);
  interleaved_scale_amplitudes(product, 2.0, &vv);

  std::vector<double> vh_unscaled(incoming);
  interleaved_scale_amplitudes(product, -1.0, &vh_unscaled);
  std::vector<double> vh;
  vh.reserve(vh_unscaled.size());
  // G divides. Multiplying here instead put the two forward models a factor
  // G^2 apart in VH, which is a bug this stack has already paid for once.
  interleaved_scale_amplitudes(vh_unscaled, 1.0 / g, &vh);

  // A mixed channel is the *union* of the scaled VV and VH components, not
  // their element-wise sum: adding would also add the time constants and so
  // double the decay times.
  spectrum_.clear();
  if (polarization_ == VV || polarization_ == VV_VH) {
    interleaved_scale_amplitudes(vv, 1.0 - l1, &spectrum_);
    interleaved_scale_amplitudes(vh, l1, &spectrum_);
  }
  if (polarization_ == VH || polarization_ == VV_VH) {
    interleaved_scale_amplitudes(vv, l2, &spectrum_);
    interleaved_scale_amplitudes(vh, 1.0 - l2, &spectrum_);
  }
  spectrum_node_detail::publish(this, spectrum_);
  if (get_output_port("kinds")) {
    // The layout above, kind by kind: a channel is the incoming components
    // followed by the product (rotation-major, so the incoming kinds repeat
    // once per rotation; a rotation does not change t e^{-t/tau} into
    // anything else); VV and VH have that layout, and the channel is their union.
    const std::vector<double> k = incoming_kinds(this, incoming.size() / 2);
    std::vector<double> one(k);
    for (int r = 0; r < n_rotations_; ++r) one.insert(one.end(), k.begin(), k.end());
    std::vector<double> kinds;
    const int unions = polarization_ == VV_VH ? 2 : 1;
    for (int u = 0; u < 2 * unions; ++u) kinds.insert(kinds.end(), one.begin(), one.end());
    publish_kinds(this, kinds);
  }
  set_valid(true);
}

std::string PhotophysicsAnisotropySpectrumNode::describe() const {
  std::ostringstream out;
  out << "polarization   : " << get_polarization_name() << "\n"
      << "rotations      : " << n_rotations_ << "\n"
      << "species out    : " << spectrum_.size() / 2 << "\n";
  return out.str();
}

std::string PhotophysicsAnisotropySpectrumNode::get_node_type() const { return "PhotophysicsAnisotropySpectrumNode"; }

void PhotophysicsAnisotropySpectrumNode::configure(
    const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  if (config.has("number_of_rotations")) {
    set_number_of_rotations(config.get_int("number_of_rotations"));
  }
  // By name, not by code: a description saying "VH" survives a renumbering
  // of the enum, and says what it means to a reader.
  if (config.has("polarization")) {
    set_polarization_name(config.get_string("polarization"));
  }
  config.apply_common(*this);
  config.require_all_used();
}

IMPBFF_END_NAMESPACE
