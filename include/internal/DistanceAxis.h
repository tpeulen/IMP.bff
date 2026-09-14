/**
 *  \file IMP/bff/internal/DistanceAxis.h
 *  \brief The distance axis a distance-distribution node is configured with.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_DISTANCE_AXIS_H
#define IMPBFF_INTERNAL_DISTANCE_AXIS_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/internal/NodeConfig.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE
namespace internal {

//! An axis from `axis` (the values) or `axis_range` `[min, max, n]` with
//! `axis_scale` `"log"` (the default, as ChiSurf's distance axis) or
//! `"linear"`. Returns an empty vector when the settings name none.
inline std::vector<double> configured_distance_axis(NodeConfig& config,
                                                    const std::string& where) {
  if (config.has("axis")) return config.get_doubles("axis");
  std::string scale = "log";
  if (config.has("axis_scale")) scale = config.get_string("axis_scale");
  if (!config.has("axis_range")) {
    if (scale != "log") {
      throw std::domain_error(where + ": 'axis_scale' without 'axis_range'");
    }
    return std::vector<double>();
  }
  const std::vector<double> range = config.get_doubles("axis_range");
  if (range.size() != 3 || !(range[1] > range[0]) || range[2] < 2.0) {
    throw std::domain_error(where + ": 'axis_range' is [min, max, n] with "
                                    "max > min and at least two points");
  }
  const int n = static_cast<int>(range[2]);
  std::vector<double> axis(static_cast<std::size_t>(n));
  if (scale == "linear") {
    for (int i = 0; i < n; ++i) {
      axis[static_cast<std::size_t>(i)] =
          range[0] + (range[1] - range[0]) * i / (n - 1);
    }
  } else if (scale == "log") {
    if (!(range[0] > 0.0)) {
      throw std::domain_error(where + ": a log axis starts above zero");
    }
    const double lo = std::log10(range[0]);
    const double hi = std::log10(range[1]);
    for (int i = 0; i < n; ++i) {
      axis[static_cast<std::size_t>(i)] =
          std::pow(10.0, lo + (hi - lo) * i / (n - 1));
    }
  } else {
    throw std::domain_error(where + ": 'axis_scale' is 'log' or 'linear', not '" +
                            scale + "'");
  }
  return axis;
}

}  // namespace internal
IMPBFF_END_NAMESPACE

#endif  // IMPBFF_INTERNAL_DISTANCE_AXIS_H
