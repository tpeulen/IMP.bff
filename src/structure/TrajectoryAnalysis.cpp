/**
 *  \file TrajectoryAnalysis.cpp
 *  \brief Axes, profiles and histograms read off the frames of a trajectory.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/TrajectoryAnalysis.h>

#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/internal/NumpyCompat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

IMPBFF_BEGIN_NAMESPACE

namespace trajectory_analysis {

namespace nc = internal::numpy_compat;

std::string fmt(const char* format, double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), format, v);
  return buf;
}

std::ofstream open_csv(const std::string& path) {
  std::ofstream out(path.c_str());
  if (!out) {
    IMP_THROW("cannot write " << path, IOException);
  }
  return out;
}

}  // namespace trajectory_analysis

std::vector<double> point_cloud_principal_axis(const std::vector<double>& coords,
                                               bool smallest_variance) {
  namespace nc = internal::numpy_compat;
  const std::size_t n = coords.size() / 3;
  if (n == 0) return std::vector<double>();
  double center[3];
  for (int k = 0; k < 3; ++k) {
    center[k] = nc::sequential_sum(&coords[k], n, 3) / static_cast<double>(n);
  }
  std::vector<double> centered(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) centered[3 * i + k] = coords[3 * i + k] - center[k];
  }
  std::vector<double> column[3];
  for (int k = 0; k < 3; ++k) {
    column[k].resize(n);
    for (std::size_t i = 0; i < n; ++i) column[k][i] = centered[3 * i + k];
  }
  double cov[9];
  const double denom = static_cast<double>(n > 1 ? n : 1);
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      cov[3 * a + b] = nc::gram_entry(&column[a][0], &column[b][0], n) / denom;
    }
  }
  double w[3], vectors[9];
  nc::eigh3(cov, w, vectors);
  // np.argmin / np.argmax: the first index on ties
  int pick = 0;
  for (int k = 1; k < 3; ++k) {
    if (smallest_variance ? (w[k] < w[pick]) : (w[k] > w[pick])) pick = k;
  }
  double axis[3] = {vectors[3 * pick], vectors[3 * pick + 1], vectors[3 * pick + 2]};
  const double norm = (std::max)(nc::norm3(axis), 1e-12);
  std::vector<double> out(6);
  for (int k = 0; k < 3; ++k) {
    out[k] = center[k];
    out[3 + k] = axis[k] / norm;
  }
  return out;
}

void write_binned_profile(const std::string& path, const std::vector<std::string>& regions,
                          const std::vector<std::vector<double> >& values, double bin_width,
                          bool from_zero) {
  namespace nc = internal::numpy_compat;
  using trajectory_analysis::fmt;
  if (values.size() != regions.size()) {
    IMP_THROW(regions.size() << " regions and " << values.size() << " value lists",
              ValueException);
  }
  std::string counts_header, density_header;
  bool any = false;
  double vmin = 0.0, vmax = 0.0;
  for (std::size_t r = 0; r < regions.size(); ++r) {
    counts_header += (r ? "," : "") + regions[r] + "_count";
    density_header += (r ? "," : "") + regions[r] + "_density";
    for (std::size_t i = 0; i < values[r].size(); ++i) {
      const double v = values[r][i];
      if (!any) {
        vmin = vmax = v;
        any = true;
      } else {
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
    }
  }
  std::ofstream out = trajectory_analysis::open_csv(path);
  out << "bin_start_A,bin_end_A,center_A," << counts_header << "," << density_header << "\n";
  if (!any) return;

  const double zmin = from_zero ? 0.0 : vmin;
  const long n_bins = (std::max)(1L, static_cast<long>(std::ceil((vmax - zmin) / bin_width)));
  std::vector<std::vector<long> > counts(regions.size(), std::vector<long>(n_bins, 0));
  for (std::size_t r = 0; r < regions.size(); ++r) {
    for (std::size_t i = 0; i < values[r].size(); ++i) {
      long idx = static_cast<long>(nc::python_floor_div(values[r][i] - zmin, bin_width));
      idx = (std::max)(0L, (std::min)(n_bins - 1, idx));
      ++counts[r][idx];
    }
  }
  std::vector<double> scale(regions.size());
  for (std::size_t r = 0; r < regions.size(); ++r) {
    long total = 0;
    for (long b = 0; b < n_bins; ++b) total += counts[r][b];
    scale[r] = static_cast<double>((std::max)(1L, total)) * bin_width;
  }
  for (long i = 0; i < n_bins; ++i) {
    const double lo = zmin + static_cast<double>(i) * bin_width;
    const double hi = zmin + static_cast<double>(i + 1) * bin_width;
    const double center = 0.5 * (lo + hi);
    out << fmt("%.4f", lo) << "," << fmt("%.4f", hi) << "," << fmt("%.4f", center) << ",";
    for (std::size_t r = 0; r < regions.size(); ++r) out << (r ? "," : "") << counts[r][i];
    out << ",";
    for (std::size_t r = 0; r < regions.size(); ++r) {
      out << (r ? "," : "") << fmt("%.8f", static_cast<double>(counts[r][i]) / scale[r]);
    }
    out << "\n";
  }
}

void write_radial_histogram(const std::vector<double>& distances, const std::string& path,
                            double bin_width) {
  namespace nc = internal::numpy_compat;
  using trajectory_analysis::fmt;
  std::ofstream out = trajectory_analysis::open_csv(path);
  out << "bin_start_A,bin_end_A,count\n";
  if (distances.empty()) return;
  const double max_d = *std::max_element(distances.begin(), distances.end());
  const long n_bins = (std::max)(1L, static_cast<long>(std::ceil(max_d / bin_width)));
  std::vector<long> counts(n_bins, 0);
  for (std::size_t i = 0; i < distances.size(); ++i) {
    const long idx = (std::min)(n_bins - 1,
                                static_cast<long>(nc::python_floor_div(distances[i], bin_width)));
    ++counts[idx];
  }
  for (long i = 0; i < n_bins; ++i) {
    const double lo = static_cast<double>(i) * bin_width;
    const double hi = lo + bin_width;
    out << fmt("%.3f", lo) << "," << fmt("%.3f", hi) << "," << counts[i] << "\n";
  }
}

void write_orientation_profile(const std::string& path, const std::vector<int>& frame_index,
                               const std::vector<double>& angle_deg,
                               const std::vector<double>& abs_cos_theta,
                               const std::vector<double>& cos_theta) {
  using trajectory_analysis::fmt;
  const std::size_t n = frame_index.size();
  if (angle_deg.size() != n || abs_cos_theta.size() != n || cos_theta.size() != n) {
    IMP_THROW("the four orientation columns differ in length", ValueException);
  }
  std::ofstream out = trajectory_analysis::open_csv(path);
  out << "frame_index,angle_deg,abs_cos_theta,cos_theta\n";
  for (std::size_t i = 0; i < n; ++i) {
    out << frame_index[i] << "," << fmt("%.6f", angle_deg[i]) << ","
        << fmt("%.8f", abs_cos_theta[i]) << "," << fmt("%.8f", cos_theta[i]) << "\n";
  }
}

IMPBFF_END_NAMESPACE
