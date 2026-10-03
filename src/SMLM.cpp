/** \file SMLM.cpp
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SMLM.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <utility>

IMPBFF_BEGIN_NAMESPACE
namespace {
constexpr double smlm_index_log_gaussian_coefficient = -2.756815599614018;

void smlm_index_require(bool condition, const char* message) {
  if (!condition) IMP_THROW(message, IMP::ValueException);
}

void smlm_index_finite_values(const std::vector<double>& values, const char* message) {
  for (double value : values) smlm_index_require(std::isfinite(value), message);
}

void smlm_index_coordinates_valid(const std::vector<double>& coordinates) {
  smlm_index_require(coordinates.size() % 3 == 0,
          "SMLM coordinates must be flattened N x 3 (3D only)");
  smlm_index_require(coordinates.size() / 3 <=
              static_cast<std::size_t>(std::numeric_limits<int>::max()),
          "Too many SMLM localizations");
  smlm_index_finite_values(coordinates, "SMLM coordinates must be finite");
}

double smlm_index_weight_sum(const std::vector<double>& weights, std::size_t count) {
  smlm_index_require(weights.size() == count, "SMLM weight length must equal point count");
  long double sum = 0;
  for (double weight : weights) {
    smlm_index_require(std::isfinite(weight) && weight >= 0,
            "SMLM weights must be finite and nonnegative");
    sum += weight;
  }
  smlm_index_require(sum <= std::numeric_limits<double>::max(),
          "SMLM total weight is not smlm_index_representable");
  return static_cast<double>(sum);
}

void smlm_index_cutoff_valid(double cutoff) {
  smlm_index_require(std::isfinite(cutoff) && cutoff > 0,
          "SMLM cutoff_sigma must be finite and positive");
}

std::vector<double> smlm_index_identity_transform() {
  return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
}

void smlm_index_transform_valid(const double* transform) {
  for (int i = 0; i < 12; ++i)
    smlm_index_require(std::isfinite(transform[i]), "SMLM transform must be finite");
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      long double dot = 0;
      for (int k = 0; k < 3; ++k)
        dot += static_cast<long double>(transform[4*a+k]) * transform[4*b+k];
      smlm_index_require(std::abs(dot - (a == b ? 1.0L : 0.0L)) <= 1e-6L,
              "SMLM transform rotation must be orthonormal");
    }
  }
  const double determinant =
      transform[0]*(transform[5]*transform[10] - transform[6]*transform[9]) -
      transform[1]*(transform[4]*transform[10] - transform[6]*transform[8]) +
      transform[2]*(transform[4]*transform[9] - transform[5]*transform[8]);
  smlm_index_require(std::abs(determinant - 1.0) <= 1e-6,
          "SMLM transform must have a proper rotation (determinant +1)");
}

std::array<double, 3> smlm_index_transform_point(const double* transform,
                                    const double* point) {
  std::array<double, 3> result;
  for (int a = 0; a < 3; ++a) {
    long double value = transform[4*a+3];
    for (int k = 0; k < 3; ++k)
      value += static_cast<long double>(transform[4*a+k]) * point[k];
    smlm_index_require(std::abs(value) <= std::numeric_limits<double>::max(),
            "Transformed SMLM coordinate is not smlm_index_representable");
    result[a] = static_cast<double>(value);
  }
  return result;
}

long double smlm_index_distance_squared(const double* a, const double* b) {
  long double distance = 0;
  for (int k = 0; k < 3; ++k) {
    const long double delta = static_cast<long double>(a[k]) - b[k];
    distance += delta * delta;
  }
  return distance;
}

long double smlm_index_box_distance_squared(const double* point, const double* lower,
                                 const double* upper) {
  long double distance = 0;
  for (int k = 0; k < 3; ++k) {
    const long double delta = point[k] < lower[k]
        ? static_cast<long double>(lower[k]) - point[k]
        : (point[k] > upper[k]
           ? static_cast<long double>(point[k]) - upper[k] : 0.0L);
    distance += delta * delta;
  }
  return distance;
}

void smlm_index_representable(long double value, const char* message) {
  smlm_index_require(std::isfinite(value) &&
              std::abs(value) <= std::numeric_limits<double>::max(), message);
}

// Clip in floating point BEFORE conversion to integer; remote/wide kernels
// must not overflow the integer voxel bounds.
bool smlm_index_voxel_bounds(long double centre, long double extent, double origin,
                  double spacing, int shape, int& lower, int& upper) {
  long double low = std::ceil((centre - extent - origin) / spacing);
  long double high = std::floor((centre + extent - origin) / spacing);
  if (high < 0 || low > shape - 1) return false;
  low = std::max(low, 0.0L);
  high = std::min(high, static_cast<long double>(shape - 1));
  if (low > high) return false;
  lower = static_cast<int>(low);
  upper = static_cast<int>(high);
  return true;
}
}  // namespace

SMLMIndex::SMLMIndex(const std::vector<double>& coordinates,
                     const std::vector<double>& sigmas,
                     const std::vector<double>& weights,
                     const std::vector<int>& particle_ids, int leaf_size)
    : coordinates_(coordinates), weights_(weights), particle_ids_(particle_ids),
      leaf_size_(leaf_size) {
  smlm_index_coordinates_valid(coordinates_);
  smlm_index_require(leaf_size > 0, "SMLM leaf_size must be positive");
  const std::size_t count = coordinates_.size() / 3;
  smlm_index_require(sigmas.size() == count || sigmas.size() == 3*count,
          "SMLM sigmas must have N isotropic or N x 3 anisotropic entries");
  for (double sigma : sigmas)
    smlm_index_require(std::isfinite(sigma) && sigma > 0,
            "SMLM sigmas must be finite and positive");
  sigmas_.resize(3*count);
  for (std::size_t i = 0; i < count; ++i)
    for (int k = 0; k < 3; ++k)
      sigmas_[3*i+k] = sigmas.size() == count ? sigmas[i] : sigmas[3*i+k];
  if (weights_.empty()) weights_.assign(count, 1.0);
  total_weight_ = smlm_index_weight_sum(weights_, count);
  if (particle_ids_.empty()) particle_ids_.assign(count, -1);
  smlm_index_require(particle_ids_.size() == count,
          "SMLM particle ID length must equal point count");
  log_normalizers_.resize(count);
  order_.resize(count);
  std::iota(order_.begin(), order_.end(), 0);
  for (std::size_t i = 0; i < count; ++i) {
    log_normalizers_[i] = smlm_index_log_gaussian_coefficient;
    for (int k = 0; k < 3; ++k)
      log_normalizers_[i] -= std::log(sigmas_[3*i+k]);
    if (particle_ids_[i] >= 0)
      particles_[particle_ids_[i]].push_back(static_cast<int>(i));
  }
  if (count) build_node(0, static_cast<int>(count));
}

int SMLMIndex::get_number_of_localizations() const {
  return static_cast<int>(coordinates_.size() / 3);
}

std::vector<int> SMLMIndex::get_particle_localizations(int particle_id) const {
  const auto found = particles_.find(particle_id);
  return found == particles_.end() ? std::vector<int>() : found->second;
}

int SMLMIndex::build_node(int begin, int end) {
  Node node;
  node.begin = begin;
  node.end = end;
  for (int k = 0; k < 3; ++k) {
    node.lower[k] = std::numeric_limits<double>::infinity();
    node.upper[k] = -std::numeric_limits<double>::infinity();
    node.max_sigma[k] = 0;
  }
  for (int position = begin; position < end; ++position) {
    const std::size_t offset = 3*static_cast<std::size_t>(order_[position]);
    for (int k = 0; k < 3; ++k) {
      node.lower[k] = std::min(node.lower[k], coordinates_[offset+k]);
      node.upper[k] = std::max(node.upper[k], coordinates_[offset+k]);
      node.max_sigma[k] = std::max(node.max_sigma[k], sigmas_[offset+k]);
    }
  }
  const int id = static_cast<int>(nodes_.size());
  nodes_.push_back(node);
  if (end - begin > leaf_size_) {
    int axis = 0;
    for (int k = 1; k < 3; ++k)
      if (static_cast<long double>(node.upper[k]) - node.lower[k] >
          static_cast<long double>(node.upper[axis]) - node.lower[axis]) axis = k;
    const int middle = begin + (end - begin)/2;
    std::nth_element(order_.begin()+begin, order_.begin()+middle, order_.begin()+end,
        [&](int a, int b) {
          const double ca = coordinates_[3*static_cast<std::size_t>(a)+axis];
          const double cb = coordinates_[3*static_cast<std::size_t>(b)+axis];
          return ca < cb || (ca == cb && a < b);
        });
    // Recursive push_back can reallocate nodes_; never retain a Node reference.
    const int left = build_node(begin, middle);
    const int right = build_node(middle, end);
    nodes_[id].left = left;
    nodes_[id].right = right;
  }
  return id;
}

std::vector<int> SMLMIndex::radius_search(const std::vector<double>& point,
                                        double radius) const {
  smlm_index_require(point.size() == 3, "SMLM search point must have three coordinates");
  smlm_index_finite_values(point, "SMLM search point must be finite");
  smlm_index_require(std::isfinite(radius) && radius >= 0,
          "SMLM search radius must be finite and nonnegative");
  std::vector<int> result, stack;
  const long double radius_squared = static_cast<long double>(radius)*radius;
  if (!nodes_.empty()) stack.push_back(0);
  while (!stack.empty()) {
    const Node& node = nodes_[stack.back()];
    stack.pop_back();
    if (smlm_index_box_distance_squared(point.data(), node.lower, node.upper) > radius_squared)
      continue;
    if (node.left >= 0) {
      stack.push_back(node.left);
      stack.push_back(node.right);
    } else {
      for (int position = node.begin; position < node.end; ++position) {
        const int i = order_[position];
        if (smlm_index_distance_squared(point.data(),
                coordinates_.data()+3*static_cast<std::size_t>(i)) <= radius_squared)
          result.push_back(i);
      }
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

std::vector<int> SMLMIndex::nearest_search(const std::vector<double>& point, int k,
                                         double max_distance) const {
  smlm_index_require(point.size() == 3, "SMLM search point must have three coordinates");
  smlm_index_finite_values(point, "SMLM search point must be finite");
  smlm_index_require(k > 0, "SMLM nearest neighbor count must be positive");
  smlm_index_require(std::isfinite(max_distance), "SMLM max_distance must be finite");
  using Neighbor = std::pair<long double, int>;
  std::priority_queue<Neighbor> candidates;
  std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor> > nodes;
  const long double limit = max_distance < 0
      ? std::numeric_limits<long double>::infinity()
      : static_cast<long double>(max_distance)*max_distance;
  if (!nodes_.empty())
    nodes.emplace(smlm_index_box_distance_squared(point.data(), nodes_[0].lower,
                                        nodes_[0].upper), 0);
  while (!nodes.empty()) {
    const Neighbor next = nodes.top();
    nodes.pop();
    const long double bound = candidates.size() == static_cast<std::size_t>(k)
        ? std::min(limit, candidates.top().first) : limit;
    if (next.first > bound) break;
    const Node& node = nodes_[next.second];
    if (node.left >= 0) {
      for (int child : {node.left, node.right}) {
        const Node& c = nodes_[child];
        const long double distance = smlm_index_box_distance_squared(point.data(), c.lower, c.upper);
        if (distance <= bound) nodes.emplace(distance, child);
      }
    } else {
      for (int position = node.begin; position < node.end; ++position) {
        const int i = order_[position];
        const Neighbor candidate(smlm_index_distance_squared(point.data(),
            coordinates_.data()+3*static_cast<std::size_t>(i)), i);
        if (candidate.first > limit) continue;
        if (candidates.size() < static_cast<std::size_t>(k)) candidates.push(candidate);
        else if (candidate < candidates.top()) {
          candidates.pop();
          candidates.push(candidate);
        }
      }
    }
  }
  std::vector<int> result(candidates.size());
  for (std::size_t i = result.size(); i > 0; --i) {
    result[i-1] = candidates.top().second;
    candidates.pop();
  }
  return result;
}

double SMLMIndex::evaluate_point(const double* point, double cutoff,
                                bool normalize_weights, double* gradient) const {
  long double density = 0, derivative[3] = {0, 0, 0};
  if (gradient) std::fill(gradient, gradient+3, 0.0);
  if (total_weight_ == 0) return 0;
  const long double cutoff_squared = static_cast<long double>(cutoff)*cutoff;
  std::vector<int> stack;
  if (!nodes_.empty()) stack.push_back(0);
  while (!stack.empty()) {
    const Node& node = nodes_[stack.back()];
    stack.pop_back();
    // Every kernel centre is in this box and sigma[k] <= max_sigma[k].
    // This is a lower bound on every descendant's Mahalanobis distance,
    // including kernels whose support extends far beyond their centre box.
    long double lower_bound = 0;
    for (int k = 0; k < 3; ++k) {
      const long double delta = point[k] < node.lower[k]
          ? static_cast<long double>(node.lower[k]) - point[k]
          : (point[k] > node.upper[k]
             ? static_cast<long double>(point[k]) - node.upper[k] : 0.0L);
      const long double scaled = delta/node.max_sigma[k];
      lower_bound += scaled*scaled;
    }
    if (lower_bound > cutoff_squared) continue;
    if (node.left >= 0) {
      stack.push_back(node.left);
      stack.push_back(node.right);
      continue;
    }
    for (int position = node.begin; position < node.end; ++position) {
      const int i = order_[position];
      if (weights_[i] == 0) continue;
      const std::size_t offset = 3*static_cast<std::size_t>(i);
      long double q2 = 0, scaled[3];
      for (int k = 0; k < 3; ++k) {
        scaled[k] = (static_cast<long double>(point[k])-coordinates_[offset+k]) /
                    sigmas_[offset+k];
        q2 += scaled[k]*scaled[k];
      }
      if (q2 > cutoff_squared) continue;
      const long double log_weight = std::log(static_cast<long double>(weights_[i])) -
          (normalize_weights ? std::log(static_cast<long double>(total_weight_)) : 0.0L);
      const long double kernel = std::exp(log_weight + log_normalizers_[i] - 0.5L*q2);
      density += kernel;
      if (gradient)
        for (int k = 0; k < 3; ++k)
          derivative[k] -= kernel*scaled[k]/sigmas_[offset+k];
    }
  }
  smlm_index_representable(density, "SMLM density is not smlm_index_representable");
  if (gradient)
    for (int k = 0; k < 3; ++k) {
      smlm_index_representable(derivative[k], "SMLM density derivative is not smlm_index_representable");
      gradient[k] = static_cast<double>(derivative[k]);
    }
  return static_cast<double>(density);
}

std::vector<double> SMLMIndex::evaluate_density(
    const std::vector<double>& query_coordinates, double cutoff_sigma,
    bool normalize_weights) const {
  smlm_index_coordinates_valid(query_coordinates);
  smlm_index_cutoff_valid(cutoff_sigma);
  std::vector<double> result(query_coordinates.size()/3);
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = evaluate_point(query_coordinates.data()+3*i, cutoff_sigma,
                               normalize_weights, nullptr);
  return result;
}

SMLMScoreResult SMLMIndex::evaluate_score(
    const std::vector<double>& query_coordinates,
    const std::vector<double>& query_weights, double cutoff_sigma,
    double background, bool normalize_weights) const {
  smlm_index_coordinates_valid(query_coordinates);
  smlm_index_cutoff_valid(cutoff_sigma);
  smlm_index_require(std::isfinite(background) && background > 0,
          "SMLM background must be finite and positive");
  const std::size_t count = query_coordinates.size()/3;
  const double total = query_weights.empty() ? static_cast<double>(count)
                                            : smlm_index_weight_sum(query_weights, count);
  SMLMScoreResult result;
  result.densities.resize(count);
  result.gradient.assign(query_coordinates.size(), 0);
  long double score = 0;
  for (std::size_t i = 0; i < count; ++i) {
    double derivative[3];
    const double weight = query_weights.empty() ? 1 : query_weights[i];
    const bool active = weight > 0 && total > 0;
    const double density = evaluate_point(query_coordinates.data()+3*i, cutoff_sigma,
                                          normalize_weights, active ? derivative : nullptr);
    result.densities[i] = density;
    if (!active) continue;
    const long double probability = static_cast<long double>(density)+background;
    const long double fraction = static_cast<long double>(weight)/total;
    score -= fraction*std::log(probability);
    for (int k = 0; k < 3; ++k) {
      const long double value = -fraction*derivative[k]/probability;
      smlm_index_representable(value, "SMLM score derivative is not smlm_index_representable");
      result.gradient[3*i+k] = static_cast<double>(value);
    }
  }
  smlm_index_representable(score, "SMLM score is not smlm_index_representable");
  result.score = static_cast<double>(score);
  return result;
}

SMLMAverageResult average_smlm_particles(
    const SMLMIndex& index, const std::vector<int>& particle_ids,
    const std::vector<double>& transforms, const std::vector<int>& grid_shape,
    const std::vector<double>& origin, const std::vector<double>& spacing,
    const SMLMAverageOptions& options) {
  smlm_index_require(!particle_ids.empty(), "SMLM averaging needs at least one selected particle");
  smlm_index_require(grid_shape.size() == 3 && origin.size() == 3 && spacing.size() == 3,
          "SMLM grid shape, origin and spacing must each have three entries");
  smlm_index_finite_values(origin, "SMLM grid origin must be finite");
  smlm_index_cutoff_valid(options.cutoff_sigma);
  smlm_index_require(options.min_localizations >= 0 && options.max_localizations >= 0 &&
          (options.max_localizations == 0 ||
           options.max_localizations >= options.min_localizations),
          "Invalid SMLM localization count gates");
  const auto& scores = options.particle_scores;
  smlm_index_require(scores.empty() || scores.size() == particle_ids.size(),
          "SMLM particle_scores must follow requested particle IDs");
  smlm_index_finite_values(scores, "SMLM particle scores must be finite");
  if (options.use_score_gate) {
    smlm_index_require(scores.size() == particle_ids.size(), "SMLM score gate needs particle scores");
    smlm_index_require(std::isfinite(options.min_score) && std::isfinite(options.max_score) &&
            options.min_score <= options.max_score, "Invalid SMLM particle score gates");
  }
  smlm_index_require(transforms.empty() || transforms.size()/12 == particle_ids.size(),
          "SMLM transforms must have 12 entries per requested particle");
  smlm_index_require(transforms.size() % 12 == 0,
          "SMLM transforms must be flattened source -> reference 3 x 4 matrices");
  std::size_t grid_size = 1;
  for (int k = 0; k < 3; ++k) {
    smlm_index_require(grid_shape[k] > 0, "SMLM grid dimensions must be positive");
    smlm_index_require(std::isfinite(spacing[k]) && spacing[k] > 0,
            "SMLM grid spacing must be finite and positive");
    const long double last = static_cast<long double>(origin[k]) +
        static_cast<long double>(spacing[k])*(grid_shape[k]-1);
    smlm_index_representable(last, "SMLM grid coordinate is not smlm_index_representable");
    smlm_index_require(grid_size <= std::vector<double>().max_size() /
                static_cast<std::size_t>(grid_shape[k]), "SMLM grid size overflow");
    grid_size *= static_cast<std::size_t>(grid_shape[k]);
  }
  const auto identity = smlm_index_identity_transform();
  std::set<int> unique;
  std::vector<std::vector<int> > groups;
  groups.reserve(particle_ids.size());
  for (std::size_t p = 0; p < particle_ids.size(); ++p) {
    smlm_index_require(particle_ids[p] >= 0 && unique.insert(particle_ids[p]).second,
            "Selected SMLM particle IDs must be unique and nonnegative");
    groups.push_back(index.get_particle_localizations(particle_ids[p]));
    smlm_index_require(!groups.back().empty(), "Selected SMLM particle ID is absent from the index");
    smlm_index_transform_valid(transforms.empty() ? identity.data() : transforms.data()+12*p);
  }
  SMLMAverageResult result;
  result.grid_shape = grid_shape;
  result.origin = origin;
  result.spacing = spacing;
  result.values.assign(grid_size, 0);
  result.half1.assign(grid_size, 0);
  result.half2.assign(grid_size, 0);
  const auto& coordinates = index.get_coordinates();
  const auto& sigmas = index.get_sigmas();
  const auto& weights = index.get_weights();
  const long double cutoff_squared =
      static_cast<long double>(options.cutoff_sigma)*options.cutoff_sigma;
  long double denominators[2] = {0, 0};
  for (std::size_t p = 0; p < particle_ids.size(); ++p) {
    const auto& group = groups[p];
    long double particle_weight = 0;
    for (int i : group) particle_weight += weights[i];
    const bool rejected = group.size() < static_cast<std::size_t>(options.min_localizations) ||
        (options.max_localizations > 0 &&
         group.size() > static_cast<std::size_t>(options.max_localizations)) ||
        (options.use_score_gate && (scores[p] < options.min_score || scores[p] > options.max_score)) ||
        particle_weight == 0;
    if (rejected) {
      result.rejected_ids.push_back(particle_ids[p]);
      continue;
    }
    const int half = static_cast<int>(result.selected_ids.size() % 2);
    auto& map = half == 0 ? result.half1 : result.half2;
    auto& half_ids = half == 0 ? result.half1_particle_ids : result.half2_particle_ids;
    half_ids.push_back(particle_ids[p]);
    result.selected_ids.push_back(particle_ids[p]);
    result.localization_counts.push_back(static_cast<int>(group.size()));
    result.selected_localizations += static_cast<int>(group.size());
    if (half == 0) result.half1_localizations += static_cast<int>(group.size());
    else result.half2_localizations += static_cast<int>(group.size());
    const double* transform = transforms.empty() ? identity.data() : transforms.data()+12*p;
    result.transforms.insert(result.transforms.end(), transform, transform+12);
    const long double normalization = options.normalize_particles ? particle_weight : 1.0L;
    denominators[half] += options.normalize_particles ? 1.0L : particle_weight;
    for (int i : group) {
      if (weights[i] == 0) continue;
      const std::size_t offset = 3*static_cast<std::size_t>(i);
      const auto centre = smlm_index_transform_point(transform, coordinates.data()+offset);
      int lower[3], upper[3];
      bool intersects = true;
      for (int a = 0; a < 3; ++a) {
        // diag(R * diag(sigma^2) * R^T) supplies a conservative support box.
        // Evaluation below retains all off-diagonal terms via R^T*delta.
        long double variance = 0;
        for (int k = 0; k < 3; ++k) {
          const long double component =
              static_cast<long double>(transform[4*a+k])*sigmas[offset+k];
          variance += component*component;
        }
        const long double extent = options.cutoff_sigma*std::sqrt(variance);
        if (!smlm_index_voxel_bounds(centre[a], extent, origin[a], spacing[a], grid_shape[a],
                          lower[a], upper[a])) intersects = false;
      }
      if (!intersects) continue;
      long double log_scale = smlm_index_log_gaussian_coefficient +
          std::log(static_cast<long double>(weights[i])) - std::log(normalization);
      for (int k = 0; k < 3; ++k) log_scale -= std::log(sigmas[offset+k]);
      for (int z = lower[2]; z <= upper[2]; ++z) {
        const long double dz = static_cast<long double>(origin[2]) +
                               static_cast<long double>(spacing[2])*z - centre[2];
        for (int y = lower[1]; y <= upper[1]; ++y) {
          const long double dy = static_cast<long double>(origin[1]) +
                                 static_cast<long double>(spacing[1])*y - centre[1];
          for (int x = lower[0]; x <= upper[0]; ++x) {
            const long double dx = static_cast<long double>(origin[0]) +
                                   static_cast<long double>(spacing[0])*x - centre[0];
            // R^T maps reference offsets to the source Gaussian's axes.
            long double q2 = 0;
            for (int k = 0; k < 3; ++k) {
              const long double scaled = (transform[k]*dx + transform[4+k]*dy +
                                           transform[8+k]*dz)/sigmas[offset+k];
              q2 += scaled*scaled;
            }
            if (q2 > cutoff_squared) continue;
            const long double kernel = std::exp(log_scale - 0.5L*q2);
            smlm_index_representable(kernel, "SMLM map density is not smlm_index_representable");
            const std::size_t voxel = static_cast<std::size_t>(x) +
                static_cast<std::size_t>(grid_shape[0]) *
                (static_cast<std::size_t>(y) + static_cast<std::size_t>(grid_shape[1])*z);
            map[voxel] += static_cast<double>(kernel);
          }
        }
      }
    }
  }
  const long double denominator = denominators[0]+denominators[1];
  for (std::size_t i = 0; i < grid_size; ++i) {
    long double first = result.half1[i], second = result.half2[i];
    long double combined = first + second;
    if (options.normalize_density) {
      if (denominator > 0) combined /= denominator;
      if (denominators[0] > 0) first /= denominators[0];
      if (denominators[1] > 0) second /= denominators[1];
    }
    smlm_index_representable(combined, "SMLM average density overflow");
    smlm_index_representable(first, "SMLM half map density overflow");
    smlm_index_representable(second, "SMLM half map density overflow");
    result.values[i] = static_cast<double>(combined);
    result.half1[i] = static_cast<double>(first);
    result.half2[i] = static_cast<double>(second);
  }
  return result;
}

SMLMRegistrationResult refine_smlm_translation(
    const SMLMIndex& reference, const std::vector<double>& moving_coordinates,
    const std::vector<double>& initial_transform,
    const std::vector<double>& query_weights,
    const SMLMRegistrationOptions& options) {
  smlm_index_coordinates_valid(moving_coordinates);
  smlm_index_require(!moving_coordinates.empty(), "SMLM registration needs moving points");
  smlm_index_require(reference.get_number_of_localizations() > 0,
          "SMLM registration needs reference points");
  smlm_index_require(options.max_iterations >= 0, "SMLM registration iterations must be nonnegative");
  smlm_index_require(std::isfinite(options.max_translation_step) && options.max_translation_step > 0,
          "SMLM registration step limit must be finite and positive");
  smlm_index_require(std::isfinite(options.tolerance) && options.tolerance > 0,
          "SMLM registration tolerance must be finite and positive");
  smlm_index_cutoff_valid(options.cutoff_sigma);
  smlm_index_require(std::isfinite(options.background) && options.background > 0,
          "SMLM registration background must be finite and positive");
  smlm_index_require(query_weights.empty() ||
              smlm_index_weight_sum(query_weights, moving_coordinates.size()/3) > 0,
          "SMLM registration query weights must have positive total weight");
  smlm_index_require(smlm_index_weight_sum(reference.get_weights(),
              reference.get_number_of_localizations()) > 0,
          "SMLM registration reference weights must have positive total weight");
  SMLMRegistrationResult result;
  result.transform = initial_transform.empty() ? smlm_index_identity_transform() : initial_transform;
  smlm_index_require(result.transform.size() == 12, "SMLM initial transform must have 12 entries");
  smlm_index_transform_valid(result.transform.data());
  std::vector<double> points(moving_coordinates.size());
  for (std::size_t i = 0; i < points.size()/3; ++i) {
    const auto point = smlm_index_transform_point(result.transform.data(), moving_coordinates.data()+3*i);
    std::copy(point.begin(), point.end(), points.begin()+3*i);
  }
  auto current = reference.evaluate_score(points, query_weights,
                                          options.cutoff_sigma, options.background);
  result.initial_score = result.score = current.score;
  for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
    long double gradient[3] = {0, 0, 0};
    for (std::size_t i = 0; i < points.size()/3; ++i)
      for (int k = 0; k < 3; ++k) gradient[k] += current.gradient[3*i+k];
    const long double norm = std::sqrt(gradient[0]*gradient[0] +
                                       gradient[1]*gradient[1] + gradient[2]*gradient[2]);
    if (norm <= options.tolerance) {
      // A background-only plateau is not evidence of successful registration.
      result.converged = std::any_of(current.densities.begin(), current.densities.end(),
                                     [](double density) { return density > 0; });
      break;
    }
    double step = options.max_translation_step;
    bool accepted = false;
    std::vector<double> trial(points.size());
    SMLMScoreResult candidate;
    double delta[3] = {0, 0, 0};
    for (int backtrack = 0; backtrack < 40; ++backtrack) {
      for (int k = 0; k < 3; ++k)
        delta[k] = static_cast<double>(-step*gradient[k]/norm);
      for (std::size_t i = 0; i < points.size(); ++i) trial[i] = points[i]+delta[i%3];
      candidate = reference.evaluate_score(trial, query_weights,
                                           options.cutoff_sigma, options.background);
      if (candidate.score < current.score) { accepted = true; break; }
      step *= 0.5;
      if (step <= options.tolerance) break;
    }
    if (!accepted) {
      result.converged = step <= options.tolerance;
      break;
    }
    for (int k = 0; k < 3; ++k) result.transform[4*k+3] += delta[k];
    points.swap(trial);
    current = std::move(candidate);
    result.score = current.score;
    ++result.iterations;
    if (step <= options.tolerance) { result.converged = true; break; }
  }
  return result;
}

IMPBFF_END_NAMESPACE
