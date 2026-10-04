/** \file SMLMParticleRegistration.cpp
 *  \brief Coarse-to-fine native rigid SMLM particle fitting.
 */
#include <IMP/bff/SMLMParticleRegistration.h>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <set>

IMPBFF_BEGIN_NAMESPACE
namespace smlm_registration_internal {
using Matrix34 = Eigen::Matrix<double,3,4,Eigen::RowMajor>;
std::vector<double> flat(const Matrix34& m) {return {m.data(),m.data()+12};}
Matrix34 inverse(const Matrix34& m) {
  const Eigen::Matrix3d r=m.leftCols<3>();
  Matrix34 out;out.leftCols<3>()=r.transpose();out.col(3)=-r.transpose()*m.col(3);return out;
}
}

SMLMParticleRegistrationResult register_smlm_particles(const SMLMIndex& index,
    const SMLMParticleFrames& frames,const SMLMPointModel& model,
    const SMLMParticleRegistrationOptions& options) {
  using namespace smlm_registration_internal;
  if(frames.transforms.size()!=12*frames.particle_ids.size() || frames.particle_ids.empty())
    throw std::invalid_argument("registration needs one rigid frame per selected particle");
  if(options.angular_samples<1 || !(options.angular_period>0.0) ||
     options.angular_period>6.283185307179587 || !std::isfinite(options.angular_period) ||
     !std::isfinite(options.intrinsic_sigma) || options.intrinsic_sigma<0.0 ||
     !std::isfinite(options.background_fraction) || options.background_fraction<0.0 ||
     options.background_fraction>=1.0 || !(options.roi_padding>0.0) ||
     !std::isfinite(options.roi_padding) || !std::isfinite(options.min_signal_fraction) ||
     options.min_signal_fraction<0.0 || options.min_signal_fraction>1.0)
    throw std::invalid_argument("invalid particle registration settings");
  const auto& xyz=index.get_coordinates();const auto& sigma=index.get_sigmas();
  const auto& weights=index.get_weights();
  SMLMParticleRegistrationResult out;std::set<int> unique;
  for(std::size_t ordinal=0;ordinal<frames.particle_ids.size();++ordinal) {
    const int id=frames.particle_ids[ordinal];
    if(id<0 || !unique.insert(id).second) throw std::invalid_argument("particle IDs must be unique and assigned");
    const auto rows=index.get_particle_localizations(id);
    if(rows.empty()) throw std::invalid_argument("particle ID is absent from the index");
    std::vector<double> coords,precisions,mass;
    SMLMLikelihoodOptions likelihood;
    likelihood.roi_min.assign(3,std::numeric_limits<double>::infinity());
    likelihood.roi_max.assign(3,-std::numeric_limits<double>::infinity());
    likelihood.background_fraction=options.background_fraction;
    likelihood.intrinsic_sigma=options.intrinsic_sigma;
    likelihood.compute_model_gradient=false;
    for(int row:rows) {
      mass.push_back(weights[row]);
      for(int d=0;d<3;++d) {
        const double v=xyz[3*row+d];coords.push_back(v);precisions.push_back(sigma[3*row+d]);
        likelihood.roi_min[d]=std::min(likelihood.roi_min[d],v-options.roi_padding);
        likelihood.roi_max[d]=std::max(likelihood.roi_max[d],v+options.roi_padding);
      }
    }
    SMLMIndex observed(coords,precisions,mass);
    Matrix34 forward=Eigen::Map<const Matrix34>(frames.transforms.data()+12*ordinal);
    // Validate the caller's frame through the common rigid transform contract.
    transform_smlm_points(std::vector<double>{0,0,0},flat(forward));
    const Matrix34 initial=inverse(forward);
    std::vector<std::pair<double,Matrix34>> starts;starts.reserve(options.angular_samples);
    const double initial_nll=model.evaluate(observed,likelihood,flat(initial)).mean_nll;
    for(int angle=0;angle<options.angular_samples;++angle) {
      const double theta=options.angular_period*angle/options.angular_samples;
      Matrix34 pose=initial;
      pose.leftCols<3>()=initial.leftCols<3>()*
          Eigen::AngleAxisd(theta,Eigen::Vector3d::UnitZ()).toRotationMatrix();
      starts.emplace_back(model.evaluate(observed,likelihood,flat(pose)).mean_nll,pose);
    }
    // MSVC's stable_sort temporary buffer rejects over-aligned Eigen payloads.
    // Rank scalar indexes instead, preserving sample order for tied scores and
    // leaving the native pose storage and its alignment untouched.
    std::vector<std::size_t> order(starts.size());
    std::iota(order.begin(),order.end(),std::size_t{0});
    std::stable_sort(order.begin(),order.end(),[&](std::size_t a,std::size_t b){
      return starts[a].first<starts[b].first;
    });
    SMLMLikelihoodFitOptions fit_options;fit_options.max_iterations=options.max_iterations;
    fit_options.max_translation_step=options.max_translation_step;
    fit_options.max_rotation_step=options.max_rotation_step;
    SMLMLikelihoodFitResult best;best.likelihood.mean_nll=std::numeric_limits<double>::infinity();
    for(std::size_t j=0;j<std::min<std::size_t>(2,starts.size());++j) {
      auto fitted=fit_smlm_likelihood_rigid(observed,model,likelihood,flat(starts[order[j]].second),fit_options);
      if(fitted.likelihood.mean_nll<best.likelihood.mean_nll) best=std::move(fitted);
    }
    double signal=0.0,total=0.0;
    for(std::size_t i=0;i<mass.size() && i<best.likelihood.signal_fraction.size();++i) {
      signal+=mass[i]*best.likelihood.signal_fraction[i];total+=mass[i];
    }
    signal=total>0.0?signal/total:0.0;
    if(!std::isfinite(best.likelihood.mean_nll) || signal<options.min_signal_fraction) {
      out.rejected_ids.push_back(id);continue;
    }
    const Matrix34 pose=Eigen::Map<const Matrix34>(best.transform.data());
    const auto transform=flat(inverse(pose));
    out.transforms.insert(out.transforms.end(),transform.begin(),transform.end());
    out.particle_ids.push_back(id);out.initial_nll.push_back(initial_nll);
    out.final_nll.push_back(best.likelihood.mean_nll);out.signal_fraction.push_back(signal);
    out.iterations.push_back(best.iterations);out.converged.push_back(best.converged?1:0);
  }
  return out;
}
IMPBFF_END_NAMESPACE
