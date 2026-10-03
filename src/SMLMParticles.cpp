/** \file SMLMParticles.cpp
 *  \brief Particle frames and rigid KDE registration for SMLM.
 */
#include <IMP/bff/SMLMParticles.h>
#include <IMP/bff/StructureIO.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE
namespace smlm_particles_internal {
using SmlmParticleVector3 = Eigen::Vector3d;
using SmlmParticleMatrix3 = Eigen::Matrix3d;
using SmlmParticleTransform34 = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

SmlmParticleTransform34 checked_transform(const std::vector<double>& v) {
  if (v.size() != 12) throw std::invalid_argument("rigid transform needs 12 entries");
  SmlmParticleTransform34 t = Eigen::Map<const SmlmParticleTransform34>(v.data());
  const SmlmParticleMatrix3 r = t.leftCols<3>();
  if (!t.allFinite() || (r.transpose()*r-SmlmParticleMatrix3::Identity()).norm() > 1e-6 ||
      std::abs(r.determinant()-1.0) > 1e-6)
    throw std::invalid_argument("transform must contain a finite proper rotation");
  return t;
}
std::vector<double> flattened(const SmlmParticleTransform34& t) {
  return std::vector<double>(t.data(), t.data()+12);
}
void validate_coordinates(const std::vector<double>& c) {
  if (c.size()%3) throw std::invalid_argument("coordinates need flat N x 3 entries");
  for (double x:c) if (!std::isfinite(x))
    throw std::invalid_argument("coordinates must be finite");
}
}

std::vector<double> transform_smlm_points(const std::vector<double>& c,
                                         const std::vector<double>& transform) {
  using namespace smlm_particles_internal;
  validate_coordinates(c);
  const SmlmParticleTransform34 t = checked_transform(transform);
  std::vector<double> out(c.size());
  for (std::size_t i=0;i<c.size();i+=3) {
    const SmlmParticleVector3 p = t.leftCols<3>()*Eigen::Map<const SmlmParticleVector3>(c.data()+i)+t.col(3);
    for(int d=0;d<3;++d) out[i+d]=p[d];
  }
  return out;
}

SMLMParticleFrames align_smlm_particles(const SMLMIndex& index,
    const std::vector<int>& ids, double robust_radius,
    int min_localizations, int max_localizations) {
  using namespace smlm_particles_internal;
  if (!std::isfinite(robust_radius) || robust_radius<0.0)
    throw std::invalid_argument("robust_radius must be finite and nonnegative");
  if(min_localizations<3 || max_localizations<0 ||
     (max_localizations>0 && max_localizations<min_localizations))
    throw std::invalid_argument("invalid particle count gates");
  std::vector<int> requested=ids;
  if(requested.empty()) {
    std::set<int> assigned;
    for(int id:index.get_particle_ids()) if(id>=0) assigned.insert(id);
    requested.assign(assigned.begin(),assigned.end());
  }
  std::set<int> seen;
  SMLMParticleFrames out;
  const auto& c=index.get_coordinates(); const auto& w=index.get_weights();
  for(int id:requested) {
    if(id<0 || !seen.insert(id).second)
      throw std::invalid_argument("particle IDs must be unique and nonnegative");
    const auto rows=index.get_particle_localizations(id);
    if(static_cast<int>(rows.size())<min_localizations ||
       (max_localizations>0 && static_cast<int>(rows.size())>max_localizations)) continue;
    if(rows.size()<3) throw std::invalid_argument("a particle needs at least three localizations");
    // Coordinate medians are a robust starting centre, not a radial mask.
    SmlmParticleVector3 center;
    for(int d=0;d<3;++d) {
      std::vector<double> col; col.reserve(rows.size());
      for(int i:rows) if(w[i]>0.0) col.push_back(c[3*i+d]);
      if(col.size()<3) throw std::invalid_argument("a particle needs three positive-weight points");
      auto mid=col.begin()+col.size()/2; std::nth_element(col.begin(),mid,col.end());
      center[d]=*mid;
    }
    std::vector<double> rw(rows.size()); double total=0.0;
    for(int iteration=0;iteration<12;++iteration) {
      SmlmParticleVector3 sum=SmlmParticleVector3::Zero(); total=0.0;
      for(std::size_t j=0;j<rows.size();++j) {
        int i=rows[j]; SmlmParticleVector3 p=Eigen::Map<const SmlmParticleVector3>(c.data()+3*i);
        double distance=(p-center).norm();
        rw[j]=w[i]*(robust_radius>0.0 && distance>robust_radius
                        ? robust_radius/distance:1.0);
        sum+=rw[j]*p; total+=rw[j];
      }
      if(!(total>0.0)) throw std::invalid_argument("particle has zero total weight");
      SmlmParticleVector3 next=sum/total;
      if((next-center).norm()<1e-8) {center=next;break;}
      center=next;
    }
    SmlmParticleMatrix3 covariance=SmlmParticleMatrix3::Zero(); total=0.0;
    for(std::size_t j=0;j<rows.size();++j) {
      int i=rows[j]; SmlmParticleVector3 delta=Eigen::Map<const SmlmParticleVector3>(c.data()+3*i)-center;
      double distance=delta.norm();
      double weight=w[i]*(robust_radius>0.0 && distance>robust_radius
                           ? robust_radius/distance:1.0);
      covariance+=weight*delta*delta.transpose(); total+=weight;
    }
    covariance/=total;
    Eigen::SelfAdjointEigenSolver<SmlmParticleMatrix3> eig(covariance);
    if(eig.info()!=Eigen::Success || eig.eigenvalues()[1]<=1e-12)
      throw std::invalid_argument("particle is degenerate or collinear");
    SmlmParticleVector3 normal=eig.eigenvectors().col(0);
    if(normal.z()<0.0) normal=-normal;
    SmlmParticleMatrix3 rotation=Eigen::Quaterniond::FromTwoVectors(normal,SmlmParticleVector3::UnitZ()).toRotationMatrix();
    SmlmParticleTransform34 transform; transform.leftCols<3>()=rotation; transform.col(3)=-rotation*center;
    const auto f=flattened(transform);
    out.transforms.insert(out.transforms.end(),f.begin(),f.end());
    for(int d=0;d<3;++d) {out.centers.push_back(center[d]);out.eigenvalues.push_back(eig.eigenvalues()[d]);}
    out.particle_ids.push_back(id); out.localization_counts.push_back(static_cast<int>(rows.size()));
  }
  return out;
}

SMLMIndex select_smlm_precision(const SMLMIndex& index,
                               const std::vector<double>& max_sigma) {
  if(!max_sigma.empty() && max_sigma.size()!=3)
    throw std::invalid_argument("precision limits must be three standard deviations");
  for(double v:max_sigma) if(!(v>0.0) || !std::isfinite(v))
    throw std::invalid_argument("precision limits must be finite and positive");
  const auto& c=index.get_coordinates();const auto& s=index.get_sigmas();
  const auto& w=index.get_weights();const auto& ids=index.get_particle_ids();
  std::vector<double> coords,sigmas,weights;std::vector<int> kept_ids;
  for(std::size_t i=0;i<w.size();++i) {
    bool keep=w[i]>0.0;
    for(int d=0;d<3 && keep;++d) if(!max_sigma.empty() && s[3*i+d]>max_sigma[d]) keep=false;
    if(!keep) continue;
    coords.insert(coords.end(),c.begin()+3*i,c.begin()+3*i+3);
    sigmas.insert(sigmas.end(),s.begin()+3*i,s.begin()+3*i+3);
    weights.push_back(w[i]);kept_ids.push_back(ids[i]);
  }
  return SMLMIndex(coords,sigmas,weights,kept_ids);
}

std::vector<double> get_smlm_assembly_frame(const std::vector<double>& operators,
                                          const std::vector<double>& landmarks) {
  using namespace smlm_particles_internal;
  if(operators.empty() || operators.size()%12 || landmarks.empty())
    throw std::invalid_argument("assembly frame needs rigid operators and 3D landmarks");
  validate_coordinates(landmarks);
  const std::size_t count=operators.size()/12;
  Eigen::MatrixXd a(3*count,3);Eigen::VectorXd b(3*count);
  for(std::size_t i=0;i<count;++i) {
    std::vector<double> values(operators.begin()+12*i,operators.begin()+12*i+12);
    const auto t=checked_transform(values);
    a.block<3,3>(3*i,0)=SmlmParticleMatrix3::Identity()-t.leftCols<3>();
    b.segment<3>(3*i)=t.col(3);
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(a,Eigen::ComputeThinU|Eigen::ComputeFullV);
  svd.setThreshold(1e-6);
  if(svd.rank()!=2) throw std::invalid_argument("assembly operators need one common rotational axis");
  SmlmParticleVector3 center=svd.solve(b),axis=svd.matrixV().col(2).normalized();
  Eigen::Index strongest=0;axis.cwiseAbs().maxCoeff(&strongest);
  if(axis[strongest]<0.0) axis=-axis;
  double axial=0.0;
  for(std::size_t i=0;i<landmarks.size();i+=3)
    axial+=axis.dot(Eigen::Map<const SmlmParticleVector3>(landmarks.data()+i)-center);
  center+=axis*(axial/(landmarks.size()/3));
  const double residual=(a*center-b).cwiseAbs().maxCoeff();
  return {center[0],center[1],center[2],axis[0],axis[1],axis[2],residual};
}

SMLMRegistrationResult refine_smlm_rigid(const SMLMIndex& ref,
    const std::vector<double>& moving,const std::vector<double>& initial,
    const std::vector<double>& weights,const SMLMRegistrationOptions& options,
    double max_rotation_step) {
  using namespace smlm_particles_internal;
  validate_coordinates(moving);
  if(moving.empty() || options.max_iterations<0 ||
     !(options.max_translation_step>0.0) || !std::isfinite(options.max_translation_step) ||
     !(max_rotation_step>0.0) || !std::isfinite(max_rotation_step) ||
     !(options.tolerance>0.0) || !std::isfinite(options.tolerance))
    throw std::invalid_argument("invalid rigid refinement coordinates or limits");
  SmlmParticleTransform34 t=checked_transform(initial);
  auto placed=transform_smlm_points(moving,flattened(t));
  auto current=ref.evaluate_score(placed,weights,options.cutoff_sigma,options.background);
  SMLMRegistrationResult result; result.initial_score=current.score;
  for(int iteration=0;iteration<options.max_iterations;++iteration) {
    SmlmParticleVector3 gt=SmlmParticleVector3::Zero(), gr=SmlmParticleVector3::Zero(); double scale=0.0;
    for(std::size_t i=0;i<moving.size();i+=3) {
      const SmlmParticleVector3 p=Eigen::Map<const SmlmParticleVector3>(placed.data()+i)-t.col(3);
      const SmlmParticleVector3 g=Eigen::Map<const SmlmParticleVector3>(current.gradient.data()+i);
      gt+=g;gr+=p.cross(g);scale+=p.squaredNorm();
    }
    if(gt.norm()+gr.norm()<options.tolerance) {result.converged=true;break;}
    SmlmParticleVector3 step_t=-gt, step_r=-gr/std::max(scale/static_cast<double>(moving.size()/3),1e-12);
    if(step_t.norm()>options.max_translation_step) step_t*=options.max_translation_step/step_t.norm();
    if(step_r.norm()>max_rotation_step) step_r*=max_rotation_step/step_r.norm();
    bool accepted=false;
    for(double alpha=1.0;alpha>=1e-8;alpha*=0.5) {
      SmlmParticleVector3 omega=alpha*step_r; double angle=omega.norm();
      SmlmParticleMatrix3 delta=angle>0.0?Eigen::AngleAxisd(angle,omega/angle).toRotationMatrix():SmlmParticleMatrix3::Identity();
      SmlmParticleTransform34 candidate;candidate.leftCols<3>()=delta*t.leftCols<3>();
      candidate.col(3)=t.col(3)+alpha*step_t;
      auto candidate_points=transform_smlm_points(moving,flattened(candidate));
      auto next=ref.evaluate_score(candidate_points,weights,options.cutoff_sigma,options.background);
      if(next.score<current.score) {
        double improvement=current.score-next.score;
        t=candidate;placed=std::move(candidate_points);current=std::move(next);
        accepted=true;result.iterations=iteration+1;
        if(improvement<options.tolerance) result.converged=true;
        break;
      }
    }
    if(!accepted || result.converged) break;
  }
  result.transform=flattened(t);result.score=current.score;
  return result;
}

void write_smlm_average_mrc(const SMLMAverageResult& average,
                           const std::string& path,int map) {
  if(map<0 || map>2 || average.grid_shape.size()!=3 || average.spacing.size()!=3)
    throw std::invalid_argument("map must be 0, 1 or 2 with valid grid dimensions");
  if(std::abs(average.spacing[0]-average.spacing[1])>1e-10 ||
     std::abs(average.spacing[0]-average.spacing[2])>1e-10)
    throw std::invalid_argument("MRC export requires isotropic spacing");
  const auto& v=map==0?average.values:(map==1?average.half1:average.half2);
  int nx=average.grid_shape[0],ny=average.grid_shape[1],nz=average.grid_shape[2];
  if(nx<=0 || ny<=0 || nz<=0 || v.size()!=static_cast<std::size_t>(nx)*ny*nz)
    throw std::invalid_argument("average map shape mismatch");
  std::vector<double> c_order(v.size());
  for(int x=0;x<nx;++x) for(int y=0;y<ny;++y) for(int z=0;z<nz;++z)
    c_order[(static_cast<std::size_t>(x)*ny+y)*nz+z]=v[x+static_cast<std::size_t>(nx)*(y+ny*z)];
  write_mrc_grid(path,c_order,nx,ny,nz,average.origin,average.spacing[0]);
}

IMPBFF_END_NAMESPACE
