/*
 * Axes, profiles and histograms read off the frames of a trajectory: the
 * array kernels of `imp_bff analyze-trajectories` (TrajectoryAnalysis.h,
 * core). The IMP half is IMP_bff.probetrajectorydensity.i in layer.i.
 */

%feature("kwargs") IMP::bff::point_cloud_principal_axis;
%feature("kwargs") IMP::bff::write_binned_profile;
%feature("kwargs") IMP::bff::write_radial_histogram;
%feature("kwargs") IMP::bff::write_orientation_profile;

%include "IMP/bff/TrajectoryAnalysis.h"
