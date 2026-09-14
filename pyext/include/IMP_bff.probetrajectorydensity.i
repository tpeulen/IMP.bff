/*
 * Where a probe went over a trajectory (ProbeTrajectoryDensity.h, connection
 * layer): the particle axes, the region density maps, and the options of the
 * whole analysis. The run itself is the program's -- `imp_bff
 * analyze-trajectories` -- and is not wrapped.
 */

IMP_SWIG_VALUE(IMP::bff, ProbeTrajectoryDensityOptions, ProbeTrajectoryDensityOptionsList);

%feature("kwargs") IMP::bff::probe_atom_name;
%feature("kwargs") IMP::bff::probe_fixed_axis;
%feature("kwargs") IMP::bff::probe_long_axis;
%feature("kwargs") IMP::bff::write_probe_region_density;

%include "IMP/bff/ProbeTrajectoryDensity.h"
