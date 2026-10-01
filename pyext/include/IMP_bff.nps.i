/* Fast-NPS direct dye and dye-model metadata, the network forward model,
   and the direct-FRET likelihood and cloud prior terms (plain functions:
   the core builds without IMP). */

/* Sequences of numeric rows convert through the shared
   `const std::vector<std::vector<double> >&` typemap in IMP_bff.types.i
   (the mean-field one); NPS adds no wrapper machinery of its own. */

IMP_SWIG_VALUE(IMP::bff, NPSDirectDye, NPSDirectDyes);
IMP_SWIG_VALUE(IMP::bff, NPSDyeModel, NPSDyeModels);
IMP_SWIG_VALUE(IMP::bff, NPSNetworkDye, NPSNetworkDyes);
IMP_SWIG_VALUE(IMP::bff, NPSMeasurement, NPSMeasurements);
/* The objective derives GraphNode (already %shared_ptr'd above in
   core.i), so the sampler's set_objective takes it polymorphically. */
%shared_ptr(IMP::bff::NPSNetworkObjective);
/* The convergence estimators (Vehtari et al. 2021), already used by the
   C++ samplers; the chain type is the shared rows-of-doubles template. */
%include "IMP/bff/SamplerDiagnostics.h"
%feature("kwargs") IMP::bff::nps_isotropic_direct_score;
%feature("kwargs") IMP::bff::nps_cloud_prior_score;
%include "IMP/bff/NPS.h"