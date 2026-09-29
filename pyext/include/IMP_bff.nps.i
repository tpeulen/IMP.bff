/* Fast-NPS direct dye and dye-model metadata, the network forward model,
   plus the thin Bayesian direct-FRET likelihood restraint on live existing
   IMP label sites. */

/* Sequences of numeric rows convert through the shared
   `const std::vector<std::vector<double> >&` typemap in IMP_bff.types.i
   (the mean-field one); NPS adds no wrapper machinery of its own. */

IMP_SWIG_VALUE(IMP::bff, NPSDirectDye, NPSDirectDyes);
IMP_SWIG_VALUE(IMP::bff, NPSDyeModel, NPSDyeModels);
IMP_SWIG_VALUE(IMP::bff, NPSNetworkDye, NPSNetworkDyes);
IMP_SWIG_VALUE(IMP::bff, NPSMeasurement, NPSMeasurements);
IMP_SWIG_OBJECT(IMP::bff, NPSIsotropicFRETEfficiencyRestraint,
                NPSIsotropicFRETEfficiencyRestraints);
IMP_SWIG_OBJECT(IMP::bff, NPSCloudPositionPriorRestraint,
                NPSCloudPositionPriorRestraints);
/* The objective derives GraphNode (already %shared_ptr'd above in
   core.i), so the sampler's set_objective takes it polymorphically. */
%shared_ptr(IMP::bff::NPSNetworkObjective);
/* The convergence estimators (Vehtari et al. 2021), already used by the
   C++ samplers; the chain type is the shared rows-of-doubles template. */
%include "IMP/bff/SamplerDiagnostics.h"
%feature("kwargs") IMP::bff::NPSIsotropicFRETEfficiencyRestraint::NPSIsotropicFRETEfficiencyRestraint;
%feature("kwargs") IMP::bff::NPSCloudPositionPriorRestraint::NPSCloudPositionPriorRestraint;
%include "IMP/bff/NPS.h"