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
%feature("kwargs") IMP::bff::NPSIsotropicFRETEfficiencyRestraint::NPSIsotropicFRETEfficiencyRestraint;
%include "IMP/bff/NPS.h"