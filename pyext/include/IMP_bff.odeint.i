/*
 * odeint(): LSODA on a grid of output times, scipy.integrate.odeint's shape.
 *
 * OdeFunction is a director the Python caller subclasses (ChiSurf's
 * chisurf.core.math.numerics.odeint does). Its `evaluate` returns dy/dt as
 * an ndarray; the `std::vector<double> evaluate` directorout typemap in
 * IMP_bff.core.i (written for FitResidualFunction) reads it through the
 * buffer here too.
 */
IMP_SWIG_OBJECT(IMP::bff, OdeFunction, OdeFunctions);
IMP_SWIG_DIRECTOR(IMP::bff, OdeFunction);
%include "IMP/bff/Odeint.h"
