/*
 * Bounded scalar minimisation: minimize_lbfgsb() and minimize_nelder_mead().
 *
 * MinimizeObjective is a director the Python caller subclasses (ChiSurf's
 * chisurf.core.math.numerics.minimize does); IMP_SWIG_DIRECTOR rather than a
 * bare director feature for the same lifetime reason as FitMinimizerObserver
 * -- the C++ side calls it for the whole run.
 *
 * evaluate_with_gradient() returns f and the gradient as one ndarray; it is
 * read through the buffer, not walked as a Python sequence (see the
 * FitResidualFunction typemap in IMP_bff.core.i for the measured cost).
 */
IMP_SWIG_OBJECT(IMP::bff, MinimizeObjective, MinimizeObjectives);
IMP_SWIG_DIRECTOR(IMP::bff, MinimizeObjective);
%typemap(directorout) std::vector<double> evaluate_with_gradient {
  PyObject* bff_arr = PyArray_FROMANY($input, NPY_DOUBLE, 0, 1,
                                      NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  if (!bff_arr) throw Swig::DirectorMethodException();
  const npy_intp bff_n = PyArray_SIZE(reinterpret_cast<PyArrayObject*>(bff_arr));
  const double* bff_d =
      static_cast<const double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(bff_arr)));
  $result.assign(bff_d, bff_d + bff_n);
  Py_DECREF(bff_arr);
}
%include "IMP/bff/Minimize.h"
