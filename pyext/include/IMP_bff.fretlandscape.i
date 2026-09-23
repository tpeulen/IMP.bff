/*
 * The photon-by-photon free-energy landscape likelihood (Dingeldein & Covino,
 * arXiv:2608.21061): the SqRA grid, its tridiagonal eigensolver, the natural
 * cubic spline landscape, and FRETLandscapeModel.
 *
 * Matrices come back flat and row-major; `.reshape(n, n)` restores them.
 */
%{
#include <IMP/bff/FRETLandscapeGrid.h>
%}

IMP_SWIG_VALUE(IMP::bff, TridiagonalEigenSystem, TridiagonalEigenSystems);
IMP_SWIG_VALUE(IMP::bff, NaturalCubicSpline, NaturalCubicSplines);

%ignore IMP::bff::NaturalCubicSpline::evaluate_point;

%include "IMP/bff/FRETLandscapeGrid.h"
