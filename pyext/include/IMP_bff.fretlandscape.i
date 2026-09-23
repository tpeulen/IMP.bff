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

%{
#include <IMP/bff/FRETLandscape.h>
%}

IMP_SWIG_VALUE(IMP::bff, FRETLandscapeFitOptions, FRETLandscapeFitOptionsList);
IMP_SWIG_VALUE(IMP::bff, FRETLandscapeFit, FRETLandscapeFits);
IMP_SWIG_VALUE(IMP::bff, FRETLandscapePhotons, FRETLandscapePhotonsList);
IMP_SWIG_VALUE(IMP::bff, FRETLandscapeInitialGuessOptions, FRETLandscapeInitialGuessOptionsList);
IMP_SWIG_VALUE(IMP::bff, FRETLandscapeInitialGuess, FRETLandscapeInitialGuesses);
IMP_SWIG_VALUE(IMP::bff, FRETLandscapeLaplace, FRETLandscapeLaplaces);
IMP_SWIG_VALUE(IMP::bff, FRETLandscapeModel, FRETLandscapeModels);
%naturalvar IMP::bff::FRETLandscapeInitialGuessOptions::bin_widths;
%naturalvar IMP::bff::FRETLandscapeInitialGuessOptions::diffusions;
%naturalvar IMP::bff::FRETLandscapeInitialGuessOptions::backgrounds;
%naturalvar IMP::bff::FRETLandscapeFitOptions::fixed;

%include "IMP/bff/FRETLandscape.h"

/* Burstwise, microtime-resolved inference of a hidden conformational process
   shared by a network of FRET pairs (FRETNetwork.h). */
%{
#include <IMP/bff/FRETNetwork.h>
%}

IMP_SWIG_VALUE(IMP::bff, FRETHiddenProcess, FRETHiddenProcesses);
IMP_SWIG_VALUE(IMP::bff, FRETDye, FRETDyes);
IMP_SWIG_VALUE(IMP::bff, FRETInstrument, FRETInstruments);
IMP_SWIG_VALUE(IMP::bff, FRETMeasurement, FRETMeasurements);

%include "IMP/bff/FRETNetwork.h"
