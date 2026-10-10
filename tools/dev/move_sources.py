#!/usr/bin/env python3
"""Lay src/ out by theme (one level, the depth IMP's setup_cmake globs).

Headers stay flat in include/ -- IMP links only include/*.h and
include/internal/*.h into its build tree -- so each theme names its public
headers in src/<theme>/Headers.cmake. The script is idempotent: sources that
already sit in their theme are left alone. Run from the repo root.
"""
import os
import re
import subprocess
import sys

THEMES = {
    "util": "BffSettings CommandLine ComputeBackend IMPCompatibility Numerics Odeint OpenMP Registry SparseLinearAlgebra SpecialFunctions",
    "graph": "GraphEvaluation GraphExpression GraphNode GraphNodeRegistry GraphPort GraphSession",
    "fit": "Convolution FitChiSquared FitDataset FitJointChiSquared FitMinimizer FitObjective FitStatistics GeneralizedNormalCurve InferenceCanonicalForm InferenceFactorGraph InferenceFactorGraphFromFit InferenceGaussianElimination LinearLeastSquares MaxEnt MaxEntSpectrum Minimize Optimization SpectrumGrid StripMask",
    "bayesian": "CausalLinearGaussian CounterfactualDistanceNetwork CounterfactualMarkovChain BayesianDecayModel BayesianDecayPosterior BayesianDecaySampling BayesianDeltaMethod BayesianFisherScoring BayesianLaplace BayesianMeasuredResponse BayesianPSpline BayesianTransferTensors BayesianTransformedGaussianPrior BayesianTransforms Distributions MCMCSampler NutsKernel SamplerDiagnostics SamplerKernels SamplerWarmup Sampling",
    "spectroscopy": "AcceptorDensityDecay FCS KineticNetwork KineticSchemeNode LifetimeSpectrumMixture PhotophysicsAnisotropySpectrumNode PhotophysicsCrosstalkMatrix PhotophysicsInteractionTerms PhotophysicsLifetimeSpectrum PhotophysicsLifetimeSpectrumNode PhotophysicsPhotonSimulation PhotophysicsPolarisation PhotophysicsQuenching PhotophysicsQuenchingModel PhotophysicsTransferKinetics PhotophysicsTransferKineticsNode States TCSPCDecay",
    "fret": "DiscreteDistances FRET FRETAcceptorDensity FRETExchange FRETLandscape FRETLandscapeGrid FRETNetwork FRETNetworkSimulation FRETOrientationFactor FRETRotamer FRETSpectrumNode GaussianDistances LabelizerFeatures LabelizerFRET LabelizerIO LabelizerScore PolymerChain PolymerDistances TabulatedDistances",
    "search": "ModelSearch ModelSearchPolicy ModelSearchSelfPlay ModelSearchSpec PhotonExperiment",
    "learn": "EmbeddingIndex HMMSurrogate NeuralNet NeuralNetTraining ProteinLanguageModel",
    "probe": "BrownianWalk ContactPotentials DensityGrid DiffusionSolver ElasticNetwork FPS FPSRotamer GridDiffusionSolver Linker NPS OccupancyGrid PathMap ProbeAccessibleVolume ProbeAccessibleVolumeBuilder ProbeComponentTemplate ProbeDataPaths ProbeDiffusionSimulation ProbeForceFieldCIF ProbeLibrary ProbeNetworkSelection ProbePairSelection ProbePotentialTables ProbeRestraints ProbeRotamer ProbeRotamerLibrary ProbeSampling ProbeSimulation ProbeTopology RRT",
    "structure": "Clustering Consurf DockingPrecision HierarchyFrame MMFDBProfile Mol2IO MolecularGraph ProteinSidechainRotamerLibrary RmfIO RotamerScoring SelectionExpression SolventAccessibleSurface StructureIO StructureTable TrajectoryAnalysis TrajectoryIO VdwRadii ZMatrix",
    "sequence": "SequenceAlignment SequenceClusters SequenceCoevolution SequenceConservation SequenceDatabase SequenceHomologs SequenceMSA SequenceSearch SequenceServer",
    "smlm": "SMLM SMLMGaussianOverlap SMLMIO SMLMLikelihood SMLMParticleRegistration SMLMParticles",
}
THEMES = {k: v.split() for k, v in THEMES.items()}
# sources with no public header of their own go with the theme they serve
PRIVATE = {
    "probe": "CommandLineFpsDistance DiffusionSolverKrylov",
    "fret": "CommandLineLabelizer FRETLandscapeSimulation FRETNetworkEmission FRETNetworkFilter FRETNetworkFit FRETNetworkStates",
    "structure": "CommandLinePotentials CommandLineRmsd CommandLineTrajectory",
    "sequence": "CommandLineSequence",
}
PRIVATE = {k: v.split() for k, v in PRIVATE.items()}
# stays flat: the connection layer is reached through it
FLAT = {"ImpLayer", "ThemeSources"}

TITLE = {
    "util": "Utilities: settings, command line, numerics, registry",
    "graph": "Expression graph: nodes, ports, sessions, evaluation",
    "fit": "Curve fitting: objectives, minimizers, factor-graph inference, MaxEnt",
    "bayesian": "Bayesian inference: decay posteriors, samplers, causal models",
    "spectroscopy": "Photophysics and decays: lifetimes, quenching, kinetics, FCS",
    "fret": "FRET: transfer, networks, landscapes, labelizer, polymer distances",
    "search": "Model search: MCTS policy, self-play, search specs",
    "learn": "Learned models: neural nets, embeddings, protein language model",
    "probe": "Dye probes: accessible volumes, rotamers, diffusion, FPS",
    "structure": "Structure and trajectory I/O, selections, clustering, surfaces",
    "sequence": "Sequences: alignment, MSA, conservation, coevolution",
    "smlm": "Single-molecule localisation: particles, likelihoods, registration",
}


def git(*a):
    subprocess.check_call(["git", *a])


def main():
    names = {}
    for theme, hs in THEMES.items():
        for h in hs:
            names[h] = theme
    for theme, ss in PRIVATE.items():
        for s in ss:
            names[s] = theme
    for theme in THEMES:
        os.makedirs(f"src/{theme}", exist_ok=True)
    moved = 0
    for fn in sorted(os.listdir("src")):
        if not fn.endswith(".cpp"):
            continue
        stem = fn[:-4]
        if stem in FLAT:
            continue
        theme = names[stem]
        git("mv", f"src/{fn}", f"src/{theme}/{fn}")
        moved += 1
    # sibling includes of src/internal move one level down
    for theme in THEMES:
        for fn in os.listdir(f"src/{theme}"):
            if not fn.endswith(".cpp"):
                continue
            p = f"src/{theme}/{fn}"
            s = open(p).read()
            n = re.sub(r'#include "internal/', '#include "../internal/', s)
            if n != s:
                open(p, "w").write(n)
    for theme, hs in THEMES.items():
        priv = PRIVATE.get(theme, [])
        out = [f"# Public headers of the `{theme}` theme. IMP links only include/*.h and",
               "# include/internal/*.h into its build tree, so headers stay flat and this",
               "# list is what assigns them (test/test_theme_headers.py checks it).",
               f'set(imp_bff_{theme}_headers "' + ";".join(h + ".h" for h in hs) + '")',
               "# Sources without a public header of their own.",
               f'set(imp_bff_{theme}_private_sources "' + ";".join(p + ".cpp" for p in priv) + '")', ""]
        open(f"src/{theme}/Headers.cmake", "w").write("\n".join(out))
    open("src/Themes.cmake", "w").write(
        "# The themes under src/, in dependency order (a theme may use those before it).\n"
        "# Each src/<theme>/Headers.cmake names its public headers; src/imp/ is the\n"
        "# connection layer to IMP and is listed in src/imp/Headers.cmake.\n"
        'set(imp_bff_themes "' + ";".join(THEMES) + '")\n')
    out = ["// IMP's unity build (tools/build/setup_all.py) compiles src/*.cpp and",
           "// src/internal/*.cpp into one translation unit and looks no deeper; the themed",
           "// sources are reached through this file, as the connection layer is through",
           "// ImpLayer.cpp. Per-cpp and standalone builds name the sources directly and",
           "// leave this file out. test/test_theme_headers.py checks the list is complete.", ""]
    # by file name, the order the flat directory had: the unity translation
    # unit shares anonymous namespaces, so a different order can change which
    # helper a name resolves to.
    files = sorted((fn, theme) for theme in THEMES for fn in os.listdir(f"src/{theme}")
                   if fn.endswith(".cpp"))
    out += [f'#include "{theme}/{fn}"' for fn, theme in files]
    open("src/ThemeSources.cpp", "w").write("\n".join(out) + "\n")
    print("moved", moved)


if __name__ == "__main__":
    sys.exit(main())
