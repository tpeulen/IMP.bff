"""Fit user-specified expected emitter positions to one measured SMLM particle.

This is an IMP structural-modelling example of the forward observation
likelihood in Wu et al., DOI 10.1038/s41592-022-01676-z (PMC9834062).
Each measured localization retains its own positional precision. A rigid body
carries the supplied emitter positions; SMLMRestraint evaluates its native
likelihood and derivatives, and IMP's native ConjugateGradients optimizer moves
the body. There is no Python objective callback or coordinate export per step.

The model is arbitrary: no ring, symmetry, scaffold-to-label mapping, or scale
fit is imposed. Supply EXPECTED FLUOROPHORE POSITIONS, not a whole scaffold or
all atom coordinates. Choosing attachment sites, fluorophore offsets and any
linker/label uncertainty is a separate modelling decision. This example cannot
infer those from a structure.

Input formats
-------------
Measured CSV: native SMLMIO aliases, e.g. xnm,ynm,znm,locprecnm,locprecznm,
sitenumbers (or x,y,z,uncertainty_x,uncertainty_y,uncertainty_z,particle_id).
SMAP IDs are positive; zero means unassigned by default. Explicitly opt in to
zero-based source IDs with --zero-based-particle-ids to select particle 0.
Missing measured z/precision is rejected; photon/PSF widths never replace
positional uncertainty.
Model CSV: x,y,z with optional explicitly named emitter-weight column.
Model NPZ: ``coordinates`` of shape (N,3), optionally ``weights`` of shape (N,).
Change the coordinate key with --model-key. NPZ is decoded once and transcribed
to a temporary CSV without coordinate arithmetic; native SMLMIO performs unit
conversion and origin subtraction. CSV model precision defaults are placeholders
used by the reader only; they are never used as observation noise.

Units and frames
----------------
All IMP coordinates and likelihood options are in Angstrom. --source-origin is
the local ROI origin in the measured CSV's SOURCE unit, near the selected
particle centre. Native import computes (source_position - source_origin)*scale
and scales measured sigmas without shifting them. --roi-min/--roi-max are
explicit bounds in this resulting local Angstrom frame; all selected rows must
lie inside. --model-origin is subtracted in the model file's source unit before
native conversion. Initial/final poses map this model frame to the local
measurement frame. Absolute density log scores depend on that physical unit.

Example (replace the source origin and ROI with this particle's actual frame)::

    python examples/imaging/smlm_structure.py cell1_sml.csv emitters.npz \
        --particle-id 920 --measurement-unit nm --model-unit angstrom \
        --source-origin 12000 8000 0 \
        --roi-min -1500 -1500 -1500 --roi-max 1500 1500 1500 \
        --steps 200 --max-change 30 --intrinsic-sigma 20

This is local optimization: provide an initial pose with model/data overlap.
--initial-quaternion is unit (w,x,y,z); --initial-translation is Angstrom.
The background mixture gives finite outlier scores but supplies no attraction
when every emitter is beyond the Gaussian cutoff. A lower final score alone
does not establish a unique pose or validate the biological label model.
--max-change bounds changes in IMP's optimized attribute coordinates, including
quaternion components; it is NOT a maximum rotation angle or a hard pose bound.
"""

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import IMP.algebra
import IMP.core
import numpy as np

import IMP
from IMP import bff

UNIT_TO_ANGSTROM = {"angstrom": 1.0, "nm": 10.0}


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def nonnegative_float(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_integer(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def read_emitter_model(args):
    """Decode inputs once; native import owns coordinate/weight validation."""
    settings = bff.SMLMCSVOptions()
    settings.allow_missing_sigmas = True
    settings.default_sigmas = [1.0, 1.0, 1.0]
    settings.weight_column = args.model_weight_column
    scale = UNIT_TO_ANGSTROM[args.model_unit]
    if args.model_points.suffix.lower() == ".csv":
        return bff.read_smlm_csv(str(args.model_points), scale, args.model_origin, settings)
    if args.model_points.suffix.lower() != ".npz":
        raise ValueError("model_points must be a CSV or NPZ file")
    if args.model_weight_column:
        raise ValueError("--model-weight-column is for CSV; NPZ uses the optional weights array")
    with np.load(args.model_points, allow_pickle=False) as archive:
        if args.model_key not in archive:
            raise ValueError(f"NPZ lacks coordinate array {args.model_key!r}")
        coordinates = archive[args.model_key]
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not len(coordinates):
            raise ValueError("NPZ model coordinates must have nonempty shape (N,3)")
        weights = archive.get("weights")
        if weights is not None and weights.shape != (len(coordinates),):
            raise ValueError("NPZ weights must have shape (N,)")
        with tempfile.TemporaryDirectory(prefix="smlm-emitter-input-") as directory:
            path = Path(directory) / "emitters.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["x", "y", "z"] + (["model_weight"] if weights is not None else []))
                for row, coordinate in enumerate(coordinates):
                    writer.writerow(coordinate.tolist() + ([weights[row].item()] if weights is not None else []))
            settings.weight_column = "model_weight" if weights is not None else ""
            return bff.read_smlm_csv(str(path), scale, args.model_origin, settings)


def pose_for_report(body):
    """Export one completed pose for human-readable reporting, not computation."""
    transform = body.get_reference_frame().get_transformation_to()
    rotation, translation = transform.get_rotation(), transform.get_translation()
    matrix = [list(rotation.get_rotation_matrix_row(axis)) + [translation[axis]] for axis in range(3)]
    return {"model_to_local_measurement_3x4": matrix,
            "quaternion_wxyz": list(rotation.get_quaternion()),
            "translation_angstrom": list(translation)}


def run(args):
    quaternion_norm = math.hypot(*args.initial_quaternion)
    if not math.isfinite(quaternion_norm) or abs(quaternion_norm-1.0) > 1e-6:
        raise ValueError("initial quaternion must have unit length")
    if args.particle_id < 0 or (args.particle_id == 0 and not args.zero_based_particle_ids):
        raise ValueError("particle 0 requires --zero-based-particle-ids; IDs must be nonnegative")
    csv_options = bff.SMLMCSVOptions()
    csv_options.weight_column = args.observation_weight_column
    csv_options.zero_particle_id_is_unassigned = not args.zero_based_particle_ids
    full_data = bff.read_smlm_csv(str(args.localizations), UNIT_TO_ANGSTROM[args.measurement_unit],
                                 args.source_origin, csv_options)
    measured = bff.select_smlm_particles(full_data, [args.particle_id])
    del full_data
    emitters = read_emitter_model(args)
    if emitters.get_number_of_localizations() == 0:
        raise ValueError("the expected-emitter model is empty")

    settings = bff.SMLMLikelihoodOptions()
    settings.roi_min, settings.roi_max = args.roi_min, args.roi_max
    settings.background_fraction = args.background_fraction
    settings.intrinsic_sigma = args.intrinsic_sigma
    settings.cutoff_sigma = args.cutoff_sigma

    IMP.set_log_level(IMP.SILENT)
    model = IMP.Model()
    body_particle = IMP.Particle(model)
    identity = IMP.algebra.get_identity_transformation_3d()
    body = IMP.core.RigidBody.setup_particle(body_particle, IMP.algebra.ReferenceFrame3D(identity))
    coordinates = emitters.get_coordinates()
    particles = []
    # One-time model construction. The objective subsequently reads XYZs in C++.
    for offset in range(0, len(coordinates), 3):
        particle = IMP.Particle(model)
        IMP.core.XYZ.setup_particle(particle, IMP.algebra.Vector3D(*coordinates[offset:offset+3]))
        body.add_member(particle)
        particles.append(particle)
    rotation = IMP.algebra.Rotation3D(*args.initial_quaternion)
    initial_transform = IMP.algebra.Transformation3D(rotation, IMP.algebra.Vector3D(*args.initial_translation))
    body.set_reference_frame(IMP.algebra.ReferenceFrame3D(initial_transform))
    body.set_coordinates_are_optimized(True)
    model.update()

    restraint = bff.SMLMRestraint(model, particles, measured, settings,
                                 emitters.get_weights(), args.sum_score)
    restraint.set_weight(args.restraint_weight)
    scoring = IMP.core.RestraintsScoringFunction([restraint])
    initial_score = scoring.evaluate(False)
    initial_pose = pose_for_report(body)
    optimizer = IMP.core.ConjugateGradients(model)
    optimizer.set_scoring_function(scoring)
    optimizer.set_gradient_threshold(args.gradient_threshold)
    optimizer.set_max_change(args.max_change)
    optimizer.optimize(args.steps)  # All optimization and derivative work stays native.
    final_score = scoring.evaluate(False)

    return {"particle_id": args.particle_id,
            "zero_based_particle_ids": args.zero_based_particle_ids,
            "measured_localizations": measured.get_number_of_localizations(),
            "expected_emitters": len(particles), "coordinate_unit": "angstrom",
            "measurement_source_unit": args.measurement_unit,
            "source_origin": args.source_origin, "model_source_unit": args.model_unit,
            "model_origin": args.model_origin,
            "roi_min_angstrom": args.roi_min, "roi_max_angstrom": args.roi_max,
            "score_mode": "weighted_sum_nll" if args.sum_score else "weighted_mean_nll",
            "restraint_weight": args.restraint_weight,
            "initial_score": initial_score, "final_score": final_score,
            "initial_pose": initial_pose, "final_pose": pose_for_report(body),
            "limits": {"native_steps": args.steps, "max_attribute_change": args.max_change,
                       "gradient_threshold": args.gradient_threshold,
                       "cutoff_sigma": args.cutoff_sigma,
                       "intrinsic_sigma_angstrom": args.intrinsic_sigma,
                       "background_fraction": args.background_fraction}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("localizations", type=Path, help="measured 3D paper/localization CSV")
    parser.add_argument("model_points", type=Path, help="user-specified expected emitter CSV or NPZ")
    parser.add_argument("--particle-id", type=nonnegative_integer, required=True)
    parser.add_argument("--zero-based-particle-ids", action="store_true",
                        help="explicitly treat source group 0 as an assigned particle")
    parser.add_argument("--measurement-unit", choices=UNIT_TO_ANGSTROM, required=True)
    parser.add_argument("--model-unit", choices=UNIT_TO_ANGSTROM, required=True)
    parser.add_argument("--source-origin", nargs=3, type=finite_float, required=True)
    parser.add_argument("--model-origin", nargs=3, type=finite_float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--roi-min", nargs=3, type=finite_float, required=True, help="local Angstrom bounds")
    parser.add_argument("--roi-max", nargs=3, type=finite_float, required=True, help="local Angstrom bounds")
    parser.add_argument("--model-key", default="coordinates", help="NPZ coordinate array key")
    parser.add_argument("--model-weight-column", default="", help="CSV emitter weights; default uniform")
    parser.add_argument("--observation-weight-column", default="", help="explicit CSV weights; default equal")
    parser.add_argument("--initial-quaternion", nargs=4, type=finite_float, default=[1.0, 0.0, 0.0, 0.0])
    parser.add_argument("--initial-translation", nargs=3, type=finite_float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--background-fraction", type=nonnegative_float, default=0.03)
    parser.add_argument("--intrinsic-sigma", type=nonnegative_float, default=0.0, help="independent blur in Angstrom")
    parser.add_argument("--cutoff-sigma", type=positive_float, default=6.0)
    parser.add_argument("--steps", type=positive_integer, default=200)
    parser.add_argument("--gradient-threshold", type=positive_float, default=1e-6)
    parser.add_argument("--max-change", type=positive_float, default=30.0, help="native attribute step bound")
    parser.add_argument("--restraint-weight", type=positive_float, default=1.0)
    parser.add_argument("--sum-score", action="store_true", help="weighted sum NLL instead of weighted mean")
    args = parser.parse_args()
    if args.particle_id == 0 and not args.zero_based_particle_ids:
        parser.error("--particle-id 0 requires --zero-based-particle-ids")
    if args.background_fraction >= 1:
        parser.error("--background-fraction must be in [0,1)")
    try:
        report = run(args)
        output = json.dumps(report, indent=2, allow_nan=False)
    except (ValueError, RuntimeError, OSError, TypeError) as error:
        parser.exit(1, f"SMLM structural modelling failed: {error}\n")
    print(output)


if __name__ == "__main__":
    main()
