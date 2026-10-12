#ifndef IMPBFF_PROBE_ACCESSIBLE_VOLUME_DEFAULTS_H
#define IMPBFF_PROBE_ACCESSIBLE_VOLUME_DEFAULTS_H

/**
 *  \file IMP/bff/ProbeAccessibleVolumeDefaults.h
 *  \brief The accessible-volume defaults, read from FPS and Olga.
 *
 * Every door onto an accessible volume -- the decorator, the array builder,
 * the fps.json readers, the schema, the command line -- takes its defaults
 * from here, so an omitted parameter means the same thing whichever door it
 * was omitted at. They are the reference programs' own values (owner,
 * 2026-10-11: "av defaults should be fps/olga defaults, take from there"),
 * read from their source, not inferred:
 *
 * | parameter | FPS | Olga | here |
 * |---|---|---|---|
 * | linker length | 20 (`Fps/data/linker.txt` row 1, "Alexa488 L linker", the dye `AVinterface.cs:74` selects first) | none (0, must be set) | 20 |
 * | linker width | 4.5 (same row) | none (0) | 4.5 |
 * | AV1 dye radius | 3.5 (same row) | none (0) | 3.5 |
 * | AV3 dye radii | 5, 4.5, 1.5 (same row) | none (0, 0, 0) | 5, 4.5, 1.5 |
 * | grid | `max(0.2 min(L, W, 2R_i), 0.4)` (`AVEngine.cs:568`): 0.9 for every AV1 preset, 0.6 for AV3 | 0.9 (`PositionSimulation.h`, AV1 and AV3) | 0.9 |
 * | simulation type | "Simple AV" (`simcombobox` index 0) | AV1 (`Position.h`) | AV1 |
 * | seed sphere | `LinkerInitialSphere = 0.5` x W, unconditional (`ProjectData.cs`, `av_routines.cpp:136`) | obstacles cleared within `(lround(W/2/dg) + 1) dg` (`fretAV.cpp`, `ignoreSphere`) | 0.5 W (FPS) |
 * | allowed sphere (`allowed_sphere_radius`) | the seed sphere above: `LinkerInitialSphere` 0.5 (`AVGlobalParameters`, `ProjectData.cs:169`, `AVEngine.cs:78`) x W 4.5 = **2.25 A** for its first preset | `_allowedSphereRadius = 0.0` (`Position.h`), off; when set, strips the atoms within it (`Position.cpp`, `stripExpression`) | 2.25 A, on (#AV_DEFAULT_ALLOWED_SPHERE) |
 * | attachment atom | dropped from the obstacles, the atom alone (`av_routines.cpp:67`, `if (i == atom_i) continue;`) | kept (`noself`), cleared by `ignoreSphere` | dropped, the atom alone (FPS) |
 * | linker's first step | `LinkSearchNodes = 3` voxels (a 7x7x7 cube) from every seed | the stencil alone | FPS's (PathMap::set_source_first_step) |
 * | link search | the same 3-voxel cube hop from *every* reached node (`av_routines.cpp:156-209`) | the 74 stencil | FPS's: `link_search = "fps"`, stencil 342 (PathMap::set_cube_stencil) |
 * | contact volume | none | thickness 0, trapped fraction -1 (off) | off |
 * | min. sphere volume fraction | none | 0 | 0 |
 *
 * Where the two disagree the search follows FPS: the seed sphere, the first
 * step and the link search. FPS hops `LinkSearchNodes` voxels from *every*
 * node of the search, landing on any free voxel of the 7x7x7 cube across
 * whatever lies between, and that is the default here too (owner,
 * 2026-10-12, reversing 2026-10-11's "av go fast by default"). It is FPS's
 * search to the voxel (same reached set as FPS's propagation, path lengths to
 * float precision) and, with the exact shortcuts of PathMap's lattice search,
 * ~1.1x the seed-only search's cost. On T4L 148L's CB sites under FPS's radii
 * it brings the volume from 0.93 to 0.99 of FPS's own routine. The seed-only
 * search stays selectable as `link_search = "fast"` (stencil 74);
 * `okf/validation/fps_screening_ab.md` measures both -- including that the
 * empty sites at W 4.5 are the radii set's, not the search's.
 * Every consumer, the labelizer's refinement included, takes these values
 * ("all use same defaults"). The grid is the one value both give for an AV1 (0.9); FPS's
 * relative rule would take an AV3 with a 1.5 A radius to 0.6, and that rule
 * is #IMP::bff::fps_default_grid for a caller who wants it.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/bff_config.h>
#include <string>

IMPBFF_BEGIN_NAMESPACE

//! Linker length, A (FPS's first preset).
constexpr double AV_DEFAULT_LINKER_LENGTH = 20.0;
//! Linker width (diameter), A (FPS's first preset).
constexpr double AV_DEFAULT_LINKER_WIDTH = 4.5;
//! AV1 dye radius, A (FPS's first preset).
constexpr double AV_DEFAULT_RADIUS1 = 3.5;
//! AV3 dye radii, A (FPS's first preset).
constexpr double AV_DEFAULT_AV3_RADIUS1 = 5.0;
constexpr double AV_DEFAULT_AV3_RADIUS2 = 4.5;
constexpr double AV_DEFAULT_AV3_RADIUS3 = 1.5;
//! Grid spacing, A: Olga's fixed default, and FPS's rule for every AV1 preset.
constexpr double AV_DEFAULT_SIMULATION_GRID_RESOLUTION = 0.9;
//! FPS `LinkerInitialSphere`: the seed sphere around the anchor, in linker widths.
constexpr double AV_DEFAULT_LINKER_INITIAL_SPHERE = 0.5;
//! The allowed sphere, A: the clearance around the attachment atom inside
//! which the linker ignores obstacles. **On by default** (owner, 2026-10-12:
//! "allowed sphere radius must be enabled by default"), at FPS's value:
//! `LinkerInitialSphere` (0.5, `ProjectData.cs:169`) times FPS's first-preset
//! linker width (4.5, `Fps/data/linker.txt` row 1; every FPS preset has it),
//! i.e. 2.25 A. It is the default of every typed door's
//! `allowed_sphere_radius`. A door that reads a *position* -- the fps.json
//! readers, the command line -- takes FPS's rule for that position's own
//! width instead, `LinkerInitialSphere * W` (#av_derived_allowed_sphere_radius),
//! which is this value at the default width: FPS's screening document has
//! widths of 2.5, 3.5 and 4.5 A, and a flat 2.25 for all of them moved 3GUN
//! away from FPS. A *negative* value means that rule at any door.
constexpr double AV_DEFAULT_ALLOWED_SPHERE =
        AV_DEFAULT_LINKER_INITIAL_SPHERE * AV_DEFAULT_LINKER_WIDTH;
//! FPS `LinkSearchNodes`: how many voxels the linker hops at a time.
constexpr int AV_DEFAULT_LINK_SEARCH_NODES = 3;
//! The search stencil of FPS's link search: the (2n+1)^3 cube of
//! `LinkSearchNodes` hops but the node itself, 342 neighbours for n = 3.
constexpr int AV_SEARCH_STENCIL_FPS =
        (2 * AV_DEFAULT_LINK_SEARCH_NODES + 1) * (2 * AV_DEFAULT_LINK_SEARCH_NODES + 1)
        * (2 * AV_DEFAULT_LINK_SEARCH_NODES + 1) - 1;
//! The stencil the seed-only ("fast") search walks after its first step:
//! LabelLib's 74 reference metric.
constexpr int AV_SEARCH_STENCIL_FAST = 74;
//! The default search stencil: FPS's link search.
constexpr int AV_DEFAULT_SEARCH_STENCIL = AV_SEARCH_STENCIL_FPS;
//! The default link search, by name: "fps" (the cube hop from every node,
//! #AV_SEARCH_STENCIL_FPS) or "fast" (the hop out of the seed only, then the
//! #AV_SEARCH_STENCIL_FAST stencil).
constexpr const char *AV_DEFAULT_LINK_SEARCH = "fps";

//! The stencil a link-search name selects ("fps" or "fast"); throws otherwise.
IMPBFFEXPORT int av_link_search_stencil(const std::string &link_search);
//! The link-search name of a stencil: "fps" for #AV_SEARCH_STENCIL_FPS,
//! "fast" for every radius stencil (they all hop out of the seed only).
IMPBFFEXPORT std::string av_link_search_name(int search_stencil);

//! The source clearance a negative `allowed_sphere_radius` derives.
/*! FPS's seed: every voxel within `LinkerInitialSphere * W` of the anchor is
    open to the linker whatever the obstacles say (`av_routines.cpp:136`). */
inline double av_derived_allowed_sphere_radius(double linker_width) {
    return AV_DEFAULT_LINKER_INITIAL_SPHERE * linker_width;
}

//! How far, along an axis, the linker's first step reaches from its anchor, A.
/*! FPS seeds a sphere (`seed_radius`, `LinkerInitialSphere * W` by default)
    and hops `LinkSearchNodes` voxels from every seed
    (`av_routines.cpp:156-209`): `seed + 3 dg` along an axis, further along a
    cube diagonal. PathMap::set_source_first_step takes the step itself; this
    is its straight-line extent, which a stencil does not walk. */
inline double av_first_step_reach(double seed_radius, double grid) {
    return seed_radius + AV_DEFAULT_LINK_SEARCH_NODES * grid;
}

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_PROBE_ACCESSIBLE_VOLUME_DEFAULTS_H
