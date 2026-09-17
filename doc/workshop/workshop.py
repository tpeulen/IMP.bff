"""The pieces the workshop notebooks share: structures, volumes, pictures.

Nothing here is science -- the science is in ``IMP.bff``. This module fetches
the two BmrA entries the workshop uses, cuts them down to one dimer, wraps the
accessible-volume call with the dye geometries an ``fps.json`` would carry, and
draws: a Mol* canvas for structures and volumes, and the circle plot of a FRET
network.

The notebooks import it as::

    import sys; sys.path.insert(0, "..")      # from doc/workshop/<nb>.ipynb
    import workshop as ws

Every function that needs the network downloads once into ``_data/`` beside
this file and reuses it afterwards, so a second run and an offline room both
work.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request
from typing import Iterable, Optional, Sequence

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "_data"

RCSB = "https://files.rcsb.org/download/{}.pdb"

#: The two states of BmrA this workshop works with. "Closed" and "open" name
#: the nucleotide-binding domains: 6R72 is the compact one. 6R72 holds *two*
#: dimers (A+B and C+D); C and D are a crystallographic copy, and pairing
#: across them would invent distances, so only A+B is ever used.
STATES = {
    "closed": {"pdb_id": "6R72", "chains": ("A", "B"),
               "note": "outward-facing, E504A, ATP/Mg (X-ray)"},
    "open": {"pdb_id": "8QOE", "chains": ("A", "B"),
             "note": "inward-facing (cryo-EM)"},
}

#: Dye geometries, in the fields an ``fps.json`` position carries: the linker
#: length and width in Angstrom, the one radius an AV1 uses, and the three
#: ellipsoid radii of an AV3. These are the values the FPS/Olga templates in
#: ``examples/structure`` use for these dyes; a real experiment carries its own
#: in its ``fps.json``.
#:
#: The AV1 radius is not the first AV3 radius. Alexa647's AV3 is 11.0/3.0/1.5 --
#: an 11 A half-length -- and feeding that 11 to an AV1 asks for a sphere that
#: does not fit next to a tight site: the volume comes back empty, and every
#: distance computed from it is meaningless.
DYES = {
    "Alexa488": {"linker_length": 20.0, "linker_width": 4.5,
                 "av1_radius": 3.5, "av3_radii": (5.0, 4.5, 1.5)},
    "Alexa647": {"linker_length": 22.0, "linker_width": 4.5,
                 "av1_radius": 3.5, "av3_radii": (11.0, 3.0, 1.5)},
}

#: Forster radius of the Alexa488/Alexa647 pair, in Angstrom.
R0 = 52.0

#: The rotamer library each workshop dye is represented by on the explicit
#: route. The names are the ones ``IMP.bff.probe_rotamer_library_registry()``
#: lists; the cutoff is appended by :func:`rotamer_library_name`, because a
#: library name carries its clustering cutoff ("cutoff10" is the deep sample,
#: "cutoff30" the coarse default).
ROTAMER_LIBRARIES = {
    "Alexa488": "AlexaFluor 488 C1R",
    "Alexa647": "AlexaFluor 647 C2R",
}

BACKBONE = {"N", "CA", "C", "O", "CB"}
LIGANDS = frozenset({"ATP", "MG"})

DONOR_COLOUR = 0x2E8B57      # sea green
ACCEPTOR_COLOUR = 0xC0392B   # brick red

MOLSTAR_VERSION = "4.9.0"
MOLSTAR_JS = f"https://cdn.jsdelivr.net/npm/molstar@{MOLSTAR_VERSION}/build/viewer/molstar.js"
MOLSTAR_CSS = f"https://cdn.jsdelivr.net/npm/molstar@{MOLSTAR_VERSION}/build/viewer/molstar.css"


# --- structures ---------------------------------------------------------------

def fetch_pdb(pdb_id: str, cache: Optional[pathlib.Path] = None) -> pathlib.Path:
    """The RCSB entry in legacy PDB format, downloaded once."""
    cache = pathlib.Path(cache or DATA)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{pdb_id.upper()}.pdb"
    if not path.exists() or path.stat().st_size == 0:
        urllib.request.urlretrieve(RCSB.format(pdb_id.upper()), path)
    return path


def single_dimer(state: str, cache: Optional[pathlib.Path] = None) -> pathlib.Path:
    """One BmrA dimer as its own file: the two chains of :data:`STATES`.

    Everything downstream assumes a file with exactly chains A and B, so the
    cutting happens here and only here.
    """
    spec = STATES[state]
    cache = pathlib.Path(cache or DATA)
    out = cache / f"BmrA_{state}.pdb"
    if out.exists() and out.stat().st_size > 0:
        return out
    source = fetch_pdb(spec["pdb_id"], cache)
    wanted = set(spec["chains"])
    kept = []
    for line in source.read_text().splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM", "ANISOU", "TER")):
            if line[21:22] in wanted and line[17:20].strip() not in {"HOH", "WAT"}:
                kept.append(line)
    kept.append("END\n")
    out.write_text("".join(kept))
    return out


def ca_position(pdb_path, chain: str, resi: int) -> Optional[np.ndarray]:
    """The C-alpha of one residue, or ``None`` when it is not resolved."""
    for line in pathlib.Path(pdb_path).read_text().splitlines():
        if (line.startswith("ATOM") and line[12:16].strip() == "CA"
                and line[21:22] == chain and line[22:26].strip() == str(resi)):
            return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return None


# --- accessible volumes -------------------------------------------------------

class EmptyVolume(ValueError):
    """A site where the dye has nowhere to go: no volume, and so no distance."""


def av(pdb_path, chain: str, resi: int, dye: str = "Alexa488",
       atom_name: str = "CB", grid_resolution: float = 1.5,
       allowed_sphere_radius: float = 5.0, av3: bool = False):
    """The accessible volume of one dye at one site, with that dye's geometry.

    Raises :class:`EmptyVolume` when the search finds nothing, rather than
    handing back a volume whose distances are all zero.
    """
    import IMP.bff as bff

    geometry = DYES[dye]
    radii = geometry["av3_radii"] if av3 else (geometry["av1_radius"], 0.0, 0.0)
    volume = bff.get_av_from_pdb(
        str(pdb_path), chain, int(resi), atom_name,
        geometry["linker_length"], geometry["linker_width"], radii[0],
        radii[1], radii[2], grid_resolution, allowed_sphere_radius,
    )
    if volume.get_n_points() == 0:
        raise EmptyVolume(
            f"{chain}{resi} in {pathlib.Path(pdb_path).name} has no accessible "
            f"volume with {dye}: the site is buried, or the dye radius does not "
            f"fit. It is not a distance of zero -- there is no distance."
        )
    return volume


def rda(av_donor, av_acceptor) -> float:
    """<R_DA>, the mean of the distance distribution between two volumes (A)."""
    import IMP.bff as bff

    return float(bff.average_distance(av_donor.get_points(), av_acceptor.get_points()))


def rda_e(av_donor, av_acceptor, forster_radius: float = R0) -> float:
    """<R_DA>_E: the distance that reproduces the mean transfer efficiency.

    This is what a FRET experiment reports, and it is not <R_DA> -- for a broad
    pair of volumes the two differ by several Angstrom.
    """
    import IMP.bff as bff

    return float(bff.mean_fret_distance(
        av_donor.get_points(), av_acceptor.get_points(), forster_radius))


def efficiency(distance: float, forster_radius: float = R0) -> float:
    """The transfer efficiency of one distance."""
    return 1.0 / (1.0 + (float(distance) / float(forster_radius)) ** 6)


def distance_distribution(av_donor, av_acceptor, axis=None):
    """``(axis, p)`` -- the distance distribution of a pair, as a histogram."""
    import IMP.bff as bff

    axis = np.linspace(10.0, 120.0, 111) if axis is None else np.asarray(axis, float)
    p = np.asarray(bff.cloud_distance_distribution(
        av_donor.get_points(), av_acceptor.get_points(), axis), dtype=float)
    return axis[:len(p)], p


# --- Mol* ---------------------------------------------------------------------

def protein_for_view(pdb_path) -> str:
    """Backbone and ligands only: enough for a cartoon, a fraction of the bytes.

    The page is inlined into the notebook as base64, so a four-chain entry at
    full atom detail would bloat the file for a picture that does not use it.
    """
    kept = []
    for line in pathlib.Path(pdb_path).read_text().splitlines(keepends=True):
        if line.startswith("ATOM"):
            if line[12:16].strip() in BACKBONE:
                kept.append(line)
        elif line.startswith("HETATM") and line[17:20].strip() in LIGANDS:
            kept.append(line)
    kept.append("END\n")
    return "".join(kept)


def points_to_pdb(points, chain: str = "X", resname: str = "AV",
                  max_points: int = 2500) -> str:
    """A point cloud as pseudo-atoms, for Mol* to build a surface over.

    Down-sampled on a fixed stride: the surface of a dense cloud does not
    change perceptibly, and the browser stays responsive.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        return "END\n"
    if len(points) > max_points:
        points = points[:: int(np.ceil(len(points) / max_points))]
    lines = []
    for index, row in enumerate(points, start=1):
        x, y, z = row[:3]
        lines.append(
            f"HETATM{index % 100000:5d}  C   {resname} {chain}"
            f"{1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    lines.append("END\n")
    return "".join(lines)


def rotamer_library_name(dye: str, cutoff: int = 10) -> str:
    """The registry name of a workshop dye's rotamer library at one cutoff."""
    return f"{ROTAMER_LIBRARIES[dye]} cutoff{int(cutoff)}"


_TWO_LETTER_ELEMENTS = frozenset({"CL", "BR", "NA", "MG", "ZN", "FE", "SE"})


def _element_of(atom_name: str) -> str:
    letters = "".join(c for c in atom_name if c.isalpha()).upper()
    if len(letters) >= 2 and letters[:2] in _TWO_LETTER_ELEMENTS:
        return letters[:2]
    return letters[:1] or "C"


def rotamer_pdb(ensemble, n_rotamers: int = 12, drop_hydrogens: bool = True,
                chain: str = "Y", resname: Optional[str] = None) -> str:
    """The heaviest-weighted conformers of a rotamer ensemble, as PDB text.

    The counterpart of :func:`av_pdb` for the explicit route: where an AV has
    only points, a :class:`IMP.bff.ProbeRotamerEnsemble` carries the dye's
    atoms, and Mol* can draw them as a molecule. Each conformer becomes its own
    residue, so nothing is bonded across conformers.

    ``n_rotamers`` conformers are kept, in order of weight -- the whole
    ensemble at once is a hairball, and the weights are usually concentrated on
    a handful anyway (see ``get_effective_sample_size``).
    """
    atoms = np.asarray(ensemble.get_atoms(), dtype=float)
    n_atoms = int(ensemble.get_n_atoms())
    if n_atoms == 0 or atoms.size == 0:
        return "END\n"
    atoms = atoms.reshape(-1, n_atoms, 3)
    weights = np.asarray(ensemble.get_weights(), dtype=float).ravel()
    names = list(ensemble.get_atom_names())
    resnames = list(ensemble.get_resnames())
    label = resname or (resnames[0][:3] if resnames else "DYE")

    order = np.argsort(weights)[::-1][: max(int(n_rotamers), 1)]
    lines = []
    serial = 0
    for residue_index, rotamer in enumerate(order, start=1):
        for atom_index in range(n_atoms):
            name = names[atom_index] if atom_index < len(names) else "C"
            element = _element_of(name)
            if drop_hydrogens and element == "H":
                continue
            x, y, z = atoms[rotamer, atom_index]
            serial += 1
            lines.append(
                f"HETATM{serial % 100000:5d} {name:<4s}{label:>3s} {chain}"
                f"{residue_index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}\n"
            )
    lines.append("END\n")
    return "".join(lines)


def av_points(volume) -> np.ndarray:
    """The cloud as ``(N, 4)`` -- x, y, z, weight. The C++ side hands it over flat."""
    return np.asarray(volume.get_points(), dtype=float).reshape(-1, 4)


def av_pdb(volume, **kwargs) -> str:
    """The pseudo-atom PDB of an accessible volume's cloud."""
    return points_to_pdb(av_points(volume)[:, :3], **kwargs)



def score_coloured_pdb(pdb_path, values, chains=None, default: float = 0.0) -> str:
    """The structure with a per-residue number in the B-factor column.

    ``values`` maps ``(chain, residue)`` to the number -- a label score, a
    deviation, a weight. Mol*'s "uncertainty" theme colours by that column, so
    a panel built with ``colour_by="score"`` paints the score onto the fold,
    which is the picture the old labelizer script made by hand.
    """
    kept = []
    for line in pathlib.Path(pdb_path).read_text().splitlines(keepends=True):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        chain = line[21:22]
        if chains is not None and chain not in chains:
            continue
        if line.startswith("ATOM") and line[12:16].strip() not in BACKBONE:
            continue
        try:
            resi = int(line[22:26])
        except ValueError:
            continue
        value = values.get((chain, resi), default)
        value = default if value is None or not np.isfinite(value) else float(value)
        kept.append(f"{line[:60]}{min(value, 999.99):6.2f}{line[66:]}")
    kept.append("END\n")
    return "".join(kept)


def morph_pdb(frames, chains=None) -> str:
    """Several structures as one multi-model PDB, for Mol* to animate through.

    ``frames`` are paths or PDB texts, in the order they should play. Mol*'s
    trajectory animation steps through the MODEL records, so a morph plays in
    the browser with nothing pre-rendered.

    **Every frame must hold the same atoms in the same order.** A viewer reads
    a multi-model file as a trajectory only then; two PDB entries of the same
    protein almost never qualify, because they resolve different atoms, and
    the file silently loads as a single model instead. :func:`morph_frames`
    builds frames that do qualify.
    """
    texts = [frame if "\n" in str(frame) else pathlib.Path(frame).read_text()
             for frame in frames]
    kept = []
    for text in texts:
        kept.append([line for line in text.splitlines(keepends=True)
                     if line.startswith(("ATOM", "HETATM"))
                     and (chains is None or line[21:22] in chains)])
    counts = {len(atoms) for atoms in kept}
    if len(counts) > 1:
        raise ValueError(
            f"the frames hold different numbers of atoms ({sorted(counts)}), so a "
            f"viewer will read this as one model and the animation will not play. "
            f"Build the frames on the atoms the structures share -- morph_frames() "
            f"does that."
        )
    out = []
    for index, atoms in enumerate(kept, start=1):
        out.append(f"MODEL     {index:4d}\n")
        out.extend(atoms)
        out.append("ENDMDL\n")
    out.append("END\n")
    return "".join(out)



#: hGBP1's conformational transition: a coarse-grained trajectory of the
#: open/closed change, and the topology its frames are laid onto. A DCD holds
#: coordinates and nothing else -- no atom names, no residues, no chains -- so
#: the PDB is not optional.
HGBP1 = {
    "topology": "examples/structure/GBP/hgbp1_cg.pdb",
    "trajectory": "examples/structure/GBP/hgbp1_transition.dcd",
    "network": "examples/structure/GBP/hGBP1.fps.json",
}


def repo_root() -> pathlib.Path:
    """The checkout this module lives in."""
    return HERE.parent.parent


def hgbp1_frames(every: int = 4, atoms: str = "CA"):
    """The hGBP1 transition as PDB texts, one per frame, all with the same atoms.

    ``every`` subsamples the 58 recorded frames; ``atoms`` selects what to keep
    ("CA" for the trace, "all" for every coarse-grained bead). A backbone trace
    is a tenth of the bytes and renders the change just as clearly, which
    matters when the frames are inlined into a notebook.
    """
    import IMP.bff as bff

    root = repo_root()
    topology = (root / HGBP1["topology"]).read_text().splitlines(keepends=True)
    template = [line for line in topology
                if line.startswith("ATOM")
                and (atoms == "all" or line[12:16].strip() == atoms)]
    indices = [i for i, line in enumerate(
        [l for l in topology if l.startswith("ATOM")])
        if atoms == "all" or l[12:16].strip() == atoms] if False else None

    keep = [i for i, line in enumerate([l for l in topology if l.startswith("ATOM")])
            if atoms == "all" or line[12:16].strip() == atoms]

    header = bff.read_dcd_header(str(root / HGBP1["trajectory"]))
    flat = np.asarray(bff.read_dcd(str(root / HGBP1["trajectory"])), dtype=float)
    xyz = flat.reshape(header.n_frames, header.n_atoms, 3)

    frames = []
    for f in range(0, header.n_frames, max(1, int(every))):
        lines = []
        for line, index in zip(template, keep):
            x, y, z = xyz[f, index]
            lines.append(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
        frames.append("".join(lines) + "END\n")
    return frames

def _atom_key(line: str):
    """chain, residue, insertion code, atom name -- what makes an atom the same atom."""
    return line[21:22], line[22:27].strip(), line[12:16].strip(), line[17:20].strip()


def morph_frames(pdb_a, pdb_b, n_frames: int = 9, chains=None):
    """A straight-line morph between two structures, on the atoms they share.

    Returns a list of PDB texts, all holding the same atoms in the same order,
    which is what :func:`morph_pdb` and every viewer's trajectory reader need.

    The line is a ruler, not a mechanism: it says where the two ends are and
    how far apart, and nothing about the path between them.
    """
    def atoms_of(path):
        table = {}
        for line in pathlib.Path(path).read_text().splitlines(keepends=True):
            if line.startswith("ATOM") and (chains is None or line[21:22] in chains):
                table[_atom_key(line)] = line
        return table

    first, second = atoms_of(pdb_a), atoms_of(pdb_b)
    shared = [key for key in first if key in second]
    if not shared:
        raise ValueError(f"{pdb_a} and {pdb_b} share no atoms by chain/residue/name")

    start = np.array([[float(first[k][30:38]), float(first[k][38:46]),
                       float(first[k][46:54])] for k in shared])
    end = np.array([[float(second[k][30:38]), float(second[k][38:46]),
                     float(second[k][46:54])] for k in shared])

    frames = []
    for t in np.linspace(0.0, 1.0, int(n_frames)):
        xyz = (1.0 - t) * start + t * end
        lines = []
        for key, (x, y, z) in zip(shared, xyz):
            line = first[key]
            lines.append(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
        frames.append("".join(lines) + "END\n")
    return frames

def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def viewer_html(panels: Sequence[dict], height: int = 520) -> str:
    """One Mol* canvas per panel, side by side.

    A panel is ``{"title": str, "structure": <PDB text>, "clouds":
    [{"pdb": <PDB text>, "colour": 0xRRGGBB, "label": str}]}``. Mol* comes from
    a CDN and renders the clouds as gaussian surfaces, so nothing is meshed in
    Python and no notebook extension is involved -- the viewer needs internet
    access from the *browser*, and says so in the panel if it cannot load.

    A cloud may carry ``"style"`` to be drawn as something other than a
    surface -- ``"ball-and-stick"`` for the atoms of an explicit dye, say,
    where the point is to see the molecule and not its envelope.
    """
    payload = [{
        "title": panel.get("title", ""),
        "structure": _b64(panel["structure"]),
        # Mol*'s "uncertainty" theme reads the B-factor column; a structure
        # written by `score_coloured_pdb` carries the label score there.
        "colour_by": panel.get("colour_by", ""),
        # Animation: "spin" turns the camera, "models" steps through the
        # MODEL records of a multi-model PDB -- a morph between two states.
        "spin": bool(panel.get("spin", False)),
        "animate_models": bool(panel.get("animate_models", False)),
        "n_models": panel["structure"].count("\nMODEL ") + panel["structure"].startswith("MODEL "),
        "frame_ms": int(panel.get("frame_ms", 500)),
        "clouds": [{"pdb": _b64(c["pdb"]), "colour": c.get("colour", DONOR_COLOUR),
                    "label": c.get("label", ""),
                    "style": c.get("style", "gaussian-surface")}
                   for c in panel.get("clouds", [])],
    } for panel in panels]
    data = json.dumps(payload)
    unique = "ws-molstar-" + str(abs(hash(data)) % 10 ** 8)

    return f"""
<link rel="stylesheet" type="text/css" href="{MOLSTAR_CSS}" />
<div id="{unique}" style="display:flex;gap:10px;flex-wrap:wrap"></div>
<script>
(function() {{
  var PANELS = {data};
  var HEIGHT = {height};
  var root = document.getElementById("{unique}");

  function panelFor(panel) {{
    var wrap = document.createElement("div");
    wrap.style.cssText = "flex:1 1 380px;min-width:340px";
    var title = document.createElement("div");
    title.textContent = panel.title;
    title.style.cssText = "font:600 12.5px system-ui;margin:0 0 4px 2px";
    var host = document.createElement("div");
    host.style.cssText = "position:relative;width:100%;height:" + HEIGHT +
                         "px;border:1px solid #d8d8d8;border-radius:6px";
    wrap.appendChild(title); wrap.appendChild(host); root.appendChild(wrap);
    return host;
  }}

  function render(panel, host) {{
    return molstar.Viewer.create(host, {{
      layoutIsExpanded: false, layoutShowControls: false,
      layoutShowSequence: false, layoutShowLog: false,
      layoutShowLeftPanel: false, viewportShowExpand: true,
      viewportShowSelectionMode: false, viewportShowAnimation: false
    }}).then(function(viewer) {{
      var plugin = viewer.plugin;
      (window.__workshop_viewers = window.__workshop_viewers || []).push(viewer);
      var B = plugin.builders.structure;

      function parse(b64) {{
        return plugin.builders.data.rawData({{ data: atob(b64) }})
          .then(function(raw) {{ return B.parseTrajectory(raw, "pdb"); }});
      }}

      if (panel.spin) {{
        plugin.canvas3d.setProps({{ trackball: {{ animate: {{
          name: "spin", params: {{ speed: 0.6 }} }} }} }});
      }}

      return parse(panel.structure)
        .then(function(traj) {{
          return B.hierarchy.applyPreset(traj, "default",
            {{ representationPreset: "polymer-and-ligand",
               theme: panel.colour_by === "score"
                 ? {{ globalName: "uncertainty",
                      globalColorParams: {{ list: {{ kind: "interpolate",
                        colors: ["#2166ac", "#f7f7f7", "#b2182b"] }} }} }}
                 : undefined }});
        }})
        .then(function() {{
          return panel.clouds.reduce(function(chain, cloud) {{
            return chain.then(function() {{
              return parse(cloud.pdb)
                .then(function(traj) {{
                  return B.hierarchy.applyPreset(traj, "default",
                    {{ representationPreset: "empty" }});
                }})
                .then(function(built) {{
                  var ref = built && (built.structure || built.structureRef);
                  if (!ref) return;
                  return B.tryCreateComponentStatic(ref, "all");
                }})
                .then(function(component) {{
                  if (!component) return;
                  var style = cloud.style || "gaussian-surface";
                  var params = style === "gaussian-surface"
                    ? {{ alpha: 0.55, radiusOffset: 0.4, smoothness: 0.5,
                        visuals: ["gaussian-surface-mesh"] }}
                    : {{}};
                  return B.representation.addRepresentation(component, {{
                    type: style,
                    typeParams: params,
                    color: "uniform",
                    colorParams: {{ value: cloud.colour }}
                  }});
                }});
            }});
          }}, Promise.resolve());
        }})
        .then(function() {{
          // Step through the MODEL records of the trajectory.
          //
          // Not Mol*'s own "animate-model-index": in an embedded Viewer it
          // reports itself as playing and never advances a frame (measured --
          // its parameters are accepted, and ticking the manager by hand does
          // nothing either). Setting the model index on the state tree is what
          // that animation does internally, and it works here.
          if (!panel.animate_models || panel.n_models < 2) return;
          var cell = Array.from(plugin.state.data.cells.values()).filter(function(c) {{
            return c.transform && c.transform.params &&
                   ("modelIndex" in c.transform.params);
          }})[0];
          if (!cell) return;
          var index = 0;
          setInterval(function() {{
            index = (index + 1) % panel.n_models;
            var params = Object.assign({{}}, cell.transform.params,
                                       {{ modelIndex: index }});
            plugin.state.data.build().to(cell.transform.ref)
              .update(params).commit();
          }}, panel.frame_ms);
        }})
        .catch(function(error) {{
          host.innerHTML = "<pre style='padding:10px;font:12px monospace;" +
            "color:#b00;white-space:pre-wrap'>Mol* could not render this " +
            "panel: " + String(error && error.message || error) + "</pre>";
        }});
    }});
  }}

  function build() {{
    PANELS.reduce(function(chain, panel) {{
      var host = panelFor(panel);
      return chain.then(function() {{ return render(panel, host); }});
    }}, Promise.resolve());
  }}

  if (window.molstar) {{ build(); }}
  else {{
    var script = document.createElement("script");
    script.src = "{MOLSTAR_JS}";
    script.onload = build;
    script.onerror = function() {{
      root.innerHTML = "<pre style='padding:10px;font:12px monospace;" +
        "white-space:pre-wrap'>Mol* could not be loaded from the CDN. Every " +
        "number in this notebook is unaffected; only this 3D view needs " +
        "internet access from your browser.</pre>";
    }};
    document.head.appendChild(script);
  }}
}})();
</script>
"""


def show(panels: Sequence[dict], height: int = 520):
    """Display :func:`viewer_html` in the notebook."""
    from IPython.display import HTML, display

    return display(HTML(viewer_html(panels, height=height)))


def site_panel(state: str, sites: Sequence[dict], title: Optional[str] = None,
               pdb_path=None) -> dict:
    """One panel: a BmrA state with the volumes of the given sites on it.

    ``sites`` are ``{"chain", "resi", "dye", "colour"}`` dictionaries; the
    volume is computed with :func:`av`, the same call the tables use, so the
    picture and the number cannot drift apart.
    """
    path = pdb_path or single_dimer(state)
    clouds = []
    for site in sites:
        volume = av(path, site["chain"], site["resi"], site.get("dye", "Alexa488"))
        clouds.append({
            "pdb": av_pdb(volume),
            "colour": site.get("colour", DONOR_COLOUR),
            "label": f'{site["chain"]}{site["resi"]}',
        })
    return {"title": title or f"BmrA {state}", "structure": protein_for_view(path),
            "clouds": clouds}


# --- the network as a picture -------------------------------------------------

def deviation_scale(vmax: float, cmap: str = "coolwarm"):
    """``(norm, cmap)`` for a deviation in Angstrom: symmetric, zero in the middle.

    :func:`circle_plot` colours its chords with exactly this, so a figure that
    wants one shared colour bar for several panels can build the bar from here
    and pass the same ``vmax`` to every panel. Two panels drawn on two scales
    invite a comparison that is not there.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    norm = colors.TwoSlopeNorm(vmin=-abs(vmax), vcenter=0.0, vmax=abs(vmax))
    return norm, plt.get_cmap(cmap)


def deviation_mappable(vmax: float, cmap: str = "coolwarm"):
    """A ``ScalarMappable`` on :func:`deviation_scale`, ready for ``figure.colorbar``."""
    from matplotlib import cm as _cm

    norm, colour_map = deviation_scale(vmax, cmap)
    mappable = _cm.ScalarMappable(norm=norm, cmap=colour_map)
    mappable.set_array([])
    return mappable


def _chord_path(p1, p2, bend: float, offset: float):
    """A cubic Bezier from ``p1`` to ``p2``, pulled ``bend`` of the way inwards.

    ``bend`` of 0 is the straight line; 1 sends the chord through the origin.
    ``offset`` nudges the two control points perpendicular to the chord, which
    is what keeps chords between near-opposite sites from lying on top of each
    other. It is a drawing device and carries no information.
    """
    from matplotlib.path import Path

    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    span = p2 - p1
    length = float(np.hypot(*span)) or 1.0
    perpendicular = np.array([-span[1], span[0]]) / length
    shift = perpendicular * offset * length
    c1 = (1.0 - bend) * p1 + shift
    c2 = (1.0 - bend) * p2 + shift
    return Path([tuple(p1), tuple(c1), tuple(c2), tuple(p2)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])


def _ha(x: float) -> str:
    """Horizontal alignment that pushes a rim label away from the circle."""
    return "left" if x > 0.15 else ("right" if x < -0.15 else "center")


def _va(y: float) -> str:
    """Vertical alignment that pushes a rim label away from the circle."""
    return "bottom" if y > 0.15 else ("top" if y < -0.15 else "center")


def circle_plot(pairs: Sequence[tuple], values: dict, ax=None, title: str = "",
                vmax: Optional[float] = None, label_fmt: str = "{}",
                widths: Optional[dict] = None, groups: Optional[dict] = None,
                order: Optional[Sequence] = None, colorbar: bool = True,
                cmap: str = "coolwarm", bend: float = 0.55,
                spread: float = 0.045, lw_range: tuple = (0.9, 4.6),
                value_label: str = "model - measured (A)"):
    """The FRET network as a circle: sites on the rim, one chord per pair.

    ``pairs`` are ``(site_a, site_b)`` tuples of hashable site names, and
    ``values`` maps a pair to the number that colours its chord -- a deviation
    in Angstrom, say. Red is a large deviation, blue a small one, and the
    colour scale is symmetric so "too short" and "too long" are told apart.

    The chords curve through the inside of the circle rather than crossing it
    in a straight line: a straight-line version of a network with more than a
    handful of pairs is a knot in the middle of the figure and nothing can be
    read out of it.

    The optional arguments:

    ``widths``
        A second number per pair, driving the line width -- how many standard
        deviations the pair is off, say, where the colour carries how many
        Angstrom. Left out, the width follows ``abs(value)`` and says the same
        thing as the colour twice.
    ``groups``
        ``site -> group name``. Sites are then laid out group by group with a
        gap between the groups and a labelled arc outside the rim, so "every
        bad chord touches one domain" becomes visible rather than inferred.
    ``order``
        The site order around the rim. The default is the order the sites first
        appear in ``pairs``.
    ``label_fmt``
        A format string applied to each site name, or a callable turning a site
        into its label -- which is what a network whose sites are tuples such
        as ``("A", 195)`` wants.
    ``colorbar``
        Draw a colour bar next to this axes. Turn it off and use
        :func:`deviation_mappable` for one shared bar when several panels are
        compared.
    ``vmax``
        The end of the colour scale, in the units of ``values``. Pass the same
        one to every panel that is meant to be compared.
    """
    import matplotlib.pyplot as plt

    seen = []
    for a, b in pairs:
        for site in (a, b):
            if site not in seen:
                seen.append(site)
    sites = [s for s in (order or seen) if s in seen]
    sites += [s for s in seen if s not in sites]
    n = len(sites)

    if groups:
        keys, ordered = [], []
        for site in sites:
            key = groups.get(site, "")
            if key not in keys:
                keys.append(key)
        for key in keys:
            ordered += [s for s in sites if groups.get(s, "") == key]
        sites = ordered
    else:
        keys = [""]

    gap = 0.16 if len(keys) > 1 else 0.0
    step = (2.0 * np.pi - gap * len(keys)) / max(n, 1)
    angle, at = {}, np.pi / 2.0 + gap / 2.0
    group_span = {}
    for key in keys:
        members = [s for s in sites if groups.get(s, "") == key] if groups else sites
        start = at
        for site in members:
            angle[site] = at + step / 2.0
            at += step
        group_span[key] = (start, at)
        at += gap
    xy = {site: (np.cos(a), np.sin(a)) for site, a in angle.items()}

    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 5.4))
    finite = [abs(v) for v in values.values() if np.isfinite(v)]
    vmax = vmax if vmax is not None else (max(finite) if finite else 1.0)
    vmax = abs(vmax) or 1.0
    norm, colour_map = deviation_scale(vmax, cmap)

    if widths is None:
        width_of = {p: abs(values.get(p, np.nan)) for p in pairs}
        width_max = vmax
    else:
        width_of = {p: abs(widths.get(p, widths.get(tuple(reversed(p)), np.nan)))
                    for p in pairs}
        clean = [w for w in width_of.values() if np.isfinite(w)]
        width_max = max(clean) if clean else 1.0
    lw_lo, lw_hi = lw_range

    # The chords, thin ones first, so a big deviation is never hidden under a
    # small one.
    drawn = sorted(range(len(pairs)),
                   key=lambda i: (width_of.get(pairs[i], 0.0)
                                  if np.isfinite(width_of.get(pairs[i], np.nan)) else -1.0))
    for rank, index in enumerate(drawn):
        pair = pairs[index]
        value = values.get(pair, values.get(tuple(reversed(pair)), np.nan))
        colour = colour_map(norm(value)) if np.isfinite(value) else "0.85"
        w = width_of.get(pair, np.nan)
        lw = lw_lo + (lw_hi - lw_lo) * (w / width_max if np.isfinite(w) and width_max else 0.0)
        # Alternating offsets: neighbouring chords fan out instead of merging.
        offset = spread * (1 if index % 2 else -1) * (0.6 + 0.4 * ((index // 2) % 3))
        path = _chord_path(xy[pair[0]], xy[pair[1]], bend, offset)
        ax.add_patch(plt.matplotlib.patches.PathPatch(
            path, edgecolor=colour, facecolor="none", lw=lw, zorder=1 + rank,
            capstyle="round"))

    group_colours = ["#3f6ea8", "#a8663f", "#4f8f5a", "#8a5ba8", "#8a8a3f"]
    for i, key in enumerate(keys):
        if not key:
            continue
        a0, a1 = group_span[key]
        arc = np.linspace(a0 + step * 0.15, a1 - step * 0.15, 60)
        ax.plot(1.22 * np.cos(arc), 1.22 * np.sin(arc), lw=3.2,
                color=group_colours[i % len(group_colours)], solid_capstyle="round",
                zorder=3)
        mid = 0.5 * (a0 + a1)
        ax.annotate(key, (1.44 * np.cos(mid), 1.44 * np.sin(mid)),
                    ha=_ha(np.cos(mid)), va=_va(np.sin(mid)), fontsize=9.5,
                    color=group_colours[i % len(group_colours)])

    name_of = label_fmt if callable(label_fmt) else label_fmt.format
    for site, (x, y) in xy.items():
        ax.plot([x], [y], "o", color="0.25", ms=6.5, zorder=4)
        ax.annotate(name_of(site), (x * 1.08, y * 1.08), ha=_ha(x),
                    va=_va(y), fontsize=8.5, zorder=4)

    limit = 1.75 if any(keys) else 1.35
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    if colorbar:
        bar = ax.figure.colorbar(deviation_mappable(vmax, cmap), ax=ax,
                                 fraction=0.045, pad=0.02)
        bar.set_label(value_label, fontsize=9)
    return ax


__all__ = [
    "STATES", "DYES", "R0", "ROTAMER_LIBRARIES", "fetch_pdb", "single_dimer",
    "ca_position",
    "av", "av_points", "EmptyVolume", "rda", "rda_e", "efficiency", "distance_distribution",
    "protein_for_view", "points_to_pdb", "score_coloured_pdb", "morph_pdb", "morph_frames", "hgbp1_frames", "HGBP1", "av_pdb", "viewer_html", "show",
    "site_panel", "circle_plot", "deviation_scale", "deviation_mappable",
    "rotamer_library_name", "rotamer_pdb",
]
