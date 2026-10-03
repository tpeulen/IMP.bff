"""Paper-guardrailed regression checks for the public NPS example notebooks.

The physical pins follow Muschielok & Michaelis, J Phys Chem B 115 (2011),
doi:10.1021/jp2060377, and the actual Fast-NPS conventions of Eilert et al.,
Comput Phys Commun 219 (2017), doi:10.1016/j.cpc.2017.05.027. No invented
measurements or fitted outputs are used as external experimental results.
"""

import math
from pathlib import Path

import nbformat
import pytest

import IMP.bff as bff


NB_DIR = Path(__file__).resolve().parents[2] / "ipynb" / "example"
NAMES = ("nps.ipynb", "nps_phase3.ipynb")
PAPER_NAME = "nps_paper_structure.ipynb"


def _notebook(name):
    return nbformat.read(NB_DIR / name, as_version=4)


def _sources(notebook, kind):
    return "\n".join("".join(cell.source) for cell in notebook.cells
                     if cell.cell_type == kind)


@pytest.mark.parametrize("name", NAMES)
def test_notebook_schema_is_valid(name):
    nbformat.validate(_notebook(name))


@pytest.mark.parametrize("name", NAMES)
def test_primary_sources_and_scope_are_explicit(name):
    markdown = _sources(_notebook(name), "markdown")
    assert "10.1021/jp2060377" in markdown
    assert "10.1016/j.cpc.2017.05.027" in markdown
    assert "Muschielok" in markdown and "Eilert" in markdown
    assert "10.1111/j.2517-6161.1992.tb01796.x" not in markdown


def test_direct_example_pins_isotropic_limit_and_transfer_anisotropy():
    cells = _sources(_notebook("nps.ipynb"), "code")
    text = _sources(_notebook("nps.ipynb"), "markdown")
    assert "m=-\\cos\\theta" in text.replace(" ", "")
    assert "3/2" in text or "\\frac{3}{2}" in text
    assert "math.isclose" in cells and "eff_iso" in cells
    assert "ta_rigid" in cells

    rigid = bff.NPSDirectDye()
    rigid.x, rigid.y, rigid.z = 0.0, 0.0, 0.0
    rigid.m, rigid.phi = 0.0, 0.0
    rigid.steady_state_anisotropy = 0.4
    rigid_partner = bff.NPSDirectDye()
    rigid_partner.x, rigid_partner.y, rigid_partner.z = 0.0, 0.0, 55.0
    rigid_partner.m, rigid_partner.phi = 0.0, 0.0
    rigid_partner.steady_state_anisotropy = 0.4
    assert bff.nps_direct_transfer_anisotropy(rigid, rigid_partner) == pytest.approx(0.4)
    rigid.steady_state_anisotropy = rigid_partner.steady_state_anisotropy = 0.0
    assert bff.nps_direct_fret_efficiency(rigid, rigid_partner, 55.0) == pytest.approx(0.5)


def test_phase3_example_has_proper_prior_and_radial_distance():
    cells = _sources(_notebook("nps_phase3.ipynb"), "code")
    text = _sources(_notebook("nps_phase3.ipynb"), "markdown")
    assert ".set_prior(" in cells  # explicit priors on all six graph ports
    assert "np.linalg.norm(chain[:, 3:6] - chain[:, :3], axis=1)" in cells
    assert "nps_cloud_log_prior" in text and "Gaussian" in text
    assert "n^{-1/5}" in text and "n^{-1/7}" in text
    assert "set_output_is_log_likelihood(True)" in cells
    assert "abs(np.median(radial) - r_opt) < 12.0" in cells
    assert "normalized Gaussian" in text


def test_paper_structure_notebook_has_provenance_data_and_approximation_guardrails():
    """The structural example must identify its sources and its limits."""
    notebook = _notebook(PAPER_NAME)
    nbformat.validate(notebook)
    markdown = _sources(notebook, "markdown")
    code = _sources(notebook, "code")
    combined = markdown + "\n" + code

    assert "10.1038/nmeth.1259" in combined
    assert "1Y1W" in combined
    assert "Rpb7-Cys150: 60.4 ± 0.6%, R0=65 Å" in markdown
    assert "Rpb7-Cys94: 52.2 ± 1.1%, R0=66 Å" in markdown
    assert "Rpb7-S16C: 73.6 ± 0.7%, R0=62 Å" in markdown
    assert "approximation" in combined.lower()
    assert "attachment-site surrogates" in combined
    assert "5'" in combined and "ABSENT" in combined
    assert "NPSNetworkDye" in code
    assert "NPSMeasurement" in code
    assert "nps_network_fret_efficiency" in code
    assert "nps_network_log_likelihood" in code


def test_paper_structure_plot_uses_backbones_and_distinct_candidate_labels():
    """An atom-order polyline is not a nucleic-acid backbone or a legend."""
    code = _sources(_notebook(PAPER_NAME), "code")
    assert "backbone" in code
    assert "O5'" in code and "res['P']" in code
    assert "label='candidate 1" in code
    assert "label='candidate 2" in code
    assert "atom.coord for atom in model[chain_id].get_atoms()" not in code


def test_paper_structure_trilateration_does_not_seed_known_answers():
    """Derive sphere intersections from measured radii, not hardcoded roots."""
    code = _sources(_notebook(PAPER_NAME), "code")
    assert "np.cross" in code
    assert "target_distances" in code
    assert "seeds =" not in code


def test_paper_structure_pdb_provenance_and_atoms():
    """The notebook's offline structure input must be the pinned PDB stream."""
    import gzip
    import hashlib
    import io
    from Bio.PDB import PDBParser

    path = NB_DIR / "data" / "1Y1W.pdb.gz"
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == (
        "4225117bfc9f1dbb2c2edd278c7087cbf4acffe6af8654a74466726dbba9ca6f"
    )
    pdb_text = raw.decode("ascii")
    # PDB author-chain IDs: D is the 32-kDa B32/Rpb4 subunit and G is
    # the 19-kDa B16/Rpb7 subunit (distinct from mmCIF label_asym_id).
    assert "MOLECULE: DNA-DIRECTED RNA POLYMERASE II 32 KDA POLYPEPTIDE;" in pdb_text
    assert "CHAIN: D;" in pdb_text and "SYNONYM: B32;" in pdb_text
    assert "MOLECULE: DNA-DIRECTED RNA POLYMERASE II 19 KDA POLYPEPTIDE;" in pdb_text
    assert "CHAIN: G;" in pdb_text and "SYNONYM: B16;" in pdb_text
    structure = PDBParser(QUIET=True).get_structure("1Y1W", io.StringIO(pdb_text))
    model = structure[0]
    assert "G" in model and "D" in model
    assert model["G"][150]["SG"].get_name() == "SG"
    assert model["G"][94]["SG"].get_name() == "SG"
    assert model["G"][16]["OG"].get_name() == "OG"
    assert model["D"][73]["OG"].get_name() == "OG"
    assert len(list(model["P"].get_residues())) == 10


def test_paper_structure_notebook_executes_in_clean_kernel():
    """The offline paper example runs from a fresh kernel and emits its checks."""
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    notebook = _notebook(PAPER_NAME)
    try:
        result = NotebookClient(
            notebook, timeout=360, kernel_name="python3",
            resources={"metadata": {"path": str(NB_DIR)}}).execute()
    except CellExecutionError as exc:
        pytest.fail(f"{PAPER_NAME} did not execute: {exc}")
    outputs = "\n".join(
        out.get("text", "") for cell in result.cells if cell.cell_type == "code"
        for out in cell.get("outputs", []) if out.output_type == "stream"
    )
    assert "candidate 1 measured / BFF predicted E" in outputs
    assert "candidate 2 measured / BFF predicted E" in outputs
    assert "clashes" in outputs


@pytest.mark.parametrize("name", NAMES)
def test_notebook_executes_in_clean_kernel(name):
    """An independent kernel must run the public example, not saved outputs."""
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    notebook = _notebook(name)
    try:
        result = NotebookClient(notebook, timeout=360, kernel_name="python3",
                                resources={"metadata": {"path": str(NB_DIR)}}).execute()
    except CellExecutionError as exc:
        pytest.fail(f"{name} did not execute: {exc}")
    assert not any(out.output_type == "error" for cell in result.cells
                   if cell.cell_type == "code" for out in cell.get("outputs", []))
    if name == "nps_phase3.ipynb":
        text = "".join("".join(o.get("text", "") for o in cell.get("outputs", [])
                                if o.output_type == "stream") for cell in result.cells
                       if cell.cell_type == "code")
        assert "proper prior" in text
        assert "radial separation" in text
        assert "minimum recorded -2 log L" in text
