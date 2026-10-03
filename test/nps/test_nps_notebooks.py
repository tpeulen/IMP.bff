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
