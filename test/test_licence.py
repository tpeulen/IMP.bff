"""Every place that states IMP.bff's licence states the same one: LGPL-2.1-or-later.

IMP.bff is an IMP module, and IMP licenses its LGPL modules under the GNU Lesser
GPL, version 2.1 or later; ``COPYING.LGPL`` is that text. The pip package once
declared ``MPL-2.0`` while shipping ``COPYING.LGPL``, and the Sphinx docs said
LGPL-3 -- three licences for one library, which is what a downstream package
(ChiSurf's installer, the labelizer site) then has to guess between. This test
ties every declaration to the one text the repository ships.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPDX = "LGPL-2.1-or-later"


def test_the_shipped_text_is_lgpl_2_1():
    head = (ROOT / "COPYING.LGPL").read_text(encoding="utf-8")[:400]
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in head
    assert "Version 2.1" in head


def test_pyproject_declares_lgpl():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^license\s*=\s*(.+)$', text, re.M)
    assert match, "pyproject.toml declares no licence"
    assert SPDX in match.group(1)


def test_recipes_declare_lgpl():
    for name in ("recipe.yaml", "recipe-core.yaml"):
        text = (ROOT / "conda-recipe" / name).read_text(encoding="utf-8")
        assert re.search(r"^\s*license:\s*" + re.escape(SPDX) + r"\s*$", text, re.M), name
        assert "COPYING.LGPL" in text, name


def test_no_document_names_another_licence_for_the_module():
    for rel in ("README.md", "doc/about.rst", "pyproject.toml",
                "conda-recipe/recipe.yaml", "conda-recipe/recipe-core.yaml"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "MPL-2.0" not in text, rel
        assert "LGPL-3" not in text, rel
        assert "lgpl-3.0" not in text, rel
