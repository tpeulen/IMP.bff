"""Every ``IMP.bff.<name>`` a notebook, example or doc page names must exist.

The Python surface was renamed wholesale (``ll_*`` to ``labelizer_*``, ``AV``
to ``ProbeAccessibleVolumeDecorator``, ``select_informative_pairs`` to
``select_probe_pairs``), and the consumers under ``ipynb/`` and ``doc/`` were
missed: four notebooks called fifteen functions that no longer existed, and
nothing noticed, because no notebook is executed anywhere in the test suite.

This is the cheap half of that gap -- it reads the files and checks the names
against the built module, in under a second, without running a single cell.
"""

from __future__ import annotations

import pathlib
import re

import pytest

bff = pytest.importorskip("IMP.bff")

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: ``IMP.bff.x`` or ``bff.x`` -- the two spellings the documents use.
REFERENCE = re.compile(r"\b(?:IMP\.bff|bff)\.([A-Za-z_][A-Za-z0-9_]*)")

#: Directories that are not documentation: a prototype is a scratch pad, and
#: the build tree is generated.
SKIP = ("build/", "prototypes/", "cmake-build", ".ipynb_checkpoints", "/_data/")

#: Names that are legitimately absent from a core-only build: they live in the
#: IMP connection layer, and the same documents are read by both builds.
IMP_LAYER_ONLY = frozenset()

#: `bff.<format>` is also how a network document names its format
#: (`"format": "bff.neural_net"`): a string in the msgpack map, not an
#: attribute of the module.
DOCUMENT_FORMATS = frozenset({"neural_net", "hmm_surrogate"})


def documents():
    for pattern in ("ipynb/**/*.ipynb", "doc/**/*.ipynb", "doc/**/*.md",
                    "examples/**/*.py"):
        for path in ROOT.glob(pattern):
            if not any(part in str(path) for part in SKIP):
                yield path


@pytest.mark.parametrize("path", sorted(documents(), key=str), ids=lambda p: str(p.relative_to(ROOT)))
def test_the_names_a_document_uses_exist(path):
    """A document that names a function the module does not have is stale."""
    text = path.read_text(errors="ignore")
    missing = sorted({
        name for name in REFERENCE.findall(text)
        if not name.startswith("_")
        and not hasattr(bff, name)
        and name not in IMP_LAYER_ONLY
        and name not in DOCUMENT_FORMATS
    })
    assert not missing, (
        f"{path.relative_to(ROOT)} names {', '.join(missing)}, which IMP.bff does "
        f"not have. The Python surface was renamed; update the document."
    )
