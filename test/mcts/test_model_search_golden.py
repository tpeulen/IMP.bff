"""Every model family keeps doing what it does, however it is declared.

These tests compare a problem against a committed record of its observable
behaviour: its canonical parameters, its topologies, the transitions between
them, and what each topology fits to.  Nothing here asserts *how* the family
was declared, which is the point -- the record is what lets the declaration
move from C++ into data without anyone having to take the port on trust.

Regenerate with ``test/mcts/generate_golden.py`` when a behaviour change is
intended, and read the diff before committing it.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _characterize  # noqa: E402
import _fixtures  # noqa: E402

GOLDEN = pathlib.Path(__file__).resolve().parent / "golden"


def _golden(name: str) -> dict:
    path = GOLDEN / f"{name}.json"
    if not path.exists():
        pytest.fail(f"no golden record for {name}; run test/mcts/generate_golden.py")
    return json.loads(path.read_text())


def _undetermined(golden: dict) -> set[str]:
    """Topologies whose fitted values are not a contract.

    Only the best-scoring topology is one a user ever sees the parameters of.
    The others are on the ladder the search climbed, and the ones above the
    answer are over-parameterised -- three lifetimes on a two-lifetime decay
    splits one component in two, and the split is decided by the starting
    point rather than the data. Their scores are compared; their parameters
    are recorded for diagnosis and not asserted.
    """
    scores = {golden["root"]["structure"]: golden["root"]["reward"]}
    scores.update({k: v["reward"] for k, v in golden["structures"].items()})
    best = max(scores, key=scores.get)
    return set(golden["structures"]) - {best}


@pytest.mark.parametrize("name", sorted(_fixtures.FIXTURES))
def test_family_matches_its_golden_record(name):
    golden = _golden(name)
    record = _characterize.characterize(_fixtures.FIXTURES[name]())
    differences = _characterize.compare(
        record, golden, undetermined=_undetermined(golden)
    )
    assert not differences, "\n".join(differences)


@pytest.mark.parametrize("name", sorted(_fixtures.FIXTURES))
def test_the_winning_topology_has_reproducible_parameters(name):
    """Whatever else moves, the answer a user is handed must not."""
    golden = _golden(name)
    record = _characterize.characterize(_fixtures.FIXTURES[name]())
    undetermined = _undetermined(golden)
    determined = set(golden["structures"]) - undetermined
    for key in determined:
        assert record["structures"][key]["values"] == pytest.approx(
            golden["structures"][key]["values"], rel=1e-6
        ), key


@pytest.mark.parametrize("name", sorted(_fixtures.FIXTURES))
def test_record_does_not_depend_on_traversal_order(name):
    """Two problems built the same way characterize the same way.

    The record walks every structure from a restored root, so a topology's
    fitted state must not depend on which sibling was visited before it.  If
    this fails the record is describing the traversal, not the family.
    """
    build = _fixtures.FIXTURES[name]
    assert not _characterize.compare(
        _characterize.characterize(build()), _characterize.characterize(build())
    )


def test_tcspc_selects_the_generating_component_count():
    """The record is not merely self-consistent; it is scientifically right.

    The decay is built from two lifetimes, so among the topologies the family
    offers, two components must score best -- one underfits and three pays
    complexity for nothing.  A port that reproduced the record while losing
    this would be reproducing arithmetic, not a model family.
    """
    golden = _golden("tcspc_lifetime")
    rewards = {golden["root"]["structure"]: golden["root"]["reward"]}
    rewards.update(
        {key: value["reward"] for key, value in golden["structures"].items()}
    )
    assert max(rewards, key=rewards.get) == "lifetime.components.2"
