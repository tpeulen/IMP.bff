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


def _free_only(record: dict) -> dict:
    """Drop the values of parameters a topology holds fixed.

    A fixed slot carries whichever declared start won, not an answer -- an
    unused third lifetime sat at 12.5 on one platform and 3.0 on another with
    every fitted parameter identical. The mask itself is still compared, so a
    parameter changing from fixed to free is still caught.
    """
    trimmed = json.loads(json.dumps(record))
    for state in list(trimmed["structures"].values()) + [trimmed["root"]]:
        mask = state["fixed"]
        state["values"] = [
            value for value, fixed in zip(state["values"], mask) if fixed == 0
        ]
    return trimmed


@pytest.mark.parametrize("name", sorted(_fixtures.FIXTURES))
def test_family_matches_its_golden_record(name):
    golden = _golden(name)
    record = _characterize.characterize(_fixtures.FIXTURES[name]())
    differences = _characterize.compare(
        _free_only(record), _free_only(golden), undetermined=_undetermined(golden)
    )
    assert not differences, "\n".join(differences)


@pytest.mark.parametrize("name", sorted(_fixtures.FIXTURES))
def test_the_ranking_is_the_contract(name):
    """Which topology beats which, exactly -- that is what selection means.

    The scores themselves are compared to what a converged fit guarantees,
    but their *order* is ordinal and has to hold outright: a port, a rebuild
    or another platform may move a reward by parts in ten thousand and must
    never move the answer.
    """
    golden = _golden(name)
    record = _characterize.characterize(_fixtures.FIXTURES[name]())

    def ranking(source):
        scores = {source["root"]["structure"]: source["root"]["reward"]}
        scores.update({k: v["reward"] for k, v in source["structures"].items()})
        return sorted(scores, key=scores.get, reverse=True)

    assert ranking(record) == ranking(golden)


@pytest.mark.parametrize("name", sorted(_fixtures.FIXTURES))
def test_the_winning_topology_has_reproducible_parameters(name):
    """Whatever else moves, the answer a user is handed must not.

    Only the parameters the winning topology *frees*. A fixed one carries
    whichever declared start happened to win, which is a seed rather than an
    answer -- on one platform an unused third lifetime slot sat at 12.5 and
    on another at 3.0, with every fitted parameter identical.
    """
    golden = _golden(name)
    record = _characterize.characterize(_fixtures.FIXTURES[name]())
    for key in set(golden["structures"]) - _undetermined(golden):
        mask = golden["structures"][key]["fixed"]
        free = [i for i, fixed in enumerate(mask) if fixed == 0]
        # A relative tolerance on a parameter whose answer is zero is a
        # lottery, not a test: an unused amplitude lands at -1.2e-4 here and
        # -2.5e-4 on another platform's libm, which is the same answer and a
        # 53 % relative difference. The absolute floor is what makes this
        # comparison mean "the user is handed the same numbers".
        assert [record["structures"][key]["values"][i] for i in free] == pytest.approx(
            [golden["structures"][key]["values"][i] for i in free], rel=1e-4, abs=1e-3
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
