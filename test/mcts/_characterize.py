"""Observable behaviour of a model-search problem, as a comparable record.

The record exists so that *how* a model family is declared can change while
what it does stays fixed.  It therefore holds nothing a factory declared --
no initial-value lists, no complexity constants, no effective sample sizes --
only what a caller can see through the problem's own API: which parameters
exist, which topologies exist, how the search may move between them, and what
each topology fits to.  A declaration that reproduces this record is
behaviourally the same declaration, whatever language it is written in.

Traversal is canonical rather than incidental.  Every structure is reached by
the shortest action path from the initial structure, and the problem is
restored to its root snapshot before each path is walked, so the record does
not depend on the order structures happen to be visited in.
"""

from __future__ import annotations

import collections

import IMP.bff as bff


def _actions(problem, structure_key):
    """The actions declared out of one structure, by structure key alone."""
    state = bff.ModelSearchState(structure_key, structure_key, 0.0, False)
    return list(problem.get_actions(state))


def action_graph(problem) -> dict:
    """Every declared transition, keyed by the structure it leaves."""
    return {
        key: [
            {
                "action": action.get_key(),
                "to": action.get_predicted_state_key(),
                "prior": action.get_prior(),
                "terminal": bool(action.get_terminal()),
            }
            for action in _actions(problem, key)
        ]
        for key in sorted(problem.get_structure_keys())
    }


def _shortest_paths(graph: dict, start: str) -> dict[str, list[str]]:
    """Shortest action path from ``start`` to every reachable structure."""
    paths = {start: []}
    queue = collections.deque([start])
    while queue:
        current = queue.popleft()
        for edge in graph.get(current, []):
            target = edge["to"]
            if target in paths or target == current:
                continue
            paths[target] = paths[current] + [edge["action"]]
            queue.append(target)
    return paths


def _walk(problem, root, path: list[str]):
    """Restore the root, then follow one action path, returning the state."""
    problem.restore_state(root.get_key())
    state = root
    for action_key in path:
        action = next(
            candidate
            for candidate in problem.get_actions(state)
            if candidate.get_key() == action_key
        )
        state = problem.evaluate(state, action)
    return state


def _fitted(problem, state) -> dict:
    """What one evaluated state holds: its score and canonical registry."""
    key = state.get_key()
    return {
        "structure": state.get_structure_key(),
        "reward": state.get_reward(),
        "acceptable": bool(state.get_acceptable()),
        "values": [float(v) for v in problem.get_cached_values(key)],
        "fixed": [int(f) for f in problem.get_cached_fixed(key)],
    }


def characterize(problem) -> dict:
    """One comparable record of everything the problem does observably."""
    root = problem.get_initial_state()
    graph = action_graph(problem)
    record = {
        "parameter_ids": list(problem.get_parameter_ids()),
        "structure_keys": sorted(problem.get_structure_keys()),
        "initial_structure": root.get_structure_key(),
        "actions": graph,
        "root": _fitted(problem, root),
        "structures": {},
    }
    for structure, path in sorted(_shortest_paths(graph, root.get_structure_key()).items()):
        if not path:
            continue
        state = _walk(problem, root, path)
        record["structures"][structure] = {"path": path, **_fitted(problem, state)}
    return record


def compare(record: dict, golden: dict, *, tolerance: float = 1.0e-3,
            undetermined: set[str] | None = None) -> list[str]:
    """Differences between two records; an empty list means they agree.

    Floats are compared with a relative tolerance set by what a converged
    nonlinear fit actually guarantees, which is not what a float guarantees.
    Levenberg-Marquardt stops on a tolerance, and near a minimum the score is
    quadratic in the parameters, so a different BLAS reaching a slightly
    different stopping point moves the reward by parts in ten thousand --
    measured at 1.0e-4 between macOS and Linux on an eight-topology FCS
    family. Asserting 1e-6 there was asserting more than the numerics
    promise. What *is* exact is compared exactly: ids, topology keys,
    transitions, priors and masks, and the ranking (see
    ``test_the_ranking_is_the_contract``) -- and a real regression in a fit
    moves the score by hundreds, not by parts in ten thousand.

    ``undetermined`` names structures whose fitted *score and values* are not
    a contract. An over-parameterised topology has no single answer: fitting
    three lifetimes to a two-lifetime decay splits one component in two, and
    where the split falls is decided by the starting point and the platform's
    linear algebra, not by the data. Its **score** is determined and is
    compared like any other -- that is what the search selects on -- but
    pinning its parameters would be pinning noise, and a record that does
    that fails on a different machine for no reason anyone can act on.

    Its *score* turned out not to be a contract either. Several topologies of
    a family sit within a few hundredths of each other, and declared
    multistart then chooses between near-equal optima -- measured at 2.5e-3
    relative between macOS and Linux on `fcs.3d.2diff.0relax`, which is a
    different optimum rather than a different stopping point. What survives
    that is the *order*, which is ordinal and is asserted exactly elsewhere,
    plus the winner's own score. A topology nobody picked contributes its
    position, not its number.
    """
    problems: list[str] = []
    undetermined = undetermined or set()

    def near(a, b) -> bool:
        return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))

    def walk(left, right, where: str) -> None:
        if (where.endswith(".values") or where.endswith(".reward")) and any(
            f"structures.{key}." in where + "." for key in undetermined
        ):
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left:
                    problems.append(f"{where}.{key}: missing, golden has {right[key]!r}")
                elif key not in right:
                    problems.append(f"{where}.{key}: unexpected, {left[key]!r}")
                else:
                    walk(left[key], right[key], f"{where}.{key}")
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                problems.append(f"{where}: length {len(left)} != golden {len(right)}")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, f"{where}[{index}]")
        elif isinstance(left, bool) or isinstance(right, bool):
            if left != right:
                problems.append(f"{where}: {left!r} != golden {right!r}")
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not near(float(left), float(right)):
                problems.append(f"{where}: {left!r} != golden {right!r}")
        elif left != right:
            problems.append(f"{where}: {left!r} != golden {right!r}")

    walk(record, golden, "record")
    return problems
