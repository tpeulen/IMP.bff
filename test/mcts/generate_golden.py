"""Rewrite the golden records from the problems the fixtures build.

Run it deliberately, never from a test:

    $E/bin/python test/mcts/generate_golden.py

Regenerating is how a *reviewed* behaviour change is recorded. If a record
moves and nobody meant it to, that is the finding, not the fix.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _characterize  # noqa: E402
import _fixtures  # noqa: E402

GOLDEN = pathlib.Path(__file__).resolve().parent / "golden"


def main() -> int:
    GOLDEN.mkdir(exist_ok=True)
    for name, build in sorted(_fixtures.FIXTURES.items()):
        record = _characterize.characterize(build())
        path = GOLDEN / f"{name}.json"
        path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"{path.name}: {len(record['structure_keys'])} structures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
