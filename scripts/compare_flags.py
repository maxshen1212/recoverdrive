"""Check that two runs saw the same deterministic check flags on every frame.

Every frame of --a must exist in --b, with identical flags; frames are joined on (scene, seed, run_fault, frame_idx).
A new baseline run on a split that a frozen run already covered must pass this before its numbers are compared.

Usage: uv run python scripts/compare_flags.py --a test_final --b test_checks_only
"""

import argparse
import json
import sys
from pathlib import Path


def load(tag: str) -> dict[tuple, list[str]]:
    out = {}
    for f in sorted((Path("outputs/eval") / tag / "runs").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                out[(r["scene"], r["seed"], r["run_fault"], r["frame_idx"])] = r["flags"]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="reference run tag under outputs/eval/")
    p.add_argument("--b", required=True, help="run tag that must contain every frame of --a with the same flags")
    args = p.parse_args()
    a, b = load(args.a), load(args.b)
    missing = [k for k in a if k not in b]
    differ = [k for k in a if k in b and a[k] != b[k]]
    same = len(a) - len(missing) - len(differ)
    print(f"flags identical on {same}/{len(a)} frames ({len(missing)} missing in {args.b}, {len(differ)} different)")
    for k in (missing + differ)[:10]:
        print("  e.g.", k, a[k], "vs", b.get(k))
    return 0 if same == len(a) and len(a) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
