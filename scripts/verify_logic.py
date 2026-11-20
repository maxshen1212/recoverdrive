"""Exhaustive verification of the safety shield (and, from Task 5, the recovery controller). CPU only.

    uv run python scripts/verify_logic.py --out /data/maxshen/recoverdrive/report_figures/formal
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from recoverdrive.formal import verify_shield
from recoverdrive.runner import git_state

REPO = Path(__file__).resolve().parents[1]


def table(results) -> list[str]:
    lines = ["| property | holds | cases | statement | counterexample |", "|---|---|---|---|---|"]
    for r in results:
        cex = json.dumps(r.counterexample) if r.counterexample else ""
        lines.append(f"| {r.name} | {'yes' if r.holds else 'NO'} | {r.n_cases} | {r.statement} | {cex} |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True, help="output path without suffix")
    args = p.parse_args()
    shield_results = verify_shield()
    intro = ("Every property is checked on every case of its finite domain by calling the real functions "
             "(`src/recoverdrive/formal.py`); a failing property shows its first counterexample.")
    md = ["# Formal verification of the safety logic", "", intro, "", "## Safety shield", ""] + table(shield_results)
    md += ["", f"git: {git_state(REPO)}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"shield": [asdict(r) for r in shield_results], "git": git_state(REPO)}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
