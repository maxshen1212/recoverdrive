"""Exhaustive verification of the safety shield (and, from Task 5, the recovery controller). CPU only.

    uv run python scripts/verify_logic.py --out /data/maxshen/recoverdrive/report_figures/formal
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from recoverdrive.formal import (
    DEFAULT_CONTROLLER_SENSORS,
    explore_controller,
    verify_controller,
    verify_shield,
)
from recoverdrive.recovery import RecoveryConfig
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

    cfg = RecoveryConfig()  # the frozen Ours-full settings
    graph = explore_controller(DEFAULT_CONTROLLER_SENSORS, cfg)
    ctrl_results = verify_controller(graph, cfg)
    sanity_cfg = RecoveryConfig(reobserve=False)
    sanity = {r.name: r for r in verify_controller(explore_controller(DEFAULT_CONTROLLER_SENSORS, sanity_cfg),
                                                   sanity_cfg)}["RC2"]
    method = (f"Explicit-state model checking of `RecoveryController` with the frozen settings (tau = "
              f"{cfg.conf_threshold}). Sensors: {', '.join(DEFAULT_CONTROLLER_SENSORS)}. The controller never looks at "
              "sensor names and the shield only checks membership in the front set, so two front and two non-front "
              "sensors cover every role combination of the 7 sensors. Inputs per step: no valid answer, ok, "
              "uncertain or fault on each sensor, with confidence 0.5 or 1.0, and every set of flagged sensors.")
    sanity_line = (f"Sanity check: with re-observation switched off, RC2 {'holds' if sanity.holds else 'fails'} "
                   f"({json.dumps(sanity.counterexample)}), so the checker can find violations.")
    md += ["", "## Recovery controller", "", method, "",
           f"Reachable states: {len(graph.states)}; transitions checked: {len(graph.edges)}.", ""]
    md += table(ctrl_results)
    md += ["", sanity_line]
    md += ["", f"git: {git_state(REPO)}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"shield": [asdict(r) for r in shield_results], "controller": [asdict(r) for r in ctrl_results],
         "controller_states": len(graph.states), "controller_edges": len(graph.edges),
         "sanity_c2_without_reobserve": asdict(sanity), "git": git_state(REPO)}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
