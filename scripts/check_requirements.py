"""Check the formal requirements R1-R6 on logged runs (no model calls, CPU only).

nuScenes: every run file of each --eval folder, every method in it. CARLA: every run of --carla. Writes
<out>.md and <out>.json with, per method and requirement: runs satisfied, obligations, violations and the lowest
robustness; per fault type, the episode-level R1 violation rate with its Clopper-Pearson upper bound; and a cross-check
that R1 with no grace flags exactly the frames the E2 metric counts as missed fallbacks.

    uv run python scripts/check_requirements.py \
        --eval /data/maxshen/recoverdrive/outputs/eval/test_final /data/maxshen/recoverdrive/outputs/eval/test_checks_only \
        --carla /data/maxshen/recoverdrive/outputs/e3/e3_final --out /data/maxshen/recoverdrive/report_figures/requirements
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from recoverdrive.bounds import episode_violation_bounds
from recoverdrive.requirements import (
    CARLA_SENSORS,
    CARLA_SPECS,
    NUSCENES_SENSORS,
    NUSCENES_SPECS,
    carla_trace,
    evaluate,
    nuscenes_trace,
    spec,
)
from recoverdrive.runner import git_state
from recoverdrive.shield import acceptable_actions, classify_action
from recoverdrive.stl import series
from recoverdrive.types import FaultLabel

REPO = Path(__file__).resolve().parents[1]
BOUND_METHODS = ("ours_full", "checks_only")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def new_cell() -> dict:
    return {"runs": 0, "runs_ok": 0, "obligations": 0, "violations": 0, "min_rho": math.inf}


def add(cell: dict, res) -> None:
    cell["runs"] += 1
    cell["runs_ok"] += res.satisfied
    cell["obligations"] += res.n_obligations
    cell["violations"] += res.n_violations
    cell["min_rho"] = min(cell["min_rho"], res.robustness)


def check_nuscenes(dirs: list[Path]) -> dict:
    table: dict = defaultdict(lambda: defaultdict(new_cell))
    bound_rows: dict = defaultdict(list)
    cross = defaultdict(lambda: {"r1_grace0_violations": 0, "missed_frames": 0, "frames": 0})
    skipped = 0
    for d in dirs:
        for f in sorted((d / "runs").glob("*.jsonl")):
            recs = read_jsonl(f)
            methods = sorted({m for r in recs for m in r.get("methods", {})})
            for m in methods:
                tr, sk = nuscenes_trace(recs, m, NUSCENES_SENSORS)
                skipped += sk
                for name, res in evaluate(tr, NUSCENES_SPECS, NUSCENES_SENSORS).items():
                    add(table[m][name], res)
                ok = sorted((r for r in recs if "error" not in r and m in r.get("methods", {})),
                            key=lambda r: r["frame_idx"])
                g0 = series(spec("R1", grace=0), tr)
                c = cross[m]
                c["r1_grace0_violations"] += int((g0 < 0).sum())
                c["frames"] += len(ok)
                c["missed_frames"] += sum(classify_action(r["methods"][m]["action"]["action"],
                                                          acceptable_actions(FaultLabel(**r["label"]))) == "missed"
                                          for r in ok)
                if m in BOUND_METHODS:
                    g1 = series(spec("R1"), tr)
                    for i, r in enumerate(ok):
                        if r["label"]["fault_id"] != "none":
                            bound_rows[m].append({"run": f.stem, "fault_id": r["label"]["fault_id"],
                                                  "episode_id": r["label"]["episode_id"],
                                                  "violated": i < len(g1) and g1[i] < 0})
    return {"table": table, "bounds": {m: episode_violation_bounds(rows) for m, rows in bound_rows.items()},
            "cross_check": dict(cross), "skipped_records": skipped}


def check_carla(d: Path) -> dict:
    table: dict = defaultdict(lambda: defaultdict(new_cell))
    for summary_path in sorted(d.glob("*.json")):
        if summary_path.name == "meta.json":
            continue
        summary = json.loads(summary_path.read_text())
        steps_path = summary_path.with_suffix(".jsonl")
        if "method" not in summary or not steps_path.exists():
            continue
        tr, _ = carla_trace(read_jsonl(steps_path), summary, CARLA_SENSORS)
        key = f"{summary['method']} / {summary['condition']}"
        for name, res in evaluate(tr, CARLA_SPECS, CARLA_SENSORS).items():
            add(table[key][name], res)
    return {"table": table}


def fmt_table(table: dict, specs: tuple[str, ...], label: str) -> list[str]:
    lines = [f"| {label} | " + " | ".join(f"{s} runs ok | {s} viol. rate | {s} min ρ" for s in specs) + " |",
             "|---|" + "---|---|---|" * len(specs)]
    for m in sorted(table):
        cells = []
        for s in specs:
            c = table[m][s]
            rate = c["violations"] / c["obligations"] if c["obligations"] else 0.0
            cells += [f"{c['runs_ok']}/{c['runs']}", f"{rate:.3f}", f"{c['min_rho']:.2f}"]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return lines


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, float) and math.isinf(obj):
        return None
    return obj


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--eval", nargs="+", type=Path, required=True, help="eval run folders (with runs/*.jsonl)")
    p.add_argument("--carla", type=Path, default=None, help="E3 output folder")
    p.add_argument("--out", type=Path, required=True, help="output path without suffix")
    args = p.parse_args()

    nus = check_nuscenes(args.eval)
    carla = check_carla(args.carla) if args.carla else None
    intro = ("R1-R6 as defined in `src/recoverdrive/requirements.py` (STL, 0.5 s steps). \"runs ok\" = runs in which "
             "every obligation holds; \"viol. rate\" = violated obligations / obligations; \"min ρ\" = lowest "
             "robustness (negative = violated). In CARLA, R1/R2 judge the monitor's action and R5 the applied action "
             "(after the vehicle's TTC brake).")
    md = ["# Formal requirements on the logged runs", "", intro, "",
          f"Sources: {', '.join(str(d) for d in args.eval)}" + (f"; CARLA: {args.carla}" if args.carla else ""), "",
          "## nuScenes (test split)", ""]
    md += fmt_table(nus["table"], NUSCENES_SPECS, "method")
    md += ["", "Cross-check: R1 with no grace must flag exactly the frames the E2 metric counts as missed.", "",
           "| method | R1 (grace 0) violations | E2 missed frames | frames |", "|---|---|---|---|"]
    md += [f"| {m} | {c['r1_grace0_violations']} | {c['missed_frames']} | {c['frames']} |"
           for m, c in sorted(nus["cross_check"].items())]
    for m, rows in nus["bounds"].items():
        md += ["", f"Episode-level R1 violations of {m} (95% Clopper-Pearson upper bound):", "",
               "| fault | episodes | violated | rate | upper bound |", "|---|---|---|---|---|"]
        md += [f"| {r['fault_id']} | {r['episodes']} | {r['violated']} | {r['rate']:.3f} | {r['upper']:.3f} |"
               for r in rows]
    if carla:
        md += ["", "## CARLA closed loop (E3)", ""] + fmt_table(carla["table"], CARLA_SPECS, "method / condition")
    md += ["", f"Skipped records (frame errors or missing method): {nus['skipped_records']}",
           f"git: {git_state(REPO)}"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(to_jsonable(
        {"nuscenes": nus, "carla": carla, "sources": [str(d) for d in args.eval], "git": git_state(REPO)}), indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
