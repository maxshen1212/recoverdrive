"""Classify every violation a falsification run found (CPU only, reads the trial logs).

For each violating trial, take its first violated obligation and name the cause from the logged frame:
- the label, the flags, and the reference objects of the faulty camera (`n_gt`);
- the severity level relative to the label boundaries.
A class that no assumption or boundary explains is listed as a bug candidate.

    uv run python scripts/triage_falsification.py --run /data/maxshen/recoverdrive/outputs/falsify/nuscenes_cpu_v1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from recoverdrive.checks import flagged_sensors
from recoverdrive.faults import C2_LEVELS, C3_LEVELS, L1_LEVELS, L2_LEVELS
from recoverdrive.requirements import NUSCENES_SENSORS, nuscenes_trace, spec
from recoverdrive.runner import git_state
from recoverdrive.stl import series

REPO = Path(__file__).resolve().parents[1]
MIN_REFERENCE = 2  # assumption A3
NEAR = 1.25  # a level below 1.25 x the "medium" value counts as near the low/medium label boundary


def near_boundary(fault: str, p: dict, severity: str) -> bool:
    if severity != "medium":
        return False
    if fault == "L2":
        return p["translation_m"] < NEAR * L2_LEVELS[1][0] and abs(p["yaw_deg"]) < NEAR * L2_LEVELS[1][1]
    table = {"C2": C2_LEVELS, "C3": C3_LEVELS, "L1": L1_LEVELS}.get(fault)
    return bool(table) and p.get("level") is not None and p["level"] < NEAR * table[p["subtype"]][1]


def flags_at(r: dict) -> set[str]:
    return flagged_sensors(r["flags"]) - {"agent"}


def n_ref(r: dict, sensor: str) -> int | None:
    if not sensor.startswith("camera:"):
        return None
    return r["report"]["cameras"][sensor.split(":", 1)[1]]["n_gt"]


def classify(spec_name: str, recs: list[dict], t: int, method: str, p: dict, fault: str) -> str:
    r = recs[t]
    lab = r["label"]
    s = lab["sensor"]
    if spec_name == "R1":
        window = recs[t:t + 2]
        if not any(s in flags_at(x) for x in window):
            if (n := n_ref(r, s)) is not None and n < MIN_REFERENCE:
                return "missed: camera has < 2 reference objects (A3)"
            if near_boundary(fault, p, lab["severity"]):
                return "missed: level near the low/medium label boundary"
            return "missed: fault below the detection limit of the checks"
        return "BUG? flagged but the action stayed too permissive"
    if spec_name == "R2":
        if lab["fault_id"] != "none":
            state = r["methods"][method]["recovery"]["final_state"]
            return f"cautious on a fault frame ({state}: other sensors flagged too)"
        prev = recs[t - 1] if t > 0 else None
        if prev is not None and prev["label"]["fault_id"] != "none":
            return "first clean frame after the episode (isolation not yet re-observed)"
        if flags_at(r):
            return "false alarm of the checks on a clean frame"
        return "BUG? cautious on a clean frame without flags"
    if spec_name == "R3":
        prev_sensor = recs[t - 1]["label"]["sensor"]
        window = recs[t:t + 3]
        if any(prev_sensor in flags_at(x) for x in window):
            return "the faulty sensor stays flagged after the episode"
        if any(flags_at(x) for x in window):
            return "other sensors flagged after the episode (false alarm)"
        return "BUG? not back to continue without any flag"
    if spec_name == "R6":
        nxt = recs[t + 1]
        if s in flags_at(nxt):
            return "BUG? restored while still flagged"
        if (n := n_ref(nxt, s)) is not None and n < MIN_REFERENCE:
            return "evidence lost: < 2 reference objects (A3)"
        return "evidence intermittent: the check misses one frame of an active fault"
    raise ValueError(spec_name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", type=Path, required=True, help="falsification output folder")
    p.add_argument("--method", default="checks_only")
    args = p.parse_args()
    rows = [json.loads(line) for line in (args.run / "trials.jsonl").read_text().splitlines() if line.strip()]
    counts: dict = defaultdict(Counter)
    examples: dict = {}
    for row in rows:
        if row["satisfied"]:
            continue
        recs = []
        for f in (Path(row["run_dir"]) / "runs").glob("*.jsonl"):
            recs += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        recs = sorted((r for r in recs if "error" not in r), key=lambda r: r["frame_idx"])
        trace, _ = nuscenes_trace(recs, args.method, NUSCENES_SENSORS)
        vals = series(spec(row["spec"], NUSCENES_SENSORS), trace)
        t = int(next(i for i, v in enumerate(vals) if v < 0))
        cls = classify(row["spec"], recs, t, args.method, row["params"], row["fault"])
        key = (row["fault"], row["spec"])
        counts[key][cls] += 1
        examples.setdefault((row["fault"], row["spec"], cls), row["run_dir"])
    md = [f"# Falsification triage ({args.run.name})", "",
          "First violated obligation of every violating trial (both samplers), classified from the logged frame.", "",
          "| fault | spec | cause | trials | example |", "|---|---|---|---|---|"]
    total = Counter()
    for (fault, spec_name), c in sorted(counts.items()):
        for cls, k in c.most_common():
            total[cls] += k
            md.append(f"| {fault} | {spec_name} | {cls} | {k} | {examples[(fault, spec_name, cls)]} |")
    md += ["", "Totals by cause:", "", "| cause | trials |", "|---|---|"]
    md += [f"| {cls} | {k} |" for cls, k in total.most_common()]
    md += ["", f"Bug candidates: {sum(k for cls, k in total.items() if cls.startswith('BUG'))}",
           f"git: {git_state(REPO)}"]
    (args.run / "triage.md").write_text("\n".join(md) + "\n")
    (args.run / "triage.json").write_text(json.dumps(
        {"by_campaign": {f"{f}/{s}": dict(c) for (f, s), c in counts.items()}, "totals": dict(total),
         "git": git_state(REPO)}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
