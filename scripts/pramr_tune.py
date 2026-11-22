"""Offline replay and threshold tuning of the PRAM-R baseline (plan 2026-11-22).

The router prompt never depends on the thresholds, so the logged reliability scores can be replayed through the same
hysteresis (pramr.latch_step) with any theta / delta.

    uv run python scripts/pramr_tune.py --tag dev_pramr_s0 --check            # replay == live outputs?
    uv run python scripts/pramr_tune.py --tag dev_pramr_s0 --max-fa 0.10      # grid search (dev tags only)

Selection rule, fixed before looking at results: the highest F1 among settings with false-alarm rate <= --max-fa;
ties go to the lower false-alarm rate, then the smaller delta, then the lower thresholds. Writes pramr_tune.{json,md}
next to the run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from recoverdrive import pramr
from recoverdrive.metrics import e1_counts, e2_counts
from recoverdrive.runner import RunSpec, load_records
from recoverdrive.shield import shield
from recoverdrive.types import RecoveryResult

REPO = Path(__file__).resolve().parents[1]
THETAS = range(300, 801, 50)
DELTAS = (0, 50, 100)


def load(tag: str) -> tuple[list[dict], dict]:
    out = Path("outputs/eval") / tag
    a = json.loads((out / "meta.json").read_text())["args"]
    splits = yaml.safe_load((REPO / "configs" / "splits.yaml").read_text())
    scenes = a.get("scenes") or splits[a["split"]]
    faults = a["faults"]
    if faults == ["all"]:
        from recoverdrive.faults import ALL_FAULTS
        faults = list(ALL_FAULTS)
    specs = [RunSpec(a["split"], sc, f, s) for s in a["seeds"] for f in faults for sc in scenes]
    return [r for r in load_records(out, specs) if "error" not in r], a


def replay(records: list[dict], method: str, tc: int, tl: int, delta: int) -> list[tuple[dict, dict]]:
    """(record, replayed output) pairs, one run at a time in frame order, starting with every sensor on."""
    runs: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if method in r.get("methods", {}):
            runs[(r["scene"], r["run_fault"], r["seed"])].append(r)
    pairs = []
    for key in sorted(runs):
        active, last = None, None
        for r in sorted(runs[key], key=lambda x: x["frame_idx"]):
            ex = r["methods"][method]["extra"]
            keys = list(ex["active_before"])
            active = active if active is not None else dict.fromkeys(keys, True)
            scores = ex["scores"]
            before = dict(active)
            active = pramr.latch_step(before, scores, tc, tl, delta)
            last = scores if scores is not None else last
            isolated, pred = pramr.outcome(active, last, keys)
            result = RecoveryResult(["pramr"], "degraded" if isolated else "healthy", isolated, None)
            out = {"alarm": bool(isolated), "pred_sensor": pred, "action": {"action": shield(result).action},
                   "recovery": {"final_state": result.final_state, "isolated_sensors": isolated}, "extra": {}}
            if r["label"]["sensor"] == "agent":
                clean = ex["clean_scores"]
                valid = r["methods"][method]["valid"]
                out["extra"]["agent_recovered"] = bool(valid and clean is not None
                                                       and active == pramr.latch_step(before, clean, tc, tl, delta))
            pairs.append((r, out))
    return pairs


def score(pairs) -> dict:
    e1, e2 = e1_counts(pairs), e2_counts(pairs, True)
    return {"F1": e1["F1"], "P": e1["P"], "R": e1["R"], "sensor_acc": e1["sensor_acc"],
            "false_alarm": e1["false_alarm"], "unnecessary": e2["unnecessary"], "missed": e2["missed"],
            "rec_success": e2["rec_success"]}


def check(records: list[dict], method: str) -> tuple[int, int]:
    params = next(r["methods"][method]["extra"]["params"] for r in records if method in r.get("methods", {}))
    pairs = replay(records, method, params["theta_camera"], params["theta_lidar"], params["delta"])
    same = 0
    for r, out in pairs:
        live = r["methods"][method]
        same += (live["alarm"] == out["alarm"] and live["pred_sensor"] == out["pred_sensor"]
                 and live["recovery"]["isolated_sensors"] == out["recovery"]["isolated_sensors"]
                 and live["action"]["action"] == out["action"]["action"]
                 and live["extra"].get("agent_recovered") == out["extra"].get("agent_recovered"))
    return same, len(pairs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--check", action="store_true", help="only verify that the replay equals the live outputs")
    p.add_argument("--max-fa", type=float, default=0.10)
    args = p.parse_args()
    records, a = load(args.tag)
    methods = [m for m in a["methods"] if m.startswith("pramr")]
    if args.check:
        bad = False
        for m in methods:
            same, n = check(records, m)
            print(f"{m}: replay reproduces {same}/{n} live frames")
            bad |= same != n
        sys.exit(1 if bad else 0)
    if a["split"] != "dev":
        sys.exit("refusing to tune on a non-dev run (spec §5.4)")
    rows = []
    for m in methods:
        for tc, tl, d in itertools.product(THETAS, THETAS, DELTAS):
            rows.append({"method": m, "theta_camera": tc, "theta_lidar": tl, "delta": d,
                         **score(replay(records, m, tc, tl, d))})
    ok = [r for r in rows if r["F1"] is not None and r["false_alarm"] is not None and r["false_alarm"] <= args.max_fa]
    ok.sort(key=lambda r: (-r["F1"], r["false_alarm"], r["delta"], r["theta_camera"], r["theta_lidar"]))
    best = ok[0] if ok else None
    out = Path("outputs/eval") / args.tag
    (out / "pramr_tune.json").write_text(json.dumps({"max_fa": args.max_fa, "best": best, "rows": rows}, indent=1))
    lines = [f"# PRAM-R tuning on {args.tag} (false alarm <= {args.max_fa})", "",
             "| method | theta_cam | theta_lidar | delta | F1 | P | R | sensor acc. | false alarm | unnecessary | missed |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        for r in [x for x in ok if x["method"] == m][:8]:
            lines.append(f"| {m} | {r['theta_camera']} | {r['theta_lidar']} | {r['delta']} | {r['F1']:.3f} | "
                         f"{r['P']:.3f} | {r['R']:.3f} | {r['sensor_acc']:.3f} | {r['false_alarm']:.3f} | "
                         f"{r['unnecessary']:.3f} | {r['missed']:.3f} |")
    lines += ["", f"Chosen: {best}"]
    (out / "pramr_tune.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
