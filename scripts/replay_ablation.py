"""E5 recovery ablations and the §4.2 VLM gate, replayed offline from a finished run (no GPU, no new VLM calls).

Every Ours-full frame record keeps the raw VLM answers (first attempt, re-asks, the 3 samples) and the check flags.
The recovery controller is deterministic, so re-running it with one switch changed gives the exact result that a live
run with that switch would give. Variant `full` must reproduce the logged Ours-full on every frame (--check).

Usage: uv run python scripts/replay_ablation.py --tag test_final [--check]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from recoverdrive.checks import flagged_sensors
from recoverdrive.methods import full_outcome, same_verdict
from recoverdrive.metrics import aggregate_seeds, e1_table, e2_table, markdown_table
from recoverdrive.recovery import RecoveryConfig, RecoveryController, vote
from recoverdrive.runner import git_state, sensors_of
from recoverdrive.shield import shield
from recoverdrive.types import Diagnosis
from recoverdrive.vlm import ours_schema, to_diagnosis, validate_json

VARIANTS = {
    "full": {},
    "no_sampling": {"sampling": False},
    "no_reobserve": {"reobserve": False},
    "no_isolation": {"isolate": False},
    "gate_flags": {"gate": "flags"},
}
E1_COLS = ["method", "fault", "n", "P", "R", "F1", "sensor_acc", "false_alarm"]
E2_COLS = ["method", "fault", "rec_success", "ep_recovered", "frames_to_recover", "unnecessary", "missed"]


def replay_run(records: list[dict], rcfg: RecoveryConfig, schema: dict) -> list[dict]:
    """Ours-full outputs for one run (records in frame order) under `rcfg`."""
    ctrl = RecoveryController(rcfg)
    outs = []
    for r in records:
        logged = r["methods"]["ours_full"]
        flagged = flagged_sensors(r["flags"]) - {"agent"}
        attempts = logged["extra"].get("attempts", [])
        samples = logged["extra"].get("samples", [])
        agent_fault = r["label"]["sensor"] == "agent"
        if rcfg.gate == "flags" and not flagged:
            diag = Diagnosis("ok", "none", 1.0, "none", "gated")
            steps, valid, n_calls = ["gated"], True, 0
            recovered = True
        else:
            steps = ["reask"] * (len(attempts) - 1)
            valid = attempts[-1]["valid"]
            diag = to_diagnosis(validate_json(attempts[-1]["raw"], schema)) if valid else None
            if rcfg.sampling and valid and (diag.status != "ok" or flagged):
                steps.append("resample")
                voted = [to_diagnosis(validate_json(a["raw"], schema)) for a in samples if a["valid"]]
                if voted:
                    diag, _ = vote(voted)
                n_calls = len(attempts) + len(samples)
            else:
                n_calls = len(attempts)
            recovered = valid and same_verdict(diag, attempts[0].get("clean_data"))
        result = ctrl.step(diag if valid else None, valid, flagged, steps)
        alarm, pred = full_outcome(result, steps, diag)
        out = {"alarm": alarm, "pred_sensor": pred, "action": {"action": shield(result).action},
               "recovery": {"final_state": result.final_state, "isolated_sensors": result.isolated_sensors},
               "n_calls": n_calls, "extra": {}}
        if agent_fault:
            out["extra"]["agent_recovered"] = recovered
        outs.append(out)
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="finished run_eval output under outputs/eval/")
    p.add_argument("--check", action="store_true", help="only verify that variant 'full' reproduces the logs")
    args = p.parse_args()
    root = Path("outputs/eval") / args.tag
    meta = json.loads((root / "meta.json").read_text())
    cfg = meta["config"]
    schema = ours_schema(sensors_of(cfg["data"]["cameras"]))
    runs: dict[str, list[dict]] = defaultdict(list)
    for f in sorted((root / "runs").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r and "ours_full" in r.get("methods", {}):
                runs[f.stem].append(r)
    base = {k: v for k, v in cfg.get("recovery", {}).items()}
    replayed: dict[str, list[dict]] = defaultdict(list)  # variant -> records with methods = {variant: out}
    mismatch = n = 0
    for name, recs in runs.items():
        recs.sort(key=lambda r: r["frame_idx"])
        for var, sw in VARIANTS.items():
            outs = replay_run(recs, RecoveryConfig(**{**base, **sw}), schema)
            for r, o in zip(recs, outs):
                if var == "full":
                    lg = r["methods"]["ours_full"]
                    n += 1
                    same = (o["action"]["action"] == lg["action"]["action"] and o["alarm"] == lg["alarm"]
                            and o["pred_sensor"] == lg["pred_sensor"]
                            and o["recovery"]["final_state"] == lg["recovery"]["final_state"]
                            and o["recovery"]["isolated_sensors"] == lg["recovery"]["isolated_sensors"])
                    mismatch += not same
                replayed[var].append({**{k: r[k] for k in ("scene", "frame_idx", "run_fault", "seed", "label")},
                                      "methods": {f"ours_full[{var}]": o}})
    print(f"replay check: variant 'full' reproduces {n - mismatch}/{n} logged Ours-full frames")
    if args.check or mismatch:
        return 0 if mismatch == 0 else 1

    seeds = sorted({r["seed"] for recs in runs.values() for r in recs})
    faults = list(dict.fromkeys(r["run_fault"] for recs in runs.values() for r in recs))
    names = [f"ours_full[{v}]" for v in VARIANTS]
    e1_all, e2_all, rows_calls = [], [], []
    for var, name in zip(VARIANTS, names):
        recs = replayed[var]
        e1 = aggregate_seeds([e1_table([r for r in recs if r["seed"] == s], [name], faults) for s in seeds])
        e2 = aggregate_seeds([e2_table([r for r in recs if r["seed"] == s], [name], faults, recovery_methods=(name,))
                              for s in seeds])
        e1_all += e1
        e2_all += e2
        calls = [r["methods"][name]["n_calls"] for r in recs]
        rows_calls.append({"method": name, "fault": "all", "vlm_calls_per_frame": float(np.mean(calls)),
                           "frames_without_vlm": float(np.mean([c == 0 for c in calls]))})
    text = "\n".join([
        f"# Offline ablation replay ({args.tag}): {n} frames, seeds {seeds}; 'full' reproduces the logs exactly", "",
        "## E1 (all faults)", "", markdown_table([r for r in e1_all if r["fault"] == "all"], E1_COLS), "",
        "## E2 (all faults)", "", markdown_table([r for r in e2_all if r["fault"] == "all"], E2_COLS), "",
        "## VLM calls", "", markdown_table(rows_calls, ["method", "vlm_calls_per_frame", "frames_without_vlm"]), "",
        "## E1 by fault", "", markdown_table(e1_all, E1_COLS), "",
        "## E2 by fault", "", markdown_table(e2_all, E2_COLS)])
    out = root / "ablation_replay.md"
    out.write_text(text + "\n")
    (root / "ablation_replay.json").write_text(json.dumps({"git": git_state(Path(__file__).resolve().parent.parent), "e1": e1_all, "e2": e2_all,
                                                         "calls": rows_calls}))
    print(text.split("## E1 by fault")[0])
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
