"""Calibrate the check flag thresholds on dev scenes (spec §4.1: thresholds are calibrated on dev, then fixed).

1. Checks-only runs (no VLM) of every dev scene x sensor fault (C1-C4, L1, L2, L4), seed 0.
2. Coordinate ascent over a grid per threshold: maximize correct-sensor recall (the faulty sensor is among the
   flagged sensors) on faulty frames, subject to a clean-frame false-alarm rate (any fault flag) <= --max-fa.
Writes outputs/calib/<tag>/calibration.md with the chosen values and a YAML block for configs/eval.yaml.

Usage: CUDA_VISIBLE_DEVICES=3 uv run python scripts/calibrate.py --tag calib_dev
"""

import argparse
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from recoverdrive.adapters.nuscenes import NuScenesAdapter
from recoverdrive.cache import KVCache
from recoverdrive.checks import CheckThresholds, compute_flags, flagged_sensors
from recoverdrive.detectors.sam3 import CachedDetector
from recoverdrive.faults import SENSOR_FAULTS
from recoverdrive.runner import EvalEnv, RunSpec, load_records, run_all
from recoverdrive.types import CameraCheck, CheckReport

REPO = Path(__file__).resolve().parent.parent
GRID = {
    "match_rate_min": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "min_gt": [1, 2, 3, 4],
    "match_drop": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01],
    "min_points": [25000, 28000, 30000, 32000, 33000, 34000],
    "point_drop": [0.03, 0.05, 0.1, 0.2, 0.3],
    "sector_min_points": [50, 100, 200, 400, 600, 800],
    "visible_ratio_min": [0.0, 0.1, 0.2, 0.3, 0.4],
    "agreement_min": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6],
}


def to_report(d: dict) -> CheckReport:
    cams = {k: CameraCheck(**v) for k, v in d["cameras"].items()}
    return CheckReport(**{**d, "cameras": cams, "flags": []})


def score(rows: list[tuple], thr: CheckThresholds) -> dict:
    """rows: (report, frozen_flags, true_sensor or None, fault_id)."""
    fa = n_clean = hit = n_fault = 0
    per_fault: dict[str, list[int]] = {}
    for report, frozen, sensor, fid in rows:
        flagged = flagged_sensors(compute_flags(report, thr, frozen))
        if sensor is None:
            n_clean += 1
            fa += bool(flagged)
        else:
            n_fault += 1
            ok = sensor in flagged
            hit += ok
            per_fault.setdefault(fid, []).append(ok)
    return {"fa": fa / max(n_clean, 1), "recall": hit / max(n_fault, 1),
            "per_fault": {f: float(np.mean(v)) for f, v in per_fault.items()}}


def coordinate_ascent(rows, start: CheckThresholds, max_fa: float, passes: int = 3) -> CheckThresholds:
    best = start

    def key(t):
        s = score(rows, t)
        return (s["fa"] <= max_fa, s["recall"] if s["fa"] <= max_fa else -s["fa"])

    best_key = key(best)
    for _ in range(passes):
        improved = False
        for name, values in GRID.items():
            for v in values:
                cand = dataclasses.replace(best, **{name: v})
                k = key(cand)
                if k > best_key:
                    best, best_key, improved = cand, k, True
        if not improved:
            break
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--tag", default="calib_dev")
    p.add_argument("--max-fa", type=float, default=0.10)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--restarts", type=int, default=12)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    scenes = yaml.safe_load((REPO / "configs" / "splits.yaml").read_text())["dev"]  # dev only, never test
    out = Path("outputs/calib") / args.tag
    out.mkdir(parents=True, exist_ok=True)

    cache = KVCache(cfg["cache"])
    adapter = NuScenesAdapter(os.environ.get("NUSCENES_ROOT", "data/nuscenes"), cfg["data"]["version"],
                              cfg["data"]["cameras"], cfg["categories"])
    s = cfg["sam3"]
    env = EvalEnv(cfg, adapter, CachedDetector(cache, s["prompts"], s["score_threshold"], s["model_id"]), None, out,
                  [])
    specs = [RunSpec("dev", sc, f, 0) for f in SENSOR_FAULTS for sc in scenes]
    results = run_all(specs, env, workers=args.workers)
    errors = sum(r["errors"] for r in results)
    records = [r for r in load_records(out, specs) if "error" not in r]
    rows = [(to_report(r["report"]), [f for f in r["flags"] if f.endswith(":frozen")],
             None if r["label"]["fault_id"] == "none" else r["label"]["sensor"], r["label"]["fault_id"])
            for r in records]
    start = CheckThresholds(**cfg["thresholds"])
    best = coordinate_ascent(rows, start, args.max_fa)
    rng = np.random.default_rng(0)
    for _ in range(args.restarts):  # random restarts against local optima
        init = dataclasses.replace(start, **{k: v[int(rng.integers(len(v)))] for k, v in GRID.items()})
        cand = coordinate_ascent(rows, init, args.max_fa)
        sc, sb = score(rows, cand), score(rows, best)
        if (sc["fa"] <= args.max_fa, sc["recall"]) > (sb["fa"] <= args.max_fa, sb["recall"]):
            best = cand
    s0, s1 = score(rows, start), score(rows, best)
    lines = [f"# Threshold calibration ({args.tag}): dev scenes {scenes}, faults {list(SENSOR_FAULTS)}, seed 0",
             f"- runs {len(specs)}, frames {len(records)}, frame errors {errors}",
             f"- objective: max correct-sensor recall s.t. clean-frame flag rate <= {args.max_fa}", "",
             "| setting | clean FA | correct-sensor recall | " + " | ".join(SENSOR_FAULTS) + " |",
             "|---|---|---|" + "---|" * len(SENSOR_FAULTS)]
    for name, sc in (("initial", s0), ("calibrated", s1)):
        lines.append(f"| {name} | {sc['fa']:.3f} | {sc['recall']:.3f} | "
                     + " | ".join(f"{sc['per_fault'].get(f, float('nan')):.3f}" for f in SENSOR_FAULTS) + " |")
    flag_fa = {}
    for report, frozen, sensor, _ in rows:
        if sensor is None:
            for f in compute_flags(report, best, frozen):
                flag_fa[f.split(":")[-1]] = flag_fa.get(f.split(":")[-1], 0) + 1
    n_clean = sum(r[2] is None for r in rows)
    lines += ["", "Clean frames raising each flag (calibrated): "
              + ", ".join(f"{k} {v / n_clean:.3f}" for k, v in sorted(flag_fa.items(), key=lambda kv: -kv[1]))]
    lines += ["", "```yaml", "thresholds:"] + [f"  {k}: {v}" for k, v in dataclasses.asdict(best).items()] + ["```"]
    text = "\n".join(lines)
    (out / "calibration.md").write_text(text + "\n")
    print(text)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
