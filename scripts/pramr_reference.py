"""Normal ranges of the PRAM-R indicators (plan 2026-11-22): p5 / median / p95 over every clean dev keyframe.

    uv run python scripts/pramr_reference.py --split dev --out configs/pramr_reference.yaml

Only the dev split is allowed (spec §5.4). The file is committed before the PRAM-R test run. It also records the share
of clean frames with at least one indicator outside its range; that number is reported, never tuned on.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from recoverdrive import pramr
from recoverdrive.adapters.nuscenes import NuScenesAdapter
from recoverdrive.runner import git_state

REPO = Path(__file__).resolve().parents[1]


def ranges(values: list[float]) -> dict[str, float]:
    p5, med, p95 = np.percentile(np.asarray(values, float), [5, 50, 95])
    return {"p5": float(p5), "median": float(med), "p95": float(p95)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--split", default="dev")
    p.add_argument("--out", default="configs/pramr_reference.yaml")
    args = p.parse_args()
    if args.split != "dev":
        sys.exit("refusing: the reference ranges come from the dev split only (spec §5.4)")
    cfg = yaml.safe_load(Path(args.config).read_text())
    scenes = yaml.safe_load((REPO / "configs" / "splits.yaml").read_text())["dev"]
    cams = cfg["data"]["cameras"]
    adapter = NuScenesAdapter(os.environ.get("NUSCENES_ROOT", "data/nuscenes"), cfg["data"]["version"], cams,
                              cfg["categories"])
    cam_vals = {c: {k: [] for k in pramr.CAMERA_KEYS} for c in cams}
    lid_vals: dict[str, list[float]] = {"density": [], "noise_ratio": [], "mean_intensity": []}
    light: list[float] = []
    per_frame = []
    for scene in scenes:
        for idx, token in enumerate(tqdm(adapter.scene_sample_tokens(scene), desc=scene)):
            frame = adapter.load_frame(scene, idx, token)
            cam = {c: pramr.camera_indicators(frame.cameras[c].image) for c in cams}
            lid = pramr.lidar_raw_indicators(frame.lidar.points)
            for c in cams:
                for k, v in cam[c].items():
                    cam_vals[c][k].append(v)
            for k, v in lid.items():
                lid_vals[k].append(v)
            light.append(float(np.mean([cam[c]["brightness"] for c in cams])))
            per_frame.append((cam, lid, light[-1]))
    out = {c: {k: ranges(v) for k, v in cam_vals[c].items()} for c in cams}
    median_int = float(np.median(lid_vals["mean_intensity"]))
    out[pramr.LIDAR_KEY] = {"density": ranges(lid_vals["density"]), "noise_ratio": ranges(lid_vals["noise_ratio"]),
                            "reflectivity_ratio": ranges([v / median_int for v in lid_vals["mean_intensity"]]),
                            "mean_intensity": ranges(lid_vals["mean_intensity"])}
    out["light"] = {"brightness": ranges(light)}

    def outside(cam, lid, lt) -> bool:
        vals = [(out[c][k], v) for c in cams for k, v in cam[c].items()]
        vals += [(out[pramr.LIDAR_KEY]["density"], lid["density"]), (out[pramr.LIDAR_KEY]["noise_ratio"],
                                                                     lid["noise_ratio"]),
                 (out[pramr.LIDAR_KEY]["reflectivity_ratio"], lid["mean_intensity"] / median_int),
                 (out["light"]["brightness"], lt)]
        return any(pramr.label(v, r) != "normal" for r, v in vals)

    share = float(np.mean([outside(*f) for f in per_frame]))
    doc = {"description": "PRAM-R indicator ranges over clean dev keyframes (plan 2026-11-22)", "split": "dev",
           "scenes": scenes, "frames": len(per_frame), "share_frames_any_outside_range": round(share, 4),
           "git": git_state(REPO), "ranges": out}
    Path(args.out).write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"{len(per_frame)} clean dev frames; {share:.1%} have at least one indicator outside its range -> {args.out}")


if __name__ == "__main__":
    main()
