"""Falsification campaigns on nuScenes dev scenes: random vs. guided (TPE) search for fault settings that violate
R1, R2, R3 or R6. One campaign = one (fault, requirement, sampler); campaigns run in parallel processes.

Phase A (no GPU) runs the faults whose images are all original images, with a cache-only SAM 3 detector:

    CUDA_VISIBLE_DEVICES="" uv run python scripts/falsify_nuscenes.py --cache-only \
        --faults C1 C4 L1 L2 L4 --specs R1 R2 R3 R6 --budget 60 --tag cpu_v1 \
        --cache /data/maxshen/recoverdrive/outputs/cache/eval.sqlite \
        --nuscenes-root /data/maxshen/recoverdrive/data/nuscenes --out-root /data/maxshen/recoverdrive/outputs/falsify

Phase B adds C2/C3 (new images, SAM 3 on a GPU) and `--confirm K --method ours_full` (VLM).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from recoverdrive.adapters.nuscenes import NuScenesAdapter
from recoverdrive.cache import KVCache
from recoverdrive.detectors.sam3 import CachedDetector
from recoverdrive.falsify import CPU_FAULTS, FAULT_SPACES, GPU_FAULTS, campaign, evaluate_trial
from recoverdrive.runner import EvalEnv, git_state

REPO = Path(__file__).resolve().parents[1]
CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")


def search_space(fault_id: str, scenes: list[str]) -> dict:
    """Normalized search space; `decode` turns a sample into evaluate_trial parameters."""
    space = {"scene": ("cat", scenes), "start_u": ("float", 0.0, 1.0), "duration": ("int", 2, 6),
             "seed": ("int", 0, 9999)}
    if fault_id in ("C2", "C3", "C4"):
        space["camera"] = ("cat", list(CAMERAS))
    if fault_id in ("C2", "C3", "L1"):
        space["subtype"] = ("cat", list(FAULT_SPACES[fault_id]))
        space["level_u"] = ("float", 0.0, 1.0)
    if fault_id == "C2":
        space["sign"] = ("cat", [-1, 1])
    if fault_id == "C4":
        space["lag"] = ("cat", list(FAULT_SPACES["C4"]["lag"]))
    if fault_id == "L1":
        space["azimuth_deg"] = ("float", 0.0, 360.0)
    if fault_id == "L2":
        space.update(translation_m=("float", *FAULT_SPACES["L2"]["translation_m"]),
                     yaw_deg=("float", *FAULT_SPACES["L2"]["yaw_deg"]), yaw_sign=("cat", [-1, 1]),
                     direction_deg=("float", 0.0, 360.0))
    return space


def decode(fault_id: str, raw: dict, n_frames: int) -> dict:
    p = {k: v for k, v in raw.items() if k not in ("scene", "start_u", "level_u")}
    p["start"] = round(raw["start_u"] * (n_frames - 2))
    if "level_u" in raw:
        bounds = FAULT_SPACES[fault_id][raw["subtype"]]
        p["level"] = None if bounds is None else bounds[0] + raw["level_u"] * (bounds[1] - bounds[0])
    return p


def make_env(args, out_dir: Path) -> EvalEnv:
    cfg = yaml.safe_load((REPO / "configs" / "eval.yaml").read_text())
    adapter = NuScenesAdapter(str(args.nuscenes_root), cfg["data"]["version"], cfg["data"]["cameras"],
                              cfg["categories"])
    s = cfg["sam3"]
    detector = CachedDetector(KVCache(args.cache), s["prompts"], s["score_threshold"], s["model_id"],
                              allow_miss=not args.cache_only)
    return EvalEnv(cfg, adapter, detector, None, out_dir, [args.method])


def run_campaign(args, fault_id: str, spec_name: str, sampler: str, scenes: list[str]) -> list[dict]:
    name = f"{fault_id}_{spec_name}_{sampler}"
    cdir = args.out / name
    env = make_env(args, cdir)
    n_frames = {sc: len(env.adapter.scene_sample_tokens(sc)) for sc in scenes}
    counter = iter(range(10**9))

    def objective(raw):
        i = next(counter)
        p = decode(fault_id, raw, n_frames[raw["scene"]])
        res = evaluate_trial(env, raw["scene"], fault_id, p, args.method, spec_name, cdir / f"trial_{i:04d}")
        res.params = {"raw": raw, "decoded": p}
        return res

    results = campaign(objective, search_space(fault_id, scenes), sampler, args.budget, args.seed)
    rows = [{"campaign": name, "fault": fault_id, "spec": spec_name, "sampler": sampler, "trial": i,
             "scene": r.params["raw"]["scene"], "params": r.params["decoded"],
             "robustness": r.robustness if math.isfinite(r.robustness) else None, "satisfied": r.satisfied,
             "n_violations": r.n_violations, "run_dir": r.run_dir} for i, r in enumerate(results)]
    (cdir / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def failure_type(row: dict) -> str:
    p = row["params"]
    role = "lidar" if row["fault"].startswith("L") else ("front" if p.get("camera", "CAM_FRONT") == "CAM_FRONT"
                                                          else "rear/side")
    return f"{row['fault']}/{p.get('subtype', p.get('lag', '-'))}/{role}"


def summarize(rows: list[dict]) -> list[str]:
    by = {}
    for r in rows:
        by.setdefault((r["fault"], r["spec"]), {}).setdefault(r["sampler"], []).append(r)
    header = ("| fault | spec | sampler | trials | violations | min robustness | first violation (trial) | "
              "failure types (violations) |")
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for (fault, spec_name), samplers in sorted(by.items()):
        for sampler, rs in sorted(samplers.items()):
            rs = sorted(rs, key=lambda r: r["trial"])
            viol = [r for r in rs if not r["satisfied"]]
            finite = [r["robustness"] for r in rs if r["robustness"] is not None]
            first = viol[0]["trial"] if viol else "-"
            types = ", ".join(f"{k}: {v}" for k, v in Counter(failure_type(r) for r in viol).most_common())
            lines.append(f"| {fault} | {spec_name} | {sampler} | {len(rs)} | {len(viol)} | "
                         f"{min(finite) if finite else 'inf'} | {first} | {types} |")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--faults", nargs="+", default=list(CPU_FAULTS))
    p.add_argument("--specs", nargs="+", default=["R1", "R2", "R3", "R6"])
    p.add_argument("--samplers", nargs="+", default=["random", "tpe"])
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="checks_only")
    p.add_argument("--scenes", nargs="*", default=None, help="subset of the dev scenes")
    p.add_argument("--cache", type=Path, required=True, help="SAM 3 / VLM cache (outputs/cache/eval.sqlite)")
    p.add_argument("--nuscenes-root", type=Path, default=Path(os.environ.get("NUSCENES_ROOT", "data/nuscenes")))
    p.add_argument("--cache-only", action="store_true", help="never load SAM 3: a cache miss stops the campaign")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-root", type=Path, default=Path("outputs/falsify"))
    args = p.parse_args()

    splits = yaml.safe_load((REPO / "configs" / "splits.yaml").read_text())
    scenes = args.scenes or splits["dev"]
    if not set(scenes) <= set(splits["dev"]):
        raise SystemExit("falsification runs on dev scenes only")
    if args.cache_only and set(args.faults) & set(GPU_FAULTS):
        raise SystemExit(f"{sorted(set(args.faults) & set(GPU_FAULTS))} create new images and need SAM 3 (no "
                         "--cache-only)")
    args.out = args.out_root / f"nuscenes_{args.tag}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                    "scenes": scenes, "git": git_state(REPO)}, indent=2))
    jobs = [(f, s, sm) for f in args.faults for s in args.specs for sm in args.samplers]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_campaign, args, f, s, sm, scenes): (f, s, sm) for f, s, sm in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            out = fut.result()
            rows += out
            viol = sum(not r["satisfied"] for r in out)
            print(f"[{i}/{len(jobs)}] {out[0]['campaign']}: {len(out)} trials, {viol} violations", flush=True)
    (args.out / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    md = [f"# Falsification on nuScenes ({args.tag})", "",
          f"Method: {args.method}; scenes: {', '.join(scenes)}; budget {args.budget} per campaign; seed {args.seed}.",
          "Robustness < 0 = the requirement is violated in that trial window.", ""] + summarize(rows)
    md += ["", f"git: {git_state(REPO)}"]
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
