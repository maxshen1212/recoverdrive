"""E1/E2 evaluation on nuScenes-mini (spec §5): methods x faults x scenes x seeds -> metric tables.

Usage (after `source scripts/env.sh` and starting scripts/serve_vlm.sh):
    CUDA_VISIBLE_DEVICES=3 uv run python scripts/run_eval.py --split dev --faults C1 L1 A1 \
        --methods ours_detect --seeds 0 --tag p1

Runs are resumable: finished runs (runs/<scene>_<fault>_s<seed>.done) are skipped.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from recoverdrive.adapters.nuscenes import NuScenesAdapter
from recoverdrive.cache import KVCache
from recoverdrive.detectors.sam3 import CachedDetector
from recoverdrive.faults import ALL_FAULTS
from recoverdrive.metrics import aggregate_seeds, e1_recall_by_severity, e1_table, e2_table, markdown_table
from recoverdrive.runner import EvalEnv, RunSpec, check_test_allowed, load_records, run_all
from recoverdrive.vlm import VLMClient, healthy_servers

REPO = Path(__file__).resolve().parent.parent
E1_COLS = ["method", "fault", "n", "n_fault", "P", "R", "F1", "sensor_acc", "false_alarm"]
E2_COLS = ["method", "fault", "n", "rec_success", "ep_recovered", "frames_to_recover", "unnecessary", "missed",
           "unnecessary_clean", "missed_fault"]


def precompute_detections(env: EvalEnv, scenes: list[str]) -> None:
    """SAM 3 on every clean image of the split, so parallel runs mostly hit the cache."""
    for scene in scenes:
        tokens = env.adapter.scene_sample_tokens(scene)
        for idx, tok in enumerate(tqdm(tokens, desc=f"sam3 {scene}", leave=False)):
            frame = env.adapter.load_frame(scene, idx, tok)
            for cam in env.cameras:
                env.detector.detect(frame.cameras[cam].image)
    print(f"SAM 3 cache: {env.detector.hits} hits, {env.detector.misses} new")


def tables(records: list[dict], methods: list[str], faults: list[str], seeds: list[int]) -> str:
    per_seed = [e1_table([r for r in records if r["seed"] == s], methods, faults) for s in seeds]
    e1 = aggregate_seeds(per_seed) if len(seeds) > 1 else per_seed[0]
    sev = e1_recall_by_severity(records, methods, faults)
    per_seed2 = [e2_table([r for r in records if r["seed"] == s], methods, faults) for s in seeds]
    e2 = aggregate_seeds(per_seed2) if len(seeds) > 1 else per_seed2[0]
    out = ["## E1: detection and diagnosis", "", markdown_table(e1, E1_COLS), "",
           "## E1: recall by severity (all seeds pooled)", "",
           markdown_table(sev, ["method", "fault", "low", "n_low", "medium", "n_medium", "high", "n_high"]), "",
           "## E2: recovery and fallback", "", markdown_table(e2, E2_COLS)]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--faults", nargs="+", default=["C1", "L1", "A1"], help="fault ids, or 'all'")
    p.add_argument("--methods", nargs="+", default=["ours_detect"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--scenes", nargs="*", default=None, help="subset of the split's scenes (debugging)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--tag", required=True, help="output folder name under output.root")
    p.add_argument("--no-precompute", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = yaml.safe_load((REPO / "configs" / "splits.yaml").read_text())
    commit = None
    if args.split == "test":
        try:
            commit = check_test_allowed(cfg, REPO)
        except RuntimeError as exc:
            sys.exit(str(exc))
    faults = list(ALL_FAULTS) if args.faults == ["all"] else args.faults
    scenes = args.scenes or splits[args.split]
    if not set(scenes) <= set(splits[args.split]):
        sys.exit(f"scenes {scenes} are not all in split {args.split}")

    out = Path(cfg["output"]["root"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    meta = {"args": vars(args), "config": cfg, "commit": commit}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    cache = KVCache(cfg["cache"])
    v = cfg["vlm"]
    servers = healthy_servers(v["base_url"])
    if not servers:
        sys.exit(f"no healthy VLM server among {v['base_url']}")
    print(f"VLM servers: {servers}")
    client = VLMClient(servers, v["model"], max_tokens=v["max_tokens"], timeout_s=v["timeout_s"], cache=cache)
    adapter = NuScenesAdapter(os.environ.get("NUSCENES_ROOT", "data/nuscenes"), cfg["data"]["version"],
                              cfg["data"]["cameras"], cfg["categories"])
    s = cfg["sam3"]
    detector = CachedDetector(cache, s["prompts"], s["score_threshold"], s["model_id"])
    env = EvalEnv(cfg, adapter, detector, client, out, args.methods)

    if not args.no_precompute:
        precompute_detections(env, scenes)
    specs = [RunSpec(args.split, sc, f, seed) for seed in args.seeds for f in faults for sc in scenes]
    results = run_all(specs, env, workers=args.workers)
    n_err = sum(r["errors"] for r in results)
    n_frames = sum(r["frames"] for r in results)
    records = [r for r in load_records(out, specs) if "error" not in r]
    text = (f"# {args.tag}: split={args.split}, {len(specs)} runs, {n_frames} frames, {n_err} frame errors\n\n"
            + tables(records, args.methods, faults, args.seeds))
    (out / "summary.md").write_text(text + "\n")
    print(text)
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
