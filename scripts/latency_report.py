"""E4 latency breakdown (spec §5.3) from the controlled runs (fresh cache, 1 worker, methods sequential).

Usage: uv run python scripts/latency_report.py --ours e4_ours --baselines e4_baselines
Writes outputs/e4/latency.md and latency.json. The first frame of each run is dropped (model loading, warm-up).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from recoverdrive.metrics import markdown_table
from recoverdrive.runner import git_state

COMPONENTS = ["load", "sam3", "checks", "render", "vlm_first", "vlm_reask", "vlm_samples", "recovery"]


def records(tag: str) -> list[dict]:
    out = []
    for f in sorted((Path("outputs/eval") / tag / "runs").glob("*.jsonl")):
        recs = [json.loads(line) for line in f.read_text().splitlines()]
        out += [r for r in recs if "error" not in r and r["frame_idx"] > 0]
    return out


def stats(x) -> dict:
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", default="e4_ours")
    p.add_argument("--baselines", default="e4_baselines")
    args = p.parse_args()
    ours = records(args.ours)
    comp = {c: [] for c in COMPONENTS}
    total_full, total_detect, total_gated, flagged, method_t = [], [], [], [], []
    for r in ours:
        lat, out = r["latency_s"], r["methods"]["ours_full"]
        t = out["extra"]["timing"]
        vals = {"load": lat["load"], "sam3": lat["sam3"], "checks": lat["checks"], "render": lat["render"],
                "vlm_first": t["vlm_first"], "vlm_reask": t["vlm_reask"], "vlm_samples": t["vlm_samples"],
                "recovery": t["recovery"]}
        for c in COMPONENTS:
            comp[c].append(vals[c])
        pre = vals["load"] + vals["sam3"] + vals["checks"] + vals["render"]
        total_full.append(pre + lat["method:ours_full"])
        total_detect.append(pre + vals["vlm_first"])
        is_flagged = bool(out["extra"]["flagged"])
        flagged.append(is_flagged)
        method_t.append(lat["method:ours_full"])
        total_gated.append(pre + (lat["method:ours_full"] if is_flagged else 0.0))
    mean_total = float(np.mean(total_full))
    rows = [{"component": c, **stats(v), "share": float(np.mean(v)) / mean_total} for c, v in comp.items()]

    base = records(args.baselines)
    caption, totals = [], {"B1": [], "B2": [], "B3": []}
    for r in base:
        lat, m = r["latency_s"], r["methods"]
        own = {k: m[k]["extra"]["attempts"][0]["latency_s"] for k in totals}
        cap = lat["method:B1"] - own["B1"]  # B1 ran first with a fresh cache: it paid for the 6 captions
        caption.append(cap)
        for k, values in totals.items():
            values.append(lat["load"] + cap + own[k])
    method_rows = [
        {"method": "Ours-detect (1 call)", **stats(total_detect)},
        {"method": "Ours-full", **stats(total_full)},
        {"method": "Ours-full + VLM gate (call only on flags)", **stats(total_gated)},
        *[{"method": f"{k} (6 captions + 1 call)", **stats(v)} for k, v in totals.items()],
    ]
    text = "\n".join([
        f"# E4 latency ({len(ours)} Ours frames, {len(base)} baseline frames; seconds per frame)", "",
        ("Controlled runs: dev scene-0061, fresh cache, 1 worker, methods sequential, one dedicated vLLM server "
         "(Qwen3-VL-8B, RTX 6000 Ada shared with other users' jobs), SAM 3 on another GPU."), "",
        "## Ours-full: time per component", "",
        markdown_table(rows, ["component", "mean", "median", "p90", "share"]), "",
        f"Frames with a deterministic flag (VLM needed under the gate): {np.mean(flagged):.3f}", "",
        "## Per-frame latency by method", "", markdown_table(method_rows, ["method", "mean", "median", "p90"]), "",
        f"Shared caption time of B1-B3 (6 cameras, parallel): mean {np.mean(caption):.2f} s"])
    out = Path("outputs/e4")
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency.md").write_text(text + "\n")
    fl, mt = np.array(flagged), np.array(method_t)
    pre_mean = float(np.mean(total_full) - np.mean(mt))
    (out / "latency.json").write_text(json.dumps({"git": git_state(Path(__file__).resolve().parent.parent),
                                                  "components": rows, "methods": method_rows,
                                                  "flagged_share": float(fl.mean()),
                                                  "caption_mean": float(np.mean(caption)),
                                                  "pre_vlm_mean": pre_mean,
                                                  "method_flagged_mean": float(mt[fl].mean()),
                                                  "method_unflagged_mean": float(mt[~fl].mean())}))
    print(text)


if __name__ == "__main__":
    main()
