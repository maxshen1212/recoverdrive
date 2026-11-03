"""E0 pilot (spec §7): one dev scene, CAM_FRONT, C1 on about half the frames.

Checks that the modules connect and that the LiDAR-camera match rate separates clean frames from C1 frames.

Usage (after `source scripts/env.sh` and starting scripts/serve_vlm.sh):
    uv run python scripts/run_e0.py [--config configs/e0.yaml] [--limit N] [--no-vlm]
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from recoverdrive.build import build_adapter, build_pipeline, load_config
from recoverdrive.faults import FaultInjector, schedule_episodes
from recoverdrive.recording import JsonlWriter, RerunRecorder, draw_overlay

LABELS = ["none", "C1"]
STATUSES = ["ok", "fault", "uncertain", "invalid"]
LATENCY_KEYS = ["lidar", "sam3", "checks", "vlm", "total"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/e0.yaml")
    p.add_argument("--out", default=None, help="output dir (default: <output.root>/<timestamp>)")
    p.add_argument("--limit", type=int, default=None, help="only run the first N frames")
    p.add_argument("--no-vlm", action="store_true", help="skip the Qwen diagnosis (checks only)")
    args = p.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or Path(cfg["output"]["root"]) / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    scene = cfg["data"]["scene"]
    adapter = build_adapter(cfg)
    tokens = adapter.scene_sample_tokens(scene)[: args.limit]
    f = cfg["faults"]
    schedule = schedule_episodes(len(tokens), f["fault_ratio"], np.random.default_rng(cfg["seed"]),
                                 f["min_episode_len"], f["max_episode_len"])
    injector = FaultInjector(schedule, fault_id="C1", params=f["c1"])

    pipeline = build_pipeline(cfg, with_vlm=not args.no_vlm)
    cam = pipeline.cameras[0]

    # Warm up CUDA kernels and the VLM so frame 0 latency is not an outlier.
    warm = adapter.load_frame(scene, 0, tokens[0])
    pipeline.step(warm, None, None)

    jsonl = JsonlWriter(out / "frames.jsonl")
    rec = RerunRecorder(out / "e0.rrd", app_id="recoverdrive_e0")
    rows, errors = [], []
    prev_frame, prev_report = None, None
    t_run = time.perf_counter()
    for idx, token in enumerate(tqdm(tokens, desc=scene)):
        try:
            frame, label = injector.apply(adapter.load_frame(scene, idx, token))
            res = pipeline.step(frame, prev_frame, prev_report)
        except Exception as exc:  # noqa: BLE001 - keep going so one bad frame does not hide the rest
            errors.append({"frame_idx": idx, "error": repr(exc)})
            print(f"frame {idx}: {exc!r}")
            continue
        check = res.report.cameras[cam]
        diag = res.diagnosis
        row = {
            "frame_idx": idx, "sample_token": token, "fault_id": label.fault_id, "episode_id": label.episode_id,
            "n_lidar_objects": res.n_lidar_objects, "n_gt": check.n_gt, "n_det": check.n_det,
            "n_matched": check.n_matched, "match_rate": check.match_rate,
            "lidar_visible_ratio": res.report.lidar_visible_ratio,
            "status": diag.status if diag else ("invalid" if res.vlm else None),
            "sensor": diag.sensor if diag else None, "confidence": diag.confidence if diag else None,
            "vlm_valid": res.vlm.valid if res.vlm else None,
            **{f"latency_{k}": res.latency_s.get(k) for k in LATENCY_KEYS},
        }
        rows.append(row)
        c = res.consistency[cam]
        jsonl.write({**row, "label": label, "report": res.report, "summary": res.summary,
                     "vlm_raw": res.vlm.raw if res.vlm else None, "matches": c.matches,
                     "gt_boxes_2d": c.gt_boxes_2d, "gt_labels": c.gt_labels,
                     "detections": res.detections[cam]})
        rec.log_camera(idx, cam, frame.cameras[cam].image, c.gt_boxes_2d, c.gt_labels, c.det_boxes_2d,
                       [d.label for d in res.detections[cam]])
        rec.log_scalar(idx, "metrics/match_rate", check.match_rate)
        rec.log_scalar(idx, "metrics/is_fault", float(label.fault_id != "none"))
        rec.log_text(idx, "diagnosis", f"label={label.fault_id}\n{res.summary}\nVLM: {res.vlm.raw if res.vlm else '-'}")
        prev_frame, prev_report = frame, res.report
    jsonl.close()
    rec.close()
    wall = time.perf_counter() - t_run

    save_overlays(out, rows, adapter, scene, tokens, injector, cam, cfg["output"]["overlay_frames"])
    summary = analyze(out, rows, errors, len(tokens), wall, with_vlm=not args.no_vlm)
    print(json.dumps(summary, indent=2))
    print(f"outputs: {out}")


def save_overlays(out, rows, adapter, scene, tokens, injector, cam, n):
    """Overlay images for a few clean and a few C1 frames, spread over the scene."""
    jsonl = {r["frame_idx"]: r for r in map(json.loads, (out / "frames.jsonl").read_text().splitlines())}
    picks = []
    for fault in LABELS:
        idxs = [r["frame_idx"] for r in rows if r["fault_id"] == fault]
        picks += [idxs[i] for i in np.linspace(0, len(idxs) - 1, min(n // 2, len(idxs))).astype(int)] if idxs else []
    for idx in sorted(set(picks)):
        frame, label = injector.apply(adapter.load_frame(scene, idx, tokens[idx]))
        r = jsonl[idx]
        rate = "n/a" if r["match_rate"] is None else f"{r['match_rate']:.2f}"
        title = (f"frame {idx}  label={label.fault_id}  match={r['n_matched']}/{r['n_gt']} ({rate})  "
                 f"dets={r['n_det']}  VLM={r['status']} {r['confidence']}")
        img = draw_overlay(frame.cameras[cam].image, np.array(r["gt_boxes_2d"]),
                           np.array([d["box"] for d in r["detections"]]), title)
        img.save(out / f"overlay_{idx:03d}_{label.fault_id}.png")


def analyze(out: Path, rows: list[dict], errors: list[dict], n_frames: int, wall_s: float, with_vlm: bool) -> dict:
    is_fault = np.array([r["fault_id"] != "none" for r in rows])
    rates = np.array([np.nan if r["match_rate"] is None else r["match_rate"] for r in rows])
    has_ref = ~np.isnan(rates)
    clean, faulty = rates[has_ref & ~is_fault], rates[has_ref & is_fault]

    auroc = None
    if len(clean) and len(faulty):
        auroc = float(roc_auc_score(is_fault[has_ref], 1 - rates[has_ref]))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(clean, bins=bins, alpha=0.6, label=f"clean (n={len(clean)})", color="#3677A3")
    ax.hist(faulty, bins=bins, alpha=0.6, label=f"C1 (n={len(faulty)})", color="#D95D45")
    ax.set_xlabel("CAM_FRONT match rate (LiDAR-confirmed GT boxes matched by SAM 3, IoU >= 0.3)")
    ax.set_ylabel("frames")
    ax.set_title(f"E0 match rate, AUROC = {auroc:.3f}" if auroc is not None else "E0 match rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "match_rate_hist.png", dpi=150)
    plt.close(fig)

    summary = {
        "frames_total": n_frames, "frames_ok": len(rows), "frame_errors": errors, "wall_time_s": round(wall_s, 1),
        "n_clean": int((~is_fault).sum()), "n_c1": int(is_fault.sum()),
        "frames_without_reference": int((~has_ref).sum()),
        "match_rate_clean": describe(clean), "match_rate_c1": describe(faulty), "auroc_match_rate": auroc,
    }

    if with_vlm:
        cm = np.zeros((len(LABELS), len(STATUSES)), dtype=int)
        for r in rows:
            cm[LABELS.index(r["fault_id"]), STATUSES.index(r["status"])] += 1
        with (out / "confusion_matrix.csv").open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["label \\ status", *STATUSES])
            for lab, line in zip(LABELS, cm):
                w.writerow([lab, *line.tolist()])
        plot_confusion(cm, out / "confusion_matrix.png")
        c1_fault = [r for r in rows if r["fault_id"] == "C1" and r["status"] == "fault"]
        tp, fn = cm[1, 1], cm[1, 0] + cm[1, 2]
        fp_, tn = cm[0, 1], cm[0, 0] + cm[0, 2]
        summary["vlm"] = {
            "json_valid_rate": float(np.mean([r["vlm_valid"] for r in rows])) if rows else None,
            "confusion_matrix": {lab: dict(zip(STATUSES, line.tolist())) for lab, line in zip(LABELS, cm)},
            "fault_recall": float(tp / (tp + fn + cm[1, 3])) if is_fault.any() else None,
            "false_alarm_rate": float(fp_ / (fp_ + tn + cm[0, 3])) if (~is_fault).any() else None,
            "correct_sensor_rate_on_detected_c1": (
                float(np.mean([r["sensor"] == "camera:CAM_FRONT" for r in c1_fault])) if c1_fault else None),
        }

    keys = [k for k in LATENCY_KEYS if any(r.get(f"latency_{k}") is not None for r in rows)]
    with (out / "latency.csv").open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["frame_idx", *keys])
        for r in rows:
            w.writerow([r["frame_idx"], *[r[f"latency_{k}"] for k in keys]])
    summary["latency_s"] = {k: describe(np.array([r[f"latency_{k}"] for r in rows], float)) for k in keys}
    summary["latency_recorded_every_frame"] = all(r[f"latency_{k}"] is not None for r in rows for k in keys)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(out, summary)
    return summary


def describe(x: np.ndarray) -> dict | None:
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    return {"n": len(x), "mean": round(float(x.mean()), 4), "median": round(float(np.median(x)), 4),
            "p10": round(float(np.percentile(x, 10)), 4), "p90": round(float(np.percentile(x, 90)), 4),
            "min": round(float(x.min()), 4), "max": round(float(x.max()), 4)}


def plot_confusion(cm: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(STATUSES)), STATUSES)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Qwen3-VL status")
    ax.set_ylabel("injected fault")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_markdown(out: Path, s: dict):
    lines = [
        "# E0 pilot summary", "",
        (f"- Frames: {s['frames_ok']}/{s['frames_total']} processed, {len(s['frame_errors'])} errors; "
        f"{s['n_clean']} clean, {s['n_c1']} C1; {s['frames_without_reference']} frames had no LiDAR reference."),
        f"- Match rate clean: {s['match_rate_clean']}",
        f"- Match rate C1: {s['match_rate_c1']}",
        f"- AUROC (1 - match rate as C1 score): {s['auroc_match_rate']}",
    ]
    if "vlm" in s:
        v = s["vlm"]
        lines += [f"- Qwen JSON valid: {v['json_valid_rate']}",
                  (f"- Qwen fault recall: {v['fault_recall']}, false alarm rate: {v['false_alarm_rate']}, "
                  f"correct sensor on detected C1: {v['correct_sensor_rate_on_detected_c1']}"),
                  f"- Confusion matrix: {v['confusion_matrix']}"]
    lines += [f"- Latency (s): {s['latency_s']}", f"- Latency recorded every frame: {s['latency_recorded_every_frame']}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
