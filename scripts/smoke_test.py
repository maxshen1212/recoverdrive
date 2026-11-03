"""Phase 0 check: read one frame, call SAM 3 and Qwen once each, and log the result to Rerun.

Usage (after `source scripts/env.sh` and starting scripts/serve_vlm.sh):
    uv run python scripts/smoke_test.py
"""

import argparse
from pathlib import Path

from recoverdrive.build import build_adapter, build_pipeline, load_config
from recoverdrive.recording import RerunRecorder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/e0.yaml")
    p.add_argument("--out", default="outputs/smoke")
    args = p.parse_args()

    cfg = load_config(args.config)
    adapter = build_adapter(cfg)
    token = adapter.scene_sample_tokens(cfg["data"]["scene"])[0]
    frame = adapter.load_frame(cfg["data"]["scene"], 0, token)
    print(f"frame: {len(frame.lidar.points)} LiDAR points, {len(frame.gt_boxes)} GT boxes, "
          f"cameras {list(frame.cameras)}")

    pipeline = build_pipeline(cfg)
    result = pipeline.step(frame, None, None)
    cam = pipeline.cameras[0]
    print(f"SAM 3: {len(result.detections[cam])} detections")
    print(result.summary)
    print(f"Qwen raw output: {result.vlm.raw}  (valid={result.vlm.valid})")
    print("latency (s):", {k: round(v, 3) for k, v in result.latency_s.items()})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = RerunRecorder(out / "smoke.rrd")
    c = result.consistency[cam]
    rec.log_camera(0, cam, frame.cameras[cam].image, c.gt_boxes_2d, c.gt_labels, c.det_boxes_2d,
                   [d.label for d in result.detections[cam]])
    rec.close()
    print(f"Rerun recording: {out / 'smoke.rrd'}")
    assert result.vlm.valid, "Qwen output did not match the schema"


if __name__ == "__main__":
    main()
