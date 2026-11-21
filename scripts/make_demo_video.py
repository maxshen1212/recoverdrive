"""Side-by-side demo video (MP4) from two runs recorded with `run_e3.py --record`: a baseline on the left and the
monitor on the right, played in simulation time, with a title card and an end card. Needs ffmpeg on the PATH.

Usage:
    python scripts/make_demo_video.py outputs/e3/demo/r1_pedestrian_C1_B0_s0 \
        outputs/e3/demo/r1_pedestrian_C1_ours_full_s0 report_figures/demo_c1_pedestrian.mp4
(each run argument is the run path without suffix: <run>.json and <run>_frames/ must exist)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

BG, INK, MUTED, RED, GREEN = (24, 26, 31), (240, 240, 240), (170, 174, 182), (226, 73, 72), (27, 175, 122)
LABELS = {"B0": "B0: no monitor", "ours_full": "Ours-full: monitor + recovery + shield",
          "checks_only": "Checks-only monitor"}


def load(run: Path) -> tuple[dict, list[dict], Path]:
    frames_dir = Path(f"{run}_frames")
    return json.loads(Path(f"{run}.json").read_text()), json.loads((frames_dir / "frames.json").read_text()), frames_dir


def card(size: tuple[int, int], lines: list[tuple[str, int, tuple]]) -> Image.Image:
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    total = sum(s + 18 for _, s, _ in lines)
    y = (size[1] - total) // 2
    for text, s, color in lines:
        w = d.textlength(text, font_size=s)
        d.text(((size[0] - w) / 2, y), text, fill=color, font_size=s)
        y += s + 18
    return img


def side(frames_dir: Path, log: list[dict], i: int, result: dict, w: int, h: int) -> Image.Image:
    """Frame i of one run; after its last frame, the last frame dimmed with the outcome."""
    last = i >= len(log)
    f = log[min(i, len(log) - 1)]
    img = Image.open(frames_dir / f"{log.index(f):04d}.jpg").convert("RGB").resize((w, h))
    if last:
        img = Image.blend(img, Image.new("RGB", (w, h), (0, 0, 0)), 0.45)
        d = ImageDraw.Draw(img)
        text = "COLLISION" if result.get("collision") else "route completed"
        color = RED if result.get("collision") else GREEN
        tw = d.textlength(text, font_size=44)
        d.text(((w - tw) / 2, h / 2 - 26), text, fill=color, font_size=44)
    return img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--width", type=int, default=800, help="width of each side")
    p.add_argument("--fps", type=int, default=10, help="simulation frames per second of video (10 = real time)")
    args = p.parse_args()
    (lres, llog, ldir), (rres, rlog, rdir) = load(args.left), load(args.right)
    w = args.width
    h = round(w * 9 / 16)
    head = 56
    size = (2 * w + 12, h + head)
    n = max(len(llog), len(rlog)) + 2 * args.fps  # hold both outcomes for 2 s at the end
    fault = "front camera shows the rear view (C1)" if lres.get("condition") == "C1" else lres.get("condition")
    title = card(size, [("RecoverDrive: closed-loop demo in CARLA 0.9.16", 34, INK),
                        (f"Fault: {fault}, starting 25 m before a {lres.get('hazard', '')} hazard", 22, MUTED),
                        ((f"Left: {LABELS.get(lres['method'], lres['method'])}    "
                         f"Right: {LABELS.get(rres['method'], rres['method'])}"), 22, MUTED)])

    def outcome(r: dict) -> str:
        if r.get("collision"):
            return f"collision at t = {r.get('sim_s')} s"
        return f"no collision, route completion {r.get('route_completion', 0):.0%}"

    end = card(size, [("Result (same route, same seed)", 30, INK),
                      (f"{LABELS.get(lres['method'], lres['method'])}: {outcome(lres)}", 24,
                       RED if lres.get("collision") else GREEN),
                      (f"{LABELS.get(rres['method'], rres['method'])}: {outcome(rres)}", 24,
                       RED if rres.get("collision") else GREEN),
                      ("The monitor detects the faulty front camera from the camera-LiDAR cross-check, isolates it,", 19,
                       MUTED),
                      ("and the rule-based shield slows the vehicle down before the hazard.", 19, MUTED)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
           "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-r", "30",
           str(args.out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for _ in range(3 * args.fps):
        proc.stdin.write(title.tobytes())
    for i in range(n):
        canvas = Image.new("RGB", size, BG)
        d = ImageDraw.Draw(canvas)
        t = rlog[min(i, len(rlog) - 1)]["t"] if i < len(rlog) else rlog[-1]["t"]
        for k, (res, log, fdir) in enumerate(((lres, llog, ldir), (rres, rlog, rdir))):
            x = k * (w + 12)
            canvas.paste(side(fdir, log, i, res, w, h), (x, head))
            d.text((x + 12, 14), LABELS.get(res["method"], res["method"]), fill=INK, font_size=24)
        d.text((size[0] - 150, 16), f"t = {t:4.1f} s", fill=MUTED, font_size=22)
        proc.stdin.write(canvas.tobytes())
    for _ in range(5 * args.fps):
        proc.stdin.write(end.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {args.out} ({(3 * args.fps + n + 5 * args.fps) / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
