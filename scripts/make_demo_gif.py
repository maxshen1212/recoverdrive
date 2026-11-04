"""Turn the frames saved by `run_e3.py --record` into a GIF for the README.

Usage: uv run python scripts/make_demo_gif.py outputs/e3/demo/<run>_frames docs/media/<name>.gif \
           [--start 3.0] [--end 30.0] [--speed 2.0] [--width 480] [--fps 10]
"""

import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--start", type=float, default=0.0, help="first second of simulation time to keep")
    p.add_argument("--end", type=float, default=None, help="last second of simulation time to keep")
    p.add_argument("--speed", type=float, default=2.0, help="playback speed relative to real time")
    p.add_argument("--fps", type=float, default=10.0, help="GIF frame rate")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--colors", type=int, default=128)
    args = p.parse_args()
    log = json.loads((args.frames_dir / "frames.json").read_text())
    keep = [f for f in log if f["t"] >= args.start and (args.end is None or f["t"] <= args.end)]
    sim_dt = keep[1]["t"] - keep[0]["t"]
    step = max(1, round(args.speed / (args.fps * sim_dt)))  # e.g. 10 Hz sim, 2x speed, 10 fps -> every 2nd frame
    frames = []
    for f in keep[::step]:
        im = Image.open(args.frames_dir / f"{f['tick']:04d}.jpg").convert("RGB")
        im = im.resize((args.width, round(im.height * args.width / im.width)), Image.LANCZOS)
        frames.append(im.quantize(colors=args.colors, dither=Image.Dither.NONE))  # no dithering: much smaller GIFs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=round(1000 / args.fps), loop=0,
                   optimize=True)
    print(f"{args.out}: {len(frames)} frames, {args.out.stat().st_size / 1e6:.1f} MB, "
          f"{len(frames) / args.fps:.1f} s at {args.speed}x speed")


if __name__ == "__main__":
    main()
