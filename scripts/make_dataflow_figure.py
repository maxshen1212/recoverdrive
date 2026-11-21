"""Draw the system data-flow block diagram (docs/media/dataflow.png): what goes in and out of every module, the
diagnosis-agent variants, and the verification loop (STL requirements, falsification, proofs).

Run: uv run python scripts/make_dataflow_figure.py [--out docs/media/dataflow.png]
Same visual style as scripts/make_architecture_figure.py.
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

STYLE = {  # (fill, edge)
    "code": ("#e8f0fb", "#2f6db5"),  # deterministic Python
    "vlm": ("#fff2dd", "#c27400"),  # Qwen3-VL call
    "data": ("#eaf2ea", "#4f7a52"),  # data source, simulator or output
    "eval": ("#f3f3f3", "#7a7a7a"),  # evaluation and verification only (dashed)
}
INK, MUTED, EDGE = "#1b2230", "#4d5664", "#5b6270"
W, H = 1700, 1290


def box(ax, x, y, w, h, kind, title, sub=None, dashed=False):
    fill, edge = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14", fc=fill, ec=edge,
                                lw=2.0, ls=(0, (6, 4)) if dashed else "-"))
    if sub:
        n = sub.count("\n") + 1
        ax.text(x + w / 2, y + 26, title, ha="center", va="center", fontsize=14, weight="bold", color=INK)
        ax.text(x + w / 2, y + 26 + 22 + 10.5 * n, sub, ha="center", va="center", fontsize=10.5, color=MUTED,
                linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=14, weight="bold", color=INK)


def arrow(ax, pts, label=None, label_xy=None, ha="left", dashed=False):
    """Polyline from pts[0] to pts[-1] with an arrowhead at the end."""
    ls = (0, (5, 4)) if dashed else "-"
    for a, b in pairwise(pts[:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=EDGE, lw=1.8, ls=ls, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=18, color=EDGE, lw=1.8,
                                 linestyle=ls, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_xy, label, ha=ha, va="center", fontsize=10.5, color=MUTED, style="italic")


def draw(out: Path) -> None:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    L, R = 60, 1150  # main column
    V0, VW = 1235, 420  # verification column

    box(ax, L, 30, 520, 100, "data", "nuScenes-mini (offline, 2 Hz)",
        "6 cameras 1600×900 · 32-beam LiDAR, ~34k points\nannotated 3D boxes")
    box(ax, 630, 30, 520, 100, "data", "CARLA 0.9.16 (closed loop)",
        "2 cameras · ray-cast LiDAR · perception every 0.5 s\npedestrian and lead-car-braking hazards")
    box(ax, L, 170, R - L, 84, "code", "Source adapter → Frame",
        "per sensor: image or point cloud, calibration, ego pose at the sensor's own timestamp")
    box(ax, L, 294, R - L, 104, "eval", "Fault injector (evaluation and falsification only)",
        "C1–C4 change camera images · L1, L2, L4 change the point cloud · A1–A3 change the VLM's answers\n"
        "out: faulted Frame + ground-truth FaultLabel (fault, sensor, severity)", dashed=True)
    box(ax, L, 440, 340, 112, "code", "SAM 3 camera detector", "in: each camera image\nout: 2D boxes, 6 classes, score ≥ 0.5")
    box(ax, 430, 440, 340, 112, "code", "LiDAR objects", "in: point cloud + 3D boxes\nout: objects with ≥ 5 points ≤ 50 m")
    box(ax, 800, 440, 350, 112, "code", "Render", "out: camera mosaic (now and 0.5 s ago)\nLiDAR top view (±50 m)")
    box(ax, L, 594, 710, 104, "code", "Consistency checks (deterministic)",
        "project LiDAR objects into each camera, match SAM 3 boxes (IoU ≥ 0.3),\n"
        "point counts per sector, frozen images → CheckReport: match rates, flags")
    box(ax, L, 742, R - L, 160, "vlm", "Diagnosis agent: Qwen3-VL-8B on vLLM (JSON-schema output)",
        "in: mosaic, previous mosaic, top view, check report · out: verdict (ok / uncertain / fault:<sensor>), "
        "fault type, confidence\n"
        "Ours (frozen): 1 call, checks are reliable · think: reasons first, checks may be wrong (_trust: reliable)\n"
        "perceive: notes on every sensor, then decides (_trust: frozen guidance) · per-sensor: 6 camera + 1 LiDAR notes\n"
        "image flag: 1 call marks abnormal camera images → one more CheckReport flag · _nochecks: no check report")
    box(ax, L, 944, R - L, 92, "code", "Recovery controller",
        "re-ask a bad answer · 3-sample vote (T = 0.7) · re-observe · isolate the sensor · verify\n"
        "out: RecoveryResult (state, isolated sensors)")
    box(ax, L, 1078, 530, 100, "code", "Safety shield (rules, no LLM)",
        "in: RecoveryResult and severity\nout: continue < slow_down < stop")
    box(ax, 620, 1078, 530, 100, "data", "Vehicle and logs",
        "action to the controller (CARLA)\nJSONL + Rerun logs → metrics E1–E4")

    box(ax, V0, 294, VW, 118, "eval", "Falsification (Optuna)",
        "random vs. guided (TPE) search over\nfault type, level, timing and scenario\nminimizes STL robustness", dashed=True)
    box(ax, V0, 594, VW, 132, "eval", "STL requirements R1–R6",
        "robustness on the logged traces\nR1 missed and R2 unnecessary fallback\nR3 recovery · R6 no early restore\n"
        "R4/R5 closed-loop slow-down, braking", dashed=True)
    box(ax, V0, 944, VW, 108, "eval", "Formal proofs",
        "shield: exhaustive over all inputs\nrecovery: explicit-state model checking", dashed=True)

    mid = (L + R) / 2
    arrow(ax, [(310, 130), (310, 170)])
    arrow(ax, [(890, 130), (890, 170)])
    arrow(ax, [(mid, 254), (mid, 294)], "Frame", (mid + 10, 274))
    arrow(ax, [(230, 398), (230, 440)])
    arrow(ax, [(600, 398), (600, 440)])
    arrow(ax, [(975, 398), (975, 440)])
    arrow(ax, [(230, 552), (230, 594)], "2D boxes", (240, 573))
    arrow(ax, [(600, 552), (600, 594)], "3D objects", (610, 573))
    arrow(ax, [(415, 698), (415, 742)], "CheckReport", (425, 720))
    arrow(ax, [(975, 552), (975, 742)], "images", (985, 650))
    arrow(ax, [(mid, 902), (mid, 944)], "Diagnosis", (mid + 10, 923))
    arrow(ax, [(325, 1036), (325, 1078)], "RecoveryResult", (335, 1057))
    arrow(ax, [(590, 1128), (620, 1128)])
    arrow(ax, [(R, 1160), (1680, 1160), (1680, 660), (V0 + VW, 660)], "logged traces", (1400, 1176))
    arrow(ax, [(V0 + VW / 2, 594), (V0 + VW / 2, 412)], "robustness", (V0 + VW / 2 + 10, 503))
    arrow(ax, [(V0, 346), (R, 346)], "fault +\nscenario", (R + 8, 378), dashed=True)
    arrow(ax, [(V0, 990), (R, 990)], "proofs", (R + 8, 1010), dashed=True)

    ly = 1222
    for i, (kind, text) in enumerate([("code", "deterministic Python"), ("vlm", "Qwen3-VL call"),
                                      ("data", "data, simulator, output"), ("eval", "evaluation and verification")]):
        x = L + i * 300
        fill, edge = STYLE[kind]
        ax.add_patch(FancyBboxPatch((x, ly), 34, 24, boxstyle="round,pad=0,rounding_size=5", fc=fill, ec=edge, lw=2,
                                    ls=(0, (4, 3)) if kind == "eval" else "-"))
        ax.text(x + 46, ly + 12, text, va="center", fontsize=11.5, color=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/media/dataflow.png"))
    draw(p.parse_args().out)
