"""Draw the RecoverDrive architecture diagram (docs/media/architecture.png). Run: uv run python scripts/make_architecture_figure.py"""

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
    "data": ("#eaf2ea", "#4f7a52"),  # data source or simulator
    "eval": ("#f3f3f3", "#7a7a7a"),  # evaluation only (dashed)
}
INK, MUTED, EDGE = "#1b2230", "#4d5664", "#5b6270"
W, H = 1400, 1430


def box(ax, x, y, w, h, kind, title, sub=None, dashed=False):
    fill, edge = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14", fc=fill, ec=edge,
                                lw=2.0, ls=(0, (6, 4)) if dashed else "-"))
    cy = y + h / 2
    if sub:
        two_lines = "\n" in sub
        ax.text(x + w / 2, cy - (27 if two_lines else 15), title, ha="center", va="center", fontsize=15,
                weight="bold", color=INK)
        ax.text(x + w / 2, cy + 20, sub, ha="center", va="center", fontsize=11.5,
                color=MUTED, linespacing=1.5)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=15, weight="bold", color=INK)


def arrow(ax, pts, label=None, label_xy=None, ha="left", rotation=0):
    """Polyline from pts[0] to pts[-1] with an arrowhead at the end."""
    for a, b in pairwise(pts[:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=EDGE, lw=1.8, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=18, color=EDGE, lw=1.8,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_xy, label, ha=ha, va="center", fontsize=11, color=MUTED, style="italic", rotation=rotation)


def draw(out: Path) -> None:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    L, R = 290, 1110  # main column
    mid = (L + R) / 2

    box(ax, 120, 30, 520, 90, "data", "nuScenes-mini", "offline evaluation (detection, fallback, latency, ablation)")
    box(ax, 760, 30, 520, 90, "data", "CARLA 0.9.16", "closed loop: does the car crash?")
    box(ax, L, 185, R - L, 90, "eval", "Fault injector (evaluation only)",
        "camera C1–C4  ·  LiDAR L1, L2, L4", dashed=True)
    box(ax, L, 340, 390, 100, "code", "LiDAR objects", "ground-truth boxes with ≥ 5 points\n(oracle detector)")
    box(ax, 720, 340, 390, 100, "code", "SAM 3 camera detection", "2D boxes on 6 cameras\n6 text prompts")
    box(ax, L, 505, R - L, 110, "code", "Consistency checks (deterministic)",
        "cross-sensor match rate  ·  LiDAR health  ·  LiDAR–camera agreement  ·  frozen image")
    box(ax, L, 680, R - L, 120, "vlm", "Diagnosis agent: Qwen3-VL-8B",
        "camera mosaic (now + previous)  ·  LiDAR top view  ·  check report\n"
        "→ JSON verdict: ok / uncertain / fault:<sensor>")
    box(ax, 1150, 650, 230, 80, "eval", "Agent faults", "A1–A3 at the VLM call", dashed=True)
    box(ax, L, 865, R - L, 110, "code", "Recovery controller",
        "re-ask ≤ 2  ·  3-sample vote  ·  re-observe  ·  isolate  →  verify")
    box(ax, L, 1040, R - L, 130, "code", "Safety shield (rules, no LLM)",
        "healthy → continue   ·   degraded, front camera or LiDAR usable → slow_down\n"
        "otherwise → stop   ·   time-to-collision < 2 s → brake (CARLA)")
    box(ax, L, 1235, R - L, 80, "code", "Action + logs", "continue / slow_down / stop  ·  JSONL + Rerun")

    arrow(ax, [(380, 120), (380, 150), (500, 150), (500, 185)])
    arrow(ax, [(1020, 120), (1020, 150), (900, 150), (900, 185)])
    arrow(ax, [(485, 275), (485, 340)], "points", (497, 308))
    arrow(ax, [(915, 275), (915, 340)], "images", (927, 308))
    arrow(ax, [(485, 440), (485, 505)], "3D boxes → projected", (497, 472))
    arrow(ax, [(915, 440), (915, 505)], "2D boxes", (927, 472))
    arrow(ax, [(mid, 615), (mid, 680)], "flags + check report", (mid + 12, 648))
    arrow(ax, [(1150, 690), (R, 690)])
    arrow(ax, [(mid, 800), (mid, 865)], "verdict", (mid + 12, 833))
    arrow(ax, [(mid, 975), (mid, 1040)], "state: healthy / degraded / unresolved", (mid + 12, 1008))
    arrow(ax, [(mid, 1170), (mid, 1235)])
    # recovery loops
    arrow(ax, [(R, 895), (1135, 895), (1135, 770), (R, 770)], "re-ask, 3 samples", (1147, 832))
    arrow(ax, [(L, 945), (235, 945), (235, 560), (L, 560)], "verify: re-check the remaining sensors",
          (212, 752), ha="center", rotation=90)

    # legend
    lx, ly = L, 1355
    items = [("code", "deterministic Python", False), ("vlm", "Qwen3-VL call", False),
             ("data", "data source / simulator", False), ("eval", "evaluation only", True)]
    for kind, text, dashed in items:
        fill, edge = STYLE[kind]
        ax.add_patch(FancyBboxPatch((lx, ly), 34, 22, boxstyle="round,pad=0,rounding_size=5", fc=fill, ec=edge,
                                    lw=1.8, ls=(0, (4, 3)) if dashed else "-"))
        ax.text(lx + 44, ly + 11, text, va="center", fontsize=11.5, color=MUTED)
        lx += 64 + 9 * len(text)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("docs/media/architecture.png"))
    draw(p.parse_args().out)
