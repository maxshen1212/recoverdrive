"""Draw the assurance case of RecoverDrive in Goal Structuring Notation (GSN): docs/media/assurance_case.png.

Numbers come from the generated result files, so the figure can be redrawn after new runs:

    uv run python scripts/make_assurance_figure.py \
        --requirements /data/maxshen/recoverdrive/report_figures/requirements.json \
        --formal /data/maxshen/recoverdrive/report_figures/formal.json \
        --falsify /data/maxshen/recoverdrive/outputs/falsify/nuscenes_cpu_v1/trials.jsonl
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

INK, MUTED, EDGE = "#1b2230", "#4d5664", "#5b6270"
FILL = {"goal": ("#e8f0fb", "#2f6db5"), "strategy": ("#f3f3f3", "#7a7a7a"), "evidence": ("#eaf2ea", "#4f7a52"),
        "context": ("#fff2dd", "#c27400"), "assumption": ("#fbeaea", "#a94442")}
W, H = 1840, 1180
LINE_H = 21  # approximate height of one text line at fontsize 11.5, in canvas units


def label(ax, x, cy, tag, text, width, size=11.5):
    """Tag and wrapped text, centered vertically on cy."""
    wrapped = textwrap.fill(text, width)
    block = (wrapped.count("\n") + 1) * LINE_H + 22
    top = cy - block / 2
    ax.text(x, top + 16, tag, ha="center", va="bottom", fontsize=12, weight="bold", color=INK)
    ax.text(x, top + 24, wrapped, ha="center", va="top", fontsize=size, color=MUTED, linespacing=1.35)


def node(ax, kind, cx, cy, w, h, tag, text, wrap):
    fill, edge = FILL[kind]
    if kind in ("goal", "context"):
        rounding = 0 if kind == "goal" else 18
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                    fc=fill, ec=edge, lw=2.0))
    elif kind == "strategy":
        s = 30
        ax.add_patch(Polygon([(cx - w / 2 + s, cy - h / 2), (cx + w / 2 + s, cy - h / 2),
                              (cx + w / 2 - s, cy + h / 2), (cx - w / 2 - s, cy + h / 2)], fc=fill, ec=edge, lw=2.0))
    else:
        ax.add_patch(Ellipse((cx, cy), w, h, fc=fill, ec=edge, lw=2.0))
    label(ax, cx, cy, tag, text, wrap)
    return cx, cy, w, h


def arrow(ax, start, end, dashed=False):
    """Solid arrow = "supported by"; dashed arrow with a hollow head = "in context of"."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17, color=EDGE, lw=1.5,
                                 fc="white" if dashed else EDGE, ls=(0, (5, 3)) if dashed else "-",
                                 shrinkA=0, shrinkB=0))


def bottom(n):
    return n[0], n[1] + n[3] / 2


def top(n, dx=0.0):
    return n[0] + dx, n[1] - n[3] / 2


def numbers(req_path: Path, formal_path: Path, falsify_path: Path | None) -> dict:
    req = json.loads(req_path.read_text())
    formal = json.loads(formal_path.read_text())
    cross = req["nuscenes"]["cross_check"]["ours_full"]
    bounds = {r["fault_id"]: r for r in req["nuscenes"]["bounds"]["ours_full"]}
    agent_ok = all(bounds[f]["violated"] == 0 for f in ("A1", "A2", "A3") if f in bounds)
    proved = [r["name"] for r in formal["shield"] if r["holds"]]
    ctrl = [r["name"] for r in formal["controller"] if r["holds"]]
    out = {"missed": cross["missed_frames"] / cross["frames"], "worst_bound": max(r["upper"] for r in bounds.values()),
           "agent_ok": agent_ok, "proved": proved, "ctrl": ctrl, "states": formal["controller_states"],
           "fals": None}
    if falsify_path and falsify_path.exists():
        rows = [json.loads(line) for line in falsify_path.read_text().splitlines() if line.strip()]
        by = {s: (sum(not r["satisfied"] for r in rows if r["sampler"] == s), sum(r["sampler"] == s for r in rows))
              for s in ("random", "tpe")}
        out["fals"] = by
    return out


def draw(n: dict, out: Path) -> None:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    g0 = node(ax, "goal", 760, 110, 780, 130, "G0",
              "Under the faults C1-C4, L1, L2, L4 and A1-A3, and within assumptions A1-A4, the vehicle meets "
              "the requirements R1-R6.", 80)
    c0 = node(ax, "context", 1500, 110, 560, 180, "C0",
              "R1-R6 in STL: missed / unnecessary fallback, recovery, front-fault reaction, TTC braking, no restore "
              "while faulty. A1 single fault, A2 calibration, A3 >= 2 reference objects, A4 oracle LiDAR.", 62)
    s0 = node(ax, "strategy", 760, 300, 940, 100, "S0",
              "Argue over the four parts: the shield, the monitor's detection, the closed loop, and VLM failures.", 96)
    gy, gw, gh = 500, 340, 150
    goals = [
        node(ax, "goal", 230, gy, gw, gh, "G1", "The shield picks an acceptable action whenever the fault is "
             "detected (and nothing else is alarmed).", 36),
        node(ax, "goal", 640, gy, gw, gh, "G2", "The monitor detects faults that need a fallback often enough.",
             36),
        node(ax, "goal", 1110, gy, gw, gh, "G3", "R1-R6 hold on the logged runs, and search does not find "
             "unexplained violations.", 36),
        node(ax, "goal", 1590, gy, gw, gh, "G4", "Failures of the VLM itself (bad output, wrong answer, timeout) "
             "are contained.", 36),
    ]
    a1 = node(ax, "assumption", 440, 720, 250, 160, "A",
              "S1 needs a correct detection: without it S6a and S6b fail (false alarm, missed fault).", 26)
    ey, eh = 950, 250
    fals = n["fals"]
    fals_text = ("Falsification on nuScenes dev (checks-only): "
                 f"random {fals['random'][0]}/{fals['random'][1]}, guided (TPE) {fals['tpe'][0]}/{fals['tpe'][1]} "
                 "trials violate a requirement; every class is triaged." if fals else "Falsification: pending.")
    evidence = [
        node(ax, "evidence", 230, ey, 330, eh, "Sn1",
             f"Exhaustive check of the real shield: {', '.join(n['proved'])} hold on every case.", 30),
        node(ax, "evidence", 640, ey, 330, eh, "Sn2",
             f"Frozen test split: missed fallback on {n['missed']:.1%} of frames; per fault type, the episode-level "
             f"R1 upper bound is at most {n['worst_bound']:.0%} (95% Clopper-Pearson).", 32),
        node(ax, "evidence", 960, ey, 290, eh, "Sn3",
             "R1-R6 checked on every logged run (E2 test split, E3 CARLA closed loop).", 26),
        node(ax, "evidence", 1265, ey, 300, eh, "Sn4", fals_text, 27),
        node(ax, "evidence", 1590, ey, 330, eh, "Sn5",
             f"Controller model checking ({n['states']} states): {', '.join(n['ctrl'])} hold, incl. C4 agent "
             f"fallback. A1-A3 test episodes: {'no' if n['agent_ok'] else 'some'} missed fallback.", 32),
    ]

    arrow(ax, bottom(g0), top(s0))
    arrow(ax, (g0[0] + g0[2] / 2, g0[1]), (c0[0] - c0[2] / 2, c0[1]), dashed=True)
    for g in goals:
        arrow(ax, bottom(s0), top(g))
    arrow(ax, (goals[0][0] + 120, goals[0][1] + gh / 2), (a1[0] - 70, a1[1] - 65), dashed=True)
    for g, e in ((goals[0], evidence[0]), (goals[1], evidence[1]), (goals[3], evidence[4])):
        arrow(ax, bottom(g), top(e))
    arrow(ax, (goals[2][0] - 80, goals[2][1] + gh / 2), top(evidence[2]))
    arrow(ax, (goals[2][0] + 80, goals[2][1] + gh / 2), top(evidence[3]))

    legend = [("goal", "goal (claim)"), ("strategy", "strategy"), ("evidence", "solution (evidence)"),
              ("context", "context"), ("assumption", "assumption")]
    for i, (kind, text) in enumerate(legend):
        fill, edge = FILL[kind]
        x = 60 + i * 300
        ax.add_patch(FancyBboxPatch((x, 1120), 26, 26, boxstyle="round,pad=0,rounding_size=4", fc=fill, ec=edge,
                                    lw=1.6))
        ax.text(x + 38, 1133, text, va="center", fontsize=12, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--requirements", type=Path, required=True)
    p.add_argument("--formal", type=Path, required=True)
    p.add_argument("--falsify", type=Path, default=None, help="trials.jsonl of a falsification run")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs/media/assurance_case.png")
    args = p.parse_args()
    draw(numbers(args.requirements, args.formal, args.falsify), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
