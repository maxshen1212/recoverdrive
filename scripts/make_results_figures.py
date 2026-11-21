"""Figures and tables of the 2026-11-21 run (plan 2026-11-21-finish-and-vlm-variants): falsification on nuScenes and
in CARLA, the replays of the counterexamples with the VLM, and the VLM-centric variants on the dev split.

Reads finished runs under outputs/ and writes report_figures/ (not tracked): fig6-fig9 PNGs, results_2026-11-21.md
and results_2026-11-21.json. Colors follow make_figures.py (dataviz reference palette, light surface); every mark has
a value label, so identity never relies on color alone.

Usage, from the checkout that holds outputs/:
    python scripts/make_results_figures.py --screen vlm_screen_s0 [--seeds vlm_best_s12] [--confirm-variant TAG ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from recoverdrive.bootstrap import bootstrap_metrics
from recoverdrive.metrics import aggregate_seeds, e1_table, e2_table, fmt
from recoverdrive.runner import RunSpec, load_records

REPO = Path(__file__).resolve().parents[1]
OUT = Path("report_figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NAMES = {"B0": "B0 (no monitor)", "checks_only": "Checks-only", "ours_full": "Ours-full (frozen)",
         "think_full": "Think", "think_nochecks_full": "Think, no checks",
         "perceive_full": "Perceive → decide", "perceive_nochecks_full": "Perceive → decide, no checks",
         "percam_full": "Per-sensor → decide", "think_trust_full": "Think, trusting the checks",
         "perceive_trust_full": "Perceive → decide, trusting the checks", "imgflag_full": "VLM image flag → Ours",
         "imgflag_checks_full": "VLM image flag → checks-only"}
REFERENCE = ("B0", "checks_only", "ours_full")
VARIANTS = ("think_trust_full", "perceive_trust_full", "imgflag_full", "imgflag_checks_full", "think_full",
            "think_nochecks_full", "perceive_full", "perceive_nochecks_full", "percam_full")
SENSOR_FAULTS = ["C1", "C2", "C3", "C4", "L1", "L2", "L4"]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


# --- evaluation runs -------------------------------------------------------------------------------------------

def merge_tags(recs: list[dict], extra: list[dict]) -> list[dict]:
    """Add the method outputs of another run over the same frames (same split, scenes, faults and seed)."""
    by = {(r["scene"], r["run_fault"], r["seed"], r["frame_idx"]): r for r in extra}
    out = []
    for r in recs:
        o = by.get((r["scene"], r["run_fault"], r["seed"], r["frame_idx"]))
        if o is not None:
            assert o["label"] == r["label"], "frames differ between the merged runs"
            r = {**r, "methods": {**r["methods"], **o["methods"]}}
        out.append(r)
    return out


def union_tags(tags: list[str]) -> tuple[list[dict], list[str], list[int]]:
    """Records of several runs: frames with the same (scene, fault, seed, frame) get their method outputs merged."""
    by: dict[tuple, dict] = {}
    faults, seeds = [], []
    for tag in tags:
        recs, a = load_eval(tag)
        faults += [f for f in a["faults"] if f not in faults]
        seeds += [s for s in a["seeds"] if s not in seeds]
        for r in recs:
            k = (r["scene"], r["run_fault"], r["seed"], r["frame_idx"])
            if k in by:
                assert by[k]["label"] == r["label"], "frames differ between the merged runs"
                by[k] = {**by[k], "methods": {**by[k]["methods"], **r["methods"]}}
            else:
                by[k] = r
    return list(by.values()), faults, sorted(seeds)


def load_eval(tag: str) -> tuple[list[dict], dict]:
    meta = json.loads((Path("outputs/eval") / tag / "meta.json").read_text())
    a = meta["args"]
    splits = yaml.safe_load((REPO / "configs" / "splits.yaml").read_text())
    scenes = a.get("scenes") or splits[a["split"]]
    specs = [RunSpec(a["split"], sc, f, s) for s in a["seeds"] for f in a["faults"] for sc in scenes]
    recs = [r for r in load_records(Path("outputs/eval") / tag, specs) if "error" not in r]
    return recs, a


def method_rows(recs: list[dict], methods: list[str], faults: list[str], seeds: list[int]) -> dict[str, dict]:
    """Pooled (fault 'all') E1 + E2 + cost per method; mean and std over seeds when there are several."""
    e1 = aggregate_seeds([e1_table([r for r in recs if r["seed"] == s], methods, faults) for s in seeds]) \
        if len(seeds) > 1 else e1_table(recs, methods, faults)
    e2 = aggregate_seeds([e2_table([r for r in recs if r["seed"] == s], methods, faults) for s in seeds]) \
        if len(seeds) > 1 else e2_table(recs, methods, faults)
    out = {}
    for m in methods:
        r1 = next(r for r in e1 if r["method"] == m and r["fault"] == "all")
        r2 = next(r for r in e2 if r["method"] == m and r["fault"] == "all")
        outs = [r["methods"][m] for r in recs if m in r.get("methods", {})]
        calls = float(np.mean([o.get("n_calls", 0) for o in outs])) if outs else 0.0
        out[m] = {**{k: v for k, v in r1.items() if k not in ("method", "fault")},
                  **{k: v for k, v in r2.items() if k not in ("method", "fault", "n")},
                  "calls_per_frame": calls, "frames": len(outs)}
    return out


def per_fault_f1(recs: list[dict], methods: list[str], faults: list[str]) -> dict[str, dict[str, float]]:
    e1 = e1_table(recs, methods, faults)
    return {m: {r["fault"]: r["F1"] for r in e1 if r["method"] == m} for m in methods}


# --- falsification ---------------------------------------------------------------------------------------------

def nuscenes_falsification() -> list[dict]:
    rows = []
    for tag in ("cpu_v1", "gpu_v1"):
        trials = read_jsonl(Path("outputs/falsify") / f"nuscenes_{tag}" / "trials.jsonl")
        for f in sorted({t["fault"] for t in trials}):
            for s in ("random", "tpe"):
                ts = [t for t in trials if t["fault"] == f and t["sampler"] == s]
                rows.append({"fault": f, "sampler": s, "trials": len(ts), "violations": sum(not t["satisfied"] for t in ts)})
    return rows


def carla_falsification() -> list[dict]:
    rows = []
    for hazard in ("pedestrian", "lead_brake"):
        for s in ("random", "tpe"):
            ts = read_jsonl(Path("outputs/falsify") / f"carla_v1_{hazard}_{s}" / "trials.jsonl")
            rows.append({"hazard": hazard, "sampler": s, "runs": len(ts),
                         "collisions": sum(bool(t["collision"]) for t in ts),
                         "R5_violations": sum(t["R5"] is not None and t["R5"] < 0 for t in ts),
                         "R4_violations": sum(t["R4"] is not None and t["R4"] < 0 for t in ts),
                         "min_R5": min((t["R5"] for t in ts if t["R5"] is not None), default=None)})
    return rows


def replays(tags: list[str]) -> list[dict]:
    """Counterexamples of the checks replayed with a VLM method: how many are still violated."""
    out = []
    for tag in tags:
        rows = read_jsonl(Path("outputs/falsify") / f"nuscenes_{tag}" / "confirm.jsonl")
        if rows:
            out.append({"tag": tag, "method": rows[0]["method"], "replayed": len(rows),
                        "still_violated": sum(not r["satisfied"] for r in rows)})
    return out


# --- figures -----------------------------------------------------------------------------------------------------

def label_hbars(ax, bars, values, fmtstr="{:.2f}", errs=None):
    errs = errs or [0.0] * len(values)
    span = ax.get_xlim()[1]
    for b, v, e in zip(bars, values, errs):
        ax.text(b.get_width() + e + 0.015 * span, b.get_y() + b.get_height() / 2, fmtstr.format(v), va="center",
                ha="left", fontsize=8.5, color=INK)


def fig6_falsification(nus: list[dict], carla: list[dict]) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [7, 4]})
    faults = sorted({r["fault"] for r in nus}, key=lambda f: (f[0] != "C", f))
    x = np.arange(len(faults))
    w = 0.38
    for i, s in enumerate(("random", "tpe")):
        vals = [next(r["violations"] for r in nus if r["fault"] == f and r["sampler"] == s) for f in faults]
        bars = a.bar(x + (i - 0.5) * w, vals, w - 0.04, color=SLOTS[i], label="Random" if s == "random" else "Guided (TPE)")
        for bb, v in zip(bars, vals):
            a.text(bb.get_x() + bb.get_width() / 2, v + 2, str(v), ha="center", va="bottom", fontsize=8.5)
    a.set_ylim(0, max(r["violations"] for r in nus) * 1.35)
    a.set_xticks(x, faults)
    a.set_ylabel("violating trials (of 240 per fault)")
    a.set_title("nuScenes: violations of R1/R2/R3/R6 found, same budget", fontsize=10.5, loc="left")
    a.legend(frameon=False, loc="upper left", ncol=2)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    labels = [f"{r['hazard'].replace('_', ' ')}\n{r['sampler']}" for r in carla]
    y = np.arange(len(carla))
    hb = b.barh(y, [r["R4_violations"] for r in carla], 0.55,
                color=[SLOTS[0] if r["sampler"] == "random" else SLOTS[1] for r in carla])
    b.set_xlim(0, max([r["R4_violations"] for r in carla] + [1]) * 2.6)
    for bb, r in zip(hb, carla):
        b.text(bb.get_width() + 0.15, bb.get_y() + bb.get_height() / 2,
               f"R4 {r['R4_violations']} · R5 {r['R5_violations']} · collisions {r['collisions']} ({r['runs']} runs)",
               va="center", fontsize=8.5)
    b.set_yticks(y, labels)
    b.invert_yaxis()
    b.set_xlabel("R4 violations (no slow-down within 3 s of a front fault)")
    b.set_title("CARLA closed loop (checks-only)", fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_falsification.png", dpi=160)
    plt.close(fig)


def fig7_replays(tags: list[str]) -> None:
    """Share of the checks' worst counterexamples a VLM method still violates, per requirement."""
    data: dict[str, dict[str, list[int]]] = {}
    for tag in tags:
        for r in read_jsonl(Path("outputs/falsify") / f"nuscenes_{tag}" / "confirm.jsonl"):
            acc = data.setdefault(r["method"], {}).setdefault(r["spec"], [0, 0])
            acc[0] += 1
            acc[1] += not r["satisfied"]
    methods = list(data)
    specs = sorted({s for d in data.values() for s in d})
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(specs))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        vals = [data[m][s][1] / data[m][s][0] for s in specs]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w - 0.03, color=SLOTS[i], label=NAMES.get(m, m))
        for b, s in zip(bars, specs):
            n, v = data[m][s]
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v}/{n}", ha="center", va="bottom",
                    fontsize=8)
    ax.set_xticks(x, [{"R1": "R1 missed", "R2": "R2 unnecessary", "R3": "R3 recovery", "R6": "R6 early restore"}
                      .get(s, s) for s in specs])
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("share still violated")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.02))
    ax.set_title("The checks' worst counterexamples (214) replayed with VLM methods", fontsize=10.5, loc="left", pad=24)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_replays.png", dpi=160)
    plt.close(fig)


def fig8_variants(rows: dict[str, dict], title: str, name: str) -> None:
    methods = [m for m in (*REFERENCE, *VARIANTS) if m in rows]
    metrics = [("F1", "F1 (higher is better)"), ("sensor_acc", "right sensor named"), ("false_alarm", "false alarm rate"),
               ("unnecessary", "unnecessary fallback"), ("missed", "missed fallback")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 0.9 + 0.42 * len(methods)), sharey=True)
    y = np.arange(len(methods))
    for ax, (key, label) in zip(axes, metrics):
        vals = [rows[m].get(key) or 0.0 for m in methods]
        errs = [rows[m].get(f"{key}_std") or 0.0 for m in methods]
        colors = [SLOTS[0] if m in REFERENCE else SLOTS[1] for m in methods]
        bars = ax.barh(y, vals, 0.6, color=colors, xerr=errs if any(errs) else None,
                       error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
        ax.set_xlim(0, max(max(vals) * 1.45, 0.05))
        label_hbars(ax, bars, vals, errs=errs)
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [NAMES.get(m, m) for m in methods])
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=SLOTS[0]), plt.Rectangle((0, 0), 1, 1, color=SLOTS[1])]
    fig.legend(handles, ["reference methods", "VLM-centric variants (with recovery)"], loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def fig9_cost(rows: dict[str, dict], name: str) -> None:
    methods = [m for m in ("checks_only", "ours_full", *VARIANTS) if m in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = [rows[m]["calls_per_frame"] for m in methods]
    ys = [rows[m]["F1"] for m in methods]
    ax.scatter(xs, ys, s=64, color=SLOTS[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_xlabel("VLM calls per frame (mean, incl. re-ask and vote)")
    ax.set_ylabel("F1 (all frames of the run faults)")
    ax.set_xlim(-0.3, max(xs) * 1.35 + 0.5)
    ax.set_ylim(min(ys) - 0.05, max(ys) + 0.08)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Detection quality vs. VLM cost (dev split)", fontsize=10.5, loc="left")
    fig.tight_layout()
    place_labels(fig, ax, xs, ys, [NAMES.get(m, m) for m in methods])
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


CANDIDATES = [(8, 0, "left"), (-8, 0, "right"), (8, 10, "left"), (8, -10, "left"), (-8, 10, "right"),
              (-8, -10, "right"), (8, 20, "left"), (8, -20, "left"), (-8, 20, "right"), (-8, -20, "right"),
              (8, 30, "left"), (8, -30, "left")]


def place_labels(fig, ax, xs, ys, texts, fontsize=8.5) -> None:
    """Greedy label placement: for each point, the first candidate offset whose box overlaps no placed label, no
    point and stays inside the axes; the least-overlapping candidate when none is free. Leader line when moved."""
    renderer = fig.canvas.get_renderer()
    fig.canvas.draw()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    r = 6.0
    point_boxes = [(px - r, py - r, px + r, py + r) for px, py in pts]
    inside = ax.get_window_extent(renderer)
    placed: list[tuple[float, float, float, float]] = []

    def overlap(b, c) -> float:
        w = min(b[2], c[2]) - max(b[0], c[0])
        h = min(b[3], c[3]) - max(b[1], c[1])
        return w * h if w > 0 and h > 0 else 0.0

    for i in sorted(range(len(texts)), key=lambda i: -ys[i]):
        best = None
        for dx, dy, ha in CANDIDATES:
            ann = ax.annotate(texts[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                              va="center", fontsize=fontsize, color=INK)
            bb = ann.get_window_extent(renderer)
            box = (bb.x0 - 2, bb.y0 - 1, bb.x1 + 2, bb.y1 + 1)
            out = (max(0, inside.x0 - box[0]) + max(0, box[2] - inside.x1) + max(0, inside.y0 - box[1]) +
                   max(0, box[3] - inside.y1)) * 50
            cost = out + sum(overlap(box, c) for c in placed) + sum(overlap(box, c) for c in point_boxes)
            ann.remove()
            if best is None or cost < best[0]:
                best = (cost, dx, dy, ha, box)
            if cost == 0:
                break
        _, dx, dy, ha, box = best
        arrow = {"arrowstyle": "-", "color": INK2, "lw": 0.6, "shrinkA": 0, "shrinkB": 5} if abs(dy) > 5 else None
        ax.annotate(texts[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=fontsize, color=INK, arrowprops=arrow)
        placed.append(box)


BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def fig10_per_fault(pf: dict[str, dict[str, float]], faults: list[str], name: str) -> None:
    """Per-fault F1 heatmap: one sequential hue, every cell labeled."""
    from matplotlib.colors import LinearSegmentedColormap

    methods = [m for m in ("checks_only", "ours_full", *VARIANTS) if m in pf]
    vals = np.array([[pf[m].get(f) if pf[m].get(f) is not None else np.nan for f in faults] for m in methods])
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(8.2, 0.7 + 0.42 * len(methods)))
    ax.imshow(vals, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(methods)):
        for j in range(len(faults)):
            v = vals[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if not np.isnan(v) and v >= 0.55 else INK)
    ax.set_xticks(range(len(faults)), faults)
    ax.set_yticks(range(len(methods)), [NAMES.get(m, m) for m in methods])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("F1 per fault type on the dev split (seed 0); darker is better", fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


# --- tables ------------------------------------------------------------------------------------------------------

COLS = [("F1", "F1"), ("P", "P"), ("R", "R"), ("sensor_acc", "sensor acc."), ("false_alarm", "false alarm"),
        ("unnecessary", "unnecessary"), ("missed", "missed"), ("rec_success", "rec. success"),
        ("calls_per_frame", "VLM calls/frame")]


def md_method_table(rows: dict[str, dict]) -> str:
    methods = [m for m in (*REFERENCE, *VARIANTS) if m in rows]
    lines = ["| method | " + " | ".join(h for _, h in COLS) + " |", "|---|" + "---|" * len(COLS)]
    for m in methods:
        cells = [fmt(rows[m].get(k), rows[m].get(f"{k}_std")) for k, _ in COLS]
        lines.append(f"| {NAMES.get(m, m)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--screen", default="vlm_screen_s0", help="dev screening tag (seed 0)")
    p.add_argument("--screen-extra", nargs="*", default=[], help="more tags over the same frames (e.g. think_trust)")
    p.add_argument("--seeds", nargs="*", default=[], help="tags whose union holds 3 seeds of the best variants")
    p.add_argument("--confirm-variant", nargs="*", default=[], help="nuScenes confirm tags of a variant")
    p.add_argument("--agent", default=None, help="dev tag with the agent faults A1-A3 (seed 0)")
    p.add_argument("--carla-confirm", nargs="*", default=[], help="CARLA confirm tags (outputs/falsify/carla_<tag>)")
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    summary: dict = {}
    md = ["# Results of the 2026-11-21 run", ""]

    nus, carla = nuscenes_falsification(), carla_falsification()
    summary["falsification_nuscenes"], summary["falsification_carla"] = nus, carla
    fig6_falsification(nus, carla)
    md += ["## Falsification: random vs. guided search", "", "| fault | random | TPE | trials per sampler |",
           "|---|---|---|---|"]
    for f in sorted({r["fault"] for r in nus}, key=lambda f: (f[0] != "C", f)):
        rr = {r["sampler"]: r for r in nus if r["fault"] == f}
        md.append(f"| {f} | {rr['random']['violations']} | {rr['tpe']['violations']} | {rr['random']['trials']} |")
    md += ["", "| CARLA hazard | sampler | runs | collisions | R5 violations | R4 violations | min R5 |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {r['hazard']} | {r['sampler']} | {r['runs']} | {r['collisions']} | {r['R5_violations']} | "
           f"{r['R4_violations']} | {r['min_R5']} |" for r in carla]

    rep_tags = ["confirm_cpu_v1", "confirm_gpu_v1", *args.confirm_variant]
    rep = replays(rep_tags)
    summary["replays"] = rep
    if rep:
        fig7_replays(rep_tags)
        md += ["", "## Counterexamples of the checks replayed with a VLM method", "",
               "| source | method | replayed | still violated |", "|---|---|---|---|"]
        md += [f"| {r['tag']} | {r['method']} | {r['replayed']} | {r['still_violated']} |" for r in rep]

    if (Path("outputs/eval") / args.screen / "meta.json").exists():
        recs, a = load_eval(args.screen)
        for tag in args.screen_extra:
            recs = merge_tags(recs, load_eval(tag)[0])
        methods = [m for m in (*REFERENCE, *VARIANTS) if any(m in r["methods"] for r in recs)]
        rows = method_rows([r for r in recs if all(m in r["methods"] for m in methods)], methods, a["faults"],
                           a["seeds"])
        summary["screen"] = {"tag": args.screen, "faults": a["faults"], "seeds": a["seeds"], "frames": len(recs),
                             "rows": rows, "per_fault_f1": per_fault_f1(recs, methods, a["faults"])}
        same = [r for r in recs if all(m in r["methods"] for m in methods)]
        boot = bootstrap_metrics(same, [m for m in methods if m != "B0"], "ours_full")
        summary["screen"]["bootstrap"] = boot
        fig8_variants(rows, f"VLM-centric variants on the dev split (seed {a['seeds'][0]}, faults "
                            f"{' '.join(a['faults'])}; {len(recs)} frames)", "fig8_vlm_variants_screen.png")
        fig9_cost(rows, "fig9_cost_vs_f1.png")
        fig10_per_fault(summary["screen"]["per_fault_f1"], a["faults"], "fig10_per_fault_f1.png")
        md += ["", f"## VLM-centric variants, dev screening ({args.screen}: {len(recs)} frames)", "",
               md_method_table(rows), "", "Per-fault F1:", "",
               "| method | " + " | ".join(a["faults"]) + " |", "|---|" + "---|" * len(a["faults"])]
        pf = summary["screen"]["per_fault_f1"]
        md += [f"| {NAMES.get(m, m)} | " + " | ".join(fmt(pf[m].get(f)) for f in a["faults"]) + " |" for m in methods]
        md += ["", "Ours-full minus each method, 95% block-bootstrap interval over fault episodes (2,000 draws):", "",
               "| comparison | F1 | sensor acc. | false alarm | unnecessary | missed |", "|---|---|---|---|---|---|"]
        for name in sorted({b["method"] for b in boot if b["method"].startswith("ours_full - ")}):
            cells = []
            for k in ("F1", "sensor_acc", "false_alarm", "unnecessary", "missed"):
                b = next(x for x in boot if x["method"] == name and x["metric"] == k)
                cells.append("—" if b["point"] is None else f"{b['point']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]")
            other = name.split(" - ", 1)[1]
            md.append(f"| Ours-full − {NAMES.get(other, other)} | " + " | ".join(cells) + " |")
    if args.seeds:
        recs, faults, seeds = union_tags(args.seeds)
        methods = [m for m in (*REFERENCE, *VARIANTS)
                   if all(any(m in r["methods"] for r in recs if r["seed"] == s) for s in seeds)]
        recs = [r for r in recs if all(m in r["methods"] for m in methods)]
        rows = method_rows(recs, methods, faults, seeds)
        boot = bootstrap_metrics(recs, [m for m in methods if m != "B0"], "ours_full")
        summary["seeds"] = {"tags": args.seeds, "faults": faults, "seeds": seeds, "frames": len(recs), "rows": rows,
                            "bootstrap": boot}
        fig8_variants(rows, f"Dev split, mean ± std over seeds {seeds} (faults {' '.join(faults)}; {len(recs)} frames)",
                      "fig8_vlm_variants_seeds.png")
        md += ["", f"## Seeds {seeds} ({', '.join(args.seeds)}: {len(recs)} frames)", "", md_method_table(rows)]
    if args.agent:
        recs, a = load_eval(args.agent)
        methods = [m for m in ("ours_full", *VARIANTS) if any(m in r["methods"] for r in recs)]
        recs = [r for r in recs if all(m in r["methods"] for m in methods)]
        e2 = e2_table(recs, methods, a["faults"])
        rows = method_rows(recs, methods, a["faults"], a["seeds"])
        summary["agent"] = {"tag": args.agent, "faults": a["faults"], "frames": len(recs), "rows": rows,
                            "rec_success": {m: {r["fault"]: r["rec_success"] for r in e2 if r["method"] == m}
                                            for m in methods}}
        md += ["", f"## Agent faults ({args.agent}: {len(recs)} frames, seed {a['seeds'][0]})", "",
               "| method | " + " | ".join(f"rec. success {f}" for f in a["faults"]) + " | rec. success all | "
               "unnecessary | missed | VLM calls/frame |", "|---|" + "---|" * (len(a["faults"]) + 4)]
        for m in methods:
            rs = summary["agent"]["rec_success"][m]
            md.append(f"| {NAMES.get(m, m)} | " + " | ".join(fmt(rs.get(f)) for f in a["faults"]) +
                      f" | {fmt(rs.get('all'))} | {fmt(rows[m]['unnecessary'])} | {fmt(rows[m]['missed'])} | "
                      f"{fmt(rows[m]['calls_per_frame'])} |")
    if args.carla_confirm:
        crow = [x | {"tag": t} for t in args.carla_confirm
                for x in read_jsonl(Path("outputs/falsify") / f"carla_{t}" / "confirm.jsonl")]
        summary["carla_confirm"] = crow
        md += ["", "## CARLA scenarios replayed with another method", "",
               ("| tag | hazard | source | fault | R4 checks-only | R4 method | R5 checks-only | R5 method | "
               "collision checks-only | collision method | front camera isolated in fault steps, checks-only / "
               "method | clean steps with a sensor isolated, checks-only / method |"),
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        src = {(t, r["trial"]): r for t in {x["from_tag"] for x in crow}
               for r in read_jsonl(Path("outputs/falsify") / f"carla_{t}" / "trials.jsonl")}

        def isolation_counts(run_dir: str) -> tuple[int, int, int, int]:
            steps = [s for f in sorted(Path(run_dir).glob("*.jsonl")) for s in read_jsonl(f)]
            fault = [s for s in steps if s["label"]["fault_id"] != "none"]
            clean = [s for s in steps if s["label"]["fault_id"] == "none"]
            return (sum("camera:CAM_FRONT" in (s.get("isolated") or []) for s in fault), len(fault),
                    sum(bool(s.get("isolated")) for s in clean), len(clean))

        for x in crow:
            m = next(k for k in x if k.endswith("_full") or k == "ours_full")
            c, v = x["checks_only"], x[m]
            row = src.get((x["from_tag"], x["trial"]), {})
            sc = row.get("scenario", {})
            x["method"], x["fault"], x["fault_level"] = m, sc.get("fault"), sc.get("fault_level")
            x["front_isolated_in_fault"], x["fault_steps"], x["clean_steps_degraded"], x["clean_steps"] = \
                isolation_counts(x["run_dir"])
            x["checks_only_counts"] = list(isolation_counts(row["run_dir"])) if row.get("run_dir") else None
            fault = f"{sc.get('fault', '')} {sc['fault_level']:.1f}" if sc.get("fault_level") is not None else \
                sc.get("fault", "")
            cc = x["checks_only_counts"] or ["?"] * 4
            md.append(f"| {x['tag']} | {x['hazard']} | {x['from_tag']} #{x['trial']} | {fault} | "
                      f"{fmt(c['R4'])} | {fmt(v['R4'])} | {fmt(c['R5'])} | {fmt(v['R5'])} | {c['collision']} | "
                      f"{v['collision']} | {cc[0]}/{cc[1]} / {x['front_isolated_in_fault']}/{x['fault_steps']} | "
                      f"{cc[2]}/{cc[3]} / {x['clean_steps_degraded']}/{x['clean_steps']} |")
    (OUT / "results_2026-11-21.md").write_text("\n".join(md) + "\n")
    (OUT / "results_2026-11-21.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
