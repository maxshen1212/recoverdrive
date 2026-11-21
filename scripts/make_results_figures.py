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

from recoverdrive.metrics import aggregate_seeds, e1_table, e2_table, fmt
from recoverdrive.runner import RunSpec, load_records

REPO = Path(__file__).resolve().parents[1]
OUT = Path("report_figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NAMES = {"B0": "B0 (no monitor)", "checks_only": "Checks-only", "ours_full": "Ours-full (frozen)",
         "think_full": "Think", "think_nochecks_full": "Think, no checks",
         "perceive_full": "Perceive → decide", "perceive_nochecks_full": "Perceive → decide, no checks",
         "percam_full": "Per-sensor → decide"}
REFERENCE = ("B0", "checks_only", "ours_full")
VARIANTS = ("think_full", "think_nochecks_full", "perceive_full", "perceive_nochecks_full", "percam_full")
SENSOR_FAULTS = ["C1", "C2", "C3", "C4", "L1", "L2", "L4"]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


# --- evaluation runs -------------------------------------------------------------------------------------------

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


def fig7_replays(rows: list[dict]) -> None:
    by_method: dict[str, list[int]] = {}
    for r in rows:
        acc = by_method.setdefault(r["method"], [0, 0])
        acc[0] += r["replayed"]
        acc[1] += r["still_violated"]
    methods = list(by_method)
    fig, ax = plt.subplots(figsize=(7.5, 0.6 + 0.55 * len(methods)))
    y = np.arange(len(methods))
    share = [by_method[m][1] / by_method[m][0] for m in methods]
    bars = ax.barh(y, share, 0.55, color=[SLOTS[0] if m == "ours_full" else SLOTS[1] for m in methods])
    ax.set_xlim(0, 1.25)
    for bb, m in zip(bars, methods):
        n, v = by_method[m]
        ax.text(bb.get_width() + 0.015, bb.get_y() + bb.get_height() / 2, f"{v} of {n} still violated", va="center",
                fontsize=8.5)
    ax.set_yticks(y, [NAMES.get(m, m) for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("share of the checks' worst counterexamples that the VLM method still violates")
    ax.set_title("Replaying the counterexamples of the deterministic checks with a VLM", fontsize=10.5, loc="left")
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
    for x, yv, m in zip(xs, ys, methods):
        ax.annotate(NAMES.get(m, m), (x, yv), xytext=(7, 4), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("VLM calls per frame (mean, incl. re-ask and vote)")
    ax.set_ylabel("F1 (all frames of the run faults)")
    ax.set_xlim(-0.3, max(xs) * 1.35 + 0.5)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Detection quality vs. VLM cost (dev split)", fontsize=10.5, loc="left")
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
    p.add_argument("--seeds", default=None, help="tag with the 3-seed runs of the best variants, if done")
    p.add_argument("--confirm-variant", nargs="*", default=[], help="nuScenes confirm tags of a variant")
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

    rep = replays(["confirm_cpu_v1", "confirm_gpu_v1", *args.confirm_variant])
    summary["replays"] = rep
    if rep:
        fig7_replays(rep)
        md += ["", "## Counterexamples of the checks replayed with a VLM method", "",
               "| source | method | replayed | still violated |", "|---|---|---|---|"]
        md += [f"| {r['tag']} | {r['method']} | {r['replayed']} | {r['still_violated']} |" for r in rep]

    if (Path("outputs/eval") / args.screen / "meta.json").exists():
        recs, a = load_eval(args.screen)
        methods = [m for m in (*REFERENCE, *VARIANTS) if any(m in r["methods"] for r in recs[:50])]
        rows = method_rows(recs, methods, a["faults"], a["seeds"])
        summary["screen"] = {"tag": args.screen, "faults": a["faults"], "seeds": a["seeds"], "frames": len(recs),
                             "rows": rows, "per_fault_f1": per_fault_f1(recs, methods, a["faults"])}
        fig8_variants(rows, f"VLM-centric variants on the dev split (seed {a['seeds'][0]}, faults "
                            f"{' '.join(a['faults'])}; {len(recs)} frames)", "fig8_vlm_variants_screen.png")
        fig9_cost(rows, "fig9_cost_vs_f1.png")
        md += ["", f"## VLM-centric variants, dev screening ({args.screen}: {len(recs)} frames)", "",
               md_method_table(rows), "", "Per-fault F1:", "",
               "| method | " + " | ".join(a["faults"]) + " |", "|---|" + "---|" * len(a["faults"])]
        pf = summary["screen"]["per_fault_f1"]
        md += [f"| {NAMES.get(m, m)} | " + " | ".join(fmt(pf[m].get(f)) for f in a["faults"]) + " |" for m in methods]
    if args.seeds and (Path("outputs/eval") / args.seeds / "meta.json").exists():
        recs, a = load_eval(args.seeds)
        methods = [m for m in (*REFERENCE, *VARIANTS) if any(m in r["methods"] for r in recs[:50])]
        rows = method_rows(recs, methods, a["faults"], a["seeds"])
        summary["seeds"] = {"tag": args.seeds, "faults": a["faults"], "seeds": a["seeds"], "frames": len(recs),
                            "rows": rows}
        fig8_variants(rows, f"Best variants on the dev split, mean ± std over seeds {a['seeds']} (faults "
                            f"{' '.join(a['faults'])})", "fig8_vlm_variants_seeds.png")
        md += ["", f"## Best variants, seeds {a['seeds']} ({args.seeds}: {len(recs)} frames)", "",
               md_method_table(rows)]
    (OUT / "results_2026-11-21.md").write_text("\n".join(md) + "\n")
    (OUT / "results_2026-11-21.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
