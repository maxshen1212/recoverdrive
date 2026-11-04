"""Figures and summary tables for REPORT.md (Phases 1-5). Reads finished runs; writes report_figures/.

Colors: the dataviz reference palette (light surface), categorical slots 1-4 in their validated order, the blue
sequential ramp for the heatmap; every mark carries a value label, so identity never relies on color alone.

Usage: uv run python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from recoverdrive.metrics import aggregate_seeds, e1_table, e2_table
from recoverdrive.runner import RunSpec, load_records

OUT = Path("report_figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FAULTS = ["C1", "C2", "C3", "C4", "L1", "L2", "L4", "A1", "A2", "A3"]
TEST_SCENES = ["scene-0103", "scene-0757", "scene-0796", "scene-1094"]
NAMES = {"B0": "B0", "B1": "B1", "B2": "B2", "B3": "B3", "ours_detect": "Ours-detect", "ours_full": "Ours-full"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})


def tables(tag: str, methods: list[str], seeds: list[int], recovery=("ours_full",)):
    specs = [RunSpec("test", sc, f, s) for s in seeds for f in FAULTS for sc in TEST_SCENES]
    recs = [r for r in load_records(Path("outputs/eval") / tag, specs) if "error" not in r]
    e1 = aggregate_seeds([e1_table([r for r in recs if r["seed"] == s], methods, FAULTS) for s in seeds])
    e2 = aggregate_seeds([e2_table([r for r in recs if r["seed"] == s], methods, FAULTS, recovery_methods=recovery)
                          for s in seeds])
    return e1, e2, len(recs)


def row(table, method, fault="all"):
    return next(r for r in table if r["method"] == method and r["fault"] == fault)


def label_bars(ax, bars, values, fmt="{:.2f}", horizontal=False):
    for b, v in zip(bars, values):
        if horizontal:
            ax.text(b.get_width() + 0.01 * ax.get_xlim()[1], b.get_y() + b.get_height() / 2, fmt.format(v),
                    va="center", ha="left", fontsize=9, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom",
                    fontsize=8.5, color=INK)


def fig_e1(e1):
    methods = ["B1", "B2", "B3", "ours_detect", "ours_full"]
    cols = FAULTS + ["all"]
    data = np.array([[row(e1, m, f)["F1"] or 0.0 for f in cols] for m in methods])
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(methods)):
        for j in range(len(cols)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color=SURFACE if data[i, j] > 0.62 else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(methods)), [NAMES[m] for m in methods])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("E1 detection F1 on the test split (mean of 3 seeds)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_e1_f1.png", dpi=160)
    plt.close(fig)


def fig_e2(e2):
    methods = ["B0", "B1", "B2", "B3", "ours_detect", "ours_full"]
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for k, (col, lab) in enumerate((("unnecessary", "unnecessary fallback"), ("missed", "missed fallback"))):
        vals = [row(e2, m)[col] for m in methods]
        errs = [row(e2, m).get(f"{col}_std") or 0 for m in methods]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, color=SLOTS[k], label=lab, edgecolor=SURFACE, linewidth=1.5,
                      yerr=errs, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
        label_bars(ax, bars, vals)
    ax.set_xticks(x, [NAMES[m] for m in methods])
    ax.set_ylabel("share of frames")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_ylim(0, 0.75)
    ax.set_title("E2 fallback errors on the test split (lower is better; mean ± std over 3 seeds)", loc="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_e2_fallback.png", dpi=160)
    plt.close(fig)


def e3_rows():
    runs = [json.loads(f.read_text()) for f in Path("outputs/e3/e3_final").glob("r*_s*.json")]
    out = []
    for m in ("B0", "B1", "ours_full"):
        for cond in ("clean", "C1"):
            rs = [r for r in runs if r["method"] == m and r["condition"] == cond]
            per_seed = {s: [r for r in rs if r["seed"] == s] for s in sorted({r["seed"] for r in rs})}
            rowd = {"method": NAMES[m], "condition": cond, "runs": len(rs)}
            for col, key in (("collision_rate", "collision"), ("route_completion", "route_completion"),
                             ("unnecessary_stops", "unnecessary_stop_rate")):
                vals = [np.mean([r[key] for r in v]) for v in per_seed.values()]
                rowd[col], rowd[f"{col}_std"] = float(np.mean(vals)), float(np.std(vals))
            out.append(rowd)
    return out


def fig_e3(rows):
    c1 = [r for r in rows if r["condition"] == "C1"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for ax, col, title in ((axes[0], "collision_rate", "collision rate (C1 runs)"),
                           (axes[1], "route_completion", "route completion (C1 runs)")):
        vals = [r[col] for r in c1]
        bars = ax.bar([r["method"] for r in c1], vals, 0.55, color=SLOTS[0], edgecolor=SURFACE, linewidth=1.5,
                      yerr=[r[f"{col}_std"] for r in c1], error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
        label_bars(ax, bars, vals)
        ax.set_ylim(0, 1.15)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10.5)
    fig.suptitle("E3 closed loop in CARLA, front camera fault during the hazard (12 runs per method)", x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_e3_closed_loop.png", dpi=160)
    plt.close(fig)


def fig_e4(lat, full_test: float, gated_test: float):
    comps = {c["component"]: c["mean"] for c in lat["components"]}
    segs = [("checks, render, load", comps["load"] + comps["checks"] + comps["render"]), ("SAM 3 (6 cameras)",
            comps["sam3"]), ("VLM first call", comps["vlm_first"]), ("VLM 3 samples (parallel)",
                                                                      comps["vlm_samples"] + comps["vlm_reask"])]
    measured = {m["method"].split(" (")[0]: m["mean"] for m in lat["methods"]}
    methods = [("Ours-detect", measured["Ours-detect"]), ("Ours-full", measured["Ours-full"]),
               ("B1", measured["B1"]), ("B2", measured["B2"]), ("B3", measured["B3"]),
               ("Ours-full, test flag rate*", full_test), ("Ours-full + VLM gate, test flag rate*", gated_test)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    names, vals = [m[0] for m in methods][::-1], [m[1] for m in methods][::-1]
    bars = ax.barh(names, vals, 0.6, color=SLOTS[0], edgecolor=SURFACE, linewidth=1.5)
    ax.set_xlim(0, max(vals) * 1.2)
    label_bars(ax, bars, vals, "{:.1f} s", horizontal=True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("seconds per frame (mean)")
    ax.set_title("per-frame latency by method", loc="left", fontsize=10.5)
    ax = axes[1]
    left = 0.0
    for k, (lab, v) in enumerate(segs):
        ax.barh(["Ours-full"], [v], 0.5, left=left, color=SLOTS[k], edgecolor=SURFACE, linewidth=2,
                label=f"{lab}: {v:.1f} s")
        if v > 0.6:
            ax.text(left + v / 2, 0, f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color=SURFACE if k == 0 else INK)
        left += v
    ax.set_xlim(0, left * 1.05)
    ax.set_xlabel("seconds per frame (mean, dev scene-0061)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.28), ncol=1, fontsize=9)
    ax.set_title("Ours-full: where the time goes", loc="left", fontsize=10.5)
    fig.suptitle("E4 latency (Qwen3-VL-8B on one shared RTX 6000 Ada)", x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.02, "* measured component times re-weighted to the test split, where 30 % of frames carry a flag",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig4_e4_latency.png", dpi=160)
    plt.close(fig)


def fig_e5(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(rows) + 1.4), sharey=True)
    names = [r["variant"] for r in rows][::-1]
    for ax, col, title in ((axes[0], "F1", "E1 F1 (higher is better)"),
                           (axes[1], "unnecessary", "E2 unnecessary fallback (lower is better)")):
        vals = [r[col] for r in rows][::-1]
        bars = ax.barh(names, vals, 0.6, color=SLOTS[0], edgecolor=SURFACE, linewidth=1.5)
        ref = rows[0][col]
        ax.axvline(ref, color=INK2, linestyle="--", linewidth=1)
        ax.set_xlim(0, max(vals) * 1.25)
        label_bars(ax, bars, vals, "{:.3f}", horizontal=True)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10.5)
    fig.suptitle("E5 ablation of Ours-full on the test split (dashed line = full system)", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_e5_ablation.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    summary = {}
    e1, e2, n = tables("test_final", list(NAMES), [0, 1, 2])
    summary["test_main"] = {"frames": n, "e1": e1, "e2": e2}
    fig_e1(e1)
    fig_e2(e2)
    rows3 = e3_rows()
    summary["e3"] = rows3
    fig_e3(rows3)

    replay = json.loads(Path("outputs/eval/test_final/ablation_replay.json").read_text())
    flag_share = 1 - next(c for c in replay["calls"] if "gate" in c["method"])["frames_without_vlm"]
    lat = json.loads(Path("outputs/e4/latency.json").read_text())
    # Re-weight the measured per-frame times by the test split's flag rate (30 % of frames carry a flag there).
    full_test = (lat["pre_vlm_mean"] + flag_share * lat["method_flagged_mean"]
                 + (1 - flag_share) * lat["method_unflagged_mean"])
    gated_test = lat["pre_vlm_mean"] + flag_share * lat["method_flagged_mean"]
    summary["e4"] = {**lat, "test_flag_share": flag_share, "full_latency_test_estimate": full_test,
                     "gated_latency_test_estimate": gated_test}
    fig_e4(lat, full_test, gated_test)

    variants = [("Ours-full (reference)", "full")]
    variants += [("− 3-sample voting", "no_sampling"), ("− re-observation", "no_reobserve"),
                 ("− sensor isolation", "no_isolation"), ("+ VLM gate (call only on flags)", "gate_flags")]
    rows5 = []
    for label, key in variants:
        r1 = next(r for r in replay["e1"] if r["method"] == f"ours_full[{key}]" and r["fault"] == "all")
        r2 = next(r for r in replay["e2"] if r["method"] == f"ours_full[{key}]" and r["fault"] == "all")
        rows5.append({"variant": label, "seeds": 3, **{k: r1[k] for k in ("F1", "R", "P", "sensor_acc",
                                                                           "false_alarm")},
                      **{k: r2[k] for k in ("unnecessary", "missed", "rec_success")}})
    online = [("− consistency checks", "test_abl_no_checks", [0, 1, 2]),
              ("SAM 3 → Qwen grounding", "test_abl_qwen_grounding", [0, 1, 2]),
              ("Qwen3-VL 8B → 4B", "test_abl_4b", [0, 1, 2]),
              ("Instruct → Thinking (seed 0)", "test_abl_thinking", [0])]
    for label, tag, seeds in online:
        if not (Path("outputs/eval") / tag / "summary.md").exists():
            continue
        a1, a2, _ = tables(tag, ["ours_full"], seeds)
        rows5.append({"variant": label, "seeds": len(seeds),
                      **{k: row(a1, "ours_full")[k] for k in ("F1", "R", "P", "sensor_acc", "false_alarm")},
                      **{k: row(a2, "ours_full")[k] for k in ("unnecessary", "missed", "rec_success")}})
    s0e1, s0e2, _ = tables("test_final", ["ours_full"], [0])  # reference for the seed-0-only Thinking run
    summary["e5"] = rows5
    summary["e5_reference_seed0"] = {**{k: row(s0e1, "ours_full")[k] for k in ("F1", "R", "P", "sensor_acc",
                                                                            "false_alarm")},
                                     **{k: row(s0e2, "ours_full")[k] for k in ("unnecessary", "missed",
                                                                               "rec_success")}}
    fig_e5(rows5)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"e5": rows5, "e5_reference_seed0": summary["e5_reference_seed0"],
                      "full_latency_test_estimate": full_test, "gated_latency_test_estimate": gated_test,
                      "test_flag_share": flag_share}, indent=1))


if __name__ == "__main__":
    main()
