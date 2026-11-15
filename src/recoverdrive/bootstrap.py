"""Block bootstrap confidence intervals for the E1/E2 metrics (FutureWork Step 1).

Frames inside a fault episode, and frames inside one clean gap, are strongly dependent (the same scene, and methods
keep state across frames). The resampling unit is therefore a block: one fault episode or one clean gap of one run.
Blocks are resampled with replacement within strata (run fault, fault/clean), and every method uses the same draws,
so differences between methods are paired. Seeds are pooled; point estimates are pooled too, so they can differ a
little from the mean-over-seeds numbers of the main tables.
"""

from __future__ import annotations

import numpy as np

from recoverdrive.shield import acceptable_actions, classify_action
from recoverdrive.types import FaultLabel

COUNT_KEYS = ("tp", "fp", "fn", "tn", "correct", "unnecessary", "missed", "n")
METRICS = ("F1", "sensor_acc", "false_alarm", "unnecessary", "missed")
_K = {k: i for i, k in enumerate(COUNT_KEYS)}


def _run(r: dict) -> tuple:
    return r["scene"], r["seed"], r["run_fault"]


def _order(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda r: (*_run(r), r["frame_idx"]))


def block_ids(records: list[dict]) -> list[tuple]:
    """One block id per record; records of one run must be in frame order. A fault frame belongs to its episode, a
    clean frame to the k-th clean gap of its run."""
    ids, gap, in_clean = [], {}, {}
    for r in records:
        run = _run(r)
        if r["label"]["fault_id"] != "none":
            ids.append((*run, "fault", r["label"]["episode_id"]))
            in_clean[run] = False
            continue
        if not in_clean.get(run, False):
            gap[run] = gap.get(run, -1) + 1
        in_clean[run] = True
        ids.append((*run, "clean", gap[run]))
    return ids


def frame_counts(r: dict, out: dict) -> np.ndarray:
    """Indicator vector (COUNT_KEYS) of one frame for one method's output."""
    v = np.zeros(len(COUNT_KEYS))
    gt = r["label"]["fault_id"] != "none"
    alarm = bool(out["alarm"])
    v[_K["tp"]], v[_K["fn"]] = gt and alarm, gt and not alarm
    v[_K["fp"]], v[_K["tn"]] = (not gt) and alarm, (not gt) and not alarm
    v[_K["correct"]] = gt and alarm and out["pred_sensor"] == r["label"]["sensor"]
    cls = classify_action(out["action"]["action"], acceptable_actions(FaultLabel(**r["label"])))
    v[_K["unnecessary"]], v[_K["missed"]] = cls == "unnecessary", cls == "missed"
    v[_K["n"]] = 1
    return v


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def metrics_from_counts(c: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics from summed counts (last axis = COUNT_KEYS). F1 is 0 when precision and recall are both 0."""
    g = {k: c[..., i] for k, i in _K.items()}
    p, rec = _div(g["tp"], g["tp"] + g["fp"]), _div(g["tp"], g["tp"] + g["fn"])
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(p + rec > 0, 2 * p * rec / np.where(p + rec > 0, p + rec, 1), np.where(np.isnan(p + rec), np.nan, 0.0))
    return {"F1": f1, "sensor_acc": _div(g["correct"], g["tp"]), "false_alarm": _div(g["fp"], g["fp"] + g["tn"]),
            "unnecessary": _div(g["unnecessary"], g["n"]), "missed": _div(g["missed"], g["n"])}


def _value(x) -> float | None:
    x = float(x)
    return None if np.isnan(x) else x


def bootstrap_metrics(records: list[dict], methods: list[str], reference: str, n_boot: int = 2000,
                      seed: int = 0) -> list[dict]:
    """Point estimate and 95 % percentile interval of every metric per method, and of `reference - m` for every
    other method (paired: the same block draws for all methods)."""
    recs = _order(records)
    ids = block_ids(recs)
    blocks = list(dict.fromkeys(ids))
    index = {b: i for i, b in enumerate(blocks)}
    counts = {m: np.zeros((len(blocks), len(COUNT_KEYS))) for m in methods}
    for r, b in zip(recs, ids):
        for m in methods:
            out = r.get("methods", {}).get(m)
            if out is not None:
                counts[m][index[b]] += frame_counts(r, out)

    strata: dict[tuple, list[int]] = {}
    for i, b in enumerate(blocks):
        strata.setdefault((b[2], b[3]), []).append(i)  # (run fault, fault/clean)
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_boot, len(blocks)))
    for members in strata.values():
        draws = rng.choice(np.asarray(members), size=(n_boot, len(members)), replace=True)
        for row in range(n_boot):
            np.add.at(weights[row], draws[row], 1)

    point = {m: metrics_from_counts(counts[m].sum(0)) for m in methods}
    boot = {m: metrics_from_counts(weights @ counts[m]) for m in methods}
    rows = []

    def add(name: str, pt: dict, bs: dict) -> None:
        for k in METRICS:
            vals = bs[k][~np.isnan(bs[k])]
            lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan))
            rows.append({"method": name, "metric": k, "point": _value(pt[k]), "lo": _value(lo), "hi": _value(hi)})

    for m in methods:
        add(m, point[m], boot[m])
    for m in methods:
        if m != reference:
            add(f"{reference} - {m}", {k: point[reference][k] - point[m][k] for k in METRICS},
                {k: boot[reference][k] - boot[m][k] for k in METRICS})
    return rows


def merge_runs(runs: list[tuple[list[dict], str | None]]) -> list[dict]:
    """Join the records of several runs on (scene, seed, run_fault, frame_idx) and merge their method outputs.
    A prefix renames that run's methods to '<prefix>:<name>'. Raises ValueError on differing labels or on the same
    method name from two runs."""
    merged: dict[tuple, dict] = {}
    for recs, prefix in runs:
        for r in recs:
            key = (*_run(r), r["frame_idx"])
            methods = {(f"{prefix}:{k}" if prefix else k): v for k, v in r.get("methods", {}).items()}
            if key not in merged:
                merged[key] = {"scene": r["scene"], "seed": r["seed"], "run_fault": r["run_fault"],
                               "frame_idx": r["frame_idx"], "label": r["label"], "methods": {}}
            elif merged[key]["label"] != r["label"]:
                raise ValueError(f"labels differ for frame {key}")
            dup = set(methods) & set(merged[key]["methods"])
            if dup:
                raise ValueError(f"method(s) {sorted(dup)} come from two runs; give one of them a prefix")
            merged[key]["methods"].update(methods)
    return list(merged.values())
