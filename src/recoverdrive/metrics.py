"""E1 / E2 metrics over per-frame run records, seed aggregation, and markdown tables (spec §5.2-5.3)."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def _safe(num: float, den: float) -> float | None:
    return num / den if den else None


def method_frames(records: list[dict], method: str):
    """(record, method output) pairs for frames where the method produced an output."""
    for r in records:
        out = r.get("methods", {}).get(method)
        if out is not None:
            yield r, out


def e1_counts(pairs) -> dict:
    tp = fp = fn = tn = 0
    correct_sensor = 0
    for r, out in pairs:
        gt = r["label"]["fault_id"] != "none"
        if gt and out["alarm"]:
            tp += 1
            correct_sensor += out["pred_sensor"] == r["label"]["sensor"]
        elif gt:
            fn += 1
        elif out["alarm"]:
            fp += 1
        else:
            tn += 1
    p, rec = _safe(tp, tp + fp), _safe(tp, tp + fn)
    f1 = None if p is None or rec is None or p + rec == 0 else 2 * p * rec / (p + rec)
    return {"n": tp + fp + fn + tn, "n_fault": tp + fn, "P": p, "R": rec, "F1": f1,
            "sensor_acc": _safe(correct_sensor, tp), "false_alarm": _safe(fp, fp + tn)}


def e1_table(records: list[dict], methods: list[str], faults: list[str]) -> list[dict]:
    """One row per method x run fault (plus 'all'). False alarm uses the clean frames of those runs."""
    rows = []
    for m in methods:
        for f in [*faults, "all"]:
            recs = [r for r in records if f == "all" or r["run_fault"] == f]
            rows.append({"method": m, "fault": f, **e1_counts(method_frames(recs, m))})
    return rows


def e1_recall_by_severity(records: list[dict], methods: list[str], faults: list[str]) -> list[dict]:
    rows = []
    for m in methods:
        for f in faults:
            row = {"method": m, "fault": f}
            for sev in ("low", "medium", "high"):
                pairs = [(r, o) for r, o in method_frames(records, m)
                         if r["run_fault"] == f and r["label"]["fault_id"] != "none" and r["label"]["severity"] == sev]
                row[sev] = _safe(sum(o["alarm"] for _, o in pairs), len(pairs))
                row[f"n_{sev}"] = len(pairs)
            rows.append(row)
    return rows


def aggregate_seeds(tables: list[list[dict]], keys: tuple[str, ...] = ("method", "fault")) -> list[dict]:
    """Mean and std (population, over seeds) of every numeric column; rows matched on `keys`."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    order: list[tuple] = []
    for table in tables:
        for row in table:
            k = tuple(row[c] for c in keys)
            if k not in grouped:
                order.append(k)
            grouped[k].append(row)
    out = []
    for k in order:
        rows = grouped[k]
        agg = dict(zip(keys, k))
        for col in rows[0]:
            if col in keys:
                continue
            vals = [r[col] for r in rows if isinstance(r.get(col), (int, float)) and r[col] is not None]
            if not vals:
                agg[col] = None
                continue
            agg[col] = float(np.mean(vals))
            agg[f"{col}_std"] = float(np.std(vals)) if len(vals) > 1 else None
        out.append(agg)
    return out


def fmt(v, std=None, pct=False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, int) and not isinstance(v, bool) and std is None:
        return str(v)
    scale = 100 if pct else 1
    text = f"{v * scale:.1f}" if pct else f"{v:.3f}"
    if std is not None:
        text += f" ± {std * scale:.1f}" if pct else f" ± {std:.3f}"
    return text


def markdown_table(rows: list[dict], cols: list[str], pct: tuple[str, ...] = ()) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(fmt(v, r.get(f"{c}_std"), c in pct) if not isinstance(v, str) else v)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- E2: recovery and fallback ----------------------------------------------------------------------------

RECOVERY_METHODS = ("ours_full", "checks_only")


def _label(r: dict):
    from recoverdrive.types import FaultLabel

    return FaultLabel(**r["label"])


def recovery_success(r: dict, out: dict) -> bool | None:
    """Spec §5.2. Sensor fault: degraded and exactly the faulty sensor isolated. Agent fault: valid answer that
    matches the fault-free one after re-asking. None on clean frames."""
    label = r["label"]
    if label["fault_id"] == "none":
        return None
    if label["sensor"] == "agent":
        return bool(out.get("extra", {}).get("agent_recovered", False))
    rec = out.get("recovery") or {}
    isolated = set(rec.get("isolated_sensors", [])) - {"agent"}
    return rec.get("final_state") == "degraded" and isolated == {label["sensor"]}


def e2_counts(pairs, with_recovery: bool) -> dict:
    from recoverdrive.shield import acceptable_actions, classify_action

    n = unnecessary = missed = 0
    n_clean = unn_clean = n_fault = missed_fault = 0
    successes: list[bool] = []
    episodes: dict[tuple, list[tuple[int, bool]]] = defaultdict(list)
    for r, out in pairs:
        cls = classify_action(out["action"]["action"], acceptable_actions(_label(r)))
        n += 1
        unnecessary += cls == "unnecessary"
        missed += cls == "missed"
        if r["label"]["fault_id"] == "none":
            n_clean += 1
            unn_clean += cls == "unnecessary"
            continue
        n_fault += 1
        missed_fault += cls == "missed"
        if with_recovery:
            ok = bool(recovery_success(r, out))
            successes.append(ok)
            key = (r["scene"], r["seed"], r["run_fault"], r["label"]["episode_id"])
            episodes[key].append((r["frame_idx"], ok))
    row = {"n": n, "unnecessary": _safe(unnecessary, n), "missed": _safe(missed, n),
           "unnecessary_clean": _safe(unn_clean, n_clean), "missed_fault": _safe(missed_fault, n_fault),
           "rec_success": None, "ep_recovered": None, "frames_to_recover": None}
    if with_recovery and successes:
        row["rec_success"] = float(np.mean(successes))
        firsts = []
        for frames in episodes.values():
            frames.sort()
            hit = next((i for i, (_, ok) in enumerate(frames) if ok), None)
            if hit is not None:
                firsts.append(hit + 1)
        row["ep_recovered"] = len(firsts) / len(episodes)
        row["frames_to_recover"] = float(np.mean(firsts)) if firsts else None
    return row


def e2_table(records: list[dict], methods: list[str], faults: list[str],
             recovery_methods: tuple[str, ...] = RECOVERY_METHODS) -> list[dict]:
    rows = []
    for m in methods:
        for f in [*faults, "all"]:
            recs = [r for r in records if f == "all" or r["run_fault"] == f]
            rows.append({"method": m, "fault": f, **e2_counts(method_frames(recs, m), m in recovery_methods)})
    return rows


# --- Easy vs. hard faults (FutureWork Step 1) ---------------------------------------------------------------

EASY_ANY_SEVERITY = ("L4", "A1", "A3")  # caught by a point count, or by JSON validation that every VLM method has


def difficulty(label: dict) -> str | None:
    """'easy' for L4, A1, A3 and high-severity L1; 'hard' for every other fault; None for clean frames."""
    fault = label["fault_id"]
    if fault == "none":
        return None
    if fault in EASY_ANY_SEVERITY or (fault == "L1" and label.get("severity") == "high"):
        return "easy"
    return "hard"


def difficulty_table(records: list[dict], methods: list[str],
                     recovery_methods: tuple[str, ...] = RECOVERY_METHODS) -> list[dict]:
    """Recall, sensor accuracy, missed fallback and recovery success on fault frames, split by difficulty
    (seeds pooled)."""
    rows = []
    for m in methods:
        for diff in ("easy", "hard", "all"):
            pairs = [(r, o) for r, o in method_frames(records, m)
                     if difficulty(r["label"]) is not None and (diff == "all" or difficulty(r["label"]) == diff)]
            e1 = e1_counts(pairs)
            e2 = e2_counts(pairs, m in recovery_methods)
            rows.append({"method": m, "difficulty": diff, "n_fault": e1["n_fault"], "R": e1["R"],
                         "sensor_acc": e1["sensor_acc"], "missed_fault": e2["missed_fault"],
                         "rec_success": e2["rec_success"]})
    return rows


def latency_estimates(pre_vlm_mean: float, flagged_mean: float, unflagged_mean: float, flag_share: float) -> dict:
    """Per-frame latency re-weighted to a split's share of flagged frames: 'full' calls the VLM on every frame,
    'gated' only on flagged frames (spec §4.2 latency optimization)."""
    return {"full": pre_vlm_mean + flag_share * flagged_mean + (1 - flag_share) * unflagged_mean,
            "gated": pre_vlm_mean + flag_share * flagged_mean}


def sensor_fault_view(records: list[dict]) -> list[dict]:
    """Records without agent-fault frames (clean frames and sensor-fault frames stay). Agent faults are failures of the
    VLM itself, which a monitor without a VLM cannot have, so they must be left out when the two are compared."""
    return [r for r in records if r["label"]["sensor"] != "agent"]

