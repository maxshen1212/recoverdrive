"""Formal requirements R1-R6 of RecoverDrive in Signal Temporal Logic, and trace builders for the logged runs.

One step is 0.5 s (nuScenes keyframes at 2 Hz; CARLA perception every 5 ticks of 0.1 s). Action ranks follow
`shield.ACTIONS`: continue 0, slow_down 1, stop 2. `need` and `allow` are the ranks of the least and the most
conservative acceptable action of the frame (`shield.acceptable_actions`).

Integer and boolean signals are compared against half-step thresholds (for example `slack_lo >= -0.5` for
"a >= need"), so robustness is never exactly zero on them: +0.5 means "holds", -0.5 "violated by one level".

| ID | Meaning                                           | Formula                                                  |
|----|---------------------------------------------------|----------------------------------------------------------|
| R1 | no missed fallback (grace k1 steps)               | G( need >= 1 -> F[0,k1] (a >= need) )                     |
| R2 | no unnecessary fallback                           | G( a <= allow )                                           |
| R3 | back to continue within 2 steps after a fault     | G( fault_end -> F[0,2] (a = continue or fault) )          |
| R4 | front-camera fault: slow down within 3 s (CARLA)  | G( front_onset -> F[0,6] (speed <= 3.06 m/s) )            |
| R5 | TTC < 2 s -> stop within 1 step; no collision     | G( (ttc_low -> F[0,1] (a_final = stop)) and gap >= 0 )    |
| R6 | no sensor restored while still faulty (each s)    | G( (fault_s and iso_s) -> X (iso_s or not fault_s) )      |
"""

from __future__ import annotations

import numpy as np

from recoverdrive.shield import ACTIONS, acceptable_actions
from recoverdrive.stl import (
    Always,
    And,
    Eventually,
    Formula,
    Implies,
    Or,
    Pred,
    SpecResult,
    check_always,
)
from recoverdrive.types import FaultLabel

ACTION_RANK = {a: i for i, a in enumerate(ACTIONS)}
NUSCENES_SPECS = ("R1", "R2", "R3", "R6")
CARLA_SPECS = ("R1", "R2", "R4", "R5", "R6")
NUSCENES_SENSORS = ("camera:CAM_FRONT", "camera:CAM_FRONT_LEFT", "camera:CAM_FRONT_RIGHT", "camera:CAM_BACK",
                    "camera:CAM_BACK_LEFT", "camera:CAM_BACK_RIGHT", "lidar")
CARLA_SENSORS = ("camera:CAM_FRONT", "camera:CAM_BACK", "lidar")
STEP_S = 0.5
RECOVERY_STEPS = 2  # R3
FRONT_REACTION_STEPS = 6  # R4: 3 s
SLOW_SPEED_MPS = 3.06  # R4: 10 km/h slow speed + 1 km/h tolerance
TTC_S = 2.0  # R5


def spec(name: str, sensors: tuple[str, ...] = (), grace: int = 1) -> Formula:
    """The STL formula of requirement `name` (without the outer G, which `check_always` applies)."""
    on = 0.5  # threshold of boolean signals
    if name == "R1":
        return Implies(Pred("need", ">=", on), Eventually(0, grace, Pred("slack_lo", ">=", -0.5)))
    if name == "R2":
        return Pred("slack_hi", ">=", -0.5)
    if name == "R3":
        return Implies(Pred("fault_end", ">=", on),
                       Eventually(0, RECOVERY_STEPS, Or(Pred("a", "<=", 0.5), Pred("fault", ">=", on))))
    if name == "R4":
        return Implies(Pred("front_onset", ">=", on),
                       Eventually(0, FRONT_REACTION_STEPS, Pred("speed", "<=", SLOW_SPEED_MPS)))
    if name == "R5":
        return And(Implies(Pred("ttc_low", ">=", on), Eventually(0, 1, Pred("a_final", ">=", 1.5))),
                   Pred("gap", ">=", 0.0))
    if name == "R6":
        if not sensors:
            raise ValueError("R6 needs at least one sensor")
        return And(*(Implies(And(Pred(f"fault_{s}", ">=", on), Pred(f"iso_{s}", ">=", on)),
                             Always(1, 1, Or(Pred(f"iso_{s}", ">=", on), Pred(f"fault_{s}", "<=", on))))
                     for s in sensors))
    raise ValueError(f"unknown requirement {name}")


def _ranks(acceptable) -> tuple[int, int]:
    ranks = [ACTION_RANK[a] for a in acceptable]
    return min(ranks), max(ranks)


def _common(actions: list[str], acceptable: list, fault: list[bool]) -> dict[str, np.ndarray]:
    a = np.array([ACTION_RANK[x] for x in actions], float)
    bounds = np.array([_ranks(acc) for acc in acceptable], float).reshape(-1, 2)
    f = np.array(fault, float)
    fault_end = np.zeros_like(f)
    fault_end[1:] = (f[:-1] > 0) & (f[1:] == 0)
    return {"a": a, "need": bounds[:, 0], "allow": bounds[:, 1], "slack_lo": a - bounds[:, 0],
            "slack_hi": bounds[:, 1] - a, "fault": f, "fault_end": fault_end}


def nuscenes_trace(records: list[dict], method: str,
                   sensors: tuple[str, ...]) -> tuple[dict[str, np.ndarray], int]:
    """Trace of one run (its records, any order) for `method`, and the number of records skipped (frame errors or
    no output of `method`)."""
    rows = [r for r in records if "error" not in r and method in r.get("methods", {})]
    skipped = len(records) - len(rows)
    rows.sort(key=lambda r: r["frame_idx"])
    labels = [r["label"] for r in rows]
    outs = [r["methods"][method] for r in rows]
    tr = _common([o["action"]["action"] for o in outs],
                 [acceptable_actions(FaultLabel(**lab)) for lab in labels],
                 [lab["fault_id"] != "none" for lab in labels])
    for s in sensors:
        tr[f"fault_{s}"] = np.array([lab["fault_id"] != "none" and lab["sensor"] == s for lab in labels], float)
        tr[f"iso_{s}"] = np.array([s in (o.get("recovery") or {}).get("isolated_sensors", []) for o in outs], float)
    return tr, skipped


def carla_trace(steps: list[dict], summary: dict,
                sensors: tuple[str, ...] = ()) -> tuple[dict[str, np.ndarray], int]:
    """Trace of one CARLA run from its step log and its summary, and the number of steps skipped (none so far).
    `a` is the monitor's action (`monitor_action`, before the vehicle's TTC brake), which R1/R2 judge; `a_final` is
    the applied action, which R5 judges. `gap` is the per-step `gap_m` when logged; older logs only know whether the
    run collided, which then counts as gap -1 at the last step."""
    steps = sorted(steps, key=lambda s: s["step"])
    labels = [s["label"] for s in steps]
    fault = [lab["fault_id"] != "none" for lab in labels]
    tr = _common([s.get("monitor_action", s["action"]) for s in steps], [s["acceptable"] for s in steps], fault)
    tr["a_final"] = np.array([ACTION_RANK[s["action"]] for s in steps], float)
    front = np.array([f and lab["sensor"] == "camera:CAM_FRONT" for f, lab in zip(fault, labels)], float)
    onset = front.copy()
    onset[1:] = (front[1:] > 0) & (front[:-1] == 0)
    tr["front_onset"] = onset
    tr["speed"] = np.array([s["speed_mps"] for s in steps], float)
    tr["ttc_low"] = np.array([s["ttc"] is not None and s["ttc"] < TTC_S for s in steps], float)
    if steps and all("gap_m" in s for s in steps):
        tr["gap"] = np.array([s["gap_m"] for s in steps], float)
    else:
        gap = np.ones(len(steps))
        if summary.get("collision") and len(steps):
            gap[-1] = -1.0
        tr["gap"] = gap
    for s in sensors:
        tr[f"fault_{s}"] = np.array([f and lab["sensor"] == s for f, lab in zip(fault, labels)], float)
        tr[f"iso_{s}"] = np.array([s in st.get("isolated", []) for st in steps], float)
    return tr, 0


def evaluate(trace: dict[str, np.ndarray], names: tuple[str, ...], sensors: tuple[str, ...] = (),
             grace: int = 1) -> dict[str, SpecResult]:
    return {n: check_always(spec(n, sensors, grace), trace) for n in names}
