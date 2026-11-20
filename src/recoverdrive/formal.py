"""Formal verification of the rule-based safety logic by exhaustive enumeration.

The shield and the recovery controller have finite input domains once the sensors are fixed, so every property below
is checked by running the real functions (`shield.shield`, `shield.ttc_override`, `RecoveryController.step`) on every
case. There is no separate model of the code, so nothing can drift between the model and the implementation. A
property either holds on all cases or comes back with the first counterexample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import chain, combinations

from recoverdrive.faults import AGENT_FAULTS, FIXED_HIGH, SENSOR_FAULTS, SEVERITIES
from recoverdrive.requirements import NUSCENES_SENSORS
from recoverdrive.shield import ACTIONS, FRONT_SENSORS, acceptable_actions, shield, ttc_override
from recoverdrive.types import Action, FaultLabel, RecoveryResult

SENSORS_ALL = NUSCENES_SENSORS
STATES = ("healthy", "degraded", "unresolved")
TTC_VALUES = (None, 0.5, 1.99, 2.0, 5.0)
RANK = {a: i for i, a in enumerate(ACTIONS)}


@dataclass
class PropertyResult:
    name: str
    statement: str
    holds: bool
    n_cases: int
    counterexample: dict | None = None


def subsets(items) -> list[tuple[str, ...]]:
    items = tuple(items)
    return list(chain.from_iterable(combinations(items, k) for k in range(len(items) + 1)))


def single_fault_labels() -> list[FaultLabel]:
    """Every single-fault label (assumption A1): each fault type on each sensor it can hit, at each severity."""
    cams = [s for s in SENSORS_ALL if s.startswith("camera:")]
    out = []
    for f in SENSOR_FAULTS:
        targets = cams if f.startswith("C") else ["lidar"]
        sevs = ("high",) if f in FIXED_HIGH else SEVERITIES
        out += [FaultLabel(f, s, sev, 0) for s in targets for sev in sevs]
    out += [FaultLabel(f, "agent", sev, 0) for f in AGENT_FAULTS for sev in SEVERITIES]
    return out


def correct_result(label: FaultLabel) -> RecoveryResult:
    if label.fault_id == "none":
        return RecoveryResult([], "healthy", [], None)
    return RecoveryResult([], "degraded", [label.sensor], None)


def _cex(label: FaultLabel, result: RecoveryResult, action: str) -> dict:
    return {"label": asdict(label), "result": {"final_state": result.final_state,
                                               "isolated_sensors": result.isolated_sensors},
            "action": action, "acceptable": list(acceptable_actions(label))}


def _check(name: str, statement: str, cases) -> PropertyResult:
    """cases: iterable of (ok: bool, counterexample: dict)."""
    n = 0
    for ok, cex in cases:
        n += 1
        if not ok:
            return PropertyResult(name, statement, False, n, cex)
    return PropertyResult(name, statement, True, n, None)


def verify_shield() -> list[PropertyResult]:
    labels = [FaultLabel.none()] + single_fault_labels()
    iso_sets = subsets(SENSORS_ALL + ("agent",))
    results = [RecoveryResult([], st, list(iso), None) for st in STATES for iso in iso_sets]

    def s1():
        for lab in labels:
            r = correct_result(lab)
            a = shield(r).action
            yield a in acceptable_actions(lab), _cex(lab, r, a)

    def s2():
        for st in STATES:
            for small in iso_sets:
                for big in iso_sets:
                    if set(small) <= set(big):
                        a = shield(RecoveryResult([], st, list(small), None)).action
                        b = shield(RecoveryResult([], st, list(big), None)).action
                        yield RANK[a] <= RANK[b], {"state": st, "isolated": list(small), "action": a,
                                                   "isolated_more": list(big), "action_more": b}

    def s3():
        for r in results:
            if r.final_state == "degraded" and set(FRONT_SENSORS) <= set(r.isolated_sensors):
                a = shield(r).action
                yield a == "stop", {"isolated": r.isolated_sensors, "action": a}

    def s4():
        for r in results:
            a = shield(r).action
            want = {"healthy": "continue", "unresolved": "stop"}.get(r.final_state)
            if want:
                yield a == want, {"state": r.final_state, "isolated": r.isolated_sensors, "action": a}

    def s5():
        for act in ACTIONS:
            for ttc in TTC_VALUES:
                out = ttc_override(Action(act, "monitor"), ttc).action
                want = "stop" if ttc is not None and ttc < 2.0 else act
                yield out == want, {"action": act, "ttc": ttc, "result": out}

    def s6a():  # false alarm: a clean frame with any outcome other than healthy
        lab = FaultLabel.none()
        for r in results:
            a = shield(r).action
            yield a in acceptable_actions(lab), _cex(lab, r, a)

    def s6b():  # missed detection: a fault frame reported healthy
        r = RecoveryResult([], "healthy", [], None)
        for lab in labels[1:]:
            a = shield(r).action
            yield a in acceptable_actions(lab), _cex(lab, r, a)

    def s7():
        for lab in labels[1:]:
            for s in SENSORS_ALL + ("agent",):
                r = RecoveryResult([], "degraded", [s], None)
                a = shield(r).action
                yield a in acceptable_actions(lab), _cex(lab, r, a)

    return [
        _check("S1", "Assume-guarantee: under a single fault (A1), if the monitor detects it and isolates the right "
                     "sensor (a clean frame stays healthy), the shield's action is acceptable.", s1()),
        _check("S2", "Monotone: for the same state, isolating more sensors never gives a less cautious action.", s2()),
        _check("S3", "No front perception: degraded with CAM_FRONT and LiDAR both isolated gives stop.", s3()),
        _check("S4", "State rules: unresolved gives stop; healthy gives continue.", s4()),
        _check("S5", "TTC dominance: TTC < 2 s gives stop whatever the monitor chose; otherwise the action is "
                     "unchanged.", s5()),
        _check("S6a", "Detection is necessary (false alarm): on a clean frame, any outcome other than healthy is "
                      "acceptable. Expected to fail.", s6a()),
        _check("S6b", "Detection is necessary (missed fault): on a fault frame, a healthy outcome is acceptable. "
                      "Expected to fail.", s6b()),
        _check("S7", "Misattribution tolerance: on any fault frame, isolating exactly one sensor, even the wrong one, "
                     "gives an acceptable action.", s7()),
    ]
