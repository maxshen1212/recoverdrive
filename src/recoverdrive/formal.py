"""Formal verification of the rule-based safety logic by exhaustive enumeration.

The shield and the recovery controller have finite input domains once the sensors are fixed, so every property below
is checked by running the real functions (`shield.shield`, `shield.ttc_override`, `RecoveryController.step`) on every
case. There is no separate model of the code, so nothing can drift between the model and the implementation. A
property either holds on all cases or comes back with the first counterexample.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from itertools import chain, combinations

from recoverdrive.faults import AGENT_FAULTS, FIXED_HIGH, SENSOR_FAULTS, SEVERITIES
from recoverdrive.recovery import RecoveryConfig, RecoveryController
from recoverdrive.requirements import NUSCENES_SENSORS
from recoverdrive.shield import ACTIONS, FRONT_SENSORS, acceptable_actions, shield, ttc_override
from recoverdrive.types import Action, Diagnosis, FaultLabel, RecoveryResult

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
    """Every single-fault label (assumption AS1): each fault type on each sensor it can hit, at each severity."""
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
        _check("SH1", "Assume-guarantee: under a single fault (AS1), if the monitor detects it and isolates the right "
                     "sensor (a clean frame stays healthy), the shield's action is acceptable.", s1()),
        _check("SH2", "Monotone: for the same state, isolating more sensors never gives a less cautious action.", s2()),
        _check("SH3", "No front perception: degraded with CAM_FRONT and LiDAR both isolated gives stop.", s3()),
        _check("SH4", "State rules: unresolved gives stop; healthy gives continue.", s4()),
        _check("SH5", "TTC dominance: TTC < 2 s gives stop whatever the monitor chose; otherwise the action is "
                     "unchanged.", s5()),
        _check("SH6a", "Detection is necessary (false alarm): on a clean frame, any outcome other than healthy is "
                      "acceptable. Expected to fail.", s6a()),
        _check("SH6b", "Detection is necessary (missed fault): on a fault frame, a healthy outcome is acceptable. "
                      "Expected to fail.", s6b()),
        _check("SH7", "Misattribution tolerance: on any fault frame, isolating exactly one sensor, even the wrong one, "
                     "gives an acceptable action.", s7()),
    ]


# --- recovery controller: explicit-state model checking -------------------------------------------------------------

# Two front and two non-front sensors. The controller never looks at sensor names (only at the sets it is given) and
# the shield only checks membership in FRONT_SENSORS, so these four cover every role combination of the 7 sensors.
DEFAULT_CONTROLLER_SENSORS = ("camera:CAM_FRONT", "lidar", "camera:CAM_BACK", "camera:CAM_FRONT_LEFT")


@dataclass
class ControllerGraph:
    """Reachable states and every transition. A state is (sorted isolated items, sorted suspected_prev). An input is
    (kind, sensor, conf, flagged) with kind in none (the agent failed) / ok / uncertain / fault. An edge is
    (src, input, dst, final_state, isolated_sensors, isolated_agent_step)."""
    states: list[tuple]
    edges: list[tuple]


def _state(rc: RecoveryController) -> tuple:
    return tuple(sorted(rc.isolated.items())), tuple(sorted(rc.suspected_prev))


def _inputs(sensors: tuple[str, ...], confs: tuple[float, ...]) -> list[tuple]:
    flagged_sets = subsets(sensors)
    diags = [("none", "none", 0.0)] + [("ok", "none", c) for c in confs]
    diags += [(k, s, c) for k in ("uncertain", "fault") for s in sensors for c in confs]
    return [(k, s, c, fl) for k, s, c in diags for fl in flagged_sets]


def _step(cfg: RecoveryConfig, state: tuple, inp: tuple) -> tuple[RecoveryController, RecoveryResult]:
    rc = RecoveryController(cfg, isolated=dict(state[0]), suspected_prev=set(state[1]))
    kind, sensor, conf, flagged = inp
    diag = None if kind == "none" else Diagnosis(kind, sensor, conf)
    return rc, rc.step(diag, kind != "none", set(flagged), [])


def explore_controller(sensors: tuple[str, ...], cfg: RecoveryConfig, confs: tuple[float, ...] = (0.5, 1.0),
                       max_states: int = 100_000) -> ControllerGraph:
    """Breadth-first search over every state the real controller can reach from its initial state."""
    inputs = _inputs(sensors, confs)
    start = ((), ())
    seen, queue, edges = {start}, deque([start]), []
    while queue:
        src = queue.popleft()
        for inp in inputs:
            rc, res = _step(cfg, src, inp)
            dst = _state(rc)
            edges.append((src, inp, dst, res.final_state, tuple(res.isolated_sensors),
                          "isolate:agent" in res.steps_taken))
            if dst not in seen:
                if len(seen) >= max_states:
                    raise RuntimeError(f"more than {max_states} states")
                seen.add(dst)
                queue.append(dst)
    return ControllerGraph(sorted(seen), edges)


def verify_controller(graph: ControllerGraph, cfg: RecoveryConfig) -> list[PropertyResult]:
    tau = cfg.conf_threshold

    def cex(e) -> dict:
        src, inp, _dst, state, iso, _ = e
        return {"from_isolated": dict(src[0]), "from_suspected": list(src[1]),
                "input": {"diagnosis": inp[0], "sensor": inp[1], "conf": inp[2], "flagged": list(inp[3])},
                "final_state": state, "isolated_sensors": list(iso)}

    def c1():
        for e in graph.edges:
            kind, s, conf, _ = e[1]
            if kind == "fault" and conf >= tau:
                yield s in e[4], cex(e)

    def c2():
        for e in graph.edges:
            for s in set(e[0][1]) & set(e[2][1]):  # suspected before and in this step
                yield s in e[4], cex(e)

    def c3():
        for e in graph.edges:
            for s, _ in e[0][0]:
                if s not in e[1][3] and s not in e[2][1]:  # not flagged and not suspected in this step
                    yield s not in e[4], cex(e)

    def c4():
        for e in graph.edges:
            if e[1][0] == "none":
                yield "agent" in e[4] and e[5], cex(e)

    def c5():
        for e in graph.edges:
            if e[3] == "degraded":
                yield set(e[1][3]) <= set(e[4]) and all(v >= tau for _, v in e[2][0]), cex(e)

    def c6():
        for e in graph.edges:
            if e[3] == "healthy":
                yield e[4] == (), cex(e)

    return [
        _check("RC1", "Confident isolation: a fault diagnosis on s with confidence >= tau isolates s in that step.",
               c1()),
        _check("RC2", "Persistence: a sensor suspected in two consecutive steps is isolated.", c2()),
        _check("RC3", "Restoration: an isolated sensor that is neither flagged nor suspected is restored.", c3()),
        _check("RC4", "Agent fallback: when the VLM gives no valid answer, the agent is isolated and the deterministic "
                     "flags decide.", c4()),
        _check("RC5", "Verify soundness: degraded means every flagged sensor is isolated and every isolation has "
                     "confidence >= tau.", c5()),
        _check("RC6", "Healthy soundness: healthy means nothing is isolated.", c6()),
    ]
