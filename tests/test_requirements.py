import pytest
from fakes import carla_step, record

from recoverdrive.requirements import carla_trace, evaluate, nuscenes_trace, spec
from recoverdrive.stl import check_always

FRONT = "camera:CAM_FRONT"
LIDAR = ("lidar",)


def trace_of(recs, sensors=LIDAR):
    return nuscenes_trace(recs, "m", sensors)[0]


def test_r1_missed_front_fault():
    recs = [record(0), record(1, "C1", FRONT, "high", 0), record(2, "C1", FRONT, "high", 0), record(3)]
    res = check_always(spec("R1"), trace_of(recs))
    assert not res.satisfied and res.worst_t == 1


def test_r1_grace_allows_one_late_frame():
    recs = [record(0), record(1, "C1", FRONT, "high", 0, action="continue"),
            record(2, "C1", FRONT, "high", 0, action="slow_down", state="degraded", iso=(FRONT,)), record(3)]
    tr = trace_of(recs)
    assert check_always(spec("R1", grace=1), tr).satisfied
    assert not check_always(spec("R1", grace=0), tr).satisfied


def test_r2_unnecessary_stop_on_clean():
    res = check_always(spec("R2"), trace_of([record(0), record(1, action="stop", state="unresolved")]))
    assert res.robustness == pytest.approx(-1.5) and res.worst_t == 1


def test_r3_recovery_window():
    def run(actions):
        recs = [record(0, "L1", "lidar", "medium", 0, action="stop", state="unresolved")]
        recs += [record(i + 1, action=a, state="healthy" if a == "continue" else "unresolved")
                 for i, a in enumerate(actions)]
        return check_always(spec("R3"), trace_of(recs))
    assert not run(["stop", "stop", "stop", "continue"]).satisfied
    assert run(["stop", "stop", "continue", "continue"]).satisfied


def test_r6_restore_while_faulty():
    faulty = [record(0, "L1", "lidar", "medium", 0, action="slow_down", state="degraded", iso=("lidar",)),
              record(1, "L1", "lidar", "medium", 0, action="stop", state="unresolved")]
    healed = [faulty[0], record(1)]
    assert not check_always(spec("R6", LIDAR), trace_of(faulty)).satisfied
    assert check_always(spec("R6", LIDAR), trace_of(healed)).satisfied


def test_r6_needs_sensors():
    with pytest.raises(ValueError):
        spec("R6", ())


def test_trace_skips_error_records():
    recs = [record(0), {"scene": "s", "seed": 0, "run_fault": "L1", "frame_idx": 1, "error": "boom"}, record(2)]
    tr, skipped = nuscenes_trace(recs, "m", LIDAR)
    assert skipped == 1 and len(tr["a"]) == 2
    tr, skipped = nuscenes_trace(recs, "other_method", LIDAR)
    assert skipped == 3 and len(tr["a"]) == 0


def test_carla_trace_ttc_none():
    tr, _ = carla_trace([carla_step(0, ttc=None), carla_step(1, ttc=1.5)], {"collision": False})
    assert list(tr["ttc_low"]) == [0, 1]
    assert list(tr["gap"]) == [1, 1]
    tr, _ = carla_trace([carla_step(0), carla_step(1)], {"collision": True})
    assert list(tr["gap"]) == [1, -1]


def test_r4_front_onset_slowdown():
    def run(slow_at):
        steps = [carla_step(0)]
        for t in range(1, 10):
            speed = 2.8 if slow_at is not None and t >= slow_at else 8.3
            steps.append(carla_step(t, "C1", FRONT, "high", "slow_down", ("slow_down", "stop"), speed=speed))
        tr, _ = carla_trace(steps, {"collision": False})
        return check_always(spec("R4"), tr)
    assert run(5).satisfied
    assert not run(None).satisfied


def test_carla_ttc_brake_is_not_an_unnecessary_fallback():
    # the vehicle's own TTC brake overrides the monitor; R1/R2 judge the monitor, R5 the final action
    steps = [carla_step(0, ttc=1.5, monitor_action="continue", action="stop"),
             carla_step(1, ttc=1.0, monitor_action="continue", action="stop"), carla_step(2)]
    tr, _ = carla_trace(steps, {"collision": False})
    assert check_always(spec("R2"), tr).satisfied
    assert check_always(spec("R5"), tr).satisfied


def test_r5_ttc_brake_and_collision():
    ok = [carla_step(0, ttc=1.5), carla_step(1, action="stop", acceptable=("continue",)), carla_step(2)]
    late = [carla_step(0, ttc=1.5), carla_step(1), carla_step(2)]
    assert check_always(spec("R5"), carla_trace(ok, {"collision": False})[0]).satisfied
    assert not check_always(spec("R5"), carla_trace(late, {"collision": False})[0]).satisfied
    assert not check_always(spec("R5"), carla_trace(ok, {"collision": True})[0]).satisfied


def test_r1_grace0_matches_missed_fallback():
    from recoverdrive.shield import acceptable_actions, classify_action
    from recoverdrive.types import FaultLabel
    recs = [record(0), record(1, "C1", FRONT, "high", 0), record(2, "C1", FRONT, "high", 0, action="stop"),
            record(3, "L1", "lidar", "low", 1), record(4, "L2", "lidar", "high", 2, action="continue"),
            record(5, "C3", "camera:CAM_BACK", "high", 3, action="stop"), record(6, action="slow_down"),
            record(7, "A1", "agent", "medium", 4)]
    missed = sum(classify_action(r["methods"]["m"]["action"]["action"],
                                 acceptable_actions(FaultLabel(**r["label"]))) == "missed" for r in recs)
    res = check_always(spec("R1", grace=0), trace_of(recs))
    assert missed == 2 and res.n_violations == missed and res.n_obligations == len(recs)


def test_series_matches_check_always():
    from recoverdrive.stl import series
    tr = trace_of([record(0), record(1, "C1", FRONT, "high", 0), record(2)])
    vals = series(spec("R1", grace=0), tr)
    assert len(vals) == 3 and vals.min() == check_always(spec("R1", grace=0), tr).robustness


def test_evaluate_returns_each_spec():
    out = evaluate(trace_of([record(0), record(1), record(2), record(3)]), ("R1", "R2", "R3", "R6"), LIDAR)
    assert set(out) == {"R1", "R2", "R3", "R6"} and all(r.satisfied for r in out.values())
