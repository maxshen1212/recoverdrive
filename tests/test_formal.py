from recoverdrive.formal import verify_shield


def by_name(results):
    return {r.name: r for r in results}


def test_shield_properties_hold():
    res = by_name(verify_shield())
    for name in ("S1", "S2", "S3", "S4", "S5", "S7"):
        assert res[name].holds, (name, res[name].counterexample)
        assert res[name].n_cases > 0 and res[name].counterexample is None


def test_s6_without_correct_detection_has_counterexamples():
    res = by_name(verify_shield())
    false_alarm, missed = res["S6a"], res["S6b"]
    assert not false_alarm.holds and false_alarm.counterexample["label"]["fault_id"] == "none"
    assert not missed.holds and missed.counterexample["result"]["final_state"] == "healthy"
    assert missed.counterexample["action"] == "continue"
    assert {"label", "result", "action", "acceptable"} <= set(missed.counterexample)
