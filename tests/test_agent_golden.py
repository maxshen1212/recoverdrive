"""Golden values for the agent-fault corruptions (A1 missing_field, A2 flip), captured before the PRAM-R baseline
changed agent.py and faults.py (plan 2026-11-22). They pin the behaviour every logged run used."""

import json

from fakes import GOOD, FakeOpenAI

from recoverdrive.agent import AgentCaller
from recoverdrive.faults import AgentFaultState
from recoverdrive.vlm import VLMClient, diagnosis_schema, ours_schema, textchain_schema

SENSORS = [f"camera:{c}" for c in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
                                   "CAM_BACK_RIGHT")] + ["lidar"]
CASES = [(0, 0, "camera:CAM_BACK_RIGHT"), (1, 0, "camera:CAM_BACK"), (7, 1, "lidar"), (42, 2, "camera:CAM_FRONT_LEFT")]


def _a2(schema, reply, seed, idx):
    caller = AgentCaller(VLMClient("http://x", "m", client=FakeOpenAI(reply)))
    caller.new_frame(AgentFaultState("A2", "low", {"n_affected": 3, "seed": seed}))
    return caller.call([{"role": "user", "content": "x"}], schema, attempt_idx=idx).raw


def test_a2_golden_ours_schema():
    for seed, idx, sensor in CASES:
        assert _a2(ours_schema(SENSORS), GOOD, seed, idx) == json.dumps(
            {"verdict": f"fault:{sensor}", "fault_type": "other", "confidence": 0.9, "evidence": "all consistent"})
    fault = {**GOOD, "verdict": "fault:lidar", "fault_type": "point_dropout"}
    assert _a2(ours_schema(SENSORS), fault, 0, 0) == json.dumps(GOOD)


def test_a2_golden_textchain_schema():
    reply = {"reasoning": "r", "verdict": "ok", "fault_type": "none", "confidence": 0.9}
    for seed, idx, sensor in CASES:
        assert _a2(textchain_schema("cot", SENSORS), reply, seed, idx) == json.dumps(
            {"reasoning": "r", "verdict": f"fault:{sensor}", "fault_type": "other", "confidence": 0.9})


def test_a2_golden_diagnosis_schema():
    for seed, idx, sensor in CASES:
        assert _a2(diagnosis_schema(SENSORS), {"status": "ok", "sensor": "none", "confidence": 0.9}, seed, idx) == \
            json.dumps({"status": "fault", "sensor": sensor, "confidence": 0.9})
    assert _a2(diagnosis_schema(SENSORS), {"status": "fault", "sensor": "lidar", "confidence": 0.8}, 0, 0) == \
        json.dumps({"status": "ok", "sensor": "none", "confidence": 0.9})


def test_a1_missing_field_golden():
    st = AgentFaultState("A1", "medium", {"n_affected": 2, "kind": "missing_field"})
    assert st.corrupt_output(json.dumps(GOOD)) == json.dumps({"fault_type": "none", "evidence": "all consistent"})
    assert st.corrupt_output(json.dumps({"status": "ok", "sensor": "none", "confidence": 0.9})) == "{}"
