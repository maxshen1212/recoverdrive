import json

from fakes import GOOD, FakeOpenAI, lidar_object, make_ctx

from recoverdrive.faults import AgentFaultState
from recoverdrive.methods import B0, Ours, TextChain, build_methods, lidar_text
from recoverdrive.vlm import VLMClient, textchain_schema

SENSORS = ["camera:CAM_FRONT", "lidar"]


def client(reply=GOOD):
    return VLMClient("http://x", "m", client=FakeOpenAI(reply))


def test_b0_always_continue():
    out = B0().step(make_ctx(flags=["lidar:low_points"]))["B0"]
    assert out.action.action == "continue" and not out.alarm and out.recovery.final_state == "healthy"


def test_textchain_schema_fields_order():
    assert list(textchain_schema("zero_shot", SENSORS)["properties"])[:1] == ["verdict"]
    assert list(textchain_schema("cot", SENSORS)["properties"])[:2] == ["reasoning", "verdict"]
    assert list(textchain_schema("self_refine", SENSORS)["properties"])[:3] == ["initial_analysis", "critique",
                                                                                "refined_analysis"]


def test_textchain_maps_invalid_to_unresolved():
    def reply(req):
        name = req["response_format"]["json_schema"]["name"]
        return {"description": "a car ahead"} if name == "caption" else "not json"

    out = TextChain("B2", client(reply)).step(make_ctx())["B2"]
    assert not out.valid and out.alarm and out.pred_sensor == "agent"
    assert out.recovery.final_state == "unresolved" and out.action.action == "stop"


def test_textchain_fault_verdict_and_a1():
    def reply(req):
        name = req["response_format"]["json_schema"]["name"]
        if name == "caption":
            return {"description": "a car ahead"}
        return {"verdict": "fault:lidar", "fault_type": "point_dropout", "confidence": 0.8}

    tc = TextChain("B1", client(reply))
    out = tc.step(make_ctx())["B1"]
    assert out.alarm and out.pred_sensor == "lidar" and out.action.action == "stop"
    a1 = AgentFaultState("A1", "low", {"n_affected": 1, "kind": "garbage"})
    assert not tc.step(make_ctx(agent_fault=a1))["B1"].valid  # captions are not hit, the diagnosis is


def test_lidar_text_format():
    text = lidar_text(make_ctx(objects=[lidar_object(10, 0, 12), lidar_object(-5, 2, 40, "pedestrian")]))
    assert text.startswith("2 objects within 50 m: pedestrian at (-5.0, 2.0) m with 40 points; car at (10.0, 0.0)")
    assert "Total LiDAR points this sweep: 0." in text


def test_ours_full_reask_recovers_a1():
    ours = Ours(client(), {"recovery": {}})
    a1 = AgentFaultState("A1", "low", {"n_affected": 1, "kind": "truncate"})
    outs = ours.step(make_ctx(agent_fault=a1, label=None))
    assert outs["ours_detect"].alarm and outs["ours_detect"].action.action == "stop"
    full = outs["ours_full"]
    assert full.valid and full.recovery.final_state == "healthy" and full.extra["agent_recovered"]
    assert full.recovery.steps_taken == ["reask"] and full.n_calls == 2
    assert full.alarm and full.pred_sensor == "agent"  # E1: the agent fault was detected (and recovered)


def test_ours_full_resamples_and_isolates():
    fault = {"verdict": "fault:lidar", "fault_type": "point_dropout", "confidence": 0.9, "evidence": "e"}
    ours = Ours(client(fault), {"recovery": {}})
    full = ours.step(make_ctx(flags=["lidar:low_points"]))["ours_full"]
    assert "resample" in full.recovery.steps_taken and full.n_calls == 4
    assert full.recovery.isolated_sensors == ["lidar"] and full.action.action == "slow_down"
    assert json.loads(full.extra["samples"][0]["raw"])["verdict"] == "fault:lidar"


def test_build_methods():
    names = [m.names for m in build_methods(["B0", "B3", "ours_full"], client(), {"recovery": {}})]
    assert names == [("B0",), ("B3",), ("ours_full",)]


def test_gate_skips_unflagged_frames():
    fake = FakeOpenAI({"verdict": "fault:lidar", "fault_type": "point_dropout", "confidence": 0.9, "evidence": "e"})
    ours = Ours(VLMClient("http://x", "m", client=fake), {"recovery": {"gate": "flags"}})
    outs = ours.step(make_ctx())
    assert len(fake.calls) == 0 and outs["ours_full"].action.action == "continue" and outs["ours_full"].n_calls == 0
    assert not outs["ours_detect"].alarm
    full = ours.step(make_ctx(flags=["lidar:low_points"]))["ours_full"]
    assert len(fake.calls) == 4 and full.recovery.isolated_sensors == ["lidar"]


def test_no_sampling_uses_first_answer():
    fault = {"verdict": "fault:lidar", "fault_type": "point_dropout", "confidence": 0.9, "evidence": "e"}
    ours = Ours(client(fault), {"recovery": {"sampling": False}})
    full = ours.step(make_ctx(flags=["lidar:low_points"]))["ours_full"]
    assert full.n_calls == 1 and "resample" not in full.recovery.steps_taken
    assert full.recovery.isolated_sensors == ["lidar"]
