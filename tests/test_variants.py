"""VLM-centric variants (plan 2026-11-21): message content, stage-1 sharing, agent faults on the deciding call."""

import json

from fakes import FakeOpenAI, make_ctx

from recoverdrive.faults import AgentFaultState
from recoverdrive.methods import VLMVariant, build_methods
from recoverdrive.vlm import VLMClient

OK = {"analysis": "all views plausible", "verdict": "ok", "fault_type": "none", "confidence": 0.9, "evidence": "e"}
LIDAR = {"analysis": "empty cloud", "verdict": "fault:lidar", "fault_type": "point_dropout", "confidence": 0.9,
         "evidence": "e"}
CFG = {"recovery": {}, "data": {"nominal_lidar_points": 34000}}


def schema_name(req: dict) -> str:
    return req["response_format"]["json_schema"]["name"]


def text_of(req: dict) -> str:
    return " ".join(p["text"] for m in req["messages"] if isinstance(m["content"], list)
                    for p in m["content"] if p.get("type") == "text")


def replier(decide: dict):
    def reply(req):
        name = schema_name(req)
        if name == "observe":
            return {"CAM_FRONT": "a sharp road view", "LIDAR_TOP": "dense cloud all around"}
        if name == "observe_sensor":
            return {"observation": "looks normal"}
        return decide
    return reply


def variant(name: str, decide: dict = OK) -> tuple[VLMVariant, FakeOpenAI]:
    fake = FakeOpenAI(replier(decide))
    return VLMVariant(name, VLMClient("http://x", "m", client=fake), CFG, (f"{name}_detect", f"{name}_full")), fake


def test_think_reasons_first_and_sees_the_report():
    m, fake = variant("think")
    outs = m.step(make_ctx())
    assert set(outs) == {"think_detect", "think_full"} and not outs["think_full"].alarm
    req = fake.calls[0]
    assert next(iter(req["response_format"]["json_schema"]["schema"]["properties"])) == "analysis"
    assert "Consistency report:\nsummary" in text_of(req) and "flags can be wrong" in text_of(req)
    m2, fake2 = variant("think_nochecks")
    m2.step(make_ctx(flags=["lidar:low_points"]))
    assert "Consistency report" not in text_of(fake2.calls[0]) and len(fake2.calls) == 1  # flags ignored: no vote


def test_perceive_stage1_once_per_frame_and_notes_reach_the_decision():
    m, fake = variant("perceive", LIDAR)
    full = m.step(make_ctx(flags=["lidar:low_points"]))["perceive_full"]
    names = [schema_name(r) for r in fake.calls]
    assert names.count("observe") == 1 and names.count("diagnosis") == 4  # first answer + 3 samples share the notes
    assert full.n_calls == 5 and len(full.extra["stage1"]) == 1
    assert "- CAM_FRONT: a sharp road view" in text_of(fake.calls[-1])
    assert full.recovery.isolated_sensors == ["lidar"]


def test_percam_one_call_per_sensor_then_one_decision():
    m, fake = variant("percam")
    full = m.step(make_ctx())["percam_full"]
    names = sorted(schema_name(r) for r in fake.calls)
    assert names == ["diagnosis", "observe_sensor", "observe_sensor"]  # CAM_FRONT, LIDAR_TOP, decision
    assert full.n_calls == 3
    decide = next(r for r in fake.calls if schema_name(r) == "diagnosis")
    assert "- LIDAR_TOP: looks normal" in text_of(decide) and "Image 2" not in text_of(decide)


def test_agent_fault_hits_only_the_deciding_call():
    m, _fake = variant("perceive")
    a1 = AgentFaultState("A1", "low", {"n_affected": 1, "kind": "truncate"})
    outs = m.step(make_ctx(agent_fault=a1))
    full = outs["perceive_full"]
    assert full.extra["stage1"][0]["valid"] and not full.extra["stage1"][0]["corrupted"]
    assert full.recovery.steps_taken == ["reask"] and full.extra["agent_recovered"]
    assert json.loads(full.extra["attempts"][-1]["raw"])["verdict"] == "ok"


def test_build_methods_variants():
    names = [m.names for m in build_methods(["ours_full", "think_full", "percam_detect", "percam_full"], None, CFG)]
    assert names == [("ours_full",), ("think_full",), ("percam_detect", "percam_full")]


def test_think_trust_keeps_the_frozen_guide_and_adds_reasoning():
    m, fake = variant("think_trust")
    m.step(make_ctx())
    text = text_of(fake.calls[0])
    assert "The deterministic checks are reliable" in text and "flags can be wrong" not in text
    assert "write your step-by-step reasoning in the field analysis" in text
    assert next(iter(fake.calls[0]["response_format"]["json_schema"]["schema"]["properties"])) == "analysis"
