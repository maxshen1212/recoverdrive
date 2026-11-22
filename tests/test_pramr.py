"""PRAM-R-style baseline (plan 2026-11-22): indicators, prompt, hysteresis, agent faults, registration."""

import json

import numpy as np
from fakes import FakeOpenAI, make_ctx

from recoverdrive import pramr
from recoverdrive.faults import AgentFaultState
from recoverdrive.methods import PramR, build_methods, needs_vlm
from recoverdrive.vlm import VLMClient, validate_json

REF = {"CAM_FRONT": {"brightness": {"p5": 50.0, "median": 100.0, "p95": 150.0},
                     "contrast": {"p5": 20.0, "median": 40.0, "p95": 60.0},
                     "edge_density": {"p5": 0.02, "median": 0.06, "p95": 0.12}},
       "LIDAR_TOP": {"density": {"p5": 3.0, "median": 4.0, "p95": 5.0},
                     "noise_ratio": {"p5": 0.001, "median": 0.01, "p95": 0.03},
                     "reflectivity_ratio": {"p5": 0.8, "median": 1.0, "p95": 1.2},
                     "mean_intensity": {"p5": 15.0, "median": 20.0, "p95": 25.0}},
       "light": {"brightness": {"p5": 40.0, "median": 90.0, "p95": 140.0}}}


def answer(front=0.9, lidar=0.9):
    return {"CAM_FRONT_reliability": front, "LIDAR_TOP_reliability": lidar, "scene_complexity": 0.4,
            "CAM_FRONT_use": True, "LIDAR_TOP_use": True}


def method(replies, name="pramr_text_full", **kw):
    it = iter(replies)
    fake = FakeOpenAI(lambda req: next(it))
    return PramR(name, VLMClient("http://x", "m", client=fake), {}, reference=REF, **kw), fake


def text_of(req):
    return " ".join(p["text"] for m in req["messages"] if isinstance(m["content"], list)
                    for p in m["content"] if p.get("type") == "text")


def test_camera_and_lidar_indicators():
    assert pramr.camera_indicators(np.zeros((90, 160, 3), np.uint8)) == {"brightness": 0.0, "contrast": 0.0,
                                                                          "edge_density": 0.0}
    noisy = pramr.camera_indicators(np.random.default_rng(0).integers(0, 256, (90, 160, 3), dtype=np.uint8))
    assert noisy["edge_density"] > 0.5 and noisy["contrast"] > 40
    assert pramr.lidar_raw_indicators(np.zeros((0, 4))) == {"density": 0.0, "noise_ratio": 0.0, "mean_intensity": 0.0}
    ang = np.radians(np.arange(0, 360, 0.1))
    ring = np.c_[10 * np.cos(ang), 10 * np.sin(ang), np.zeros_like(ang), np.full_like(ang, 5.0)]
    lone = np.array([[30.0, 0.0, 0.0, 5.0]])
    raw = pramr.lidar_raw_indicators(np.vstack([ring, lone]))
    assert raw["noise_ratio"] == 1 / (len(ring) + 1) and raw["mean_intensity"] == 5.0
    assert raw["density"] == (len(ring) + 1) / pramr.N_CELLS
    half = pramr.lidar_raw_indicators(ring[np.degrees(ang) < 180])
    assert half["density"] < raw["density"] / 1.9  # a lost sector lowers the density


def test_labels_and_prompt_content():
    assert [pramr.label(v, REF["CAM_FRONT"]["brightness"]) for v in (10, 100, 200)] == ["low", "normal", "high"]
    m, fake = method([answer()])
    m.step(make_ctx())
    text = text_of(fake.calls[0])
    assert "- CAM_FRONT: brightness 0 (normal 50-150; illumination level low)" in text
    assert "- LIDAR_TOP: no points received" in text and "Memory: none, this is the first step." in text
    assert fake.calls[0]["response_format"]["json_schema"]["name"] == "pramr"
    assert not any(p.get("type") == "image_url" for p in fake.calls[0]["messages"][1]["content"])


def test_image_variant_sends_the_mosaic():
    m, fake = method([answer()], name="pramr_image_full")
    m.step(make_ctx())
    assert "Scene context: current frame, all cameras." in text_of(fake.calls[0])


def test_latch_boundaries_in_thousandths():
    on = {"CAM_FRONT": True, "LIDAR_TOP": True}
    # off only when r < theta - delta: 0.55 is exactly on the boundary and stays on (no float error at 0.6 - 0.05)
    assert pramr.latch_step(on, {"CAM_FRONT": 0.55, "LIDAR_TOP": 0.9}, 600, 600, 50)["CAM_FRONT"] is True
    assert pramr.latch_step(on, {"CAM_FRONT": 0.549, "LIDAR_TOP": 0.9}, 600, 600, 50)["CAM_FRONT"] is False
    assert pramr.latch_step(on, {"CAM_FRONT": 0.6, "LIDAR_TOP": 0.9}, 600, 600, 0)["CAM_FRONT"] is True
    off = {"CAM_FRONT": False, "LIDAR_TOP": True}
    assert pramr.latch_step(off, {"CAM_FRONT": 0.64, "LIDAR_TOP": 0.9}, 600, 600, 50)["CAM_FRONT"] is False
    assert pramr.latch_step(off, {"CAM_FRONT": 0.65, "LIDAR_TOP": 0.9}, 600, 600, 50)["CAM_FRONT"] is True
    assert pramr.latch_step(off, None, 600, 600, 50) == off
    assert pramr.latch_step(on, {"CAM_FRONT": 0.9, "LIDAR_TOP": 0.45}, 600, 500, 0)["LIDAR_TOP"] is False


def test_prompts_do_not_depend_on_thresholds():
    replies = [answer(0.2, 0.9), answer(0.7, 0.4), answer(0.9, 0.9)]
    a, fa = method(replies, theta_camera=300, theta_lidar=300, delta=0)
    b, fb = method(replies, theta_camera=800, theta_lidar=700, delta=100)
    for _ in range(3):
        a.step(make_ctx())
        b.step(make_ctx())
    assert [c["messages"] for c in fa.calls] == [c["messages"] for c in fb.calls]
    assert "Reliability scores you gave: CAM_FRONT 0.20, LIDAR_TOP 0.90." in text_of(fa.calls[1])


def test_isolation_blame_and_shield():
    m, _ = method([answer(0.2, 0.9), answer(0.3, 0.3)], theta_camera=500, theta_lidar=500, delta=0)
    first = m.step(make_ctx())["pramr_text_full"]
    assert first.alarm and first.pred_sensor == "camera:CAM_FRONT" and first.action.action == "slow_down"
    assert first.recovery.isolated_sensors == ["camera:CAM_FRONT"] and first.recovery.final_state == "degraded"
    both = m.step(make_ctx())["pramr_text_full"]  # equal scores: the first sensor in order is blamed
    assert both.pred_sensor == "camera:CAM_FRONT" and both.action.action == "stop"


def test_invalid_answer_keeps_the_state():
    m, _ = method([answer(0.2, 0.9), {"oops": 1}, answer(0.9, 0.9)])
    m.step(make_ctx())
    held = m.step(make_ctx())["pramr_text_full"]
    assert not held.valid and held.recovery.isolated_sensors == ["camera:CAM_FRONT"] and held.n_calls == 1
    back = m.step(make_ctx())["pramr_text_full"]
    assert back.valid and not back.alarm and back.action.action == "continue"


def test_agent_faults_on_the_pramr_schema():
    st = AgentFaultState("A1", "medium", {"n_affected": 1, "kind": "missing_field"})
    schema = pramr.pramr_schema(["CAM_FRONT"])
    assert validate_json(st.corrupt_output(json.dumps(answer())), schema) is None
    m, _ = method([answer(0.9, 0.9)])
    out = m.step(make_ctx(agent_fault=AgentFaultState("A2", "low", {"n_affected": 1, "seed": 0})))["pramr_text_full"]
    flipped = out.extra["scores"]
    assert out.extra["clean_scores"] == {"CAM_FRONT": 0.9, "LIDAR_TOP": 0.9}
    assert sorted(flipped.values()) == [0.1, 0.9] and out.alarm and out.extra["agent_recovered"] is False
    m2, _ = method([answer(0.9, 0.9)])
    ok = m2.step(make_ctx(agent_fault=AgentFaultState("A2", "low", {"n_affected": 0, "seed": 0})))["pramr_text_full"]
    assert ok.extra["agent_recovered"] is True and not ok.alarm


def test_registration():
    fake = VLMClient("http://x", "m", client=FakeOpenAI(answer()))
    names = [m.names for m in build_methods(["pramr_full", "pramr_image_full"], fake, {})]
    assert ("pramr_image_full",) in names and ("pramr_full",) in names
    assert needs_vlm(["pramr_full"], "none")
