import json

import numpy as np
from fakes import GOOD, FakeOpenAI

from recoverdrive.agent import AgentCaller
from recoverdrive.cache import KVCache
from recoverdrive.faults import AgentFaultState
from recoverdrive.render import camera_mosaic, lidar_bev
from recoverdrive.vlm import VLMClient, image_part, ours_messages, ours_schema, to_diagnosis, validate_json

SENSORS = ["camera:CAM_FRONT", "camera:CAM_BACK", "lidar"]
def test_render_mosaic_shape():
    names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    imgs = {n: np.full((900, 1600, 3), i * 40, np.uint8) for i, n in enumerate(names)}
    m = camera_mosaic(imgs)
    assert m.shape == (540, 1440, 3) and m.dtype == np.uint8


def test_render_bev_empty():
    bev = lidar_bev(np.zeros((0, 4)))
    assert bev.shape == (448, 448, 3)
    full = lidar_bev(np.array([[10.0, 0, 0, 1], [0, -10, 0, 1], [200, 0, 0, 1]]))
    assert full.shape == (448, 448, 3) and full.sum() > bev.sum()


def test_validate_json_enums_and_ranges():
    schema = ours_schema(SENSORS)
    assert validate_json(json.dumps(GOOD), schema) == GOOD
    assert validate_json("not json", schema) is None
    assert validate_json(json.dumps({**GOOD, "verdict": "fault:camera:CAM_LEFT"}), schema) is None
    assert validate_json(json.dumps({**GOOD, "confidence": 1.5}), schema) is None
    assert validate_json(json.dumps({**GOOD, "confidence": True}), schema) is None
    assert validate_json(json.dumps({**GOOD, "extra": 1}), schema) is None
    assert validate_json(json.dumps({k: v for k, v in GOOD.items() if k != "fault_type"}), schema) is None
    d = to_diagnosis(validate_json(json.dumps(GOOD), schema))
    assert d.status == "ok" and d.sensor == "none" and d.fault_type == "none" and d.evidence == "all consistent"
    f = to_diagnosis({**GOOD, "verdict": "fault:camera:CAM_FRONT"})
    assert f.status == "fault" and f.sensor == "camera:CAM_FRONT"
    assert next(iter(schema["properties"])) == "verdict"


def _messages(seed_px=0):
    img = np.full((10, 10, 3), seed_px, np.uint8)
    return ours_messages("summary", image_part(img), None, image_part(img))


def test_chat_json_uses_cache(tmp_path):
    fake = FakeOpenAI()
    client = VLMClient("http://x", "m", cache=KVCache(tmp_path / "c.sqlite"), client=fake)
    schema = ours_schema(SENSORS)
    a = client.chat_json(_messages(), schema)
    b = client.chat_json(_messages(), schema)
    assert len(fake.calls) == 1 and not a.cached and b.cached and a.raw == b.raw
    sent = fake.calls[0]["messages"][1]["content"]
    assert any(p.get("type") == "image_url" and p["image_url"]["url"].startswith("data:image/jpeg") for p in sent)
    client.chat_json(_messages(seed_px=5), schema)  # different image -> new call
    client.chat_json(_messages(), schema, temperature=0.7, seed=1)
    client.chat_json(_messages(), schema, temperature=0.7, seed=2)
    assert len(fake.calls) == 4


def test_caller_applies_a1():
    caller = AgentCaller(VLMClient("http://x", "m", client=FakeOpenAI()))
    caller.new_frame(AgentFaultState("A1", "low", {"n_affected": 1, "kind": "truncate"}))
    schema = ours_schema(SENSORS)
    first = caller.call(_messages(), schema)
    assert not first.valid and first.corrupted and first.clean_data == GOOD
    second = caller.call(_messages(), schema)
    assert second.valid and second.data == GOOD
    caller.new_frame(None)
    assert caller.call(_messages(), schema).valid


def test_caller_records_exception_as_invalid():
    caller = AgentCaller(VLMClient("http://x", "m", client=FakeOpenAI(exc=RuntimeError("server down"))))
    caller.new_frame(None)
    att = caller.call(_messages(), ours_schema(SENSORS))
    assert not att.valid and att.data is None and "server down" in att.error


def test_caller_applies_a2_flip():
    caller = AgentCaller(VLMClient("http://x", "m", client=FakeOpenAI()))
    caller.new_frame(AgentFaultState("A2", "low", {"n_affected": 1, "seed": 0}))
    att = caller.call(_messages(), ours_schema(SENSORS))
    assert att.valid and att.data["verdict"].startswith("fault:") and att.clean_data == GOOD


def test_chat_json_dedupes_concurrent_identical_requests(tmp_path):
    import threading
    import time

    calls = []

    def slow(req):
        calls.append(1)
        time.sleep(0.05)
        return GOOD

    client = VLMClient("http://x", "m", cache=KVCache(tmp_path / "c.sqlite"), client=FakeOpenAI(slow))
    schema = ours_schema(SENSORS)
    threads = [threading.Thread(target=client.chat_json, args=(_messages(), schema)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1


def test_caller_explicit_attempt_index():
    caller = AgentCaller(VLMClient("http://x", "m", client=FakeOpenAI()))
    caller.new_frame(AgentFaultState("A1", "medium", {"n_affected": 2, "kind": "garbage"}))
    schema = ours_schema(SENSORS)
    assert caller.call(_messages(), schema, attempt_idx=2).valid  # index 2 is past the affected attempts
    assert not caller.call(_messages(), schema, attempt_idx=1).valid
    assert caller.attempt == 3


def test_chat_json_spreads_requests_over_servers():
    a, b = FakeOpenAI(), FakeOpenAI()
    client = VLMClient(["http://a", "http://b"], "m")
    client.clients = [a, b]
    schema = ours_schema(SENSORS)
    for px in range(20):
        client.chat_json(_messages(seed_px=px), schema)
    assert len(a.calls) + len(b.calls) == 20 and a.calls and b.calls


def test_grounding_parse_and_rescale():
    from recoverdrive.detectors.qwen_grounding import grounding_schema, parse_grounding

    raw = json.dumps({"objects": [{"label": "car", "bbox_2d": [100, 200, 300, 400]},
                                  {"label": "tree", "bbox_2d": [0, 0, 10, 10]},
                                  {"label": "bus", "bbox_2d": [500, 500, 400, 600]}]})
    dets = parse_grounding(raw, 1600, 900, ["car", "bus"])
    assert len(dets) == 1 and dets[0].label == "car"
    np.testing.assert_allclose(dets[0].box, [160, 180, 480, 360])
    assert parse_grounding("nope", 1600, 900, ["car"]) is None
    assert grounding_schema(["car"])["properties"]["objects"]["items"]["properties"]["label"]["enum"] == ["car"]


def test_no_checks_messages_have_no_report():
    img = image_part(np.zeros((4, 4, 3), np.uint8))
    text = " ".join(p["text"] for p in ours_messages(None, img, None, img)[1]["content"] if p.get("type") == "text")
    assert "Consistency report" not in text and "no consistency report" in text
    with_report = " ".join(p["text"] for p in ours_messages("S", img, None, img)[1]["content"] if p.get("type") == "text")
    assert "Consistency report:\nS" in with_report
