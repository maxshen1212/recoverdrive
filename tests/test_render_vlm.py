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
