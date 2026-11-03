"""AgentCaller: the diagnosis-call wrapper shared by all methods.

It counts diagnosis attempts within a frame, applies the frame's agent fault (A1-A3) to the affected attempts,
and turns any exception from the model server into an invalid attempt, so one bad call never stops a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from recoverdrive.faults import AgentFaultState
from recoverdrive.vlm import VLMClient, validate_json


@dataclass
class Attempt:
    raw: str
    data: dict | None  # validated output, None when invalid
    valid: bool
    timeout: bool
    latency_s: float
    cached: bool = False
    clean_data: dict | None = None  # the model's real answer before agent-fault corruption
    error: str | None = None
    corrupted: bool = False


def flip_verdict(data: dict, sensors: list[str], rng: np.random.Generator) -> dict:
    """A2: a well-formed but wrong answer (ok <-> fault)."""
    out = dict(data)
    if "verdict" in data:
        healthy = data["verdict"] == "ok"
        out["verdict"] = f"fault:{rng.choice(sensors)}" if healthy else "ok"
        out["confidence"] = 0.9
        if "fault_type" in out:
            out["fault_type"] = "other" if healthy else "none"
        return out
    if data.get("status") == "ok":
        out.update(status="fault", sensor=str(rng.choice(sensors)), confidence=0.9)
        if "fault_type" in out:
            out["fault_type"] = "other"
    else:
        out.update(status="ok", sensor="none", confidence=0.9)
        if "fault_type" in out:
            out["fault_type"] = "none"
    return out


class AgentCaller:
    def __init__(self, client: VLMClient):
        self.client = client
        self.fault: AgentFaultState | None = None
        self.attempt = 0
        self.n_calls = 0

    def new_frame(self, fault: AgentFaultState | None) -> None:
        self.fault = fault
        self.attempt = 0

    def call(self, messages: list[dict], schema: dict, temperature: float = 0.0, seed: int | None = None,
             max_tokens: int | None = None, diagnosis: bool = True, name: str = "output") -> Attempt:
        """One model call. Only diagnosis calls count as attempts and can be hit by agent faults."""
        idx = self.attempt
        if diagnosis:
            self.attempt += 1
        self.n_calls += 1
        try:
            res = self.client.chat_json(messages, schema, temperature=temperature, seed=seed, max_tokens=max_tokens,
                                        name=name)
        except Exception as exc:  # noqa: BLE001 - server errors become invalid attempts
            timeout = "timeout" in type(exc).__name__.lower() or isinstance(exc, TimeoutError)
            return Attempt("", None, False, timeout, self.client.timeout_s if timeout else 0.0, error=repr(exc))
        clean = validate_json(res.raw, schema)
        raw, corrupted = res.raw, False
        if diagnosis and self.fault is not None and self.fault.affects(idx):
            corrupted = True
            fid = self.fault.fault_id
            if fid == "A3":
                return Attempt("", None, False, True, self.client.timeout_s, res.cached, clean,
                               error="simulated timeout (A3)", corrupted=True)
            if fid == "A1":
                raw = self.fault.corrupt_output(raw)
            elif fid == "A2" and clean is not None:
                props = schema["properties"]
                sensors = ([v[len("fault:"):] for v in props["verdict"]["enum"] if v.startswith("fault:")]
                           if "verdict" in props else [s for s in props["sensor"]["enum"] if s != "none"])
                rng = np.random.default_rng([self.fault.params.get("seed", 0), idx])
                raw = json.dumps(flip_verdict(clean, sensors, rng))
        data = validate_json(raw, schema)
        return Attempt(raw, data, data is not None, False, res.latency_s, res.cached, clean, corrupted=corrupted)
