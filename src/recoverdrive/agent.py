"""AgentCaller: the diagnosis-call wrapper shared by all methods.

It counts diagnosis attempts within a frame, applies the frame's agent fault (A1-A3) to the affected attempts,
and turns any exception from the model server into an invalid attempt, so one bad call never stops a run.
"""

from __future__ import annotations

import json
import threading
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


def flip_reliability(data: dict, rng: np.random.Generator) -> dict:
    """A2 for per-sensor reliability answers (PRAM-R baseline): one wrong answer, as flip_verdict does for verdicts.
    If every sensor looks reliable (r >= 0.5), one random sensor gets 1 - r; otherwise every sensor below 0.5 does."""
    out = dict(data)
    keys = [k for k in data if k.endswith("_reliability")]
    low = [k for k in keys if data[k] < 0.5]
    for k in low or [str(rng.choice(keys))]:
        out[k] = round(1.0 - float(data[k]), 3)
    return out


class AgentCaller:
    def __init__(self, client: VLMClient):
        self.client = client
        self.fault: AgentFaultState | None = None
        self.attempt = 0
        self.n_calls = 0
        self._lock = threading.Lock()

    def new_frame(self, fault: AgentFaultState | None) -> None:
        self.fault = fault
        self.attempt = 0

    def call(self, messages: list[dict], schema: dict, temperature: float = 0.0, seed: int | None = None,
             max_tokens: int | None = None, diagnosis: bool = True, name: str = "output",
             attempt_idx: int | None = None) -> Attempt:
        """One model call. Only diagnosis calls count as attempts and can be hit by agent faults.
        Pass `attempt_idx` for calls issued in parallel, so fault injection stays deterministic."""
        with self._lock:
            idx = self.attempt if attempt_idx is None else attempt_idx
            if diagnosis:
                self.attempt = max(self.attempt, idx + 1)
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
                rng = np.random.default_rng([self.fault.params.get("seed", 0), idx])
                if any(k.endswith("_reliability") for k in props):  # PRAM-R baseline schema (plan 2026-11-22)
                    raw = json.dumps(flip_reliability(clean, rng))
                else:
                    sensors = ([v[len("fault:"):] for v in props["verdict"]["enum"] if v.startswith("fault:")]
                               if "verdict" in props else [s for s in props["sensor"]["enum"] if s != "none"])
                    raw = json.dumps(flip_verdict(clean, sensors, rng))
        data = validate_json(raw, schema)
        return Attempt(raw, data, data is not None, False, res.latency_s, res.cached, clean, corrupted=corrupted)
