"""Methods compared in E1/E2 (spec §5.1). Every method sees the same FrameContext and returns MethodOutputs."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from PIL import Image

from recoverdrive.agent import AgentCaller, Attempt
from recoverdrive.checks import CheckOutput, flagged_sensors
from recoverdrive.faults import AGENT_FAULTS, AgentFaultState
from recoverdrive.geometry import to_sensor_frame
from recoverdrive.recovery import RecoveryConfig, RecoveryController, no_recovery, vote
from recoverdrive.shield import shield
from recoverdrive.types import Action, Det2D, Diagnosis, FaultLabel, Frame, Pose, RecoveryResult
from recoverdrive.vlm import (
    CAPTION_SCHEMA,
    CAPTION_SYSTEM,
    CAPTION_USER,
    VLMClient,
    image_part,
    ours_messages,
    ours_schema,
    textchain_messages,
    textchain_schema,
    to_diagnosis,
)


@dataclass
class FrameContext:
    frame: Frame
    label: FaultLabel
    check: CheckOutput
    summary: str
    mosaic: dict  # image part (vlm.image_part) of the current camera mosaic
    prev_mosaic: dict | None
    bev: dict
    sensors: list[str]  # "camera:<C>" for each monitored camera, then "lidar"
    frame_seed: int
    agent_fault: AgentFaultState | None = None
    detections: dict[str, list[Det2D]] = field(default_factory=dict)
    prev_frame: Frame | None = None


@dataclass
class MethodOutput:
    diagnosis: Diagnosis | None
    valid: bool
    alarm: bool  # E1 positive (plan D3)
    pred_sensor: str  # sensor blamed, "agent" for an unusable agent output, "none" if no alarm
    recovery: RecoveryResult | None = None
    action: Action | None = None
    n_calls: int = 0
    latency_s: float = 0.0
    extra: dict = field(default_factory=dict)


def attempt_record(a: Attempt) -> dict:
    return {"raw": a.raw, "valid": a.valid, "timeout": a.timeout, "corrupted": a.corrupted, "cached": a.cached,
            "latency_s": round(a.latency_s, 4), "clean_data": a.clean_data, "error": a.error}


def detect_output(diag: Diagnosis | None, valid: bool) -> tuple[bool, str]:
    """Alarm and blamed sensor for a single diagnosis without recovery."""
    if not valid or diag is None:
        return True, "agent"
    if diag.status == "ok":
        return False, "none"
    return True, diag.sensor


def same_verdict(a: Diagnosis | None, b: dict | None) -> bool:
    """Same status and (for non-ok answers) same sensor."""
    ref = to_diagnosis(b)
    if a is None or ref is None:
        return False
    return a.status == ref.status and (a.status == "ok" or a.sensor == ref.sensor)


def full_outcome(result: RecoveryResult, steps: list[str], diag: Diagnosis | None) -> tuple[bool, str]:
    """E1 alarm and blamed sensor of Ours-full (shared by the method and the offline ablation replay)."""
    isolated = [s for s in result.isolated_sensors if s != "agent"]
    reasked = "reask" in steps  # a bad agent output was noticed, even if re-asking then fixed it
    if result.final_state == "healthy":
        pred = "agent" if reasked else "none"
    elif isolated:
        pred = isolated[0]
    elif "agent" in result.isolated_sensors:
        pred = "agent"
    else:
        pred = diag.sensor if diag is not None else "agent"
    return result.final_state != "healthy" or reasked, pred


def no_recovery_output(diag: Diagnosis | None, valid: bool, **kw) -> MethodOutput:
    alarm, sensor = detect_output(diag, valid)
    result = no_recovery(diag, valid)
    return MethodOutput(diag, valid, alarm, sensor, recovery=result, action=shield(result), **kw)


class Ours:
    """Consistency checks + SAM 3 + Qwen diagnosis.

    ours_detect: the first answer, no recovery. ours_full: + re-ask, 3-sample vote, re-observe, isolate, verify,
    shield (spec §4.3-4.5). Both come from one pass, so the first diagnosis call is shared.
    """

    def __init__(self, client: VLMClient, cfg: dict, names: tuple[str, ...] = ("ours_detect", "ours_full")):
        self.caller = AgentCaller(client)
        self.cfg = cfg
        self.names = names
        self.rcfg = RecoveryConfig(**cfg.get("recovery", {}))
        self.controller = RecoveryController(self.rcfg)
        self.no_checks = cfg.get("ablation") == "no_checks"  # E5: the VLM and recovery get no check results

    def diagnose(self, ctx: FrameContext, temperature: float = 0.0, seed: int | None = None,
                 attempt_idx: int | None = None) -> Attempt:
        msgs = ours_messages(None if self.no_checks else ctx.summary, ctx.mosaic, ctx.prev_mosaic, ctx.bev)
        return self.caller.call(msgs, ours_schema(ctx.sensors), temperature=temperature, seed=seed, name="diagnosis",
                                attempt_idx=attempt_idx)

    def step(self, ctx: FrameContext) -> dict[str, MethodOutput]:
        self.caller.new_frame(ctx.agent_fault)
        flagged = set() if self.no_checks else flagged_sensors(ctx.check.report.flags) - {"agent"}
        if self.rcfg.gate == "flags" and not flagged:  # §4.2 latency optimization: no flag, no VLM call
            return self._gated(ctx)
        timing: dict[str, float] = {}
        first = self.diagnose(ctx)
        timing["vlm_first"] = first.latency_s
        outputs = {}
        if "ours_detect" in self.names:
            outputs["ours_detect"] = no_recovery_output(to_diagnosis(first.data), first.valid, n_calls=1,
                                                        latency_s=first.latency_s,
                                                        extra={"attempts": [attempt_record(first)]})
        if "ours_full" not in self.names:
            return outputs

        attempts, steps = [first], []
        while not attempts[-1].valid and len(attempts) <= self.rcfg.max_reask:  # 1. re-ask on bad format
            steps.append("reask")
            attempts.append(self.diagnose(ctx))
        timing["vlm_reask"] = sum(a.latency_s for a in attempts[1:])
        valid = attempts[-1].valid
        diag = to_diagnosis(attempts[-1].data)
        samples: list[Attempt] = []
        t0 = time.perf_counter()
        if self.rcfg.sampling and valid and (diag.status != "ok" or flagged):  # suspected fault: sample and vote
            steps.append("resample")
            base = len(attempts)
            with ThreadPoolExecutor(self.rcfg.n_samples) as pool:
                samples = list(pool.map(
                    lambda i: self.diagnose(ctx, self.rcfg.sample_temperature, ctx.frame_seed * 10 + i, base + i),
                    range(self.rcfg.n_samples)))
            voted = [to_diagnosis(a.data) for a in samples if a.valid]
            if voted:
                diag, _ = vote(voted)
        timing["vlm_samples"] = time.perf_counter() - t0 if samples else 0.0
        t0 = time.perf_counter()
        result = self.controller.step(diag if valid else None, valid, flagged, steps)  # 2-3. re-observe, isolate
        action = shield(result)
        timing["recovery"] = time.perf_counter() - t0
        alarm, pred = full_outcome(result, steps, diag)
        extra = {"attempts": [attempt_record(a) for a in attempts], "samples": [attempt_record(a) for a in samples],
                 "flagged": sorted(flagged), "timing": timing}
        if ctx.agent_fault is not None and ctx.agent_fault.fault_id in AGENT_FAULTS:
            # Spec §5.2: recovered if the final answer is valid and matches the fault-free answer.
            extra["agent_recovered"] = valid and same_verdict(diag, first.clean_data)
        all_calls = attempts + samples
        outputs["ours_full"] = MethodOutput(
            diag if valid else None, valid, alarm, pred, recovery=result, action=action,
            n_calls=len(all_calls), latency_s=sum(a.latency_s for a in all_calls), extra=extra)
        return outputs

    def _gated(self, ctx: FrameContext) -> dict[str, MethodOutput]:
        """No flag and the gate is on: the frame is treated as healthy without calling the VLM."""
        diag = Diagnosis(status="ok", sensor="none", confidence=1.0, fault_type="none", evidence="gated: no flag")
        outputs = {}
        if "ours_detect" in self.names:
            outputs["ours_detect"] = no_recovery_output(diag, True, extra={"gated": True})
        if "ours_full" in self.names:
            result = self.controller.step(diag, True, set(), ["gated"])
            alarm, pred = full_outcome(result, ["gated"], diag)
            extra = {"gated": True, "timing": {"vlm_first": 0.0, "vlm_reask": 0.0, "vlm_samples": 0.0}}
            if ctx.agent_fault is not None:
                extra["agent_recovered"] = True  # no call was made, so the fault had no effect
            outputs["ours_full"] = MethodOutput(diag, True, alarm, pred, recovery=result, action=shield(result),
                                                extra=extra)
        return outputs


class B0:
    """No monitor: always continue."""

    names = ("B0",)

    def step(self, ctx: FrameContext) -> dict[str, MethodOutput]:
        result = RecoveryResult([], "healthy", [], None)
        return {"B0": MethodOutput(None, True, False, "none", recovery=result, action=shield(result))}


IDENTITY = Pose(np.array([1.0, 0, 0, 0]), np.zeros(3))


def lidar_text(ctx: FrameContext, max_objects: int = 25) -> str:
    """DriveAgent-style LiDAR object list: class and (x, y) in the ego frame (x forward, y left)."""
    lidar = ctx.frame.lidar
    n_points = 0 if lidar is None else len(lidar.points)
    objs = sorted(ctx.check.objects, key=lambda o: o.distance)[:max_objects]
    parts = []
    for o in objs:
        b = to_sensor_frame(o.box, lidar.ego_pose, IDENTITY)
        parts.append(f"{o.box.category} at ({b.center[0]:.1f}, {b.center[1]:.1f}) m with {o.n_points} points")
    listing = "; ".join(parts) if parts else "no objects detected"
    return f"{len(ctx.check.objects)} objects within 50 m: {listing}. Total LiDAR points this sweep: {n_points}."


class TextChain:
    """B1-B3: DriveAgent's vehicle-abnormality chain re-implemented on the same Qwen3-VL (spec §5.1).

    Qwen captions each camera (cached; T = 0), the LiDAR objects become text, and one text-only call returns the
    verdict. No recovery: the verdict feeds the shared shield through no_recovery (plan D2, D9)."""

    VARIANTS: ClassVar[dict[str, str]] = {"B1": "zero_shot", "B2": "cot", "B3": "self_refine"}
    MAX_TOKENS: ClassVar[dict[str, int]] = {"zero_shot": 256, "cot": 2048, "self_refine": 2048}

    def __init__(self, name: str, client: VLMClient):
        self.names = (name,)
        self.variant = self.VARIANTS[name]
        self.caller = AgentCaller(client)

    def caption(self, camera: str, image: np.ndarray) -> str:
        small = np.asarray(Image.fromarray(image).resize((800, 450), Image.BILINEAR))
        msgs = [{"role": "system", "content": CAPTION_SYSTEM},
                {"role": "user", "content": [image_part(small), {"type": "text", "text": CAPTION_USER.format(camera=camera)}]}]
        att = self.caller.call(msgs, CAPTION_SCHEMA, max_tokens=160, diagnosis=False, name="caption")
        return att.data["description"] if att.valid else "(no description available)"

    def step(self, ctx: FrameContext) -> dict[str, MethodOutput]:
        self.caller.new_frame(ctx.agent_fault)
        cams = [s.split(":", 1)[1] for s in ctx.sensors if s.startswith("camera:")]
        with ThreadPoolExecutor(len(cams)) as pool:
            captions = list(pool.map(lambda c: self.caption(c, ctx.frame.cameras[c].image), cams))
        visual = "\n".join(f"{c}: {text}" for c, text in zip(cams, captions))
        msgs = textchain_messages(self.variant, lidar_text(ctx), visual, ctx.sensors)
        att = self.caller.call(msgs, textchain_schema(self.variant, ctx.sensors),
                               max_tokens=self.MAX_TOKENS[self.variant], name="diagnosis")
        out = no_recovery_output(to_diagnosis(att.data), att.valid, n_calls=len(cams) + 1, latency_s=att.latency_s,
                                 extra={"attempts": [attempt_record(att)]})
        return {self.names[0]: out}


def build_methods(names: list[str], client: VLMClient | None, cfg: dict) -> list:
    """One object per method family; each keeps its own per-run state."""
    methods: list = []
    known = ("B0", "B1", "B2", "B3", "ours_detect", "ours_full")
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    if "B0" in names:
        methods.append(B0())
    methods += [TextChain(n, client) for n in ("B1", "B2", "B3") if n in names]
    ours = tuple(n for n in ("ours_detect", "ours_full") if n in names)
    if ours:
        methods.append(Ours(client, cfg, ours))
    return methods
