"""Methods compared in E1/E2 (spec §5.1). Every method sees the same FrameContext and returns MethodOutputs."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np
from PIL import Image

from recoverdrive import pramr
from recoverdrive.agent import AgentCaller, Attempt
from recoverdrive.checks import CheckOutput, flagged_sensors, rank_sensors
from recoverdrive.faults import AGENT_FAULTS, AgentFaultState
from recoverdrive.geometry import to_sensor_frame
from recoverdrive.recovery import RecoveryConfig, RecoveryController, no_recovery, vote
from recoverdrive.render import lidar_bev
from recoverdrive.shield import shield
from recoverdrive.types import Action, Det2D, Diagnosis, FaultLabel, Frame, Pose, RecoveryResult
from recoverdrive.vlm import (
    CAPTION_SCHEMA,
    CAPTION_SYSTEM,
    CAPTION_USER,
    OBSERVATION_SCHEMA,
    VLMClient,
    camera_observe_messages,
    decide_messages,
    format_notes,
    image_check_messages,
    image_check_schema,
    image_part,
    lidar_observe_messages,
    observe_keys,
    observe_messages,
    observe_schema,
    ours_messages,
    ours_schema,
    textchain_messages,
    textchain_schema,
    think_schema,
    think_trust_messages,
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
        self.detect_name, self.full_name = "ours_detect", "ours_full"
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
        if self.detect_name in self.names:
            outputs[self.detect_name] = no_recovery_output(to_diagnosis(first.data), first.valid, n_calls=1,
                                                        latency_s=first.latency_s,
                                                        extra={"attempts": [attempt_record(first)]})
        if self.full_name not in self.names:
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
        outputs[self.full_name] = MethodOutput(
            diag if valid else None, valid, alarm, pred, recovery=result, action=action,
            n_calls=len(all_calls), latency_s=sum(a.latency_s for a in all_calls), extra=extra)
        return outputs

    def _gated(self, ctx: FrameContext) -> dict[str, MethodOutput]:
        """No flag and the gate is on: the frame is treated as healthy without calling the VLM."""
        diag = Diagnosis(status="ok", sensor="none", confidence=1.0, fault_type="none", evidence="gated: no flag")
        outputs = {}
        if self.detect_name in self.names:
            outputs[self.detect_name] = no_recovery_output(diag, True, extra={"gated": True})
        if self.full_name in self.names:
            result = self.controller.step(diag, True, set(), ["gated"])
            alarm, pred = full_outcome(result, ["gated"], diag)
            extra = {"gated": True, "timing": {"vlm_first": 0.0, "vlm_reask": 0.0, "vlm_samples": 0.0}}
            if ctx.agent_fault is not None:
                extra["agent_recovered"] = True  # no call was made, so the fault had no effect
            outputs[self.full_name] = MethodOutput(diag, True, alarm, pred, recovery=result, action=shield(result),
                                                extra=extra)
        return outputs


class VLMVariant(Ours):
    """VLM-centric variants (plan 2026-11-21): the VLM reasons before it answers and, in the two-stage variants,
    first describes the raw sensor data itself. Only the diagnosis messages differ from Ours: the recovery
    controller, the 3-sample vote and the shield are the same, so `<variant>_full` is Ours-full with another
    diagnosis agent. `_nochecks` variants get neither the check report nor the flags.

    think            one call: images + check report, reasoning first
    think_trust      one call: the frozen Ours prompt (checks are reliable) + reasoning first (ablation of `think`)
    perceive_trust   stage 1 notes (as perceive); stage 2 = think_trust with the notes as one more piece of evidence
    imgflag          stage 1 asks the VLM only about image quality; an abnormal camera becomes one more flag
                     (camera:<C>:vlm_image) in the report and the recovery; then the frozen Ours call decides
    think_nochecks   one call: images only, reasoning first
    perceive         stage 1 notes on every sensor from the images (no checks); stage 2 decides from notes + report + images
    perceive_nochecks  the same without the report
    percam           stage 1 one call per camera (higher resolution, with its previous image) and one for the LiDAR;
                     stage 2 decides from the notes + report + current mosaic and top view
    Stage-1 calls are not diagnosis calls: agent faults (A1-A3) hit only the deciding call, as for Ours."""

    VARIANTS: ClassVar[tuple[str, ...]] = ("think", "think_nochecks", "perceive", "perceive_nochecks", "percam",
                                           "think_trust", "perceive_trust", "imgflag")
    MAX_TOKENS: ClassVar[dict[str, int]] = {"decide": 768, "observe": 640, "sensor": 200}
    CAMERA_PX: ClassVar[tuple[int, int]] = (960, 540)

    def __init__(self, variant: str, client: VLMClient, cfg: dict, names: tuple[str, ...]):
        super().__init__(client, cfg, names)
        self.variant = variant
        self.detect_name, self.full_name = f"{variant}_detect", f"{variant}_full"
        self.no_checks = variant.endswith("_nochecks")
        self.notes: str | None = None

    def _camera_part(self, image: np.ndarray) -> dict:
        return image_part(np.asarray(Image.fromarray(image).resize(self.CAMERA_PX, Image.BILINEAR)))

    def observe(self, ctx: FrameContext) -> tuple[str | None, list[Attempt]]:
        """Stage 1: the VLM's own notes on every sensor (none for the one-call variants and imgflag)."""
        if self.variant.startswith("think") or self.variant == "imgflag":
            return None, []
        cams = [s.split(":", 1)[1] for s in ctx.sensors if s.startswith("camera:")]
        if self.variant.startswith("perceive"):
            att = self.caller.call(observe_messages(ctx.mosaic, ctx.prev_mosaic, ctx.bev), observe_schema(cams),
                                   max_tokens=self.MAX_TOKENS["observe"], diagnosis=False, name="observe")
            return format_notes(att.data if att.valid else None, observe_keys(cams)), [att]
        from recoverdrive.runner import points_in_ego  # runner imports this module

        prev = ctx.prev_frame
        jobs = [(c, camera_observe_messages(c, self._camera_part(ctx.frame.cameras[c].image),
                                            self._camera_part(prev.cameras[c].image) if prev is not None else None))
                for c in cams]
        lidar = ctx.frame.lidar
        n_points = 0 if lidar is None else len(lidar.points)
        prev_bev = image_part(lidar_bev(points_in_ego(prev))) if prev is not None else None
        jobs.append(("LIDAR_TOP", lidar_observe_messages(ctx.bev, prev_bev, self.cfg["data"].get("nominal_lidar_points"),
                                                         n_points)))
        with ThreadPoolExecutor(len(jobs)) as pool:
            atts = list(pool.map(lambda j: self.caller.call(j[1], OBSERVATION_SCHEMA, max_tokens=self.MAX_TOKENS["sensor"],
                                                            diagnosis=False, name="observe_sensor"), jobs))
        notes = {k: (a.data["observation"] if a.valid else "(no observation)") for (k, _), a in zip(jobs, atts)}
        return format_notes(notes, [k for k, _ in jobs]), atts

    def diagnose(self, ctx: FrameContext, temperature: float = 0.0, seed: int | None = None,
                 attempt_idx: int | None = None) -> Attempt:
        summary = None if self.no_checks else ctx.summary
        prev = None if self.variant == "percam" else ctx.prev_mosaic  # percam's notes already compare frames
        if self.variant == "imgflag":  # the frozen Ours call on the report extended by the VLM image check
            return super().diagnose(ctx, temperature, seed, attempt_idx)
        if self.variant in ("think_trust", "perceive_trust"):
            msgs = think_trust_messages(ctx.summary, ctx.mosaic, ctx.prev_mosaic, ctx.bev, self.notes)
        else:
            msgs = decide_messages(summary, self.notes, ctx.mosaic, prev, ctx.bev)
        return self.caller.call(msgs, think_schema(ctx.sensors), temperature=temperature, seed=seed,
                                max_tokens=self.MAX_TOKENS["decide"], name="diagnosis", attempt_idx=attempt_idx)

    def step(self, ctx: FrameContext) -> dict[str, MethodOutput]:
        self.caller.new_frame(ctx.agent_fault)
        self.notes, stage1 = self.observe(ctx)
        if self.variant == "imgflag":
            flags, details, att = image_flags(self.caller, ctx)
            ctx, stage1 = with_image_flags(ctx, flags, details), [att]
        outputs = super().step(ctx)
        lat = sum(a.latency_s for a in stage1)
        for out in outputs.values():
            out.n_calls += len(stage1)
            out.latency_s += lat
            out.extra["stage1"] = [attempt_record(a) for a in stage1]
            out.extra.setdefault("timing", {})["vlm_stage1"] = lat
        return outputs


def image_flags(caller: AgentCaller, ctx: FrameContext) -> tuple[list[str], str, Attempt]:
    """The VLM's image-quality check (not a diagnosis call): flags camera:<C>:vlm_image for abnormal cameras."""
    cams = [s.split(":", 1)[1] for s in ctx.sensors if s.startswith("camera:")]
    att = caller.call(image_check_messages(ctx.mosaic, ctx.prev_mosaic), image_check_schema(cams), max_tokens=200,
                      diagnosis=False, name="image_check")
    if not att.valid:
        return [], "", att
    bad = [c for c in cams if att.data.get(c) == "abnormal"]
    return [f"camera:{c}:vlm_image" for c in bad], str(att.data.get("details", "")).strip(), att


def with_image_flags(ctx: FrameContext, flags: list[str], details: str) -> FrameContext:
    """A copy of the frame context with the VLM image flags added to the report flags and one summary line. With no
    flag the context is returned unchanged, so the deciding call is the frozen Ours call (and hits its cache)."""
    if not flags:
        return ctx
    cams = ", ".join(f.split(":")[1] for f in flags)
    line = f"A separate image-quality check by the VLM marks as abnormal: {cams} ({details})."
    report = replace(ctx.check.report, flags=[*ctx.check.report.flags, *flags])
    return replace(ctx, check=replace(ctx.check, report=report), summary=ctx.summary + "\n" + line)


def rules_outcome(result: RecoveryResult, ranked: list[str]) -> tuple[bool, str]:
    """E1 alarm and blamed sensor of a monitor without a VLM: the first ranked sensor that is isolated, else the
    first ranked sensor."""
    alarm = result.final_state != "healthy"
    if not alarm:
        return False, "none"
    isolated = set(result.isolated_sensors)
    pred = next((s for s in ranked if s in isolated), ranked[0] if ranked else "none")
    return True, pred


class ChecksOnly:
    """Rules only, no VLM: every sensor with a deterministic fault flag is suspected with confidence 1.0, then the same
    re-observe / isolate / verify and shield as Ours-full (FutureWork Step 1)."""

    names = ("checks_only",)

    def __init__(self, cfg: dict):
        self.controller = RecoveryController(RecoveryConfig(**cfg.get("recovery", {})))

    def step(self, ctx: FrameContext) -> dict[str, MethodOutput]:
        ranked = rank_sensors(ctx.check.report.flags)
        result = self.controller.step_suspects(set(ranked), 1.0, set(ranked), ["rules"])
        alarm, pred = rules_outcome(result, ranked)
        extra: dict = {"flagged": ranked}
        if ctx.agent_fault is not None:
            extra["agent_recovered"] = True  # no VLM call, so the agent fault has no effect (as with the VLM gate)
        return {"checks_only": MethodOutput(None, True, alarm, pred, recovery=result, action=shield(result),
                                            n_calls=0, extra=extra)}


class ImageFlagChecks:
    """Checks-only plus the VLM's image-quality check as one more flag, with no deciding VLM call: the VLM is used only
    as a feature extractor for image faults (plan 2026-11-21). Its call is not a diagnosis call, so agent faults do
    not reach it (as for the stage-1 calls of the variants)."""

    names = ("imgflag_checks_full",)

    def __init__(self, client: VLMClient, cfg: dict):
        self.caller = AgentCaller(client)
        self.controller = RecoveryController(RecoveryConfig(**cfg.get("recovery", {})))

    def step(self, ctx: FrameContext) -> dict[str, MethodOutput]:
        self.caller.new_frame(ctx.agent_fault)
        flags, _, att = image_flags(self.caller, ctx)
        ranked = rank_sensors([*ctx.check.report.flags, *flags])
        result = self.controller.step_suspects(set(ranked), 1.0, set(ranked), ["rules"])
        alarm, pred = rules_outcome(result, ranked)
        extra: dict = {"flagged": ranked, "vlm_flags": flags, "stage1": [attempt_record(att)]}
        if ctx.agent_fault is not None:
            extra["agent_recovered"] = True
        return {"imgflag_checks_full": MethodOutput(None, True, alarm, pred, recovery=result, action=shield(result),
                                                    n_calls=1, latency_s=att.latency_s, extra=extra)}


class PramR:
    """PRAM-R-style baseline (arXiv 2603.04222), one reliability score per sensor (plan 2026-11-22, src/recoverdrive/
    pramr.py). Indicators -> one router call per frame -> threshold with hysteresis -> switched-off sensors are
    isolated -> rule-based shield. No re-ask, vote or verification (the paper has none): an invalid or late answer
    keeps the previous state and raises no alarm. Names: pramr_text_full / pramr_image_full (dev comparison of the
    scene context) and pramr_full (the frozen choice, pramr.SCENE_IMAGE)."""

    NAMES: ClassVar[tuple[str, ...]] = ("pramr_text_full", "pramr_image_full", "pramr_full")

    def __init__(self, name: str, client: VLMClient, cfg: dict, reference: dict | None = None,
                 theta_camera: int = pramr.THETA_CAMERA, theta_lidar: int = pramr.THETA_LIDAR, delta: int = pramr.DELTA):
        self.name, self.names = name, (name,)
        self.image = {"pramr_text_full": False, "pramr_image_full": True, "pramr_full": pramr.SCENE_IMAGE}[name]
        self.caller = AgentCaller(client)
        self.reference = reference  # loaded from configs/pramr_reference.yaml at the first step when None
        self.params = {"theta_camera": theta_camera, "theta_lidar": theta_lidar, "delta": delta,
                       "scene_image": self.image}
        self.active: dict[str, bool] | None = None
        self.prev_indicators: dict | None = None
        self.prev_scores: dict[str, float] | None = None
        self.last_scores: dict[str, float] | None = None

    def _latch(self, active: dict[str, bool], scores: dict[str, float] | None) -> dict[str, bool]:
        p = self.params
        return pramr.latch_step(active, scores, p["theta_camera"], p["theta_lidar"], p["delta"])

    def step(self, ctx: FrameContext) -> dict[str, MethodOutput]:
        self.caller.new_frame(ctx.agent_fault)
        cams = [s.split(":", 1)[1] for s in ctx.sensors if s.startswith("camera:")]
        keys = pramr.sensor_keys(cams)
        if self.active is None:
            self.active = dict.fromkeys(keys, True)  # every run starts with all sensors on
        if self.reference is None:
            self.reference = pramr.load_reference()
        indicators = pramr.frame_indicators(ctx.frame, cams, self.reference)
        msgs = pramr.pramr_messages(cams, indicators, self.reference, self.prev_indicators, self.prev_scores,
                                    ctx.mosaic if self.image else None)
        att = self.caller.call(msgs, pramr.pramr_schema(cams), max_tokens=pramr.MAX_TOKENS, name="pramr")
        scores = pramr.scores_of(att.data, cams)
        clean = pramr.scores_of(att.clean_data, cams) if att.clean_data is not None else None
        before = dict(self.active)
        self.active = self._latch(before, scores)
        if scores is not None:
            self.last_scores = scores
        isolated, pred = pramr.outcome(self.active, self.last_scores, keys)
        result = RecoveryResult(["pramr"], "degraded" if isolated else "healthy", isolated, None)
        extra: dict = {"attempts": [attempt_record(att)], "scores": scores, "clean_scores": clean,
                       "indicators": indicators, "active_before": before, "active_after": dict(self.active),
                       "params": self.params}
        if ctx.agent_fault is not None:  # spec §5.2: valid answer whose switching matches the fault-free answer
            extra["agent_recovered"] = bool(att.valid and clean is not None
                                            and self.active == self._latch(before, clean))
        self.prev_indicators, self.prev_scores = indicators, scores
        return {self.name: MethodOutput(None, att.valid, bool(isolated), pred, recovery=result, action=shield(result),
                                        n_calls=1, latency_s=att.latency_s, extra=extra)}


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


VLM_FREE_METHODS = ("B0", "checks_only")


def needs_vlm(methods: list[str], ablation: str) -> bool:
    """True if a VLM server is needed: a method outside VLM_FREE_METHODS, or Qwen grounding as the camera detector."""
    return any(m not in VLM_FREE_METHODS for m in methods) or ablation == "qwen_grounding"


def build_methods(names: list[str], client: VLMClient | None, cfg: dict) -> list:
    """One object per method family; each keeps its own per-run state."""
    methods: list = []
    variant_names = tuple(f"{v}_{k}" for v in VLMVariant.VARIANTS for k in ("detect", "full"))
    known = ("B0", "B1", "B2", "B3", "ours_detect", "ours_full", "checks_only", "imgflag_checks_full", *PramR.NAMES,
             *variant_names)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    if "B0" in names:
        methods.append(B0())
    if "checks_only" in names:
        methods.append(ChecksOnly(cfg))
    if "imgflag_checks_full" in names:
        methods.append(ImageFlagChecks(client, cfg))
    methods += [PramR(n, client, cfg) for n in PramR.NAMES if n in names]
    methods += [TextChain(n, client) for n in ("B1", "B2", "B3") if n in names]
    ours = tuple(n for n in ("ours_detect", "ours_full") if n in names)
    if ours:
        methods.append(Ours(client, cfg, ours))
    for v in VLMVariant.VARIANTS:
        own = tuple(n for n in (f"{v}_detect", f"{v}_full") if n in names)
        if own:
            methods.append(VLMVariant(v, client, cfg, own))
    return methods
