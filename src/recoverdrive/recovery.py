"""Recovery controller (spec §4.3) and verify (spec §4.4).

The controller holds the decision logic only; the VLM calls (re-ask, sampling) are made by the method and passed in.
Re-observation is causal (plan D6): a sensor suspected in two consecutive keyframes is persistent; an isolated
sensor is restored after a frame in which neither the checks nor the VLM point at it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

from recoverdrive.types import Diagnosis, RecoveryResult


@dataclass
class RecoveryConfig:
    max_reask: int = 2
    n_samples: int = 3
    sample_temperature: float = 0.7
    conf_threshold: float = 0.6
    # Ablation switches (Phase 5). The defaults are the frozen Ours-full.
    sampling: bool = True  # False: no 3-sample vote; the last valid answer is final
    reobserve: bool = True  # False: no memory across frames (no persistence promotion, no carried isolation)
    isolate: bool = True  # False: never isolate a sensor or the agent; any confirmed fault is unresolved
    gate: str = "none"  # "flags": call the VLM only when a deterministic flag is raised (latency optimization)


def vote(diags: list[Diagnosis]) -> tuple[Diagnosis, float]:
    """Majority vote over (status, sensor). Confidence = share of samples that agree with the majority.
    Ties go to the earliest sample."""
    keys = [(d.status, "none" if d.status == "ok" else d.sensor) for d in diags]
    counts = Counter(keys)
    best = max(keys, key=lambda k: (counts[k], -keys.index(k)))
    rep = diags[keys.index(best)]
    conf = counts[best] / len(keys)
    return replace(rep, confidence=conf), conf


def no_recovery(diagnosis: Diagnosis | None, valid: bool) -> RecoveryResult:
    """Methods without recovery (plan D2): ok -> healthy, anything else -> unresolved."""
    healthy = valid and diagnosis is not None and diagnosis.status == "ok"
    return RecoveryResult(steps_taken=[], final_state="healthy" if healthy else "unresolved", isolated_sensors=[],
                          final_diagnosis=diagnosis)


@dataclass
class RecoveryController:
    cfg: RecoveryConfig = field(default_factory=RecoveryConfig)
    isolated: dict[str, float] = field(default_factory=dict)  # sensor -> confidence when isolated
    suspected_prev: set[str] = field(default_factory=set)

    def step(self, diag: Diagnosis | None, agent_ok: bool, flagged: set[str], steps: list[str]) -> RecoveryResult:
        """One round for one frame.

        diag: final (voted) diagnosis, or None when the agent gave no valid answer after re-asking.
        flagged: sensors with deterministic fault flags this frame (camera:<C> / lidar).
        """
        steps = list(steps)
        tau = self.cfg.conf_threshold
        flagged = set(flagged) - {"agent"}
        if not self.cfg.reobserve:
            self.isolated, self.suspected_prev = {}, set()
        conf = 1.0
        if not agent_ok or diag is None:
            steps.append("isolate:agent")
            suspects = set(flagged)  # rule-based fallback: trust the deterministic flags
        elif diag.status == "fault" and diag.sensor != "none":
            suspects, conf = {diag.sensor}, diag.confidence
        elif diag.status == "uncertain" and flagged:
            suspects, conf = (set(flagged) & {diag.sensor}) or set(flagged), min(diag.confidence, tau - 1e-9)
        else:
            suspects = set()

        if not self.cfg.isolate:  # ablation: nothing can be isolated, so any suspicion or agent failure stops
            self.suspected_prev = suspects
            unresolved = bool(suspects) or not agent_ok or diag is None
            return RecoveryResult(steps_taken=steps, final_state="unresolved" if unresolved else "healthy",
                                  isolated_sensors=[], final_diagnosis=diag)

        # Re-observe: restore isolated sensors that look healthy again.
        for s in sorted(self.isolated):
            if s not in suspects and s not in flagged:
                steps.append(f"reobserve:restored:{s}")
                del self.isolated[s]
        pending = set()
        for s in sorted(suspects):
            if s in self.isolated:
                steps.append(f"reobserve:persistent:{s}")
                self.isolated[s] = max(self.isolated[s], conf)
            elif s in self.suspected_prev:
                steps += [f"reobserve:persistent:{s}", f"isolate:{s}"]
                self.isolated[s] = max(conf, tau)
            elif conf >= tau:
                steps.append(f"isolate:{s}")
                self.isolated[s] = conf
            else:
                steps.append(f"reobserve:pending:{s}")
                pending.add(s)
        self.suspected_prev = suspects

        isolated = sorted(self.isolated)
        if not agent_ok or diag is None:
            isolated = sorted({*isolated, "agent"})
        # Verify: the remaining sensors must be consistent, and the isolation confident.
        remaining_flagged = flagged - set(isolated)
        if pending or (isolated and remaining_flagged):
            state = "unresolved"
        elif isolated:
            confident = all(self.isolated[s] >= tau for s in self.isolated)
            state = "degraded" if confident else "unresolved"
        else:
            state = "healthy"
        if isolated:
            steps.append(f"verify:{state}")
        return RecoveryResult(steps_taken=steps, final_state=state, isolated_sensors=isolated, final_diagnosis=diag)
