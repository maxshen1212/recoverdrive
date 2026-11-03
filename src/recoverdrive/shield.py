"""Rule-based safety shield (spec §4.5) and the acceptable-action labels behind the fallback metrics (spec §5.2).
No LLM is involved here."""

from __future__ import annotations

from recoverdrive.faults import AGENT_FAULTS, FIXED_HIGH
from recoverdrive.types import Action, FaultLabel, RecoveryResult

ACTIONS = ("continue", "slow_down", "stop")  # ordered by conservativeness
FRONT_SENSORS = ("camera:CAM_FRONT", "lidar")  # plan D7: front perception = CAM_FRONT or LiDAR


def shield(result: RecoveryResult, front_sensors: tuple[str, ...] = FRONT_SENSORS) -> Action:
    if result.final_state == "healthy":
        return Action("continue", "healthy")
    if result.final_state == "unresolved":
        return Action("stop", "unresolved fault")
    if any(s not in result.isolated_sensors for s in front_sensors):
        return Action("slow_down", f"degraded, isolated {result.isolated_sensors}")
    return Action("stop", "degraded without front perception")


def ttc_override(action: Action, ttc_s: float | None, threshold_s: float = 2.0) -> Action:
    """Any state: brake when time-to-collision drops below the threshold (CARLA)."""
    if ttc_s is not None and ttc_s < threshold_s:
        return Action("stop", f"TTC {ttc_s:.2f} s < {threshold_s} s")
    return action


def acceptable_actions(label: FaultLabel, front_sensors: tuple[str, ...] = FRONT_SENSORS) -> tuple[str, ...]:
    """Spec §5.2 table. C1 and L4 count as high severity."""
    if label.fault_id == "none":
        return ("continue",)
    if label.fault_id in AGENT_FAULTS:
        return ("continue", "slow_down")
    severity = "high" if label.fault_id in FIXED_HIGH else label.severity
    if severity == "low" or label.sensor not in front_sensors:
        return ("continue", "slow_down")
    return ("slow_down", "stop")


def classify_action(action: str, acceptable: tuple[str, ...]) -> str:
    """'unnecessary' if more conservative than the most conservative acceptable action, 'missed' if less
    conservative than the least conservative one, else 'ok'."""
    idx = ACTIONS.index(action)
    levels = [ACTIONS.index(a) for a in acceptable]
    if idx > max(levels):
        return "unnecessary"
    if idx < min(levels):
        return "missed"
    return "ok"
