"""Closed-loop test scenarios for CARLA (E3 and its falsification).

A `Scenario` fixes what the E3 runner used to hard-code: where the hazard triggers, how fast the pedestrian walks, how
long the lead car brakes, and which front-sensor fault is active over which stretch of the route. The defaults
reproduce E3 exactly (`Scenario.from_condition`), so E3 runs do not change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from recoverdrive.faults import inject_c1, inject_c2, inject_c3, inject_l1
from recoverdrive.types import FaultLabel, Frame

FAULTS = ("none", "C1", "C3_black", "C2_rotate", "L1_sector")
LEVEL_RANGES = {"C2_rotate": (0.0, 12.0), "L1_sector": (0.0, 360.0)}  # degrees
BASE_TRIGGER_M = 40.0


@dataclass
class Scenario:
    hazard: str  # "lead_brake" or "pedestrian"
    trigger_offset_m: float | None = None  # None: E3's random offset in [-3, 3] m
    walker_speed_mps: float = 1.6
    lead_brake_s: float = 3.0
    fault: str = "C1"
    fault_level: float | None = None  # C2_rotate: degrees; L1_sector: degrees of the dropped band
    fault_start_m: float = -25.0  # fault window, relative to the trigger point
    fault_end_m: float = 30.0

    @staticmethod
    def from_condition(hazard: str, condition: str) -> Scenario:
        """The E3 scenario of a condition ("clean" or "C1")."""
        return Scenario(hazard, fault="C1" if condition == "C1" else "none")

    def trigger_m(self, random_offset: float) -> float:
        """The trigger point; E3 draws `random_offset` in [-3, 3] m, a falsification scenario fixes it."""
        return BASE_TRIGGER_M + (random_offset if self.trigger_offset_m is None else self.trigger_offset_m)

    def apply_fault(self, frame: Frame, progress_m: float, trigger_m: float,
                    rng: np.random.Generator) -> tuple[Frame, FaultLabel]:
        """Inject the scenario's fault while the ego is inside the fault window."""
        if self.fault == "none" or not trigger_m + self.fault_start_m <= progress_m <= trigger_m + self.fault_end_m:
            return frame, FaultLabel.none()
        if self.fault == "C1":
            return inject_c1(frame, "CAM_FRONT", "CAM_BACK"), FaultLabel("C1", "camera:CAM_FRONT", "high", 0)
        if self.fault in ("C3_black", "C2_rotate"):
            from recoverdrive.falsify import severity_of
            cam = frame.cameras["CAM_FRONT"]
            if self.fault == "C3_black":
                img, label = inject_c3(cam.image, "black", None, rng), FaultLabel("C3", "camera:CAM_FRONT", "high", 0)
            else:
                lvl = self._level()
                img = inject_c2(cam.image, "rotate", lvl, 1)
                label = FaultLabel("C2", "camera:CAM_FRONT", severity_of("C2", "rotate", lvl), 0)
            cams = dict(frame.cameras)
            cams["CAM_FRONT"] = replace(cam, image=img, image_path=None)
            return replace(frame, cameras=cams), label
        if self.fault == "L1_sector":
            from recoverdrive.falsify import severity_of
            lvl = self._level()
            pts = inject_l1(frame.lidar.points, "sector", lvl, rng, azimuth_deg=0.0)  # band centered straight ahead
            label = FaultLabel("L1", "lidar", severity_of("L1", "sector", lvl), 0)
            return replace(frame, lidar=replace(frame.lidar, points=pts)), label
        raise ValueError(f"unknown scenario fault {self.fault}")

    def _level(self) -> float:
        lo, hi = LEVEL_RANGES[self.fault]
        if self.fault_level is None:
            raise ValueError(f"{self.fault} needs fault_level")
        return float(min(max(self.fault_level, lo), hi))


# Falsification search space per hazard (same format as falsify.campaign); "route" is filled in with the route indices
# of that hazard by the script, because they depend on the town.
_COMMON = {"trigger_offset_m": ("float", -5.0, 5.0), "fault": ("cat", ["C1", "C3_black", "C2_rotate", "L1_sector"]),
           "level_u": ("float", 0.0, 1.0), "fault_start_m": ("float", -30.0, 0.0),
           "fault_len_m": ("float", 10.0, 60.0), "seed": ("int", 0, 99)}
CARLA_SEARCH = {
    "pedestrian": {"route": ("cat", []), **_COMMON, "walker_speed_mps": ("float", 1.0, 2.5)},
    "lead_brake": {"route": ("cat", []), **_COMMON, "lead_brake_s": ("float", 1.0, 4.0)},
}


def scenario_from_params(hazard: str, p: dict) -> Scenario:
    """A Scenario from one falsification sample; parameters of the other hazard are ignored."""
    fault = p["fault"]
    level = None
    if fault in LEVEL_RANGES:
        lo, hi = LEVEL_RANGES[fault]
        level = lo + float(p["level_u"]) * (hi - lo)
    return Scenario(hazard, trigger_offset_m=float(p["trigger_offset_m"]),
                    walker_speed_mps=float(p["walker_speed_mps"]) if hazard == "pedestrian" else 1.6,
                    lead_brake_s=float(p["lead_brake_s"]) if hazard == "lead_brake" else 3.0, fault=fault,
                    fault_level=level, fault_start_m=float(p["fault_start_m"]),
                    fault_end_m=float(p["fault_start_m"]) + float(p["fault_len_m"]))
