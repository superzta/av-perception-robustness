"""Closed-loop perception-driven vehicle controller.

This module provides a real autonomous-driving controller that converts the
output of the project's perception stack (camera-only or camera+LiDAR fusion)
into an actual ``carla.VehicleControl`` command applied to the ego vehicle.

Design:

* Longitudinal control (throttle / brake) is driven directly by perception:
    - Hard brake when any critical detection is found within ``brake_distance_m``
      or when the symbolic decision is ``BRAKE``.
    - Partial coast / soft brake when the decision is ``SLOW_DOWN`` or an
      obstacle is within ``slow_distance_m``.
    - Otherwise a P-controller tracks ``target_speed_kmh``.

* Lateral control (steering) follows the CARLA map's lane waypoints using a
  small look-ahead pure-pursuit style controller. This mirrors how real AV
  stacks separate planning (routing) from perception (when to stop).

The controller is intentionally simple, deterministic and stateless enough to
support paired clean vs. attacked comparisons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - import is exercised at runtime only
    from . import carla_utils  # type: ignore
    carla = carla_utils.carla
except Exception:  # noqa: BLE001
    import carla  # type: ignore


@dataclass
class ControllerConfig:
    mode: str = "perception_closed_loop"
    target_speed_kmh: float = 25.0
    lookahead_m: float = 6.0
    max_steer: float = 0.7
    steer_kp: float = 0.9
    throttle_kp: float = 0.5
    max_throttle: float = 0.65
    brake_distance_m: float = 12.0
    slow_distance_m: float = 22.0
    slow_throttle: float = 0.2
    stop_speed_threshold_mps: float = 0.3
    emergency_brake: float = 1.0

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "ControllerConfig":
        raw = dict(raw or {})
        cfg = cls()
        for key, value in raw.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


@dataclass
class ControlStepResult:
    control: "carla.VehicleControl"
    ego_speed_mps: float
    throttle: float
    brake: float
    steer: float
    reason: str


def _vehicle_speed_mps(vehicle) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _yaw_rad(transform) -> float:
    return math.radians(float(transform.rotation.yaw))


def _wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class PerceptionDrivingController:
    """Closed-loop controller that drives the ego using perception output."""

    def __init__(self, world, vehicle, config: ControllerConfig, logger=None) -> None:
        self.world = world
        self.vehicle = vehicle
        self.cfg = config
        self.logger = logger
        self.carla_map = world.get_map()
        self._first_log = True

    # ------------------------------------------------------------------ lateral
    def _next_waypoint(self):
        loc = self.vehicle.get_transform().location
        current_wp = self.carla_map.get_waypoint(
            loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if current_wp is None:
            return None
        candidates = current_wp.next(float(self.cfg.lookahead_m))
        if not candidates:
            return current_wp
        return candidates[0]

    def _steer_to_waypoint(self, waypoint) -> float:
        ego_tf = self.vehicle.get_transform()
        ego_loc = ego_tf.location
        tgt_loc = waypoint.transform.location

        dx = tgt_loc.x - ego_loc.x
        dy = tgt_loc.y - ego_loc.y

        target_yaw = math.atan2(dy, dx)
        ego_yaw = _yaw_rad(ego_tf)
        error = _wrap_pi(target_yaw - ego_yaw)

        steer = self.cfg.steer_kp * error
        return max(-self.cfg.max_steer, min(self.cfg.max_steer, steer))

    # -------------------------------------------------------------- longitudinal
    def _longitudinal_command(
        self,
        decision: str,
        front_distance_m: float,
        speed_mps: float,
    ) -> (float, float, str):
        target_mps = float(self.cfg.target_speed_kmh) / 3.6

        decision_upper = str(decision or "").upper()
        has_obstacle = not (front_distance_m is None or math.isnan(front_distance_m))

        if decision_upper.startswith("BRAKE") or (
            has_obstacle and front_distance_m <= self.cfg.brake_distance_m
        ):
            return 0.0, float(self.cfg.emergency_brake), "brake"

        if decision_upper.startswith("SLOW") or (
            has_obstacle and front_distance_m <= self.cfg.slow_distance_m
        ):
            throttle = float(self.cfg.slow_throttle) if speed_mps < (target_mps * 0.6) else 0.0
            brake = 0.2 if speed_mps > (target_mps * 0.8) else 0.0
            return throttle, brake, "slow"

        error = target_mps - speed_mps
        throttle = max(0.0, min(self.cfg.max_throttle, self.cfg.throttle_kp * error))
        brake = 0.0
        if speed_mps > target_mps * 1.15:
            throttle = 0.0
            brake = 0.3
        return throttle, brake, "cruise"

    # ----------------------------------------------------------------------- API
    def step(
        self,
        decision: str,
        front_distance_m: float,
    ) -> ControlStepResult:
        speed = _vehicle_speed_mps(self.vehicle)

        throttle, brake, reason = self._longitudinal_command(decision, front_distance_m, speed)

        steer = 0.0
        wp = self._next_waypoint()
        if wp is not None:
            steer = self._steer_to_waypoint(wp)

        control = carla.VehicleControl()
        control.throttle = float(max(0.0, min(1.0, throttle)))
        control.brake = float(max(0.0, min(1.0, brake)))
        control.steer = float(max(-1.0, min(1.0, steer)))
        control.hand_brake = False
        control.manual_gear_shift = False
        control.reverse = False

        self.vehicle.apply_control(control)

        if self._first_log and self.logger is not None:
            self.logger.info(
                "PerceptionDrivingController active target_kmh=%.1f brake_dist=%.1f slow_dist=%.1f",
                self.cfg.target_speed_kmh,
                self.cfg.brake_distance_m,
                self.cfg.slow_distance_m,
            )
            self._first_log = False

        return ControlStepResult(
            control=control,
            ego_speed_mps=float(speed),
            throttle=float(control.throttle),
            brake=float(control.brake),
            steer=float(control.steer),
            reason=reason,
        )
