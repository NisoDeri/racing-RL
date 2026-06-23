"""
Phase 5 curriculum opponents.

Each stage of the multi-car curriculum is characterized by an :class:`OpponentSpec`
that names the opponent driving mode, the count, the relative speed, and the
spawn offsets along the centerline ahead of the ego car.

The :class:`Opponent` class wraps a Box2D :class:`Car` driven kinematically by
the env (positions and velocities are set directly each step), mirroring the
static control car in ``main.py``. Opponents bypass the physics force loop so
collisions with the ego car do not perturb them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from src.physics.car import Car
from src.track.track import Track


OpponentMode = Literal["stationary", "centerline_follower"]


@dataclass(frozen=True)
class OpponentSpec:
    """Configuration for one curriculum stage's opponent population."""

    mode: OpponentMode
    count: int
    speed_fraction: float
    spawn_offsets: Tuple[float, ...]

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be non-negative")
        if not 0.0 <= self.speed_fraction <= 1.0:
            raise ValueError("speed_fraction must be in [0, 1]")
        if len(self.spawn_offsets) != self.count:
            raise ValueError("spawn_offsets length must match count")
        if any(offset <= 0.0 for offset in self.spawn_offsets):
            raise ValueError("spawn_offsets must be strictly positive (ahead of ego)")


class Opponent:
    """Kinematic car driven by the env (not by Box2D forces).

    ``stationary``  → position and velocity never change.
    ``centerline_follower`` → advances along the track centerline at
    ``speed`` m/s every step (mirrors the static control car block in
    ``main.py``).
    """

    def __init__(self, car: Car, mode: OpponentMode, speed: float, initial_s: float) -> None:
        if mode not in ("stationary", "centerline_follower"):
            raise ValueError(f"Unknown opponent mode: {mode}")
        if speed < 0.0:
            raise ValueError("speed must be non-negative")
        self.car = car
        self.mode = mode
        self.speed = float(speed)
        self.s = float(initial_s)

    @property
    def position(self) -> np.ndarray:
        return self.car.position

    def update(self, track: Track, dt: float) -> None:
        if self.mode == "stationary":
            self.car.body.linearVelocity = (0.0, 0.0)
            self.car.body.angularVelocity = 0.0
            return

        # centerline_follower
        self.s = (self.s + self.speed * dt) % track.total_length
        pos, heading, _ = track.get_pose_at_s(self.s)
        self.car.body.position = (float(pos[0]), float(pos[1]))
        self.car.body.angle = float(heading)
        forward = np.array([np.cos(heading), np.sin(heading)]) * self.speed
        self.car.body.linearVelocity = (float(forward[0]), float(forward[1]))
        self.car.body.angularVelocity = 0.0


CURRICULUM_OPPONENTS: dict[str, OpponentSpec] = {
    # 5a: no opponents — kept for API symmetry. Phase-4 model is reused here.
    "5a": OpponentSpec(mode="stationary", count=0, speed_fraction=0.0, spawn_offsets=()),
    # 5b: one stationary obstacle 40m ahead of ego.
    "5b": OpponentSpec(mode="stationary", count=1, speed_fraction=0.0, spawn_offsets=(40.0,)),
    # 5c: one centerline follower at 50% target speed, 40m ahead.
    "5c": OpponentSpec(
        mode="centerline_follower",
        count=1,
        speed_fraction=0.5,
        spawn_offsets=(40.0,),
    ),
    # 5d: three centerline followers at 50% target speed, evenly spread ahead.
    "5d": OpponentSpec(
        mode="centerline_follower",
        count=3,
        speed_fraction=0.5,
        spawn_offsets=(35.0, 80.0, 125.0),
    ),
}


CURRICULUM_STAGES: tuple[str, ...] = tuple(CURRICULUM_OPPONENTS.keys())
