"""
Phase 5 curriculum opponents.

Each stage of the multi-car curriculum is characterized by an :class:`OpponentSpec`
that names the opponent driving mode, the count, the relative speed, and the
spawn offsets along the centerline ahead of the ego car.

The :class:`Opponent` class wraps a Box2D :class:`Car` driven kinematically by
the env (positions and velocities are set directly each step), mirroring the
static control car in ``main.py``. Opponents bypass the physics force loop so
collisions with the ego car do not perturb them.

The :class:`PolicyOpponent` class (Phase 5e) drives via a PPO checkpoint sampled
from the pool. It is physics-driven (Box2D forces apply) so it can crash, spin
out, and interact realistically with the ego.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from src.physics.car import Car
from src.track.track import Track


OpponentMode = Literal["stationary", "centerline_follower", "pool_agent"]

# Must match the values in racing_env.py.
_SPEED_NORM: float = 95.0
_LAT_VEL_NORM: float = 50.0
_N_STACK: int = 4
_OBS_DIM: int = 34  # single-frame observation dimension


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
        self._fixed_position = tuple(float(v) for v in car.position)
        self._fixed_angle = float(car.angle)

    @property
    def position(self) -> np.ndarray:
        return self.car.position

    def update(self, track: Track, dt: float) -> None:
        if self.mode == "stationary":
            self.car.body.position = self._fixed_position
            self.car.body.angle = self._fixed_angle
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


class PolicyOpponent:
    """Physics-driven opponent controlled by a PPO checkpoint from the pool.

    Unlike :class:`Opponent`, the car is NOT kinematic: Box2D forces apply
    via ``car.set_controls() + car.update()`` each step.  The opponent can
    crash, spin out, and interact realistically with the ego.

    Each instance maintains its own frame-stack buffer (shape N_STACK × OBS_DIM)
    so the policy sees the same observation structure it was trained with.
    Per-process model caching in :class:`CheckpointPool` ensures that distinct
    subprocesses (SubprocVecEnv) each maintain their own copy without IPC.
    """

    def __init__(self, car: Car, model, initial_s: float) -> None:
        self.car = car
        self.model = model
        self.s = float(initial_s)
        self._obs_buf = np.zeros((_N_STACK, _OBS_DIM), dtype=np.float32)
        self._last_action = np.zeros(2, dtype=np.float32)

    @property
    def position(self) -> np.ndarray:
        return np.array(self.car.position)

    def reset_obs_buffer(self) -> None:
        """Zero the frame stack and last action on episode reset."""
        self._obs_buf[:] = 0.0
        self._last_action[:] = 0.0

    def update(
        self,
        track: Track,
        dt: float,  # unused; kept for API symmetry with Opponent
        *,
        inner_boundary=None,
        outer_boundary=None,
        all_cars=None,
        raycaster=None,
    ) -> None:
        """Predict action from current obs, apply forces, update Frenet s."""
        obs_raw = self._compute_obs(inner_boundary, outer_boundary, all_cars, raycaster)
        # Roll oldest frame out and push new obs in.
        self._obs_buf = np.roll(self._obs_buf, -1, axis=0)
        self._obs_buf[-1] = obs_raw
        stacked = self._obs_buf.flatten()[None]  # (1, N_STACK * OBS_DIM)
        action, _ = self.model.predict(stacked, deterministic=True)
        throttle = float(action[0, 0])
        steering = float(action[0, 1])
        self._last_action = np.array([throttle, steering], dtype=np.float32)
        self.car.set_controls(throttle, steering)
        self.car.update()
        # Keep Frenet s current so _compute_lead_count in the env stays accurate.
        frenet = track.get_frenet_coordinates(self.car.position, self.car.angle)
        self.s = float(frenet["s"])

    def _compute_obs(
        self,
        inner_boundary,
        outer_boundary,
        all_cars,
        raycaster,
    ) -> np.ndarray:
        distances, _ = raycaster.cast(
            self.car.position,
            self.car.angle,
            inner_boundary,
            outer_boundary,
            cars=all_cars,
            ego_car=self.car,
        )
        rays = raycaster.get_normalized(distances)
        lat_vel = np.clip(
            self.car.get_lateral_velocity() / _LAT_VEL_NORM, -2.0, 2.0
        )
        return np.concatenate([
            rays,
            [self.car.speed / _SPEED_NORM],
            [lat_vel],
            self._last_action,
        ]).astype(np.float32)


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
    # 5e: three policy opponents sampled from the checkpoint pool each episode.
    "5e": OpponentSpec(
        mode="pool_agent",
        count=3,
        speed_fraction=0.0,  # unused — physics controls speed for pool agents
        spawn_offsets=(35.0, 80.0, 125.0),
    ),
}


CURRICULUM_STAGES: tuple[str, ...] = tuple(CURRICULUM_OPPONENTS.keys())
