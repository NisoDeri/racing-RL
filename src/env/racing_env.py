"""
Gymnasium Racing Environment.

Phase 1 RL contract:
  Observation:
    [30 normalized ray distances, normalized speed, normalized lateral velocity,
     last throttle, last steering]

  Action:
    [throttle, steering], both in [-1, 1]

  Reward profiles:
    v1 preserves the Phase 2 baseline.
    v2 adds forward-only speed, time and wall-contact costs, and backwards
    driving termination for the Phase 3 reward-shaping experiment.
    v3 extends v2 with car-aware penalties and an overtaking bonus for the
    Phase 5 multi-car curriculum.
"""
import os
import sys
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config import SIM, RACE, SENSOR, RENDER, CAR
from src.physics.world import World
from src.physics.car import Car
from src.track.track import Track
from src.rendering.renderer import Renderer
from src.sensors.sensor import RayCaster
from src.env.opponents import (
    CURRICULUM_OPPONENTS,
    CURRICULUM_STAGES,
    Opponent,
    OpponentSpec,
    PolicyOpponent,
)


_SPEED_NORM = 95.0
_LAT_VEL_NORM = 50.0


@dataclass(frozen=True)
class RewardConfig:
    """Weights and safeguards for one reward-shaping experiment."""

    progress_weight: float = 1.0
    speed_weight: float = 0.01
    lateral_weight: float = 0.1
    heading_weight: float = 0.05
    steering_smoothness_weight: float = 0.5
    wall_hit_penalty: float = 10.0
    off_track_penalty: float = 50.0
    time_penalty: float = 0.0
    wall_contact_penalty: float = 0.0
    backwards_terminal_penalty: float = 0.0
    forward_speed_only: bool = False
    reward_while_touching_wall: bool = True
    max_backwards_steps: int | None = None
    backwards_progress_threshold: float = -0.01
    # Phase 5 car-aware terms; default to 0.0 so v1/v2 stay byte-identical.
    car_hit_penalty: float = 0.0
    car_contact_penalty: float = 0.0
    overtake_bonus: float = 0.0


REWARD_PROFILES = {
    # Exact Phase 1/2 reward, kept for reproducible before/after experiments.
    "v1": RewardConfig(),
    # Phase 3 fixes for reversing, wall scraping, and camping.
    "v2": RewardConfig(
        time_penalty=0.001,
        wall_contact_penalty=0.25,
        backwards_terminal_penalty=25.0,
        forward_speed_only=True,
        reward_while_touching_wall=False,
        max_backwards_steps=120,
    ),
    # Phase 5 multi-car: v2 + car-aware penalties and overtaking bonus.
    "v3": RewardConfig(
        time_penalty=0.001,
        wall_contact_penalty=0.25,
        backwards_terminal_penalty=25.0,
        forward_speed_only=True,
        reward_while_touching_wall=False,
        max_backwards_steps=120,
        car_hit_penalty=5.0,
        car_contact_penalty=0.15,
        overtake_bonus=2.0,
    ),
}


class RacingEnv(gym.Env):
    """
    Top-down racing environment for reinforcement learning.

    The policy observes raycast distances plus ego proprioception. Frenet
    coordinates are intentionally kept internal for reward shaping and metrics.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        track_creator=None,
        track_generator=None,
        randomize_start=False,
        start_lateral_jitter=0.1,
        start_heading_jitter=np.radians(5.0),
        max_episode_steps=6000,
        reward_profile=None,
        reward_config=None,
        opponent_spec=None,
        pool_dir: str = "",
    ):
        """
        Args:
            render_mode: "human" for Pygame window, "rgb_array" for pixel array,
                None for headless.
            track_creator: Optional callable that returns a Track instance. If
                None, uses Track.create_complex_track().
            track_generator: Optional callable accepting the seeded Gymnasium RNG
                and returning a new Track. Used for per-episode randomization.
            randomize_start: Randomize longitudinal, lateral, and heading start pose.
            start_lateral_jitter: Maximum lateral offset as a fraction of half-width.
            start_heading_jitter: Maximum absolute heading offset in radians.
            max_episode_steps: Max steps before truncation, about 100 seconds
                at 60 FPS.
            reward_profile: Named reward configuration ("v1", "v2", or "v3").
                Defaults to the Phase 3 "v2" profile.
            reward_config: Optional RewardConfig override for experiments/tests.
            opponent_spec: Phase 5 OpponentSpec describing the kinematic
                opponents to spawn each episode. Defaults to no opponents.
        """
        super().__init__()

        if track_creator is not None and track_generator is not None:
            raise ValueError("Pass either track_creator or track_generator, not both")
        if not 0.0 <= start_lateral_jitter < 1.0:
            raise ValueError("start_lateral_jitter must be in [0, 1)")
        if start_heading_jitter < 0.0:
            raise ValueError("start_heading_jitter must be non-negative")

        if reward_config is not None and reward_profile is not None:
            raise ValueError("Pass either reward_profile or reward_config, not both")
        if reward_config is not None:
            if not isinstance(reward_config, RewardConfig):
                raise TypeError("reward_config must be a RewardConfig")
            self.reward_profile = "custom"
            self.reward_config = reward_config
        else:
            reward_profile = reward_profile or "v2"
            try:
                self.reward_config = REWARD_PROFILES[reward_profile]
            except KeyError as exc:
                choices = ", ".join(sorted(REWARD_PROFILES))
                raise ValueError(
                    f"Unknown reward_profile {reward_profile!r}; choose from {choices}"
                ) from exc
            self.reward_profile = reward_profile

        if opponent_spec is not None and not isinstance(opponent_spec, OpponentSpec):
            raise TypeError("opponent_spec must be an OpponentSpec")
        self.opponent_spec = opponent_spec or CURRICULUM_OPPONENTS["5a"]
        if len(self.opponent_spec.spawn_offsets) >= 2:
            gaps = np.diff(np.asarray(self.opponent_spec.spawn_offsets, dtype=np.float64))
            if np.any(gaps < RACE.curriculum_min_opponent_gap):
                raise ValueError(
                    "opponent_spec.spawn_offsets must be at least "
                    f"{RACE.curriculum_min_opponent_gap} m apart"
                )

        self.pool_dir = pool_dir
        self._pool = None  # lazily initialized on first pool_agent reset
        self.render_mode = render_mode
        self.track_creator = track_creator
        self.track_generator = track_generator
        self.randomize_start = randomize_start
        self.start_lateral_jitter = start_lateral_jitter
        self.start_heading_jitter = start_heading_jitter
        self.max_episode_steps = max_episode_steps

        self.raycaster = RayCaster(
            num_forward_rays=SENSOR.num_forward_rays,
            forward_spread=SENSOR.forward_spread,
            num_mirror_rays=SENSOR.num_mirror_rays,
            mirror_start=SENSOR.mirror_angle_start,
            mirror_end=SENSOR.mirror_angle_end,
            max_distance=SENSOR.max_ray_distance,
        )

        self.obs_dim = self.raycaster.num_rays + 4
        low = np.zeros(self.obs_dim, dtype=np.float32)
        high = np.ones(self.obs_dim, dtype=np.float32)
        speed_idx = self.raycaster.num_rays
        lateral_idx = speed_idx + 1
        low[lateral_idx] = -2.0
        high[lateral_idx] = 2.0
        low[lateral_idx + 1:] = -1.0
        high[speed_idx] = 2.0

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.world = None
        self.car = None
        self.track = None
        self.renderer = None
        self.window_closed = False
        self.inner_boundary = None
        self.outer_boundary = None
        self.opponents: list = []

        self.steps = 0
        self.prev_s = 0.0
        self.total_progress = 0.0
        self.laps_completed = 0
        self.last_throttle = 0.0
        self.last_steering = 0.0
        self.prev_steering = 0.0
        self.prev_wall_hits = 0
        self.prev_car_collisions = 0
        self.stuck_wall_steps = 0
        self.backwards_steps = 0
        self.total_abs_steering_change = 0.0
        self.car_contact_steps = 0
        self.overtake_count = 0
        self.prev_lead_count = 0
        self.termination_reason = None

        self.last_ray_distances = None
        self.last_ray_hits = None
        self.last_reward_terms = {}
        self.last_wall_hit_this_step = False
        self.last_car_hit_this_step = False

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Returns:
            observation: Initial state observation.
            info: Additional information dict.
        """
        super().reset(seed=seed)

        self.world = World()

        if self.track_generator:
            self.track = self.track_generator(self.np_random)
        elif self.track_creator:
            self.track = self.track_creator()
        else:
            self.track = Track.create_complex_track(track_width=14)

        self.inner_boundary, self.outer_boundary = self.track.get_boundary_points()
        self.track.create_walls(self.world)
        self.world.collision_handler.ignore_car_collision_count_until_step = (
            RACE.startup_collision_grace_steps
        )

        start_s, lateral_offset, heading_offset = self._get_start_state(options)
        start_pos, start_heading, start_segment = self.track.get_pose_at_s(start_s)
        start_pos = start_pos + self.track.normals[start_segment] * lateral_offset
        start_heading += heading_offset
        self.car = Car(self.world, position=start_pos, angle=start_heading, car_id=0,
                       is_main_player=True)

        self.opponents = self._spawn_opponents(start_s)
        for opp in self.opponents:
            if isinstance(opp, PolicyOpponent):
                opp.reset_obs_buffer()

        self.steps = 0
        self.prev_s = start_s
        self.total_progress = 0.0
        self.laps_completed = 0
        self.last_throttle = 0.0
        self.last_steering = 0.0
        self.prev_steering = 0.0
        self.prev_wall_hits = 0
        self.prev_car_collisions = 0
        self.stuck_wall_steps = 0
        self.backwards_steps = 0
        self.total_abs_steering_change = 0.0
        self.car_contact_steps = 0
        self.overtake_count = 0
        self.prev_lead_count = self._compute_lead_count(start_s)
        self.termination_reason = None
        self.episode_start_s = start_s
        self.episode_start_lateral_offset = lateral_offset
        self.episode_start_heading_offset = heading_offset
        self.last_reward_terms = {}
        self.last_wall_hit_this_step = False
        self.last_car_hit_this_step = False

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._init_renderer()
            self.render()

        return obs, info

    def _spawn_opponents(self, ego_s):
        """Spawn the curriculum-stage opponents ahead of the ego on the centerline."""
        opponents = []
        if self.opponent_spec.count == 0:
            return opponents

        L = self.track.total_length
        mode = self.opponent_spec.mode

        if mode == "pool_agent":
            from src.env.pool import CheckpointPool
            if self._pool is None:
                self._pool = CheckpointPool(self.pool_dir)
            models = self._pool.sample(n=self.opponent_spec.count, device="cpu")
            for idx, (offset, model) in enumerate(
                zip(self.opponent_spec.spawn_offsets, models)
            ):
                s_i = (ego_s + offset) % L
                pos, heading, _ = self.track.get_pose_at_s(s_i)
                car = Car(
                    self.world,
                    position=(float(pos[0]), float(pos[1])),
                    angle=float(heading),
                    car_id=idx + 1,
                    is_static_control=False,
                )
                opponents.append(PolicyOpponent(car=car, model=model, initial_s=float(s_i)))
        else:
            opponent_target_speed = (
                self.opponent_spec.speed_fraction * RACE.static_control_speed
            )
            for idx, offset in enumerate(self.opponent_spec.spawn_offsets):
                s_i = (ego_s + offset) % L
                pos, heading, _ = self.track.get_pose_at_s(s_i)
                car = Car(
                    self.world,
                    position=(float(pos[0]), float(pos[1])),
                    angle=float(heading),
                    car_id=idx + 1,
                    is_static_control=True,
                )
                opponents.append(
                    Opponent(
                        car=car,
                        mode=mode,
                        speed=opponent_target_speed,
                        initial_s=float(s_i),
                    )
                )

        return opponents

    def _compute_lead_count(self, ego_s):
        """Number of opponents currently behind the ego in lap-relative terms."""
        if not self.opponents:
            return 0
        L = self.track.total_length
        count = 0
        for opp in self.opponents:
            ds = (ego_s - opp.s) % L
            # ds in (0, L/2] means ego is ahead of this opponent within half a lap.
            if 0.0 < ds <= L / 2.0:
                count += 1
        return count

    def _get_start_state(self, options):
        options = options or {}
        if self.randomize_start:
            default_s = self.np_random.uniform(0.0, self.track.total_length)
            lateral_limit = self.start_lateral_jitter * self.track.half_width
            default_lateral = self.np_random.uniform(-lateral_limit, lateral_limit)
            default_heading = self.np_random.uniform(
                -self.start_heading_jitter, self.start_heading_jitter
            )
        else:
            default_s = 0.0
            default_lateral = 0.0
            default_heading = 0.0

        start_s = float(options.get("start_s", default_s)) % self.track.total_length
        lateral_offset = float(options.get("lateral_offset", default_lateral))
        heading_offset = float(options.get("heading_offset", default_heading))
        if not np.all(np.isfinite([start_s, lateral_offset, heading_offset])):
            raise ValueError("Start pose values must be finite")
        if abs(lateral_offset) >= self.track.half_width:
            raise ValueError("lateral_offset must remain within the track")
        return start_s, lateral_offset, heading_offset

    def step(self, action):
        """
        Execute one environment step.

        Args:
            action: [throttle, steering], both in [-1, 1].

        Returns:
            observation, reward, terminated, truncated, info.
        """
        prev_steering = self.prev_steering
        throttle = float(np.clip(action[0], -1.0, 1.0))
        steering = float(np.clip(action[1], -1.0, 1.0))
        self.total_abs_steering_change += abs(steering - prev_steering)
        self.car.set_controls(throttle, steering)

        all_cars = [self.car] + [opp.car for opp in self.opponents]
        for opp in self.opponents:
            if isinstance(opp, PolicyOpponent):
                opp.update(
                    self.track, SIM.time_step,
                    inner_boundary=self.inner_boundary,
                    outer_boundary=self.outer_boundary,
                    all_cars=all_cars,
                    raycaster=self.raycaster,
                )
            else:
                opp.update(self.track, SIM.time_step)

        self.car.update()
        self.world.step()
        self.steps += 1

        frenet = self.track.get_frenet_coordinates(self.car.position, self.car.angle)
        on_track = self.track.is_inside_track(self.car.position)
        ds = self._compute_progress_delta(frenet['s'])

        self.total_progress += ds
        if self.total_progress >= self.track.total_length * (self.laps_completed + 1):
            self.laps_completed += 1
        self.prev_s = frenet['s']

        wall_stats = self.world.collision_handler.get_car_stats(self.car.car_id)
        wall_hits = int(wall_stats['wall_hit_count'])
        new_wall_hits = max(0, wall_hits - self.prev_wall_hits)
        self.prev_wall_hits = wall_hits
        self.last_wall_hit_this_step = new_wall_hits > 0

        car_collisions = int(wall_stats['car_collision_count'])
        new_car_hits = max(0, car_collisions - self.prev_car_collisions)
        self.prev_car_collisions = car_collisions
        self.last_car_hit_this_step = new_car_hits > 0
        touching_car = bool(wall_stats['touching_car'])
        if touching_car:
            self.car_contact_steps += 1

        current_lead_count = self._compute_lead_count(frenet['s'])
        lead_delta = max(0, current_lead_count - self.prev_lead_count)
        self.overtake_count += lead_delta
        self.prev_lead_count = current_lead_count

        if ds < self.reward_config.backwards_progress_threshold:
            self.backwards_steps += 1
        else:
            self.backwards_steps = 0

        backwards_terminated = (
            self.reward_config.max_backwards_steps is not None
            and self.backwards_steps >= self.reward_config.max_backwards_steps
        )

        reward, reward_terms = self._compute_reward(
            frenet=frenet,
            on_track=on_track,
            ds=ds,
            steering=steering,
            prev_steering=prev_steering,
            new_wall_hits=new_wall_hits,
            touching_wall=bool(wall_stats['touching_wall']),
            backwards_terminated=backwards_terminated,
            new_car_hits=new_car_hits,
            touching_car=touching_car,
            overtakes_this_step=lead_delta,
        )
        self.last_reward_terms = reward_terms

        terminated = False
        truncated = False

        if not on_track:
            terminated = True
            self.termination_reason = "off_track"

        if wall_stats['touching_wall'] and max(ds, 0.0) < 0.01:
            self.stuck_wall_steps += 1
        else:
            self.stuck_wall_steps = 0

        if self.stuck_wall_steps >= 30:
            terminated = True
            self.termination_reason = "stuck_wall"

        if backwards_terminated:
            terminated = True
            self.termination_reason = "driving_backwards"

        if self.steps >= self.max_episode_steps:
            truncated = True
            if not terminated:
                self.termination_reason = "max_steps"

        self.last_throttle = throttle
        self.last_steering = steering
        self.prev_steering = steering

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _compute_progress_delta(self, current_s):
        ds = current_s - self.prev_s

        if ds < -self.track.total_length / 2:
            ds += self.track.total_length
        elif ds > self.track.total_length / 2:
            ds -= self.track.total_length

        return ds

    def _get_observation(self):
        """
        Build the raycast-first observation vector.

        Returns:
            numpy array of shape (34,) with normalized values.
        """
        ray_kwargs = {}
        if SENSOR.detect_cars_as_obstacles and self.opponents:
            all_cars = [self.car] + [opp.car for opp in self.opponents]
            ray_kwargs = {"cars": all_cars, "ego_car": self.car}
        distances, hit_points = self.raycaster.cast(
            self.car.position,
            self.car.angle,
            self.inner_boundary,
            self.outer_boundary,
            **ray_kwargs,
        )
        self.last_ray_distances = distances
        self.last_ray_hits = hit_points

        rays = self.raycaster.get_normalized(distances)
        lateral_velocity = np.clip(self.car.get_lateral_velocity() / _LAT_VEL_NORM, -2.0, 2.0)

        return np.array(
            [
                *rays,
                self.car.speed / _SPEED_NORM,
                lateral_velocity,
                self.last_throttle,
                self.last_steering,
            ],
            dtype=np.float32,
        )

    def _compute_reward(
        self,
        frenet,
        on_track,
        ds,
        steering,
        prev_steering,
        new_wall_hits,
        touching_wall,
        backwards_terminated,
        new_car_hits=0,
        touching_car=False,
        overtakes_this_step=0,
    ):
        config = self.reward_config
        clean_driving = on_track and (
            config.reward_while_touching_wall or not touching_wall
        )
        progress_reward = (
            config.progress_weight * max(ds, 0.0) if clean_driving else 0.0
        )
        speed = (
            max(self.car.get_forward_velocity(), 0.0)
            if config.forward_speed_only
            else self.car.speed
        )
        speed_bonus = config.speed_weight * speed if clean_driving else 0.0
        lateral_penalty = (
            -config.lateral_weight * abs(frenet['e_y']) / self.track.half_width
        )
        heading_penalty = -config.heading_weight * abs(frenet['e_psi']) / np.pi
        steering_penalty = (
            -config.steering_smoothness_weight * abs(steering - prev_steering)
        )
        wall_penalty = -config.wall_hit_penalty * new_wall_hits
        wall_contact_penalty = (
            -config.wall_contact_penalty if touching_wall else 0.0
        )
        time_penalty = -config.time_penalty
        off_track_penalty = -config.off_track_penalty if not on_track else 0.0
        backwards_penalty = (
            -config.backwards_terminal_penalty if backwards_terminated else 0.0
        )
        car_hit_penalty = -config.car_hit_penalty * new_car_hits
        car_contact_penalty = (
            -config.car_contact_penalty if touching_car else 0.0
        )
        overtake_reward = config.overtake_bonus * overtakes_this_step

        reward_terms = {
            'progress': float(progress_reward),
            'speed': float(speed_bonus),
            'lateral': float(lateral_penalty),
            'heading': float(heading_penalty),
            'steering_smoothness': float(steering_penalty),
            'wall_hit': float(wall_penalty),
            'wall_contact': float(wall_contact_penalty),
            'time': float(time_penalty),
            'off_track': float(off_track_penalty),
            'backwards': float(backwards_penalty),
            'car_hit': float(car_hit_penalty),
            'car_contact': float(car_contact_penalty),
            'overtake': float(overtake_reward),
        }

        return float(sum(reward_terms.values())), reward_terms

    def _get_info(self):
        """Return additional info dict."""
        frenet = self.track.get_frenet_coordinates(self.car.position, self.car.angle)
        wall_stats = self.world.collision_handler.get_car_stats(self.car.car_id)
        ray_min_distance = (
            float(np.min(self.last_ray_distances))
            if self.last_ray_distances is not None
            else SENSOR.max_ray_distance
        )

        return {
            'speed': float(self.car.speed),
            'forward_velocity': float(self.car.get_forward_velocity()),
            'speed_kmh': float(self.car.speed * 3.6),
            's': float(frenet['s']),
            'e_y': float(frenet['e_y']),
            'e_psi': float(frenet['e_psi']),
            'steps': int(self.steps),
            'on_track': bool(self.track.is_inside_track(self.car.position)),
            'laps': int(self.laps_completed),
            'total_progress': float(self.total_progress),
            'progress_fraction': float(self.total_progress / self.track.total_length),
            'wall_hits': int(wall_stats['wall_hit_count']),
            'wall_hit_this_step': bool(self.last_wall_hit_this_step),
            'car_collisions': int(wall_stats['car_collision_count']),
            'car_hit_this_step': bool(self.last_car_hit_this_step),
            'touching_car': bool(wall_stats['touching_car']),
            'car_contact_steps': int(self.car_contact_steps),
            'overtake_count': int(self.overtake_count),
            'num_opponents': len(self.opponents),
            'opponent_mode': self.opponent_spec.mode,
            'ray_min_distance': ray_min_distance,
            'lateral_velocity': float(self.car.get_lateral_velocity()),
            'last_throttle': float(self.last_throttle),
            'last_steering': float(self.last_steering),
            'prev_steering': float(self.prev_steering),
            'backwards_steps': int(self.backwards_steps),
            'mean_abs_steering_change': float(
                self.total_abs_steering_change / max(self.steps, 1)
            ),
            'termination_reason': self.termination_reason,
            'reward_profile': self.reward_profile,
            'track_name': self.track.name,
            'track_seed': self.track.generation_seed,
            'track_length': float(self.track.total_length),
            'track_width': float(self.track.width),
            'start_s': float(self.episode_start_s),
            'start_lateral_offset': float(self.episode_start_lateral_offset),
            'start_heading_offset': float(self.episode_start_heading_offset),
            'reward_terms': self.last_reward_terms,
        }

    def _init_renderer(self):
        """Initialize the Pygame renderer."""
        if self.renderer is None:
            self.renderer = Renderer()
        track_span = np.max(self.track.centerline, axis=0) - np.min(
            self.track.centerline, axis=0
        )
        max_span = max(track_span)
        self.renderer.zoom = (
            min(RENDER.screen_width, RENDER.screen_height)
            / (max_span + 50)
            / SIM.pixels_per_meter
        )

    def render(self):
        """
        Render the current state.

        For render_mode="human": displays in a Pygame window.
        For render_mode="rgb_array": returns a pixel array.
        """
        if self.render_mode is None or self.window_closed:
            return None

        if self.renderer is None:
            self._init_renderer()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.window_closed = True
                self.close()
                return None

        self.renderer.set_camera(self.car.position[0], self.car.position[1])
        self.renderer.clear()
        self.renderer.draw_track(self.track)
        for opp in self.opponents:
            self.renderer.draw_car(opp.car)
        self.renderer.draw_car(self.car)

        if self.last_ray_distances is not None and self.last_ray_hits is not None:
            self.renderer.draw_rays(
                self.car.position,
                self.last_ray_distances,
                self.last_ray_hits,
                self.raycaster.max_distance,
                is_mirror=self.raycaster.is_mirror,
            )

        frenet = self.track.get_frenet_coordinates(self.car.position, self.car.angle)
        self.renderer.draw_frenet_debug(self.car, frenet)
        self.renderer.draw_hud(self.car, frenet)

        on_track = self.track.is_inside_track(self.car.position)
        status_color = (0, 255, 0) if on_track else (255, 0, 0)
        status_text = "ON TRACK" if on_track else "OFF TRACK!"
        self.renderer._draw_text(status_text, (RENDER.screen_width - 120, 10), status_color)
        self.renderer._draw_text(
            f"Step: {self.steps} | Laps: {self.laps_completed} | "
            f"Progress: {self.total_progress:.0f}m",
            (RENDER.screen_width - 400, 35),
            (200, 200, 200),
        )

        self.renderer.update()

        if self.render_mode == "rgb_array":
            return np.array(pygame.surfarray.array3d(self.renderer.screen))

        self.renderer.tick(self.metadata["render_fps"])
        return None

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.quit()
            self.renderer = None
