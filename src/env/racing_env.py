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
"""
import os
import sys
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config import SIM, RACE, SENSOR, RENDER
from src.physics.world import World
from src.physics.car import Car
from src.track.track import Track
from src.rendering.renderer import Renderer
from src.sensors.sensor import RayCaster
from src.env.opponents import Opponent, OpponentSpec, PolicyOpponent
from src.env.pool import CheckpointPool


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
    car_hit_penalty: float = 0.0
    car_contact_penalty: float = 0.0
    backwards_terminal_penalty: float = 0.0
    forward_speed_only: bool = False
    reward_while_touching_wall: bool = True
    max_backwards_steps: int | None = None
    backwards_progress_threshold: float = -0.01


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
    # Phase 5: v2 shaping plus car-aware contact cost for multi-car racing.
    "v3": RewardConfig(
        time_penalty=0.001,
        wall_contact_penalty=0.25,
        car_contact_penalty=2.0,
        backwards_terminal_penalty=25.0,
        forward_speed_only=True,
        reward_while_touching_wall=False,
        max_backwards_steps=120,
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
        pool_dir=None,
        pool_device="cpu",
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
            reward_profile: Named reward configuration ("v1" or "v2"). Defaults
                to the Phase 3 "v2" profile.
            reward_config: Optional RewardConfig override for experiments/tests.
            opponent_spec: Optional OpponentSpec for Phase 5 multi-car curricula.
            pool_dir: Checkpoint directory for "pool_agent" opponents.
            pool_device: Device used when loading pooled PPO checkpoint opponents.
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

        self.render_mode = render_mode
        self.track_creator = track_creator
        self.track_generator = track_generator
        self.randomize_start = randomize_start
        self.start_lateral_jitter = start_lateral_jitter
        self.start_heading_jitter = start_heading_jitter
        self.max_episode_steps = max_episode_steps
        if opponent_spec is not None and not isinstance(opponent_spec, OpponentSpec):
            raise TypeError("opponent_spec must be an OpponentSpec")
        if (
            opponent_spec is not None
            and opponent_spec.mode == "pool_agent"
            and pool_dir is None
        ):
            raise ValueError("pool_dir is required for pool_agent opponents")
        self.opponent_spec = opponent_spec
        self.pool_dir = pool_dir
        self.pool_device = pool_device
        self.checkpoint_pool = (
            CheckpointPool(pool_dir)
            if opponent_spec is not None and opponent_spec.mode == "pool_agent"
            else None
        )

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
        self.follow_camera = True
        self.show_sensors = True
        self.show_checkpoints = True
        self.show_minimap = False
        self.camera_pan_speed = 60.0
        self.zoom_multiplier = 1.12
        self.min_zoom = 0.05
        self.max_zoom = 5.0
        self.inner_boundary = None
        self.outer_boundary = None
        self.opponents = []

        self.steps = 0
        self.prev_s = 0.0
        self.prev_lead_count = 0
        self.overtake_count = 0
        self.car_contact_steps = 0
        self.total_progress = 0.0
        self.laps_completed = 0
        self.lap_step_marks = []
        self.last_throttle = 0.0
        self.last_steering = 0.0
        self.prev_steering = 0.0
        self.prev_wall_hits = 0
        self.prev_car_collisions = 0
        self.stuck_wall_steps = 0
        self.backwards_steps = 0
        self.total_abs_steering_change = 0.0
        self.termination_reason = None

        self.last_ray_distances = None
        self.last_ray_hits = None
        self.last_reward_terms = {}
        self.last_wall_hit_this_step = False
        self._cached_frenet = None
        self._cached_on_track = True
        self._minimap_race_state = {}

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
        self.car = Car(self.world, position=start_pos, angle=start_heading)
        self.opponents = self._spawn_opponents(start_s)

        self.steps = 0
        self.prev_s = start_s
        self.prev_lead_count = self._compute_lead_count(start_s)
        self.overtake_count = 0
        self.car_contact_steps = 0
        self.total_progress = 0.0
        self.laps_completed = 0
        self.lap_step_marks = []
        self.last_throttle = 0.0
        self.last_steering = 0.0
        self.prev_steering = 0.0
        self.prev_wall_hits = 0
        self.prev_car_collisions = 0
        self.stuck_wall_steps = 0
        self.backwards_steps = 0
        self.total_abs_steering_change = 0.0
        self.termination_reason = None
        self.episode_start_s = start_s
        self.episode_start_lateral_offset = lateral_offset
        self.episode_start_heading_offset = heading_offset
        self.last_reward_terms = {}
        self.last_wall_hit_this_step = False
        self._cached_frenet = self.track.get_frenet_coordinates(
            self.car.position, self.car.angle
        )
        self._cached_on_track = self.track.is_inside_track(self.car.position)
        self._minimap_race_state = {}

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._init_renderer()
            self.render()

        return obs, info

    def _spawn_opponents(self, start_s):
        spec = self.opponent_spec
        if spec is None or spec.count == 0:
            return []
        if spec.mode == "pool_agent":
            if self.checkpoint_pool is None:
                raise ValueError("pool_dir is required for pool_agent opponents")
            models = self.checkpoint_pool.sample(spec.count, device=self.pool_device)
        else:
            models = [None] * spec.count

        opponents = []
        speed = spec.speed_fraction * RACE.static_control_speed
        for index, (offset, model) in enumerate(zip(spec.spawn_offsets, models), start=1):
            opp_s = (start_s + offset) % self.track.total_length
            pos, heading, _ = self.track.get_pose_at_s(opp_s)
            car = Car(
                self.world,
                position=pos,
                angle=heading,
                car_id=index,
                is_static_control=(spec.mode != "pool_agent"),
            )
            if spec.mode == "pool_agent":
                opponent = PolicyOpponent(car, model, opp_s)
                opponent.reset_obs_buffer()
            else:
                opponent = Opponent(car, spec.mode, speed, opp_s)
            opponents.append(opponent)
        return opponents

    def _all_cars(self):
        cars = [self.car] if self.car is not None else []
        cars.extend(opp.car for opp in self.opponents)
        return cars

    def _initial_minimap_progress(self, s):
        """Treat cars just behind s=0 as being behind the start line, not a lap ahead."""
        progress = float(s)
        if progress > self.track.total_length / 2:
            progress -= self.track.total_length
        return progress

    def _update_minimap_race_state(self):
        """Update continuous race progress used by the minimap leaderboard."""
        if self.track is None:
            return
        active_ids = set()
        for car in self._all_cars():
            frenet = self.track.get_frenet_coordinates(car.position, car.angle)
            current_s = float(frenet["s"])
            active_ids.add(car.car_id)
            state = self._minimap_race_state.get(car.car_id)
            if state is None:
                progress = self._initial_minimap_progress(current_s)
                state = {"prev_s": current_s, "progress": progress}
            else:
                ds = current_s - state["prev_s"]
                if ds < -self.track.total_length / 2:
                    ds += self.track.total_length
                elif ds > self.track.total_length / 2:
                    ds -= self.track.total_length
                state["progress"] += ds
                state["prev_s"] = current_s
            self._minimap_race_state[car.car_id] = state
            car.minimap_race_progress = float(state["progress"])
            car.minimap_laps = max(
                0,
                int(np.floor(state["progress"] / self.track.total_length)),
            )

        for car_id in list(self._minimap_race_state):
            if car_id not in active_ids:
                self._minimap_race_state.pop(car_id, None)

    def _update_opponents(self):
        for opponent in self.opponents:
            if isinstance(opponent, PolicyOpponent):
                opponent.update(
                    self.track,
                    SIM.time_step,
                    inner_boundary=self.inner_boundary,
                    outer_boundary=self.outer_boundary,
                    all_cars=self._all_cars(),
                    raycaster=self.raycaster,
                )
            else:
                opponent.update(self.track, SIM.time_step)

    def _snap_kinematic_opponents(self):
        for opponent in self.opponents:
            if not isinstance(opponent, PolicyOpponent):
                opponent.update(self.track, 0.0)

    def _compute_lead_count(self, ego_s):
        lead_count = 0
        for opponent in self.opponents:
            delta = (ego_s - opponent.s) % self.track.total_length
            if 0.0 < delta < self.track.total_length / 2:
                lead_count += 1
        return lead_count

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

        self.car.update()
        self._update_opponents()
        self.world.step()
        self._snap_kinematic_opponents()
        self.steps += 1

        frenet = self.track.get_frenet_coordinates(self.car.position, self.car.angle)
        on_track = self.track.is_inside_track(self.car.position)
        self._cached_frenet = frenet
        self._cached_on_track = on_track
        ds = self._compute_progress_delta(frenet['s'])
        lead_count = self._compute_lead_count(frenet['s'])
        if lead_count > self.prev_lead_count:
            self.overtake_count += lead_count - self.prev_lead_count
        self.prev_lead_count = lead_count

        self.total_progress += ds
        if self.total_progress >= self.track.total_length * (self.laps_completed + 1):
            self.laps_completed += 1
            self.lap_step_marks.append(self.steps)
        self.prev_s = frenet['s']

        wall_stats = self.world.collision_handler.get_car_stats(self.car.car_id)
        wall_hits = int(wall_stats['wall_hit_count'])
        new_wall_hits = max(0, wall_hits - self.prev_wall_hits)
        self.prev_wall_hits = wall_hits
        self.last_wall_hit_this_step = new_wall_hits > 0
        car_collisions = int(wall_stats['car_collision_count'])
        new_car_collisions = max(0, car_collisions - self.prev_car_collisions)
        self.prev_car_collisions = car_collisions
        touching_car = bool(wall_stats['touching_car'])
        if touching_car:
            self.car_contact_steps += 1

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
            new_car_collisions=new_car_collisions,
            touching_car=touching_car,
            backwards_terminated=backwards_terminated,
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
        distances, hit_points = self.raycaster.cast(
            self.car.position,
            self.car.angle,
            self.inner_boundary,
            self.outer_boundary,
            cars=self._all_cars() if SENSOR.detect_cars_as_obstacles else None,
            ego_car=self.car,
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
        new_car_collisions=0,
        touching_car=False,
        backwards_terminated=False,
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
        car_hit_penalty = -config.car_hit_penalty * new_car_collisions
        car_contact_penalty = -config.car_contact_penalty if touching_car else 0.0
        time_penalty = -config.time_penalty
        off_track_penalty = -config.off_track_penalty if not on_track else 0.0
        backwards_penalty = (
            -config.backwards_terminal_penalty if backwards_terminated else 0.0
        )

        reward_terms = {
            'progress': float(progress_reward),
            'speed': float(speed_bonus),
            'lateral': float(lateral_penalty),
            'heading': float(heading_penalty),
            'steering_smoothness': float(steering_penalty),
            'wall_hit': float(wall_penalty),
            'wall_contact': float(wall_contact_penalty),
            'car_hit': float(car_hit_penalty),
            'car_contact': float(car_contact_penalty),
            'time': float(time_penalty),
            'off_track': float(off_track_penalty),
            'backwards': float(backwards_penalty),
        }

        return float(sum(reward_terms.values())), reward_terms

    def _lap_times(self):
        """Per-lap durations in seconds, derived from completion step marks.

        One env step advances the physics by ``SIM.time_step`` seconds, so a
        lap's duration is the number of steps between consecutive completions
        times the physics timestep. The first lap is timed from episode start.
        """
        previous = 0
        times = []
        for mark in self.lap_step_marks:
            times.append(float((mark - previous) * SIM.time_step))
            previous = mark
        return times

    def _mean_lap_time(self):
        """Mean completed-lap time in seconds, or ``None`` if no lap finished."""
        times = self._lap_times()
        if not times:
            return None
        return float(np.mean(times))

    def _get_info(self):
        """Return additional info dict."""
        frenet = self._cached_frenet
        wall_stats = self.world.collision_handler.get_car_stats(self.car.car_id)
        ray_distances = self.last_ray_distances
        ray_min_distance = (
            float(np.min(ray_distances))
            if ray_distances is not None
            else SENSOR.max_ray_distance
        )
        ray_mean_distance = (
            float(np.mean(ray_distances))
            if ray_distances is not None
            else SENSOR.max_ray_distance
        )
        lap_times = self._lap_times()
        mean_lap_time = float(np.mean(lap_times)) if lap_times else None

        return {
            'speed': float(self.car.speed),
            'car_x': float(self.car.position[0]),
            'car_y': float(self.car.position[1]),
            'forward_velocity': float(self.car.get_forward_velocity()),
            'speed_kmh': float(self.car.speed * 3.6),
            's': float(frenet['s']),
            'e_y': float(frenet['e_y']),
            'e_psi': float(frenet['e_psi']),
            'kappa': float(frenet['kappa']),
            'steps': int(self.steps),
            'on_track': bool(self._cached_on_track),
            'laps': int(self.laps_completed),
            'lap_times': lap_times,
            'mean_lap_time': mean_lap_time,
            'total_progress': float(self.total_progress),
            'progress_fraction': float(self.total_progress / self.track.total_length),
            'wall_hits': int(wall_stats['wall_hit_count']),
            'wall_hit_this_step': bool(self.last_wall_hit_this_step),
            'touching_car': bool(wall_stats['touching_car']),
            'car_collisions': int(wall_stats['car_collision_count']),
            'car_contact_steps': int(self.car_contact_steps),
            'overtake_count': int(self.overtake_count),
            'num_opponents': len(self.opponents),
            'opponent_mode': (
                self.opponent_spec.mode if self.opponent_spec is not None else None
            ),
            'ray_min_distance': ray_min_distance,
            'ray_mean_distance': ray_mean_distance,
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
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    self.follow_camera = not self.follow_camera
                elif event.key == pygame.K_v:
                    self.show_sensors = not self.show_sensors
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.renderer.zoom = min(
                        self.max_zoom,
                        self.renderer.zoom * self.zoom_multiplier,
                    )
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.renderer.zoom = max(
                        self.min_zoom,
                        self.renderer.zoom / self.zoom_multiplier,
                    )

        if self.follow_camera:
            self.renderer.set_camera(self.car.position[0], self.car.position[1])
        else:
            keys = pygame.key.get_pressed()
            pan = self.camera_pan_speed * SIM.time_step / max(self.renderer.zoom, 0.1)
            dx = (1 if keys[pygame.K_d] else 0) - (1 if keys[pygame.K_a] else 0)
            dy = (1 if keys[pygame.K_w] else 0) - (1 if keys[pygame.K_s] else 0)
            if dx or dy:
                if dx and dy:
                    pan *= 0.70710678
                self.renderer.set_camera(
                    self.renderer.camera_x + dx * pan,
                    self.renderer.camera_y + dy * pan,
                )

        self.renderer.clear()
        self.renderer.draw_track(self.track)
        if self.show_checkpoints:
            self.renderer.draw_checkpoints(self.track.get_checkpoint_positions())

        self.renderer.draw_cars(
            [self.car] + [opp.car for opp in self.opponents],
            collision_handler=self.world.collision_handler,
        )
        if self.show_minimap:
            self._update_minimap_race_state()
            self.renderer.draw_minimap(self.track, self._all_cars())

        if (
            self.show_sensors
            and self.last_ray_distances is not None
            and self.last_ray_hits is not None
        ):
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
