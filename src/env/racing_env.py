"""
Gymnasium Racing Environment.

Phase 1 RL contract:
  Observation:
    [30 normalized ray distances, normalized speed, normalized lateral velocity,
     last throttle, last steering]

  Action:
    [throttle, steering], both in [-1, 1]

  Reward:
    + positive forward progress along the centerline
    + small speed bonus
    - lateral deviation from the centerline
    - heading error
    - steering jitter
    - wall hits
    - off-track termination penalty
"""
import os
import sys

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


class RacingEnv(gym.Env):
    """
    Top-down racing environment for reinforcement learning.

    The policy observes raycast distances plus ego proprioception. Frenet
    coordinates are intentionally kept internal for reward shaping and metrics.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, track_creator=None, max_episode_steps=6000):
        """
        Args:
            render_mode: "human" for Pygame window, "rgb_array" for pixel array,
                None for headless.
            track_creator: Optional callable that returns a Track instance. If
                None, uses Track.create_complex_track().
            max_episode_steps: Max steps before truncation, about 100 seconds
                at 60 FPS.
        """
        super().__init__()

        self.render_mode = render_mode
        self.track_creator = track_creator
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
        low[lateral_idx:] = -1.0
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
        self.inner_boundary = None
        self.outer_boundary = None

        self.steps = 0
        self.prev_s = 0.0
        self.total_progress = 0.0
        self.laps_completed = 0
        self.last_throttle = 0.0
        self.last_steering = 0.0
        self.prev_wall_hit_count = 0
        self.stuck_wall_steps = 0

        self.last_ray_distances = None
        self.last_ray_hits = None
        self.last_reward_terms = {}
        self.last_wall_hit_this_step = False

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Returns:
            observation: Initial state observation.
            info: Additional information dict.
        """
        super().reset(seed=seed)

        self.world = World()

        if self.track_creator:
            self.track = self.track_creator()
        else:
            self.track = Track.create_complex_track(track_width=14)

        self.inner_boundary, self.outer_boundary = self.track.get_boundary_points()
        self.track.create_walls(self.world)
        self.world.collision_handler.ignore_car_collision_count_until_step = (
            RACE.startup_collision_grace_steps
        )

        start_pos = self.track.centerline[0]
        start_heading = np.arctan2(self.track.tangents[0, 1], self.track.tangents[0, 0])
        self.car = Car(self.world, position=start_pos, angle=start_heading)

        self.steps = 0
        self.prev_s = 0.0
        self.total_progress = 0.0
        self.laps_completed = 0
        self.last_throttle = 0.0
        self.last_steering = 0.0
        self.prev_wall_hit_count = 0
        self.stuck_wall_steps = 0
        self.last_reward_terms = {}
        self.last_wall_hit_this_step = False

        obs = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._init_renderer()
            self.render()

        return obs, info

    def step(self, action):
        """
        Execute one environment step.

        Args:
            action: [throttle, steering], both in [-1, 1].

        Returns:
            observation, reward, terminated, truncated, info.
        """
        prev_steering = self.last_steering
        throttle = float(np.clip(action[0], -1.0, 1.0))
        steering = float(np.clip(action[1], -1.0, 1.0))
        self.car.set_controls(throttle, steering)

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
        new_wall_hits = max(0, wall_hits - self.prev_wall_hit_count)
        self.prev_wall_hit_count = wall_hits
        self.last_wall_hit_this_step = new_wall_hits > 0

        reward, reward_terms = self._compute_reward(
            frenet=frenet,
            on_track=on_track,
            ds=ds,
            steering=steering,
            prev_steering=prev_steering,
            new_wall_hits=new_wall_hits,
        )
        self.last_reward_terms = reward_terms

        terminated = False
        truncated = False

        if not on_track:
            terminated = True

        if wall_stats['touching_wall'] and max(ds, 0.0) < 0.01:
            self.stuck_wall_steps += 1
        else:
            self.stuck_wall_steps = 0

        if self.stuck_wall_steps >= 30:
            terminated = True

        if self.steps >= self.max_episode_steps:
            truncated = True

        self.last_throttle = throttle
        self.last_steering = steering

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
        )
        self.last_ray_distances = distances
        self.last_ray_hits = hit_points

        rays = self.raycaster.get_normalized(distances)
        lateral_velocity = np.clip(self.car.get_lateral_velocity() / 50.0, -1.0, 1.0)

        return np.array(
            [
                *rays,
                self.car.speed / 95.0,
                lateral_velocity,
                self.last_throttle,
                self.last_steering,
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, frenet, on_track, ds, steering, prev_steering, new_wall_hits):
        progress_reward = max(ds, 0.0) if on_track else 0.0
        speed_bonus = 0.01 * self.car.speed if on_track else 0.0
        lateral_penalty = -0.1 * abs(frenet['e_y']) / self.track.half_width
        heading_penalty = -0.05 * abs(frenet['e_psi']) / np.pi
        steering_penalty = -0.5 * abs(steering - prev_steering)
        wall_penalty = -10.0 * new_wall_hits
        off_track_penalty = -50.0 if not on_track else 0.0

        reward_terms = {
            'progress': float(progress_reward),
            'speed': float(speed_bonus),
            'lateral': float(lateral_penalty),
            'heading': float(heading_penalty),
            'steering_smoothness': float(steering_penalty),
            'wall_hit': float(wall_penalty),
            'off_track': float(off_track_penalty),
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
            'speed_kmh': float(self.car.speed * 3.6),
            's': float(frenet['s']),
            'e_y': float(frenet['e_y']),
            'e_psi': float(frenet['e_psi']),
            'steps': int(self.steps),
            'on_track': bool(self.track.is_inside_track(self.car.position)),
            'laps': int(self.laps_completed),
            'total_progress': float(self.total_progress),
            'wall_hits': int(wall_stats['wall_hit_count']),
            'wall_hit_this_step': bool(self.last_wall_hit_this_step),
            'ray_min_distance': ray_min_distance,
            'lateral_velocity': float(self.car.get_lateral_velocity()),
            'last_throttle': float(self.last_throttle),
            'last_steering': float(self.last_steering),
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
        if self.render_mode is None:
            return None

        if self.renderer is None:
            self._init_renderer()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        self.renderer.set_camera(self.car.position[0], self.car.position[1])
        self.renderer.clear()
        self.renderer.draw_track(self.track)
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
