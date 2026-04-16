"""
Gymnasium Racing Environment
Wraps the Box2D simulation as a standard Gymnasium environment.

Observation (state-based, Frenet frame):
  [speed, e_y, e_psi, kappa, lookahead_kappa_1, ..., lookahead_kappa_N]
  
Action (continuous):
  [throttle, steering]  both in [-1, 1]

Reward:
  + progress along track (ds)
  - penalty for lateral deviation from centerline
  - penalty for heading error
  - big penalty for going off-track (terminates episode)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config import SIM, CAR, TRACK, RENDER
from src.physics.world import World
from src.physics.car import Car
from src.track.track import Track
from src.rendering.renderer import Renderer


class RacingEnv(gym.Env):
    """
    Top-down racing environment for reinforcement learning.
    
    The agent controls a car on a race track using throttle and steering.
    The goal is to complete laps as fast as possible while staying on track.
    
    State observation uses Frenet frame coordinates (not vision/raycasting):
    - Speed, lateral deviation, heading error, curvature
    - Look-ahead curvature values (AI can "see" upcoming turns)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None, track_creator=None, max_episode_steps=6000):
        """
        Args:
            render_mode: "human" for Pygame window, "rgb_array" for pixel array, None for headless
            track_creator: Optional callable that returns a Track instance.
                          If None, uses Track.create_complex_track()
            max_episode_steps: Max steps before truncation (~100 seconds at 60fps)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.track_creator = track_creator
        self.max_episode_steps = max_episode_steps
        
        # ============================
        # OBSERVATION SPACE
        # ============================
        # [speed, e_y, e_psi, kappa, lookahead_kappa_1, ..., lookahead_kappa_N]
        # All values are normalized to roughly [-1, 1] range for better training
        obs_dim = 4 + TRACK.num_curvature_samples  # 4 + 10 = 14
        
        # Bounds: speed [0,2], e_y [-2,2], e_psi [-1,1], curvatures [-50,50]
        low = np.full(obs_dim, -50.0, dtype=np.float32)
        low[0] = 0.0   # speed is non-negative
        high = np.full(obs_dim, 50.0, dtype=np.float32)
        high[0] = 2.0   # max ~100 m/s normalized
        high[1] = 2.0   # e_y can slightly exceed 1 when off-track
        low[1] = -2.0
        high[2] = 1.0   # e_psi normalized by pi
        low[2] = -1.0
        
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )
        
        # ============================
        # ACTION SPACE
        # ============================
        # [throttle, steering] both continuous in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # These get created on reset()
        self.world = None
        self.car = None
        self.track = None
        self.renderer = None
        
        # Episode state
        self.steps = 0
        self.prev_s = 0.0
        self.total_progress = 0.0
        self.laps_completed = 0
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        
        Returns:
            observation: Initial state observation
            info: Additional information dict
        """
        super().reset(seed=seed)
        
        # === Create fresh physics world ===
        self.world = World()
        
        # === Create track ===
        if self.track_creator:
            self.track = self.track_creator()
        else:
            self.track = Track.create_complex_track(track_width=14)
        
        # === Place car at start ===
        start_pos = self.track.centerline[0]
        start_heading = np.arctan2(
            self.track.tangents[0, 1], 
            self.track.tangents[0, 0]
        )
        self.car = Car(self.world, position=start_pos, angle=start_heading)
        
        # === Reset episode state ===
        self.steps = 0
        self.prev_s = 0.0
        self.total_progress = 0.0
        self.laps_completed = 0
        
        # === Get initial observation ===
        obs = self._get_observation()
        info = self._get_info()
        
        # === Render if needed ===
        if self.render_mode == "human":
            self._init_renderer()
            self.render()
        
        return obs, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: [throttle, steering] both in [-1, 1]
            
        Returns:
            observation: New state
            reward: Scalar reward
            terminated: True if episode ended (off-track, crash)
            truncated: True if max steps reached
            info: Additional information
        """
        # === Apply action to car ===
        throttle = float(np.clip(action[0], -1, 1))
        steering = float(np.clip(action[1], -1, 1))
        self.car.set_controls(throttle, steering)
        
        # === Step physics ===
        self.car.update()
        self.world.step()
        self.steps += 1
        
        # === Get state ===
        frenet = self.track.get_frenet_coordinates(self.car.position, self.car.angle)
        on_track = self.track.is_inside_track(self.car.position)
        
        # === Calculate progress (handle lap wraparound) ===
        current_s = frenet['s']
        ds = current_s - self.prev_s
        
        # Handle wraparound (crossing start/finish line)
        if ds < -self.track.total_length / 2:
            ds += self.track.total_length    # Crossed finish forwards
        elif ds > self.track.total_length / 2:
            ds -= self.track.total_length    # Went backwards past start
            
        self.total_progress += ds
        
        # Track lap completion
        if self.total_progress >= self.track.total_length * (self.laps_completed + 1):
            self.laps_completed += 1
        
        self.prev_s = current_s
        
        # === Compute reward ===
        reward = self._compute_reward(frenet, on_track, ds)
        
        # === Check termination ===
        terminated = False
        truncated = False
        
        # Off track → episode ends
        if not on_track:
            terminated = True
            reward -= 10.0  # Big penalty for leaving track
        
        # Max steps → truncation
        if self.steps >= self.max_episode_steps:
            truncated = True
        
        # === Get observation ===
        obs = self._get_observation()
        info = self._get_info()
        info['on_track'] = on_track
        info['laps'] = self.laps_completed
        info['total_progress'] = self.total_progress
        
        # === Render ===
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Build the observation vector from Frenet frame coordinates.
        
        Returns:
            numpy array of shape (obs_dim,) with normalized values
        """
        frenet = self.track.get_frenet_coordinates(self.car.position, self.car.angle)
        lookahead = self.track.get_lookahead_curvature(frenet['s'])
        
        # Normalize values for better neural network training:
        # - speed: divide by ~50 m/s (reasonable max speed)
        # - e_y: divide by half_width (so ±1 = at track edge)
        # - e_psi: divide by pi (already in [-pi, pi])
        # - kappa: multiply by 100 (curvature values are small)
        obs = np.array([
            self.car.speed / 95.0,                          # Normalized speed (~340 km/h max)
            frenet['e_y'] / self.track.half_width,          # Normalized lateral error
            frenet['e_psi'] / np.pi,                        # Normalized heading error
            frenet['kappa'] * 100.0,                        # Scaled curvature
            *(lookahead * 100.0)                            # Scaled lookahead curvatures
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self, frenet, on_track, ds):
        """
        Compute the reward for this step.
        
        Reward design:
          + Forward progress along track (ds > 0 is good)
          - Lateral deviation penalty (stay near center... for now)
          - Heading error penalty (face forward)
          - Speed bonus (go fast!)
        
        Args:
            frenet: Frenet coordinates dict
            on_track: Whether car is on track
            ds: Progress made this step (meters)
            
        Returns:
            Scalar reward
        """
        reward = 0.0
        
        # === Progress reward (most important!) ===
        # Positive for going forward, negative for going backward
        reward += ds * 1.0
        
        # === Lateral deviation penalty ===
        # Penalize being far from center (normalized by track half-width)
        lateral_ratio = abs(frenet['e_y']) / self.track.half_width
        reward -= 0.1 * lateral_ratio
        
        # === Heading error penalty ===
        # Penalize pointing away from track direction
        heading_error = abs(frenet['e_psi']) / np.pi
        reward -= 0.05 * heading_error
        
        # === Speed bonus (small) ===
        # Encourage going faster (but progress reward already does this)
        reward += 0.01 * self.car.speed
        
        return reward
    
    def _get_info(self):
        """Return additional info dict."""
        frenet = self.track.get_frenet_coordinates(self.car.position, self.car.angle)
        return {
            'speed': self.car.speed,
            'speed_kmh': self.car.speed * 3.6,
            's': frenet['s'],
            'e_y': frenet['e_y'],
            'e_psi': frenet['e_psi'],
            'steps': self.steps,
        }
    
    # ============================
    # RENDERING
    # ============================
    
    def _init_renderer(self):
        """Initialize the Pygame renderer (only for render_mode='human')."""
        if self.renderer is None:
            self.renderer = Renderer()
            
            # Set zoom to fit the track
            track_span = np.max(self.track.centerline, axis=0) - np.min(self.track.centerline, axis=0)
            max_span = max(track_span)
            self.renderer.zoom = min(RENDER.screen_width, RENDER.screen_height) / (max_span + 50) / SIM.pixels_per_meter
    
    def render(self):
        """
        Render the current state.
        
        For render_mode="human": displays in Pygame window
        For render_mode="rgb_array": returns pixel array (for video recording)
        """
        if self.render_mode is None:
            return None
            
        if self.renderer is None:
            self._init_renderer()
        
        # Process pygame events to prevent window freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None
        
        # Camera follows car
        self.renderer.set_camera(self.car.position[0], self.car.position[1])
        
        # Draw everything
        self.renderer.clear()
        self.renderer.draw_track(self.track)
        self.renderer.draw_car(self.car)
        
        # Draw Frenet debug line
        frenet = self.track.get_frenet_coordinates(self.car.position, self.car.angle)
        self.renderer.draw_frenet_debug(self.car, frenet)
        
        # HUD with training info
        self.renderer.draw_hud(self.car, frenet)
        
        # Draw training-specific info
        on_track = self.track.is_inside_track(self.car.position)
        status_color = (0, 255, 0) if on_track else (255, 0, 0)
        status_text = "ON TRACK" if on_track else "OFF TRACK!"
        self.renderer._draw_text(status_text, (RENDER.screen_width - 120, 10), status_color)
        
        # Episode info
        self.renderer._draw_text(
            f"Step: {self.steps} | Laps: {self.laps_completed} | Progress: {self.total_progress:.0f}m",
            (RENDER.screen_width - 400, 35),
            (200, 200, 200)
        )
        
        self.renderer.update()
        
        if self.render_mode == "rgb_array":
            return np.array(pygame.surfarray.array3d(self.renderer.screen))
        
        self.renderer.tick(self.metadata["render_fps"])
    
    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.quit()
            self.renderer = None

