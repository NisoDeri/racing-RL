"""
Configuration constants for the F1 simulation.
All tunable parameters in one place for easy experimentation.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class SimConfig:
    """Simulation settings"""
    # Physics
    time_step: float = 1.0 / 60.0  # 60 FPS physics
    velocity_iterations: int = 8   # Box2D solver iterations
    position_iterations: int = 3

    # World scale: pixels per meter
    pixels_per_meter: float = 20.0


@dataclass
class RaceConfig:
    """Race/session settings"""
    num_players: int = 3
    enable_static_control_car: bool = True
    static_control_speed: float = 28.0  # m/s
    player_spawn_gap: float = 8.0       # meters along centerline
    static_control_spawn_ahead: float = 10.0  # meters ahead of start line (along track direction)
    startup_collision_grace_steps: int = 3     # ignore car-car counts for first N physics steps
    min_safe_spawn_gap: float = 6.5            # hard floor to reduce overlap on spawn
    curriculum_min_opponent_gap: float = 25.0  # minimum meters between consecutive opponent spawn offsets


@dataclass
class CarConfig:
    """Car physics parameters — tuned to real F1 2024 data"""
    # Dimensions (meters) - real F1 car
    length: float = 5.6
    width: float = 2.0
    
    # Mass (kg) - F1 minimum 798kg with driver
    mass: float = 798.0
    
    # Engine — real F1: ~1000 HP, 0-100 in 2.6s, 0-200 in 4.8s
    max_forward_force: float = 22000.0   # N — gives ~27.6 m/s² initial accel (~2.8G)
    max_backward_force: float = 36000.0  # N — F1 braking is ~5G (real: ~40kN)
    
    # Steering — F1 uses small steering angles, high downforce does the turning
    max_steer_angle: float = np.radians(20)
    
    # Friction/Grip — racing slicks (μ ~1.5-1.8)
    lateral_friction: float = 0.92       # Base grip (enhanced by aero downforce at speed)
    drag_coefficient: float = 0.29       # v_max ≈ 22000/(798*0.29) ≈ 95 m/s ≈ 342 km/h
    rolling_resistance: float = 0.015


@dataclass
class TrackConfig:
    """Track parameters"""
    # Track dimensions
    width: float = 12.0  # meters - width of driveable surface
    
    # For state-based observation
    num_curvature_samples: int = 10   # How many look-ahead curvature points
    curvature_sample_distance: float = 5.0  # Meters between samples
    
    # Wall collision physics
    wall_friction: float = 0.5        # How much car slides along wall
    wall_restitution: float = 0.1     # Bounciness off walls (low = realistic barrier)
    
    # Checkpoints & Timing
    num_sectors: int = 3              # Sectors for timing (like real F1)


@dataclass
class SensorConfig:
    """Sensor parameters for AI observation (Phase 5)"""
    # Forward rays (semicircle ahead of car)
    num_forward_rays: int = 24
    forward_spread: float = np.pi      # 180 deg: -90 to +90

    # Rear mirror rays (per side, like F1 mirrors)
    num_mirror_rays: int = 3           # per side (6 total)
    mirror_angle_start: float = np.radians(135)   # inner edge of mirror arc
    mirror_angle_end: float = np.radians(165)     # outer edge (closer to rear)

    max_ray_distance: float = 100.0    # meters
    detect_cars_as_obstacles: bool = True

    # Frenet lookahead (curvature preview)
    num_lookahead: int = 10
    lookahead_spacing: float = 5.0     # meters between samples


@dataclass 
class RenderConfig:
    """Rendering settings"""
    # Window
    screen_width: int = 1200
    screen_height: int = 800
    
    # Colors (RGB)
    background_color: tuple = (30, 30, 30)       # Dark gray
    track_color: tuple = (50, 50, 50)            # Asphalt gray
    track_border_color: tuple = (255, 255, 255)  # White lines
    centerline_color: tuple = (255, 200, 0)      # Yellow center
    car_color: tuple = (220, 30, 30)             # Ferrari red

    # Camera
    follow_car: bool = True
    zoom: float = 1.0


# Global config instances
SIM = SimConfig()
RACE = RaceConfig()
CAR = CarConfig()
TRACK = TrackConfig()
SENSOR = SensorConfig()
RENDER = RenderConfig()
