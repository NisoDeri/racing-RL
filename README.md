# F1 Top-Down Racing Simulation & RL Environment

A 2D top-down Formula 1 racing simulator built with **Box2D** physics and **Pygame** rendering, designed as a **Gymnasium-compatible reinforcement learning environment** for training autonomous racing agents.

University project — the goal is to train RL agents (PPO, SAC) that learn to drive a race car around procedurally generated tracks using only on-board sensor observations.

---

## Project Structure

```
f1/
├── main.py                  # Interactive demo (keyboard-controlled driving)
├── config.py                # All tunable parameters (physics, car, track, sensors, rendering)
├── requirements.txt         # Python dependencies
├── .gitignore
│
└── src/
    ├── physics/
    │   ├── world.py         # Box2D world wrapper + CollisionHandler (car/wall contacts)
    │   └── car.py           # Car body: engine force, steering torque, lateral grip, downforce
    │
    ├── track/
    │   └── track.py         # Centerline geometry, Frenet frame, Shapely boundaries, walls
    │
    ├── sensors/
    │   └── sensor.py        # RayCaster (30-ray LiDAR) + FrenetObserver
    │
    ├── rendering/
    │   └── renderer.py      # Pygame drawing: track, car, HUD, rays, lap timer
    │
    └── env/
        └── racing_env.py    # Gymnasium environment wrapper for RL training
```

---

## How to Run

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Interactive Demo

```bash
python main.py
```

**Controls:**

| Key | Action |
|-----|--------|
| W / Up Arrow | Accelerate |
| S / Down Arrow | Brake / Reverse |
| A / Left Arrow | Steer Left |
| D / Right Arrow | Steer Right |
| T | Switch track (Sprint / Grand Prix) |
| R | Reset car to start |
| C | Toggle camera follow |
| V | Toggle sensor ray visualization |
| P | Take screenshot |
| +/- | Zoom in/out |
| ESC | Quit |

### Using the Gymnasium Environment

```python
from src.env.racing_env import RacingEnv

env = RacingEnv(render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # random agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## Implementation Overview

### Physics Engine (Box2D)

The simulation uses **Box2D** as its 2D rigid-body physics engine with **zero gravity** (top-down perspective). A single rectangular body represents the car chassis, with forces applied each frame:

1. **Lateral friction + aerodynamic downforce** — Cancels sideways slip. Base grip = 0.92 (racing slick coefficient). At high speed, a downforce bonus grows with v² (up to +0.07), emulating real F1 aerodynamic loading. This is why the simulated car can corner at 250+ km/h.

2. **Engine / braking force** — Forward force along the car's heading (22 kN forward, 36 kN braking ≈ 5G deceleration, matching real F1 data).

3. **Steering torque** — Rotates the car body. A power-curve speed factor `(1 - (v/120)^1.5)` reduces steering authority at high speed more gently than a linear falloff, preserving mid-speed responsiveness.

4. **Rolling resistance** — Gentle deceleration when coasting (no throttle input).

The physics world steps at **60 Hz** with 8 velocity and 3 position solver iterations.

### Track Generation

Tracks are procedurally generated from **polar Fourier harmonics**:

```
r(θ) = R₀ + Σ aₖ·cos(kθ) + bₖ·sin(kθ)
```

- **Sprint Circuit** (~750m): base radius 100m, harmonics at k=2,3,5,7
- **Grand Prix Circuit** (~3.5km): base radius 480m, same harmonics + a Gaussian-windowed high-frequency esses section (k=14, localized near θ≈1.2 rad)

Track boundaries are computed using **Shapely's polygon buffer** operation (Minkowski sum), which cleanly handles self-intersection artifacts at tight turns. Inner/outer boundaries become Box2D static edge chains for collision.

### Frenet Frame Coordinate System

Instead of Cartesian (x, y) coordinates, the car's state is expressed in the **Frenet frame** — a curvilinear coordinate system defined relative to the track centerline:

| Variable | Symbol | Meaning |
|----------|--------|---------|
| **Progress** | `s` | Arc-length distance traveled along the centerline (meters) |
| **Lateral error** | `eᵧ` | Perpendicular distance from centerline (+ = right, − = left) |
| **Heading error** | `eᵩ` | Angle between car heading and track tangent direction |
| **Curvature** | `κ` | Local curvature of the centerline (1/radius; + = left turn) |

This representation is **track-agnostic** — the same observation semantics apply to any track shape, enabling transfer learning across different circuits.

### Sensor Systems

Two complementary observation channels:

#### 1. RayCaster (LiDAR-like)

Casts **30 rays** outward from the car:
- 24 forward rays spanning a 180° semicircle (−90° to +90°)
- 3 left-mirror rays (135°–165°) and 3 right-mirror rays (−165° to −135°)
- Maximum range: 100m per ray

Each ray returns the distance to the nearest wall. The resulting 30-dimensional vector is a 1D "depth image" of the track geometry around the car — wall proximity patterns encode turn direction, severity, and approach distance.

#### 2. FrenetObserver (State-based)

Computes the Frenet frame observation plus **lookahead curvature** — the curvature values at 10 points sampled every 5m ahead of the car. This gives the agent predictive knowledge of upcoming turns (analogous to a driver who has studied the track map).

Combined, the agent has both **reactive perception** (rays: "where are the walls right now?") and **predictive knowledge** (curvature lookahead: "what turns are coming?").

### Collision Detection

The `CollisionHandler` (a Box2D `b2ContactListener`) tracks car-wall contacts in real time:
- **touching_wall**: boolean flag for current frame
- **total_wall_hits**: cumulative count (new hit only on first contact edge)
- **wall_hit_speed**: impact velocity at moment of collision

This data feeds into the HUD display, RL reward penalties, and session statistics.

---

## Gymnasium Environment (`RacingEnv`)

The environment follows the standard **Gymnasium API** (`reset`, `step`, `render`, `close`).

### Observation Space

14-dimensional continuous vector (`Box`):

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | Speed | ÷ 95 m/s (≈340 km/h theoretical max) |
| 1 | Lateral error (eᵧ) | ÷ track half-width (±1 = at edge) |
| 2 | Heading error (eᵩ) | ÷ π (full range = ±1) |
| 3 | Curvature (κ) | × 100 (raw values are small) |
| 4–13 | Lookahead curvatures (10 pts) | × 100 |

All values are scaled to approximately [-1, 1] for stable neural network training.

### Action Space

2-dimensional continuous vector (`Box[-1, 1]²`):

| Index | Control | Range |
|-------|---------|-------|
| 0 | Throttle | −1 (full brake) to +1 (full throttle) |
| 1 | Steering | −1 (full left) to +1 (full right) |

### Reward Function

The reward is designed to encourage fast, clean driving:

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Progress** | `+1.0 × ds` | Forward movement along the track (dominant signal) |
| **Lateral penalty** | `−0.1 × \|eᵧ\| / half_width` | Stay near the centerline |
| **Heading penalty** | `−0.05 × \|eᵩ\| / π` | Face the track direction |
| **Speed bonus** | `+0.01 × speed` | Go faster |
| **Off-track penalty** | `−10.0` (+ episode termination) | Don't leave the track |

### Episode Termination

- **Terminated**: car leaves the track boundaries
- **Truncated**: episode exceeds `max_episode_steps` (default 6000 ≈ 100 seconds at 60 fps)

### Info Dict

Each step returns additional metrics:

```python
{
    'speed': float,          # m/s
    'speed_kmh': float,      # km/h
    's': float,              # progress along track (meters)
    'e_y': float,            # lateral deviation (meters)
    'e_psi': float,          # heading error (radians)
    'steps': int,            # steps elapsed this episode
    'on_track': bool,        # within track boundaries
    'laps': int,             # completed laps
    'total_progress': float, # cumulative distance traveled (meters)
}
```

---

## Metrics for RL Training

When training and evaluating RL agents, the following metrics should be tracked:

### Training Metrics (per episode)

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Episode return** | Sum of rewards over an episode | Primary training signal; should increase over time |
| **Episode length** | Number of steps before termination/truncation | Longer = agent survives longer on track |
| **Total progress** | Cumulative arc-length traveled (meters) | Direct measure of how far the agent drives |
| **Laps completed** | Number of full track laps | Binary success metric — completing laps is the goal |

### Performance Metrics (evaluation)

| Metric | Description | Target |
|--------|-------------|--------|
| **Lap time** | Wall-clock time for one complete lap | Lower is better; compare against human baseline |
| **Off-track rate** | Fraction of episodes ending with off-track termination | Should decrease toward 0 |
| **Average speed** | Mean speed across an episode (km/h) | Higher indicates more confident driving |
| **Lateral deviation (mean \|eᵧ\|)** | Average distance from centerline | Lower = more precise driving |
| **Heading error (mean \|eᵩ\|)** | Average misalignment with track direction | Lower = smoother trajectories |
| **Wall hit count** | Total wall contacts per episode | Should be 0 for a well-trained agent |

### Transfer Learning Metrics

Since the observation space is track-agnostic, a key experiment is to train on one track and evaluate on another:

| Metric | Description |
|--------|-------------|
| **Zero-shot success rate** | % of episodes where agent completes a lap on unseen track without fine-tuning |
| **Fine-tuning efficiency** | Episodes needed to match single-track performance on a new track |
| **Cross-track lap time ratio** | Lap time on unseen track ÷ lap time on training track |

---

## Key Terms and Concepts

| Term | Definition |
|------|------------|
| **Box2D** | Open-source 2D rigid-body physics engine used for the simulation |
| **Pygame** | Python library for rendering the visual output and handling input |
| **Gymnasium** | Standard Python API for reinforcement learning environments (successor to OpenAI Gym) |
| **Frenet frame** | Curvilinear coordinate system defined relative to a reference curve; describes position as progress along the curve + perpendicular deviation |
| **Curvature (κ)** | Inverse of the turning radius at a point on the track; higher magnitude = sharper turn |
| **Lateral friction** | Tire grip force that resists sideways sliding; the core mechanism enabling cornering |
| **Aerodynamic downforce** | Vertical force from car bodywork that increases tire grip proportionally to speed² |
| **PPO** | Proximal Policy Optimization — on-policy RL algorithm well-suited for continuous control |
| **SAC** | Soft Actor-Critic — off-policy RL algorithm with entropy regularization for exploration |
| **Stable-Baselines3** | Python library providing reliable implementations of PPO, SAC, and other RL algorithms |
| **Lookahead curvature** | Curvature sampled at future points along the track; gives the agent preview of upcoming turns |
| **Episode** | One run from reset to termination/truncation; the fundamental unit of RL training |
| **Reward shaping** | Designing the reward signal to guide learning toward desired behavior (fast + on-track driving) |
| **Transfer learning** | Applying knowledge from one task (track A) to improve learning on a related task (track B) |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `Box2D` | 2D rigid-body physics simulation |
| `numpy` | Numerical computation (vectors, matrices) |
| `shapely` | Computational geometry (robust track boundary offsetting) |
| `pygame` | Real-time rendering and user input |
| `gymnasium` | Standard RL environment API |
| `stable-baselines3` | RL algorithm implementations (PPO, SAC) |

---

## Configuration

All tunable parameters live in `config.py` as dataclass instances:

- **`SIM`** — Physics timestep, solver iterations, world scale
- **`CAR`** — Dimensions, mass, engine force, braking force, grip, drag
- **`TRACK`** — Width, curvature samples, wall physics, sectors
- **`SENSOR`** — Ray count, spread angles, mirror layout, max range
- **`RENDER`** — Window size, colors, camera settings
