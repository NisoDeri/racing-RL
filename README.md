# F1 Top-Down Racing Simulation & RL Environment

A 2D top-down Formula 1 racing simulator built with **Box2D** physics and **Pygame** rendering, designed as a **Gymnasium-compatible reinforcement learning environment** for training autonomous racing agents.

University project — the goal is to train RL agents (PPO, SAC) that learn to drive a race car around procedurally generated tracks using only on-board sensor observations.

---

## Project Structure

```
f1/
├── main.py                  # Interactive demo (keyboard-controlled driving)
├── train.py                 # PPO fixed/random-track training runner
├── train_sac.py             # SAC runner for Phase 8 PPO-vs-SAC comparison
├── evaluate.py              # Held-out track metrics and JSON output
├── phase7_ablation.py       # Phase 7 GAE / auxiliary / normalization command builder
├── phase8_compare.py        # Phase 8 PPO-vs-SAC command builder
├── phase9_figures.py        # Phase 9 report figure generators (trajectory/success/curves/raycast)
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
    │   ├── track.py         # Centerline geometry, Frenet frame, boundaries, walls
    │   └── random_track.py  # Seeded Phase 4 tracks and held-out track set
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

- Python 3.12+
- pip

### Installation

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Interactive Demo

```bash
.venv/bin/python main.py
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

### Training and Evaluation

`v1` reproduces the Phase 2 reward. `v2` is the Phase 3 default with reward-hacking
protections. Use distinct run names so ablation outputs never overwrite each other.

```bash
# Phase 2 control run
.venv/bin/python train.py --reward-profile v1 --seed 42 --run-name ppo_sprint_v1_seed42

# Phase 3 shaped-reward run
.venv/bin/python train.py --reward-profile v2 --seed 42 --run-name ppo_sprint_v2_seed42

# Phase 4: 5M transitions over parallel per-episode randomized tracks
.venv/bin/python train.py --track-mode random --reward-profile v2 \
  --timesteps 5000000 --n-envs 8 --seed 42 --run-name ppo_random_v2_seed42

# Phase 7a: PPO plus auxiliary next-raycast prediction
.venv/bin/python train.py --track-mode random --reward-profile v2 \
  --timesteps 5000000 --n-envs 8 --seed 42 \
  --aux-raycast-prediction --aux-loss-coef 0.05 \
  --run-name ppo_random_v2_seed42_auxray

# Phase 7b: build the GAE / n-step ablation commands
.venv/bin/python phase7_ablation.py \
  --gae-values 0.0 0.5 0.9 0.95 1.0 \
  --seeds 42 43 44 \
  --track-mode random \
  --reward-profile v2 \
  --timesteps 5000000 \
  --n-envs 8 \
  --manifest results/phase7_gae_manifest.json

# Add reward normalization from Phase 7d; omit --execute to print commands only
.venv/bin/python phase7_ablation.py \
  --gae-values 0.0 0.5 0.9 0.95 1.0 \
  --seeds 42 43 44 \
  --track-mode random \
  --reward-profile v2 \
  --timesteps 5000000 \
  --n-envs 8 \
  --vec-normalize-reward \
  --execute

# Combine Phase 7a/7b/7d in one ablation manifest
.venv/bin/python phase7_ablation.py \
  --gae-values 0.0 0.5 0.9 0.95 1.0 \
  --seeds 42 43 44 \
  --track-mode random \
  --reward-profile v2 \
  --timesteps 5000000 \
  --n-envs 8 \
  --aux-raycast-prediction \
  --vec-normalize-reward \
  --manifest results/phase7_aux_gae_norm_manifest.json

# Phase 8: train SAC as the PPO comparison algorithm
.venv/bin/python train_sac.py --track-mode random --reward-profile v2 \
  --timesteps 5000000 --n-envs 8 --seed 42 \
  --run-name phase8_sac_random_v2_seed42

# Phase 8: build the full 3-seed PPO-vs-SAC comparison manifest
.venv/bin/python phase8_compare.py \
  --seeds 42 43 44 \
  --timesteps 5000000 \
  --n-envs 8 \
  --manifest results/phase8_manifest.json

# Evaluate on Sprint, Grand Prix, and procedural seeds 1001/1002/1003
.venv/bin/python evaluate.py models/ppo_random_v2_seed42_final.zip \
  --tracks held-out --episodes 20 --output results/phase4_seed42.json

# Evaluate a Phase 8 SAC model on the same held-out set
.venv/bin/python evaluate.py models/phase8/sac/phase8_sac_random_v2_seed42_final.zip \
  --algo sac --tracks held-out --episodes 20 \
  --output results/phase8/phase8_sac_random_v2_seed42_heldout.json

# Evaluate a Phase 7d normalized run with its saved VecNormalize stats
.venv/bin/python evaluate.py models/phase7/<run>_final.zip \
  --vec-normalize models/phase7/<run>_final_vecnormalize.pkl \
  --tracks held-out --episodes 20

# Phase 3 single-track metrics remain available
.venv/bin/python evaluate.py models/ppo_sprint_v2_seed42_final.zip \
  --tracks sprint --reward-profile v2 --episodes 20

# Phase 9: held-out evaluation with lap time + trajectory capture
.venv/bin/python evaluate.py models/phase4/v2/seed42/phase4_v2_seed42/best_model.zip \
  --tracks held-out --episodes 100 --record-trajectories 1 \
  --output results/phase9/phase4_v2_seed42_heldout.json

# Phase 9: report figures from the evaluation artifacts
.venv/bin/python phase9_figures.py trajectory \
  results/phase9/phase4_v2_seed42_heldout.trajectories.json \
  --track grand-prix --out results/phase9/figures/traj_grand-prix.png
.venv/bin/python phase9_figures.py success \
  --group "Phase 4" results/phase4_v2_seed4*_heldout_final.json \
  --out results/phase9/figures/success.png
.venv/bin/python phase9_figures.py curves \
  --group PPO logs/phase4/v2/seed4*/phase4_v2_seed4*/evaluations.npz \
  --out results/phase9/figures/curves.png
.venv/bin/python phase9_figures.py raycast \
  models/phase4/v2/seed42/phase4_v2_seed42/best_model.zip \
  --tracks held-out --episodes 5 --out results/phase9/figures/raycast.png
```

The committed model and TensorBoard files use Git LFS. Run `git lfs pull` before
evaluating those artifacts.

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
- **Phase 4 training distribution**: a new validated 300-point Fourier circuit per
  episode, with base radius 150–500m, randomized harmonics/phases, and width 18–28m

Track boundaries are computed using **Shapely's polygon buffer** operation (Minkowski sum), which cleanly handles self-intersection artifacts at tight turns. Inner/outer boundaries become Box2D static edge chains for collision.

Procedural generation is deterministic from Gymnasium's seed. Invalid/self-intersecting
samples are rejected. Seed `2001` is reserved for validation/checkpoint selection.
Seeds `1001`, `1002`, and `1003` are separately excluded from training and, together
with Sprint and Grand Prix, form the fixed five-track final evaluation set.
Tracks with curvature above `0.2 m⁻¹` (turn radius below 5m) are rejected as
incompatible with the car geometry. Validation and held-out episodes use reproducible
random longitudinal starts with small lateral and heading perturbations.

To render a sequence of generated tracks with a random policy:

```bash
.venv/bin/python watch_env.py --track-mode random --seed 42
```

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

Two observation implementations are available. The current raycast-only policy uses
the first; Frenet data is kept internal for reward shaping and optional later ablations.

#### 1. RayCaster (LiDAR-like)

Casts **30 rays** outward from the car:
- 24 forward rays spanning a 180° semicircle (−90° to +90°)
- 3 left-mirror rays (135°–165°) and 3 right-mirror rays (−165° to −135°)
- Maximum range: 100m per ray

Each ray returns the distance to the nearest wall. The resulting 30-dimensional vector is a 1D "depth image" of the track geometry around the car — wall proximity patterns encode turn direction, severity, and approach distance.

#### 2. FrenetObserver (State-based)

Computes the Frenet frame observation plus **lookahead curvature** — the curvature values at 10 points sampled every 5m ahead of the car. This gives the agent predictive knowledge of upcoming turns (analogous to a driver who has studied the track map).

The baseline deliberately does not expose Frenet features to the policy.

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

34-dimensional continuous vector (`Box`):

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0–29 | Raycast distances | ÷ 100m maximum range, clipped to `[0, 1]` |
| 30 | Speed | ÷ 95 m/s; declared range `[0, 2]` |
| 31 | Lateral velocity | ÷ 50 m/s, clipped to `[-2, 2]` |
| 32 | Previous throttle | `[-1, 1]` |
| 33 | Previous steering | `[-1, 1]` |

Training stacks four consecutive observations with `VecFrameStack`, producing a
136-dimensional policy input that can encode motion over time.

### Action Space

2-dimensional continuous vector (`Box[-1, 1]²`):

| Index | Control | Range |
|-------|---------|-------|
| 0 | Throttle | −1 (full brake) to +1 (full throttle) |
| 1 | Steering | −1 (full left) to +1 (full right) |

### Reward Function

Two named profiles make the reward-shaping ablation reproducible. `v1` is the Phase 2
baseline. `v2` is the Phase 3 default and retains the same core terms while closing
three known exploits:

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Progress** | `+1.0 × ds` | Forward movement along the track (dominant signal) |
| **Lateral penalty** | `−0.1 × \|eᵧ\| / half_width` | Stay near the centerline |
| **Heading penalty** | `−0.05 × \|eᵩ\| / π` | Face the track direction |
| **Speed bonus** | `+0.01 × max(forward_speed, 0)` | Reward forward driving, not reversing |
| **Steering smoothness** | `−0.5 × \|Δsteering\|` | Avoid control jitter |
| **Wall hit** | `−10.0` per new hit | Avoid collisions |
| **Wall contact (v2)** | `−0.25` per touching step | Make sustained wall scraping costly |
| **Time (v2)** | `−0.001` per step | Make camping costly |
| **Off-track** | `−50.0` (+ termination) | Stay on the circuit |

In `v2`, progress and speed bonuses are zero while touching a wall. After 120
consecutive backwards-progress steps, the episode terminates with a `−25` penalty.

### Episode Termination

- **Terminated**: off-track, stuck against a wall for 30 steps, or sustained reverse
  progress for 120 steps (`v2`)
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
    'progress_fraction': float,
    'wall_hits': int,
    'mean_abs_steering_change': float,
    'termination_reason': str | None,
    'reward_profile': str,
    'track_name': str,
    'track_seed': int | None,
    'track_length': float,
    'track_width': float,
    'start_s': float,
    'start_lateral_offset': float,
    'start_heading_offset': float,
    'reward_terms': dict,
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
- **`RACE`** — Multi-car race/session settings (players, static control car, spawn spacing)
- **`CAR`** — Dimensions, mass, engine force, braking force, grip, drag
- **`TRACK`** — Width, curvature samples, wall physics, sectors
- **`SENSOR`** — Ray count, spread angles, mirror layout, max range
- **`RENDER`** — Window size, colors, camera settings

---

## Multi-Car Simulation (New)

The simulator now supports:
- `RACE.num_players` dynamic player cars (main agent is index `0`)
- optional `RACE.enable_static_control_car` centerline-following control car
- car-to-car collision tracking in addition to wall collisions

### New Config Keys (`config.py`)
- `RACE.num_players`
- `RACE.enable_static_control_car`
- `RACE.static_control_speed` (m/s)
- `RACE.player_spawn_gap` (meters along centerline)
- `RACE.static_control_spawn_ahead` (meters ahead of start line)
- `RACE.min_safe_spawn_gap` (minimum enforced spawn spacing)
- `RACE.startup_collision_grace_steps` (ignore car-hit counting for first N physics steps)

When `num_players=1` and static control car is disabled, behavior remains equivalent to the original single-car setup.

---

## Sensor update

Ray sensors can now treat cars as obstacles (same way walls are handled in ray intersection tests).

New config key:
- `SENSOR.detect_cars_as_obstacles` (default `True`)

When enabled, ray distance returns nearest hit among:
- inner/outer track boundaries
- other cars (ego car excluded)

---

## Startup contact note
At initial load, Box2D can emit immediate contacts while bodies settle/spawn.  
To avoid false `WALL HITS` / `CAR HITS`, counters are gated by `RACE.startup_collision_grace_steps` and reset logic is shared between first init and manual reset.
