# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 2D top-down F1 racing simulator (Box2D physics + Pygame rendering) designed as a Gymnasium-compatible RL environment. University MSc project — the goal is to train PPO/SAC agents to race autonomously using only on-board sensors. The simulation infrastructure is complete; Phase 1 (RL contract) is complete; active work is Phase 2 (PPO baseline training).

## Commands

The project uses a `.venv` at the repo root (Python 3.12). Always activate it or prefix commands with `.venv/bin/python`.

```bash
# One-time setup
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Interactive keyboard demo
.venv/bin/python main.py

# Watch a random agent in real time (sanity-check the env)
.venv/bin/python watch_env.py

# Train PPO baseline (outputs to logs/ and models/)
# Hardware auto-detected: SubprocVecEnv for CPUs, MPS/CUDA for neural net
.venv/bin/python train.py
# Override hardware:  N_ENVS=4 DEVICE=cpu .venv/bin/python train.py
tensorboard --logdir logs/

# Watch a trained model drive
.venv/bin/python watch_trained.py                                    # loads models/best_model.zip
.venv/bin/python watch_trained.py models/ppo_sprint_500000_steps.zip # specific checkpoint

# Tests (headless, no display required)
.venv/bin/pytest tests/ -v
.venv/bin/pytest tests/test_racing_env.py::test_observation_shape   # single test
```

## Architecture

The system is split into five modules under `src/`, all wired together in `main.py` and `src/env/racing_env.py`:

**`src/physics/`** — Box2D wrapper. `World` holds the b2World and the `CollisionHandler` (a `b2ContactListener` that tracks per-car wall and car-to-car contact counts). `Car` applies engine force, steering torque, lateral grip, and aerodynamic downforce each step. Physics runs at 60 Hz.

**`src/track/`** — `Track` generates centerline geometry from polar Fourier harmonics, computes Shapely-buffered inner/outer boundaries, and creates Box2D static edge chains for collision. Two tracks available: `Track.create_sprint_track()` (~750m) and `Track.create_complex_track()` (~3.5km). The track exposes `get_frenet_coordinates(pos, angle)` used for reward computation (not in the observation).

**`src/sensors/`** — `RayCaster` casts 30 rays (24 forward semicircle + 6 mirror-like rear rays) against track boundaries and optionally other car bodies. When `SENSOR.detect_cars_as_obstacles=True`, ray intersections check other car polygons too.

**`src/rendering/`** — `Renderer` draws track, cars, HUD, sensor rays, lap timer, and Frenet debug overlays via Pygame. Stateless except for camera position and zoom.

**`src/env/racing_env.py`** — `RacingEnv` is the Gymnasium wrapper (single-car). Observation is a **34-dim raycast-only vector**: 30 normalized ray distances `[0,1]` + normalized speed `[0,2]` + normalized lateral velocity `[-2,2]` + last throttle + last steering `[-1,1]`. Action is `[throttle, steering]` ∈ [-1,1]². Episode terminates on off-track, 30+ steps stuck against a wall, or after 6000 steps.

**`main.py`** — Multi-car interactive demo. Spawns `RACE.num_players` player cars plus an optional kinematic centerline-following control car. Useful as reference for setting up the full stack manually.

## Configuration

All physics, sensor, and render parameters live in `config.py` as global dataclass instances: `SIM`, `RACE`, `CAR`, `TRACK`, `SENSOR`, `RENDER`. No CLI flags exist. Key values:

- `RACE.num_players` / `RACE.enable_static_control_car` — multi-car setup
- `RACE.startup_collision_grace_steps` — ignores contacts on first N physics steps at spawn
- `SENSOR.detect_cars_as_obstacles` — whether rays hit other cars

## Key Design Decisions

**Observation is raycast-only; Frenet is reward-only.** The 34-dim obs uses only what an on-board sensor would see. Frenet coordinates (`s`, `e_y`, `e_psi`) are computed internally each step for reward shaping but are NOT in the observation — Frenet still lives in `src/track/track.py` for this purpose. Frenet computed via closest-point projection onto the centerline; lap wraparound uses a ±half-track-length guard on `ds`.

**Reward function (v1).** `r_t = ds + 0.01·speed − 0.1·|e_y|/half_width − 0.05·|e_psi|/π − 0.5·|Δsteering| − 10·wall_hits − 50·off_track`. The `ds` term is clipped to `max(0, ds)` to prevent backwards driving.

**Collision tracking is per-car.** `CollisionHandler.get_car_stats(car_id)` returns counts for a specific car. The `ignore_car_collision_count_until_step` gate prevents Box2D's initial contact cascade at spawn from inflating hit counts.

**Training uses frame stacking.** `train.py` wraps the env with `VecFrameStack(n_stack=4)` so the policy can infer velocity from ray distance deltas across frames. `watch_trained.py` must use the same wrapper to load a trained model correctly.

**Static control car is kinematically driven.** It bypasses Box2D force integration — position and velocity are set directly each frame. This prevents it from being perturbed by collisions while still participating in raycast obstacle detection.

## RL Training Roadmap

See `NEXT_PHASE.md` for the full 10-phase plan. Current status:
- **Phase 1** (RL contract — obs/action/reward) ✓ complete
- **Phase 2** (PPO baseline on Sprint Circuit) — `train.py` is ready; run it
- **Phases 3–10** — reward iteration, domain randomization, curriculum, self-play, SAC comparison
