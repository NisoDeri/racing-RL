# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 2D top-down F1 racing simulator (Box2D physics + Pygame rendering) designed as a Gymnasium-compatible RL environment. University MSc project — the goal is to train PPO/SAC agents to race autonomously using only on-board sensors. Phases 1–3 are complete; active work is Phase 4 domain randomization and zero-shot evaluation.

## Commands

The project uses a `.venv` at the repo root (Python 3.12). Always activate it or prefix commands with `.venv/bin/python`.

```bash
# One-time setup
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Interactive keyboard demo
.venv/bin/python main.py

# Watch a random agent in real time (sanity-check the env)
.venv/bin/python watch_env.py
.venv/bin/python watch_env.py --track-mode random --seed 42

# Train Phase 2 control or Phase 3 reward variant
.venv/bin/python train.py --reward-profile v1 --run-name ppo_sprint_v1_seed42
.venv/bin/python train.py --reward-profile v2 --run-name ppo_sprint_v2_seed42
.venv/bin/python train.py --track-mode random --timesteps 5000000 --n-envs 8 \
  --seed 42 --run-name ppo_random_v2_seed42
tensorboard --logdir logs/

# Headless evaluation to JSON
.venv/bin/python evaluate.py models/ppo_random_v2_seed42_final.zip \
  --tracks held-out --episodes 20 --output results/phase4_seed42.json

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

**`src/track/`** — `Track` computes centerline/Frenet geometry, Shapely-buffered boundaries, and Box2D walls. `random_track.py` generates validated, seeded Fourier tracks per episode and defines the held-out Sprint, Grand Prix, and procedural 1001/1002/1003 evaluation set. Training explicitly excludes those procedural seeds.

**`src/sensors/`** — `RayCaster` casts 30 rays (24 forward semicircle + 6 mirror-like rear rays) against track boundaries and optionally other car bodies. When `SENSOR.detect_cars_as_obstacles=True`, ray intersections check other car polygons too.

**`src/rendering/`** — `Renderer` draws track, cars, HUD, sensor rays, lap timer, and Frenet debug overlays via Pygame. Stateless except for camera position and zoom.

**`src/env/racing_env.py`** — `RacingEnv` is the Gymnasium wrapper (single-car). Observation is a **34-dim raycast-only vector**: 30 normalized ray distances `[0,1]` + normalized speed `[0,2]` + normalized lateral velocity `[-2,2]` + last throttle + last steering `[-1,1]`. Action is `[throttle, steering]` ∈ [-1,1]². The Phase 3 `v2` profile terminates on off-track, 30+ stuck-wall steps, or 120 backwards-progress steps; all profiles truncate after 6000 steps.

**`train.py` / `evaluate.py`** — PPO experiment runner and headless evaluator. `--track-mode random` regenerates the training track at each reset and defaults to 5M steps. `--tracks held-out` reports per-track and aggregate metrics over all five unseen circuits.

**`main.py`** — Multi-car interactive demo. Spawns `RACE.num_players` player cars plus an optional kinematic centerline-following control car. Useful as reference for setting up the full stack manually.

## Configuration

All physics, sensor, and render parameters live in `config.py` as global dataclass instances: `SIM`, `RACE`, `CAR`, `TRACK`, `SENSOR`, `RENDER`. Training/evaluation options are CLI flags. Key simulation values:

- `RACE.num_players` / `RACE.enable_static_control_car` — multi-car setup
- `RACE.startup_collision_grace_steps` — ignores contacts on first N physics steps at spawn
- `SENSOR.detect_cars_as_obstacles` — whether rays hit other cars

## Key Design Decisions

**Observation is raycast-only; Frenet is reward-only.** The 34-dim obs uses only what an on-board sensor would see. Frenet coordinates (`s`, `e_y`, `e_psi`) are computed internally each step for reward shaping but are NOT in the observation — Frenet still lives in `src/track/track.py` for this purpose. Frenet computed via closest-point projection onto the centerline; lap wraparound uses a ±half-track-length guard on `ds`.

**Reward profiles.** `v1` exactly retains the Phase 2 reward for control runs. The default `v2` uses forward velocity for the speed bonus, withholds progress/speed reward during wall contact, adds `−0.25` per wall-contact step and `−0.001` per time step, and terminates sustained reverse driving with `−25`.

**Collision tracking is per-car.** `CollisionHandler.get_car_stats(car_id)` returns counts for a specific car. The `ignore_car_collision_count_until_step` gate prevents Box2D's initial contact cascade at spawn from inflating hit counts.

**Training uses frame stacking.** `train.py` wraps the env with `VecFrameStack(n_stack=4)` so the policy can infer velocity from ray distance deltas across frames. `watch_trained.py` must use the same wrapper to load a trained model correctly.

**Training, model selection, and held-out evaluation are separated by seed.** `RandomTrackGenerator` draws a new 32-bit seed per reset from `env.np_random` and rejects validation seed 2001 plus held-out procedural seeds 1001–1003. `EvalCallback` uses only seed 2001; final evaluation uses Sprint, Grand Prix, and 1001–1003. Invalid polygons, collapsed inner boundaries, and curvature above `0.2 m⁻¹` are rejected. Evaluation starts are seeded and randomized across longitudinal position with small lateral/heading perturbations.

**Static control car is kinematically driven.** It bypasses Box2D force integration — position and velocity are set directly each frame. This prevents it from being perturbed by collisions while still participating in raycast obstacle detection.

## RL Training Roadmap

See `NEXT_PHASE.md` for the full 10-phase plan. Current status:
- **Phase 1** (RL contract — obs/action/reward) ✓ complete
- **Phase 2** (PPO baseline on Sprint Circuit) ✓ complete; artifacts use Git LFS
- **Phase 3** (reward iteration + reproducible evaluation) ✓ complete
- **Phase 4** (domain randomization + held-out track evaluation) — active
- **Phases 5–10** — curriculum, self-play, advanced experiments, SAC, report
