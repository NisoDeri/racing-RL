# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 2D top-down F1 racing simulator (Box2D physics + Pygame rendering) designed as a Gymnasium-compatible RL environment. University MSc project — the goal is to train PPO/SAC agents to race autonomously using only on-board sensors. Phases 1–5 (seed 42) are complete; seeds 43/44 and final held-out evaluation remain for Phase 5.

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
.venv/bin/python watch_trained.py models/5c_v3_seed42_best.zip --track grand-prix
.venv/bin/python watch_trained.py models/5d_v3_seed42_best.zip --curriculum-stage 5d
.venv/bin/python watch_trained.py models/5d_v3_seed42_best.zip --opponents 3

# Phase 5 curriculum training (load from previous stage checkpoint)
.venv/bin/python train.py --curriculum-stage 5b --track-mode random --n-envs 8 \
  --load-checkpoint models/phase4/v2/seed42/phase4_v2_seed42/best_model.zip \
  --timesteps 1500000 --seed 42 --log-dir logs/phase5 --model-dir models/phase5/v3/seed42
.venv/bin/python train.py --curriculum-stage 5c --track-mode random --n-envs 8 \
  --load-checkpoint models/phase5/v3/seed42/5b_v3_seed42/best_model.zip \
  --timesteps 1500000 --seed 42 --log-dir logs/phase5 --model-dir models/phase5/v3/seed42
.venv/bin/python train.py --curriculum-stage 5d --track-mode random --n-envs 8 \
  --load-checkpoint models/phase5/v3/seed42/5c_v3_seed42/best_model.zip \
  --timesteps 2500000 --seed 42 --log-dir logs/phase5 --model-dir models/phase5/v3/seed42

# Phase 5 held-out evaluation with opponents
.venv/bin/python evaluate.py models/phase5/v3/seed42/5d_v3_seed42/best_model.zip \
  --curriculum-stage 5d --tracks held-out --episodes 20 \
  --output results/phase5d_seed42_heldout.json

# Tests (headless, no display required)
.venv/bin/pytest tests/ -v
.venv/bin/pytest tests/test_racing_env.py::test_observation_shape   # single test
.venv/bin/pytest tests/test_curriculum.py -v                        # Phase 5 curriculum tests
```

## Architecture

The system is split into five modules under `src/`, all wired together in `main.py` and `src/env/racing_env.py`:

**`src/physics/`** — Box2D wrapper. `World` holds the b2World and the `CollisionHandler` (a `b2ContactListener` that tracks per-car wall and car-to-car contact counts). `Car` applies engine force, steering torque, lateral grip, and aerodynamic downforce each step. Physics runs at 60 Hz.

**`src/track/`** — `Track` computes centerline/Frenet geometry, Shapely-buffered boundaries, and Box2D walls. `random_track.py` generates validated, seeded Fourier tracks per episode and defines the held-out Sprint, Grand Prix, and procedural 1001/1002/1003 evaluation set. Training explicitly excludes those procedural seeds.

**`src/sensors/`** — `RayCaster` casts 30 rays (24 forward semicircle + 6 mirror-like rear rays) against track boundaries and optionally other car bodies. When `SENSOR.detect_cars_as_obstacles=True`, ray intersections check other car polygons too.

**`src/rendering/`** — `Renderer` draws track, cars, HUD, sensor rays, lap timer, and Frenet debug overlays via Pygame. Stateless except for camera position and zoom.

**`src/env/opponents.py`** — Phase 5 opponent definitions. `OpponentSpec` (frozen dataclass) encodes a stage's opponent population: mode (`stationary` or `centerline_follower`), count, speed fraction, and spawn offsets. `Opponent` drives a kinematic `Car` each step. `CURRICULUM_OPPONENTS` maps stage IDs (`5a`–`5d`) to their specs.

**`src/env/racing_env.py`** — `RacingEnv` is the Gymnasium wrapper. Accepts an optional `opponent_spec` for Phase 5. Observation is a **34-dim raycast-only vector**: 30 normalized ray distances `[0,1]` + normalized speed `[0,2]` + normalized lateral velocity `[-2,2]` + last throttle + last steering `[-1,1]`. Action is `[throttle, steering]` ∈ [-1,1]². The `v2` profile terminates on off-track, 30+ stuck-wall steps, or 120 backwards-progress steps; all profiles truncate after 6000 steps. The `v3` profile extends `v2` with car-collision penalties and an overtake bonus.

**`train.py` / `evaluate.py`** — PPO experiment runner and headless evaluator. `--track-mode random` regenerates the training track at each reset and defaults to 5M steps. `--tracks held-out` reports per-track and aggregate metrics over all five unseen circuits.

**`main.py`** — Multi-car interactive demo. Spawns `RACE.num_players` player cars plus an optional kinematic centerline-following control car. Useful as reference for setting up the full stack manually.

## Configuration

All physics, sensor, and render parameters live in `config.py` as global dataclass instances: `SIM`, `RACE`, `CAR`, `TRACK`, `SENSOR`, `RENDER`. Training/evaluation options are CLI flags. Key simulation values:

- `RACE.num_players` / `RACE.enable_static_control_car` — multi-car setup
- `RACE.startup_collision_grace_steps` — ignores contacts on first N physics steps at spawn
- `SENSOR.detect_cars_as_obstacles` — whether rays hit other cars

## Key Design Decisions

**Observation is raycast-only; Frenet is reward-only.** The 34-dim obs uses only what an on-board sensor would see. Frenet coordinates (`s`, `e_y`, `e_psi`) are computed internally each step for reward shaping but are NOT in the observation — Frenet still lives in `src/track/track.py` for this purpose. Frenet computed via closest-point projection onto the centerline; lap wraparound uses a ±half-track-length guard on `ds`.

**Reward profiles.** `v1` exactly retains the Phase 2 reward for control runs. The default `v2` uses forward velocity for the speed bonus, withholds progress/speed reward during wall contact, adds `−0.25` per wall-contact step and `−0.001` per time step, and terminates sustained reverse driving with `−25`. `v3` extends `v2` with `−5.0` per new car-car collision event, `−0.15` per step while touching another car, and `+2.0` per opponent overtaken.

**Collision tracking is per-car.** `CollisionHandler.get_car_stats(car_id)` returns counts for a specific car. The `ignore_car_collision_count_until_step` gate prevents Box2D's initial contact cascade at spawn from inflating hit counts.

**Training uses frame stacking.** `train.py` wraps the env with `VecFrameStack(n_stack=4)` so the policy can infer velocity from ray distance deltas across frames. `watch_trained.py` must use the same wrapper to load a trained model correctly.

**Training, model selection, and held-out evaluation are separated by seed.** `RandomTrackGenerator` draws a new 32-bit seed per reset from `env.np_random` and rejects validation seed 2001 plus held-out procedural seeds 1001–1003. `EvalCallback` uses only seed 2001; final evaluation uses Sprint, Grand Prix, and 1001–1003. Invalid polygons, collapsed inner boundaries, and curvature above `0.2 m⁻¹` are rejected. Evaluation starts are seeded and randomized across longitudinal position with small lateral/heading perturbations.

**Static control car is kinematically driven.** It bypasses Box2D force integration — position and velocity are set directly each frame. This prevents it from being perturbed by collisions while still participating in raycast obstacle detection.

## RL Training Roadmap

See `NEXT_PHASE.md` for the full 10-phase plan. Current status:
- **Phase 1** (RL contract — obs/action/reward) ✓ complete
- **Phase 2** (PPO baseline on Sprint Circuit) ✓ complete; artifacts use Git LFS
- **Phase 3** (reward iteration + reproducible evaluation) ✓ complete; three seeds per reward profile
- **Phase 4** (domain randomization + held-out track evaluation) ✓ complete; three 5M-step seeds
- **Phase 5** (multi-car curriculum) ✓ seed 42 complete (5b→5c→5d); seeds 43/44 + final held-out eval pending
- **Phases 6–10** — self-play, advanced experiments, SAC, evaluation, report

## Verified Experiment Results

Results below were produced on seeds 42, 43, and 44. Model selection uses only the reserved validation track (procedural seed 2001); held-out results use Sprint, Grand Prix, and procedural seeds 1001–1003.

### Phase 3 — Reward Shaping

Both profiles were trained for 1M transitions per seed with the same PPO configuration. Each final model was evaluated on five seeded randomized starts on Sprint.

| Metric (mean ± seed std) | v1 control | v2 shaped reward |
|---|---:|---:|
| Lap success | 0% ± 0% | 100% ± 0% |
| Laps per episode | 0 ± 0 | 8 ± 0 |
| Progress fraction | −8.012 ± 0.144 | 8.491 ± 0.067 |
| Wall hits | 29.40 ± 1.73 | 0.47 ± 0.66 |

The v1 agent learned the documented reverse-driving exploit: it accumulated positive reward while completing no valid laps. The v2 profile eliminated that exploit and generalized across randomized Sprint starts. Raw v1 and v2 returns are not directly comparable because their reward definitions differ.

### Phase 4 — Domain Randomization

PPO v2 was trained from scratch for 5M transitions on a new validated procedural track each episode, with eight parallel environments. The best checkpoint for each seed was selected on validation seed 2001. Final evaluation used 20 episodes per held-out track per seed: 300 episodes total.

| Metric (mean ± seed std) | Result |
|---|---:|
| Held-out lap success | 100% ± 0% |
| Progress fraction | 3.829 ± 0.050 |
| Wall hits | 11.59 ± 4.87 |
| Mean speed | 78.37 ± 0.24 |

All five held-out tracks achieved 100% lap success over 60 episodes each. Sprint remains the main limitation: it averaged 38.05 wall hits because its 14m width is narrower than the 18–28m procedural training distribution. One seed-44 Sprint episode terminated as `stuck_wall` after already completing a lap, so it still counted as successful under the `laps > 0` criterion.

Final evaluation files are `results/phase4_v2_seed{42,43,44}_heldout_final.json`; validation-selected models are `models/phase4/v2/seed*/phase4_v2_seed*/best_model.zip`. The full suite passes: **50 tests passed** on 2026-06-22.

### Phase 5 — Multi-Car Curriculum (seed 42, validation track only)

Curriculum stages trained sequentially: each stage warm-starts from the previous stage's best model. Reward profile `v3` adds car-collision penalty (−5.0/event, −0.15/step) and overtake bonus (+2.0). Eval every 250k steps on validation seed 2001, 3 episodes per eval.

| Stage | Opponents | Best eval step | Best val reward | Notes |
|-------|-----------|---------------|-----------------|-------|
| 5b | 1 stationary | 250k / 1.5M | ~11,147 | Trivially solved — Phase 4 raycast already handles parked cars like walls |
| 5c | 1 moving (50% speed) | 1M / 1.5M | 11,765 | Genuine learning signal; reward spike at 1M confirms overtaking behavior emerged |
| 5d | 3 moving (50% speed) | 500k / 2.5M | 11,322 | High variance (±300–1377); agent handles chain of cars but inconsistently |

Best models: `models/phase5/v3/seed42/{5b,5c,5d}_v3_seed42/best_model.zip`. Final held-out evaluation (all 5 tracks × 20 episodes) pending. Seeds 43/44 not yet run.
