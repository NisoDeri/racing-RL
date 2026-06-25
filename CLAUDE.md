# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 2D top-down F1 racing simulator (Box2D physics + Pygame rendering) designed as a Gymnasium-compatible RL environment. University MSc project — the goal is to train PPO/SAC agents to race autonomously using only on-board sensors. Phases 1–4 are complete; the next planned work is the Phase 5 multi-car curriculum.

## Commands

The project uses a `.venv` at the repo root (Python 3.12). Always activate it or prefix commands with `.venv/bin/python` (POSIX) / `.venv\Scripts\python.exe` (Windows).

**Windows / environment notes** (verified on Windows 11, Python 3.12, SB3 2.9):
- Set up with [`uv`](https://docs.astral.sh/uv/): `uv venv --python 3.12 && uv pip install -r requirements.txt`. `box2d` installs from a prebuilt wheel — no compiler needed.
- **Set `PYTHONUTF8=1`** when training/evaluating on Windows: the startup banner prints `→`/`×`, which crash under the default cp1252 console codec (`UnicodeEncodeError`) before training starts.
- **SB3 ≥ 2.9 fix (already applied in `train.py`):** with `--vec-normalize-reward` (`norm_obs=False`), SB3 no longer creates `obs_rms`, so `SyncVecNormalizeCallback._on_step` must guard `obs_rms`/`ret_rms` behind `hasattr` or it crashes on the first step. *(Still needs merging to `master`.)*
- The commands below use POSIX `.venv/bin/python`; on Windows swap in `.venv\Scripts\python.exe` and prefix `PYTHONUTF8=1`.

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

### Phase 7 — GAE ablation + auxiliary task + reward normalization (verified, seed 44)

```bash
# Full GAE ablation (5 values), with Phase 7a aux head + Phase 7d reward norm.
# Each value trains 5M steps; phase7_ablation runs them sequentially.
# (Windows: prefix PYTHONUTF8=1 and use .venv\Scripts\python.exe.)
.venv/bin/python phase7_ablation.py --gae-values 0.0 0.5 0.9 0.95 1.0 --seeds 44 \
  --track-mode random --reward-profile v2 --timesteps 5000000 --n-envs 8 \
  --vec-normalize-reward --aux-raycast-prediction \
  --manifest results/phase7_seed44_manifest.json --execute

# Held-out evaluation for one GAE value (best_model is validation-seed-2001 selected)
.venv/bin/python evaluate.py \
  models/phase7/phase7_gae0p95_random_v2_seed44_normrew_auxray/best_model.zip \
  --reward-profile v2 --tracks held-out --episodes 20 \
  --output results/phase7/phase7_gae0p95_seed44_heldout.json

# Ablation figures (one --group per GAE value; add seeds 42/43 npz later for bands)
.venv/bin/python phase9_figures.py curves \
  --group "GAE 0.95" logs/phase7/phase7_gae0p95_random_v2_seed44_normrew_auxray/evaluations.npz \
  --out results/phase7/figures/phase7_lambda_curves.png

# Watch a Phase 7 (aux-policy) model on a chosen unseen track. Needed because
# watch_trained.py hardcodes Sprint and does not import AuxRaycastActorCriticPolicy.
.venv/bin/python watch_phase7.py \
  models/phase7/phase7_gae0p95_random_v2_seed44_normrew_auxray/best_model.zip \
  --track grand-prix     # random | sprint | grand-prix | procedural-1001/1002/1003
```

Seed-44 result (single seed): **GAE λ=0.95 is the bias–variance sweet spot** — 99% zero-shot lap success, 2.1 wall hits, ~262 km/h. λ=0.0 is safe-but-slow (high bias), λ=1.0 fast-but-scrappier (high variance), λ=0.5 weakest (86%).

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
- **Phase 3** (reward iteration + reproducible evaluation) ✓ complete; three seeds per reward profile
- **Phase 4** (domain randomization + held-out track evaluation) ✓ complete; three 5M-step seeds
- **Phase 5** (multi-car curriculum) — next
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
