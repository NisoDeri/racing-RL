# F1 Top-Down Racing Simulation — Project Structure

Top-down 2D Formula 1 racing simulator built on Box2D + Pygame, with a Gymnasium-compatible RL environment. The simulation models F1 physics (downforce, slick-tire grip, aero drag), procedurally generated tracks (Fourier-harmonic centerlines), multi-car racing with collision detection, raycast + Frenet-frame sensors, and lap/sector timing.

## Directory tree

```
f1/
├── main.py                       # Interactive playable simulation entry point
├── train.py                      # PPO training and reward-profile experiments
├── evaluate.py                   # Headless model evaluation and JSON metrics
├── config.py                     # All tunable parameters (physics, race, sensors, rendering)
├── requirements.txt              # Python dependencies
├── README.md                     # Project readme
├── plan.txt                      # Development plan / roadmap notes
├── TODO.txt                      # Outstanding tasks
├── knowledge.txt                 # Reference notes (F1 physics, formulas, etc.)
├── features_to_add.txt           # Wishlist of features
├── links.txt                     # Reference URLs
├── youtube_video.txt             # Video references
├── background.pdf                # Background/reference document
├── screenshots/                  # PNG screenshots of running sim (various tracks)
└── src/
    ├── __init__.py               # Package marker (comment only)
    ├── env/
    │   ├── __init__.py           # Re-exports RacingEnv
    │   └── racing_env.py         # Gymnasium RL environment wrapper
    ├── physics/
    │   ├── __init__.py           # Re-exports World, Car
    │   ├── world.py              # Box2D world + collision listener
    │   └── car.py                # Car body, control inputs, F1-tuned forces
    ├── rendering/
    │   ├── __init__.py           # Re-exports Renderer
    │   └── renderer.py           # Pygame rendering of track, cars, HUD, sensors
    ├── sensors/
    │   ├── __init__.py           # Re-exports RayCaster, FrenetObserver
    │   └── sensor.py             # RayCaster + Frenet-frame observation
    └── track/
        ├── __init__.py           # Re-exports Track
        └── track.py              # Centerline, Frenet math, walls, checkpoints
```

## Python files — responsibility and implementation status

### Top-level

- **`main.py`** — **Fully implemented.** Entry point for the interactive (keyboard-controlled) simulation. Builds the World/Track/Cars/Renderer/Sensors, runs the main game loop at 60 FPS, handles keyboard input (WASD/arrows for steer & throttle, R reset, T switch track, C camera toggle, V sensors toggle, P screenshot, +/- zoom, ESC quit), spawns multiple player cars plus an optional static "control car" that drives on the centerline, runs the physics step, casts rays, tracks lap/sector timing with best-lap detection, and renders the full frame.

- **`train.py`** — **Fully implemented for Phases 2–3.** Configurable SB3 PPO runner with four-frame stacking, parallel environments, named `v1`/`v2` reward profiles, deterministic evaluation, checkpoints, TensorBoard logging, seeds, and Phase 3 driving metrics.

- **`evaluate.py`** — **Fully implemented for Sprint Circuit evaluation.** Loads a PPO model, runs deterministic headless episodes, and reports return, success, progress, wall hits, speed, Frenet errors, steering smoothness, and termination reasons as console/JSON output.

- **`config.py`** — **Fully implemented.** Centralized configuration via `@dataclass` instances. Defines `SimConfig` (timestep, solver iterations, pixels/meter), `RaceConfig` (number of players, static-control car settings, spawn gaps, collision grace steps), `CarConfig` (F1-tuned dimensions, mass 798kg, ~22kN forward force, braking, steering angle, grip/drag), `TrackConfig` (track width, curvature sampling, wall friction/restitution, sector count), `SensorConfig` (24 forward rays + 6 mirror rays, max distance, Frenet lookahead), and `RenderConfig` (window size, colors, camera). Exposes global singletons `SIM`, `RACE`, `CAR`, `TRACK`, `SENSOR`, `RENDER`.

### `src/` — package

- **`src/__init__.py`** — **Empty** (just a one-line comment marking the package).

### `src/env/` — RL environment

- **`src/env/__init__.py`** — **Minimal.** Re-exports `RacingEnv`.

- **`src/env/racing_env.py`** — **Fully implemented (single-car version).** Gymnasium `Env` wrapper around the simulation. Observation space is the 34-dim raycast-first vector (30 normalized rays, speed, lateral velocity, previous throttle, previous steering). Frenet values remain internal to reward shaping. Named `v1` and `v2` reward profiles support the Phase 3 ablation; `v2` adds forward-only speed reward, sustained-wall and time costs, and reverse-driving termination. Action space is continuous `[throttle, steering]` in `[-1, 1]`. Implements `reset()`, `step()`, `render()`, and `close()` and tracks detailed episode diagnostics.

### `src/physics/` — Box2D physics

- **`src/physics/__init__.py`** — **Minimal.** Re-exports `World`, `Car`.

- **`src/physics/world.py`** — **Fully implemented.** Wraps the Box2D `b2World` (zero gravity, top-down). Contains the `CollisionHandler` (a `b2ContactListener`) which tracks per-car wall hits and car-car collisions via `userData` tags, maintains contact counts, exposes per-car stats through `get_car_stats(car_id)`, and supports a startup grace window so spawn-overlap doesn't get counted. Provides `step()`, `create_dynamic_body()`, `create_static_body()`. Includes a `reset()` for episode/lap resets.

- **`src/physics/car.py`** — **Fully implemented.** F1-flavoured car body. Builds a rectangular dynamic Box2D body sized to a real F1 (5.6m × 2.0m, 798kg). `update()` applies four physics effects every step: (1) lateral grip impulse with speed²-scaled aero downforce bonus (so the car sticks better at high speed), (2) throttle force (forward or reverse), (3) steering as yaw torque with a gentle speed-falloff curve so steering authority survives at high speed, (4) rolling resistance when throttle is near zero. Tags the body with `userData` identifying car ID, main-player flag, and static-control flag. Exposes properties for position, angle, velocity, speed, forward/right vectors.

### `src/rendering/` — visualization

- **`src/rendering/__init__.py`** — **Minimal.** Re-exports `Renderer`.

- **`src/rendering/renderer.py`** — **Fully implemented.** Pygame renderer. Handles world↔screen coordinate transforms (with camera + zoom), draws the track surface as a filled polygon between inner/outer boundaries, draws the centerline and start/finish line and sector lines (red/blue/yellow like real F1 sectors), draws all cars in distinct colors (player = red, others = orange, static control car = blue) with collision-state outlines (red flash when touching wall, magenta when touching another car), draws rays + hit points with color gradients (forward = red→green, mirror = magenta→cyan), draws a side panel with the ray bar-chart and closest-wall readout, a lap-timer panel (current/best lap, sector, wall hits, car hits), HUD with speed/throttle/steering/Frenet info, track name banner, and frame-rate limiting.

### `src/sensors/` — perception

- **`src/sensors/__init__.py`** — **Minimal.** Re-exports `RayCaster`, `FrenetObserver`.

- **`src/sensors/sensor.py`** — **Fully implemented.** Two sensor classes for RL observations. `RayCaster` casts 30 rays (24 forward in a 180° arc + 3 left-mirror + 3 right-mirror, with a realistic blind spot behind the car), vectorized ray-segment intersection against inner+outer track boundaries and optionally against other cars (so cars register as obstacles in the rays). Returns distances + hit points and provides a `[0,1]` normalizer. `FrenetObserver` computes the Frenet-frame state vector `[speed, e_y, e_psi, kappa, lookahead_κ_1..N]` from the car + track for RL.

### `src/track/` — track geometry

- **`src/track/__init__.py`** — **Minimal.** Re-exports `Track`.

- **`src/track/track.py`** — **Fully implemented.** Core track geometry. Builds a `Track` from a closed centerline polyline, precomputes segment vectors, lengths, cumulative arc-length, tangents, normals, and per-point curvature. Implements `project_point()` (find nearest centerline point), `get_frenet_coordinates()` (the heart of state-based observation: returns `s`, `e_y`, `e_psi`, `kappa`, projected point), `get_lookahead_curvature()` (curvature at N points ahead), `is_inside_track()`. Computes inner/outer boundaries via Shapely polygon buffer (with a per-point-offset naive fallback), and aligns the inner ring to the outer ring to avoid diagonal seam artifacts at sharp corners. `create_walls()` builds Box2D static edge bodies for both boundaries tagged for collision detection. `get_checkpoint_positions()` produces sector lines. Includes three static factory methods: `create_oval_track()` (simple oval, used for testing), `create_sprint_track()` (~750m compact circuit via Fourier harmonics), and `create_complex_track()` (~3.5km Grand Prix circuit with two straights, varied radii, hairpins, chicanes, and an "esses" section built from a Gaussian-windowed high-frequency oscillation). `get_pose_at_s()` returns position+heading at any arc-length (used for grid spawn).

## Key entry points

- **Run the playable sim:** `python main.py` (keyboard control, switchable tracks, HUD).
- **Use the RL environment:** `from src.env import RacingEnv; env = RacingEnv(render_mode="human")` — standard Gymnasium API.
- **Train PPO:** `python train.py --reward-profile v2 --seed 42`.
- **Evaluate PPO:** `python evaluate.py MODEL.zip --reward-profile v2 --episodes 100`.

## Notes on implementation maturity

- The interactive simulation (`main.py` + `physics/` + `rendering/` + `track/` + `sensors/`) is complete and supports multi-car play with collision tracking.
- The RL environment is currently single-car. Raycasts are wired into observations, but opponent spawning and curriculum/self-play remain Phase 5–6 work.
- `train.py` trains PPO with four-frame stacking and named reward profiles. `evaluate.py` produces reproducible JSON evaluation metrics. SAC and domain-randomized training remain later phases.
