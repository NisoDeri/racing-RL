"""Watch a PPO driver race against three PPO-controlled opponents.

Examples:
    .venv/Scripts/python.exe watch_multicar.py
    .venv/Scripts/python.exe watch_multicar.py --model models/ppo_sprint_final.zip
    .venv/Scripts/python.exe watch_multicar.py --headless --max-steps 600


    run this in PowerShell to watch a Phase 4v2 seed42 model

    .venv/Scripts/python.exe watch_multicar.py `
   --model models/phase4/v2/seed42/phase4_v2_seed42/best_model.zip `
   --pool-dir models/pool-phase4 `
   --opponent-model models/phase4/v2/seed42/phase4_v2_seed42/best_model.zip


The opponent pool is created automatically from ``--model`` when empty.  To
race against different checkpoints, repeat ``--opponent-model`` for each one.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from config import RACE
from src.env.opponents import Opponent, OpponentSpec
from src.env.pool import CheckpointPool
from src.env.racing_env import REWARD_PROFILES, RacingEnv
from src.physics.car import Car
from src.track.track import Track

CONTROL_CAR_FLAG = False  # Flag to indicate that the control car should be added
N_STACK = 4
# A compact five-car grid.  The finish line is at s=0 and every car begins
# behind it, with 9 m between consecutive cars.
GRID_START_S = -45.0

GRID_OPPONENTS = OpponentSpec(
    mode="pool_agent",
    count=3,
    speed_fraction=1.0,
    spawn_offsets=(9.0, 18.0, 27.0),
)

# The three Control cars opponents are spawned at 9, 30, and 60 m behind the main car.
'''
GRID_OPPONENTS = OpponentSpec(
    mode="centerline_follower",
    count=3,
    speed_fraction=1.0,
    spawn_offsets=(9.0, 30.0, 60.0),
    #spawn_offsets=(9.0,),
)
'''
CONTROL_START_S = -10.0
RACE_COLORS = (
    (225, 55, 55),    # main PPO car: red
    (245, 155, 45),   # opponent 1: orange
    (55, 200, 95),    # opponent 2: green
    (170, 85, 220),   # opponent 3: purple
    (65, 170, 255),   # centerline control car: blue
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_model.zip"),
                        help="PPO checkpoint that drives the main (red) car.")
    parser.add_argument("--pool-dir", type=Path, default=Path("models/pool"),
                        help="Directory containing PPO opponent snapshots.")
    parser.add_argument("--opponent-model", type=Path, action="append", default=[],
                        help=("PPO checkpoint for opponents. Pass one model to share it, "
                              "or pass exactly three to assign orange, green, and purple cars."))
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many physics steps (0 means run until the window closes).")
    parser.add_argument(
        "--reward-profile",
        choices=sorted(REWARD_PROFILES),
        default="v3",
        help="Reward and wrong-way safety profile used by the live race.",
    )
    parser.add_argument("--headless", action="store_true",
                        help="Run without a Pygame window; useful as a smoke test.")
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    if len(args.opponent_model) not in (0, 1, 3):
        parser.error("pass --opponent-model either once or exactly three times")
    return args


def bootstrap_pool(pool_dir: Path, ego_model: Path, opponent_models: list[Path]) -> None:
    """Seed an empty pool so Phase 5e can sample PPO-controlled opponents."""
    pool = CheckpointPool(str(pool_dir))
    if pool.size():
        return

    sources = opponent_models or [ego_model]
    for index, source in enumerate(sources):
        if not source.is_file():
            raise FileNotFoundError(f"Opponent checkpoint not found: {source}")
        pool.add_from_path(str(source), step=index)
    print(f"Created opponent pool at {pool_dir} with {len(sources)} checkpoint(s).")


def validate_model(model: PPO, raw_obs_dim: int) -> None:
    expected_shape = (N_STACK * raw_obs_dim,)
    actual_shape = model.observation_space.shape
    if actual_shape != expected_shape:
        raise ValueError(
            f"Model expects observations shaped {actual_shape}, but this race supplies "
            f"{expected_shape}. Use a PPO model trained with {N_STACK}-frame stacking."
        )


def add_control_car(env: RacingEnv) -> None:
    """Add the blue centerline-following control car to a live race grid."""
    control_s = CONTROL_START_S % env.track.total_length
    position, heading, _ = env.track.get_pose_at_s(control_s)
    car = Car(
        env.world,
        position=position,
        angle=heading,
        car_id=len(env._all_cars()),
        is_static_control=True,
    )
    env.opponents.append(
        Opponent(
            car,
            mode="centerline_follower",
            speed=RACE.static_control_speed,
            initial_s=control_s,
        )
    )


def assign_fixed_opponent_models(env: RacingEnv, model_paths: list[Path]) -> None:
    """Use explicitly selected models rather than random samples from the pool."""
    if not model_paths:
        return

    paths = model_paths * GRID_OPPONENTS.count if len(model_paths) == 1 else model_paths
    if len(env.opponents) != GRID_OPPONENTS.count:
        raise RuntimeError("Expected the three PPO opponents before adding the control car")

    color_names = ("orange", "green", "purple")
    for opponent, path, color_name in zip(env.opponents, paths, color_names):
        if not path.is_file():
            raise FileNotFoundError(f"Opponent checkpoint not found: {path}")
        model = PPO.load(str(path), device="cpu")
        validate_model(model, env.observation_space.shape[0])
        opponent.model = model
        opponent.reset_obs_buffer()
        print(f"{color_name.capitalize()} opponent: {path}")


def assign_race_colors(env: RacingEnv) -> None:
    """Give every car in this race a stable, distinct display color."""
    for car, color in zip(env._all_cars(), RACE_COLORS):
        car.render_color = color


def phase_label(path: Path) -> str:
    """Return the first short token from the model filename for the GUI legend."""
    return re.split(r"[_-]+", path.stem, maxsplit=1)[0]


def assign_minimap_labels(env: RacingEnv, ego_path: Path, opponent_paths: list[Path]) -> None:
    """Attach the color-to-agent legend used by the renderer minimap."""
    paths = opponent_paths * GRID_OPPONENTS.count if len(opponent_paths) == 1 else opponent_paths
    opponent_labels = (
        [phase_label(path) for path in paths]
        if paths
        else ["pool model"] * GRID_OPPONENTS.count
    )
    color_names = ("Orange", "Green", "Purple", "Blue")
    labels = [f"Red - {phase_label(ego_path)}"]
    ppo_index = 0
    for index, opponent in enumerate(env.opponents):
        color_name = color_names[min(index, len(color_names) - 1)]
        if getattr(opponent.car, "is_static_control", False):
            labels.append(f"{color_name} - control")
        else:
            label = (
                opponent_labels[ppo_index]
                if ppo_index < len(opponent_labels)
                else "pool model"
            )
            labels.append(f"{color_name} - {label}")
            ppo_index += 1
    for car, label in zip(env._all_cars(), labels):
        car.minimap_label = label
    return
    labels = [
        f"Red — {phase_label(ego_path)} (main)",
        f"Orange — {opponent_labels[0]}",
        f"Green — {opponent_labels[1]}",
        f"Purple — {opponent_labels[2]}",
        "Blue — Control car",
    ]
    labels = [
        f"Red - {phase_label(ego_path)}",
        f"Orange - {opponent_labels[0]}",
        f"Green - {opponent_labels[1]}",
        f"Purple - {opponent_labels[2]}",
        "Blue - control",
    ]
    for car, label in zip(env._all_cars(), labels):
        car.minimap_label = label


def main() -> None:
    args = parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Main checkpoint not found: {args.model}")

    bootstrap_pool(args.pool_dir, args.model, args.opponent_model)
    model = PPO.load(str(args.model), device="cpu")

    env = RacingEnv(
        render_mode=None if args.headless else "human",
        track_creator=lambda: Track.create_sprint_track(track_width=14),
        # This runner intentionally never resets a race.  Give Gymnasium a
        # practically unreachable per-episode limit; --max-steps controls the
        # whole continuous run instead.
        max_episode_steps=args.max_steps or 2_000_000_000,
        reward_profile=args.reward_profile,
        opponent_spec=GRID_OPPONENTS,
        pool_dir=str(args.pool_dir),
        pool_device="cpu",
    )
    validate_model(model, env.observation_space.shape[0])

    print(
        "Continuous PPO race: 1 main agent vs 3 PPO opponents + 1 control car. "
        "Free camera: W/A/S/D pan, +/- zoom, V sensors, C follows the main car."
    )
    if args.headless and args.max_steps == 0:
        args.max_steps = 6000

    observation, _ = env.reset(options={"start_s": GRID_START_S})
    env.follow_camera = False
    if env.renderer is not None:
        track_center = np.mean(env.track.centerline, axis=0)
        env.renderer.set_camera(float(track_center[0]), float(track_center[1]))
    assign_fixed_opponent_models(env, args.opponent_model)
    if CONTROL_CAR_FLAG:
        add_control_car(env) # Add the control car to the environment after the initial reset. This ensures that the control car is present
    assign_race_colors(env)
    assign_minimap_labels(env, args.model, args.opponent_model)
    env.show_minimap = True
    # Include the new control car in the main agent's very first sensor frame.
    observation = env._get_observation()
    frames = np.zeros((N_STACK, observation.size), dtype=np.float32)
    steps = 0
    try:
        while args.max_steps == 0 or steps < args.max_steps:
            frames = np.roll(frames, -1, axis=0)
            frames[-1] = observation
            action, _ = model.predict(frames.reshape(1, -1), deterministic=True)
            observation, _, _, _, _ = env.step(action[0])
            steps += 1

            if env.window_closed:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
