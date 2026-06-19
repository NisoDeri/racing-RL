"""Watch RacingEnv with a random agent on fixed or procedural tracks."""

import argparse

import numpy as np

from src.env.racing_env import RacingEnv
from src.track.random_track import RandomTrackGenerator
from src.track.track import Track


def _sprint():
    return Track.create_sprint_track(track_width=14)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-mode", choices=("sprint", "random"), default="sprint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    track_kwargs = (
        {"track_generator": RandomTrackGenerator()}
        if args.track_mode == "random"
        else {"track_creator": _sprint}
    )
    env = RacingEnv(
        render_mode="human",
        max_episode_steps=args.max_episode_steps,
        **track_kwargs,
    )
    obs, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    episode = 1

    print("Random agent running - close the window to stop.")
    print(f"Track: {info['track_name']} | seed: {info['track_seed']}")
    print(f"Observation shape: {obs.shape}")

    try:
        while True:
            action = rng.uniform(-1, 1, size=2).astype(np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            if env.window_closed:
                break

            if terminated or truncated:
                print(
                    f"Episode {episode} ended ({info['termination_reason']}) - "
                    f"steps: {info['steps']}, wall hits: {info['wall_hits']}"
                )
                obs, info = env.reset()
                episode += 1
                print(f"Track: {info['track_name']} | seed: {info['track_seed']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
