"""
Watch a trained Phase 7 model drive in real time (or headless-test it).

Handles the Phase 7 config the stock watch_trained.py doesn't:
  - aux-raycast policy (AuxRaycastActorCriticPolicy) reconstruction
  - frame stacking (n_stack=4), matching training
  - choice of track: random (unseen each episode), a held-out circuit, or sprint

Examples:
  # Watch the gae=0.95 agent on fresh random tracks (a window opens)
  python watch_phase7.py models/phase7/phase7_gae0p95_random_v2_seed44_normrew_auxray/best_model.zip
  # Watch on a specific held-out circuit
  python watch_phase7.py <model.zip> --track grand-prix
  # Headless self-test (no window): confirm the model loads and drives
  python watch_phase7.py <model.zip> --no-render --episodes 1 --max-steps 400
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.env.racing_env import RacingEnv
from src.track.track import Track
from src.track.random_track import (
    RandomTrackGenerator,
    HELD_OUT_TRACK_IDS,
    create_held_out_track,
)
# Imported so SB3 can reconstruct the aux policy class when loading the zip.
from src.rl.auxiliary import AuxRaycastActorCriticPolicy  # noqa: F401

N_STACK = 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="Path to a trained .zip model")
    p.add_argument(
        "--track", default="random",
        help="'random' (new unseen track each episode), 'sprint', or a held-out id: "
             + ", ".join(HELD_OUT_TRACK_IDS),
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--episodes", type=int, default=0, help="0 = run forever (close window to stop)")
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--no-render", action="store_true", help="Headless self-test, no window")
    return p.parse_args()


def make_env_fn(track, max_steps, render_mode):
    def _init():
        if track == "random":
            kwargs = {"track_generator": RandomTrackGenerator()}
        elif track == "sprint":
            kwargs = {"track_creator": lambda: Track.create_sprint_track(track_width=14)}
        elif track in HELD_OUT_TRACK_IDS:
            kwargs = {"track_creator": lambda t=track: create_held_out_track(t)}
        else:
            raise SystemExit(f"Unknown --track '{track}'. Use 'random', 'sprint', or one of {HELD_OUT_TRACK_IDS}.")
        return RacingEnv(
            render_mode=render_mode,
            randomize_start=True,
            max_episode_steps=max_steps,
            **kwargs,
        )
    return _init


def main():
    args = parse_args()
    render_mode = None if args.no_render else "human"

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model, device="cpu")

    env = DummyVecEnv([make_env_fn(args.track, args.max_steps, render_mode)])
    env = VecFrameStack(env, n_stack=N_STACK)
    env.seed(args.seed)

    obs = env.reset()
    episode = 1
    steps = 0
    print(f"Track: {args.track} | {'headless self-test' if args.no_render else 'close the window to stop'}")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        steps += 1
        if done[0]:
            i = info[0]
            print(f"Episode {episode}: laps={i.get('laps', 0)} wall_hits={i.get('wall_hits', 0)} "
                  f"progress_frac={i.get('progress_fraction', 0):.2f} mean_speed={i.get('mean_speed', 0):.1f}")
            episode += 1
            obs = env.reset()
            if args.episodes and episode > args.episodes:
                break
        elif args.no_render and steps >= args.max_steps:
            print("Headless self-test reached max-steps without episode end (model is driving).")
            break
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
