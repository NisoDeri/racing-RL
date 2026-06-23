"""
Watch a trained model drive in real time.

Run after training:
    python watch_trained.py
    python watch_trained.py models/ppo_sprint_500000_steps.zip   # specific checkpoint

    # Held-out tracks (match evaluate.py track IDs)
    python watch_trained.py models/5b_v3_seed42_best.zip --track grand-prix
    python watch_trained.py models/5d_v3_seed42_best.zip --track procedural-1001

    # Match an exact curriculum stage's opponent population
    python watch_trained.py models/5c_v3_seed42_best.zip --curriculum-stage 5c

    # Ad-hoc: race against N centerline followers (50% target speed, 40m apart)
    python watch_trained.py models/5d_v3_seed42_best.zip --track grand-prix --opponents 3
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from src.env.opponents import (
    CURRICULUM_OPPONENTS,
    CURRICULUM_STAGES,
    OpponentSpec,
)
from src.env.racing_env import RacingEnv
from src.track.random_track import HELD_OUT_TRACK_IDS, create_held_out_track


N_STACK = 4
AD_HOC_OPPONENT_SPACING = 40.0   # meters between consecutive ad-hoc opponents
AD_HOC_OPPONENT_SPEED_FRAC = 0.5  # of RACE.static_control_speed


def _build_ad_hoc_opponent_spec(n):
    if n <= 0:
        return None
    offsets = tuple(AD_HOC_OPPONENT_SPACING * (i + 1) for i in range(n))
    return OpponentSpec(
        mode="centerline_follower",
        count=n,
        speed_fraction=AD_HOC_OPPONENT_SPEED_FRAC,
        spawn_offsets=offsets,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        default="models/best_model.zip",
        help="Path to an SB3 PPO .zip checkpoint",
    )
    parser.add_argument(
        "--track",
        choices=HELD_OUT_TRACK_IDS,
        default="sprint",
        help="Held-out track id (same set as evaluate.py).",
    )
    opponents = parser.add_mutually_exclusive_group()
    opponents.add_argument(
        "--curriculum-stage",
        choices=CURRICULUM_STAGES,
        default=None,
        help="Spawn opponents matching CURRICULUM_OPPONENTS[stage] exactly.",
    )
    opponents.add_argument(
        "--opponents",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Ad-hoc: spawn N centerline followers at 50%% target speed, "
            f"spaced {int(AD_HOC_OPPONENT_SPACING)} m apart. "
            "Mutually exclusive with --curriculum-stage."
        ),
    )
    args = parser.parse_args()
    if args.opponents is not None and args.opponents < 0:
        parser.error("--opponents must be >= 0")
    return args


def _resolve_opponent_spec(args):
    if args.curriculum_stage:
        return CURRICULUM_OPPONENTS[args.curriculum_stage]
    if args.opponents is not None:
        return _build_ad_hoc_opponent_spec(args.opponents)
    return None


def main():
    args = parse_args()
    opponent_spec = _resolve_opponent_spec(args)

    def make_env():
        return RacingEnv(
            render_mode="human",
            track_creator=lambda: create_held_out_track(args.track),
            max_episode_steps=6000,
            opponent_spec=opponent_spec,
        )

    print(f"Loading model: {args.model}")
    print(f"Track: {args.track}")
    if args.curriculum_stage:
        print(f"Opponents: curriculum stage {args.curriculum_stage}")
    elif args.opponents:
        print(f"Opponents: {args.opponents} ad-hoc centerline followers")
    else:
        print("Opponents: none")

    model = PPO.load(args.model)

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=N_STACK)

    obs = env.reset()
    episode = 1
    print("Watching trained agent — close the window to stop.")

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done[0]:
            hits = info[0].get("wall_hits", 0)
            laps = info[0].get("laps", 0)
            car_hits = info[0].get("car_collisions", 0)
            overtakes = info[0].get("overtake_count", 0)
            print(
                f"Episode {episode}: wall_hits={hits}, laps={laps}, "
                f"car_collisions={car_hits}, overtakes={overtakes}"
            )
            episode += 1
            obs = env.reset()


if __name__ == "__main__":
    main()
