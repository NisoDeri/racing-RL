"""Headless PPO evaluation on fixed held-out racing tracks."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.env.racing_env import REWARD_PROFILES, RacingEnv
from src.env.opponents import CURRICULUM_OPPONENTS, CURRICULUM_STAGES
from src.track.random_track import HELD_OUT_TRACK_IDS, create_held_out_track


N_STACK = 4


def _resolve_model_path(value):
    path = Path(value)
    if not path.exists() and path.suffix != ".zip":
        path = path.with_suffix(".zip")
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {value}")
    with path.open("rb") as model_file:
        header = model_file.read(64)
    if header.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not a downloaded model. "
            "Install Git LFS and run `git lfs pull` before evaluation."
        )
    return path


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def _summarize_results(results):
    reasons = Counter(result["termination_reason"] for result in results)
    return {
        "episodes": len(results),
        "success_rate": float(np.mean([result["success"] for result in results])),
        "return": _mean_std([result["return"] for result in results]),
        "steps": _mean_std([result["steps"] for result in results]),
        "progress": _mean_std([result["progress"] for result in results]),
        "progress_fraction": _mean_std(
            [result["progress_fraction"] for result in results]
        ),
        "wall_hits": _mean_std([result["wall_hits"] for result in results]),
        "mean_speed": _mean_std([result["mean_speed"] for result in results]),
        "mean_abs_e_y": _mean_std([result["mean_abs_e_y"] for result in results]),
        "mean_abs_e_psi": _mean_std(
            [result["mean_abs_e_psi"] for result in results]
        ),
        "mean_abs_steering_change": _mean_std(
            [result["mean_abs_steering_change"] for result in results]
        ),
        "car_collisions": _mean_std(
            [result["car_collisions"] for result in results]
        ),
        "overtake_count": _mean_std(
            [result["overtake_count"] for result in results]
        ),
        "car_contact_steps": _mean_std(
            [result["car_contact_steps"] for result in results]
        ),
        "termination_counts": dict(sorted(reasons.items())),
    }


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _evaluate_track(
    model,
    track_id,
    reward_profile,
    episodes,
    seed,
    max_episode_steps,
    curriculum_stage=None,
    pool_dir=None,
    track_index=0,
    total_tracks=1,
):
    opponent_spec = (
        CURRICULUM_OPPONENTS[curriculum_stage] if curriculum_stage else None
    )

    def make_env():
        return RacingEnv(
            render_mode=None,
            track_creator=lambda: create_held_out_track(track_id),
            randomize_start=True,
            max_episode_steps=max_episode_steps,
            reward_profile=reward_profile,
            opponent_spec=opponent_spec,
            pool_dir=pool_dir or "",
        )

    env = VecFrameStack(DummyVecEnv([make_env]), n_stack=N_STACK)
    env.seed(seed)
    obs = env.reset()
    results = []
    track_start = time.monotonic()
    _log(f"[{track_index}/{total_tracks}] {track_id}  (0/{episodes})")

    try:
        for episode_index in range(episodes):
            episode_return = 0.0
            speeds = []
            lateral_errors = []
            heading_errors = []

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                info = infos[0]
                episode_return += float(rewards[0])
                speeds.append(info["speed"])
                lateral_errors.append(abs(info["e_y"]))
                heading_errors.append(abs(info["e_psi"]))

                if dones[0]:
                    ep_laps = int(info["laps"])
                    ep_hits = int(info["wall_hits"])
                    ep_reason = info["termination_reason"]
                    elapsed = time.monotonic() - track_start
                    _log(
                        f"[{track_index}/{total_tracks}] {track_id} "
                        f"ep {episode_index + 1:>3}/{episodes}  "
                        f"laps={ep_laps}  wall_hits={ep_hits:>3}  "
                        f"reason={ep_reason:<16}  elapsed={elapsed:.0f}s"
                    )
                    results.append(
                        {
                            "episode": episode_index + 1,
                            "return": episode_return,
                            "steps": int(info["steps"]),
                            "laps": int(info["laps"]),
                            "success": bool(info["laps"] > 0),
                            "progress": float(info["total_progress"]),
                            "progress_fraction": float(info["progress_fraction"]),
                            "wall_hits": int(info["wall_hits"]),
                            "car_collisions": int(info.get("car_collisions", 0)),
                            "overtake_count": int(info.get("overtake_count", 0)),
                            "car_contact_steps": int(
                                info.get("car_contact_steps", 0)
                            ),
                            "mean_speed": float(np.mean(speeds)),
                            "mean_abs_e_y": float(np.mean(lateral_errors)),
                            "mean_abs_e_psi": float(np.mean(heading_errors)),
                            "mean_abs_steering_change": float(
                                info["mean_abs_steering_change"]
                            ),
                            "termination_reason": info["termination_reason"],
                            "track_name": info["track_name"],
                            "track_seed": info["track_seed"],
                            "track_length": info["track_length"],
                            "track_width": info["track_width"],
                            "start_s": info["start_s"],
                            "start_lateral_offset": info["start_lateral_offset"],
                            "start_heading_offset": info["start_heading_offset"],
                        }
                    )
                    break
    finally:
        env.close()

    summary = _summarize_results(results)
    summary["track_name"] = results[0]["track_name"]
    summary["track_seed"] = results[0]["track_seed"]
    summary["track_length"] = results[0]["track_length"]
    summary["track_width"] = results[0]["track_width"]
    summary["episode_results"] = results
    return summary


def evaluate(
    model_path,
    reward_profile,
    track_ids,
    episodes,
    seed,
    max_episode_steps,
    curriculum_stage=None,
    pool_dir=None,
):
    model_path = _resolve_model_path(model_path)
    model = PPO.load(model_path, device="cpu")
    tracks = {}
    all_results = []

    _log(f"Evaluating {len(track_ids)} track(s) × {episodes} episodes each")
    for track_index, track_id in enumerate(track_ids):
        track_summary = _evaluate_track(
            model=model,
            track_id=track_id,
            reward_profile=reward_profile,
            episodes=episodes,
            seed=seed + track_index,
            max_episode_steps=max_episode_steps,
            curriculum_stage=curriculum_stage,
            pool_dir=pool_dir,
            track_index=track_index + 1,
            total_tracks=len(track_ids),
        )
        tracks[track_id] = track_summary
        all_results.extend(track_summary["episode_results"])

    return {
        "model": str(model_path),
        "reward_profile": reward_profile,
        "curriculum_stage": curriculum_stage,
        "pool_dir": pool_dir,
        "seed": seed,
        "episodes_per_track": episodes,
        "track_ids": list(track_ids),
        "aggregate": _summarize_results(all_results),
        "tracks": tracks,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to an SB3 PPO model (.zip optional)")
    parser.add_argument(
        "--reward-profile",
        choices=sorted(REWARD_PROFILES),
        default=None,
        help="Defaults to v2; v3 is auto-selected when --curriculum-stage is given.",
    )
    parser.add_argument(
        "--curriculum-stage",
        choices=CURRICULUM_STAGES,
        default=None,
        help="Spawn Phase 5 opponents per CURRICULUM_OPPONENTS[stage].",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=("held-out", *HELD_OUT_TRACK_IDS),
        default=["sprint"],
        help="Track IDs, or 'held-out' for the complete five-track set",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    parser.add_argument(
        "--pool-dir",
        type=str,
        default=None,
        help="Checkpoint pool directory for stage 5e evaluation.",
    )
    args = parser.parse_args(argv)
    if args.reward_profile is None:
        args.reward_profile = "v3" if args.curriculum_stage else "v2"
    if args.episodes < 1 or args.max_episode_steps < 1:
        parser.error("episodes and max-episode-steps must be positive")
    if "held-out" in args.tracks:
        if len(args.tracks) != 1:
            parser.error("'held-out' cannot be combined with individual tracks")
        args.tracks = list(HELD_OUT_TRACK_IDS)
    return args


def main(argv=None):
    args = parse_args(argv)
    summary = evaluate(
        model_path=args.model,
        reward_profile=args.reward_profile,
        track_ids=args.tracks,
        episodes=args.episodes,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        curriculum_stage=args.curriculum_stage,
        pool_dir=args.pool_dir,
    )
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
