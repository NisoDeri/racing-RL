"""Headless evaluation on fixed held-out racing tracks."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from src.env.racing_env import REWARD_PROFILES, RacingEnv
from src.track.random_track import HELD_OUT_TRACK_IDS, create_held_out_track


N_STACK = 4
ALGORITHMS = {
    "ppo": PPO,
    "sac": SAC,
}


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


def _resolve_vec_normalize_path(value):
    if value is None:
        return None
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found: {value}")
    return path


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def _lap_time_stats(results):
    """Pool every completed-lap time across episodes into mean/std/count.

    Lap time is only defined for episodes that finished at least one lap, so an
    episode contributes zero or more samples. Returns ``None`` means/std when no
    lap was completed anywhere in the set.
    """
    lap_times = [lap for result in results for lap in result["lap_times"]]
    if not lap_times:
        return {"mean": None, "std": None, "laps_timed": 0}
    lap_times = np.asarray(lap_times, dtype=np.float64)
    return {
        "mean": float(np.mean(lap_times)),
        "std": float(np.std(lap_times)),
        "laps_timed": int(lap_times.size),
    }


def _summarize_results(results):
    reasons = Counter(result["termination_reason"] for result in results)
    return {
        "episodes": len(results),
        "success_rate": float(np.mean([result["success"] for result in results])),
        "lap_time": _lap_time_stats(results),
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
        "termination_counts": dict(sorted(reasons.items())),
    }


def _evaluate_track(
    model,
    track_id,
    reward_profile,
    episodes,
    seed,
    max_episode_steps,
    vec_normalize_path=None,
    record_trajectories=0,
    frame_stack=True,
):
    def make_env():
        return RacingEnv(
            render_mode=None,
            track_creator=lambda: create_held_out_track(track_id),
            randomize_start=True,
            max_episode_steps=max_episode_steps,
            reward_profile=reward_profile,
        )

    env = DummyVecEnv([make_env])
    if vec_normalize_path is not None:
        env = VecNormalize.load(str(vec_normalize_path), env)
        env.training = False
        env.norm_reward = False
    if frame_stack:
        env = VecFrameStack(env, n_stack=N_STACK)
    env.seed(seed)
    obs = env.reset()
    results = []
    trajectories = []

    try:
        for episode_index in range(episodes):
            episode_return = 0.0
            speeds = []
            lateral_errors = []
            heading_errors = []
            recording = episode_index < record_trajectories
            path = [] if recording else None

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                info = infos[0]
                episode_return += float(rewards[0])
                speeds.append(info["speed"])
                lateral_errors.append(abs(info["e_y"]))
                heading_errors.append(abs(info["e_psi"]))
                if recording:
                    path.append([info["car_x"], info["car_y"]])

                if dones[0]:
                    if recording:
                        trajectories.append(
                            {
                                "episode": episode_index + 1,
                                "track_seed": info["track_seed"],
                                "success": bool(info["laps"] > 0),
                                "path": path,
                            }
                        )
                    results.append(
                        {
                            "episode": episode_index + 1,
                            "return": episode_return,
                            "steps": int(info["steps"]),
                            "laps": int(info["laps"]),
                            "lap_times": list(info["lap_times"]),
                            "success": bool(info["laps"] > 0),
                            "progress": float(info["total_progress"]),
                            "progress_fraction": float(info["progress_fraction"]),
                            "wall_hits": int(info["wall_hits"]),
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
    if record_trajectories:
        summary["trajectories"] = trajectories
    return summary


def evaluate(
    model_path,
    reward_profile,
    track_ids,
    episodes,
    seed,
    max_episode_steps,
    vec_normalize_path=None,
    algo="ppo",
    record_trajectories=0,
    frame_stack=True,
):
    model_path = _resolve_model_path(model_path)
    vec_normalize_path = _resolve_vec_normalize_path(vec_normalize_path)
    model_cls = ALGORITHMS[algo]
    model = model_cls.load(model_path, device="cpu")
    tracks = {}
    all_results = []

    for track_index, track_id in enumerate(track_ids):
        track_summary = _evaluate_track(
            model=model,
            track_id=track_id,
            reward_profile=reward_profile,
            episodes=episodes,
            seed=seed + track_index,
            max_episode_steps=max_episode_steps,
            vec_normalize_path=vec_normalize_path,
            record_trajectories=record_trajectories,
            frame_stack=frame_stack,
        )
        tracks[track_id] = track_summary
        all_results.extend(track_summary["episode_results"])

    return {
        "model": str(model_path),
        "algorithm": algo,
        "reward_profile": reward_profile,
        "vec_normalize": str(vec_normalize_path) if vec_normalize_path else None,
        "seed": seed,
        "episodes_per_track": episodes,
        "track_ids": list(track_ids),
        "aggregate": _summarize_results(all_results),
        "tracks": tracks,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to an SB3 model (.zip optional)")
    parser.add_argument(
        "--algo",
        choices=sorted(ALGORITHMS),
        default="ppo",
        help="Model algorithm used for loading.",
    )
    parser.add_argument("--reward-profile", choices=sorted(REWARD_PROFILES), default="v2")
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
    parser.add_argument(
        "--vec-normalize",
        help="Optional VecNormalize .pkl stats saved by Phase 7d training.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    parser.add_argument(
        "--record-trajectories",
        type=int,
        default=0,
        help="Phase 9: record car (x, y) paths for the first N episodes per "
        "track. Written to <output>.trajectories.json when --output is set.",
    )
    parser.add_argument(
        "--no-frame-stack",
        action="store_true",
        help="Evaluate without VecFrameStack; required for SAC models trained "
        "with train_sac.py (which does not use frame stacking).",
    )
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.max_episode_steps < 1:
        parser.error("episodes and max-episode-steps must be positive")
    if args.record_trajectories < 0:
        parser.error("record-trajectories must be non-negative")
    if args.record_trajectories > 0 and args.output is None:
        parser.error(
            "--record-trajectories requires --output (trajectories are written "
            "to <output>.trajectories.json)"
        )
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
        vec_normalize_path=args.vec_normalize,
        algo=args.algo,
        record_trajectories=args.record_trajectories,
        frame_stack=not args.no_frame_stack,
    )

    # Split bulky trajectory paths into a sidecar so the metrics JSON stays lean.
    trajectories = {
        track_id: track.pop("trajectories")
        for track_id, track in summary["tracks"].items()
        if "trajectories" in track
    }

    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        if trajectories:
            traj_path = args.output.with_suffix(".trajectories.json")
            traj_path.write_text(
                json.dumps({"model": summary["model"], "tracks": trajectories})
                + "\n",
                encoding="utf-8",
            )
            print(f"Trajectories saved to {traj_path}")


if __name__ == "__main__":
    main()
