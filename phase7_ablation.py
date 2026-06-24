"""Phase 7 experiment runner for GAE/n-step and reward-normalization ablations."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


DEFAULT_GAE_VALUES = (0.0, 0.5, 0.9, 0.95, 1.0)


def _format_float_for_name(value):
    return f"{value:g}".replace(".", "p")


def build_run_name(
    track_mode,
    reward_profile,
    seed,
    gae_lambda,
    normalize_reward,
    aux_raycast_prediction=False,
):
    suffixes = []
    if normalize_reward:
        suffixes.append("normrew")
    if aux_raycast_prediction:
        suffixes.append("auxray")
    suffix = f"_{'_'.join(suffixes)}" if suffixes else ""
    gae_label = _format_float_for_name(gae_lambda)
    return f"phase7_gae{gae_label}_{track_mode}_{reward_profile}_seed{seed}{suffix}"


def build_train_command(args, seed, gae_lambda):
    run_name = build_run_name(
        args.track_mode,
        args.reward_profile,
        seed,
        gae_lambda,
        args.vec_normalize_reward,
        args.aux_raycast_prediction,
    )
    cmd = [
        args.python,
        "train.py",
        "--track-mode",
        args.track_mode,
        "--reward-profile",
        args.reward_profile,
        "--seed",
        str(seed),
        "--gae-lambda",
        str(gae_lambda),
        "--timesteps",
        str(args.timesteps),
        "--n-envs",
        str(args.n_envs),
        "--n-steps",
        str(args.n_steps),
        "--eval-freq",
        str(args.eval_freq),
        "--eval-episodes",
        str(args.eval_episodes),
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--log-dir",
        args.log_dir,
        "--model-dir",
        args.model_dir,
        "--run-name",
        run_name,
        "--no-progress-bar",
    ]
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.vec_normalize_reward:
        cmd.append("--vec-normalize-reward")
    if args.aux_raycast_prediction:
        cmd.append("--aux-raycast-prediction")
        cmd.extend(["--aux-loss-coef", str(args.aux_loss_coef)])
        cmd.extend(["--aux-batch-size", str(args.aux_batch_size)])
        cmd.extend(["--aux-gradient-steps", str(args.aux_gradient_steps)])
    return cmd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gae-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_GAE_VALUES),
        help="GAE lambda values for Phase 7b.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--track-mode", choices=("sprint", "random"), default="random")
    parser.add_argument("--reward-profile", default="v2")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--log-dir", default="logs/phase7")
    parser.add_argument("--model-dir", default="models/phase7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--vec-normalize-reward",
        action="store_true",
        help="Also enable Phase 7d reward normalization for every run.",
    )
    parser.add_argument(
        "--aux-raycast-prediction",
        action="store_true",
        help="Also enable Phase 7a next-raycast auxiliary prediction.",
    )
    parser.add_argument("--aux-loss-coef", type=float, default=0.05)
    parser.add_argument("--aux-batch-size", type=int, default=256)
    parser.add_argument("--aux-gradient-steps", type=int, default=1)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch training. Without this, commands are only printed.",
    )
    parser.add_argument("--manifest", type=Path, help="Optional JSON manifest path.")
    args = parser.parse_args(argv)
    if any(not 0.0 <= value <= 1.0 for value in args.gae_values):
        parser.error("all gae-values must be in [0, 1]")
    if args.timesteps < 1 or args.n_envs < 1 or args.n_steps < 2:
        parser.error("timesteps >= 1, n-envs >= 1, and n-steps >= 2 are required")
    if args.aux_loss_coef < 0.0:
        parser.error("aux-loss-coef must be non-negative")
    if args.aux_batch_size < 1 or args.aux_gradient_steps < 1:
        parser.error("aux-batch-size and aux-gradient-steps must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    commands = [
        build_train_command(args, seed, gae_lambda)
        for seed in args.seeds
        for gae_lambda in args.gae_values
    ]
    manifest = {
        "phase": "7",
        "description": "GAE/n-step lambda ablation with optional Phase 7a/7d flags",
        "vec_normalize_reward": bool(args.vec_normalize_reward),
        "aux_raycast_prediction": bool(args.aux_raycast_prediction),
        "commands": commands,
    }

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for cmd in commands:
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)

    return manifest


if __name__ == "__main__":
    main()
