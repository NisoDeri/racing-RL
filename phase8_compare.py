"""Phase 8 command builder for PPO-vs-SAC comparison experiments."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


DEFAULT_PPO_MODEL_TEMPLATE = (
    "models/phase4/v2/seed{seed}/phase4_v2_seed{seed}/best_model.zip"
)


def build_sac_run_name(track_mode, reward_profile, seed, normalize_reward=False):
    suffix = "_normrew" if normalize_reward else ""
    return f"phase8_sac_{track_mode}_{reward_profile}_seed{seed}{suffix}"


def build_sac_train_command(args, seed):
    run_name = build_sac_run_name(
        args.track_mode,
        args.reward_profile,
        seed,
        args.vec_normalize_reward,
    )
    cmd = [
        args.python,
        "train_sac.py",
        "--track-mode",
        args.track_mode,
        "--reward-profile",
        args.reward_profile,
        "--seed",
        str(seed),
        "--timesteps",
        str(args.timesteps),
        "--n-envs",
        str(args.n_envs),
        "--buffer-size",
        str(args.buffer_size),
        "--learning-starts",
        str(args.learning_starts),
        "--batch-size",
        str(args.batch_size),
        "--train-freq",
        str(args.train_freq),
        "--gradient-steps",
        str(args.gradient_steps),
        "--eval-freq",
        str(args.eval_freq),
        "--eval-episodes",
        str(args.eval_episodes),
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--log-dir",
        args.sac_log_dir,
        "--model-dir",
        args.sac_model_dir,
        "--run-name",
        run_name,
        "--no-progress-bar",
    ]
    if args.vec_normalize_reward:
        cmd.append("--vec-normalize-reward")
    return cmd


def build_sac_eval_command(args, seed):
    run_name = build_sac_run_name(
        args.track_mode,
        args.reward_profile,
        seed,
        args.vec_normalize_reward,
    )
    model_path = f"{args.sac_model_dir}/{run_name}_final.zip"
    output_path = f"{args.results_dir}/{run_name}_heldout.json"
    cmd = [
        args.python,
        "evaluate.py",
        model_path,
        "--algo",
        "sac",
        "--no-frame-stack",
        "--reward-profile",
        args.reward_profile,
        "--tracks",
        "held-out",
        "--episodes",
        str(args.heldout_episodes),
        "--output",
        output_path,
    ]
    if args.vec_normalize_reward:
        cmd.extend([
            "--vec-normalize",
            f"{args.sac_model_dir}/{run_name}_final_vecnormalize.pkl",
        ])
    return cmd


def build_ppo_eval_command(args, seed):
    model_path = args.ppo_model_template.format(seed=seed)
    output_path = f"{args.results_dir}/phase8_ppo_{args.reward_profile}_seed{seed}_heldout.json"
    return [
        args.python,
        "evaluate.py",
        model_path,
        "--algo",
        "ppo",
        "--reward-profile",
        args.reward_profile,
        "--tracks",
        "held-out",
        "--episodes",
        str(args.heldout_episodes),
        "--output",
        output_path,
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--track-mode", choices=("sprint", "random"), default="random")
    parser.add_argument("--reward-profile", default="v2")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=-1)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--heldout-episodes", type=int, default=100)
    parser.add_argument("--sac-log-dir", default="logs/phase8/sac")
    parser.add_argument("--sac-model-dir", default="models/phase8/sac")
    parser.add_argument("--results-dir", default="results/phase8")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--ppo-model-template",
        default=DEFAULT_PPO_MODEL_TEMPLATE,
        help="Format string for PPO baseline model paths; may use {seed}.",
    )
    parser.add_argument(
        "--skip-ppo-eval",
        action="store_true",
        help="Only include SAC held-out evaluation commands.",
    )
    parser.add_argument(
        "--vec-normalize-reward",
        action="store_true",
        help="Train/evaluate SAC with reward normalization stats.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run commands after printing them. Without this, only print/manifest.",
    )
    parser.add_argument("--manifest", type=Path, help="Optional JSON manifest path.")
    args = parser.parse_args(argv)

    if not args.seeds:
        parser.error("at least one seed is required")
    if args.timesteps < 1 or args.n_envs < 1:
        parser.error("timesteps >= 1 and n-envs >= 1 are required")
    if args.buffer_size < 1 or args.learning_starts < 0 or args.batch_size < 1:
        parser.error("buffer-size, learning-starts, and batch-size are invalid")
    if args.train_freq < 1 or args.gradient_steps == 0:
        parser.error("train-freq must be positive and gradient-steps cannot be 0")
    if args.eval_freq < 1 or args.eval_episodes < 1:
        parser.error("eval-freq and eval-episodes must be positive")
    if args.checkpoint_freq < 1 or args.heldout_episodes < 1:
        parser.error("checkpoint-freq and heldout-episodes must be positive")
    return args


def build_manifest(args):
    sac_train_commands = [
        build_sac_train_command(args, seed)
        for seed in args.seeds
    ]
    sac_eval_commands = [
        build_sac_eval_command(args, seed)
        for seed in args.seeds
    ]
    ppo_eval_commands = [] if args.skip_ppo_eval else [
        build_ppo_eval_command(args, seed)
        for seed in args.seeds
    ]
    return {
        "phase": "8",
        "description": "PPO-vs-SAC algorithm comparison commands",
        "seeds": list(args.seeds),
        "sac_train_commands": sac_train_commands,
        "sac_eval_commands": sac_eval_commands,
        "ppo_eval_commands": ppo_eval_commands,
    }


def main(argv=None):
    args = parse_args(argv)
    manifest = build_manifest(args)

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    commands = (
        manifest["sac_train_commands"]
        + manifest["sac_eval_commands"]
        + manifest["ppo_eval_commands"]
    )
    for cmd in commands:
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)

    return manifest


if __name__ == "__main__":
    main()
