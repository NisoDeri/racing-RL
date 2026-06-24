"""SAC training runner for the Phase 8 PPO-vs-SAC comparison."""

import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.env.racing_env import REWARD_PROFILES
from src.track.random_track import VALIDATION_RANDOM_TRACK_SEED
from train import (
    DEFAULT_GAMMA,
    N_STACK,
    RacingMetricsCallback,
    SyncVecNormalizeCallback,
    _detect_device,
    _detect_n_envs,
    _find_vec_normalize,
    _wrap_vec_env,
    make_env,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train SAC for Phase 8 PPO-vs-SAC comparison."
    )
    parser.add_argument("--reward-profile", choices=sorted(REWARD_PROFILES), default="v2")
    parser.add_argument("--track-mode", choices=("sprint", "random"), default="random")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=int(os.environ.get("N_ENVS", _detect_n_envs())),
    )
    parser.add_argument(
        "--device", default=os.environ.get("DEVICE", _detect_device())
    )
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=-1,
        help="SAC gradient steps after each rollout; -1 matches collected transitions.",
    )
    parser.add_argument("--ent-coef", default="auto")
    parser.add_argument(
        "--vec-normalize-reward",
        action="store_true",
        help="Normalize discounted returns with SB3 VecNormalize.",
    )
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--log-dir", default="logs/phase8/sac")
    parser.add_argument("--model-dir", default="models/phase8/sac")
    parser.add_argument("--run-name")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args(argv)

    if args.timesteps < 1 or args.n_envs < 1:
        parser.error("timesteps >= 1 and n-envs >= 1 are required")
    if args.buffer_size < 1 or args.learning_starts < 0 or args.batch_size < 1:
        parser.error("buffer-size, learning-starts, and batch-size are invalid")
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("gamma must be in [0, 1]")
    if not 0.0 < args.tau <= 1.0:
        parser.error("tau must be in (0, 1]")
    if args.learning_rate <= 0.0:
        parser.error("learning-rate must be positive")
    if args.train_freq < 1 or args.gradient_steps == 0:
        parser.error("train-freq must be positive and gradient-steps cannot be 0")
    if args.eval_freq < 1 or args.eval_episodes < 1 or args.checkpoint_freq < 1:
        parser.error("eval/checkpoint frequencies and eval-episodes must be positive")
    if args.max_episode_steps < 1:
        parser.error("max-episode-steps must be positive")
    if args.run_name is None:
        args.run_name = f"sac_{args.track_mode}_{args.reward_profile}"
    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    eval_label = (
        f"validation seed {VALIDATION_RANDOM_TRACK_SEED}"
        if args.track_mode == "random"
        else "sprint"
    )

    print("=" * 60)
    print("SAC training")
    print(f"  Run       : {args.run_name} (reward {args.reward_profile})")
    print(f"  Tracks    : {args.track_mode}")
    print(f"  Eval track: {eval_label}")
    print(f"  Device    : {args.device}")
    print(f"  Envs      : {args.n_envs} x {vec_cls.__name__}  (frame stack: none — SAC uses raw obs)")
    print(f"  Buffer    : {args.buffer_size:,}  batch={args.batch_size}")
    print(f"  Updates   : train_freq={args.train_freq}, gradient_steps={args.gradient_steps}")
    print(f"  Gamma/tau : gamma={args.gamma:.3f}, tau={args.tau:.3f}")
    print(f"  Entropy   : ent_coef={args.ent_coef}")
    print(f"  Normalize : reward={'yes' if args.vec_normalize_reward else 'no'}")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Logs      : {args.log_dir}  ->  tensorboard --logdir {args.log_dir}")
    print("=" * 60)
    print()

    train_env = make_vec_env(
        make_env(
            track_mode=args.track_mode,
            reward_profile=args.reward_profile,
            max_episode_steps=args.max_episode_steps,
        ),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_cls,
    )
    train_env = _wrap_vec_env(
        train_env,
        vec_normalize_reward=args.vec_normalize_reward,
        gamma=args.gamma,
        frame_stack=False,
    )

    eval_track_mode = "validation" if args.track_mode == "random" else "sprint"
    eval_env = make_vec_env(
        make_env(
            track_mode=eval_track_mode,
            reward_profile=args.reward_profile,
            max_episode_steps=args.max_episode_steps,
        ),
        n_envs=1,
        seed=args.seed + 99,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = _wrap_vec_env(
        eval_env,
        vec_normalize_reward=args.vec_normalize_reward,
        gamma=args.gamma,
        training=False,
        frame_stack=False,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
        device=args.device,
    )

    callbacks = [
        RacingMetricsCallback(),
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=args.model_dir,
            name_prefix=args.run_name,
        ),
        EvalCallback(
            eval_env,
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            best_model_save_path=os.path.join(args.model_dir, args.run_name),
            log_path=os.path.join(args.log_dir, args.run_name),
            deterministic=True,
            verbose=1,
        ),
    ]
    if args.vec_normalize_reward:
        callbacks.insert(0, SyncVecNormalizeCallback(train_env, eval_env))

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=not args.no_progress_bar,
    )

    final_path = os.path.join(args.model_dir, f"{args.run_name}_final")
    model.save(final_path)
    print(f"\nTraining complete. Final SAC model saved to {final_path}.zip")
    vec_normalize = _find_vec_normalize(train_env)
    if vec_normalize is not None:
        stats_path = f"{final_path}_vecnormalize.pkl"
        vec_normalize.save(stats_path)
        print(f"VecNormalize stats saved to {stats_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
