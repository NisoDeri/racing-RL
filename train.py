"""
PPO training for fixed-track baselines and Phase 4 domain randomization.

Run:
    python train.py

Monitor training live:
    tensorboard --logdir logs/

What to watch in TensorBoard:
    rollout/ep_rew_mean  — average episode reward      (should rise)
    rollout/ep_len_mean  — average episode length       (longer = survives longer)
    train/policy_loss    — how much the policy changes  (should shrink over time)
    train/value_loss     — critic accuracy              (should shrink)

"Learning" signal: ep_len_mean going from ~60 steps → 500+ steps is the clearest sign
the agent stopped crashing immediately and learned to stay on track.
"""
import argparse
import os
import multiprocessing
from collections import Counter
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecFrameStack, SubprocVecEnv, DummyVecEnv, VecNormalize
)
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback
)

from src.track.track import Track
from src.track.random_track import (
    VALIDATION_RANDOM_TRACK_SEED,
    RandomTrackGenerator,
    create_validation_track,
)
from src.env.racing_env import REWARD_PROFILES, RacingEnv
from src.rl.auxiliary import (
    AuxRaycastActorCriticPolicy,
    AuxRaycastPredictionCallback,
)


# ============================================================
# Hardware detection — runs once at import time
# ============================================================

def _detect_device():
    # MlpPolicy is tiny (34→256→256→2). CPU↔GPU transfer overhead dominates
    # for small batches — SB3 explicitly warns that MPS/CUDA slows MlpPolicy.
    # Switch to "cuda" or "mps" here only if you use a CNN policy (e.g. Phase 7a).
    return "cpu"

def _detect_n_envs():
    # Leave 2 cores for the main process and OS scheduler.
    # SubprocVecEnv spawns one OS process per env, so this is real parallelism.
    return max(1, multiprocessing.cpu_count() - 2)


# ============================================================
# Defaults
# ============================================================

N_STACK         = 4           # Frame stacking (gives the agent velocity info)
DEFAULT_GAMMA   = 0.99
DEFAULT_GAE_LAMBDA = 0.95


# ============================================================
# Env factory — top-level named function so SubprocVecEnv can pickle it
# ============================================================

def _sprint_track():
    return Track.create_sprint_track(track_width=14)


def make_env(track_mode="sprint", reward_profile="v2", max_episode_steps=6000):
    def _init():
        if track_mode == "random":
            track_kwargs = {"track_generator": RandomTrackGenerator()}
        elif track_mode == "validation":
            track_kwargs = {"track_creator": create_validation_track}
        else:
            track_kwargs = {"track_creator": _sprint_track}
        env = RacingEnv(
            render_mode=None,
            randomize_start=(track_mode == "validation"),
            max_episode_steps=max_episode_steps,
            reward_profile=reward_profile,
            **track_kwargs,
        )
        return env
    return _init


# ============================================================
# Custom callback — logs extra metrics SB3 doesn't track by default
# ============================================================

class SyncVecNormalizeCallback(BaseCallback):
    """Copy running VecNormalize stats from train_env to eval_env each step.

    Ensures EvalCallback selects the best checkpoint on the same normalized
    reward scale the policy was trained on, rather than on raw returns.
    Only added to the callback list when --vec-normalize-reward is active.
    """

    def __init__(self, train_env, eval_env):
        super().__init__(verbose=0)
        self._train_vn = _find_vec_normalize(train_env)
        self._eval_vn = _find_vec_normalize(eval_env)

    def _on_step(self):
        if self._train_vn is not None and self._eval_vn is not None:
            # SB3 >=2.9 only creates obs_rms/ret_rms when the matching norm flag
            # is on. With norm_obs=False (our config) obs_rms is absent, so guard
            # both before copying or this crashes on the first step.
            if hasattr(self._train_vn, "obs_rms"):
                self._eval_vn.obs_rms = self._train_vn.obs_rms
            if hasattr(self._train_vn, "ret_rms"):
                self._eval_vn.ret_rms = self._train_vn.ret_rms
        return True


class RacingMetricsCallback(BaseCallback):
    """
    Logs per-episode wall hits and laps to TensorBoard.
    These don't appear in the default SB3 logs.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_wall_hits = []
        self._ep_laps = []
        self._ep_progress = []
        self._ep_progress_fraction = []
        self._ep_smoothness = []
        self._ep_track_lengths = []
        self._ep_track_widths = []
        self._end_reasons = Counter()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:                       # episode just ended
                self._ep_wall_hits.append(info.get("wall_hits", 0))
                self._ep_laps.append(info.get("laps", 0))
                self._ep_progress.append(info.get("total_progress", 0.0))
                self._ep_progress_fraction.append(info.get("progress_fraction", 0.0))
                self._ep_smoothness.append(
                    info.get("mean_abs_steering_change", 0.0)
                )
                self._ep_track_lengths.append(info.get("track_length", 0.0))
                self._ep_track_widths.append(info.get("track_width", 0.0))
                self._end_reasons[info.get("termination_reason") or "unknown"] += 1

        if len(self._ep_wall_hits) >= 10:               # log every 10 episodes
            self.logger.record("racing/mean_wall_hits", np.mean(self._ep_wall_hits))
            self.logger.record("racing/mean_laps",      np.mean(self._ep_laps))
            self.logger.record("racing/mean_progress", np.mean(self._ep_progress))
            self.logger.record(
                "racing/mean_progress_fraction",
                np.mean(self._ep_progress_fraction),
            )
            self.logger.record(
                "racing/mean_steering_change", np.mean(self._ep_smoothness)
            )
            self.logger.record(
                "racing/lap_success_rate", np.mean(np.asarray(self._ep_laps) > 0)
            )
            self.logger.record(
                "racing/mean_track_length", np.mean(self._ep_track_lengths)
            )
            self.logger.record(
                "racing/mean_track_width", np.mean(self._ep_track_widths)
            )
            episode_count = sum(self._end_reasons.values())
            for reason, count in self._end_reasons.items():
                self.logger.record(f"racing/end_{reason}", count / episode_count)
            self._ep_wall_hits.clear()
            self._ep_laps.clear()
            self._ep_progress.clear()
            self._ep_progress_fraction.clear()
            self._ep_smoothness.clear()
            self._ep_track_lengths.clear()
            self._ep_track_widths.clear()
            self._end_reasons.clear()
        return True


def _default_batch_size(rollout_size):
    target = min(512, max(2, rollout_size // 8))
    for candidate in (512, 256, 128, 64, 32, 16, 8, 4, 2):
        if candidate <= target and rollout_size % candidate == 0:
            return candidate
    return 2


def _wrap_vec_env(
    env,
    *,
    vec_normalize_reward=False,
    gamma=DEFAULT_GAMMA,
    training=True,
    frame_stack=True,
):
    if vec_normalize_reward:
        env = VecNormalize(
            env,
            norm_obs=False,
            norm_reward=training,
            gamma=gamma,
            training=training,
        )
    if frame_stack:
        env = VecFrameStack(env, n_stack=N_STACK)
    return env


def _find_vec_normalize(env):
    current = env
    while current is not None:
        if isinstance(current, VecNormalize):
            return current
        current = getattr(current, "venv", None)
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train PPO on Sprint Circuit or randomized tracks."
    )
    parser.add_argument("--reward-profile", choices=sorted(REWARD_PROFILES), default="v2")
    parser.add_argument("--track-mode", choices=("sprint", "random"), default="sprint")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=int(os.environ.get("N_ENVS", _detect_n_envs())),
    )
    parser.add_argument(
        "--device", default=os.environ.get("DEVICE", _detect_device())
    )
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=DEFAULT_GAE_LAMBDA)
    parser.add_argument(
        "--vec-normalize-reward",
        action="store_true",
        help="Phase 7d: normalize discounted returns with SB3 VecNormalize.",
    )
    parser.add_argument(
        "--aux-raycast-prediction",
        action="store_true",
        help="Phase 7a: add an auxiliary head that predicts the next raycast vector.",
    )
    parser.add_argument(
        "--aux-loss-coef",
        type=float,
        default=0.05,
        help="Weight applied to the auxiliary raycast MSE loss.",
    )
    parser.add_argument(
        "--aux-batch-size",
        type=int,
        default=256,
        help="Mini-batch size for auxiliary raycast updates.",
    )
    parser.add_argument(
        "--aux-gradient-steps",
        type=int,
        default=1,
        help="Auxiliary optimizer passes per PPO rollout.",
    )
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--run-name")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args(argv)

    if args.timesteps is None:
        args.timesteps = 5_000_000 if args.track_mode == "random" else 1_000_000
    if args.n_envs < 1 or args.n_steps < 2 or args.timesteps < 1:
        parser.error("n-envs >= 1, n-steps >= 2, and timesteps >= 1 are required")
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("gamma must be in [0, 1]")
    if not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("gae-lambda must be in [0, 1]")
    if args.aux_loss_coef < 0.0:
        parser.error("aux-loss-coef must be non-negative")
    if args.aux_batch_size < 1 or args.aux_gradient_steps < 1:
        parser.error("aux-batch-size and aux-gradient-steps must be positive")
    rollout_size = args.n_envs * args.n_steps
    if args.batch_size is None:
        args.batch_size = _default_batch_size(rollout_size)
    if args.batch_size < 2 or rollout_size % args.batch_size != 0:
        parser.error("batch-size must be >= 2 and divide n-envs * n-steps")
    if args.run_name is None:
        args.run_name = f"ppo_{args.track_mode}_{args.reward_profile}"
    return args


# ============================================================
# Main
# ============================================================

def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv

    print("=" * 60)
    print("PPO training")
    print(f"  Run       : {args.run_name} (reward {args.reward_profile})")
    print(f"  Tracks    : {args.track_mode}")
    eval_label = (
        f"validation seed {VALIDATION_RANDOM_TRACK_SEED}"
        if args.track_mode == "random"
        else "sprint"
    )
    print(f"  Eval track: {eval_label}")
    print(f"  Device    : {args.device}")
    print(f"  Envs      : {args.n_envs} × {vec_cls.__name__}  (frame stack: {N_STACK})")
    print(f"  Batch     : {args.batch_size}  (rollout buffer: {args.n_steps * args.n_envs:,} transitions)")
    print(f"  Gamma/GAE : gamma={args.gamma:.3f}, lambda={args.gae_lambda:.3f}")
    print(f"  Normalize : reward={'yes' if args.vec_normalize_reward else 'no'}")
    aux_label = "yes" if args.aux_raycast_prediction else "no"
    print(f"  Auxiliary : next-raycast prediction={aux_label}")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Logs      : {args.log_dir}  →  tensorboard --logdir {args.log_dir}")
    print("=" * 60)
    print("Override hardware:  python train.py --n-envs 8 --device cpu")
    print()

    # --- Training envs (SubprocVecEnv = one OS process per env, true parallelism) ---
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
    )

    # Use a reserved validation track for model selection. Final held-out tracks
    # are evaluated only after training via evaluate.py.
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
    )

    # --- PPO ---
    policy = (
        AuxRaycastActorCriticPolicy
        if args.aux_raycast_prediction
        else "MlpPolicy"
    )
    model = PPO(
        policy,
        train_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
        device=args.device,
    )

    # --- Callbacks ---
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
    if args.aux_raycast_prediction:
        callbacks.insert(
            1,
            AuxRaycastPredictionCallback(
                loss_coef=args.aux_loss_coef,
                batch_size=args.aux_batch_size,
                gradient_steps=args.aux_gradient_steps,
            ),
        )

    # --- Train ---
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=not args.no_progress_bar,
    )

    final_path = os.path.join(args.model_dir, f"{args.run_name}_final")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}.zip")
    vec_normalize = _find_vec_normalize(train_env)
    if vec_normalize is not None:
        stats_path = f"{final_path}_vecnormalize.pkl"
        vec_normalize.save(stats_path)
        print(f"VecNormalize stats saved to {stats_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
