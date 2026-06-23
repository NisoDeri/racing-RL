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
    VecFrameStack, SubprocVecEnv, DummyVecEnv
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
from src.env.opponents import CURRICULUM_OPPONENTS, CURRICULUM_STAGES


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


# ============================================================
# Env factory — top-level named function so SubprocVecEnv can pickle it
# ============================================================

def _sprint_track():
    return Track.create_sprint_track(track_width=14)


def make_env(
    track_mode="sprint",
    reward_profile="v2",
    max_episode_steps=6000,
    curriculum_stage=None,
):
    opponent_spec = (
        CURRICULUM_OPPONENTS[curriculum_stage] if curriculum_stage else None
    )

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
            opponent_spec=opponent_spec,
            **track_kwargs,
        )
        return env
    return _init


# ============================================================
# Custom callback — logs extra metrics SB3 doesn't track by default
# ============================================================

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
        self._ep_car_collisions = []
        self._ep_overtakes = []
        self._ep_car_contact_steps = []
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
                self._ep_car_collisions.append(info.get("car_collisions", 0))
                self._ep_overtakes.append(info.get("overtake_count", 0))
                self._ep_car_contact_steps.append(info.get("car_contact_steps", 0))
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
            self.logger.record(
                "racing/mean_car_collisions", np.mean(self._ep_car_collisions)
            )
            self.logger.record(
                "racing/mean_overtakes", np.mean(self._ep_overtakes)
            )
            self.logger.record(
                "racing/mean_car_contact_steps",
                np.mean(self._ep_car_contact_steps),
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
            self._ep_car_collisions.clear()
            self._ep_overtakes.clear()
            self._ep_car_contact_steps.clear()
            self._end_reasons.clear()
        return True


def _default_batch_size(rollout_size):
    target = min(512, max(2, rollout_size // 8))
    for candidate in (512, 256, 128, 64, 32, 16, 8, 4, 2):
        if candidate <= target and rollout_size % candidate == 0:
            return candidate
    return 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train PPO on Sprint Circuit or randomized tracks."
    )
    parser.add_argument(
        "--reward-profile",
        choices=sorted(REWARD_PROFILES),
        default=None,
        help="Reward profile. Defaults to v2; v3 is auto-selected when "
        "--curriculum-stage is given.",
    )
    parser.add_argument("--track-mode", choices=("sprint", "random"), default="sprint")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--curriculum-stage",
        choices=CURRICULUM_STAGES,
        default=None,
        help="Phase 5 curriculum stage. Spawns opponents per CURRICULUM_OPPONENTS.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a .zip checkpoint. PPO weights are loaded; optimizer state "
        "is re-initialized so fine-tuning starts cleanly.",
    )
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
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--run-name")
    parser.add_argument("--no-progress-bar", action="store_true")
    args = parser.parse_args(argv)

    if args.reward_profile is None:
        args.reward_profile = "v3" if args.curriculum_stage else "v2"
    if args.timesteps is None:
        args.timesteps = 5_000_000 if args.track_mode == "random" else 1_000_000
    if args.n_envs < 1 or args.n_steps < 2 or args.timesteps < 1:
        parser.error("n-envs >= 1, n-steps >= 2, and timesteps >= 1 are required")
    rollout_size = args.n_envs * args.n_steps
    if args.batch_size is None:
        args.batch_size = _default_batch_size(rollout_size)
    if args.batch_size < 2 or rollout_size % args.batch_size != 0:
        parser.error("batch-size must be >= 2 and divide n-envs * n-steps")
    if args.run_name is None:
        if args.curriculum_stage:
            args.run_name = (
                f"{args.curriculum_stage}_{args.reward_profile}_seed{args.seed}"
            )
        else:
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

    curriculum_label = args.curriculum_stage or "none"
    checkpoint_label = args.load_checkpoint or "none"

    print("=" * 60)
    print("PPO training")
    print(f"  Run       : {args.run_name} (reward {args.reward_profile})")
    print(f"  Curriculum: {curriculum_label}")
    print(f"  Checkpoint: {checkpoint_label}")
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
            curriculum_stage=args.curriculum_stage,
        ),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_cls,
    )
    train_env = VecFrameStack(train_env, n_stack=N_STACK)

    # Use a reserved validation track for model selection. Final held-out tracks
    # are evaluated only after training via evaluate.py.
    eval_track_mode = "validation" if args.track_mode == "random" else "sprint"
    eval_env = make_vec_env(
        make_env(
            track_mode=eval_track_mode,
            reward_profile=args.reward_profile,
            max_episode_steps=args.max_episode_steps,
            curriculum_stage=args.curriculum_stage,
        ),
        n_envs=1,
        seed=args.seed + 99,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK)

    # --- PPO ---
    if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
        if not os.path.exists(checkpoint_path) and not checkpoint_path.endswith(".zip"):
            checkpoint_path = checkpoint_path + ".zip"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.load_checkpoint}")
        print(f"Loading PPO weights from {checkpoint_path} (optimizer state reset).")
        model = PPO.load(
            checkpoint_path,
            env=train_env,
            device=args.device,
            tensorboard_log=args.log_dir,
        )
        model.seed = args.seed
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs=dict(net_arch=[256, 256]),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
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

    # --- Train ---
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=not args.no_progress_bar,
    )

    final_path = os.path.join(args.model_dir, f"{args.run_name}_final")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
