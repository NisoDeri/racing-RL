"""
Phase 2 — PPO baseline training on Sprint Circuit.

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
import os
import multiprocessing
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecFrameStack, VecMonitor, SubprocVecEnv, DummyVecEnv
)
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback
)

from src.track.track import Track
from src.env.racing_env import RacingEnv


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
# Config
# ============================================================

TOTAL_TIMESTEPS = 1_000_000   # Phase 2 target (change to 5_000_000 for Phase 4)
N_STACK         = 4           # Frame stacking (gives the agent velocity info)
EVAL_FREQ       = 50_000      # Evaluate every N timesteps
CHECKPOINT_FREQ = 100_000     # Save model every N timesteps
LOG_DIR         = "logs/"
MODEL_DIR       = "models/"
SEED            = 42

# Resolved at startup — override by setting env vars N_ENVS / DEVICE before running.
N_ENVS = int(os.environ.get("N_ENVS", _detect_n_envs()))
DEVICE = os.environ.get("DEVICE", _detect_device())

# Larger batch amortizes GPU/MPS launch overhead; keep it a factor of n_steps*N_ENVS.
BATCH_SIZE = min(512, (2048 * N_ENVS) // 8)


# ============================================================
# Env factory — top-level named function so SubprocVecEnv can pickle it
# ============================================================

def _sprint_track():
    return Track.create_sprint_track(track_width=14)


def make_env(seed=0):
    def _init():
        env = RacingEnv(
            render_mode=None,
            track_creator=_sprint_track,
            max_episode_steps=6000,
        )
        env.reset(seed=seed)
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

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:                       # episode just ended
                self._ep_wall_hits.append(info.get("wall_hits", 0))
                self._ep_laps.append(info.get("laps", 0))

        if len(self._ep_wall_hits) >= 10:               # log every 10 episodes
            self.logger.record("racing/mean_wall_hits", np.mean(self._ep_wall_hits))
            self.logger.record("racing/mean_laps",      np.mean(self._ep_laps))
            self._ep_wall_hits.clear()
            self._ep_laps.clear()
        return True


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    vec_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv

    print("=" * 60)
    print("Phase 2 — PPO training on Sprint Circuit")
    print(f"  Device    : {DEVICE}")
    print(f"  Envs      : {N_ENVS} × {vec_cls.__name__}  (frame stack: {N_STACK})")
    print(f"  Batch     : {BATCH_SIZE}  (rollout buffer: {2048 * N_ENVS:,} transitions)")
    print(f"  Timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Logs      : {LOG_DIR}  →  tensorboard --logdir {LOG_DIR}")
    print("=" * 60)
    print("Override hardware:  N_ENVS=8 DEVICE=cpu python train.py")
    print()

    # --- Training envs (SubprocVecEnv = one OS process per env, true parallelism) ---
    train_env = make_vec_env(
        make_env(seed=SEED), n_envs=N_ENVS, seed=SEED, vec_env_cls=vec_cls
    )
    train_env = VecFrameStack(train_env, n_stack=N_STACK)
    train_env = VecMonitor(train_env)   # required for ep_rew_mean / ep_len_mean logging

    # --- Eval env (single env, always DummyVecEnv — no benefit from SubprocVecEnv) ---
    eval_env = make_vec_env(
        make_env(seed=SEED + 99), n_envs=1, seed=SEED + 99, vec_env_cls=DummyVecEnv
    )
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK)

    # --- PPO ---
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        n_steps=2048,
        batch_size=BATCH_SIZE,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=SEED,
        device=DEVICE,
    )

    # --- Callbacks ---
    callbacks = [
        RacingMetricsCallback(),
        CheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
            save_path=MODEL_DIR,
            name_prefix="ppo_sprint",
        ),
        EvalCallback(
            eval_env,
            eval_freq=max(EVAL_FREQ // N_ENVS, 1),
            n_eval_episodes=5,
            best_model_save_path=MODEL_DIR,
            log_path=LOG_DIR,
            deterministic=True,
            verbose=1,
        ),
    ]

    # --- Train ---
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(os.path.join(MODEL_DIR, "ppo_sprint_final"))
    print("\nTraining complete. Final model saved to models/ppo_sprint_final.zip")
    print("Load it later with:  model = PPO.load('models/ppo_sprint_final')")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
