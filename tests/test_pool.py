"""
Tests for CheckpointPool and PolicyOpponent (Phase 5e self-play).

All tests are headless — no display required.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.env.pool import CheckpointPool
from src.env.opponents import CURRICULUM_OPPONENTS, PolicyOpponent
from src.env.racing_env import RacingEnv
from src.track.track import Track


N_STACK = 4


def _make_pool_model(tmp_path):
    """Create a minimal PPO model matching the RacingEnv obs space."""

    def _factory():
        return RacingEnv(
            render_mode=None,
            track_creator=lambda: Track.create_sprint_track(track_width=14),
        )

    stacked_env = VecFrameStack(DummyVecEnv([_factory]), n_stack=N_STACK)
    model = PPO("MlpPolicy", stacked_env, verbose=0)
    path = str(tmp_path / "model.zip")
    model.save(path)
    stacked_env.close()
    return model, path


# ---------------------------------------------------------------------------
# Test 1 — Basic add / size / sample round-trip
# ---------------------------------------------------------------------------


def test_pool_add_and_sample(tmp_path):
    pool = CheckpointPool(str(tmp_path / "pool"))
    _, model_path = _make_pool_model(tmp_path)

    for step in (1000, 2000, 3000):
        pool.add_from_path(model_path, step=step)

    assert pool.size() == 3
    assert pool.snapshot_steps() == [1000, 2000, 3000]

    sampled = pool.sample(n=2)
    assert len(sampled) == 2
    # Both should be loaded PPO instances
    for m in sampled:
        assert hasattr(m, "predict")


# ---------------------------------------------------------------------------
# Test 2 — Eviction: pool never exceeds max_size
# ---------------------------------------------------------------------------


def test_pool_eviction(tmp_path):
    pool = CheckpointPool(str(tmp_path / "pool"), max_size=3)
    _, model_path = _make_pool_model(tmp_path)

    for i in range(5):
        pool.add_from_path(model_path, step=i * 1000)

    assert pool.size() == 3
    # Only the three most-recent snapshots should remain.
    assert pool.snapshot_steps() == [2000, 3000, 4000]


# ---------------------------------------------------------------------------
# Test 3 — Bootstrap from an existing .zip
# ---------------------------------------------------------------------------


def test_pool_bootstrap_from_path(tmp_path):
    pool = CheckpointPool(str(tmp_path / "pool"))
    assert pool.size() == 0

    _, model_path = _make_pool_model(tmp_path)
    pool.add_from_path(model_path, step=0)

    assert pool.size() == 1
    sampled = pool.sample(1)
    assert sampled[0] is not None


# ---------------------------------------------------------------------------
# Test 4 — PolicyOpponent can be stepped inside a RacingEnv
# ---------------------------------------------------------------------------


def test_policy_opponent_step(tmp_path):
    """5e env: reset + step cycle does not raise; info keys are present."""
    pool_dir = str(tmp_path / "pool")
    _, model_path = _make_pool_model(tmp_path)

    pool = CheckpointPool(pool_dir)
    pool.add_from_path(model_path, step=0)

    spec_5e = CURRICULUM_OPPONENTS["5e"]
    env = RacingEnv(
        render_mode=None,
        track_creator=lambda: Track.create_sprint_track(track_width=14),
        opponent_spec=spec_5e,
        pool_dir=pool_dir,
        reward_profile="v3",
    )

    obs, info = env.reset()
    assert obs.shape == (34,)
    assert info["num_opponents"] == 3
    assert info["opponent_mode"] == "pool_agent"

    # Verify opponents are PolicyOpponent instances.
    assert all(isinstance(opp, PolicyOpponent) for opp in env.opponents)

    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset()

    assert "car_collisions" in info
    assert "overtake_count" in info
    env.close()
