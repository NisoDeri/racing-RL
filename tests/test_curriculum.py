"""
Phase 5 curriculum tests.

Headless tests covering the OpponentSpec catalog, opponent spawning and
kinematic updates in RacingEnv, and the v3 reward profile's car-aware terms.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from config import RACE, SIM
from src.env.opponents import CURRICULUM_OPPONENTS, OpponentSpec
from src.env.racing_env import REWARD_PROFILES, RacingEnv
from src.track.track import Track


def _sprint_track():
    return Track.create_sprint_track(track_width=14)


def _make_env(stage=None, reward_profile=None):
    return RacingEnv(
        render_mode=None,
        track_creator=_sprint_track,
        max_episode_steps=200,
        reward_profile=reward_profile,
        opponent_spec=CURRICULUM_OPPONENTS[stage] if stage else None,
    )


# ============================================================
# Test 1 — Curriculum spec catalog matches the planned table
# ============================================================

def test_curriculum_specs_catalog():
    expected = {
        "5a": ("stationary", 0, 0.0, ()),
        "5b": ("stationary", 1, 0.0, (40.0,)),
        "5c": ("centerline_follower", 1, 0.5, (40.0,)),
        "5d": ("centerline_follower", 3, 0.5, (35.0, 80.0, 125.0)),
    }
    for stage, (mode, count, frac, offsets) in expected.items():
        spec = CURRICULUM_OPPONENTS[stage]
        assert isinstance(spec, OpponentSpec)
        assert spec.mode == mode
        assert spec.count == count
        assert spec.speed_fraction == pytest.approx(frac)
        assert spec.spawn_offsets == offsets


# ============================================================
# Test 2 — Stage 5d spawns exactly three opponents on reset
# ============================================================

def test_stage_5d_spawns_three_opponents():
    env = _make_env(stage="5d")
    try:
        env.reset(seed=0)
        assert len(env.opponents) == 3
        # Each opponent has a unique car_id distinct from the ego (car_id=0).
        car_ids = sorted(opp.car.car_id for opp in env.opponents)
        assert car_ids == [1, 2, 3]
        assert env.car.car_id == 0
    finally:
        env.close()


# ============================================================
# Test 3 — Stationary opponent does not move across steps
# ============================================================

def test_stationary_opponent_holds_position():
    env = _make_env(stage="5b", reward_profile="v3")
    try:
        env.reset(seed=0)
        opponent = env.opponents[0]
        initial_pos = np.array(opponent.position, copy=True)
        no_op = np.zeros(2, dtype=np.float32)
        for _ in range(100):
            env.step(no_op)
        final_pos = np.array(opponent.position, copy=True)
        # Stationary opponents must remain within a millimeter of spawn.
        assert np.linalg.norm(final_pos - initial_pos) < 1e-3
    finally:
        env.close()


# ============================================================
# Test 4 — Centerline-follower advances at the configured speed
# ============================================================

def test_centerline_follower_advances():
    env = _make_env(stage="5c", reward_profile="v3")
    try:
        env.reset(seed=0)
        opponent = env.opponents[0]
        initial_s = opponent.s
        steps = 100
        no_op = np.zeros(2, dtype=np.float32)
        for _ in range(steps):
            env.step(no_op)
        expected_ds = opponent.speed * SIM.time_step * steps
        actual_ds = (opponent.s - initial_s) % env.track.total_length
        # Tolerance covers float precision and lap wrap-around guards.
        assert actual_ds == pytest.approx(expected_ds, abs=1e-3)
        assert opponent.speed == pytest.approx(
            CURRICULUM_OPPONENTS["5c"].speed_fraction * RACE.static_control_speed
        )
    finally:
        env.close()


# ============================================================
# Test 5 — v3 reward applies the car_contact penalty
# ============================================================

def test_v3_car_contact_penalty_fires():
    env = _make_env(stage="5b", reward_profile="v3")
    try:
        env.reset(seed=0)
        # Force the collision handler to report a car-car contact for the ego car.
        ego_stats = env.world.collision_handler.car_stats[env.car.car_id]
        ego_stats["touching_car"] = True
        no_op = np.zeros(2, dtype=np.float32)
        _, _, _, _, info = env.step(no_op)
        assert info["touching_car"] is True
        assert info["reward_terms"]["car_contact"] == pytest.approx(
            -REWARD_PROFILES["v3"].car_contact_penalty
        )
    finally:
        env.close()
