"""
Regression tests for RacingEnv — Phase 1 observation/reward refactor.

Run with:
    pytest tests/test_racing_env.py -v

All tests are headless (render_mode=None) — no display required.
"""
import sys
import os
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.env.racing_env import (
    REWARD_PROFILES,
    RacingEnv,
    _LAT_VEL_NORM,
    _SPEED_NORM,
)
from src.track.track import Track


# ============================================================
# Shared fixtures
# ============================================================

def _sprint_track():
    """Sprint track creator — faster than Grand Prix for tests."""
    return Track.create_sprint_track(track_width=14)


@pytest.fixture
def env():
    """Fresh headless env, reset and ready."""
    e = RacingEnv(render_mode=None, track_creator=_sprint_track, max_episode_steps=200)
    e.reset(seed=0)
    yield e
    e.close()


# ============================================================
# Test 1 — Observation shape
# ============================================================

def test_observation_shape(env):
    obs, _ = env.reset(seed=42)
    assert obs.shape == (34,), f"Expected (34,), got {obs.shape}"


# ============================================================
# Test 2 — Observation space bounds are correct
# ============================================================

def test_observation_space_bounds():
    e = RacingEnv(render_mode=None, track_creator=_sprint_track)
    space = e.observation_space
    assert space.shape == (34,)

    # Rays [0-29]
    assert np.all(space.low[0:30] == 0.0)
    assert np.all(space.high[0:30] == 1.0)
    # Speed [30]
    assert space.low[30] == pytest.approx(0.0)
    assert space.high[30] == pytest.approx(2.0)
    # Lateral velocity [31]
    assert space.low[31] == pytest.approx(-2.0)
    assert space.high[31] == pytest.approx(2.0)
    # Throttle [32]
    assert space.low[32] == pytest.approx(-1.0)
    assert space.high[32] == pytest.approx(1.0)
    # Steering [33]
    assert space.low[33] == pytest.approx(-1.0)
    assert space.high[33] == pytest.approx(1.0)
    e.close()


# ============================================================
# Test 3 — All obs values stay within declared space bounds
# ============================================================

def test_obs_values_within_space(env):
    obs, _ = env.reset(seed=1)
    _check_obs_in_bounds(obs, env.observation_space)

    rng = np.random.default_rng(seed=1)
    for _ in range(30):
        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        _check_obs_in_bounds(obs, env.observation_space)
        if terminated or truncated:
            obs, _ = env.reset(seed=1)


def _check_obs_in_bounds(obs, space, tol=1e-5):
    too_low = obs < space.low - tol
    too_high = obs > space.high + tol
    assert not np.any(too_low), f"obs below low at indices {np.where(too_low)[0]}: {obs[too_low]}"
    assert not np.any(too_high), f"obs above high at indices {np.where(too_high)[0]}: {obs[too_high]}"


# ============================================================
# Test 4 — Wall hit counter increments on collision
#          (also verifies track.create_walls() is called in reset)
# ============================================================

def test_wall_hit_counter_increments():
    """
    Drive aggressively into the wall and confirm info['wall_hits'] rises.
    This test exercises the create_walls() fix: without it, Box2D has no
    wall bodies and wall_hits would always be 0.
    """
    e = RacingEnv(render_mode=None, track_creator=_sprint_track, max_episode_steps=600)
    e.reset(seed=0)

    wall_hit_found = False
    for _ in range(600):
        obs, reward, terminated, truncated, info = e.step(
            np.array([1.0, 0.9], dtype=np.float32)
        )
        if info['wall_hits'] > 0:
            wall_hit_found = True
            # Also confirm the penalty was deducted (reward should be affected)
            break
        if terminated or truncated:
            e.reset(seed=0)

    e.close()
    assert wall_hit_found, (
        "Expected wall_hits > 0 after 600 steps of aggressive driving. "
        "Check that track.create_walls() is called in reset() and "
        "the collision handler is active."
    )


# ============================================================
# Test 5 — Smoothness penalty: jittery steering → lower reward
# ============================================================

def test_smoothness_penalty_jittery_worse_than_smooth():
    """
    Same throttle, same env seed. Alternating ±1 steering should yield
    a lower cumulative reward than constant 0 steering due to the
    0.5 * |Δsteering| penalty per step.
    """
    N = 50

    def _run(steerings):
        e = RacingEnv(render_mode=None, track_creator=_sprint_track, max_episode_steps=N)
        e.reset(seed=7)
        total = 0.0
        for s in steerings:
            _, reward, terminated, truncated, _ = e.step(
                np.array([0.5, s], dtype=np.float32)
            )
            total += reward
            if terminated or truncated:
                break
        e.close()
        return total

    smooth_reward  = _run([0.0] * N)
    jittery_reward = _run([1.0 if i % 2 == 0 else -1.0 for i in range(N)])

    assert smooth_reward > jittery_reward, (
        f"Expected smooth ({smooth_reward:.3f}) > jittery ({jittery_reward:.3f}). "
        "Smoothness penalty may not be applied."
    )


# ============================================================
# Test 6 — reset() clears prev_steering and prev_wall_hits
# ============================================================

def test_reset_clears_phase1_state():
    e = RacingEnv(render_mode=None, track_creator=_sprint_track)
    e.reset(seed=0)

    for _ in range(20):
        e.step(np.array([1.0, 1.0], dtype=np.float32))

    e.reset(seed=1)
    assert e.prev_steering == pytest.approx(0.0), \
        f"prev_steering not cleared on reset: {e.prev_steering}"
    assert e.prev_wall_hits == 0, \
        f"prev_wall_hits not cleared on reset: {e.prev_wall_hits}"
    e.close()


# ============================================================
# Test 7 — Off-track terminates the episode
# ============================================================

def test_off_track_terminates():
    """
    Teleport the car to a position 1000m away from the track center,
    then step once. The env should detect off-track and terminate.

    We teleport rather than drive there because walls (now correctly created
    by reset()) physically prevent the car from leaving the track boundary.
    """
    e = RacingEnv(render_mode=None, track_creator=_sprint_track, max_episode_steps=2000)
    e.reset(seed=0)

    # Move car far outside the track (well beyond half_width=7m)
    far_pos = e.track.centerline[0] + np.array([1000.0, 1000.0])
    e.car.body.position = (float(far_pos[0]), float(far_pos[1]))
    e.car.body.linearVelocity = (0.0, 0.0)

    _, _, terminated, truncated, info = e.step(np.array([0.0, 0.0], dtype=np.float32))

    e.close()
    assert terminated, "Episode should terminate when car is off-track"
    assert not info['on_track'], "on_track should be False when terminated"


# ============================================================
# Test 8 — Truncation at max_episode_steps
# ============================================================

def test_truncation_at_max_steps():
    max_steps = 10
    e = RacingEnv(render_mode=None, track_creator=_sprint_track, max_episode_steps=max_steps)
    e.reset(seed=0)
    truncated = False
    for _ in range(max_steps + 5):
        _, _, terminated, truncated, _ = e.step(np.array([0.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break
    e.close()
    assert truncated, "Episode should truncate at max_episode_steps"


# ============================================================
# Test 9 — Info dict contains expected keys
# ============================================================

def test_info_dict_keys(env):
    _, reset_info = env.reset(seed=0)
    required_reset = {'speed', 's', 'e_y', 'e_psi', 'steps', 'wall_hits', 'prev_steering'}
    missing = required_reset - set(reset_info.keys())
    assert not missing, f"reset() info missing keys: {missing}"

    _, _, _, _, step_info = env.step(np.array([0.5, 0.0], dtype=np.float32))
    required_step = required_reset | {'on_track', 'laps', 'total_progress'}
    missing_step = required_step - set(step_info.keys())
    assert not missing_step, f"step() info missing keys: {missing_step}"


# ============================================================
# Test 10 — gymnasium check_env passes (SB3 compatibility)
# ============================================================

def test_gymnasium_check_env():
    """
    gymnasium.utils.env_checker.check_env validates the full Gymnasium
    contract: space shapes, dtypes, reset/step return types, obs bounds.
    Passing this guarantees compatibility with Stable-Baselines3.
    """
    from gymnasium.utils.env_checker import check_env
    e = RacingEnv(render_mode=None, track_creator=_sprint_track, max_episode_steps=200)
    check_env(e, warn=True, skip_render_check=True)
    e.close()


# ============================================================
# Phase 3 reward-shaping regression tests
# ============================================================

def test_v2_idle_step_has_time_penalty(env):
    _, reward, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert info['reward_profile'] == 'v2'
    assert info['reward_terms']['time'] == pytest.approx(-0.001)
    assert reward == pytest.approx(sum(info['reward_terms'].values()))


def test_v2_does_not_reward_reverse_speed():
    v1 = RacingEnv(render_mode=None, track_creator=_sprint_track, reward_profile='v1')
    v2 = RacingEnv(render_mode=None, track_creator=_sprint_track, reward_profile='v2')
    v1.reset(seed=0)
    v2.reset(seed=0)

    for candidate in (v1, v2):
        candidate.car.body.linearVelocity = tuple(-20.0 * candidate.car.forward_vector)

    frenet = {'e_y': 0.0, 'e_psi': 0.0}
    common = dict(
        frenet=frenet,
        on_track=True,
        ds=-0.2,
        steering=0.0,
        prev_steering=0.0,
        new_wall_hits=0,
        touching_wall=False,
        backwards_terminated=False,
    )
    _, v1_terms = v1._compute_reward(**common)
    _, v2_terms = v2._compute_reward(**common)

    assert v1_terms['speed'] == pytest.approx(0.2)
    assert v2_terms['speed'] == pytest.approx(0.0)
    v1.close()
    v2.close()


def test_v2_wall_scraping_has_no_progress_or_speed_reward(env):
    env.car.body.linearVelocity = tuple(20.0 * env.car.forward_vector)
    _, terms = env._compute_reward(
        frenet={'e_y': 0.0, 'e_psi': 0.0},
        on_track=True,
        ds=0.3,
        steering=0.0,
        prev_steering=0.0,
        new_wall_hits=0,
        touching_wall=True,
        backwards_terminated=False,
    )
    assert terms['progress'] == pytest.approx(0.0)
    assert terms['speed'] == pytest.approx(0.0)
    assert terms['wall_contact'] == pytest.approx(-0.25)


def test_sustained_backwards_progress_terminates():
    config = replace(REWARD_PROFILES['v2'], max_backwards_steps=3)
    e = RacingEnv(
        render_mode=None,
        track_creator=_sprint_track,
        max_episode_steps=100,
        reward_config=config,
    )
    e.reset(seed=0)
    e._compute_progress_delta = lambda current_s: -1.0

    for _ in range(3):
        _, _, terminated, truncated, info = e.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )

    assert terminated
    assert not truncated
    assert info['termination_reason'] == 'driving_backwards'
    assert info['reward_terms']['backwards'] == pytest.approx(-25.0)
    e.close()


def test_unknown_reward_profile_is_rejected():
    with pytest.raises(ValueError, match='Unknown reward_profile'):
        RacingEnv(reward_profile='not-a-profile')
