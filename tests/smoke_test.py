"""
Smoke test — quick manual sanity check, not part of the pytest suite.

Run with:
    python tests/smoke_test.py

Runs 200 steps with a random agent and prints key stats. No display required.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.env.racing_env import RacingEnv, _SPEED_NORM, _LAT_VEL_NORM
from src.track.track import Track


def _sprint_track():
    return Track.create_sprint_track(track_width=14)


def main():
    print("=" * 55)
    print("RacingEnv Phase 1 — Smoke Test (200 steps, random agent)")
    print("=" * 55)

    env = RacingEnv(render_mode=None, track_creator=_sprint_track, max_episode_steps=200)
    obs, info = env.reset(seed=42)

    print(f"\nObservation shape     : {obs.shape}")
    print(f"Observation space     : low={env.observation_space.low[:5]}... "
          f"high={env.observation_space.high[:5]}...")
    print(f"Action space          : {env.action_space}")
    print(f"\nNormalization constants:")
    print(f"  _SPEED_NORM         : {_SPEED_NORM} m/s")
    print(f"  _LAT_VEL_NORM       : {_LAT_VEL_NORM} m/s")
    print(f"\nInitial obs[0:5] (rays)    : {obs[0:5].round(3)}")
    print(f"Initial obs[30:34] (ego)   : {obs[30:34].round(3)}"
          "  ← [speed_norm, lat_vel_norm, throttle, steering]")

    rng = np.random.default_rng(seed=42)
    total_reward = 0.0
    wall_hits = 0
    steps_run = 0
    episodes = 0

    for step in range(200):
        action = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        wall_hits = info.get('wall_hits', 0)
        steps_run += 1

        if terminated or truncated:
            episodes += 1
            obs, info = env.reset(seed=step)

    print(f"\n--- After 200 steps ---")
    print(f"Episodes finished     : {episodes}")
    print(f"Steps run             : {steps_run}")
    print(f"Total reward          : {total_reward:.3f}")
    print(f"Wall hits (cumul.)    : {wall_hits}")
    print(f"Last obs[30:34]       : {obs[30:34].round(3)}")
    print(f"obs within space      : {env.observation_space.contains(obs)}")
    print("=" * 55)
    print("Smoke test PASSED (no exceptions raised)")
    env.close()


if __name__ == "__main__":
    main()
