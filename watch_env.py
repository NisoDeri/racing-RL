"""
Watch RacingEnv run with a random agent (rendered in real time).
Run: python watch_env.py
"""
import sys
import numpy as np
from src.env.racing_env import RacingEnv
from src.track.track import Track


def _sprint():
    return Track.create_sprint_track(track_width=14)


env = RacingEnv(render_mode="human", track_creator=_sprint, max_episode_steps=6000)
obs, info = env.reset(seed=0)

rng = np.random.default_rng(42)
episode = 1
step = 0

print("Random agent running — close the window to stop.")
print(f"Obs shape: {obs.shape}  |  obs[0:3] (first 3 rays): {obs[0:3].round(3)}")

while True:
    action = rng.uniform(-1, 1, size=2).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    if terminated or truncated:
        reason = "off-track" if terminated else "max steps"
        print(f"Episode {episode} ended ({reason}) — steps: {step}, "
              f"wall hits: {info['wall_hits']}")
        obs, info = env.reset()
        episode += 1
        step = 0
