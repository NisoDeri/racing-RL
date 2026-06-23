"""
Watch a trained model drive in real time.

Run after training:
    python watch_trained.py
    python watch_trained.py models/ppo_sprint_500000_steps.zip   # specific checkpoint
"""
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from src.env.racing_env import RacingEnv
from src.track.track import Track


MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "models/best_model.zip"
N_STACK = 4


def make_env():
    env = RacingEnv(
        render_mode="human",
        track_creator=lambda: Track.create_sprint_track(track_width=14),
        max_episode_steps=6000,
    )
    return env


print(f"Loading model: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=N_STACK)

obs = env.reset()
episode = 1

print("Watching trained agent — close the window to stop.")

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    if done[0]:
        hits = info[0].get("wall_hits", 0)
        laps = info[0].get("laps", 0)
        print(f"Episode {episode}: wall_hits={hits}, laps={laps}")
        episode += 1
        obs = env.reset()
