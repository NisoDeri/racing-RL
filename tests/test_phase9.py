"""Phase 9 evaluation-protocol and figure-generation tests."""

import json

import numpy as np
import pytest

import phase9_figures as figures
from evaluate import _lap_time_stats, evaluate, parse_args as parse_evaluate_args
from src.env.racing_env import RacingEnv


# --- Lap-time metric ---------------------------------------------------------

def test_env_lap_times_derive_from_step_marks():
    env = RacingEnv(max_episode_steps=10)
    try:
        env.reset(seed=0)
        # Two laps completed at steps 60 and 150 (one physics tick = 1/60 s).
        env.lap_step_marks = [60, 150]
        assert env._lap_times() == pytest.approx([1.0, 1.5])
        assert env._mean_lap_time() == pytest.approx(1.25)
    finally:
        env.close()


def test_env_mean_lap_time_none_without_laps():
    env = RacingEnv(max_episode_steps=10)
    try:
        env.reset(seed=0)
        assert env.lap_step_marks == []
        assert env._mean_lap_time() is None
    finally:
        env.close()


def test_lap_time_stats_pools_completed_laps():
    results = [
        {"lap_times": [10.0, 12.0]},
        {"lap_times": []},
        {"lap_times": [14.0]},
    ]
    stats = _lap_time_stats(results)
    assert stats["laps_timed"] == 3
    assert stats["mean"] == pytest.approx(12.0)
    assert stats["std"] == pytest.approx(np.std([10.0, 12.0, 14.0]))


def test_lap_time_stats_empty_is_none():
    stats = _lap_time_stats([{"lap_times": []}])
    assert stats == {"mean": None, "std": None, "laps_timed": 0}


def test_evaluate_accepts_record_trajectories_flag(tmp_path):
    out = tmp_path / "result.json"
    args = parse_evaluate_args(["model.zip", "--record-trajectories", "2", "--output", str(out)])
    assert args.record_trajectories == 2


def test_evaluate_rejects_negative_record_trajectories():
    with pytest.raises(SystemExit):
        parse_evaluate_args(["model.zip", "--record-trajectories", "-1"])


def test_evaluate_captures_lap_time_and_trajectories(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    env = DummyVecEnv([lambda: RacingEnv(max_episode_steps=4)])
    env = VecFrameStack(env, n_stack=4)
    try:
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
                    policy_kwargs=dict(net_arch=[16]), seed=0, device="cpu")
        model_path = tmp_path / "ppo.zip"
        model.save(model_path)
    finally:
        env.close()

    summary = evaluate(
        model_path=model_path,
        reward_profile="v2",
        track_ids=["sprint"],
        episodes=1,
        seed=0,
        max_episode_steps=4,
        algo="ppo",
        record_trajectories=1,
    )

    assert "lap_time" in summary["aggregate"]
    assert "laps_timed" in summary["aggregate"]["lap_time"]
    trajectories = summary["tracks"]["sprint"]["trajectories"]
    assert len(trajectories) == 1
    # 4 steps recorded, each a 2D point.
    assert len(trajectories[0]["path"]) == 4
    assert len(trajectories[0]["path"][0]) == 2


# --- Figures -----------------------------------------------------------------

def test_aggregate_success_reads_seed_jsons(tmp_path):
    paths = []
    for index, rate in enumerate([1.0, 0.8]):
        path = tmp_path / f"seed{index}.json"
        path.write_text(json.dumps({"aggregate": {"success_rate": rate}}))
        paths.append(path)
    mean, std, count = figures.aggregate_success(paths)
    assert mean == pytest.approx(0.9)
    assert std == pytest.approx(0.1)
    assert count == 2


def test_aggregate_curves_truncates_to_common_length(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    np.savez(a, timesteps=np.array([100, 200, 300]),
             results=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    np.savez(b, timesteps=np.array([100, 200]),
             results=np.array([[3.0, 3.0], [4.0, 4.0]]))
    timesteps, mean, std = figures.aggregate_curves([a, b])
    assert list(timesteps) == [100, 200]
    assert mean == pytest.approx([2.0, 3.0])
    assert std == pytest.approx([1.0, 1.0])


def test_plot_trajectory_overlay_writes_png(tmp_path):
    path = [[float(x), 0.0] for x in range(5)]
    out = figures.plot_trajectory_overlay("sprint", [path], tmp_path / "traj.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_success_bar_writes_png(tmp_path):
    out = figures.plot_success_bar(
        {"Phase 2": (0.0, 0.0), "Phase 4": (1.0, 0.0)}, tmp_path / "bar.png"
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_learning_curves_writes_png(tmp_path):
    a = tmp_path / "a.npz"
    np.savez(a, timesteps=np.array([100, 200]), results=np.array([[1.0], [2.0]]))
    out = figures.plot_learning_curves({"PPO": [a]}, tmp_path / "curve.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_raycast_heatmap_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    kappa = np.abs(rng.normal(size=200))
    ray = 50 - 20 * kappa + rng.normal(size=200)
    out = figures.plot_raycast_heatmap(kappa, ray, tmp_path / "heat.png", bins=10)
    assert out.exists() and out.stat().st_size > 0
