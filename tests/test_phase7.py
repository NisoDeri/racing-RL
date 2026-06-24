"""Phase 7 training and ablation command tests."""

import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from evaluate import parse_args as parse_evaluate_args
from phase7_ablation import build_train_command, parse_args as parse_phase7_args
from src.env.racing_env import RacingEnv
from src.rl.auxiliary import AuxRaycastActorCriticPolicy
from train import parse_args as parse_train_args


def test_train_accepts_gae_lambda_and_reward_normalization():
    args = parse_train_args([
        "--track-mode",
        "random",
        "--gae-lambda",
        "0.9",
        "--vec-normalize-reward",
    ])

    assert args.gae_lambda == pytest.approx(0.9)
    assert args.vec_normalize_reward is True
    assert args.run_name == "ppo_random_v2"


def test_train_rejects_invalid_gae_lambda():
    with pytest.raises(SystemExit):
        parse_train_args(["--gae-lambda", "1.5"])


def test_train_accepts_auxiliary_raycast_prediction_args():
    args = parse_train_args([
        "--aux-raycast-prediction",
        "--aux-loss-coef",
        "0.02",
        "--aux-batch-size",
        "32",
        "--aux-gradient-steps",
        "2",
    ])

    assert args.aux_raycast_prediction is True
    assert args.aux_loss_coef == pytest.approx(0.02)
    assert args.aux_batch_size == 32
    assert args.aux_gradient_steps == 2


def test_evaluate_accepts_vec_normalize_stats_path():
    args = parse_evaluate_args([
        "model.zip",
        "--vec-normalize",
        "models/phase7/run_vecnormalize.pkl",
    ])

    assert args.vec_normalize == "models/phase7/run_vecnormalize.pkl"


def test_phase7_ablation_builds_reproducible_commands():
    args = parse_phase7_args([
        "--python",
        ".venv/bin/python",
        "--gae-values",
        "0.0",
        "0.95",
        "--seeds",
        "42",
        "--timesteps",
        "1000",
        "--n-envs",
        "2",
        "--vec-normalize-reward",
    ])

    first = build_train_command(args, seed=42, gae_lambda=0.0)
    second = build_train_command(args, seed=42, gae_lambda=0.95)

    assert "--gae-lambda" in first
    assert first[first.index("--gae-lambda") + 1] == "0.0"
    assert "--vec-normalize-reward" in first
    assert "phase7_gae0_random_v2_seed42_normrew" in first
    assert second[second.index("--gae-lambda") + 1] == "0.95"
    assert "phase7_gae0p95_random_v2_seed42_normrew" in second


def test_phase7_ablation_includes_auxiliary_flags():
    args = parse_phase7_args([
        "--gae-values",
        "0.95",
        "--seeds",
        "42",
        "--timesteps",
        "1000",
        "--aux-raycast-prediction",
        "--aux-loss-coef",
        "0.03",
        "--aux-batch-size",
        "16",
        "--aux-gradient-steps",
        "2",
    ])

    cmd = build_train_command(args, seed=42, gae_lambda=0.95)

    assert "--aux-raycast-prediction" in cmd
    assert cmd[cmd.index("--aux-loss-coef") + 1] == "0.03"
    assert cmd[cmd.index("--aux-batch-size") + 1] == "16"
    assert cmd[cmd.index("--aux-gradient-steps") + 1] == "2"
    assert "phase7_gae0p95_random_v2_seed42_auxray" in cmd


def test_auxiliary_policy_predicts_next_raycast_shape():
    env = DummyVecEnv([lambda: RacingEnv(max_episode_steps=5)])
    env = VecFrameStack(env, n_stack=4)
    try:
        model = PPO(
            AuxRaycastActorCriticPolicy,
            env,
            policy_kwargs=dict(net_arch=[16]),
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            seed=0,
            device="cpu",
        )
        obs = env.reset()
        obs_tensor = th.as_tensor(obs, device=model.device).float()
        actions = th.zeros((1, 2), device=model.device)

        with th.no_grad():
            prediction = model.policy.predict_next_rays(obs_tensor, actions)

        assert prediction.shape == (1, 30)
        assert th.all((0.0 <= prediction) & (prediction <= 1.0))
    finally:
        env.close()
