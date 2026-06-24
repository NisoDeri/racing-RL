"""Phase 8 SAC comparison tooling tests."""

import pytest
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from evaluate import evaluate, parse_args as parse_evaluate_args
from phase8_compare import (
    build_manifest,
    build_sac_train_command,
    parse_args as parse_phase8_args,
)
from src.env.racing_env import RacingEnv
from train_sac import parse_args as parse_sac_args


def test_train_sac_defaults_to_phase8_random_tracks():
    args = parse_sac_args(["--seed", "43"])

    assert args.track_mode == "random"
    assert args.timesteps == 5_000_000
    assert args.gradient_steps == -1
    assert args.run_name == "sac_random_v2"
    assert args.seed == 43


def test_train_sac_rejects_invalid_gradient_steps():
    with pytest.raises(SystemExit):
        parse_sac_args(["--gradient-steps", "0"])


def test_evaluate_accepts_sac_algorithm():
    args = parse_evaluate_args(["model.zip", "--algo", "sac"])

    assert args.algo == "sac"


def test_phase8_compare_builds_sac_training_commands():
    args = parse_phase8_args([
        "--python",
        ".venv/bin/python",
        "--seeds",
        "42",
        "--timesteps",
        "1000",
        "--n-envs",
        "2",
        "--buffer-size",
        "5000",
        "--learning-starts",
        "100",
        "--batch-size",
        "32",
        "--gradient-steps",
        "2",
        "--vec-normalize-reward",
    ])

    cmd = build_sac_train_command(args, seed=42)

    assert cmd[:2] == [".venv/bin/python", "train_sac.py"]
    assert cmd[cmd.index("--timesteps") + 1] == "1000"
    assert cmd[cmd.index("--gradient-steps") + 1] == "2"
    assert "--vec-normalize-reward" in cmd
    assert "phase8_sac_random_v2_seed42_normrew" in cmd


def test_phase8_manifest_includes_sac_and_ppo_eval_commands():
    args = parse_phase8_args([
        "--seeds",
        "42",
        "--timesteps",
        "1000",
        "--heldout-episodes",
        "3",
    ])

    manifest = build_manifest(args)

    assert manifest["phase"] == "8"
    assert len(manifest["sac_train_commands"]) == 1
    assert len(manifest["sac_eval_commands"]) == 1
    assert len(manifest["ppo_eval_commands"]) == 1
    assert "--algo" in manifest["sac_eval_commands"][0]
    assert "sac" in manifest["sac_eval_commands"][0]
    assert "ppo" in manifest["ppo_eval_commands"][0]


def test_evaluate_loads_sac_model(tmp_path):
    env = DummyVecEnv([lambda: RacingEnv(max_episode_steps=3)])
    env = VecFrameStack(env, n_stack=4)
    try:
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[16]),
            buffer_size=100,
            learning_starts=0,
            batch_size=4,
            gradient_steps=1,
            seed=0,
            device="cpu",
        )
        model_path = tmp_path / "sac_model.zip"
        model.save(model_path)
    finally:
        env.close()

    summary = evaluate(
        model_path=model_path,
        reward_profile="v2",
        track_ids=["sprint"],
        episodes=1,
        seed=0,
        max_episode_steps=3,
        algo="sac",
    )

    assert summary["algorithm"] == "sac"
    assert summary["aggregate"]["episodes"] == 1
