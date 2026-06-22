"""Phase 4 procedural-track and domain-randomization tests."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from evaluate import parse_args as parse_evaluate_args
from src.env.racing_env import RacingEnv
from src.track.random_track import (
    HELD_OUT_RANDOM_TRACK_SEEDS,
    HELD_OUT_TRACK_IDS,
    MAX_ABS_CURVATURE,
    VALIDATION_RANDOM_TRACK_SEED,
    RandomTrackGenerator,
    create_held_out_track,
    create_random_track,
    create_validation_track,
)
from src.track.track import Track
from train import parse_args as parse_train_args


def test_random_track_is_reproducible_from_seed():
    first = create_random_track(np.random.default_rng(42), generation_seed=42)
    second = create_random_track(np.random.default_rng(42), generation_seed=42)

    np.testing.assert_allclose(first.centerline, second.centerline)
    assert first.width == pytest.approx(second.width)
    assert first.generation_parameters == second.generation_parameters


def test_different_seeds_produce_different_tracks():
    first = create_random_track(np.random.default_rng(1), generation_seed=1)
    second = create_random_track(np.random.default_rng(2), generation_seed=2)

    assert not np.allclose(first.centerline, second.centerline)


@pytest.mark.parametrize("seed", range(20))
def test_random_tracks_have_valid_geometry(seed):
    track = create_random_track(np.random.default_rng(seed), generation_seed=seed)
    inner, outer = track.get_boundary_points()

    assert Polygon(track.centerline).is_valid
    assert 18.0 <= track.width <= 28.0
    assert track.total_length > 0.0
    assert np.all(track.segment_lengths > 0.0)
    assert np.max(np.abs(track.curvature)) <= MAX_ABS_CURVATURE
    assert len(inner) >= 3
    assert len(outer) >= 3


def test_training_generator_skips_held_out_seed():
    allowed_seed = 424242

    class StubRng:
        def __init__(self):
            self.values = iter(
                [
                    HELD_OUT_RANDOM_TRACK_SEEDS[0],
                    VALIDATION_RANDOM_TRACK_SEED,
                    allowed_seed,
                ]
            )

        def integers(self, low, high):
            return next(self.values)

    track = RandomTrackGenerator()(StubRng())
    assert track.generation_seed == allowed_seed


def test_validation_track_is_reserved_and_stable():
    first = create_validation_track()
    second = create_validation_track()
    assert first.generation_seed == VALIDATION_RANDOM_TRACK_SEED
    assert first.generation_seed not in HELD_OUT_RANDOM_TRACK_SEEDS
    np.testing.assert_allclose(first.centerline, second.centerline)


def test_randomized_env_sequence_is_seeded_and_changes_each_reset():
    def track_sequence():
        env = RacingEnv(
            render_mode=None,
            track_generator=RandomTrackGenerator(num_points=120),
            max_episode_steps=10,
        )
        _, first_info = env.reset(seed=99)
        first_centerline = env.track.centerline.copy()
        _, second_info = env.reset()
        second_centerline = env.track.centerline.copy()
        env.close()
        return first_info, first_centerline, second_info, second_centerline

    sequence_a = track_sequence()
    sequence_b = track_sequence()

    assert sequence_a[0]["track_seed"] == sequence_b[0]["track_seed"]
    assert sequence_a[2]["track_seed"] == sequence_b[2]["track_seed"]
    assert sequence_a[0]["track_seed"] != sequence_a[2]["track_seed"]
    np.testing.assert_allclose(sequence_a[1], sequence_b[1])
    np.testing.assert_allclose(sequence_a[3], sequence_b[3])
    assert sequence_a[0]["track_name"].startswith("Procedural")
    assert sequence_a[0]["track_length"] > 0.0
    assert 18.0 <= sequence_a[0]["track_width"] <= 28.0


def test_track_creator_and_generator_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either track_creator or track_generator"):
        RacingEnv(
            track_creator=lambda: Track.create_sprint_track(),
            track_generator=RandomTrackGenerator(),
        )


def test_randomized_starts_are_reproducible_and_vary():
    def start_sequence():
        env = RacingEnv(
            track_creator=lambda: Track.create_sprint_track(track_width=14),
            randomize_start=True,
        )
        starts = []
        for seed in (123, None, None):
            _, info = env.reset(seed=seed)
            starts.append(
                (
                    info["start_s"],
                    info["start_lateral_offset"],
                    info["start_heading_offset"],
                )
            )
        env.close()
        return starts

    first = start_sequence()
    second = start_sequence()
    assert first == second
    assert len(set(first)) == len(first)


def test_nonzero_start_does_not_create_fake_progress():
    env = RacingEnv(
        track_creator=lambda: Track.create_sprint_track(track_width=14),
        max_episode_steps=10,
    )
    start_s = 0.4 * Track.create_sprint_track(track_width=14).total_length
    env.reset(seed=0, options={"start_s": start_s})
    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    env.close()

    assert abs(info["total_progress"]) < 0.1
    assert info["progress_fraction"] == pytest.approx(
        info["total_progress"] / info["track_length"]
    )


@pytest.mark.parametrize("track_id", HELD_OUT_TRACK_IDS)
def test_held_out_tracks_are_stable(track_id):
    first = create_held_out_track(track_id)
    second = create_held_out_track(track_id)
    np.testing.assert_allclose(first.centerline, second.centerline)
    assert first.width == pytest.approx(second.width)


def test_phase4_training_defaults():
    args = parse_train_args(["--track-mode", "random"])
    assert args.timesteps == 5_000_000
    assert args.run_name == "ppo_random_v2"


def test_held_out_evaluation_alias_expands_to_all_tracks():
    args = parse_evaluate_args(["model.zip", "--tracks", "held-out"])
    assert args.tracks == list(HELD_OUT_TRACK_IDS)
