"""Seeded procedural tracks for Phase 4 domain randomization."""

from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from src.track.track import Track


HELD_OUT_RANDOM_TRACK_SEEDS = (1001, 1002, 1003)
VALIDATION_RANDOM_TRACK_SEED = 2001
RESERVED_RANDOM_TRACK_SEEDS = (
    *HELD_OUT_RANDOM_TRACK_SEEDS,
    VALIDATION_RANDOM_TRACK_SEED,
)
HELD_OUT_TRACK_IDS = (
    "sprint",
    "grand-prix",
    *(f"procedural-{seed}" for seed in HELD_OUT_RANDOM_TRACK_SEEDS),
)
_MAX_GENERATION_SEED = np.iinfo(np.uint32).max
MAX_ABS_CURVATURE = 0.2


def _sample_centerline(rng, num_points):
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    base_radius = rng.uniform(150.0, 500.0)
    second_amplitude = rng.uniform(50.0, 200.0)
    middle_amplitude = rng.uniform(20.0, 100.0)
    high_amplitude = rng.uniform(10.0, 60.0)
    middle_harmonic = int(rng.choice([3, 4, 5]))
    high_harmonic = int(rng.choice([5, 7, 9]))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=3)
    width = float(rng.uniform(18.0, 28.0))

    radius = (
        base_radius
        + second_amplitude * np.cos(2 * angles + phases[0])
        + middle_amplitude * np.sin(middle_harmonic * angles + phases[1])
        + high_amplitude * np.cos(high_harmonic * angles + phases[2])
    )
    centerline = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    parameters = {
        "base_radius": float(base_radius),
        "second_amplitude": float(second_amplitude),
        "middle_amplitude": float(middle_amplitude),
        "middle_harmonic": middle_harmonic,
        "high_amplitude": float(high_amplitude),
        "high_harmonic": high_harmonic,
        "phases": phases.tolist(),
    }
    return centerline, radius, width, parameters


def _max_abs_curvature(centerline):
    segments = np.roll(centerline, -1, axis=0) - centerline
    segment_lengths = np.linalg.norm(segments, axis=1)
    if np.any(segment_lengths <= 1e-6):
        return np.inf

    headings = np.arctan2(segments[:, 1], segments[:, 0])
    heading_changes = headings - np.roll(headings, 1)
    heading_changes = (heading_changes + np.pi) % (2.0 * np.pi) - np.pi
    arc_lengths = (segment_lengths + np.roll(segment_lengths, 1)) / 2.0
    return float(np.max(np.abs(heading_changes / arc_lengths)))


def _is_valid_track(centerline, radius, width, max_abs_curvature):
    if not np.all(np.isfinite(centerline)) or np.min(radius) <= width:
        return False
    if _max_abs_curvature(centerline) > max_abs_curvature:
        return False

    polygon = Polygon(centerline)
    if not polygon.is_valid or polygon.is_empty or polygon.area <= 0.0:
        return False

    inner = polygon.buffer(-width / 2.0, quad_segs=8, join_style=1)
    return not inner.is_empty and not isinstance(inner, MultiPolygon)


def create_random_track(
    rng,
    *,
    generation_seed=None,
    num_points=300,
    max_attempts=50,
    max_abs_curvature=MAX_ABS_CURVATURE,
):
    """Generate a deterministic valid track from a NumPy random generator."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if num_points < 32:
        raise ValueError("num_points must be at least 32")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if max_abs_curvature <= 0.0:
        raise ValueError("max_abs_curvature must be positive")

    for attempt in range(1, max_attempts + 1):
        centerline, radius, width, parameters = _sample_centerline(rng, num_points)
        if not _is_valid_track(centerline, radius, width, max_abs_curvature):
            continue

        parameters["generation_attempt"] = attempt
        parameters["max_abs_curvature"] = _max_abs_curvature(centerline)
        return Track(
            centerline,
            width=width,
            name=(
                f"Procedural {generation_seed}"
                if generation_seed is not None
                else "Procedural"
            ),
            generation_seed=generation_seed,
            generation_parameters=parameters,
        )

    raise RuntimeError(f"Could not generate a valid track in {max_attempts} attempts")


@dataclass(frozen=True)
class RandomTrackGenerator:
    """Generate a training track while excluding validation/evaluation seeds."""

    excluded_seeds: tuple = RESERVED_RANDOM_TRACK_SEEDS
    num_points: int = 300
    max_attempts: int = 50
    max_abs_curvature: float = MAX_ABS_CURVATURE

    def __call__(self, rng):
        excluded = set(self.excluded_seeds)
        while True:
            generation_seed = int(rng.integers(0, _MAX_GENERATION_SEED))
            if generation_seed not in excluded:
                break
        return create_random_track(
            np.random.default_rng(generation_seed),
            generation_seed=generation_seed,
            num_points=self.num_points,
            max_attempts=self.max_attempts,
            max_abs_curvature=self.max_abs_curvature,
        )


def create_held_out_track(track_id):
    """Create one fixed evaluation track by its stable identifier."""
    if track_id == "sprint":
        return Track.create_sprint_track(track_width=14)
    if track_id == "grand-prix":
        return Track.create_complex_track(track_width=14)
    if track_id.startswith("procedural-"):
        try:
            seed = int(track_id.removeprefix("procedural-"))
        except ValueError as exc:
            raise ValueError(f"Invalid procedural track id: {track_id}") from exc
        if seed not in HELD_OUT_RANDOM_TRACK_SEEDS:
            raise ValueError(f"Procedural seed {seed} is not in the held-out set")
        return create_random_track(
            np.random.default_rng(seed), generation_seed=seed
        )
    raise ValueError(f"Unknown held-out track: {track_id}")


def create_validation_track():
    """Create the fixed model-selection track, separate from final evaluation."""
    return create_random_track(
        np.random.default_rng(VALIDATION_RANDOM_TRACK_SEED),
        generation_seed=VALIDATION_RANDOM_TRACK_SEED,
    )
