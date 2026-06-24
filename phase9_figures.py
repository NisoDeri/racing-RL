"""Phase 9 report figure generators.

Turns the Phase 9 evaluation artifacts (``evaluate.py`` JSON summaries,
``*.trajectories.json`` sidecars, and ``EvalCallback`` ``evaluations.npz`` logs)
into the figures the report needs:

    trajectory   — agent path(s) overlaid on a held-out track
    success      — zero-shot lap-success bar chart across model groups
    curves       — learning curves (eval return vs timesteps) with seed bands
    raycast      — heatmap of mean ray distance vs absolute track curvature

Each figure has a pure ``plot_*`` function that takes in-memory data and writes a
PNG, plus a thin CLI subcommand that loads the data from disk. Matplotlib runs on
the headless ``Agg`` backend so this works without a display.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.track.random_track import HELD_OUT_TRACK_IDS, create_held_out_track  # noqa: E402


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _save(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _closed_loop(points):
    """Append the first point so a polyline renders as a closed loop."""
    points = np.asarray(points, dtype=np.float64)
    return np.vstack([points, points[:1]])


# --- Trajectory overlay -----------------------------------------------------

def plot_trajectory_overlay(track_id, paths, out_path, title=None):
    """Overlay one or more (x, y) paths on a held-out track."""
    track = create_held_out_track(track_id)
    inner, outer = track.get_boundary_points()

    fig, ax = plt.subplots(figsize=(8, 8))
    for boundary in (inner, outer):
        loop = _closed_loop(boundary)
        ax.plot(loop[:, 0], loop[:, 1], color="0.4", linewidth=1.0)
    centerline = _closed_loop(track.centerline)
    ax.plot(
        centerline[:, 0], centerline[:, 1],
        color="0.75", linestyle="--", linewidth=0.8,
    )
    for index, path in enumerate(paths):
        path = np.asarray(path, dtype=np.float64)
        if path.size == 0:
            continue
        ax.plot(path[:, 0], path[:, 1], linewidth=1.6, label=f"episode {index + 1}")
        ax.plot(path[0, 0], path[0, 1], marker="o", color="green", markersize=6)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"Trajectory overlay — {track_id}")
    if paths:
        ax.legend(loc="upper right", fontsize="small")
    return _save(fig, out_path)


# --- Zero-shot success bar chart --------------------------------------------

def aggregate_success(result_paths):
    """Mean and seed-std of held-out success rate across result JSONs."""
    rates = [_load_json(path)["aggregate"]["success_rate"] for path in result_paths]
    rates = np.asarray(rates, dtype=np.float64)
    return float(rates.mean()), float(rates.std()), int(rates.size)


def plot_success_bar(groups, out_path, title="Zero-shot lap success"):
    """Bar chart of success rate per model group.

    ``groups`` maps a label to ``(mean, std)`` in [0, 1].
    """
    labels = list(groups)
    means = [groups[label][0] for label in labels]
    stds = [groups[label][1] for label in labels]

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 5))
    positions = np.arange(len(labels))
    ax.bar(positions, means, yerr=stds, capsize=6, color="#3b7dd8")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Held-out lap success rate")
    ax.set_title(title)
    for position, mean in zip(positions, means):
        ax.text(position, min(mean + 0.03, 1.02), f"{mean:.0%}", ha="center")
    return _save(fig, out_path)


# --- Learning curves ---------------------------------------------------------

def load_curve(npz_path):
    """Return (timesteps, mean-return-per-eval) from an evaluations.npz."""
    data = np.load(npz_path)
    return np.asarray(data["timesteps"]), np.asarray(data["results"]).mean(axis=1)

def aggregate_curves(npz_paths):
    """Mean and std across seeds of the eval-return curve.

    Curves are truncated to their shortest common length so seeds logged for
    slightly different totals still align on a shared timestep axis.
    """
    curves = [load_curve(path) for path in npz_paths]
    length = min(len(timesteps) for timesteps, _ in curves)
    timesteps = curves[0][0][:length]
    stacked = np.vstack([returns[:length] for _, returns in curves])
    return timesteps, stacked.mean(axis=0), stacked.std(axis=0)


def plot_learning_curves(groups, out_path, title="Learning curves"):
    """Plot eval return vs timesteps with a shaded seed band per group.

    ``groups`` maps a label to a list of evaluations.npz paths (one per seed).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, paths in groups.items():
        timesteps, mean, std = aggregate_curves(paths)
        ax.plot(timesteps, mean, label=label)
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Eval mean episode return")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _save(fig, out_path)


# --- Raycast heatmap ---------------------------------------------------------

def collect_raycast_curvature(
    model_path, track_ids, episodes, max_episode_steps, seed,
    algo="ppo", vec_normalize_path=None,
):
    """Run a model and gather (|curvature|, mean ray distance) per step."""
    # Imported lazily so figure plotting works without loading SB3/the env.
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

    from evaluate import ALGORITHMS, N_STACK, _resolve_vec_normalize_path
    from src.env.racing_env import RacingEnv

    vec_normalize_path = _resolve_vec_normalize_path(vec_normalize_path)
    model = ALGORITHMS[algo].load(str(model_path), device="cpu")
    frame_stack = algo != "sac"
    kappa = []
    ray = []
    for index, track_id in enumerate(track_ids):
        env = DummyVecEnv([
            lambda tid=track_id: RacingEnv(
                render_mode=None,
                track_creator=lambda: create_held_out_track(tid),
                randomize_start=True,
                max_episode_steps=max_episode_steps,
            )
        ])
        if vec_normalize_path is not None:
            env = VecNormalize.load(str(vec_normalize_path), env)
            env.training = False
            env.norm_reward = False
        if frame_stack:
            env = VecFrameStack(env, n_stack=N_STACK)
        env.seed(seed + index)
        obs = env.reset()
        try:
            for _ in range(episodes):
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, dones, infos = env.step(action)
                    info = infos[0]
                    kappa.append(abs(info["kappa"]))
                    ray.append(info["ray_mean_distance"])
                    if dones[0]:
                        break
        finally:
            env.close()
    return np.asarray(kappa), np.asarray(ray)


def plot_raycast_heatmap(kappa, ray, out_path, bins=40, title="Raycast vs curvature"):
    """2D histogram of mean ray distance against absolute track curvature.

    A binned-mean line is overlaid so the trend is readable on top of density.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    ray = np.asarray(ray, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    hist = ax.hist2d(kappa, ray, bins=bins, cmap="viridis")
    fig.colorbar(hist[3], ax=ax, label="step count")

    edges = np.linspace(kappa.min(), kappa.max(), min(bins, 20) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned = np.full(centers.shape, np.nan)
    for i in range(len(centers)):
        mask = (kappa >= edges[i]) & (kappa <= edges[i + 1])
        if np.any(mask):
            binned[i] = ray[mask].mean()
    ax.plot(centers, binned, color="white", linewidth=2.0, label="binned mean")

    ax.set_xlabel("|track curvature|  (1/m)")
    ax.set_ylabel("mean ray distance (m)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return _save(fig, out_path)


# --- CLI ---------------------------------------------------------------------

def _parse_groups(group_args):
    """Turn ``--group LABEL path...`` occurrences into ``{label: [paths]}``."""
    groups = {}
    for tokens in group_args or []:
        if len(tokens) < 2:
            raise SystemExit("each --group needs a label followed by >=1 path")
        groups[tokens[0]] = tokens[1:]
    return groups


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="figure", required=True)

    p_traj = sub.add_parser("trajectory", help="Trajectory overlay on a track")
    p_traj.add_argument("trajectories", help="*.trajectories.json from evaluate.py")
    p_traj.add_argument("--track", required=True, choices=HELD_OUT_TRACK_IDS)
    p_traj.add_argument("--episodes", type=int, default=0,
                        help="Max episodes to draw (0 = all recorded).")
    p_traj.add_argument("--out", type=Path, required=True)

    p_succ = sub.add_parser("success", help="Zero-shot success bar chart")
    p_succ.add_argument("--group", nargs="+", action="append", metavar="LABEL PATH",
                        help="Label followed by one result JSON per seed. Repeatable.")
    p_succ.add_argument("--out", type=Path, required=True)

    p_curve = sub.add_parser("curves", help="Learning curves with seed bands")
    p_curve.add_argument("--group", nargs="+", action="append", metavar="LABEL NPZ",
                         help="Label followed by one evaluations.npz per seed.")
    p_curve.add_argument("--out", type=Path, required=True)

    p_ray = sub.add_parser("raycast", help="Ray distance vs curvature heatmap")
    p_ray.add_argument("model", help="Path to an SB3 model")
    p_ray.add_argument("--algo", default="ppo", choices=("ppo", "sac"))
    p_ray.add_argument("--tracks", nargs="+", default=list(HELD_OUT_TRACK_IDS),
                       choices=HELD_OUT_TRACK_IDS)
    p_ray.add_argument("--episodes", type=int, default=3)
    p_ray.add_argument("--max-episode-steps", type=int, default=6000)
    p_ray.add_argument("--seed", type=int, default=123)
    p_ray.add_argument("--bins", type=int, default=40)
    p_ray.add_argument(
        "--vec-normalize",
        default=None,
        help="Optional VecNormalize .pkl stats; required when the model was "
        "trained with --vec-normalize-reward.",
    )
    p_ray.add_argument("--out", type=Path, required=True)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.figure == "trajectory":
        data = _load_json(args.trajectories)
        episodes = data["tracks"][args.track]
        if args.episodes:
            episodes = episodes[: args.episodes]
        paths = [episode["path"] for episode in episodes]
        out = plot_trajectory_overlay(args.track, paths, args.out)

    elif args.figure == "success":
        raw = _parse_groups(args.group)
        groups = {
            label: aggregate_success(paths)[:2] for label, paths in raw.items()
        }
        out = plot_success_bar(groups, args.out)

    elif args.figure == "curves":
        groups = _parse_groups(args.group)
        out = plot_learning_curves(groups, args.out)

    elif args.figure == "raycast":
        kappa, ray = collect_raycast_curvature(
            args.model, args.tracks, args.episodes,
            args.max_episode_steps, args.seed, args.algo,
            vec_normalize_path=args.vec_normalize,
        )
        out = plot_raycast_heatmap(kappa, ray, args.out, bins=args.bins)

    print(f"Saved {args.figure} figure to {out}")
    return out


if __name__ == "__main__":
    main()
