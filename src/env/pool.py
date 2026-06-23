"""
Checkpoint pool for Phase 5e self-play.

The pool stores snapshots of the ego policy taken during training.
At each episode reset, opponents sample uniformly from the pool so
the ego trains against a range of its past skill levels.

The pool lives on the filesystem (a directory of .zip files).
Each subprocess created by SubprocVecEnv has its own CheckpointPool
instance with its own model cache — no IPC required.  New snapshots
written by the main-process PoolSnapshotCallback become visible to
subprocesses at the next reset() when sample() re-globs the directory.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

class CheckpointPool:
    """Filesystem-backed pool of PPO checkpoint snapshots.

    Args:
        pool_dir: Directory where snapshot .zip files are stored.
        max_size: Maximum number of snapshots to keep (oldest evicted).
    """

    MAX_SIZE = 10

    def __init__(self, pool_dir: str, max_size: int = MAX_SIZE) -> None:
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._cache: dict[str, object] = {}  # path str → loaded PPO model

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(self, model, step: int) -> None:
        """Save a snapshot; evict the oldest if pool exceeds max_size."""
        from stable_baselines3 import PPO  # local import — pool is optional dep

        path = self.pool_dir / f"snapshot_{step:010d}.zip"
        model.save(str(path))
        # Invalidate cache for this path if it was previously loaded.
        self._cache.pop(str(path), None)
        self._evict()

    def add_from_path(self, src_path: str, step: int = 0) -> None:
        """Copy an existing .zip into the pool (bootstrap use)."""
        dst = self.pool_dir / f"snapshot_{step:010d}.zip"
        shutil.copy2(src_path, str(dst))
        self._cache.pop(str(dst), None)
        self._evict()

    def _evict(self) -> None:
        paths = self._sorted_paths()
        while len(paths) > self.max_size:
            oldest = paths.pop(0)
            self._cache.pop(str(oldest), None)
            oldest.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def sample(self, n: int, device: str = "cpu") -> list:
        """Return n randomly sampled (and cached) PPO models."""
        paths = self._sorted_paths()
        if not paths:
            raise RuntimeError(
                f"Checkpoint pool at '{self.pool_dir}' is empty. "
                "Bootstrap it with add_from_path() before training."
            )
        chosen = random.choices(paths, k=n)
        return [self._load(p, device) for p in chosen]

    def _load(self, path: Path, device: str):
        from stable_baselines3 import PPO

        key = str(path)
        if key not in self._cache:
            self._cache[key] = PPO.load(str(path), device=device)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Introspection (used for TensorBoard logging)
    # ------------------------------------------------------------------

    def size(self) -> int:
        return len(self._sorted_paths())

    def snapshot_steps(self) -> list[int]:
        return [int(p.stem.split("_")[1]) for p in self._sorted_paths()]

    def _sorted_paths(self) -> list[Path]:
        return sorted(self.pool_dir.glob("snapshot_*.zip"))
