"""Pieces shared by the GPU trainers (gpu/ppo.py and gpu/rsl.py): run
folders and the per-rollout statistics logged to TensorBoard under the same
tags as the CPU (SB3) runs, so the dashboard compares all runs directly."""

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

from robot3d.runs import now_iso, write_run_info
from robot3d.walk import WalkConfig


def start_run(
    run_dir: Path,
    *,
    robot: str,
    trainer: str,
    total_steps: int,
    n_envs: int,
    seed: int,
    walk: WalkConfig,
    trainer_config: dict,
    device: str,
) -> dict:
    """Create the run folder and its run.json; returns run_info (kept up to date by the trainer)."""
    if run_dir.exists():
        raise FileExistsError(f"Run folder already exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    run_info = {
        "robot": robot,
        "task": "walk",
        "backend": "gpu",
        "trainer": trainer,
        "device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else device,
        "started": now_iso(),
        "command": " ".join(sys.argv),
        "total_steps": total_steps,
        "n_envs": n_envs,
        "seed": seed,
        "walk_config": walk.to_dict(),
        "ppo_config": trainer_config,
    }
    write_run_info(run_dir, run_info)
    return run_info


class RolloutLog:
    """Collects what happened during one rollout, from GpuWalkEnv.step results."""

    def __init__(self, control_dt: float):
        self.control_dt = control_dt
        self.recent_returns: deque[float] = deque(maxlen=100)  # like SB3's ep_info_buffer
        self.recent_lengths: deque[float] = deque(maxlen=100)
        self._reset()

    def _reset(self) -> None:
        self.term_sums: dict[str, float] = {}
        self.steps = 0
        self.distance: list[float] = []
        self.speed: list[float] = []
        self.fell: list[float] = []
        self.extra: dict[str, list[float]] = {}  # StepResult.stats, averaged over the rollout

    def record(self, result) -> None:
        self.steps += 1
        for k, v in result.terms.items():
            self.term_sums[k] = self.term_sums.get(k, 0.0) + float(v.mean())
        if result.done.any():
            self.recent_returns.extend(result.episode_return.tolist())
            self.recent_lengths.extend(result.episode_length.tolist())
            seconds = result.episode_length.float() * self.control_dt
            self.distance.extend(result.episode_distance.tolist())
            self.speed.extend((result.episode_distance / seconds).tolist())
            self.fell.extend(result.episode_fell.float().tolist())
        for name, value in (getattr(result, "stats", None) or {}).items():
            self.extra.setdefault(name, []).append(float(value))

    def scalars(self) -> dict[str, float]:
        """This rollout's rollout/, episode/ and reward/ scalars; starts a new rollout."""
        out = {f"reward/{k}": v / max(self.steps, 1) for k, v in self.term_sums.items()}
        if self.recent_returns:
            out["rollout/ep_rew_mean"] = float(np.mean(self.recent_returns))
            out["rollout/ep_len_mean"] = float(np.mean(self.recent_lengths))
        if self.distance:
            out["episode/distance_m"] = float(np.mean(self.distance))
            out["episode/speed_mps"] = float(np.mean(self.speed))
            out["episode/fell"] = float(np.mean(self.fell))
        out.update({name: float(np.mean(values)) for name, values in self.extra.items()})
        self._reset()
        return out
