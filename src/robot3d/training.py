"""PPO training for the walk task: run folders, checkpoints, TensorBoard metrics.

A training run writes everything into runs/<name>/:
    run.json                          settings (robot, task, PPO) + progress
    tb/                               TensorBoard logs (uv run tensorboard --logdir runs)
    checkpoints/step_000500000.zip                  the policy after 500k steps
    checkpoints/step_000500000_vecnormalize.pkl     its observation normalizer

PPO (Proximal Policy Optimization) in one paragraph: run the current policy in
many environments for a while (a "rollout"), score how much better or worse
each action turned out than expected (the "advantage", estimated with a
learned value function), then nudge the policy toward the better actions,
but only a little per update ("proximal"), so learning doesn't lurch. Repeat.
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import psutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from robot3d.envs import WalkEnv
from robot3d.runs import RUNS_DIR
from robot3d.walk import WalkConfig


def default_run_name(robot: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}_{robot}_walk"


def default_num_envs() -> int:
    """Parallel environments (one process each): logical CPU threads - 2.

    Measured on an 8-core/16-thread i7-11700K: 8 envs 4.5k steps/s, 12 envs
    5.1k, 16 envs 5.6k. Hyperthreads help because each env process spends much
    of its time in Python overhead. Two threads stay free so the PC (and the
    web viewer) remain responsive.
    """
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 2
    return max(1, logical - 2)


@dataclass
class PPOConfig:
    n_steps: int = 1024  # steps per env per rollout (1024 x 20 ms = ~20 s, about one episode)
    minibatches: int = 4  # each rollout is split into this many batches per epoch
    n_epochs: int = 10  # passes over each rollout
    learning_rate: float = 3e-4
    gamma: float = 0.99  # discount: how much future reward counts (0.99 = ~100 steps = 2 s horizon)
    gae_lambda: float = 0.95  # advantage estimation smoothing (bias vs variance)
    clip_range: float = 0.2  # the "proximal" part: max policy change per update
    ent_coef: float = 0.0  # bonus for randomness (exploration); the policy's own noise suffices here
    net_arch: list[int] = field(default_factory=lambda: [256, 256])  # hidden layers (policy and value nets)
    log_std_init: float = -1.0  # initial exploration noise: std e^-1 = 0.37 in action units (~0.18 rad)


class CheckpointEvery(BaseCallback):
    """Save the policy + normalizer every `every` steps, and at the end."""

    def __init__(self, checkpoint_dir: Path, every: int):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.every = every
        self._next = every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self.save()
            self._next += self.every
        return True

    def save(self) -> Path:
        path = self.checkpoint_dir / f"step_{self.num_timesteps:09d}.zip"
        self.model.save(path)
        self.model.get_vec_normalize_env().save(str(path.with_name(path.stem + "_vecnormalize.pkl")))
        return path


class Heartbeat(BaseCallback):
    """Rewrite run.json with progress every `every` seconds. The dashboard
    tells a running run from a crashed one by how recently it changed."""

    def __init__(self, run_dir: Path, run_info: dict, every: float = 30.0):
        super().__init__()
        self.run_dir = run_dir
        self.run_info = run_info
        self.every = every
        self._last = 0.0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        now = time.monotonic()
        if now - self._last >= self.every:
            self._last = now
            self.run_info.update(steps_done=int(self.num_timesteps), updated=_now())
            try:
                _write_json(self.run_dir / "run.json", self.run_info)
            except OSError:
                pass  # never crash training over a progress note; next beat retries


class EpisodeStats(BaseCallback):
    """Log walk-specific numbers to TensorBoard next to SB3's own
    (rollout/ep_rew_mean etc.): per-episode distance, speed and falls, and
    the average of each reward term, which shows WHAT the policy is being
    rewarded or punished for."""

    def __init__(self, control_dt: float):
        super().__init__()
        self.control_dt = control_dt

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            for key, value in info.items():
                if key.startswith("reward_"):
                    self.logger.record_mean(f"reward/{key.removeprefix('reward_')}", value)
            if done and "episode" in info:  # "episode" is added by SB3's Monitor wrapper
                seconds = info["episode"]["l"] * self.control_dt
                self.logger.record_mean("episode/distance_m", info["distance"])
                self.logger.record_mean("episode/speed_mps", info["distance"] / seconds)
                self.logger.record_mean("episode/fell", float(info["fell"]))
        return True


def train(
    robot: str = "quadruped",
    total_steps: int = 10_000_000,
    n_envs: int | None = None,
    name: str | None = None,
    seed: int = 0,
    checkpoint_every: int = 500_000,
    walk: WalkConfig | None = None,
    ppo: PPOConfig | None = None,
    runs_dir: Path = RUNS_DIR,
    verbose: int = 1,
) -> Path:
    """Train a walking policy. Returns the run directory. Ctrl+C stops early
    and still saves a final checkpoint."""
    walk = walk or WalkConfig()
    ppo = ppo or PPOConfig()
    n_envs = n_envs or default_num_envs()
    name = name or default_run_name(robot)
    run_dir = runs_dir / name
    if run_dir.exists():
        raise FileExistsError(f"Run folder already exists: {run_dir}")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    run_info = {
        "robot": robot,
        "task": "walk",
        "started": _now(),
        "command": " ".join(sys.argv),
        "total_steps": total_steps,
        "n_envs": n_envs,
        "seed": seed,
        "walk_config": walk.to_dict(),
        "ppo_config": asdict(ppo),
    }
    _write_json(run_dir / "run.json", run_info)

    # Each env runs in its own process (SubprocVecEnv), so they step in parallel.
    # VecNormalize rescales observations (and rewards) to ~zero mean and unit
    # variance using running averages; neural networks learn much better on
    # well-scaled inputs. Its statistics are saved with every checkpoint,
    # because a policy only works with the same scaling it was trained with.
    envs = make_vec_env(
        WalkEnv, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv, env_kwargs={"robot": robot, "config": walk}
    )
    envs = VecNormalize(envs, gamma=ppo.gamma)
    model = PPO(
        "MlpPolicy",
        envs,
        n_steps=ppo.n_steps,
        batch_size=ppo.n_steps * n_envs // ppo.minibatches,
        n_epochs=ppo.n_epochs,
        learning_rate=ppo.learning_rate,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        clip_range=ppo.clip_range,
        ent_coef=ppo.ent_coef,
        policy_kwargs=dict(
            net_arch=dict(pi=ppo.net_arch, vf=ppo.net_arch),
            activation_fn=torch.nn.ELU,
            log_std_init=ppo.log_std_init,
        ),
        tensorboard_log=str(run_dir / "tb"),
        seed=seed,
        device="cpu",
        verbose=verbose,
    )
    control_dt = walk.control_dt
    checkpoints = CheckpointEvery(checkpoint_dir, checkpoint_every)
    interrupted = False
    callbacks = [checkpoints, EpisodeStats(control_dt), Heartbeat(run_dir, run_info)]
    try:
        model.learn(total_steps, callback=callbacks, tb_log_name="ppo")
    except KeyboardInterrupt:
        interrupted = True
    finally:
        final = checkpoints.save()
        envs.close()
        run_info.update(
            finished=_now(),
            steps_done=int(model.num_timesteps),
            interrupted=interrupted,
            final_checkpoint=str(final.relative_to(run_dir)),
        )
        _write_json(run_dir / "run.json", run_info)
    return run_dir


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    # Write a temp file, then swap it in, so a reader (the dashboard) never
    # sees a half-written run.json. On Windows the swap fails while another
    # process has the file open for that instant, so retry briefly.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    for _ in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.05)
    tmp.replace(path)  # last try: let the error surface
