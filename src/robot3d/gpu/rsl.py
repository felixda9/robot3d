"""Train on the GPU with RSL-RL's PPO, the reference implementation behind
legged_gym, Isaac Lab and mjlab, on our GpuWalkEnv.

We drive RSL-RL's algorithm (act / process_env_step / compute_returns /
update) from our own loop, so runs land in the same run folders, with the
same TensorBoard tags, as every other run, and the dashboard compares them
directly.

Settings follow legged_gym's standard locomotion recipe, including its
reward convention: every reward is multiplied by the control time step
(0.02 s), which keeps returns around 1-10, the scale RSL-RL's value clipping
(+-0.2) is made for. Episode returns in the logs stay unscaled.

Checkpoints: RSL-RL exports the actor (observation normalizer + network +
noise-free output) as one TorchScript module. We store it in step_N.pt
(format "robot3d-jit-v1"), which gpu/ppo.py's TorchPolicy plays on the CPU.
"""

import io
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from tensordict import TensorDict

from robot3d.gpu.common import RolloutLog, start_run
from robot3d.runs import RUNS_DIR, default_run_name, now_iso, write_run_info
from robot3d.walk import WalkConfig

JIT_CHECKPOINT_FORMAT = "robot3d-jit-v1"


@dataclass
class RslConfig:
    """legged_gym's standard PPO settings for locomotion."""

    num_envs: int = 4096
    steps_per_env: int = 24
    epochs: int = 5
    minibatches: int = 4
    learning_rate: float = 1e-3
    schedule: str = "adaptive"  # RSL-RL's KL-driven learning rate
    desired_kl: float = 0.01
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    actor_hidden: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden: list[int] = field(default_factory=lambda: [512, 256, 128])
    init_std: float = 1.0
    reward_scale: float = 0.02  # = control time step, as legged_gym


class _EnvInfo:
    """The two facts RSL-RL's algorithm needs to build itself."""

    def __init__(self, num_envs: int, num_actions: int):
        self.num_envs = num_envs
        self.num_actions = num_actions


def _rsl_cfg(cfg: RslConfig) -> dict:
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.modules import GaussianDistribution

    return {
        "num_steps_per_env": cfg.steps_per_env,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "multi_gpu": None,
        "actor": {
            "class_name": MLPModel,
            "hidden_dims": cfg.actor_hidden,
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": GaussianDistribution, "init_std": cfg.init_std, "std_type": "scalar"},
        },
        "critic": {
            "class_name": MLPModel,
            "hidden_dims": cfg.critic_hidden,
            "activation": "elu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": PPO,
            "num_learning_epochs": cfg.epochs,
            "num_mini_batches": cfg.minibatches,
            "clip_param": cfg.clip_param,
            "gamma": cfg.gamma,
            "lam": cfg.lam,
            "value_loss_coef": cfg.value_loss_coef,
            "use_clipped_value_loss": cfg.use_clipped_value_loss,
            "entropy_coef": cfg.entropy_coef,
            "learning_rate": cfg.learning_rate,
            "max_grad_norm": cfg.max_grad_norm,
            "schedule": cfg.schedule,
            "desired_kl": cfg.desired_kl,
        },
    }


def export_actor(alg, path: Path, steps: int, num_obs: int, num_actions: int) -> Path:
    module = torch.jit.script(alg.get_policy().as_jit().to("cpu"))
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    torch.save(
        {
            "format": JIT_CHECKPOINT_FORMAT,
            "steps": steps,
            "num_obs": num_obs,
            "num_actions": num_actions,
            "jit": buffer.getvalue(),
        },
        path,
    )
    return path


def train_rsl(
    robot: str = "quadruped",
    total_steps: int = 50_000_000,
    name: str | None = None,
    seed: int = 0,
    checkpoint_every: int = 5_000_000,
    walk: WalkConfig | None = None,
    rsl: RslConfig | None = None,
    runs_dir: Path = RUNS_DIR,
    device: str = "cuda:0",
    log=print,
) -> Path:
    """Train a walking policy with RSL-RL's PPO on the GPU. Returns the run folder."""
    from rsl_rl.algorithms import PPO
    from torch.utils.tensorboard import SummaryWriter

    from robot3d.gpu.env import GpuWalkEnv

    walk = walk or WalkConfig()
    rsl = rsl or RslConfig()
    torch.manual_seed(seed)
    run_dir = runs_dir / (name or default_run_name(robot, "walk_rsl"))
    run_info = start_run(
        run_dir, robot=robot, trainer="rsl-rl", total_steps=total_steps, n_envs=rsl.num_envs,
        seed=seed, walk=walk, trainer_config=asdict(rsl), device=device,
    )
    env = GpuWalkEnv(rsl.num_envs, robot, walk, device=device, seed=seed)
    N = rsl.num_envs

    def as_td(obs: torch.Tensor) -> TensorDict:
        return TensorDict({"policy": obs}, batch_size=[N], device=env.device)

    obs = env.reset()
    alg = PPO.construct_algorithm(as_td(obs), _EnvInfo(N, env.num_actions), _rsl_cfg(rsl), str(env.device))
    alg.train_mode()
    writer = SummaryWriter(str(run_dir / "tb" / "ppo_1"))
    rollout_log = RolloutLog(env.task.control_dt)

    def save(steps: int) -> Path:
        path = run_dir / "checkpoints" / f"step_{steps:09d}.pt"
        return export_actor(alg, path, steps, env.num_obs, env.num_actions)

    steps = 0
    next_checkpoint = checkpoint_every
    start = time.perf_counter()
    last_beat = 0.0
    interrupted = False
    iteration = 0
    try:
        while steps < total_steps:
            iteration += 1
            with torch.no_grad():
                for _ in range(rsl.steps_per_env):
                    actions = alg.act(as_td(obs))
                    result = env.step(actions)
                    obs = result.obs
                    alg.process_env_step(
                        as_td(obs), result.reward * rsl.reward_scale, result.done, {"time_outs": result.time_out}
                    )
                    rollout_log.record(result)
                alg.compute_returns(as_td(obs))
            losses = alg.update()
            steps += rsl.steps_per_env * N

            elapsed = time.perf_counter() - start
            scalars = {
                "time/fps": steps / elapsed,
                "train/learning_rate": alg.learning_rate,
                "train/value_loss": losses["value"],
                "train/policy_gradient_loss": losses["surrogate"],
                "train/entropy_loss": -losses["entropy"],
                "train/std": float(alg.get_policy().output_std.mean()),
                **rollout_log.scalars(),
            }
            for tag, value in scalars.items():
                writer.add_scalar(tag, float(value), steps)
            writer.flush()
            if iteration % 10 == 0 or steps >= total_steps:
                log(
                    f"{steps / 1e6:7.1f}M steps | {scalars['time/fps'] / 1e3:6.1f}k steps/s | "
                    f"reward {scalars.get('rollout/ep_rew_mean', float('nan')):7.1f} | "
                    f"speed {scalars.get('episode/speed_mps', float('nan')):5.2f} m/s | "
                    f"falls {100 * scalars.get('episode/fell', 0):4.1f}% | std {scalars['train/std']:.3f} | "
                    f"lr {alg.learning_rate:.1e}"
                )
            if steps >= next_checkpoint:
                save(steps)
                next_checkpoint += checkpoint_every
            if elapsed - last_beat > 30:
                last_beat = elapsed
                run_info.update(steps_done=steps, updated=now_iso())
                try:
                    write_run_info(run_dir, run_info)
                except OSError:
                    pass
    except KeyboardInterrupt:
        interrupted = True
    finally:
        final = save(steps)
        writer.close()
        run_info.update(
            finished=now_iso(), steps_done=steps, interrupted=interrupted,
            final_checkpoint=str(final.relative_to(run_dir)),
        )
        write_run_info(run_dir, run_info)
    return run_dir
