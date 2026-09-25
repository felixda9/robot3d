"""PPO for thousands of robots at once, on the GPU (PyTorch).

The same algorithm as Stable-Baselines3's PPO (see training.py), shaped for
massively parallel environments the way legged_gym / RSL-RL do it:
  * many robots (4096) x few steps each (24) per rollout, instead of a few
    robots x many steps: 98k samples per update, collected in well under a
    second;
  * an adaptive learning rate (RSL-RL's schedule): after each minibatch,
    smaller if the policy moved too far (KL divergence), larger if too little;
  * clipped value loss, per-minibatch advantage normalization, gradient
    clipping, time-limit bootstrapping.

Logs the same TensorBoard tags as the SB3 trainer (rollout/, episode/,
reward/, train/, time/), so the dashboard compares CPU and GPU runs directly.

Imports only PyTorch (not Warp), so the web server can load its checkpoints.
"""

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from robot3d.gpu.common import RolloutLog, start_run
from robot3d.runs import RUNS_DIR, default_run_name, now_iso, write_run_info
from robot3d.walk import WalkConfig

CHECKPOINT_FORMAT = "robot3d-ppo-v1"


@dataclass
class GpuPPOConfig:
    num_envs: int = 4096  # robots simulated in parallel on the GPU
    steps_per_env: int = 24  # rollout length per robot (24 x 20 ms = 0.48 s)
    # 10 x 8 (up to 80 gradient steps per rollout, v4) instead of RSL-RL's
    # 5 x 4: with only 5 x 4, 30M steps gave ~5k gradient steps in total
    # (CPU runs: ~28k) and the policy stayed noisy.
    epochs: int = 10  # passes over each rollout
    minibatches: int = 8
    # Learning rate schedule.
    # "adaptive" (RSL-RL's, the default since stand12_calm): after each
    #   minibatch, /1.5 if the policy moved more than 2 x desired_kl (exact
    #   Gaussian KL), x1.5 if less than half, within [1e-5, 1e-2].
    # "linear_kl_stop" (walk runs up to walk12_tidy): falls linearly to 0,
    #   and an update stops once KL passes 1.5 x target_kl. It stalls when
    #   the exploration noise gets small (KL grows as change^2 / std^2):
    #   stand12_calm, at std 0.10, managed 1 of 80 planned steps per update
    #   from 21M steps on. (The first GPU runs' troubles with "adaptive" came
    #   from joint actor/critic gradient clipping, fixed since.)
    lr_schedule: str = "adaptive"
    learning_rate: float = 1e-3  # the starting rate
    desired_kl: float = 0.01  # adaptive
    target_kl: float = 0.02  # linear_kl_stop
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 1.0
    # Experimental, off: reward scaling as SB3's VecNormalize (running std of
    # discounted returns) and no value clipping (SB3's default). Hypothesis:
    # raw returns are ~100+, so value clipping at +-0.2 throttles the critic
    # and v1-v4 kept sliding back to "stand still" between good checkpoints.
    # But the one run with both (v5) got stuck standing for all 30M steps.
    # One seed proves little either way; test with several seeds.
    normalize_rewards: bool = False
    clip_value: bool = True
    # Exploration noise: start at 0.37 in action units (~0.18 rad, as the CPU
    # runs) with only a tiny entropy bonus. walk_gpu_30m used legged_gym's
    # 0.5 + 0.005: its noise stayed high (0.28 after 30M steps), most early
    # episodes ended in falls, and some snapshots only moved thanks to the
    # noise (their noise-free "mean" action stood still).
    entropy_coef: float = 0.001
    max_grad_norm: float = 1.0
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    init_std: float = 0.37
    clip_obs: float = 10.0  # normalized observations are clipped to +-this (as SB3's VecNormalize)


# ------------------------------------------------------------------ networks


def _mlp(sizes: list[int], out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        # Orthogonal init as in SB3: hidden layers gain sqrt(2); the actor's
        # output tiny (0.01) so a fresh policy outputs ~0 = "stand still".
        nn.init.orthogonal_(linear.weight, gain=out_gain if last else math.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if not last:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Actor: observation -> mean action (a Gaussian with learned, state-independent
    std around it). Critic: observation -> expected future reward (value)."""

    def __init__(self, num_obs: int, num_actions: int, hidden: list[int], init_std: float):
        super().__init__()
        self.actor = _mlp([num_obs, *hidden, num_actions], out_gain=0.01)
        self.critic = _mlp([num_obs, *hidden, 1], out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((num_actions,), math.log(init_std)))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class RewardScaler:
    """SB3 VecNormalize's reward normalization: divide each reward by the
    running std of the discounted return, so value targets stay ~O(1)
    whatever the reward's units. (Only for learning; logged episode returns
    stay raw.)"""

    def __init__(self, num_envs: int, gamma: float, device: torch.device, clip: float = 10.0):
        self.gamma = gamma
        self.clip = clip
        self.returns = torch.zeros(num_envs, device=device)
        self.stats = ObsNormalizer(1, clip=clip).to(device)  # running mean/var of a 1-D quantity

    def __call__(self, reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        self.returns = self.returns * self.gamma + reward
        self.stats.update(self.returns.unsqueeze(1))
        scaled = (reward / torch.sqrt(self.stats.var[0] + 1e-8)).clamp(-self.clip, self.clip)
        self.returns[done] = 0.0
        return scaled


class ObsNormalizer(nn.Module):
    """Running mean/variance of observations (like SB3's VecNormalize for
    observations): the policy sees (obs - mean) / std, clipped."""

    def __init__(self, size: int, clip: float):
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        # Parallel (Chan et al.) update of mean/variance with a whole batch.
        n = batch.shape[0]
        batch_mean = batch.mean(0)
        batch_var = batch.var(0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta**2 * self.count * n / total) / total
        self.count = total

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return ((obs - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


# ---------------------------------------------------------------- inference


class TorchPolicy:
    """A trained GPU-PPO checkpoint for inference on the CPU (web viewer,
    evaluation): normalized observation -> deterministic (mean) action."""

    def __init__(self, path: Path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self._jit = None
        if saved.get("format") == "robot3d-jit-v1":
            # An RSL-RL actor exported as TorchScript (gpu/rsl.py): it
            # normalizes internally and returns the noise-free action.
            import io

            self.num_obs = int(saved["num_obs"])
            self._jit = torch.jit.load(io.BytesIO(saved["jit"]), map_location="cpu")
            self._jit.eval()
            return
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
        self.num_obs = int(saved["num_obs"])
        self.net = ActorCritic(self.num_obs, int(saved["num_actions"]), list(saved["net_arch"]), 1.0)
        self.net.load_state_dict(saved["model"])
        self.normalizer = ObsNormalizer(self.num_obs, float(saved["clip_obs"]))
        self.normalizer.load_state_dict(saved["normalizer"])
        self.net.eval()

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        if self._jit is not None:
            return self._jit(obs)[0].numpy().astype(np.float64)
        return self.net.actor(self.normalizer(obs))[0].numpy().astype(np.float64)


# ------------------------------------------------------------------ training


def train_gpu(
    robot: str = "quadruped",
    total_steps: int = 50_000_000,
    name: str | None = None,
    seed: int = 0,
    checkpoint_every: int = 5_000_000,
    walk: WalkConfig | None = None,
    ppo: GpuPPOConfig | None = None,
    runs_dir: Path = RUNS_DIR,
    device: str = "cuda:0",
    log=print,
    init_from: Path | None = None,
) -> Path:
    """Train a walking policy on the GPU. Returns the run folder. Ctrl+C
    stops early and still saves a final checkpoint.

    init_from: a checkpoint (.pt of this trainer) to start from instead of a
    fresh network: fine-tuning, e.g. with harder shoves or new reward terms.
    Same robot and observation size; its network size and std are kept."""
    from torch.utils.tensorboard import SummaryWriter

    from robot3d.gpu.env import GpuWalkEnv  # Warp: only needed for training

    walk = walk or WalkConfig()
    ppo = ppo or GpuPPOConfig()
    torch.manual_seed(seed)
    name = name or default_run_name(robot, "walk_gpu")
    run_dir = runs_dir / name
    run_info = start_run(
        run_dir, robot=robot, trainer="robot3d-ppo", total_steps=total_steps, n_envs=ppo.num_envs,
        seed=seed, walk=walk, trainer_config=asdict(ppo), device=device,
    )

    env = GpuWalkEnv(ppo.num_envs, robot, walk, device=device, seed=seed)
    dev = env.device
    T, N = ppo.steps_per_env, ppo.num_envs
    net = ActorCritic(env.num_obs, env.num_actions, ppo.net_arch, ppo.init_std).to(dev)
    normalizer = ObsNormalizer(env.num_obs, ppo.clip_obs).to(dev)
    if init_from is not None:
        saved = torch.load(init_from, map_location=dev, weights_only=False)
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{init_from}: not a {CHECKPOINT_FORMAT} checkpoint (can only fine-tune our PPO's)")
        if int(saved["num_obs"]) != env.num_obs or int(saved["num_actions"]) != env.num_actions:
            raise ValueError(f"{init_from} has {saved['num_obs']} observations / {saved['num_actions']} actions; "
                             f"this robot and task have {env.num_obs} / {env.num_actions}")
        if list(saved["net_arch"]) != list(ppo.net_arch):
            raise ValueError(f"{init_from} has network {saved['net_arch']}, the config {ppo.net_arch}")
        net.load_state_dict(saved["model"])
        normalizer.load_state_dict(saved["normalizer"])
        run_info["init_from"] = str(init_from)
        write_run_info(run_dir, run_info)
    reward_scaler = RewardScaler(N, ppo.gamma, dev) if ppo.normalize_rewards else None
    optimizer = torch.optim.Adam(net.parameters(), lr=ppo.learning_rate)
    lr = ppo.learning_rate
    writer = SummaryWriter(str(run_dir / "tb" / "ppo_1"))

    def save(steps: int) -> Path:
        path = run_dir / "checkpoints" / f"step_{steps:09d}.pt"
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "steps": steps,
                "num_obs": env.num_obs,
                "num_actions": env.num_actions,
                "net_arch": list(ppo.net_arch),
                "clip_obs": ppo.clip_obs,
                "model": {k: v.cpu() for k, v in net.state_dict().items()},
                "normalizer": {k: v.cpu() for k, v in normalizer.state_dict().items()},
            },
            path,
        )
        return path

    # Rollout storage, all on the GPU: (T steps, N robots, ...).
    buf_obs = torch.zeros((T, N, env.num_obs), device=dev)
    buf_act = torch.zeros((T, N, env.num_actions), device=dev)
    buf_logp = torch.zeros((T, N), device=dev)
    buf_mean = torch.zeros((T, N, env.num_actions), device=dev)
    buf_val = torch.zeros((T, N), device=dev)
    buf_rew = torch.zeros((T, N), device=dev)
    buf_done = torch.zeros((T, N), device=dev)

    rollout_log = RolloutLog(env.task.control_dt)
    obs = env.reset()
    normalizer.update(obs)
    steps = 0
    next_checkpoint = checkpoint_every
    start = time.perf_counter()
    last_beat = 0.0
    interrupted = False
    iteration = 0
    try:
        while steps < total_steps:
            iteration += 1
            # ------------------------------------------------ collect a rollout
            with torch.no_grad():
                for t in range(T):
                    nobs = normalizer(obs)
                    dist = net.distribution(nobs)
                    action = dist.sample()
                    buf_obs[t] = nobs
                    buf_act[t] = action
                    buf_logp[t] = dist.log_prob(action).sum(-1)
                    buf_mean[t] = dist.mean
                    buf_val[t] = net.value(nobs)
                    result = env.step(action)
                    # Time-limit endings aren't failures: the robot would have
                    # kept earning reward, so add the value it had left.
                    reward = reward_scaler(result.reward, result.done) if reward_scaler else result.reward
                    buf_rew[t] = reward + ppo.gamma * buf_val[t] * result.time_out
                    buf_done[t] = result.done.float()
                    obs = result.obs
                    normalizer.update(obs)
                    rollout_log.record(result)
                last_value = net.value(normalizer(obs))

                # GAE: how much better than expected each action turned out.
                advantages = torch.zeros_like(buf_rew)
                running = torch.zeros(N, device=dev)
                for t in reversed(range(T)):
                    next_value = last_value if t == T - 1 else buf_val[t + 1]
                    not_done = 1.0 - buf_done[t]
                    delta = buf_rew[t] + ppo.gamma * next_value * not_done - buf_val[t]
                    running = delta + ppo.gamma * ppo.gae_lambda * not_done * running
                    advantages[t] = running
                returns = advantages + buf_val

            steps += T * N

            # ---------------------------------------------------------- update
            flat = {
                "obs": buf_obs.reshape(T * N, -1),
                "act": buf_act.reshape(T * N, -1),
                "logp": buf_logp.reshape(-1),
                "mean": buf_mean.reshape(T * N, -1),
                "val": buf_val.reshape(-1),
                "adv": advantages.reshape(-1),
                "ret": returns.reshape(-1),
            }
            old_std = net.log_std.exp().detach()
            batch = T * N // ppo.minibatches
            if ppo.lr_schedule == "linear_kl_stop":
                lr = ppo.learning_rate * max(0.0, 1.0 - steps / total_steps)  # linear decay
            for group in optimizer.param_groups:
                group["lr"] = lr
            stats = {"kl": [], "approx_kl": [], "pg": [], "v": [], "ent": [], "clip": []}
            stopped_early = False
            for _ in range(ppo.epochs):
                perm = torch.randperm(T * N, device=dev)
                for i in range(ppo.minibatches):
                    idx = perm[i * batch : (i + 1) * batch]
                    mb = {k: v[idx] for k, v in flat.items()}
                    dist = net.distribution(mb["obs"])
                    logp = dist.log_prob(mb["act"]).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()
                    value = net.value(mb["obs"])

                    # How far has the policy already moved from the one that
                    # collected this rollout? (Exact KL of the two Gaussians.)
                    # Past 1.5 x target: stop this update here.
                    with torch.no_grad():
                        std = dist.stddev
                        kl = (
                            torch.log(std / old_std)
                            + (old_std**2 + (mb["mean"] - dist.mean) ** 2) / (2 * std**2)
                            - 0.5
                        ).sum(-1).mean()
                    if ppo.lr_schedule == "adaptive":
                        if kl > 2.0 * ppo.desired_kl:
                            lr = max(1e-5, lr / 1.5)
                        elif kl < 0.5 * ppo.desired_kl:
                            lr = min(1e-2, lr * 1.5)
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                    elif kl > 1.5 * ppo.target_kl:
                        stopped_early = True
                        break

                    adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
                    ratio = torch.exp(logp - mb["logp"])
                    pg_loss = -torch.min(
                        adv * ratio, adv * ratio.clamp(1 - ppo.clip_range, 1 + ppo.clip_range)
                    ).mean()
                    if ppo.clip_value:
                        v_clipped = mb["val"] + (value - mb["val"]).clamp(-ppo.clip_range, ppo.clip_range)
                        v_loss = torch.max((value - mb["ret"]) ** 2, (v_clipped - mb["ret"]) ** 2).mean()
                    else:
                        v_loss = ((value - mb["ret"]) ** 2).mean()
                    loss = pg_loss + ppo.value_coef * v_loss - ppo.entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    # Clip the policy's and the critic's gradients separately
                    # (as RSL-RL does). Clipped together, the critic's large
                    # gradients (value errors are in reward units, ~100)
                    # dominate the shared norm, so every policy update got
                    # shrunk by an arbitrary, fluctuating factor.
                    nn.utils.clip_grad_norm_([*net.actor.parameters(), net.log_std], ppo.max_grad_norm)
                    nn.utils.clip_grad_norm_(net.critic.parameters(), ppo.max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        log_ratio = logp - mb["logp"]
                        stats["kl"].append(float(kl))
                        stats["approx_kl"].append(float(((ratio - 1) - log_ratio).mean()))  # SB3's estimator
                        stats["pg"].append(float(pg_loss))
                        stats["v"].append(float(v_loss))
                        stats["ent"].append(float(entropy))
                        stats["clip"].append(float(((ratio - 1).abs() > ppo.clip_range).float().mean()))
                if stopped_early:
                    break

            # --------------------------------------------------------- logging
            elapsed = time.perf_counter() - start
            with torch.no_grad():
                explained_var = 1 - torch.var(flat["ret"] - flat["val"]) / (torch.var(flat["ret"]) + 1e-8)
            scalars = {
                "time/fps": steps / elapsed,
                "train/approx_kl": np.mean(stats["approx_kl"]),
                "train/kl": np.mean(stats["kl"]),
                "train/learning_rate": lr,
                "train/policy_gradient_loss": np.mean(stats["pg"]),
                "train/value_loss": np.mean(stats["v"]),
                "train/entropy_loss": -np.mean(stats["ent"]),
                "train/clip_fraction": np.mean(stats["clip"]),
                # Share of the planned minibatch steps done before the KL stop.
                "train/update_fraction": len(stats["pg"]) / (ppo.epochs * ppo.minibatches),
                "train/explained_variance": float(explained_var),
                "train/std": float(net.log_std.detach().exp().mean()),
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
                    f"falls {100 * scalars.get('episode/fell', 0):4.1f}% | KL {scalars['train/kl']:.4f} | lr {lr:.1e}"
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
            finished=now_iso(),
            steps_done=steps,
            interrupted=interrupted,
            final_checkpoint=str(final.relative_to(run_dir)),
        )
        write_run_info(run_dir, run_info)
    return run_dir
