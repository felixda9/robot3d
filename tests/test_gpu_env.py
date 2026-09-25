"""GPU environment (MuJoCo Warp) and GPU training, end to end.

Skipped without an NVIDIA GPU + CUDA build of PyTorch. The key check:
the same robot, started in the same state and given the same actions,
behaves the same in MuJoCo Warp (float32, GPU) as in regular MuJoCo
(float64, CPU): same observations and rewards within float32 tolerance.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU and CUDA PyTorch")


@pytest.fixture(scope="module")
def gpu_env():
    from robot3d.gpu.env import GpuWalkEnv

    from robot3d.walk import WalkConfig

    # No random shoves: the physics comparison needs identical inputs.
    env = GpuWalkEnv(num_envs=64, config=WalkConfig(push_interval=0.0), device="cuda:0", seed=0)
    env.reset()
    return env


def put_cpu_state_into_world(gpu_env, world, cpu_env):
    """Copy a CPU env's state into one GPU world (after the batch was reset)."""
    import mujoco_warp as mjw
    import warp as wp

    gpu_env.qpos[world] = torch.as_tensor(cpu_env.data.qpos, dtype=torch.float32, device=gpu_env.device)
    gpu_env.qvel[world] = torch.as_tensor(cpu_env.data.qvel, dtype=torch.float32, device=gpu_env.device)
    gpu_env.ctrl[world] = torch.as_tensor(cpu_env.data.ctrl, dtype=torch.float32, device=gpu_env.device)
    with wp.ScopedDevice(gpu_env.wp_device), wp.ScopedStream(wp.stream_from_torch(gpu_env.device)):
        mjw.kinematics(gpu_env.m, gpu_env.d)
    gpu_env.last_action[world] = 0.0
    gpu_env.air_time[world] = 0.0
    gpu_env.episode_length[world] = 0
    gpu_env.start_xy[world] = gpu_env.qpos[world, 0:2]
    xy, down = gpu_env.task.feet_state(gpu_env.geom_xpos)
    gpu_env._feet_before[0][world] = xy[world]
    gpu_env._feet_before[1][world] = down[world]


def test_gpu_physics_matches_cpu(gpu_env):
    from robot3d.envs import WalkEnv

    from robot3d.walk import WalkConfig

    cpu = WalkEnv(config=WalkConfig(push_interval=0.0))
    cpu_obs, _ = cpu.reset(seed=3)
    gpu_env.reset()
    put_cpu_state_into_world(gpu_env, 0, cpu)
    torch.testing.assert_close(gpu_env.observe()[0].cpu(), torch.tensor(cpu_obs), rtol=1e-4, atol=1e-4)

    rng = np.random.default_rng(0)
    for step in range(15):  # 0.3 s of walking-like random motion
        action = rng.uniform(-0.5, 0.5, cpu.task.num_actions)
        cpu_obs, cpu_reward, *_ , cpu_info = cpu.step(action)
        actions = torch.zeros((gpu_env.num_envs, gpu_env.num_actions), device=gpu_env.device)
        actions[0] = torch.as_tensor(action, dtype=torch.float32)
        result = gpu_env.step(actions)
        gpu_obs = result.obs[0].cpu().numpy()
        # float32 vs float64 and solver details: small differences that grow slowly.
        tol = 2e-3 * (1 + step)
        np.testing.assert_allclose(gpu_obs[:10], cpu_obs[:10], atol=tol, err_msg=f"torso state, step {step}")
        np.testing.assert_allclose(gpu_obs[10:18], cpu_obs[10:18], atol=tol, err_msg=f"joint angles, step {step}")
        assert result.reward[0].item() == pytest.approx(cpu_reward, abs=0.05 + 0.02 * step), f"reward, step {step}"


def test_fallen_robots_restart_standing(gpu_env):
    gpu_env.reset()
    gpu_env.qpos[5, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=gpu_env.device)  # upside down
    result = gpu_env.step(torch.zeros((gpu_env.num_envs, gpu_env.num_actions), device=gpu_env.device))
    assert result.done[5] and not result.time_out[5]
    assert result.episode_fell.tolist() == [True] * int(result.done.sum())
    assert gpu_env.episode_length[5] == 0  # restarted
    assert gpu_env.qpos[5, 2].item() == pytest.approx(gpu_env.task.standing_height, abs=0.02)
    assert result.obs[5, 0].item() == pytest.approx(gpu_env.task.standing_height, abs=0.02)  # its new observation


def test_random_shoves_on_the_gpu():
    from robot3d.gpu.env import GpuWalkEnv
    from robot3d.walk import WalkConfig

    env = GpuWalkEnv(num_envs=256, config=WalkConfig(push_interval=1.0), device="cuda:0", seed=1)
    env.reset(randomize_episode_start=False)
    still = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    jumps = torch.zeros(env.num_envs, device=env.device)
    last = env.qvel[:, :2].clone()
    for _ in range(round(3.0 / env.task.control_dt)):  # 3 s: 2-6 shoves each
        env.step(still)
        jumps += ((env.qvel[:, :2] - last).norm(dim=1) > 0.3).float()
        last = env.qvel[:, :2].clone()
    assert 1.5 < jumps.mean().item() < 6.5
    assert (env.next_push >= env.episode_length).all()  # every robot has its next shove scheduled (or due next step)


def test_getup_episodes_end_when_standing_steady_on_the_gpu():
    from robot3d.gpu.env import GpuWalkEnv
    from robot3d.walk import WalkConfig

    env = GpuWalkEnv(num_envs=64, robot="quadruped12", config=WalkConfig.getup(), device="cuda:0", seed=2)
    env.reset(randomize_episode_start=False)
    assert env.task.fell(env.qpos, env.xmat[:, 1]).float().mean() > 0.6  # they start fallen
    # Put them all on their feet: standing steady for success_seconds ends the episode with the bonus.
    env.qpos[:] = env.task.standing_qpos
    env.qvel[:] = 0.0
    env.steady_steps[:] = 0
    still = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    for step in range(env.task.task.success_steps):
        result = env.step(still)
        assert result.done.all().item() == (step == env.task.task.success_steps - 1), step
    assert (result.terms["success"] == env.task.config.success_bonus).all()
    assert not result.episode_fell.any()  # success isn't counted as a fall


def test_tiny_rsl_training_run_plays_in_cpu_mujoco(tmp_path):
    from robot3d.gpu.rsl import RslConfig, train_rsl
    from robot3d.policy import PolicyController
    from robot3d.runs import list_checkpoints, read_run_info
    from robot3d.simulation import Simulation

    run_dir = train_rsl(
        total_steps=256 * 24 * 3,
        name="tiny_rsl",
        checkpoint_every=256 * 24,
        rsl=RslConfig(num_envs=256),
        runs_dir=tmp_path,
        log=lambda *_: None,
    )
    assert read_run_info(run_dir)["trainer"] == "rsl-rl"
    checkpoints = list_checkpoints(run_dir)
    assert len(checkpoints) >= 2
    sim = Simulation("quadruped")
    sim.set_controller(PolicyController(checkpoints[-1], sim.model))  # TorchScript actor on the CPU
    for _ in range(50):
        sim.step()
    assert np.isfinite(sim.data.qpos).all()


def test_tiny_gpu_training_run_plays_in_cpu_mujoco(tmp_path):
    from robot3d.gpu.ppo import GpuPPOConfig, train_gpu
    from robot3d.policy import PolicyController, evaluate
    from robot3d.runs import list_checkpoints, run_summary
    from robot3d.simulation import Simulation

    run_dir = train_gpu(
        total_steps=256 * 24 * 3,
        name="tiny_gpu",
        checkpoint_every=256 * 24,
        ppo=GpuPPOConfig(num_envs=256),
        runs_dir=tmp_path,
        log=lambda *_: None,
    )
    summary = run_summary(run_dir)
    assert summary.backend == "gpu" and summary.status == "finished"
    checkpoints = list_checkpoints(run_dir)
    assert len(checkpoints) >= 2 and all(c.format == "torch" for c in checkpoints)
    assert list((run_dir / "tb").rglob("events.out.tfevents.*"))

    # A GPU-trained policy drives regular CPU MuJoCo (the viewer's path).
    sim = Simulation("quadruped")
    sim.set_controller(PolicyController(checkpoints[-1], sim.model))
    for _ in range(50):
        sim.step()
    assert np.isfinite(sim.data.qpos).all()
    results = evaluate(checkpoints[-1], episodes=1)
    assert results[0]["seconds"] > 0
