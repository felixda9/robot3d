"""The batched (GPU) walk task must compute exactly what walk.WalkTask computes.

States come from the regular CPU environment driven by random actions, so
they cover standing, stumbling and falling. Runs on the CPU (torch CPU
tensors): this checks the math, not the GPU.
"""

import mujoco
import numpy as np
import pytest
import torch

from robot3d.envs import WalkEnv
from robot3d.gpu.task import BatchedWalkTask

N = 12


@pytest.fixture(scope="module")
def snapshots():
    """N pairs of consecutive control-step states, plus what the CPU task said about them."""
    env = WalkEnv()
    task = env.task
    rng = np.random.default_rng(0)
    rows = []
    for i in range(N):
        env.reset(seed=i)
        for _ in range(int(rng.integers(1, 60))):
            env.step(rng.uniform(-1, 1, task.num_actions))
        if i % 4 == 3:  # some robots upside down
            env.data.qpos[3:7] = [0, 1, 0, 0]
            mujoco.mj_forward(env.model, env.data)
        # Note: right after mj_step, MuJoCo's xmat/geom_xpos are from the start
        # of that step (one 2 ms step behind qpos). The CPU env, the viewer
        # and MuJoCo Warp all read them at that point, so we compare exactly
        # those arrays rather than recomputing them.
        d = env.data
        before = {"qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "rot": d.xmat[1].reshape(3, 3).copy(),
                  "geom_xpos": d.geom_xpos.copy(), "last_action": env._last_action.copy()}
        phase = task.gait_phase(env._steps)
        obs_before = task.observation(d, env._last_action, phase)
        feet_before = task.feet_state(d)
        env.step(rng.uniform(-1, 1, task.num_actions))
        after = {"qpos": d.qpos.copy(), "geom_xpos": d.geom_xpos.copy(), "rot": d.xmat[1].reshape(3, 3).copy()}
        rows.append({
            "steps": env._steps - 1,  # step count at "before"
            "before": before,
            "after": after,
            "obs_before": obs_before,
            "fell": task.fell(d),
            "up_z": task.up_z(d),
            "slip": task.foot_slip(feet_before, task.feet_state(d)),
        })
    return task, rows


def stack(rows, part, key):
    return torch.tensor(np.stack([r[part][key] for r in rows]), dtype=torch.float32)


def test_observation_matches(snapshots):
    task, rows = snapshots
    batched = BatchedWalkTask(task, "cpu")
    steps = torch.tensor([r["steps"] for r in rows])
    obs = batched.observation(
        stack(rows, "before", "qpos"), stack(rows, "before", "qvel"),
        stack(rows, "before", "rot"), stack(rows, "before", "last_action"), batched.gait_phase(steps),
    )
    expected = torch.tensor(np.stack([r["obs_before"] for r in rows]))
    assert obs.shape == (N, task.obs_size)
    torch.testing.assert_close(obs, expected, rtol=1e-5, atol=1e-5)


def test_falling_and_uprightness_match(snapshots):
    task, rows = snapshots
    batched = BatchedWalkTask(task, "cpu")
    rot, qpos = stack(rows, "after", "rot"), stack(rows, "after", "qpos")
    assert batched.fell(qpos, rot).tolist() == [r["fell"] for r in rows]
    assert any(r["fell"] for r in rows) and not all(r["fell"] for r in rows)  # both cases covered
    torch.testing.assert_close(batched.up_z(rot), torch.tensor([r["up_z"] for r in rows], dtype=torch.float32))


def test_foot_slip_matches(snapshots):
    task, rows = snapshots
    batched = BatchedWalkTask(task, "cpu")
    slip = batched.foot_slip(
        batched.feet_state(stack(rows, "before", "geom_xpos")),
        batched.feet_state(stack(rows, "after", "geom_xpos")),
    )
    torch.testing.assert_close(slip, torch.tensor([r["slip"] for r in rows], dtype=torch.float32), rtol=1e-4, atol=1e-6)


def test_action_to_ctrl_and_reward_match(snapshots):
    task, _ = snapshots
    batched = BatchedWalkTask(task, "cpu")
    rng = np.random.default_rng(1)
    actions = rng.uniform(-1.5, 1.5, (N, task.num_actions))  # includes out-of-range actions
    expected = np.stack([task.action_to_ctrl(a) for a in actions])
    torch.testing.assert_close(batched.action_to_ctrl(torch.tensor(actions, dtype=torch.float32)),
                               torch.tensor(expected, dtype=torch.float32))

    nfeet = len(task.feet)
    inputs = {
        "vx": rng.uniform(-0.5, 2.0, N), "vy": rng.uniform(-0.5, 0.5, N), "motor_power": rng.uniform(0, 80, N),
        "action": rng.uniform(-1, 1, (N, task.num_actions)), "last_action": rng.uniform(-1, 1, (N, task.num_actions)),
        "up_z": rng.uniform(-1, 1, N), "fell": rng.random(N) < 0.3, "foot_slip": rng.uniform(0, 0.5, N),
        "feet_down": rng.random((N, nfeet)) < 0.6, "landed": rng.random((N, nfeet)) < 0.3,
        "air_time": rng.uniform(0, 0.8, (N, nfeet)), "foot_height": rng.uniform(-0.001, 0.06, (N, nfeet)),
        "phase": np.array([task.gait_phase(int(s)) for s in rng.integers(0, 1000, N)]),
    }
    booleans = ("fell", "feet_down", "landed")
    dtypes = {k: torch.bool for k in booleans} | {"phase": torch.float64}
    rewards, terms = batched.reward(**{k: torch.tensor(v, dtype=dtypes.get(k, torch.float32))
                                       for k, v in inputs.items()})
    for i in range(N):
        r, t = task.reward(**{k: v[i] for k, v in inputs.items()})
        assert rewards[i].item() == pytest.approx(r, rel=1e-5, abs=1e-5)
        assert set(t) == set(terms)
        for name, value in t.items():
            assert terms[name][i].item() == pytest.approx(value, rel=1e-5, abs=1e-5), name


def test_gait_clock_matches_exactly(snapshots):
    """Every step of a long episode, including the cycle boundaries, where a
    float32 phase could land on the other side of 0 and flip the schedule."""
    task, _ = snapshots
    batched = BatchedWalkTask(task, "cpu")
    steps = torch.arange(0, 2 * task.max_steps)
    phases = batched.gait_phase(steps)
    expected = [task.gait_phase(int(s)) for s in steps]
    assert phases.tolist() == expected
    down = batched.desired_down(phases)
    assert down.tolist() == [task.desired_down(p).tolist() for p in expected]


def test_air_time_update_matches(snapshots):
    task, _ = snapshots
    batched = BatchedWalkTask(task, "cpu")
    rng = np.random.default_rng(2)
    prev = rng.random((N, len(task.feet))) < 0.5
    down = rng.random((N, len(task.feet))) < 0.5
    air = rng.uniform(0, 0.5, (N, len(task.feet)))
    got = batched.air_time_update(torch.tensor(prev), torch.tensor(air, dtype=torch.float32), torch.tensor(down))
    for i in range(N):
        expected = task.air_time_update(prev[i], air[i], down[i])
        for g, e in zip(got, expected):
            np.testing.assert_allclose(g[i].numpy(), e, rtol=1e-6, atol=1e-6)


def test_reset_state_noise(snapshots):
    task, _ = snapshots
    batched = BatchedWalkTask(task, "cpu")
    qpos, qvel = batched.reset_state(500, torch.Generator().manual_seed(0))
    joint_offset = qpos[:, batched.joint_qpos] - batched.standing_qpos[batched.joint_qpos]
    assert joint_offset.abs().max() <= task.config.reset_joint_noise + 1e-6
    assert joint_offset.std() > 0.03  # it is actually random
    other = torch.ones(qpos.shape[1], dtype=torch.bool)
    other[batched.joint_qpos] = False
    torch.testing.assert_close(qpos[:, other], batched.standing_qpos[other].expand(500, -1))
    assert (qvel - batched.standing_qvel).abs().max() <= task.config.reset_velocity_noise + 1e-6
