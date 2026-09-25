"""The walk environment (Gymnasium interface, observation, action, reward)."""

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import robot3d.envs  # noqa: F401  (registers robot3d/Walk-v0)
from robot3d.envs import WalkEnv


@pytest.fixture
def env():
    return WalkEnv("quadruped")


def test_passes_gymnasium_checker():
    env = gym.make("robot3d/Walk-v0")
    with pytest.warns(UserWarning, match="infinity"):  # unbounded observations are normal for MuJoCo envs
        check_env(env.unwrapped, skip_render_check=True)
    assert env.observation_space.shape == (34,)
    assert env.action_space.shape == (8,)


def test_standing_still_survives_a_whole_episode(env):
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, info = env.step(np.zeros(8))  # action 0 = hold the home pose
        steps += 1
        if terminated or truncated:
            break
    assert not terminated and truncated
    assert steps == env.task.max_steps == 1000  # 20 s at 50 Hz
    assert abs(info["distance"]) < 0.05


def test_falling_over_ends_the_episode_with_a_penalty(env):
    env.reset(seed=0)
    env.data.qpos[3:7] = [0, 1, 0, 0]  # torso upside down (180 degrees about x)
    mujoco.mj_forward(env.model, env.data)
    _, reward, terminated, _, info = env.step(np.zeros(8))
    assert terminated and info["fell"]
    assert info["reward_fall"] == -env.task.config.fall_penalty
    assert reward < -5


def test_action_maps_to_offsets_around_home(env):
    task = env.task
    assert np.allclose(task.action_to_ctrl(np.zeros(8)), task.home_ctrl)
    ctrl = task.action_to_ctrl(np.ones(8))
    expected = np.minimum(task.home_ctrl + task.config.action_scale, task.ctrl_high)
    assert np.allclose(ctrl, expected)
    assert np.allclose(task.action_to_ctrl(5 * np.ones(8)), ctrl)  # actions are clipped to +-1


def test_observation_ignores_position_and_heading(env):
    """Walking works the same anywhere and facing any direction, so the
    observation must not change when the robot is moved or turned."""
    env.reset(seed=0)
    env.data.qvel[0] = 0.3  # moving forward
    mujoco.mj_forward(env.model, env.data)
    before = env.task.observation(env.data, np.zeros(8))

    yaw = np.pi / 2  # turn 90 degrees left and move 5 m away
    turn = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, turn, env.data.qpos[3:7].copy())
    env.data.qpos[3:7] = quat
    env.data.qpos[0:2] += [5.0, -3.0]
    # Same motion relative to the robot: the world-frame velocity turns with it
    # (angular velocity is already in the torso's frame, so it stays).
    rotate = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    env.data.qvel[0:3] = rotate @ env.data.qvel[0:3]
    mujoco.mj_forward(env.model, env.data)
    after = env.task.observation(env.data, np.zeros(8))
    assert np.allclose(before, after, atol=1e-6)


def test_moving_forward_is_rewarded_up_to_a_cap(env):
    config = env.task.config
    terms = lambda speed: env.task.reward(speed, 0.0, np.zeros(8), np.zeros(8), 1.0, False)[1]  # noqa: E731
    assert terms(0.5)["forward"] == pytest.approx(0.5 * config.forward_weight)
    assert terms(-0.5)["forward"] < 0  # walking backwards is punished
    assert terms(3.0)["forward"] == pytest.approx(config.max_reward_speed * config.forward_weight)


def test_same_seed_same_episode():
    a, b = WalkEnv(), WalkEnv()
    obs_a, _ = a.reset(seed=42)
    obs_b, _ = b.reset(seed=42)
    assert np.array_equal(obs_a, obs_b)
    action = np.full(8, 0.3)
    assert np.array_equal(a.step(action)[0], b.step(action)[0])
