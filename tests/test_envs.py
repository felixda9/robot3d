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
    assert env.observation_space.shape == (36,)  # 34 + gait clock (sin, cos)
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


def reward_terms(task, vx=0.0, vy=0.0, feet_down=(1, 1, 1, 1), landed=(0, 0, 0, 0), air_time=(0, 0, 0, 0),
                 foot_height=(0, 0, 0, 0), phase=0.0):
    return task.reward(
        vx=vx, vy=vy, motor_power=0.0, action=np.zeros(8), last_action=np.zeros(8), up_z=1.0, fell=False,
        foot_slip=0.0, feet_down=np.array(feet_down, bool), landed=np.array(landed, bool),
        air_time=np.array(air_time, float), foot_height=np.array(foot_height, float), phase=phase,
    )[1]


def test_speed_tracking_peaks_at_the_target(env):
    c = env.task.config
    at_target = reward_terms(env.task, vx=c.target_speed)["tracking"]
    assert at_target == pytest.approx(c.tracking_weight)
    assert reward_terms(env.task, vx=0.0)["tracking"] < at_target  # standing still
    assert reward_terms(env.task, vx=1.2)["tracking"] < reward_terms(env.task, vx=0.6)["tracking"]  # running
    assert reward_terms(env.task, vx=c.target_speed, vy=0.3)["tracking"] < at_target  # drifting sideways
    assert reward_terms(env.task, vx=3.0)["forward"] == 0.0  # the old raw-speed term is off


def test_gait_terms(env):
    from robot3d.walk import WalkConfig, WalkTask

    task = env.task
    feet = [task.model.geom(int(g)).name for g in task.feet]
    assert feet == ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    assert reward_terms(task, feet_down=(1, 0, 0, 1))["support"] == 0.0  # trot phase: 2 feet down
    assert reward_terms(task, feet_down=(1, 0, 0, 0))["support"] < 0  # hopping on one foot
    assert reward_terms(task, feet_down=(0, 0, 0, 0))["support"] < 0  # airborne: running

    # trot_rsl's terms (off by default now: the gait clock replaced them)
    assert reward_terms(task, feet_down=(1, 0, 0, 1))["trot"] == 0.0
    task = WalkTask(env.model, WalkConfig(trot_weight=0.5, air_time_weight=2.0))
    assert reward_terms(task, feet_down=(1, 0, 0, 1))["trot"] == pytest.approx(0.5)
    assert reward_terms(task, feet_down=(1, 1, 0, 0))["trot"] == 0.0  # front pair together: a bound, not a trot
    long_step = reward_terms(task, landed=(1, 0, 0, 0), air_time=(0.4, 0, 0, 0))["air_time"]
    short_step = reward_terms(task, landed=(1, 0, 0, 0), air_time=(0.06, 0, 0, 0))["air_time"]
    assert long_step > 0 > short_step


def test_gait_clock_schedule(env):
    task = env.task
    c = task.config
    assert task.clock and task.obs_size == 36
    steps_per_cycle = round(1 / (c.gait_frequency * task.control_dt))
    assert task.gait_phase(0) == 0.0 and task.gait_phase(steps_per_cycle) == pytest.approx(0.0, abs=1e-9)
    # FL+RR down in phase [0, 0.6), FR+RL in [0.5, 1.1): trot-walk with brief 4-feet moments
    FL, FR, RL, RR = 0, 1, 2, 3
    assert task.desired_down(0.05).tolist() == [True, True, True, True]  # both pairs down (FR+RL landing)
    assert task.desired_down(0.3).tolist() == [True, False, False, True]  # FR+RL swinging
    assert task.desired_down(0.55).tolist() == [True, True, True, True]
    assert task.desired_down(0.8).tolist() == [False, True, True, False]  # FL+RR swinging
    feet_at_30 = (1, 0, 0, 1)

    matching = reward_terms(task, feet_down=feet_at_30, foot_height=(0, 0.04, 0.04, 0), phase=0.3)
    assert matching["gait"] == pytest.approx(c.gait_weight)
    assert matching["clearance"] == pytest.approx(c.clearance_weight)
    # The trot_rsl scoot: all four planted. Half the feet are wrong, nothing lifts.
    scoot = reward_terms(task, feet_down=(1, 1, 1, 1), phase=0.3)
    assert scoot["gait"] == pytest.approx(0.5 * c.gait_weight) and scoot["clearance"] == 0.0
    # Lifting half as high earns half; stance feet's height doesn't count.
    low = reward_terms(task, feet_down=feet_at_30, foot_height=(0.1, 0.02, 0.02, 0.1), phase=0.3)
    assert low["clearance"] == pytest.approx(0.5 * c.clearance_weight)
    assert reward_terms(task, phase=0.05)["clearance"] == 0.0  # nobody is swinging
    del FL, FR, RL, RR


def test_gait_clock_in_observations(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == (36,) and obs[-2:].tolist() == pytest.approx([0.0, 1.0])  # phase 0: (sin, cos)
    obs, *_ = env.step(np.zeros(8))
    angle = 2 * np.pi * env.task.gait_phase(1)
    assert obs[-2:].tolist() == pytest.approx([np.sin(angle), np.cos(angle)], abs=1e-6)


def test_air_time_bookkeeping(env):
    task = env.task
    down, air = np.ones(4, bool), np.zeros(4)
    history = [(1, 1, 1, 1), (0, 1, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)]  # FL lifts for 2 steps, lands
    landings = []
    for state in history:
        now = np.array(state, bool)
        landed, at_landing, air = task.air_time_update(down, air, now)
        down = now
        landings.append((landed[0], round(at_landing[0], 4)))
    assert landings[-1] == (True, round(2 * task.control_dt, 4))
    assert air[0] == 0.0


def test_first_task_runs_keep_their_reward():
    from robot3d.walk import WalkConfig

    # run.json of the first task (walk_10m etc.) saved forward_weight=1.0 and no walk terms
    newer = ("tracking_weight", "support_weight", "trot_weight", "air_time_weight",
             "gait_frequency", "gait_weight", "clearance_weight")
    saved = {k: v for k, v in WalkConfig().to_dict().items() if k not in newer}
    saved["forward_weight"] = 1.0
    old = WalkConfig.from_run(saved)
    assert (old.forward_weight, old.tracking_weight, old.support_weight, old.trot_weight, old.air_time_weight) == (
        1.0, 0.0, 0.0, 0.0, 0.0)
    assert (old.gait_frequency, old.gait_weight, old.clearance_weight) == (0.0, 0.0, 0.0)


def test_runs_before_the_clock_keep_their_observation(env):
    from robot3d.walk import WalkConfig, WalkTask

    saved = WalkConfig().to_dict()
    for key in ("gait_frequency", "gait_weight", "clearance_weight"):
        del saved[key]  # like trot_rsl's run.json
    task = WalkTask(env.model, WalkConfig.from_run(saved))
    assert not task.clock and task.obs_size == 34
    assert task.observation(env.data, np.zeros(8), 0.37).shape == (34,)
    assert reward_terms(task, feet_down=(1, 1, 1, 1), phase=0.3)["gait"] == 0.0


def test_foot_slip_penalty(env):
    env.reset(seed=0)
    for _ in range(50):  # let the reset noise settle (the feet shuffle a little at first)
        _, _, _, _, info = env.step(np.zeros(8))
    assert info["reward_slip"] == pytest.approx(0.0, abs=1e-4)  # standing: feet planted

    env.reset(seed=0)
    env.data.qvel[0] = 1.5  # shove the whole robot forward: planted feet skid
    mujoco.mj_forward(env.model, env.data)
    _, _, _, _, info = env.step(np.zeros(8))
    assert info["reward_slip"] < -0.1


def test_old_runs_keep_their_reward():
    from robot3d.walk import WalkConfig

    saved = WalkConfig().to_dict()
    del saved["slip_weight"]  # a run from before the slip penalty existed
    assert WalkConfig.from_run(saved).slip_weight == 0.0
    assert WalkConfig.from_run(WalkConfig().to_dict()) == WalkConfig()


def test_same_seed_same_episode():
    a, b = WalkEnv(), WalkEnv()
    obs_a, _ = a.reset(seed=42)
    obs_b, _ = b.reset(seed=42)
    assert np.array_equal(obs_a, obs_b)
    action = np.full(8, 0.3)
    assert np.array_equal(a.step(action)[0], b.step(action)[0])
