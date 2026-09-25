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
    from robot3d.walk import WalkConfig

    return WalkEnv("quadruped", WalkConfig(push_interval=0.0))  # no random shoves: tests check exact behavior


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
                 foot_height=(0, 0, 0, 0), phase=0.0, turn_rate=0.0, fell=False, height=None, joint_offset=0.0,
                 up_z=1.0, joint_velocity=0.0, angular_velocity=(0.0, 0.0, 0.0)):
    return task.reward(
        vx=vx, vy=vy, motor_power=0.0, action=np.zeros(8), last_action=np.zeros(8), up_z=up_z, fell=fell,
        foot_slip=0.0, feet_down=np.array(feet_down, bool), landed=np.array(landed, bool),
        air_time=np.array(air_time, float), foot_height=np.array(foot_height, float), phase=phase,
        turn_rate=turn_rate, height=task.standing_height if height is None else height,
        joint_offset=np.full(task.num_actions, joint_offset),
        joint_velocity=np.full(task.num_actions, joint_velocity), angular_velocity=np.array(angular_velocity),
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
    one_second = round(1 / task.control_dt)
    assert task.gait_phase(0) == 0.0
    assert task.gait_phase(one_second) == pytest.approx(c.gait_frequency % 1.0, abs=1e-9)  # 1.5 Hz: half a cycle on
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


def test_speed_is_measured_along_the_robots_heading(env):
    """The policy can't see which way it faces, so the reward mustn't depend on it."""
    from robot3d.walk import WalkConfig, WalkTask

    env.reset(seed=0)
    env.data.qpos[3:7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]  # turned 90 deg left: nose along +y
    mujoco.mj_forward(env.model, env.data)
    forward, sideways = env.task.heading_velocity(env.data, 0.0, 0.4)  # moving along world +y
    assert (forward, sideways) == pytest.approx((0.4, 0.0), abs=1e-9)
    world = WalkTask(env.model, WalkConfig(velocity_frame="world"))  # runs before 5c
    assert world.heading_velocity(env.data, 0.0, 0.4) == (0.0, 0.4)

    c = env.task.config
    straight = reward_terms(env.task, turn_rate=0.0)["turn"]
    assert straight == pytest.approx(c.turn_weight)
    assert reward_terms(env.task, turn_rate=np.radians(30))["turn"] == pytest.approx(0.33 * c.turn_weight, abs=0.01)
    assert reward_terms(env.task, turn_rate=-1.0)["turn"] < 0.1 * straight  # circling fast


def test_stand_task_rewards_stillness():
    """Stand: hold the home pose exactly, legs and torso still; a fall ends it."""
    from robot3d.walk import WalkConfig

    stand = WalkEnv("quadruped", WalkConfig.stand()).task
    c = stand.config
    assert c.task == "stand" and c.terminate_on_fall and c.action_scale == 0.5
    still = reward_terms(stand)
    assert still["pose"] == pytest.approx(c.pose_weight) and still["tracking"] == pytest.approx(c.tracking_weight)
    assert still["joint_speed"] == 0.0 and still["wobble"] == 0.0
    assert reward_terms(stand, joint_offset=0.05)["pose"] > 0.7 * c.pose_weight  # 3 deg off: still most of it
    assert reward_terms(stand, joint_offset=0.15)["pose"] < 0.2 * c.pose_weight  # 9 deg off on all 8 joints: little
    fidget = reward_terms(stand, joint_velocity=2.0, angular_velocity=(1.0, 0.5, 0.0))
    assert fidget["joint_speed"] == pytest.approx(-c.joint_speed_weight * 8 * 4.0)
    assert fidget["wobble"] == pytest.approx(-c.wobble_weight * 1.25)
    assert reward_terms(stand, fell=True)["fall"] == -c.fall_penalty
    walk = reward_terms(WalkEnv("quadruped").task, joint_velocity=2.0, angular_velocity=(1.0, 0.5, 0.0))
    assert walk["joint_speed"] == walk["wobble"] == 0.0  # walking: moving legs is the point
    # Being shoved along (torso > 0.5 m/s): free to step, no stillness terms.
    shoved = reward_terms(stand, vx=0.8, joint_velocity=2.0, angular_velocity=(1.0, 0.5, 0.0), joint_offset=0.2)
    assert shoved["pose"] == shoved["joint_speed"] == shoved["wobble"] == 0.0
    # And a step's total never goes below 0.
    assert stand.reward(**{**reward_inputs(stand), "joint_velocity": np.full(8, 20.0)})[0] == 0.0


def test_getup_task_rewards():
    """MuJoCo Playground's Go1 getup recipe: orientation + height, then the
    standing pose once upright and holding still once also at full height."""
    from robot3d.walk import WalkConfig

    task = WalkEnv("quadruped", WalkConfig.getup()).task
    c = task.config
    assert c.task == "getup" and WalkConfig().task == "walk"
    up = reward_terms(task)  # standing in the home pose, action 0
    assert up["orientation"] == pytest.approx(c.orientation_weight) and up["height"] == pytest.approx(c.height_weight)
    assert up["pose"] == pytest.approx(c.pose_weight) and up["hold"] == pytest.approx(c.hold_weight)
    assert up["tracking"] == up["turn"] == up["down"] == up["success"] == up["upright"] == 0.0
    tilted = reward_terms(task, up_z=0.95)  # ~18 deg: not upright yet
    assert tilted["pose"] == tilted["hold"] == 0.0 and 0.8 < tilted["orientation"] < 0.85
    low = reward_terms(task, height=0.8 * task.standing_height)  # upright but crouched: pose yes, hold no
    assert low["pose"] > 0 and low["hold"] == 0.0
    on_back = reward_terms(task, up_z=-1.0, height=0.04, fell=True)
    assert on_back["orientation"] < 1e-3 and on_back["fall"] == 0.0  # a fall doesn't end it
    # Totals never go below 0 (penalties can't make ending an episode attractive).
    rough = task.reward(**{**reward_inputs(task), "motor_power": 1e5, "up_z": -1.0})[0]
    assert rough == 0.0
    walk = reward_terms(WalkEnv("quadruped").task, fell=True, height=0.08)
    assert walk["fall"] < 0 and walk["down"] == walk["height"] == walk["orientation"] == walk["hold"] == 0.0


def reward_inputs(task):
    n = task.num_actions
    return dict(
        vx=0.0, vy=0.0, motor_power=0.0, action=np.zeros(n), last_action=np.zeros(n), up_z=1.0, fell=False,
        foot_slip=0.0, feet_down=np.ones(4, bool), landed=np.zeros(4, bool), air_time=np.zeros(4),
        foot_height=np.zeros(4), phase=0.0, turn_rate=0.0, height=task.standing_height, joint_offset=np.zeros(n),
        joint_velocity=np.zeros(n), angular_velocity=np.zeros(3),
    )


def test_getup_actions_are_relative_to_the_current_pose():
    from robot3d.walk import WalkConfig, WalkTask

    task = WalkTask(WalkEnv("quadruped12").model, WalkConfig.getup())
    pose = task.home_ctrl + 0.3
    assert np.allclose(task.action_to_ctrl(np.zeros(12), pose), np.clip(pose, task.ctrl_low, task.ctrl_high))
    assert np.allclose(task.action_to_ctrl(np.ones(12), pose), np.clip(pose + 0.5, task.ctrl_low, task.ctrl_high))
    walk = WalkTask(task.model, WalkConfig())
    assert np.allclose(walk.action_to_ctrl(np.zeros(12), pose), walk.home_ctrl)  # walking: around home


def test_getup_episodes_start_mostly_fallen_and_run_their_full_length():
    from robot3d.walk import WalkConfig

    env = WalkEnv("quadruped12", WalkConfig.getup())
    qpos, qvel = env.task.fallen_states
    assert qpos.shape == (env.task.FALLEN_STATES, env.model.nq) and np.isfinite(qpos).all()
    data = mujoco.MjData(env.model)
    fallen = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(env.model, data)
        fallen.append(env.task.fell(data))
    assert np.mean(fallen) > 0.7  # most land on a side, back or belly

    starts = []
    for seed in range(30):
        env.reset(seed=seed)
        starts.append(env.task.fell(env.data))
    assert 0.35 < np.mean(starts) < 0.85  # 60% start from the fallen bank (a few of those landed on their feet)

    # Neither lying down nor standing up ends the episode: always the full 6 s.
    env.reset(seed=next(i for i, f in enumerate(starts) if f))
    for step in range(env.task.max_steps):
        _, _, terminated, truncated, info = env.step(np.zeros(12))
        assert not terminated and truncated == (step == env.task.max_steps - 1)

    # The viewer always starts it standing.
    rng = np.random.default_rng(0)
    for _ in range(10):
        env.task.reset_state(env.data, rng, allow_fallen=False)
        assert not env.task.fell(env.data)


def test_success_ending_still_works_when_switched_on():
    import dataclasses

    from robot3d.walk import WalkConfig

    # (home-based actions, so action 0 holds the standing pose; with relative
    # actions, action 0 = "target = current angle" = no holding force: it sags)
    config = dataclasses.replace(WalkConfig.getup(), success_bonus=10.0, action_mode="home")
    env = WalkEnv("quadruped12", config)
    env.reset(seed=0)
    env.task.reset_state(env.data, np.random.default_rng(0), allow_fallen=False)
    for step in range(env.task.success_steps):
        _, _, terminated, _, info = env.step(np.zeros(12))
        assert terminated == (step == env.task.success_steps - 1)
    assert info["succeeded"] and info["reward_success"] == 10.0


def test_roll_penalty():
    from robot3d.walk import WalkConfig, WalkTask

    model12 = WalkEnv("quadruped12").model
    task = WalkTask(model12, WalkConfig())
    assert [model12.actuator(int(i)).name for i in task.roll_motors] == ["FL_roll", "FR_roll", "RL_roll", "RR_roll"]
    offsets = np.zeros(12)
    offsets[task.roll_motors] = 0.3  # legs 17 deg out
    terms = task.reward(
        vx=0.4, vy=0.0, motor_power=0.0, action=np.zeros(12), last_action=np.zeros(12), up_z=1.0, fell=False,
        foot_slip=0.0, feet_down=np.ones(4, bool), landed=np.zeros(4, bool), air_time=np.zeros(4),
        foot_height=np.zeros(4), phase=0.0, turn_rate=0.0, height=task.standing_height, joint_offset=offsets,
        joint_velocity=np.zeros(12), angular_velocity=np.zeros(3),
    )[1]
    assert terms["roll"] == pytest.approx(-task.config.roll_weight * 4 * 0.09)
    assert WalkTask(model12, WalkConfig.getup()).config.roll_weight == 0.0  # getting up needs roll freely
    assert WalkTask(model12, WalkConfig.stand()).config.roll_weight == 0.0  # the pose term covers roll
    assert len(WalkEnv("quadruped").task.roll_motors) == 0  # the 8-motor robot has none


def test_random_shoves(env):
    from robot3d.walk import WalkConfig

    def speed_jumps(env, seconds):
        """Steps where the torso's horizontal velocity changed by > 0.3 m/s at once."""
        env.reset(seed=1)
        jumps, last = 0, env.data.qvel[:2].copy()
        for _ in range(round(seconds / env.task.control_dt)):
            env.step(np.zeros(env.task.num_actions))
            jumps += np.linalg.norm(env.data.qvel[:2] - last) > 0.3
            last = env.data.qvel[:2].copy()
        return jumps

    assert speed_jumps(env, 6.0) == 0  # pushes off
    pushed = WalkEnv("quadruped", WalkConfig(push_interval=1.0, push_max_speed=1.0))
    assert 3 <= speed_jumps(pushed, 6.0) <= 12  # every 0.5-1.5 s (a few kicks can be too small to count)
    assert WalkConfig().push_interval > 0  # new runs train with them


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
    newer = ("tracking_weight", "support_weight", "trot_weight", "air_time_weight", "gait_frequency",
             "gait_weight", "clearance_weight", "push_interval", "velocity_frame", "turn_weight", "roll_weight")
    saved = {k: v for k, v in WalkConfig().to_dict().items() if k not in newer}
    saved["forward_weight"] = 1.0
    old = WalkConfig.from_run(saved)
    assert (old.forward_weight, old.tracking_weight, old.support_weight, old.trot_weight, old.air_time_weight) == (
        1.0, 0.0, 0.0, 0.0, 0.0)
    assert (old.gait_frequency, old.gait_weight, old.clearance_weight) == (0.0, 0.0, 0.0)
    assert old.push_interval == 0.0  # no shoves either
    assert (old.velocity_frame, old.turn_weight) == ("world", 0.0)  # speed along world +x, no turn term
    assert old.roll_weight == 0.0


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


def test_runs_from_before_the_task_field_get_their_task():
    from robot3d.walk import WalkConfig

    getup = {k: v for k, v in WalkConfig.getup().to_dict().items() if k != "task"}  # like stand12_reach
    assert WalkConfig.from_run(getup).task == "getup"
    walk = {k: v for k, v in WalkConfig().to_dict().items() if k != "task"}
    assert WalkConfig.from_run(walk).task == "walk"


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
