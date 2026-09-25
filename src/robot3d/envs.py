"""Gymnasium environments.

    import gymnasium as gym
    import robot3d.envs  # registers the environments
    env = gym.make("robot3d/Walk-v0", robot="quadruped")

The standard Gymnasium interface (reset/step, observation_space,
action_space) is what lets off-the-shelf RL libraries like Stable-Baselines3
train on our robots.
"""

import gymnasium as gym
import mujoco
import numpy as np

from robot3d.robots import load_model
from robot3d.walk import WalkConfig, WalkTask


class WalkEnv(gym.Env):
    """Teach a robot to walk forward (+x) without falling over.

    One step = one policy decision = 20 ms of simulated time (10 physics
    steps). An episode ends when the robot falls ("terminated") or after 20 s
    ("truncated"). The observation, action and reward are in walk.py.
    """

    metadata = {"render_modes": []}  # we watch policies in the web viewer instead

    def __init__(self, robot: str = "quadruped", config: WalkConfig | None = None):
        self.robot = robot
        self.model = load_model(robot)
        self.data = mujoco.MjData(self.model)
        self.task = WalkTask(self.model, config or WalkConfig())
        n = self.task.num_actions
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.task.obs_size,), dtype=np.float32)
        self._last_action = np.zeros(n)
        self._steps = 0
        self._start_x = 0.0
        nfeet = len(self.task.feet)
        self._feet_down = np.ones(nfeet, dtype=bool)  # per foot: on the ground at the last step?
        self._air_time = np.zeros(nfeet)  # per foot: seconds in the air so far

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds self.np_random
        self.task.reset_state(self.data, self.np_random)
        self._last_action = np.zeros(self.task.num_actions)
        self._steps = 0
        self._start_xy = self.data.qpos[0:2].copy()
        self._next_push = self._push_delay()
        self._steady_steps = 0
        self._feet_down = self.task.feet_state(self.data)[1]
        self._air_time = np.zeros(len(self.task.feet))
        return self.task.observation(self.data, self._last_action, self.task.gait_phase(0)), {}

    def step(self, action):
        task, model, data = self.task, self.model, self.data
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        data.ctrl[:] = task.action_to_ctrl(action)

        c = task.config
        if c.push_interval > 0 and self._steps >= self._next_push:
            # A shove: the torso's horizontal velocity (free joint qvel[0:2],
            # world frame) jumps, as if bumped into (see WalkConfig.push_interval).
            data.qvel[0:2] += self.np_random.uniform(-c.push_max_speed, c.push_max_speed, 2)
            self._next_push = self._steps + self._push_delay()

        x_before, y_before = data.qpos[0], data.qpos[1]
        feet_before = task.feet_state(data)
        power = 0.0
        for _ in range(task.decimation):
            mujoco.mj_step(model, data)
            # Mechanical power = torque x joint speed, per motor; |.| because a
            # motor spends energy both pushing and braking.
            power += float(np.abs(data.actuator_force * data.qvel[task.joint_qvel]).sum())
        power /= task.decimation
        # Average speed over the step (smoother than the instantaneous velocity).
        forward_velocity = (data.qpos[0] - x_before) / task.control_dt
        lateral_velocity = (data.qpos[1] - y_before) / task.control_dt
        vx, vy = task.heading_velocity(data, forward_velocity, lateral_velocity)
        feet_after = task.feet_state(data)
        foot_slip = task.foot_slip(feet_before, feet_after)
        feet_down = feet_after[1]
        landed, air_time, self._air_time = task.air_time_update(self._feet_down, self._air_time, feet_down)
        self._feet_down = feet_down

        up_z = task.up_z(data)
        fell = task.fell(data)
        self._steady_steps = self._steady_steps + 1 if task.steady(data) else 0
        succeeded = task.config.success_bonus > 0 and self._steady_steps >= task.success_steps
        reward, terms = task.reward(
            vx=vx, vy=vy, motor_power=power, action=action,
            last_action=self._last_action, up_z=up_z, fell=fell, foot_slip=foot_slip,
            feet_down=feet_down, landed=landed, air_time=air_time,
            foot_height=task.foot_heights(data), phase=task.gait_phase(self._steps + 1),  # the clock after this step
            turn_rate=task.turn_rate(data), height=float(data.qpos[2]),
            joint_offset=data.qpos[task.joint_qpos] - task.home_ctrl,
            joint_velocity=data.qvel[task.joint_qvel], angular_velocity=data.qvel[3:6], succeeded=succeeded,
        )
        self._last_action = action
        self._steps += 1

        observation = task.observation(data, self._last_action, task.gait_phase(self._steps))
        # A fall ends a walk/stand episode; getting up steadily ends a get-up one.
        terminated = (fell and task.config.terminate_on_fall) or succeeded
        truncated = self._steps >= task.max_steps
        info = {
            "forward_velocity": forward_velocity,
            "distance": task.distance(data.qpos[0:2] - self._start_xy),
            "motor_power": power,
            "fell": fell,
            "succeeded": succeeded,
            **{f"reward_{name}": value for name, value in terms.items()},
        }
        return observation, reward, terminated, truncated, info

    def _push_delay(self) -> int:
        return self.task.push_delay(self.np_random.uniform()) if self.task.config.push_interval > 0 else 0


gym.register(id="robot3d/Walk-v0", entry_point="robot3d.envs:WalkEnv")
