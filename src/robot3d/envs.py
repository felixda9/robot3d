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

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds self.np_random
        self.task.reset_state(self.data, self.np_random)
        self._last_action = np.zeros(self.task.num_actions)
        self._steps = 0
        self._start_x = float(self.data.qpos[0])
        return self.task.observation(self.data, self._last_action), {}

    def step(self, action):
        task, model, data = self.task, self.model, self.data
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        data.ctrl[:] = task.action_to_ctrl(action)

        x_before = data.qpos[0]
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
        foot_slip = task.foot_slip(feet_before, task.feet_state(data))

        up_z = task.up_z(data)
        fell = task.fell(data)
        reward, terms = task.reward(forward_velocity, power, action, self._last_action, up_z, fell, foot_slip)
        self._last_action = action
        self._steps += 1

        observation = task.observation(data, self._last_action)
        terminated = fell
        truncated = self._steps >= task.max_steps
        info = {
            "forward_velocity": forward_velocity,
            "distance": float(data.qpos[0]) - self._start_x,
            "motor_power": power,
            "fell": fell,
            **{f"reward_{name}": value for name, value in terms.items()},
        }
        return observation, reward, terminated, truncated, info


gym.register(id="robot3d/Walk-v0", entry_point="robot3d.envs:WalkEnv")
