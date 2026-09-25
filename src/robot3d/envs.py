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
from robot3d.terrain import Terrain
from robot3d.walk import WalkConfig, WalkTask


class WalkEnv(gym.Env):
    """Teach a robot to walk forward (+x) without falling over.

    One step = one policy decision = 20 ms of simulated time (10 physics
    steps). An episode ends when the robot falls ("terminated") or after 20 s
    ("truncated"). The observation, action and reward are in walk.py.
    """

    metadata = {"render_modes": []}  # we watch policies in the web viewer instead

    def __init__(self, robot: str = "quadruped", config: WalkConfig | None = None, terrain: Terrain | None = None):
        """terrain: the ground (default: config.terrain's layout, or the flat
        floor). Episodes start on a random tile of it, facing a random way."""
        self.robot = robot
        config = config or WalkConfig()
        if terrain is None and config.terrain:
            terrain = Terrain.make(config.terrain, config.terrain_seed)
        self.terrain = terrain
        self.model = load_model(robot, terrain)
        self.data = mujoco.MjData(self.model)
        self.task = WalkTask(self.model, config, terrain=terrain)
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
        if self.task.randomizes_physics:
            self.task.apply_physics(self.model, *self.task.sample_physics(self.np_random))
        # options={"spawn": (x, y, heading)}: start there (e.g. a test course section)
        spawn = (options or {}).get("spawn")
        if spawn is None and self.terrain is not None and self.terrain.tiles:
            spawn = self.terrain.random_spawn(self.np_random)
        self.task.reset_state(self.data, self.np_random, spawn=spawn)
        self._last_action = np.zeros(self.task.num_actions)
        self._steps = 0
        self._start_xy = self.data.qpos[0:2].copy()
        self._next_push = self._push_delay()
        self._steady_steps = 0
        self._push_left = 0  # control steps left of a force push
        self._flight_steps, self._landed = 0, False  # jump task
        self._resample_command()
        self._feet_down = self.task.feet_state(self.data)[1]
        self._air_time = np.zeros(len(self.task.feet))
        return self._observe(), {}

    def step(self, action):
        task, model, data = self.task, self.model, self.data
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        data.ctrl[:] = task.action_to_ctrl(action, data.qpos[task.joint_qpos])

        c = task.config
        if c.commands and self._steps >= self._next_command:
            self._resample_command()
        if c.push_interval > 0 and self._steps >= self._next_push and c.push_kind == "force":
            # A push like the viewer's (see WalkConfig.push_kind).
            self.push(self.np_random.uniform(0, 2 * np.pi), self.np_random.uniform(0, c.push_max_speed),
                      self.np_random.uniform(0.0, 1.0))
            self._next_push = self._steps + self._push_delay()
        elif c.push_interval > 0 and self._steps >= self._next_push:
            # A shove: the torso's horizontal velocity (free joint qvel[0:2],
            # world frame) jumps, as if bumped into (see WalkConfig.push_interval).
            if c.push_direction == "circle":
                angle = self.np_random.uniform(0, 2 * np.pi)
                size = self.np_random.uniform(0, c.push_max_speed)
                data.qvel[0:2] += size * np.array([np.cos(angle), np.sin(angle)])
            else:
                data.qvel[0:2] += self.np_random.uniform(-c.push_max_speed, c.push_max_speed, 2)
            if c.push_max_spin > 0:  # and a twist (qvel[3:6]: torso angular velocity, its own frame)
                data.qvel[3:6] += self.np_random.uniform(-c.push_max_spin, c.push_max_spin, 3)
            self._next_push = self._steps + self._push_delay()

        if self._push_left > 0:
            self._push_left -= 1
        elif data.xfrc_applied[1].any():
            data.xfrc_applied[1] = 0.0  # the push is over

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
        airborne, self._flight_steps, self._landed = task.jump_update(self._flight_steps, self._landed, feet_down)
        succeeded = task.config.success_bonus > 0 and self._steady_steps >= task.success_steps
        reward, terms = task.reward(
            vx=vx, vy=vy, motor_power=power, action=action,
            last_action=self._last_action, up_z=up_z, fell=fell, foot_slip=foot_slip,
            feet_down=feet_down, landed=landed, air_time=air_time,
            foot_height=task.foot_heights(data), phase=task.gait_phase(self._steps + 1),  # the clock after this step
            turn_rate=task.turn_rate(data), height=task.height(data),
            joint_offset=data.qpos[task.joint_qpos] - task.home_ctrl,
            joint_velocity=data.qvel[task.joint_qvel], angular_velocity=data.qvel[3:6], succeeded=succeeded,
            jump_airborne=airborne, jump_landed=self._landed, command=self._command,
            stumbling=task.stumbling_feet(data) if c.stumble_weight > 0 else None,
        )
        self._last_action = action
        self._steps += 1

        observation = self._observe()
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

    def _observe(self) -> np.ndarray:
        task = self.task
        return task.observation(self.data, self._last_action, task.gait_phase(self._steps), self._command,
                                rng=self.np_random)

    def _resample_command(self) -> None:
        """A new steering command (WalkConfig.commands), or walking straight at target_speed."""
        task = self.task
        if task.config.commands:
            self._command = task.sample_command(self.np_random)
            self._next_command = self._steps + round(task.config.command_resample_seconds / task.control_dt)
        else:
            self._command = task.default_command()
            self._next_command = 2**62

    def push(self, angle: float, size: float, height: float) -> None:
        """Start a viewer-like force push on the torso (WalkTask.push_wrench):
        it acts for the next PUSH_SECONDS of control steps."""
        data = self.data
        force, torque = self.task.push_wrench(data.xmat[1].reshape(3, 3), data.xpos[1], data.xipos[1],
                                              angle, size, height)
        data.xfrc_applied[1, :3] = force
        data.xfrc_applied[1, 3:] = torque
        self._push_left = round(self.task.PUSH_SECONDS / self.task.control_dt)

    def _push_delay(self) -> int:
        return self.task.push_delay(self.np_random.uniform()) if self.task.config.push_interval > 0 else 0


gym.register(id="robot3d/Walk-v0", entry_point="robot3d.envs:WalkEnv")
