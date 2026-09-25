"""The "walk forward" task: what a policy observes, how its actions become
motor targets, and how it is rewarded.

Shared by the Gymnasium environment (training, envs.py) and the policy player
in the web server (watching, policy.py), so a policy sees and acts exactly
the same way in both.

Works for any robot that follows the project conventions: body 1 is the
free-floating torso, every motor is a position actuator on a joint, and a
"home" keyframe holds the standing pose's motor targets.
"""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class WalkConfig:
    # --- control
    control_dt: float = 0.02  # the policy acts at 50 Hz (every 10 physics steps of 2 ms)
    action_scale: float = 0.5  # action +-1 = target +-0.5 rad around the home pose
    episode_seconds: float = 20.0  # then the episode is cut off ("truncated")

    # --- reward: one number per control step; the policy learns to make the sum large
    forward_weight: float = 1.0  # x forward speed (m/s)...
    max_reward_speed: float = 1.0  # ...counted up to this speed, so sprinting recklessly doesn't pay
    upright_weight: float = 0.5  # torso "up" axis . world up: 1 level, 0 on its side
    energy_weight: float = 0.002  # per watt of mechanical motor power |torque x joint speed|
    smoothness_weight: float = 0.05  # per unit of squared action change (discourages jitter)
    fall_penalty: float = 10.0  # once, when the episode ends by falling

    # --- falling (ends the episode)
    min_up_z: float = 0.5  # torso tilted more than 60 degrees
    min_height_fraction: float = 0.5  # or torso lower than half its standing height

    # --- start of each episode: the settled standing pose plus random noise,
    # so the policy learns to cope with slightly different starts
    reset_joint_noise: float = 0.1  # rad
    reset_velocity_noise: float = 0.1  # rad/s and m/s

    def to_dict(self) -> dict:
        return asdict(self)


class WalkTask:
    """Model-specific constants plus the task's observation, action, and reward."""

    def __init__(self, model: mujoco.MjModel, config: WalkConfig = WalkConfig(), keyframe: str = "home"):
        self.model = model
        self.config = config
        self.decimation = max(1, round(config.control_dt / model.opt.timestep))
        self.control_dt = self.decimation * model.opt.timestep
        self.max_steps = round(config.episode_seconds / self.control_dt)

        self.home_ctrl = model.key(keyframe).ctrl.copy()
        limited = model.actuator_ctrllimited.astype(bool)
        self.ctrl_low = np.where(limited, model.actuator_ctrlrange[:, 0], -np.inf)
        self.ctrl_high = np.where(limited, model.actuator_ctrlrange[:, 1], np.inf)
        joints = model.actuator_trnid[:, 0]
        self.joint_qpos = model.jnt_qposadr[joints]  # where each motor's joint angle lives in qpos
        self.joint_qvel = model.jnt_dofadr[joints]  # ... and its velocity in qvel

        self.num_actions = model.nu
        # torso height, gravity (3), linear velocity (3), angular velocity (3),
        # then per motor: joint angle, joint speed, previous action
        self.obs_size = 1 + 3 + 3 + 3 + 3 * model.nu

        self.standing_qpos, self.standing_qvel = self._settled_standing_state(keyframe)
        self.standing_height = float(self.standing_qpos[2])

    # ----------------------------------------------------------------- actions

    def action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Policy action (each in -1..1) -> motor target angles.

        Actions are offsets around the standing pose: action 0 = stand still.
        Starting from a sensible pose makes learning much faster than letting
        the policy pick raw angles from the whole joint range.
        """
        action = np.clip(action, -1.0, 1.0)
        return np.clip(self.home_ctrl + self.config.action_scale * action, self.ctrl_low, self.ctrl_high)

    # ------------------------------------------------------------- observation

    def observation(self, data: mujoco.MjData, last_action: np.ndarray) -> np.ndarray:
        """What the policy "feels", like a real robot's IMU and joint encoders.

        Directions are expressed in the torso's own frame, so the policy
        behaves the same whichever way the robot faces, and absolute x/y
        position is left out (walking works the same anywhere on the floor).
        """
        torso_rot = data.xmat[1].reshape(3, 3)  # torso frame -> world frame
        world_to_torso = torso_rot.T
        gravity = world_to_torso @ np.array([0.0, 0.0, -1.0])  # "which way is down": tilt, without heading
        linear_velocity = world_to_torso @ data.qvel[0:3]  # free joint: linear velocity is in world frame
        angular_velocity = data.qvel[3:6]  # ... angular velocity already in the torso's frame
        joint_angles = data.qpos[self.joint_qpos] - self.home_ctrl  # relative to the standing pose
        joint_speeds = data.qvel[self.joint_qvel]
        return np.concatenate(
            [[data.qpos[2]], gravity, linear_velocity, angular_velocity, joint_angles, joint_speeds, last_action]
        ).astype(np.float32)

    # ------------------------------------------------------------------ reward

    def reward(
        self,
        forward_velocity: float,
        motor_power: float,
        action: np.ndarray,
        last_action: np.ndarray,
        up_z: float,
        fell: bool,
    ) -> tuple[float, dict[str, float]]:
        """Reward for one control step, plus each term separately (for logging)."""
        c = self.config
        terms = {
            "forward": c.forward_weight * min(float(forward_velocity), c.max_reward_speed),
            "upright": c.upright_weight * up_z,
            "energy": -c.energy_weight * motor_power,
            "smoothness": -c.smoothness_weight * float(np.sum((action - last_action) ** 2)),
            "fall": -c.fall_penalty if fell else 0.0,
        }
        return float(sum(terms.values())), terms

    # ---------------------------------------------------------------- episodes

    @staticmethod
    def up_z(data: mujoco.MjData) -> float:
        """z-component of the torso's "up" axis: 1 = level, 0 = on its side, -1 = upside down."""
        return float(data.xmat[1][8])

    def fell(self, data: mujoco.MjData) -> bool:
        too_tilted = self.up_z(data) < self.config.min_up_z
        too_low = data.qpos[2] < self.config.min_height_fraction * self.standing_height
        return bool(too_tilted or too_low)

    def reset_state(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        """Start an episode: standing pose + noise, motors targeting home."""
        mujoco.mj_resetData(self.model, data)
        data.qpos[:] = self.standing_qpos
        data.qvel[:] = self.standing_qvel
        noise = self.config.reset_joint_noise
        data.qpos[self.joint_qpos] += rng.uniform(-noise, noise, self.num_actions)
        data.qvel[:] += rng.uniform(-self.config.reset_velocity_noise, self.config.reset_velocity_noise, self.model.nv)
        data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, data)

    def _settled_standing_state(self, keyframe: str) -> tuple[np.ndarray, np.ndarray]:
        """Drop the robot from its keyframe and let it settle (1.5 s), once.
        Episodes start from this state instead of mid-air."""
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.model.key(keyframe).id)
        for _ in range(round(1.5 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, data)
        return data.qpos.copy(), data.qvel.copy()
