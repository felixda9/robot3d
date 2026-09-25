"""The "walk forward" task: what a policy observes, how its actions become
motor targets, and how it is rewarded.

Shared by the Gymnasium environment (training, envs.py) and the policy player
in the web server (watching, policy.py), so a policy sees and acts exactly
the same way in both.

Works for any robot that follows the project conventions: body 1 is the
free-floating torso, every motor is a position actuator on a joint, and a
"home" keyframe holds the standing pose's motor targets.
"""

from dataclasses import asdict, dataclass, fields

import math

import mujoco
import numpy as np


@dataclass(frozen=True)
class WalkConfig:
    # --- control
    control_dt: float = 0.02  # the policy acts at 50 Hz (every 10 physics steps of 2 ms)
    action_scale: float = 0.5  # action +-1 = target +-0.5 rad around the home pose
    episode_seconds: float = 20.0  # then the episode is cut off ("truncated")

    # --- reward: one number per control step; the policy learns to make the sum large.
    # The task (since 2026-09-25): a calm trot-walk at a target speed along +x.
    target_speed: float = 0.4  # m/s: a walk for this robot size (it runs above ~0.8 m/s)
    tracking_weight: float = 2.0  # x exp(-speed_error^2 / tracking_sigma): full reward exactly on target
    # (m/s)^2; off by 0.1 m/s -> 90% of the reward, by 0.2 -> 67%, standing still -> 20%.
    # (trot_rsl used 0.25, so loose that it settled at 0.31 m/s for 97%.)
    tracking_sigma: float = 0.1
    support_weight: float = 1.0  # penalty for each step with fewer than 2 feet down (no running/hopping)
    # Which way "forward" is for the speed reward. "body": along the robot's
    # own heading (where its nose points, flattened onto the ground), so the
    # reward only depends on what the policy can sense; it never observes its
    # heading. "world": along world +x (runs before 5c). With "world", the
    # 12-motor walk12_push learned to walk in circles: it couldn't tell why its
    # reward kept changing.
    velocity_frame: str = "body"
    turn_weight: float = 0.5  # x exp(-turn_rate^2 / turn_sigma): walk straight, don't turn
    turn_sigma: float = 0.25  # (rad/s)^2 (legged_gym's value); turning at 10 deg/s -> 88% of it, 30 deg/s -> 33%

    # Gait clock (since trot_clock): a built-in rhythm, like a metronome for the
    # legs. The policy sees the clock's phase and is rewarded when each foot is
    # on the ground exactly when the schedule says: the diagonal pairs FL+RR
    # and FR+RL take turns, each foot down for `gait_duty` of every cycle.
    # Without it, the policy found a loophole (trot_rsl: back feet never
    # lifted, it scooted). gait_frequency 0 = no clock (runs before it existed).
    # Cycles per second: each foot steps 1.5 times a second (~27 cm strides at
    # 0.4 m/s). The user preferred it to 2 Hz (trot_clock_15 vs trot_clock).
    gait_frequency: float = 1.5
    gait_duty: float = 0.6  # share of a cycle each foot is down; > 0.5 = a walk (never airborne)
    gait_weight: float = 1.0  # x share of feet whose contact matches the schedule
    swing_height: float = 0.04  # m: lift swinging feet this high ...
    clearance_weight: float = 0.5  # ... x mean over swinging feet of min(height / swing_height, 1)

    # The trot-walk's first reward (trot_rsl), replaced by the clock:
    trot_weight: float = 0.0  # diagonal feet in the same state (flaw: "all four down" counts too)
    air_time_weight: float = 0.0  # per landing: x (its swing time - air_time_target)
    air_time_target: float = 0.25
    # The first task (walk_10m .. walk_gpu_v5): forward speed, counted up to max_reward_speed.
    forward_weight: float = 0.0
    max_reward_speed: float = 1.0
    upright_weight: float = 0.5  # torso "up" axis . world up: 1 level, 0 on its side
    energy_weight: float = 0.002  # per watt of mechanical motor power |torque x joint speed|
    smoothness_weight: float = 0.05  # per unit of squared action change (discourages jitter)
    slip_weight: float = 0.5  # per (m/s)^2 of feet sliding along the floor while touching it
    fall_penalty: float = 10.0  # once, when the episode ends by falling

    # --- falling (ends the episode)
    min_up_z: float = 0.5  # torso tilted more than 60 degrees
    min_height_fraction: float = 0.5  # or torso lower than half its standing height

    # --- start of each episode: the settled standing pose plus random noise,
    # so the policy learns to cope with slightly different starts
    reset_joint_noise: float = 0.1  # rad
    reset_velocity_noise: float = 0.1  # rad/s and m/s

    # --- random shoves (since 5c, for robustness): every push_interval seconds
    # (randomly 0.5x to 1.5x that), the torso gets a sudden velocity kick,
    # forward/back and sideways each uniform in +-push_max_speed, like being
    # bumped into. The policy isn't told; it feels the stumble and must catch
    # itself. (legged_gym does the same, every 15 s.) 0 = no pushes.
    push_interval: float = 4.0  # s
    push_max_speed: float = 1.0  # m/s

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_run(cls, saved: dict) -> "WalkConfig":
        """Settings from a run's run.json. A setting added after the run was
        trained gets the value that reproduces how it was trained."""
        before_it_existed = {
            "slip_weight": 0.0,
            # the first task: run.json files from then all saved forward_weight=1.0
            "tracking_weight": 0.0,
            "support_weight": 0.0,
            "trot_weight": 0.0,
            "air_time_weight": 0.0,
            "gait_frequency": 0.0,
            "gait_weight": 0.0,
            "clearance_weight": 0.0,
            "push_interval": 0.0,
            "velocity_frame": "world",
            "turn_weight": 0.0,
        }
        unknown = sorted(set(saved) - {f.name for f in fields(cls)})
        if unknown:
            # The run was trained by newer code than this process is running
            # (typically a server started before `git pull` / a code change).
            raise ValueError(
                f"this run uses task settings this code doesn't know ({', '.join(unknown)}). "
                "If the code was updated since the server started, restart it (uv run scripts/serve.py)."
            )
        return cls(**{**before_it_existed, **saved})


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
        # then per motor: joint angle, joint speed, previous action,
        # then with the gait clock: its phase as (sin, cos)
        self.clock = config.gait_frequency > 0
        self.obs_size = 1 + 3 + 3 + 3 + 3 * model.nu + (2 if self.clock else 0)
        # Phase advance per control step. Phases are computed from the integer
        # step count in float64 (here and on the GPU), so both agree exactly,
        # even right at a cycle boundary.
        self.cycles_per_step = self.control_dt * config.gait_frequency

        # Feet (project convention: geoms named "<leg>_foot"), for the slip
        # penalty. A foot counts as on the ground when its lowest point is
        # within FOOT_CONTACT_MARGIN of the floor (spheres: center z - radius).
        self.feet = np.array([g for g in range(model.ngeom) if model.geom(g).name.endswith("_foot")], dtype=int)
        self.foot_radius = model.geom_size[self.feet, 0]
        # Diagonal leg pairs (indices into self.feet) for the trot reward:
        # front-left with rear-right, front-right with rear-left.
        foot_index = {model.geom(g).name.removesuffix("_foot"): i for i, g in enumerate(self.feet)}
        self.diagonal_pairs = [
            (foot_index[a], foot_index[b]) for a, b in (("FL", "RR"), ("FR", "RL")) if a in foot_index and b in foot_index
        ]
        # Gait clock: when in the cycle each foot's stance starts. The first
        # pair (FL+RR) at phase 0, the second (FR+RL) half a cycle later.
        self.foot_phase_offset = np.zeros(len(self.feet))
        if len(self.diagonal_pairs) == 2:
            self.foot_phase_offset[list(self.diagonal_pairs[1])] = 0.5

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

    def observation(self, data: mujoco.MjData, last_action: np.ndarray, phase: float = 0.0) -> np.ndarray:
        """What the policy "feels", like a real robot's IMU and joint encoders
        (plus, with the gait clock, where in the stepping rhythm it is).

        Directions are expressed in the torso's own frame, so the policy
        behaves the same whichever way the robot faces, and absolute x/y
        position is left out (walking works the same anywhere on the floor).

        Called right after mj_step, when MuJoCo's xmat is from the start of
        that step (2 ms behind qpos). Training, the viewer and the GPU version
        all read it at that point, so the policy always sees the same thing.
        """
        torso_rot = data.xmat[1].reshape(3, 3)  # torso frame -> world frame
        world_to_torso = torso_rot.T
        gravity = world_to_torso @ np.array([0.0, 0.0, -1.0])  # "which way is down": tilt, without heading
        linear_velocity = world_to_torso @ data.qvel[0:3]  # free joint: linear velocity is in world frame
        angular_velocity = data.qvel[3:6]  # ... angular velocity already in the torso's frame
        joint_angles = data.qpos[self.joint_qpos] - self.home_ctrl  # relative to the standing pose
        joint_speeds = data.qvel[self.joint_qvel]
        parts = [[data.qpos[2]], gravity, linear_velocity, angular_velocity, joint_angles, joint_speeds, last_action]
        if self.clock:
            # sin/cos rather than the raw phase: 0.99 and 0.01 are neighbors
            # on a circle, and the policy should see them as close.
            parts.append([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------ pushes

    def push_delay(self, uniform: float) -> int:
        """Control steps until the next push, from a uniform(0, 1) random number."""
        return max(1, round(self.config.push_interval * (0.5 + uniform) / self.control_dt))

    # -------------------------------------------------------------- gait clock

    def gait_phase(self, step: int) -> float:
        """Where in the stepping cycle (0..1) the clock is after `step` control steps of an episode."""
        return (step * self.cycles_per_step) % 1.0

    def desired_down(self, phase: float) -> np.ndarray:
        """Per foot: should it be on the ground at this phase?"""
        return (phase - self.foot_phase_offset) % 1.0 < self.config.gait_duty

    # ------------------------------------------------------------------ reward

    def reward(
        self,
        *,
        vx: float,
        vy: float,
        motor_power: float,
        action: np.ndarray,
        last_action: np.ndarray,
        up_z: float,
        fell: bool,
        foot_slip: float,
        feet_down: np.ndarray,
        landed: np.ndarray,
        air_time: np.ndarray,
        foot_height: np.ndarray,
        phase: float,
        turn_rate: float,
    ) -> tuple[float, dict[str, float]]:
        """Reward for one control step, plus each term separately (for logging).

        vx, vy: torso velocity over the step, forward and sideways (see
            heading_velocity()), m/s.
        turn_rate: how fast the torso turns about its own up axis, rad/s.
        foot_slip: sum over planted feet of (sliding speed)^2, see foot_slip().
        feet_down: which feet are on the ground (bool per foot).
        landed, air_time: which feet touched down this step, and how long each
            had been in the air (see air_time_update()).
        foot_height: each foot's lowest point above the floor, m (foot_heights()).
        phase: the gait clock at the end of the step (gait_phase()).
        """
        c = self.config
        gait = clearance = 0.0
        if self.clock:
            should_be_down = self.desired_down(phase)
            gait = float(np.mean(feet_down == should_be_down))
            swinging = ~should_be_down
            if swinging.any():
                clearance = float(np.mean(np.clip(foot_height[swinging] / c.swing_height, 0.0, 1.0)))
        speed_error_sq = (vx - c.target_speed) ** 2 + vy**2  # includes drifting sideways
        extra_air = np.clip(air_time - c.air_time_target, -c.air_time_target, 0.3)  # capped: no endless lifts
        diagonal_sync = [float(feet_down[a] == feet_down[b]) for a, b in self.diagonal_pairs]
        terms = {
            "tracking": c.tracking_weight * float(np.exp(-speed_error_sq / c.tracking_sigma)),
            "forward": c.forward_weight * min(float(vx), c.max_reward_speed),
            "upright": c.upright_weight * up_z,
            "trot": c.trot_weight * (float(np.mean(diagonal_sync)) if diagonal_sync else 0.0),
            "air_time": c.air_time_weight * float(np.sum(np.where(landed, extra_air, 0.0))),
            "gait": c.gait_weight * gait,
            "clearance": c.clearance_weight * clearance,
            "turn": c.turn_weight * math.exp(-(turn_rate**2) / c.turn_sigma),
            "energy": -c.energy_weight * motor_power,
            "smoothness": -c.smoothness_weight * float(np.sum((action - last_action) ** 2)),
            "slip": -c.slip_weight * foot_slip,
            "support": -c.support_weight * float(np.sum(feet_down) < 2),
            "fall": -c.fall_penalty if fell else 0.0,
        }
        return float(sum(terms.values())), terms

    def air_time_update(
        self, prev_down: np.ndarray, air_time: np.ndarray, down: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Swing bookkeeping per foot, once per control step.
        Returns (landed this step, air time it had when landing, new air_time state)."""
        landed = down & ~prev_down
        new_air_time = np.where(down, 0.0, air_time + self.control_dt)
        return landed, air_time.copy(), new_air_time

    # ----------------------------------------------------------------- feet

    FOOT_CONTACT_MARGIN = 0.005  # m

    # -------------------------------------------------------------- motion

    def heading_velocity(self, data: mujoco.MjData, vx: float, vy: float) -> tuple[float, float]:
        """World-frame horizontal velocity -> (forward, sideways) for the reward.

        velocity_frame "body": relative to the robot's heading, the direction
        its torso's x axis points, flattened onto the ground (so tilting
        doesn't change it). "world": unchanged.
        """
        if self.config.velocity_frame == "world":
            return vx, vy
        nose = data.xmat[1].reshape(3, 3)[:, 0]  # torso x axis in world coordinates
        heading = math.atan2(nose[1], nose[0])
        c, s = math.cos(heading), math.sin(heading)
        return c * vx + s * vy, -s * vx + c * vy

    @staticmethod
    def turn_rate(data: mujoco.MjData) -> float:
        """Turning speed about the torso's up axis, rad/s (free joint: qvel[3:6]
        is the angular velocity in the torso's own frame)."""
        return float(data.qvel[5])

    def feet_state(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(x/y position of each foot, whether each foot is on the ground)."""
        pos = data.geom_xpos[self.feet]
        on_ground = pos[:, 2] - self.foot_radius < self.FOOT_CONTACT_MARGIN
        return pos[:, :2].copy(), on_ground

    def foot_heights(self, data: mujoco.MjData) -> np.ndarray:
        """Each foot's lowest point above the floor (m; ~0 when planted)."""
        return data.geom_xpos[self.feet, 2] - self.foot_radius

    def foot_slip(self, before: tuple[np.ndarray, np.ndarray], after: tuple[np.ndarray, np.ndarray]) -> float:
        """Sum of squared sliding speeds (m/s)^2 of feet that stayed on the
        ground for the whole control step. A foot planted on the ground
        should not move; if it does, it is skidding, which wastes effort and
        on a real robot wears the feet and makes the gait unpredictable."""
        (xy0, down0), (xy1, down1) = before, after
        speed = np.linalg.norm(xy1 - xy0, axis=1) / self.control_dt
        return float(np.sum(np.where(down0 & down1, speed**2, 0.0)))

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
