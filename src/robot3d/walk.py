"""The robot's tasks: "walk" (WalkConfig()), "stand" (WalkConfig.stand()) and
"getup" (WalkConfig.getup()). What a policy observes, how its actions become
motor targets, and how it is rewarded. One task class with settings, so every
task shares the observation, the motors, the GPU pipeline and the viewer.

Shared by the Gymnasium environment (training, envs.py) and the policy player
in the web server (watching, policy.py), so a policy sees and acts exactly
the same way in both.

Works for any robot that follows the project conventions: body 1 is the
free-floating torso, every motor is a position actuator on a joint, and a
"home" keyframe holds the standing pose's motor targets.
"""

import math
from dataclasses import asdict, dataclass, fields
from functools import cached_property

import mujoco
import numpy as np


@dataclass(frozen=True)
class WalkConfig:
    # "walk", "stand" (stand still, catch shoves), "getup" (get up after a
    # fall) or "jump" (one high jump on command, then stand).
    task: str = "walk"

    # --- control
    control_dt: float = 0.02  # the policy acts at 50 Hz (every 10 physics steps of 2 ms)
    action_scale: float = 0.5  # action +-1 = target +-0.5 rad (around the home pose, see action_mode)
    # How actions become motor targets. "home": home pose + action_scale x
    # action (walk, stand: action 0 = the standing pose). "relative": each
    # joint's current angle + action_scale x action (getup: action 0 = hold
    # still wherever you are). MuJoCo Playground's Go1/Spot getup and Lee et
    # al. 2019 use relative actions; Playground found home-based ones worse.
    action_mode: str = "home"
    episode_seconds: float = 20.0  # then the episode is cut off ("truncated")

    # --- steering commands (M7): the policy observes a command (forward and
    # sideways speed in its heading frame, turn rate) and is rewarded for
    # following it; without commands it walks at target_speed, straight.
    # Commands are drawn per robot every command_resample_seconds: forward
    # 90% of the time, sideways 30%, turning 50%, each uniform in its range;
    # and command_zero_fraction of them "stand still" (all zero), where the
    # gait rules pause and `still` pays for standing in the home pose on all
    # feet (as Playground/Isaac Lab do at zero command; otherwise a clocked
    # walker marches in place).
    commands: bool = False
    command_max_forward: float = 0.6  # m/s
    command_max_backward: float = 0.3  # m/s
    command_max_sideways: float = 0.3  # m/s
    command_max_turn: float = 1.0  # rad/s
    command_resample_seconds: float = 5.0
    command_zero_fraction: float = 0.1
    still_weight: float = 0.0  # at a zero command: x share of feet down x exp(-|q - home|^2 / 0.1)

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
    # Per rad^2 of sideways leg angle (roll joints, robots that have them):
    # walking straight needs none, so without this penalty walk12_straight
    # walked splayed and lopsided (legs held 16-19 deg out). Small enough to
    # still use roll to catch a shove. (legged_gym setups call it "hip_pos".)
    roll_weight: float = 1.0
    fall_penalty: float = 10.0  # once, when the episode ends by falling (terminate_on_fall)
    # Stillness, for standing (0 while walking): stand12_reach "moved way too
    # much" (the user), with only the torso's drift penalized.
    joint_speed_weight: float = 0.0  # per (rad/s)^2 of joint speed, summed over the motors
    wobble_weight: float = 0.0  # per (rad/s)^2 of torso tipping (angular velocity about its x and y axes)
    # > 0 (m/s): pose, joint_speed and wobble apply only while the torso moves
    # slower than this. Calm when left alone, free to step when shoved (Isaac
    # Lab's Spot config gates at 0.5 m/s). stand12_calm, always penalized for
    # moving, stood still but wouldn't step to catch a shove.
    stillness_speed_gate: float = 0.0

    # Jumping (the jump task; 0 elsewhere). The robot is "airborne" with all
    # feet up; it has "landed" once it touched down after >= 3 airborne steps.
    jump_weight: float = 0.0  # x (torso height - standing height), while airborne before landing
    rejump_weight: float = 0.0  # penalty per airborne step after landing: one jump only
    settle_weight: float = 0.0  # after landing: x share of feet down x exp(-|q - home|^2 / 0.5)

    # x share of feet on the ground, while calm (standing): stand12_v3 rested
    # on three feet, the fourth 49 mm up, since nothing asked for four.
    stance_weight: float = 0.0

    # Getting up (MuJoCo Playground's Go1 getup recipe; 0 elsewhere):
    orientation_weight: float = 0.0  # x exp(-4 (1 - up_z)): 1 level, 0.02 on its side, ~0 on its back
    hold_weight: float = 0.0  # x exp(-0.5 |action|^2), once up (gates below): hold still (relative actions)
    upright_gate: float = -1.0  # pose and hold count only while up_z > this (-1: always)
    height_gate: float = 0.0  # hold counts only while the torso is above this x standing height

    # Clip each step's total reward at 0 (legged_gym's only_positive_rewards,
    # Playground). With penalties making a step's total negative, ending the
    # episode early (falling) looks good: stand12_calm fell in ~85% of its
    # training episodes. (Lee et al. 2019: unbounded costs make the agent
    # "find it more rewarding to terminate".)
    reward_floor: bool = False

    # Posture terms, for standing (0 while walking):
    height_weight: float = 0.0  # x torso height / standing height (capped at 1): be up
    pose_weight: float = 0.0  # x exp(-sum of (joint angle - home angle)^2 / pose_sigma): back in the standing pose
    pose_sigma: float = 1.0  # rad^2
    down_weight: float = 0.0  # penalty for every step spent fallen (with terminate_on_fall off)
    # Pay "stand still" (tracking), "don't turn" and "standing pose" only in
    # proportion to how upright the torso is (0 on its side or back). Without
    # it, stand12 learned that lying motionless on its back with its legs in
    # the standing pose earns 1.3 per step: a trap that made getting up look
    # costly. Only matters when falls don't end the episode.
    posture_gating: bool = False

    # --- falling: tilted more than ~80 degrees (up_z < 0.17), or the torso
    # nearly on the floor (< 30% of standing height). Was 60 deg / 50% until
    # walk12_tidy: the walker never practised saving itself from a deep lean
    # (reference projects end at 70 deg, or only upside down).
    min_up_z: float = 0.17
    min_height_fraction: float = 0.3
    terminate_on_fall: bool = True  # walking: a fall ends the episode. Standing: it goes on; get up!

    # --- start of each episode: the settled standing pose plus random noise,
    # so the policy learns to cope with slightly different starts
    reset_joint_noise: float = 0.1  # rad
    reset_velocity_noise: float = 0.1  # rad/s and m/s

    # --- random shoves (since 5c, for robustness): every push_interval seconds
    # (randomly 0.5x to 1.5x that), the torso gets a sudden velocity kick,
    # forward/back and sideways each uniform in +-push_max_speed, like being
    # bumped into. The policy isn't told; it feels the stumble and must catch
    # itself. (legged_gym does the same, every 15 s.) 0 = no pushes.
    push_interval: float = 2.0  # s (each gap 1-3 s, as mjlab; walk12_tidy: 4 s)
    push_max_speed: float = 1.0  # m/s
    # "circle": a random direction, size uniform up to the max (pure forward
    # or sideways shoves reach it too). "axes" (until walk12_tidy): x and y
    # each uniform in +-max, so straight shoves never exceeded 1 m/s.
    push_direction: str = "circle"
    # "kick": the torso's velocity jumps (legged_gym, Isaac Lab, mjlab).
    # "force": like the viewer's double-click: a horizontal force for
    # PUSH_SECONDS at a point on the torso's side, from mid-height to the top
    # edge, so it also tips the robot (a 60 N push on the upper side edge:
    # 0.8 m/s sideways and 143 deg/s of roll). Its size is still given in m/s,
    # the speed it would give the whole robot (1 m/s ~ 72 N for quadruped12).
    # stand12_v3, trained with kicks, fell to 60 N side pushes in the viewer.
    push_kind: str = "kick"
    push_max_spin: float = 0.5  # rad/s: also kick the torso's roll/pitch/yaw rate by up to this (mjlab: ~0.5)
    # Shove curriculum (GPU training; > 0 = on): each robot's max shove starts
    # at push_max_speed, grows by push_curriculum_step every episode it
    # survives, shrinks by it after a fall, up to this.
    push_curriculum_max: float = 3.0
    push_curriculum_step: float = 0.25  # (1 -> 3 m/s takes 8 survived episodes)
    # > 0 (m/s): gait, clearance, support and roll terms apply only while the
    # speed is within this of the target, so a shoved robot may step however
    # it needs to catch itself; they return once it's back on track.
    constraint_gate_speed_error: float = 0.5

    # --- starting fallen (get-up task): this share of episodes starts from
    # a random fallen pose (on its side, back, belly...).
    fallen_start_fraction: float = 0.0

    # --- success (get-up task): standing steady (WalkTask.steady) for
    # success_seconds ends the episode with success_bonus. 0 = off.
    success_bonus: float = 0.0
    success_seconds: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    def for_robot(self, robot: str) -> "WalkConfig":
        """These settings adjusted for a robot (ROBOT_SETTINGS), e.g. the
        humanoid: a biped needs its hip/ankle roll joints to balance, so no
        roll penalty, and it counts as fallen sooner than a quadruped."""
        from dataclasses import replace

        return replace(self, **ROBOT_SETTINGS.get(robot, {}).get(self.task, {}))

    @classmethod
    def stand(cls) -> "WalkConfig":
        """The stand task: stand still in the home pose; when shoved, catch
        yourself (with a step if needed) and settle back into it. A fall ends
        the episode: getting up is the get-up policy's job (getup())."""
        return cls(
            task="stand",
            target_speed=0.0,  # the speed reward pays for not drifting
            tracking_weight=1.0,
            gait_frequency=0.0,  # no stepping rhythm: step only when needed
            gait_weight=0.0,
            clearance_weight=0.0,
            upright_weight=1.0,
            pose_weight=1.0,
            pose_sigma=0.1,  # tight: 0.05 rad off on every joint keeps 74% of it, 0.1 rad 30%
            joint_speed_weight=0.01,  # legs still (1 rad/s on all 12 joints: -0.12 per step)
            wobble_weight=0.5,  # torso still (tipping at 1 rad/s: -0.5 per step)
            stillness_speed_gate=0.5,  # ... but only while not being shoved along
            smoothness_weight=0.1,
            roll_weight=0.0,  # the pose term covers the roll joints too
            reward_floor=True,
            push_kind="force",  # pushed the way the viewer pushes (see push_kind)
            push_max_speed=1.0,  # (curriculum up to push_curriculum_max = 3 m/s, ~216 N)
            push_max_spin=0.0,  # force pushes tip the robot by themselves
            stance_weight=1.0,
        )

    @classmethod
    def steer(cls) -> "WalkConfig":
        """Walking that follows steering commands (M7): forward/back, sideways
        and turning, or standing still at a zero command. Task "walk"."""
        return cls(commands=True, still_weight=1.0, turn_weight=1.0)

    @classmethod
    def jump(cls) -> "WalkConfig":
        """The jump task (5c, the user's "jump on command"): from standing,
        one jump as high as it can, land on its feet, settle into the
        standing pose. An episode is one jump (3 s); in the viewer, pressing
        J starts one. The clock (one cycle per episode) tells the policy how
        far into the jump it is."""
        return cls(
            task="jump",
            episode_seconds=3.0,
            gait_frequency=1.0 / 3.0,  # the clock input: 0..1 over the jump
            gait_weight=0.0,
            clearance_weight=0.0,
            target_speed=0.0,
            tracking_weight=0.5,  # jump in place
            upright_weight=0.0,
            orientation_weight=1.0,  # stay level, in the air too
            jump_weight=50.0,  # a 10 cm high jump: ~20 over its ~0.3 s flight
            rejump_weight=1.0,
            settle_weight=1.0,  # can only be earned after jumping
            support_weight=0.0,  # all feet up is the point
            slip_weight=0.0,
            roll_weight=0.0,
            energy_weight=0.0005,
            smoothness_weight=0.01,
            action_scale=1.0,  # deep crouch, full extension
            push_interval=0.0,
            reward_floor=True,
            min_up_z=0.5,  # a fall: tilted past 60 deg or below half height
            min_height_fraction=0.5,
        )

    @classmethod
    def getup(cls) -> "WalkConfig":
        """The get-up task (5c), MuJoCo Playground's Go1 getup recipe: from
        lying on its side, back or belly, get up and hold still. It only drives
        after a fall; the stand or walk policy takes over once it's steady.

        - Relative actions (+-0.5 rad from the current angles): big movements
          add up over steps; action 0 = hold still.
        - 60% of episodes start fallen, 40% standing (so holding still once up
          is learned too); fixed 6 s episodes. No success ending: with
          positive rewards for standing, ending at success would throw away
          the reward that follows it, punishing success.
        - Rewards: orientation (level), height, standing pose once upright
          (< ~10 deg), holding still once also at full height. Totals clipped
          at 0. Lying still earns ~0.

        (Earlier: stand12* combined standing and getting up; stand12_reach
        collapsed into lying on its belly. getup12 ended episodes on success.)"""
        return cls(
            task="getup",
            target_speed=0.0,
            tracking_weight=0.0,  # standing still is the stand policy's job
            turn_weight=0.0,
            gait_frequency=0.0,
            gait_weight=0.0,
            clearance_weight=0.0,
            support_weight=0.0,
            action_mode="relative",
            action_scale=0.5,
            upright_weight=0.0,
            orientation_weight=1.0,
            height_weight=1.0,
            pose_weight=1.0,
            pose_sigma=2.0,  # exp(-0.5 |q - home|^2), as Playground
            upright_gate=0.985,  # ~10 deg
            hold_weight=1.0,
            height_gate=0.95,
            energy_weight=0.0005,
            smoothness_weight=0.01,
            slip_weight=0.0,
            roll_weight=0.0,  # getting up needs the roll joints freely
            fall_penalty=0.0,
            terminate_on_fall=False,
            # "fallen" here (its statistics, and which starts count as fallen):
            # tilted past 60 deg or below half height, as before.
            min_up_z=0.5,
            min_height_fraction=0.5,
            fallen_start_fraction=0.6,
            push_interval=0.0,
            episode_seconds=6.0,
            reward_floor=True,
        )

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
            "roll_weight": 0.0,
            "push_direction": "axes",
            "push_max_spin": 0.0,
            "push_curriculum_max": 0.0,
            "constraint_gate_speed_error": 0.0,
        }
        if "task" not in saved:  # before the task field: the stand12* runs were get-up runs
            saved = {**saved, "task": "getup" if saved.get("terminate_on_fall") is False else "walk"}
        unknown = sorted(set(saved) - {f.name for f in fields(cls)})
        if unknown:
            # The run was trained by newer code than this process is running
            # (typically a server started before `git pull` / a code change).
            raise ValueError(
                f"this run uses task settings this code doesn't know ({', '.join(unknown)}). "
                "If the code was updated since the server started, restart it (uv run scripts/serve.py)."
            )
        return cls(**{**before_it_existed, **saved})


# Per robot and task: settings that differ from the defaults (WalkConfig.for_robot).
ROBOT_SETTINGS: dict[str, dict[str, dict]] = {
    "humanoid": {
        "walk": dict(
            roll_weight=0.0,  # hip and ankle roll shift the weight between the feet: needed
            swing_height=0.05,
            min_up_z=0.5,  # a biped is down by 60 deg of tilt ...
            min_height_fraction=0.6,  # ... or with its pelvis below 60% of standing height (kneeling)
            push_max_speed=0.5,  # gentler shoves to start (a 19 kg biped), growing to 1.5 m/s
            push_curriculum_max=1.5,
        ),
    },
}


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
        # ... and with commands: the command (forward, sideways, turn)
        self.obs_size = 1 + 3 + 3 + 3 + 3 * model.nu + (2 if self.clock else 0) + (3 if config.commands else 0)
        # Phase advance per control step. Phases are computed from the integer
        # step count in float64 (here and on the GPU), so both agree exactly,
        # even right at a cycle boundary.
        self.cycles_per_step = self.control_dt * config.gait_frequency

        # Feet (project convention: geoms named "<leg>_foot"), for the slip
        # penalty. A foot counts as on the ground when its lowest point is
        # within FOOT_CONTACT_MARGIN of the floor (spheres: center z - radius).
        self.feet = np.array([g for g in range(model.ngeom) if model.geom(g).name.endswith("_foot")], dtype=int)
        # From a foot geom's center down to its sole: the radius of a sphere
        # foot (quadrupeds), the half-height of a box foot (the humanoid).
        is_box = model.geom_type[self.feet] == mujoco.mjtGeom.mjGEOM_BOX
        self.foot_radius = np.where(is_box, model.geom_size[self.feet, 2], model.geom_size[self.feet, 0])
        # "support": at least half the feet should be down (quadruped 2, biped 1).
        self.min_feet_down = max(1, len(self.feet) // 2)
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
        elif "R" in foot_index:  # a biped: left and right take turns
            self.foot_phase_offset[foot_index["R"]] = 0.5

        self.standing_qpos, self.standing_qvel = self._settled_standing_state(keyframe)
        self.standing_height = float(self.standing_qpos[2])
        self.robot_mass = float(model.body_subtreemass[1])
        # Pushes land on the root body (1) at its first geom: the quadrupeds'
        # torso box, the humanoid's pelvis box. Its half-size:
        root_geom = model.body_geomadr[1]
        self.torso_half_size = (model.geom_size[root_geom].copy()
                                if model.geom_type[root_geom] == mujoco.mjtGeom.mjGEOM_BOX else np.array([0.1, 0.1, 0.05]))
        joints = model.actuator_trnid[:, 0]
        self.joint_range = model.jnt_range[joints].copy()  # (motors, 2) min/max angle
        # Sideways (roll) motors, by the naming convention "<leg>_roll" (quadruped12).
        self.roll_motors = np.array([i for i in range(model.nu) if model.actuator(i).name.endswith("_roll")], dtype=int)

    # ----------------------------------------------------------------- actions

    def action_to_ctrl(self, action: np.ndarray, joint_pos: np.ndarray | None = None) -> np.ndarray:
        """Policy action (each in -1..1) -> motor target angles.

        action_mode "home": offsets around the standing pose (action 0 = stand
        there). Starting from a sensible pose makes learning much faster than
        letting the policy pick raw angles from the whole joint range.
        "relative": offsets from the joints' current angles (`joint_pos`).
        """
        action = np.clip(action, -1.0, 1.0)
        base = joint_pos if self.config.action_mode == "relative" else self.home_ctrl
        return np.clip(base + self.config.action_scale * action, self.ctrl_low, self.ctrl_high)

    # ------------------------------------------------------------- observation

    def observation(self, data: mujoco.MjData, last_action: np.ndarray, phase: float = 0.0,
                    command: np.ndarray | None = None) -> np.ndarray:
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
        if self.config.commands:
            parts.append(np.zeros(3) if command is None else command)
        return np.concatenate(parts).astype(np.float32)

    # ---------------------------------------------------------------- commands

    def default_command(self) -> np.ndarray:
        """(forward, sideways, turn) when there are no commands: walk straight at target_speed."""
        return np.array([self.config.target_speed, 0.0, 0.0])

    def sample_command(self, rng: np.random.Generator) -> np.ndarray:
        """A random training command (see WalkConfig.commands)."""
        c = self.config
        if rng.uniform() < c.command_zero_fraction:
            return np.zeros(3)
        forward = rng.uniform(-c.command_max_backward, c.command_max_forward) if rng.uniform() < 0.9 else 0.0
        sideways = rng.uniform(-c.command_max_sideways, c.command_max_sideways) if rng.uniform() < 0.3 else 0.0
        turn = rng.uniform(-c.command_max_turn, c.command_max_turn) if rng.uniform() < 0.5 else 0.0
        return np.array([forward, sideways, turn])

    def clamp_command(self, command) -> np.ndarray:
        """A command limited to what the policy was trained for."""
        c = self.config
        return np.array([
            min(max(command[0], -c.command_max_backward), c.command_max_forward),
            min(max(command[1], -c.command_max_sideways), c.command_max_sideways),
            min(max(command[2], -c.command_max_turn), c.command_max_turn),
        ])

    @staticmethod
    def is_moving_command(command: np.ndarray) -> bool:
        return bool(np.abs(command).max() > 0.05)

    # ------------------------------------------------------------------ pushes

    PUSH_SECONDS = 0.1  # a force push lasts this long (as the viewer's, Simulation.PUSH_SECONDS)

    def push_wrench(self, torso_rot: np.ndarray, torso_pos: np.ndarray, torso_com: np.ndarray,
                    angle: float, size: float, height: float) -> tuple[np.ndarray, np.ndarray]:
        """(force, torque about the torso's center of mass) of a viewer-like
        push: horizontal in world direction `angle`, `size` m/s of impulse for
        the whole robot spread over PUSH_SECONDS, landing on the torso's side
        that faces the pusher, `height` (-1..1) of its half-height up."""
        direction = np.array([math.cos(angle), math.sin(angle), 0.0])
        force = direction * size * self.robot_mass / self.PUSH_SECONDS
        local = torso_rot.T @ direction  # push direction in the torso's frame
        hx, hy, hz = self.torso_half_size
        reach = 1.0 / max(abs(local[0]) / hx, abs(local[1]) / hy, 1e-9)  # to the box's side, horizontally
        point = torso_pos + torso_rot @ np.array([-local[0] * reach, -local[1] * reach, height * hz])
        return force, np.cross(point - torso_com, force)

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
        height: float,
        joint_offset: np.ndarray,
        joint_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        succeeded: bool = False,
        jump_airborne: bool = False,
        jump_landed: bool = False,
        command: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Reward for one control step, plus each term separately (for logging).

        vx, vy: torso velocity over the step, forward and sideways (see
            heading_velocity()), m/s.
        turn_rate: how fast the torso turns about its own up axis, rad/s.
        height: torso height, m. joint_offset: joint angles - home pose, rad.
        joint_velocity: joint speeds, rad/s. angular_velocity: the torso's, in
            its own frame (x, y: tipping; z: turning), rad/s.
        foot_slip: sum over planted feet of (sliding speed)^2, see foot_slip().
        feet_down: which feet are on the ground (bool per foot).
        landed, air_time: which feet touched down this step, and how long each
            had been in the air (see air_time_update()).
        foot_height: each foot's lowest point above the floor, m (foot_heights()).
        phase: the gait clock at the end of the step (gait_phase()).
        """
        c = self.config
        command = self.default_command() if command is None else command
        cmd_vx, cmd_vy, cmd_turn = float(command[0]), float(command[1]), float(command[2])
        moving = float(not c.commands or self.is_moving_command(command))
        upright = min(max(up_z, 0.0), 1.0) if c.posture_gating else 1.0  # see WalkConfig.posture_gating
        # constraint_gate_speed_error: gait rules pause while knocked off speed
        balanced = float(c.constraint_gate_speed_error <= 0
                         or math.hypot(vx - cmd_vx, vy - cmd_vy) < c.constraint_gate_speed_error)
        # stillness_speed_gate: calm terms only while not being shoved along
        calm = float(c.stillness_speed_gate <= 0 or math.hypot(vx, vy) < c.stillness_speed_gate)
        is_up = float(up_z > c.upright_gate)
        at_height = float(height > c.height_gate * self.standing_height)
        gait = clearance = 0.0
        if self.clock:
            should_be_down = self.desired_down(phase)
            gait = float(np.mean(feet_down == should_be_down))
            swinging = ~should_be_down
            if swinging.any():
                clearance = float(np.mean(np.clip(foot_height[swinging] / c.swing_height, 0.0, 1.0)))
        speed_error_sq = (vx - cmd_vx) ** 2 + (vy - cmd_vy) ** 2  # includes drifting sideways
        extra_air = np.clip(air_time - c.air_time_target, -c.air_time_target, 0.3)  # capped: no endless lifts
        diagonal_sync = [float(feet_down[a] == feet_down[b]) for a, b in self.diagonal_pairs]
        terms = {
            "tracking": c.tracking_weight * upright * float(np.exp(-speed_error_sq / c.tracking_sigma)),
            "forward": c.forward_weight * min(float(vx), c.max_reward_speed),
            "upright": c.upright_weight * up_z,
            "trot": c.trot_weight * (float(np.mean(diagonal_sync)) if diagonal_sync else 0.0),
            "air_time": c.air_time_weight * float(np.sum(np.where(landed, extra_air, 0.0))),
            "gait": c.gait_weight * balanced * moving * gait,
            "clearance": c.clearance_weight * balanced * moving * clearance,
            "turn": c.turn_weight * upright * math.exp(-((turn_rate - cmd_turn) ** 2) / c.turn_sigma),
            "energy": -c.energy_weight * motor_power,
            "smoothness": -c.smoothness_weight * float(np.sum((action - last_action) ** 2)),
            "slip": -c.slip_weight * foot_slip,
            "support": -c.support_weight * balanced * float(np.sum(feet_down) < self.min_feet_down),
            "fall": -c.fall_penalty if fell and c.terminate_on_fall else 0.0,
            "height": c.height_weight * min(max(height / self.standing_height, 0.0), 1.0),
            "pose": c.pose_weight * upright * calm * is_up * math.exp(-float(np.sum(joint_offset**2)) / c.pose_sigma),
            "down": -c.down_weight * float(fell),
            "roll": -c.roll_weight * balanced * float(np.sum(joint_offset[self.roll_motors] ** 2)),
            "success": c.success_bonus * float(succeeded),
            "joint_speed": -c.joint_speed_weight * calm * float(np.sum(joint_velocity**2)),
            "wobble": -c.wobble_weight * calm * float(angular_velocity[0] ** 2 + angular_velocity[1] ** 2),
            "orientation": c.orientation_weight * math.exp(-4.0 * (1.0 - up_z)),
            "hold": c.hold_weight * is_up * at_height * math.exp(-0.5 * float(np.sum(action**2))),
            "stance": c.stance_weight * calm * float(np.mean(feet_down)),
            "still": c.still_weight * (1.0 - moving) * float(np.mean(feet_down))
            * math.exp(-float(np.sum(joint_offset**2)) / 0.1),
            "jump": c.jump_weight * float(jump_airborne and not jump_landed) * max(height - self.standing_height, 0.0),
            "rejump": -c.rejump_weight * float(jump_airborne and jump_landed),
            "settle": c.settle_weight * float(jump_landed) * float(np.mean(feet_down))
            * math.exp(-float(np.sum(joint_offset**2)) / 0.5),
        }
        total = float(sum(terms.values()))
        return (max(total, 0.0) if c.reward_floor else total), terms

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

    def distance(self, displacement: np.ndarray) -> float:
        """How far an episode got, from its (x, y) displacement: straight-line
        distance for heading-frame tasks (after a shove turns it, the robot
        walks on in its new direction), progress along +x for older runs."""
        if self.config.velocity_frame == "world":
            return float(displacement[0])
        return float(np.linalg.norm(displacement))

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

    LANDING_AIR_STEPS = 3  # airborne this many control steps (60 ms) before a touchdown counts as landing

    def jump_update(self, flight_steps: int, landed: bool, feet_down: np.ndarray) -> tuple[bool, int, bool]:
        """Jump bookkeeping per control step: (airborne now, new flight_steps, landed)."""
        airborne = not feet_down.any()
        if not airborne and flight_steps >= self.LANDING_AIR_STEPS:
            landed = True
        return airborne, (flight_steps + 1 if airborne else 0), landed

    def steady(self, data: mujoco.MjData) -> bool:
        """Standing up properly: torso level within 20 deg and at >= 90% of
        standing height (Lee et al. 2019's hand-off, scaled to this robot).
        Held for 0.5 s, the viewer hands back from the get-up policy to
        walking/standing."""
        return self.up_z(data) > 0.94 and data.qpos[2] > 0.9 * self.standing_height

    @property
    def success_steps(self) -> int:
        return round(self.config.success_seconds / self.control_dt)

    def fell(self, data: mujoco.MjData) -> bool:
        too_tilted = self.up_z(data) < self.config.min_up_z
        too_low = data.qpos[2] < self.config.min_height_fraction * self.standing_height
        return bool(too_tilted or too_low)

    def reset_state(self, data: mujoco.MjData, rng: np.random.Generator, allow_fallen: bool = True) -> None:
        """Start an episode: standing pose + noise, motors targeting home. With
        fallen_start_fraction, some episodes start lying down instead (not
        when allow_fallen is off, e.g. in the viewer)."""
        mujoco.mj_resetData(self.model, data)
        c = self.config
        if allow_fallen and c.fallen_start_fraction > 0 and rng.uniform() < c.fallen_start_fraction:
            qpos, qvel = self.fallen_states
            i = rng.integers(len(qpos))
            data.qpos[:] = qpos[i]
            data.qvel[:] = qvel[i]
        else:
            data.qpos[:] = self.standing_qpos
            data.qvel[:] = self.standing_qvel
            noise = c.reset_joint_noise
            data.qpos[self.joint_qpos] += rng.uniform(-noise, noise, self.num_actions)
            data.qvel[:] += rng.uniform(-c.reset_velocity_noise, c.reset_velocity_noise, self.model.nv)
        data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, data)

    FALLEN_STATES = 128

    @cached_property
    def fallen_states(self) -> tuple[np.ndarray, np.ndarray]:
        """(qpos, qvel) stacks of FALLEN_STATES ways to lie on the floor, made
        once, on first use (~1 s): drop the robot from 0.5 m at a random
        orientation with its joints held at random angles, and let it settle
        for a second. Most land on a side, back or belly; some on their feet."""
        rng = np.random.default_rng(0)
        model = self.model
        data = mujoco.MjData(model)
        qpos, qvel = [], []
        low, high = self.joint_range[:, 0], self.joint_range[:, 1]
        for _ in range(self.FALLEN_STATES):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = self.standing_qpos
            data.qpos[2] = max(0.5, 1.5 * self.standing_height)  # high enough for any orientation (humanoid: 0.8 m)
            orientation = rng.normal(size=4)  # a uniformly random rotation (normalized 4D Gaussian)
            data.qpos[3:7] = orientation / np.linalg.norm(orientation)
            angles = rng.uniform(low, high)
            data.qpos[self.joint_qpos] = angles
            data.ctrl[:] = np.clip(angles, self.ctrl_low, self.ctrl_high)
            for _ in range(round(1.0 / model.opt.timestep)):
                mujoco.mj_step(model, data)
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
        return np.array(qpos), np.array(qvel)

    def _settled_standing_state(self, keyframe: str) -> tuple[np.ndarray, np.ndarray]:
        """Drop the robot from its keyframe and let it settle (1.5 s), once.
        Episodes start from this state instead of mid-air."""
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.model.key(keyframe).id)
        for _ in range(round(1.5 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, data)
        return data.qpos.copy(), data.qvel.copy()
