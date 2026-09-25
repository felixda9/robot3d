"""The walk task for N robots at once, as PyTorch tensor math.

This is walk.WalkTask again, batched: every function takes tensors with a
leading "world" dimension (one row per robot) and runs as a handful of GPU
kernels for all robots together. It must compute exactly what WalkTask
computes; tests/test_gpu_task.py compares the two on the same states.
(Float32 here vs float64 there: they agree to ~1e-5.)
"""

import torch

from robot3d.walk import WalkTask

_DOWN = (0.0, 0.0, -1.0)


class BatchedWalkTask:
    def __init__(self, task: WalkTask, device: torch.device | str):
        self.task = task
        self.config = task.config
        self.device = torch.device(device)
        self.decimation = task.decimation
        self.control_dt = task.control_dt
        self.max_steps = task.max_steps
        self.num_actions = task.num_actions
        self.obs_size = task.obs_size
        self.standing_height = task.standing_height

        def tensor(x, dtype=torch.float32):
            return torch.as_tensor(x, dtype=dtype, device=self.device)

        self.home_ctrl = tensor(task.home_ctrl)
        self.ctrl_low = tensor(task.ctrl_low)
        self.ctrl_high = tensor(task.ctrl_high)
        self.joint_qpos = tensor(task.joint_qpos, torch.long)
        self.joint_qvel = tensor(task.joint_qvel, torch.long)
        self.feet = tensor(task.feet, torch.long)
        self.foot_radius = tensor(task.foot_radius)
        self.standing_qpos = tensor(task.standing_qpos)
        self.standing_qvel = tensor(task.standing_qvel)
        pairs = task.diagonal_pairs or [(0, 0)]
        self.pair_a = tensor([a for a, _ in pairs], torch.long)
        self.pair_b = tensor([b for _, b in pairs], torch.long)
        self.has_pairs = bool(task.diagonal_pairs)
        self.clock = task.clock
        self.robot_mass = task.robot_mass
        self.torso_half_size = tensor(task.torso_half_size)
        self.roll_motors = tensor(task.roll_motors, torch.long)
        self.foot_phase_offset = tensor(task.foot_phase_offset, torch.float64)
        if self.config.fallen_start_fraction > 0:
            fallen_qpos, fallen_qvel = task.fallen_states
            self.fallen_qpos, self.fallen_qvel = tensor(fallen_qpos), tensor(fallen_qvel)

    # ----------------------------------------------------------------- actions

    def action_to_ctrl(self, actions: torch.Tensor, joint_pos: torch.Tensor | None = None) -> torch.Tensor:
        """(N, nu) actions in -1..1 -> (N, nu) motor targets (see WalkTask)."""
        base = joint_pos if self.config.action_mode == "relative" else self.home_ctrl
        targets = base + self.config.action_scale * actions.clamp(-1.0, 1.0)
        return torch.maximum(torch.minimum(targets, self.ctrl_high), self.ctrl_low)

    # ------------------------------------------------------------- observation

    def observation(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        torso_rot: torch.Tensor,
        last_action: torch.Tensor,
        phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(N, obs_size) observations. torso_rot: (N, 3, 3) torso-to-world
        rotations; phase: (N,) gait clock (gait_phase())."""
        # R^T @ v for every robot: world directions -> torso frame.
        gravity = torch.einsum("nji,j->ni", torso_rot, torso_rot.new_tensor(_DOWN))
        linear_velocity = torch.einsum("nji,nj->ni", torso_rot, qvel[:, 0:3])
        parts = [
            qpos[:, 2:3],
            gravity,
            linear_velocity,
            qvel[:, 3:6],
            qpos[:, self.joint_qpos] - self.home_ctrl,
            qvel[:, self.joint_qvel],
            last_action,
        ]
        if self.clock:
            angle = 2 * torch.pi * phase
            parts.append(torch.stack([angle.sin(), angle.cos()], dim=1).float())
        return torch.cat(parts, dim=1)

    # -------------------------------------------------------------- gait clock

    def gait_phase(self, steps: torch.Tensor) -> torch.Tensor:
        """(N,) float64 phases (0..1) from (N,) step counts; float64 like WalkTask, so they agree exactly."""
        return torch.remainder(steps.double() * self.task.cycles_per_step, 1.0)

    def desired_down(self, phase: torch.Tensor) -> torch.Tensor:
        """(N, feet) bool: should each foot be on the ground at these phases?"""
        return torch.remainder(phase[:, None] - self.foot_phase_offset, 1.0) < self.config.gait_duty

    # ------------------------------------------------------------------ reward

    def reward(
        self,
        *,
        vx: torch.Tensor,
        vy: torch.Tensor,
        motor_power: torch.Tensor,
        action: torch.Tensor,
        last_action: torch.Tensor,
        up_z: torch.Tensor,
        fell: torch.Tensor,
        foot_slip: torch.Tensor,
        feet_down: torch.Tensor,
        landed: torch.Tensor,
        air_time: torch.Tensor,
        foot_height: torch.Tensor,
        phase: torch.Tensor,
        turn_rate: torch.Tensor,
        height: torch.Tensor,
        joint_offset: torch.Tensor,
        joint_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        succeeded: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """(N,) rewards and (N,) per-term values, same terms as WalkTask.reward.
        feet_down, landed: (N, feet) bool; air_time, foot_height: (N, feet);
        phase: (N,) float64."""
        c = self.config
        upright = up_z.clamp(0.0, 1.0) if c.posture_gating else torch.ones_like(up_z)  # see WalkConfig
        if c.stillness_speed_gate > 0:
            calm = (torch.sqrt(vx**2 + vy**2) < c.stillness_speed_gate).float()
        else:
            calm = torch.ones_like(vx)
        is_up = (up_z > c.upright_gate).float()
        if c.constraint_gate_speed_error > 0:
            balanced = (torch.sqrt((vx - c.target_speed) ** 2 + vy**2) < c.constraint_gate_speed_error).float()
        else:
            balanced = torch.ones_like(vx)
        at_height = (height > c.height_gate * self.standing_height).float()
        if self.clock:
            should_be_down = self.desired_down(phase)
            gait = (feet_down == should_be_down).float().mean(dim=1)
            swinging = (~should_be_down).float()
            lift = (foot_height / c.swing_height).clamp(0.0, 1.0)
            clearance = (lift * swinging).sum(dim=1) / swinging.sum(dim=1).clamp(min=1.0)  # 0 if none swings
        else:
            gait = clearance = torch.zeros_like(vx)
        speed_error_sq = (vx - c.target_speed) ** 2 + vy**2
        extra_air = (air_time - c.air_time_target).clamp(-c.air_time_target, 0.3)
        if self.has_pairs:
            trot = (feet_down[:, self.pair_a] == feet_down[:, self.pair_b]).float().mean(dim=1)
        else:
            trot = torch.zeros_like(vx)
        terms = {
            "tracking": c.tracking_weight * upright * torch.exp(-speed_error_sq / c.tracking_sigma),
            "forward": c.forward_weight * vx.clamp(max=c.max_reward_speed),
            "upright": c.upright_weight * up_z,
            "trot": c.trot_weight * trot,
            "air_time": c.air_time_weight * torch.where(landed, extra_air, torch.zeros_like(extra_air)).sum(dim=1),
            "gait": c.gait_weight * balanced * gait,
            "clearance": c.clearance_weight * balanced * clearance,
            "turn": c.turn_weight * upright * torch.exp(-(turn_rate**2) / c.turn_sigma),
            "energy": -c.energy_weight * motor_power,
            "smoothness": -c.smoothness_weight * ((action - last_action) ** 2).sum(dim=1),
            "slip": -c.slip_weight * foot_slip,
            "support": -c.support_weight * balanced * (feet_down.sum(dim=1) < 2).float(),
            "fall": -c.fall_penalty * fell.float() * float(c.terminate_on_fall),
            "height": c.height_weight * (height / self.standing_height).clamp(0.0, 1.0),
            "pose": c.pose_weight * upright * calm * is_up * torch.exp(-(joint_offset**2).sum(dim=1) / c.pose_sigma),
            "down": -c.down_weight * fell.float(),
            "roll": -c.roll_weight * balanced * (joint_offset[:, self.roll_motors] ** 2).sum(dim=1),
            "success": c.success_bonus * (succeeded.float() if succeeded is not None else torch.zeros_like(vx)),
            "joint_speed": -c.joint_speed_weight * calm * (joint_velocity**2).sum(dim=1),
            "wobble": -c.wobble_weight * calm * (angular_velocity[:, 0] ** 2 + angular_velocity[:, 1] ** 2),
            "orientation": c.orientation_weight * torch.exp(-4.0 * (1.0 - up_z)),
            "hold": c.hold_weight * is_up * at_height * torch.exp(-0.5 * (action**2).sum(dim=1)),
            "stance": c.stance_weight * calm * feet_down.float().mean(dim=1),
        }
        total = torch.stack(list(terms.values())).sum(dim=0)
        return (total.clamp(min=0.0) if c.reward_floor else total), terms

    def air_time_update(self, prev_down: torch.Tensor, air_time: torch.Tensor, down: torch.Tensor):
        """Same as WalkTask.air_time_update, for (N, feet) tensors."""
        landed = down & ~prev_down
        new_air_time = torch.where(down, torch.zeros_like(air_time), air_time + self.control_dt)
        return landed, air_time.clone(), new_air_time

    # ------------------------------------------------------------------- state

    @staticmethod
    def up_z(torso_rot: torch.Tensor) -> torch.Tensor:
        return torso_rot[:, 2, 2]

    def steady(self, qpos: torch.Tensor, torso_rot: torch.Tensor) -> torch.Tensor:
        """Same as WalkTask.steady, batched."""
        return (self.up_z(torso_rot) > 0.94) & (qpos[:, 2] > 0.9 * self.standing_height)

    def fell(self, qpos: torch.Tensor, torso_rot: torch.Tensor) -> torch.Tensor:
        c = self.config
        return (self.up_z(torso_rot) < c.min_up_z) | (qpos[:, 2] < c.min_height_fraction * self.standing_height)

    def feet_state(self, geom_xpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """((N, feet, 2) foot x/y, (N, feet) on the ground). geom_xpos: (N, ngeom, 3)."""
        pos = geom_xpos[:, self.feet]
        on_ground = pos[..., 2] - self.foot_radius < WalkTask.FOOT_CONTACT_MARGIN
        return pos[..., :2].clone(), on_ground

    def heading_velocity(self, torso_rot: torch.Tensor, vx: torch.Tensor, vy: torch.Tensor):
        """Same as WalkTask.heading_velocity, for (N,) velocities and (N, 3, 3) rotations."""
        if self.config.velocity_frame == "world":
            return vx, vy
        heading = torch.atan2(torso_rot[:, 1, 0], torso_rot[:, 0, 0])
        c, s = heading.cos(), heading.sin()
        return c * vx + s * vy, -s * vx + c * vy

    @staticmethod
    def turn_rate(qvel: torch.Tensor) -> torch.Tensor:
        return qvel[:, 5]

    def push_wrench(self, torso_rot, torso_pos, torso_com, angle, size, height):
        """Same as WalkTask.push_wrench, for (N,) angles/sizes/heights: (N, 3) force, torque."""
        direction = torch.stack([angle.cos(), angle.sin(), torch.zeros_like(angle)], dim=1)
        force = direction * (size * self.robot_mass / self.task.PUSH_SECONDS)[:, None]
        local = torch.einsum("nji,nj->ni", torso_rot, direction)  # R^T u
        hx, hy, hz = self.torso_half_size
        reach = 1.0 / torch.maximum(torch.maximum(local[:, 0].abs() / hx, local[:, 1].abs() / hy),
                                    torch.full_like(angle, 1e-9))
        offset = torch.stack([-local[:, 0] * reach, -local[:, 1] * reach, height * hz], dim=1)
        point = torso_pos + torch.einsum("nij,nj->ni", torso_rot, offset)
        return force, torch.cross(point - torso_com, force, dim=1)

    def foot_heights(self, geom_xpos: torch.Tensor) -> torch.Tensor:
        """(N, feet) each foot's lowest point above the floor."""
        return geom_xpos[:, self.feet, 2] - self.foot_radius

    def foot_slip(self, before, after) -> torch.Tensor:
        (xy0, down0), (xy1, down1) = before, after
        speed_sq = ((xy1 - xy0) ** 2).sum(dim=-1) / self.control_dt**2
        return torch.where(down0 & down1, speed_sq, torch.zeros_like(speed_sq)).sum(dim=1)

    def reset_state(self, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """(qpos, qvel) for n fresh episodes: standing pose + the same noise as WalkTask."""
        c = self.config
        qpos = self.standing_qpos.repeat(n, 1)
        qvel = self.standing_qvel.repeat(n, 1)
        joint_noise = torch.rand((n, self.num_actions), generator=generator, device=self.device)
        qpos[:, self.joint_qpos] += (2 * joint_noise - 1) * c.reset_joint_noise
        vel_noise = torch.rand(qvel.shape, generator=generator, device=self.device)
        qvel += (2 * vel_noise - 1) * c.reset_velocity_noise
        if c.fallen_start_fraction > 0:  # some start lying down (WalkTask.fallen_states)
            fallen = torch.rand(n, generator=generator, device=self.device) < c.fallen_start_fraction
            pick = torch.randint(len(self.fallen_qpos), (n,), generator=generator, device=self.device)
            qpos = torch.where(fallen[:, None], self.fallen_qpos[pick], qpos)
            qvel = torch.where(fallen[:, None], self.fallen_qvel[pick], qvel)
        return qpos, qvel
