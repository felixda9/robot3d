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

    # ----------------------------------------------------------------- actions

    def action_to_ctrl(self, actions: torch.Tensor) -> torch.Tensor:
        """(N, nu) actions in -1..1 -> (N, nu) motor targets (see WalkTask)."""
        targets = self.home_ctrl + self.config.action_scale * actions.clamp(-1.0, 1.0)
        return torch.maximum(torch.minimum(targets, self.ctrl_high), self.ctrl_low)

    # ------------------------------------------------------------- observation

    def observation(
        self, qpos: torch.Tensor, qvel: torch.Tensor, torso_rot: torch.Tensor, last_action: torch.Tensor
    ) -> torch.Tensor:
        """(N, obs_size) observations. torso_rot: (N, 3, 3) torso-to-world rotations."""
        # R^T @ v for every robot: world directions -> torso frame.
        gravity = torch.einsum("nji,j->ni", torso_rot, torso_rot.new_tensor(_DOWN))
        linear_velocity = torch.einsum("nji,nj->ni", torso_rot, qvel[:, 0:3])
        return torch.cat(
            [
                qpos[:, 2:3],
                gravity,
                linear_velocity,
                qvel[:, 3:6],
                qpos[:, self.joint_qpos] - self.home_ctrl,
                qvel[:, self.joint_qvel],
                last_action,
            ],
            dim=1,
        )

    # ------------------------------------------------------------------ reward

    def reward(
        self,
        forward_velocity: torch.Tensor,
        motor_power: torch.Tensor,
        action: torch.Tensor,
        last_action: torch.Tensor,
        up_z: torch.Tensor,
        fell: torch.Tensor,
        foot_slip: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """(N,) rewards and (N,) per-term values, same terms as WalkTask.reward."""
        c = self.config
        terms = {
            "forward": c.forward_weight * forward_velocity.clamp(max=c.max_reward_speed),
            "upright": c.upright_weight * up_z,
            "energy": -c.energy_weight * motor_power,
            "smoothness": -c.smoothness_weight * ((action - last_action) ** 2).sum(dim=1),
            "slip": -c.slip_weight * foot_slip,
            "fall": -c.fall_penalty * fell.float(),
        }
        return torch.stack(list(terms.values())).sum(dim=0), terms

    # ------------------------------------------------------------------- state

    @staticmethod
    def up_z(torso_rot: torch.Tensor) -> torch.Tensor:
        return torso_rot[:, 2, 2]

    def fell(self, qpos: torch.Tensor, torso_rot: torch.Tensor) -> torch.Tensor:
        c = self.config
        return (self.up_z(torso_rot) < c.min_up_z) | (qpos[:, 2] < c.min_height_fraction * self.standing_height)

    def feet_state(self, geom_xpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """((N, feet, 2) foot x/y, (N, feet) on the ground). geom_xpos: (N, ngeom, 3)."""
        pos = geom_xpos[:, self.feet]
        on_ground = pos[..., 2] - self.foot_radius < WalkTask.FOOT_CONTACT_MARGIN
        return pos[..., :2].clone(), on_ground

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
        return qpos, qvel
