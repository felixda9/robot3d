"""N robots stepped at once on the GPU: MuJoCo Warp physics + the batched walk task.

The same walk task as envs.WalkEnv (the same MJCF, observation, action,
reward, and episode rules), but for thousands of robots in parallel, all
data staying on the GPU:
  * MuJoCo Warp keeps every robot's state in (N, ...) GPU arrays.
  * wp.to_torch() gives PyTorch views of those same arrays (no copying),
    so the task math and the policy read and write them directly.
  * The 10 physics steps per policy step are recorded once as a CUDA graph
    and replayed each step, which removes most per-kernel launch overhead.
  * Robots that fall or finish their 20 s restart on the spot ("auto-reset").
"""

from contextlib import contextmanager
from dataclasses import dataclass

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from robot3d.gpu.task import BatchedWalkTask
from robot3d.robots import load_model
from robot3d.walk import WalkConfig, WalkTask


@wp.kernel
def _accumulate_power(
    force: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    dof: wp.array(dtype=int),
    power: wp.array(dtype=float),
):
    """power[world] += sum over motors of |torque x joint speed| (one physics step)."""
    world, motor = wp.tid()
    wp.atomic_add(power, world, wp.abs(force[world, motor] * qvel[world, dof[motor]]))


@dataclass
class StepResult:
    obs: torch.Tensor  # (N, obs) observations after the step (after auto-reset for finished robots)
    reward: torch.Tensor  # (N,)
    done: torch.Tensor  # (N,) episode ended (fell or time limit)
    time_out: torch.Tensor  # (N,) ended by the time limit, not by falling
    terms: dict[str, torch.Tensor]  # (N,) each reward term
    # Finished episodes this step (only the robots where done is True):
    episode_return: torch.Tensor
    episode_length: torch.Tensor
    episode_distance: torch.Tensor
    episode_fell: torch.Tensor


class GpuWalkEnv:
    def __init__(
        self,
        num_envs: int = 4096,
        robot: str = "quadruped",
        config: WalkConfig | None = None,
        device: str = "cuda:0",
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.robot = robot
        self.device = torch.device(device)
        if self.device.type == "cuda":
            # CUDA graphs can't be recorded on PyTorch's default ("legacy")
            # stream, so give this process a real stream and make it
            # PyTorch's current one: PyTorch, Warp and the recorded physics
            # graph then all run in order on the same stream.
            torch.cuda.synchronize(self.device)
            self.stream = torch.cuda.Stream(device=self.device)
            torch.cuda.set_stream(self.stream)
        self.mjm = load_model(robot)
        self.task = BatchedWalkTask(WalkTask(self.mjm, config or WalkConfig()), self.device)
        self.num_obs = self.task.obs_size
        self.num_actions = self.task.num_actions

        wp.init()
        self.wp_device = wp.get_device(device)
        mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, mjd)
        with wp.ScopedDevice(self.wp_device):
            self.m = mjw.put_model(self.mjm)
            # Contact/constraint buffers per world: standing = 4 foot contacts;
            # a robot lying on the floor touches it with torso and legs.
            self.d = mjw.put_data(self.mjm, mjd, nworld=num_envs, nconmax=32, njmax=128)
            self._power = wp.zeros(num_envs, dtype=float)
            self._dof = wp.array(self.task.task.joint_qvel.astype("int32"), dtype=int)
            self._reset_mask = wp.zeros(num_envs, dtype=bool)

        # PyTorch views of MuJoCo Warp's arrays (shared memory, no copies).
        d = self.d
        self.qpos = wp.to_torch(d.qpos)  # (N, nq)
        self.qvel = wp.to_torch(d.qvel)  # (N, nv)
        self.ctrl = wp.to_torch(d.ctrl)  # (N, nu)
        self.xmat = wp.to_torch(d.xmat)  # (N, nbody, 3, 3)
        self.geom_xpos = wp.to_torch(d.geom_xpos)  # (N, ngeom, 3)
        self.power = wp.to_torch(self._power)  # (N,)
        self.reset_mask = wp.to_torch(self._reset_mask)  # (N,)

        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.last_action = torch.zeros((num_envs, self.num_actions), device=self.device)
        self.episode_length = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(num_envs, device=self.device)
        self.start_xy = torch.zeros((num_envs, 2), device=self.device)
        self.next_push = torch.zeros(num_envs, dtype=torch.long, device=self.device)  # episode step of the next shove
        # Foot x/y and on-ground at the end of the last step (for slip, and
        # as "was down" for landings), and each foot's time in the air.
        self._feet_before: tuple[torch.Tensor, torch.Tensor] | None = None
        self.air_time = torch.zeros((num_envs, len(self.task.feet)), device=self.device)
        self._graph = None

    # -------------------------------------------------------------- stepping

    def _physics(self) -> None:
        """One policy step of physics: `decimation` MuJoCo steps + power bookkeeping."""
        for _ in range(self.task.decimation):
            mjw.step(self.m, self.d)
            wp.launch(
                _accumulate_power,
                dim=(self.num_envs, self.num_actions),
                inputs=[self.d.actuator_force, self.d.qvel, self._dof, self._power],
            )

    @contextmanager
    def _warp(self):
        """Run Warp work on our device and, on a GPU, on PyTorch's CUDA stream:
        Warp and PyTorch must share a stream, or a kernel from one could run
        before the other has finished writing its input."""
        with wp.ScopedDevice(self.wp_device):
            if self.wp_device.is_cuda:
                with wp.ScopedStream(wp.stream_from_torch(self.device)):
                    yield
            else:
                yield

    def _run_physics(self) -> None:
        with self._warp():
            if not self.wp_device.is_cuda:
                self._physics()
            elif self._graph is None:
                # First call: run normally (this compiles the kernels), then
                # record the same work as a graph for every later call.
                # Recording doesn't execute anything.
                self._physics()
                with wp.ScopedCapture() as capture:
                    self._physics()
                self._graph = capture.graph
            else:
                wp.capture_launch(self._graph)

    def reset(self, randomize_episode_start: bool = True) -> torch.Tensor:
        """Start every robot on a fresh episode; returns (N, obs) observations.

        randomize_episode_start: pretend each robot is already somewhere into
        its first episode (as legged_gym does). Otherwise all N robots would
        reach the 20 s limit at the same moment, and until then the only
        finished episodes would be falls (making "falls" read 100%)."""
        everyone = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_robots(everyone)
        if randomize_episode_start:
            self.episode_length = torch.randint(
                0, self.task.max_steps, (self.num_envs,), generator=self.generator, device=self.device
            )
            self.next_push = self.episode_length + self._push_delays(self.num_envs)
        self._feet_before = self.task.feet_state(self.geom_xpos)
        return self.observe()

    def observe(self) -> torch.Tensor:
        phase = self.task.gait_phase(self.episode_length)
        return self.task.observation(self.qpos, self.qvel, self.xmat[:, 1], self.last_action, phase)

    def step(self, actions: torch.Tensor) -> StepResult:
        task = self.task
        actions = actions.clamp(-1.0, 1.0)
        self.ctrl.copy_(task.action_to_ctrl(actions))
        c = task.config
        if c.push_interval > 0:
            # Shoves (see WalkConfig.push_interval), without a GPU->CPU sync:
            # every robot draws a kick, only those due get it.
            due = self.episode_length >= self.next_push
            kick = (2 * torch.rand((self.num_envs, 2), generator=self.generator, device=self.device) - 1)
            self.qvel[:, 0:2] += kick * c.push_max_speed * due[:, None]
            self.next_push = torch.where(due, self.episode_length + self._push_delays(self.num_envs), self.next_push)
        x_before = self.qpos[:, 0].clone()
        y_before = self.qpos[:, 1].clone()
        self.power.zero_()

        self._run_physics()

        power = self.power / task.decimation
        vx = (self.qpos[:, 0] - x_before) / task.control_dt
        vy = (self.qpos[:, 1] - y_before) / task.control_dt
        feet_after = task.feet_state(self.geom_xpos)
        foot_slip = task.foot_slip(self._feet_before, feet_after)
        feet_down = feet_after[1]
        landed, air_at_landing, self.air_time = task.air_time_update(self._feet_before[1], self.air_time, feet_down)
        torso_rot = self.xmat[:, 1]
        fell = task.fell(self.qpos, torso_rot)
        vx, vy = task.heading_velocity(torso_rot, vx, vy)
        reward, terms = task.reward(
            vx=vx, vy=vy, motor_power=power, action=actions, last_action=self.last_action,
            up_z=task.up_z(torso_rot), fell=fell, foot_slip=foot_slip,
            feet_down=feet_down, landed=landed, air_time=air_at_landing,
            foot_height=task.foot_heights(self.geom_xpos),
            phase=task.gait_phase(self.episode_length + 1),  # the clock after this step
            turn_rate=task.turn_rate(self.qvel), height=self.qpos[:, 2],
            joint_offset=self.qpos[:, task.joint_qpos] - task.home_ctrl,
        )
        self.last_action = actions
        self.episode_length += 1
        self.episode_return += reward
        terminated = fell & task.config.terminate_on_fall  # standing: no; it has to get up
        time_out = (self.episode_length >= task.max_steps) & ~terminated
        done = terminated | time_out

        result = StepResult(
            obs=self.observe(),
            reward=reward,
            done=done,
            time_out=time_out,
            terms=terms,
            episode_return=self.episode_return[done].clone(),
            episode_length=self.episode_length[done].clone(),
            episode_distance=self._distance(self.qpos[done, 0:2] - self.start_xy[done]),
            episode_fell=terminated[done].clone(),
        )
        self._feet_before = feet_after
        if done.any():
            # Restart the finished robots. mjw.kinematics() inside refreshes
            # body poses for ALL robots, so everyone else's observation and
            # foot positions were taken above, before it, with the same
            # one-step-stale poses the CPU env sees after mj_step. Restarted
            # robots get fresh ones, like WalkTask.reset_state (mj_forward).
            self._reset_robots(done)
            result.obs[done] = self.observe()[done]
            fresh_xy, fresh_down = task.feet_state(self.geom_xpos)
            self._feet_before[0][done] = fresh_xy[done]
            self._feet_before[1][done] = fresh_down[done]
        return result

    def _reset_robots(self, mask: torch.Tensor) -> None:
        """Restart the robots in `mask` (bool (N,)) standing, like WalkTask.reset_state."""
        n = int(mask.sum())
        self.reset_mask.copy_(mask)
        with self._warp():
            mjw.reset_data(self.m, self.d, reset=self._reset_mask)  # clears their state, time, warmstart
        qpos, qvel = self.task.reset_state(n, self.generator)
        self.qpos[mask] = qpos
        self.qvel[mask] = qvel
        self.ctrl[mask] = self.task.home_ctrl
        with self._warp():
            mjw.kinematics(self.m, self.d)  # body/geom poses of the new states, for the observation
        self.last_action[mask] = 0.0
        self.air_time[mask] = 0.0
        self.episode_length[mask] = 0
        self.episode_return[mask] = 0.0
        self.start_xy[mask] = self.qpos[mask, 0:2]
        self.next_push[mask] = self._push_delays(self.num_envs)[mask]

    def _distance(self, displacement: torch.Tensor) -> torch.Tensor:
        """Like WalkTask.distance, for (n, 2) displacements."""
        if self.task.config.velocity_frame == "world":
            return displacement[:, 0].clone()
        return displacement.norm(dim=1)

    def _push_delays(self, n: int) -> torch.Tensor:
        """(n,) control steps until the next push, like WalkTask.push_delay."""
        c, dt = self.task.config, self.task.control_dt
        uniform = torch.rand(n, generator=self.generator, device=self.device)
        return (c.push_interval * (0.5 + uniform) / dt).round().long().clamp(min=1)
