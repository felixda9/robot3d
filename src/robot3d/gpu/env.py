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
  * On terrain (WalkConfig.terrain "park"), every robot has its own tile
    (terrain.py): the model has MAX_TILE_BOXES placeholder boxes, and each
    simulated world moves its tile's boxes into them (MuJoCo Warp lets box
    poses and sizes differ per world). A robot gets a new tile at every
    reset, at its curriculum level.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from robot3d.gpu.task import BatchedWalkTask
from robot3d.robots import load_model
from robot3d.terrain import LEAVE_DISTANCE, LEVELS, MAX_TILE_BOXES, SPAWN_JITTER, TYPES, park_tiles
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
    # Extra per-rollout averages to log (e.g. the shove curriculum's level).
    stats: dict[str, float] | None = None


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
        config = config or WalkConfig()
        if config.terrain not in ("", "park"):
            raise ValueError(f"GPU training runs on terrain 'park' or flat (''), not {config.terrain!r}")
        self.tiles = park_tiles(config.terrain_seed, config.terrain_variants) if config.terrain else None
        self.mjm = load_model(robot, box_slots=MAX_TILE_BOXES if self.tiles else 0)
        self.task = BatchedWalkTask(WalkTask(self.mjm, config), self.device, self.tiles)
        self.num_obs = self.task.obs_size
        self.num_actions = self.task.num_actions
        randomizes_physics = self.task.task.randomizes_physics

        wp.init()
        self.wp_device = wp.get_device(device)
        mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, mjd)
        with wp.ScopedDevice(self.wp_device):
            self.m = mjw.put_model(self.mjm)
            # Model values that differ per robot get one row per world
            # (MuJoCo Warp reads row world % rows): each robot's terrain boxes,
            # and its randomized physics.
            if self.tiles:
                for name in ("geom_size", "geom_aabb", "geom_rbound"):
                    self._per_world(name)
            if randomizes_physics:
                for name in ("geom_friction", "body_mass", "actuator_gainprm", "actuator_biasprm"):
                    self._per_world(name)
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
        self.xpos = wp.to_torch(d.xpos)  # (N, nbody, 3) body frame origins
        self.xipos = wp.to_torch(d.xipos)  # (N, nbody, 3) body centers of mass
        self.xfrc = wp.to_torch(d.xfrc_applied)  # (N, nbody, 6) external force + torque (force pushes)
        self.geom_xpos = wp.to_torch(d.geom_xpos)  # (N, ngeom, 3)
        self.power = wp.to_torch(self._power)  # (N,)
        self.reset_mask = wp.to_torch(self._reset_mask)  # (N,)

        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.last_action = torch.zeros((num_envs, self.num_actions), device=self.device)
        self.episode_length = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(num_envs, device=self.device)
        self.start_xy = torch.zeros((num_envs, 2), device=self.device)
        # Steering commands (WalkConfig.commands): (forward, sideways, turn) per robot.
        self.command = torch.zeros((num_envs, 3), device=self.device)
        self.command[:, 0] = self.task.config.target_speed
        self.next_command = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._command_steps = round(self.task.config.command_resample_seconds / self.task.control_dt)
        self.next_push = torch.zeros(num_envs, dtype=torch.long, device=self.device)  # episode step of the next shove
        self.steady_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)  # get-up task
        self.flight_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)  # jump task
        self.landed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        c = self.task.config
        # Each robot's current max shove (m/s): push_max_speed, or its curriculum level.
        self.push_level = torch.full((num_envs,), c.push_max_speed, device=self.device)
        self.push_left = torch.zeros(num_envs, dtype=torch.long, device=self.device)  # control steps of force push left
        self._push_steps = round(self.task.task.PUSH_SECONDS / self.task.control_dt)
        # Foot x/y and on-ground at the end of the last step (for slip, and
        # as "was down" for landings), and each foot's time in the air.
        self._feet_before: tuple[torch.Tensor, torch.Tensor] | None = None
        self.air_time = torch.zeros((num_envs, len(self.task.feet)), device=self.device)
        self._graph = None
        if self.tiles:
            self._setup_terrain()
        if randomizes_physics:
            self.friction = wp.to_torch(self.m.geom_friction)  # (N, ngeom, 3)
            self.body_mass = wp.to_torch(self.m.body_mass)  # (N, nbody)
            self.gainprm = wp.to_torch(self.m.actuator_gainprm)  # (N, nu, 10)
            self.biasprm = wp.to_torch(self.m.actuator_biasprm)  # (N, nu, 10)
            nominal = self.task.task.nominal_physics
            self._nominal_mass = float(nominal["mass"])
            self._nominal_gain = torch.as_tensor(nominal["gain"], dtype=torch.float32, device=self.device)
            self._nominal_bias = torch.as_tensor(nominal["bias"], dtype=torch.float32, device=self.device)

    def _per_world(self, name: str) -> None:
        """Give model field `name` one row per world (a copy of the shared row)."""
        shared = getattr(self.m, name)
        rows = np.repeat(shared.numpy(), self.num_envs, axis=0)
        setattr(self.m, name, wp.array(rows, dtype=shared.dtype, device=self.wp_device))

    def _setup_terrain(self) -> None:
        """The tile pool as tensors, and each robot's terrain type and level."""
        tiles, dev, n = self.tiles, self.device, self.num_envs
        slots = MAX_TILE_BOXES
        pos = np.zeros((len(tiles), slots, 3))
        pos[..., 2] = -1.0  # unused slots: tiny boxes under the floor
        mat = np.tile(np.eye(3), (len(tiles), slots, 1, 1))
        size = np.full((len(tiles), slots, 3), 0.01)
        most = max(len(t.spawns) for t in tiles)
        spawns = np.zeros((len(tiles), most, 2))
        for i, tile in enumerate(tiles):
            for k, box in enumerate(tile.boxes):
                pos[i, k], mat[i, k], size[i, k] = box.center, box.rotation, box.half
            spawns[i, :len(tile.spawns)] = tile.spawns

        def tensor(x, dtype=torch.float32):
            return torch.as_tensor(x, dtype=dtype, device=dev)

        self.tile_pos, self.tile_mat, self.tile_size = tensor(pos), tensor(mat), tensor(size)
        self.tile_spawns = tensor(spawns)
        self.tile_spawn_count = tensor([len(t.spawns) for t in tiles], torch.long)
        self.slot_geoms = tensor([self.mjm.geom(f"terrain_slot{k}").id for k in range(slots)], torch.long)
        # Views of the per-world box poses (static geoms: MuJoCo Warp computes
        # them once, so we set them) and sizes/bounds (used by collisions).
        self.geom_xmat = wp.to_torch(self.d.geom_xmat)  # (N, ngeom, 3, 3)
        self.geom_size = wp.to_torch(self.m.geom_size)  # (N, ngeom, 3)
        self.geom_aabb = wp.to_torch(self.m.geom_aabb)  # (N, ngeom, 2, 3): box center, half-sizes
        self.geom_rbound = wp.to_torch(self.m.geom_rbound)  # (N, ngeom): bounding sphere radius
        # Terrain curriculum (legged_gym's): each robot keeps a terrain type,
        # and its level goes up when it walks off its tile, down when it falls
        # or gets stuck. Levels start low; a robot past the top level gets a
        # random one, so the easy ones aren't forgotten.
        self.terrain_type = torch.randint(0, len(TYPES), (n,), generator=self.generator, device=dev)
        start = self.task.config.terrain_start_level
        self.level = torch.randint(0, start + 1, (n,), generator=self.generator, device=dev)
        self.task.tile = torch.zeros(n, dtype=torch.long, device=dev)
        self.command_distance = torch.zeros(n, device=dev)  # how far the commands asked it to walk this episode

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
            self.next_command = self.episode_length + self._command_steps
        self._feet_before = self.task.feet_state(self.geom_xpos)
        return self.observe()

    def observe(self) -> torch.Tensor:
        phase = self.task.gait_phase(self.episode_length)
        return self.task.observation(self.qpos, self.qvel, self.xmat[:, 1], self.last_action, phase, self.command,
                                     self.generator)

    def step(self, actions: torch.Tensor) -> StepResult:
        task = self.task
        actions = actions.clamp(-1.0, 1.0)
        self.ctrl.copy_(task.action_to_ctrl(actions, self.qpos[:, task.joint_qpos]))
        c = task.config
        if c.commands:  # new commands for robots whose current one has run its time
            due = self.episode_length >= self.next_command
            self.command = torch.where(due[:, None], task.sample_commands(self.num_envs, self.generator), self.command)
            self.next_command = torch.where(due, self.episode_length + self._command_steps, self.next_command)
        if c.push_interval > 0:
            # Shoves (see WalkConfig.push_interval), without a GPU->CPU sync:
            # every robot draws a kick, only those due get it.
            due = self.episode_length >= self.next_push
            if c.push_kind == "force":
                angle = 2 * torch.pi * torch.rand(self.num_envs, generator=self.generator, device=self.device)
                size = torch.rand(self.num_envs, generator=self.generator, device=self.device) * self.push_level
                height = torch.rand(self.num_envs, generator=self.generator, device=self.device)
                force, torque = task.push_wrench(self.xmat[:, 1], self.xpos[:, 1], self.xipos[:, 1],
                                                 angle, size, height)
                wrench = torch.cat([force, torque], dim=1)
                self.xfrc[:, 1] = torch.where(due[:, None], wrench, self.xfrc[:, 1])
                self.push_left = torch.where(due, torch.full_like(self.push_left, self._push_steps + 1),
                                             self.push_left)
            elif c.push_direction == "circle":
                angle = 2 * torch.pi * torch.rand(self.num_envs, generator=self.generator, device=self.device)
                size = torch.rand(self.num_envs, generator=self.generator, device=self.device) * self.push_level
                kick = torch.stack([angle.cos(), angle.sin()], dim=1) * size[:, None]
            else:
                kick = (2 * torch.rand((self.num_envs, 2), generator=self.generator, device=self.device) - 1)
                kick = kick * c.push_max_speed
            if c.push_kind != "force":
                self.qvel[:, 0:2] += kick * due[:, None]
            if c.push_max_spin > 0:
                spin = 2 * torch.rand((self.num_envs, 3), generator=self.generator, device=self.device) - 1
                self.qvel[:, 3:6] += spin * c.push_max_spin * due[:, None]
            self.next_push = torch.where(due, self.episode_length + self._push_delays(self.num_envs), self.next_push)
        if c.push_kind == "force":  # a force push acts for _push_steps control steps, then stops
            self.push_left = (self.push_left - 1).clamp(min=0)
            self.xfrc[:, 1] *= (self.push_left > 0).float()[:, None]
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
        steady = task.steady(self.qpos, torso_rot)
        self.steady_steps = torch.where(steady, self.steady_steps + 1, torch.zeros_like(self.steady_steps))
        succeeded = (self.steady_steps >= task.task.success_steps) & (task.config.success_bonus > 0)
        airborne, self.flight_steps, self.landed = task.jump_update(self.flight_steps, self.landed, feet_down)
        vx, vy = task.heading_velocity(torso_rot, vx, vy)
        reward, terms = task.reward(
            vx=vx, vy=vy, motor_power=power, action=actions, last_action=self.last_action,
            up_z=task.up_z(torso_rot), fell=fell, foot_slip=foot_slip,
            feet_down=feet_down, landed=landed, air_time=air_at_landing,
            foot_height=task.foot_heights(self.geom_xpos),
            phase=task.gait_phase(self.episode_length + 1),  # the clock after this step
            turn_rate=task.turn_rate(self.qvel), height=task.height(self.qpos),
            joint_offset=self.qpos[:, task.joint_qpos] - task.home_ctrl,
            joint_velocity=self.qvel[:, task.joint_qvel], angular_velocity=self.qvel[:, 3:6],
            succeeded=succeeded, jump_airborne=airborne, jump_landed=self.landed, command=self.command,
        )
        self.last_action = actions
        self.episode_length += 1
        self.episode_return += reward
        # A fall ends a walk/stand episode; getting up steadily ends a get-up one.
        fall_ends = fell & task.config.terminate_on_fall
        terminated = fall_ends | succeeded
        time_out = (self.episode_length >= task.max_steps) & ~terminated
        stats = {}
        if self.tiles:
            # Walked off its tile: the episode ends like a time-out (the task
            # would go on; PPO estimates the rest), and a harder tile follows.
            walked = (self.qpos[:, 0:2] - self.start_xy).norm(dim=1)
            left = (walked > LEAVE_DISTANCE) & ~terminated
            self.command_distance += self.command[:, 0:2].norm(dim=1) * task.control_dt
            stuck = time_out & (walked < (0.5 * self.command_distance).clamp(max=1.0))
            self.level += (left.long() - (fall_ends | stuck).long())
            beyond = self.level >= LEVELS
            random_level = torch.randint(0, LEVELS, (self.num_envs,), generator=self.generator, device=self.device)
            self.level = torch.where(beyond, random_level, self.level).clamp(min=0)
            time_out = time_out | left
            stats["curriculum/terrain_level"] = self.level.float().mean()
        done = terminated | time_out
        if c.push_curriculum_max > 0:
            # Survived a whole episode: harder shoves; fell: easier (see WalkConfig).
            step = c.push_curriculum_step
            self.push_level = torch.where(time_out, (self.push_level + step).clamp(max=c.push_curriculum_max),
                                          self.push_level)
            self.push_level = torch.where(fall_ends, (self.push_level - step).clamp(min=c.push_max_speed),
                                          self.push_level)

        result = StepResult(
            obs=self.observe(),
            reward=reward,
            done=done,
            time_out=time_out,
            terms=terms,
            episode_return=self.episode_return[done].clone(),
            episode_length=self.episode_length[done].clone(),
            episode_distance=self._distance(self.qpos[done, 0:2] - self.start_xy[done]),
            episode_fell=fall_ends[done].clone(),
            stats=({"curriculum/push_max_speed": self.push_level.mean()} if c.push_curriculum_max > 0 else {})
            | stats or None,
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
        if self.tiles:
            self._new_tiles(mask)
        if self.task.task.randomizes_physics:
            self._randomize_physics(mask)
        self.reset_mask.copy_(mask)
        with self._warp():
            mjw.reset_data(self.m, self.d, reset=self._reset_mask)  # clears their state, time, warmstart
        qpos, qvel = self.task.reset_state(n, self.generator)
        if self.tiles:
            self._spawn(mask, qpos, qvel)
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
        if self.task.config.commands:
            self.command[mask] = self.task.sample_commands(self.num_envs, self.generator)[mask]
            self.next_command[mask] = self._command_steps
        self.steady_steps[mask] = 0
        self.flight_steps[mask] = 0
        self.landed[mask] = False
        self.push_left[mask] = 0
        self.xfrc[mask] = 0.0
        self.next_push[mask] = self._push_delays(self.num_envs)[mask]
        if self.tiles:
            self.command_distance[mask] = 0.0

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator, device=self.device)

    def _new_tiles(self, mask: torch.Tensor) -> None:
        """A random tile of each masked robot's type and level, moved into its world's box slots."""
        variants = self.task.config.terrain_variants
        variant = (self._rand(self.num_envs) * variants).long().clamp(max=variants - 1)
        tile = (self.level * len(TYPES) + self.terrain_type) * variants + variant
        self.task.tile = torch.where(mask, tile, self.task.tile)
        worlds = mask.nonzero()[:, 0:1]  # (n, 1)
        tile = self.task.tile[worlds[:, 0]]
        slots = self.slot_geoms[None, :]  # (1, K)
        self.geom_xpos[worlds, slots] = self.tile_pos[tile]
        self.geom_xmat[worlds, slots] = self.tile_mat[tile]
        self.geom_size[worlds, slots] = self.tile_size[tile]
        self.geom_aabb[worlds, slots, 0] = 0.0
        self.geom_aabb[worlds, slots, 1] = self.tile_size[tile]
        self.geom_rbound[worlds, slots] = self.tile_size[tile].norm(dim=-1)

    def _spawn(self, mask: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor) -> None:
        """Move the masked robots' fresh standing states onto their tiles, as
        terrain.spawn_point draws: half at the middle, the rest at another spawn point; any heading."""
        tile = self.task.tile[mask]
        n = len(tile)
        count = self.tile_spawn_count[tile]
        other = 1 + (self._rand(n) * (count - 1)).long().clamp(max=(count - 2).clamp(min=0))
        index = torch.where((self._rand(n) < 0.5) | (count == 1), torch.zeros_like(other), other)
        xy = self.tile_spawns[tile, index] + (2 * self._rand(n, 2) - 1) * SPAWN_JITTER
        yaw = (2 * self._rand(n) - 1) * math.pi
        self.task.place(qpos, qvel, xy[:, 0], xy[:, 1], yaw, tile)

    def _randomize_physics(self, mask: torch.Tensor) -> None:
        """New friction, torso mass and motor strength for the masked robots (WalkTask.sample_physics)."""
        c = self.task.config
        worlds = mask.nonzero()[:, 0]
        n = len(worlds)

        def uniform(low, high):
            return low + (high - low) * self._rand(n)

        self.friction[worlds, :, 0] = uniform(c.friction_min, c.friction_max)[:, None]
        self.body_mass[worlds, 1] = self._nominal_mass + uniform(c.added_mass_min, c.added_mass_max)
        strength = uniform(c.motor_strength_min, c.motor_strength_max)
        self.gainprm[worlds, :, 0] = self._nominal_gain * strength[:, None]
        self.biasprm[worlds, :, 1:3] = self._nominal_bias * strength[:, None, None]

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
