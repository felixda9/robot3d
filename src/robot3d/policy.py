"""Load trained checkpoints and run them on a live simulation.

    checkpoint = find_checkpoint("runs/my_run")        # newest checkpoint of a run
    checkpoint = find_checkpoint("runs/my_run/checkpoints/step_002000000.zip")
    controller = PolicyController(checkpoint, sim.model)
    sim.set_controller(controller)                     # the policy now drives the motors

Two checkpoint formats, same behavior: SB3 (.zip, CPU training) and GPU-PPO
(.pt, gpu/ppo.py). Both run here on the CPU in regular (float64) MuJoCo,
which is also how GPU-trained policies are checked against the physics they
weren't trained in (float32 MuJoCo Warp).
"""

import copy
import dataclasses
import math
import pickle
from collections.abc import Callable

import mujoco
import numpy as np

# Checkpoint discovery lives in runs.py (no PyTorch); re-exported here.
from robot3d.runs import SKILL_TEST_VERSION, Checkpoint, find_checkpoint, list_checkpoints  # noqa: F401
from robot3d.terrain import Terrain
from robot3d.walk import WalkConfig, WalkTask


def _load_sb3(checkpoint: Checkpoint) -> tuple[Callable[[np.ndarray], np.ndarray], int]:
    from stable_baselines3 import PPO

    policy = PPO.load(checkpoint.model_path, device="cpu")
    with open(checkpoint.normalizer_path, "rb") as f:
        # The saved VecNormalize holds the running mean/variance of the
        # observations seen in training; the policy needs inputs scaled the
        # same way. (Unpickled without an env: we only use normalize_obs.)
        normalizer = pickle.load(f)

    def predict(observation: np.ndarray) -> np.ndarray:
        # deterministic=True: the policy's best guess, without the random
        # exploration noise used during training.
        action, _ = policy.predict(normalizer.normalize_obs(observation), deterministic=True)
        return action

    return predict, policy.observation_space.shape[0]


def _load_torch(checkpoint: Checkpoint) -> tuple[Callable[[np.ndarray], np.ndarray], int]:
    from robot3d.gpu.ppo import TorchPolicy

    policy = TorchPolicy(checkpoint.model_path)  # normalizes internally, mean action
    return policy.act, policy.num_obs


class PolicyController:
    """Drives a simulation's motors with a trained policy.

    Every `decimation` physics steps (20 ms), Simulation.step() calls act():
    observe -> normalize -> policy -> motor targets, exactly as in training
    (same WalkTask, same settings from the run's run.json).
    """

    def __init__(self, checkpoint: Checkpoint, model: mujoco.MjModel, terrain: Terrain | None = None):
        """model, terrain: the simulation it drives, and the terrain boxes in
        it (for heights above the ground and the height map)."""
        info = checkpoint.run_info()
        self.checkpoint = checkpoint
        self.task = WalkTask(model, WalkConfig.from_run(info["walk_config"]), terrain=terrain)
        self.decimation = self.task.decimation
        load = _load_torch if checkpoint.format == "torch" else _load_sb3
        self._predict, expected_obs = load(checkpoint)
        if expected_obs != self.task.obs_size:
            raise ValueError(
                f"{checkpoint.label} expects {expected_obs} observations, but robot "
                f"{info['robot']!r} with this task gives {self.task.obs_size}. Wrong robot?"
            )
        self.last_action = np.zeros(self.task.num_actions)
        self.steps = 0  # control steps since the (re)start: drives the gait clock
        self._rng = np.random.default_rng()
        # Steering command (forward, sideways, turn) for policies trained with
        # commands; zero = stand still. Others walk at their target speed.
        self.steerable = self.task.config.commands
        self.command = np.zeros(3) if self.steerable else self.task.default_command()

    def reset(self) -> None:
        self.last_action = np.zeros(self.task.num_actions)
        self.steps = 0

    def retargeted(self, model: mujoco.MjModel, terrain: Terrain | None) -> "PolicyController":
        """The same policy (network shared, not reloaded) for another model of
        the same robot, e.g. after the viewer switched the ground."""
        other = copy.copy(self)
        other.task = WalkTask(model, self.task.config, terrain=terrain)
        other.reset()
        return other

    def reset_state(self, data: mujoco.MjData) -> None:
        """Start like a training episode: standing, with the same small noise
        (always standing, even for a policy that trained starting fallen too)."""
        self.task.reset_state(data, self._rng, allow_fallen=False)

    def action(self, data: mujoco.MjData) -> np.ndarray:
        """The policy's action (-1..1 per motor) for the current state."""
        phase = self.task.gait_phase(self.steps)
        action = self._predict(self.task.observation(data, self.last_action, phase, self.command))
        self.steps += 1
        self.last_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return self.last_action

    def act(self, data: mujoco.MjData) -> None:
        data.ctrl[:] = self.task.action_to_ctrl(self.action(data), data.qpos[self.task.joint_qpos])


def evaluate(checkpoint: Checkpoint, episodes: int = 5, seed: int = 0) -> list[dict]:
    """Run full episodes headless (no noise in the actions) and measure them.
    Uses PolicyController, the same code path as the web viewer."""
    from robot3d.envs import WalkEnv  # here to keep `import robot3d.policy` light

    info = checkpoint.run_info()
    env = WalkEnv(info["robot"], nominal(WalkConfig.from_run(info["walk_config"])))
    controller = PolicyController(checkpoint, env.model, env.terrain)
    results = []
    task = env.task
    settle_steps = round(1.0 / task.control_dt)  # skip the first second (starting up) for gait numbers
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        controller.reset()
        total, steps = 0.0, 0
        feet_down = []  # per step: which feet are on the ground
        upright_steps = 0
        while True:
            _, reward, terminated, truncated, step_info = env.step(controller.action(env.data))
            total += reward
            steps += 1
            feet_down.append(task.feet_state(env.data)[1])
            upright_steps += not step_info["fell"]
            if terminated or truncated:
                break
        seconds = steps * task.control_dt
        results.append(
            {
                "return": total,
                "seconds": seconds,
                "distance": step_info["distance"],
                "speed": step_info["distance"] / seconds,
                "fell": terminated and not step_info["succeeded"],
                "upright": upright_steps / steps,
                **gait_numbers(np.array(feet_down[settle_steps:]), task),
            }
        )
    return results


def nominal(config: WalkConfig) -> WalkConfig:
    """A run's settings for testing it: exact height map, unrandomized physics."""
    return dataclasses.replace(config, height_map_noise=0.0, friction_min=1.0, friction_max=1.0, added_mass_min=0.0,
                               added_mass_max=0.0, motor_strength_min=1.0, motor_strength_max=1.0)


def gait_numbers(feet_down: np.ndarray, task: WalkTask) -> dict:
    """Walk-or-run numbers from (steps, feet) on-the-ground flags.

    duty_factor: share of time each foot is on the ground (walk > 0.5, run < 0.5).
    airborne: share of time with all feet in the air (a walk: 0).
    diagonal_sync: share of time diagonal feet are both down or both up (trot: ~1).
    cadence: touchdowns per foot per second.
    """
    if len(feet_down) < 2:
        return {"duty_factor": None, "airborne": None, "diagonal_sync": None, "cadence": None}
    touchdowns = (feet_down[1:] & ~feet_down[:-1]).sum(axis=0)
    sync = [np.mean(feet_down[:, a] == feet_down[:, b]) for a, b in task.diagonal_pairs]
    return {
        "duty_factor": float(feet_down.mean()),
        "airborne": float(np.mean(feet_down.sum(axis=1) == 0)),
        "diagonal_sync": float(np.mean(sync)) if sync else None,
        "cadence": float(touchdowns.mean() / (len(feet_down) * task.control_dt)),
    }


class Behaviors:
    """The robot's trained behaviors and the automatic switch between them.
    Drives the simulation like any controller (Simulation's Controller protocol).

    Up to three policies, one per task (the run's WalkConfig.task):
      walk:  walks (Walk mode).
      stand: stands still, catches shoves (Stand mode).
      getup: gets up after a fall. Not a mode: whenever the robot falls
             (WalkTask.fell) and it's loaded, it takes over ("recovering")
             until the robot has stood steady for RECOVERED_SECONDS, then the
             mode's policy drives again. With nothing else loaded, it also
             stands (it was trained to stand once up).
      jump:  one jump on command (jump()): it drives ("jumping") until it has
             landed and stood steady for RECOVERED_SECONDS, or its episode
             length has passed; then the mode's policy again.
    """

    RECOVERED_SECONDS = 0.5
    TASKS = ("walk", "stand", "getup", "jump")

    def __init__(self) -> None:
        self.policies: dict[str, PolicyController] = {}
        self.labels: dict[str, str] = {}
        self.mode = "walk"
        self.recovering = False
        self._steady_steps = 0
        self.jumping = False
        self._jump_steps = 0
        self._jump_flight = 0
        self._jump_landed = False

    def label(self, task: str) -> str:
        return self.labels.get(task, "")

    def install(self, controller: PolicyController, label: str) -> None:
        """Put a policy in its task's slot. Walk and stand policies switch to
        their mode (what you load is what you see); a get-up policy only
        does, to "stand", when there's nothing else to drive."""
        task = controller.task.config.task
        self.policies[task] = controller
        self.labels[task] = label
        if task in ("walk", "stand"):
            self.mode = task
        elif not self.has(self.mode):
            self.mode = "stand"
        self.reset()

    def retargeted(self, model: mujoco.MjModel, terrain: Terrain | None) -> dict[str, PolicyController]:
        """The loaded policies, for another model of the same robot (PolicyController.retargeted)."""
        return {task: policy.retargeted(model, terrain) for task, policy in list(self.policies.items())}

    def swap(self, policies: dict[str, PolicyController]) -> None:
        """Replace the loaded policies with these (from retargeted()); labels and mode stay."""
        self.policies = dict(policies)
        self.reset()

    def has(self, mode: str) -> bool:
        """Can this mode drive? (Stand mode can use the get-up policy.)"""
        return mode in self.policies or (mode == "stand" and "getup" in self.policies)

    def set_mode(self, mode: str) -> None:
        if not self.has(mode):
            raise ValueError(f"no {mode} policy loaded")
        self.mode = mode
        self.reset()

    @property
    def steerable(self) -> bool:
        walker = self.policies.get("walk")
        return walker is not None and walker.steerable

    def command_limits(self) -> tuple[float, float, float, float]:
        if not self.steerable:
            return (0.0, 0.0, 0.0, 0.0)
        c = self.policies["walk"].task.config
        return (c.command_max_forward, c.command_max_backward, c.command_max_sideways, c.command_max_turn)

    def set_command(self, forward: float, sideways: float, turn: float) -> None:
        """Steer the walker (clamped to what it was trained for)."""
        if not self.steerable:
            raise ValueError("the walk policy doesn't take steering commands (train one with --task steer)")
        walker = self.policies["walk"]
        walker.command = walker.task.clamp_command([forward, sideways, turn])

    def jump(self) -> None:
        """Start one jump (the jump policy drives until it has landed and settled)."""
        if "jump" not in self.policies:
            raise ValueError("no jump policy loaded")
        if self.recovering:
            raise ValueError("it's getting up; jump once it stands")
        self.jumping = True
        self._jump_steps = self._jump_flight = self._steady_steps = 0
        self._jump_landed = False
        self.policies["jump"].reset()  # its clock starts: 0..1 over the jump

    @property
    def active(self) -> PolicyController:
        """The policy driving right now."""
        if self.recovering:
            return self.policies["getup"]
        if self.jumping:
            return self.policies["jump"]
        return self.policies.get(self.mode) or self.policies.get("getup") or next(iter(self.policies.values()))

    # --- Controller protocol

    @property
    def decimation(self) -> int:
        return self.active.decimation

    def reset(self) -> None:
        self.recovering = False
        self.jumping = False
        self._steady_steps = 0
        for policy in self.policies.values():
            policy.reset()

    def reset_state(self, data: mujoco.MjData) -> None:
        self.active.reset_state(data)

    def act(self, data: mujoco.MjData) -> None:
        if self.jumping:
            jumper = self.policies["jump"]
            task = jumper.task
            _, self._jump_flight, self._jump_landed = task.jump_update(
                self._jump_flight, self._jump_landed, task.feet_state(data)[1])
            self._jump_steps += 1
            self._steady_steps = self._steady_steps + 1 if self._jump_landed and task.steady(data) else 0
            settled = self._steady_steps >= round(self.RECOVERED_SECONDS / task.control_dt)
            if settled or self._jump_steps >= task.max_steps:
                self.jumping = False  # back to the mode's policy
                self._steady_steps = 0
                self.active.reset()
        getup = self.policies.get("getup")
        if getup is not None and self.active is not getup or self.recovering:
            task = getup.task
            # Fallen by the driving policy's own measure (when its training
            # episodes ended): don't cut in while it can still save itself.
            if not self.recovering and self.active.task.fell(data):
                self.recovering = True  # hand over to the get-up policy
                self.jumping = False
                self._steady_steps = 0
                getup.reset()
            elif self.recovering:
                self._steady_steps = self._steady_steps + 1 if task.steady(data) else 0
                if self._steady_steps >= round(self.RECOVERED_SECONDS / task.control_dt):
                    self.recovering = False  # up again: back to the mode's policy
                    self.active.reset()
        self.active.act(data)


# ------------------------------------------------------------- skill tests

SHOVE_TEST_SPEEDS = (1.0, 2.0)  # m/s of push (quadruped12: 72 and 144 N on the viewer's slider)
SHOVE_TEST_DIRECTIONS = 16
JUMP_TEST_JUMPS = 8
GETUP_TEST_BANK_STARTS = 16
GETUP_TEST_UPSIDE_DOWN = 8


def skill_test(checkpoint: Checkpoint) -> dict:
    """A fixed, task-specific test of what the policy is for; the dashboard's
    "Skill" column and its choice of the best checkpoint. (The mean return
    over a few episodes was too noisy to pick one: one unlucky shove, or a
    lucky set of easy starts, decided it.)

    walk, stand: a push test, pushed the way the viewer pushes (a force for
        0.1 s on the torso's side, halfway up: WalkEnv.push). For each of
        SHOVE_TEST_SPEEDS and SHOVE_TEST_DIRECTIONS: 3 s of walking/standing,
        one push, survived if not fallen 3 s later. (v1 used velocity kicks at
        the center of mass: far easier than the viewer's pushes, which also
        tip the robot; stand12_v3 scored 78% yet fell to 60 N side pushes.)
    jump: JUMP_TEST_JUMPS jumps from standing; passed if it landed (after
        >= 60 ms in the air) and stood steady within the jump's 3 s. The
        description reports the median jump height (torso rise at the top).
    getup: GETUP_TEST_BANK_STARTS fallen poses from the fallen bank plus
        GETUP_TEST_UPSIDE_DOWN upside down with random leg angles; passed if
        standing steady (WalkTask.steady) for 0.5 s within 10 s.
    Returns {"skill": share passed (0..1), "skill_test": what was tested}.
    """
    from robot3d.envs import WalkEnv

    info = checkpoint.run_info()
    config = dataclasses.replace(nominal(WalkConfig.from_run(info["walk_config"])), push_interval=0.0)
    if config.terrain:
        return course_test(checkpoint, config)
    env = WalkEnv(info["robot"], config)
    controller = PolicyController(checkpoint, env.model)
    task, model, data = env.task, env.model, env.data
    steps = lambda seconds: round(seconds / task.control_dt)  # noqa: E731

    if config.task == "jump":
        passed, heights = 0, []
        for k in range(JUMP_TEST_JUMPS):
            env.reset(seed=k)
            controller.reset()
            top, steady, flight, landed, ok = data.qpos[2], 0, 0, False, False
            for _ in range(task.max_steps):
                env.step(controller.action(data))
                top = max(top, data.qpos[2])
                _, flight, landed = task.jump_update(flight, landed, task.feet_state(data)[1])
                steady = steady + 1 if landed and task.steady(data) else 0
                if task.fell(data):
                    break
                if steady >= steps(0.5):
                    ok = True
                    break
            passed += ok
            heights.append(top - task.standing_height)
        return {"skill": passed / JUMP_TEST_JUMPS,
                "skill_test": f"{SKILL_TEST_VERSION}: {JUMP_TEST_JUMPS} jumps, landed and steady "
                              f"(median height {np.median(heights) * 100:.0f} cm)"}

    if config.task != "getup":
        passed = total = 0
        for speed in SHOVE_TEST_SPEEDS:
            for k in range(SHOVE_TEST_DIRECTIONS):
                env.reset(seed=k)
                controller.reset()
                for _ in range(steps(3.0)):
                    env.step(controller.action(data))
                env.push(2 * np.pi * k / SHOVE_TEST_DIRECTIONS, speed, height=0.5)
                fell = False
                for _ in range(steps(3.0)):
                    env.step(controller.action(data))
                    if task.fell(data):
                        fell = True
                        break
                total += 1
                passed += not fell
        newtons = " and ".join(f"{round(s * task.robot_mass / task.PUSH_SECONDS)}" for s in SHOVE_TEST_SPEEDS)
        return {"skill": passed / total,
                "skill_test": f"{SKILL_TEST_VERSION}: pushes of {newtons} N (0.1 s, torso side) "
                              f"from {SHOVE_TEST_DIRECTIONS} directions"}

    rng = np.random.default_rng(0)
    bank_qpos, bank_qvel = task.fallen_states
    starts = [(bank_qpos[i], bank_qvel[i])
              for i in np.linspace(0, len(bank_qpos) - 1, GETUP_TEST_BANK_STARTS).astype(int)]
    for _ in range(GETUP_TEST_UPSIDE_DOWN):  # upside down, any heading, legs anywhere
        mujoco.mj_resetData(model, data)
        data.qpos[:] = task.standing_qpos
        angles = rng.uniform(task.joint_range[:, 0], task.joint_range[:, 1])
        data.qpos[task.joint_qpos] = angles
        data.ctrl[:] = np.clip(angles, task.ctrl_low, task.ctrl_high)
        yaw = rng.uniform(0, 2 * np.pi)
        data.qpos[3:7] = [0.0, np.cos(yaw / 2), np.sin(yaw / 2), 0.0]  # rolled over, then turned
        data.qpos[2] = 0.25
        mujoco.mj_forward(model, data)
        for _ in range(round(0.5 / model.opt.timestep)):  # land, joints held
            mujoco.mj_step(model, data)
        starts.append((data.qpos.copy(), data.qvel.copy()))
    passed = 0
    for i, (qpos, qvel) in enumerate(starts):
        env.reset(seed=i)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        env._steps = 0
        controller.reset()
        steady = 0
        for _ in range(steps(10.0)):
            env.step(controller.action(data))
            steady = steady + 1 if task.steady(data) else 0
            if steady >= steps(0.5):
                passed += 1
                break
    return {"skill": passed / len(starts),
            "skill_test": f"{SKILL_TEST_VERSION}: {len(starts)} fallen starts ({GETUP_TEST_UPSIDE_DOWN} upside down), "
                          "up within 10 s"}


COURSE_TEST_TRIES = 4
COURSE_TEST_SPEED = 0.4  # m/s forward command
COURSE_TEST_SECONDS = 90.0


def course_test(checkpoint: Checkpoint, config: WalkConfig) -> dict:
    """The skill test of terrain runs: walk the held-out test course
    (Terrain.course: shapes the training park never has), steered along its
    lane like a person with a gamepad would: forward at COURSE_TEST_SPEED,
    turning back toward the lane's center line. Skill = the share of the
    course walked before falling or running out of time, averaged over
    COURSE_TEST_TRIES tries (different small start noise)."""
    from robot3d.envs import WalkEnv

    course = Terrain.course()
    env = WalkEnv(checkpoint.run_info()["robot"], dataclasses.replace(config, terrain="course"), terrain=course)
    controller = PolicyController(checkpoint, env.model, course)
    task, data = env.task, env.data
    shares = []
    for k in range(COURSE_TEST_TRIES):
        env.reset(seed=k)
        controller.reset()
        for _ in range(round(COURSE_TEST_SECONDS / task.control_dt)):
            if controller.steerable:
                c, s = task.heading_cos_sin(data)
                aim = math.atan2(-data.qpos[1], 1.0)  # back toward y = 0, within ~1 m
                error = math.atan2(math.sin(aim - math.atan2(s, c)), math.cos(aim - math.atan2(s, c)))
                command = [COURSE_TEST_SPEED, 0.0, float(np.clip(2.0 * error, -0.8, 0.8))]
                controller.command = env._command = task.clamp_command(command)
            env.step(controller.action(data))
            if task.fell(data) or data.qpos[0] > course.length:
                break
        shares.append(min(max(data.qpos[0], 0.0) / course.length, 1.0))
    return {"skill": float(np.mean(shares)),
            "skill_test": f"{SKILL_TEST_VERSION}: test course ({course.length:.0f} m of unseen terrain), "
                          f"share walked ({COURSE_TEST_TRIES} tries)"}
