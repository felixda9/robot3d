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

import pickle
from collections.abc import Callable

import mujoco
import numpy as np

# Checkpoint discovery lives in runs.py (no PyTorch); re-exported here.
from robot3d.runs import Checkpoint, find_checkpoint, list_checkpoints  # noqa: F401
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

    def __init__(self, checkpoint: Checkpoint, model: mujoco.MjModel):
        info = checkpoint.run_info()
        self.checkpoint = checkpoint
        self.task = WalkTask(model, WalkConfig.from_run(info["walk_config"]))
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

    def reset(self) -> None:
        self.last_action = np.zeros(self.task.num_actions)
        self.steps = 0

    def reset_state(self, data: mujoco.MjData) -> None:
        """Start like a training episode: standing, with the same small noise
        (always standing, even for a policy that trained starting fallen too)."""
        self.task.reset_state(data, self._rng, allow_fallen=False)

    def action(self, data: mujoco.MjData) -> np.ndarray:
        """The policy's action (-1..1 per motor) for the current state."""
        phase = self.task.gait_phase(self.steps)
        action = self._predict(self.task.observation(data, self.last_action, phase))
        self.steps += 1
        self.last_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return self.last_action

    def act(self, data: mujoco.MjData) -> None:
        data.ctrl[:] = self.task.action_to_ctrl(self.action(data))


def evaluate(checkpoint: Checkpoint, episodes: int = 5, seed: int = 0) -> list[dict]:
    """Run full episodes headless (no noise in the actions) and measure them.
    Uses PolicyController, the same code path as the web viewer."""
    from robot3d.envs import WalkEnv  # here to keep `import robot3d.policy` light

    info = checkpoint.run_info()
    env = WalkEnv(info["robot"], WalkConfig.from_run(info["walk_config"]))
    controller = PolicyController(checkpoint, env.model)
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
                "fell": terminated,
                "upright": upright_steps / steps,
                **gait_numbers(np.array(feet_down[settle_steps:]), task),
            }
        )
    return results


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
    """The robot's trained behaviors, a walker and a stand policy, with an
    automatic switch. Drives the simulation like any controller (Simulation's
    Controller protocol).

    mode "stand": the stand policy drives (it stays up, catches shoves, and
        gets up by itself after a fall).
    mode "walk": the walker drives; if the robot falls and a stand policy is
        loaded, the stand policy takes over ("recovering") until the robot
        has stood steady for RECOVERED_SECONDS, then the walker resumes.
    """

    RECOVERED_SECONDS = 0.5

    def __init__(self) -> None:
        self.walker: PolicyController | None = None
        self.stander: PolicyController | None = None
        self.walk_label = ""
        self.stand_label = ""
        self.mode = "walk"
        self.recovering = False
        self._steady_steps = 0

    def install(self, controller: PolicyController, label: str) -> None:
        """Put a policy in its slot (stand policies: trained with the stand
        task) and switch to its mode, so what you load is what you see."""
        if controller.task.config.is_stand:
            self.stander, self.stand_label, self.mode = controller, label, "stand"
        else:
            self.walker, self.walk_label, self.mode = controller, label, "walk"
        self.reset()

    def has(self, mode: str) -> bool:
        return (self.stander if mode == "stand" else self.walker) is not None

    def set_mode(self, mode: str) -> None:
        if not self.has(mode):
            raise ValueError(f"no {mode} policy loaded")
        self.mode = mode
        self.reset()

    @property
    def active(self) -> PolicyController:
        """The policy driving right now."""
        if self.mode == "stand" or self.recovering or self.walker is None:
            return self.stander
        return self.walker

    # --- Controller protocol

    @property
    def decimation(self) -> int:
        return self.active.decimation

    def reset(self) -> None:
        self.recovering = False
        self._steady_steps = 0
        for policy in (self.walker, self.stander):
            if policy is not None:
                policy.reset()

    def reset_state(self, data: mujoco.MjData) -> None:
        self.active.reset_state(data)

    def act(self, data: mujoco.MjData) -> None:
        if self.mode == "walk" and self.walker is not None and self.stander is not None:
            task = self.walker.task
            if not self.recovering and task.fell(data):
                self.recovering = True  # hand over to the stand policy to get up
                self._steady_steps = 0
                self.stander.reset()
            elif self.recovering:
                steady = task.up_z(data) > 0.9 and data.qpos[2] > 0.8 * task.standing_height
                self._steady_steps = self._steady_steps + 1 if steady else 0
                if self._steady_steps >= round(self.RECOVERED_SECONDS / task.control_dt):
                    self.recovering = False  # up again: walk on
                    self.walker.reset()
        self.active.act(data)
