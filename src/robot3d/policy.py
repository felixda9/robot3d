"""Load trained checkpoints and run them on a live simulation.

    checkpoint = find_checkpoint("runs/my_run")        # newest checkpoint of a run
    checkpoint = find_checkpoint("runs/my_run/checkpoints/step_002000000.zip")
    controller = PolicyController(checkpoint, sim.model)
    sim.set_controller(controller)                     # the policy now drives the motors
"""

import pickle

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Checkpoint discovery lives in runs.py (no PyTorch); re-exported here.
from robot3d.runs import Checkpoint, find_checkpoint, list_checkpoints  # noqa: F401
from robot3d.walk import WalkConfig, WalkTask


class PolicyController:
    """Drives a simulation's motors with a trained policy.

    Every `decimation` physics steps (20 ms), Simulation.step() calls act():
    observe -> normalize -> policy -> motor targets, exactly as in training
    (same WalkTask, same settings from the run's run.json).
    """

    def __init__(self, checkpoint: Checkpoint, model: mujoco.MjModel):
        info = checkpoint.run_info()
        self.checkpoint = checkpoint
        self.task = WalkTask(model, WalkConfig(**info["walk_config"]))
        self.decimation = self.task.decimation
        self.policy = PPO.load(checkpoint.model_path, device="cpu")
        expected = self.policy.observation_space.shape
        if expected != (self.task.obs_size,):
            raise ValueError(
                f"{checkpoint.label} expects observations of shape {expected}, but robot "
                f"{info['robot']!r} with this task gives ({self.task.obs_size},). Wrong robot?"
            )
        with open(checkpoint.normalizer_path, "rb") as f:
            # The saved VecNormalize holds the running mean/variance of the
            # observations seen in training; the policy needs inputs scaled
            # the same way. (Unpickled without an env: we only use normalize_obs.)
            self.normalizer: VecNormalize = pickle.load(f)
        self.last_action = np.zeros(self.task.num_actions)
        self._rng = np.random.default_rng()

    def reset(self) -> None:
        self.last_action = np.zeros(self.task.num_actions)

    def reset_state(self, data: mujoco.MjData) -> None:
        """Start like a training episode: standing, with the same small noise."""
        self.task.reset_state(data, self._rng)

    def action(self, data: mujoco.MjData) -> np.ndarray:
        """The policy's action (-1..1 per motor) for the current state."""
        observation = self.task.observation(data, self.last_action)
        normalized = self.normalizer.normalize_obs(observation)
        # deterministic=True: use the policy's best guess, without the random
        # exploration noise used during training.
        action, _ = self.policy.predict(normalized, deterministic=True)
        self.last_action = np.clip(action.astype(np.float64), -1.0, 1.0)
        return self.last_action

    def act(self, data: mujoco.MjData) -> None:
        data.ctrl[:] = self.task.action_to_ctrl(self.action(data))


def evaluate(checkpoint: Checkpoint, episodes: int = 5, seed: int = 0) -> list[dict]:
    """Run full episodes headless (no noise in the actions) and measure them.
    Uses PolicyController, the same code path as the web viewer."""
    from robot3d.envs import WalkEnv  # here to keep `import robot3d.policy` light

    info = checkpoint.run_info()
    env = WalkEnv(info["robot"], WalkConfig(**info["walk_config"]))
    controller = PolicyController(checkpoint, env.model)
    results = []
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        controller.reset()
        total, steps = 0.0, 0
        while True:
            _, reward, terminated, truncated, step_info = env.step(controller.action(env.data))
            total += reward
            steps += 1
            if terminated or truncated:
                break
        seconds = steps * env.task.control_dt
        results.append(
            {
                "return": total,
                "seconds": seconds,
                "distance": step_info["distance"],
                "speed": step_info["distance"] / seconds,
                "fell": terminated,
            }
        )
    return results
