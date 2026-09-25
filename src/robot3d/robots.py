"""Finding and loading robot models.

Robots are data, not code: each robot is an MJCF file in the top-level
`robots/` folder, e.g. `robots/quadruped.xml` is the robot named "quadruped".
"""

from pathlib import Path

import mujoco

# src/robot3d/robots.py -> repo root is two levels above the package folder.
ROBOTS_DIR = Path(__file__).resolve().parents[2] / "robots"


def available_robots() -> list[str]:
    return sorted(p.stem for p in ROBOTS_DIR.glob("*.xml"))


def robot_path(name: str) -> Path:
    path = ROBOTS_DIR / f"{name}.xml"
    if not path.is_file():
        raise FileNotFoundError(
            f"No robot named {name!r} in {ROBOTS_DIR}. "
            f"Available: {', '.join(available_robots()) or '(none)'}"
        )
    return path


def load_model(name: str) -> mujoco.MjModel:
    """Compile a robot's MJCF file into an MjModel (the static description:
    bodies, joints, masses, motors). The changing state (positions,
    velocities, time) lives separately in an MjData."""
    return mujoco.MjModel.from_xml_path(str(robot_path(name)))


def reset_to_keyframe(model: mujoco.MjModel, data: mujoco.MjData, key: str = "home") -> None:
    """Reset the simulation to a named <keyframe>: joint positions, velocities,
    and motor targets (ctrl) all come from the keyframe, and time goes back to 0."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
    if key_id < 0:
        raise ValueError(f"Model has no keyframe named {key!r}")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    # mj_forward computes everything derived from the state (body positions,
    # contacts, ...) without advancing time, so the first frame is correct.
    mujoco.mj_forward(model, data)
