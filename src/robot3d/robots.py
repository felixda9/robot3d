"""Finding and loading robot models.

Robots are data, not code: each robot is an MJCF file in the top-level
`robots/` folder, e.g. `robots/quadruped.xml` is the robot named "quadruped".
"""

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco

if TYPE_CHECKING:
    from robot3d.terrain import Terrain

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


def load_model(name: str, terrain: "Terrain | None" = None, box_slots: int = 0) -> mujoco.MjModel:
    """Compile a robot's MJCF file into an MjModel (the static description:
    bodies, joints, masses, motors). The changing state (positions,
    velocities, time) lives separately in an MjData.

    terrain: add its boxes to the world (terrain.py). box_slots: add this
    many placeholder boxes, hidden under the floor, for the GPU env to move
    each robot's own terrain tile into (named terrain_slot0, 1, ...)."""
    if terrain is None and box_slots == 0:
        return mujoco.MjModel.from_xml_path(str(robot_path(name)))
    spec = mujoco.MjSpec.from_file(str(robot_path(name)))
    if terrain is not None:
        terrain.add_to(spec)
    for k in range(box_slots):
        g = spec.worldbody.add_geom()
        g.name = f"terrain_slot{k}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = [0.0, 0.0, -1.0]
        g.size = [0.01, 0.01, 0.01]
    return spec.compile()


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
