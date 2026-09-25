"""Turn MuJoCo's model and state into protocol messages: what the browser needs to draw."""

import mujoco
import numpy as np

from robot3d.protocol import CameraInfo, FrameMessage, GeomInfo, GeomType, SceneMessage

_GEOM_TYPES: dict[int, GeomType] = {
    mujoco.mjtGeom.mjGEOM_PLANE: "plane",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
}

# MuJoCo's default geom color. A geom still at this default shows its
# material's color instead (same rule as MuJoCo's renderer).
_DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)

# Frames round poses to 10 micrometers / 1e-5: invisible on screen, and the
# JSON gets much shorter than full float64 precision.
_FRAME_DECIMALS = 5


def frame_geom_ids(model: mujoco.MjModel) -> list[int]:
    """Geoms that can move, i.e. are attached to a body not welded to the world.

    body_weldid is the body's nearest ancestor-or-self that has joints (0 =
    the world). Static scenery (the floor, later obstacles) is sent once and
    never streamed. Mocap bodies move without joints, so include them too.
    """
    ids = []
    for g in range(model.ngeom):
        body = model.geom_bodyid[g]
        if model.body_weldid[body] != 0 or model.body_mocapid[body] >= 0:
            ids.append(g)
    return ids


def build_scene(model: mujoco.MjModel, data: mujoco.MjData, robot: str) -> SceneMessage:
    """The static scene: every geom's shape, size, color and parent body.

    `data` must be up to date (mj_forward or mj_step already called); it
    provides the initial pose of every geom.
    """
    frame_geoms = frame_geom_ids(model)
    dynamic = set(frame_geoms)
    geoms = []
    for g in range(model.ngeom):
        geom_type = _GEOM_TYPES.get(int(model.geom_type[g]))
        if geom_type is None:
            raise ValueError(
                f"Geom {model.geom(g).name or g!r} has type {mujoco.mjtGeom(model.geom_type[g]).name}; "
                "only primitive shapes are supported (see CLAUDE.md)."
            )
        geoms.append(
            GeomInfo(
                id=g,
                name=model.geom(g).name,
                type=geom_type,
                size=tuple(model.geom_size[g].tolist()),
                rgba=_geom_rgba(model, g),
                body=model.body(model.geom_bodyid[g]).name,
                group=int(model.geom_group[g]),
                dynamic=g in dynamic,
                pos=tuple(data.geom_xpos[g].tolist()),
                mat=tuple(data.geom_xmat[g].tolist()),
            )
        )

    # The same default viewpoint MuJoCo's viewer starts with (from the MJCF's
    # <statistic> and <visual><global azimuth/elevation>).
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    camera = CameraInfo(
        lookat=tuple(cam.lookat.tolist()),
        distance=cam.distance,
        azimuth=cam.azimuth,
        elevation=cam.elevation,
        fovy=float(model.vis.global_.fovy),
    )
    return SceneMessage(
        robot=robot,
        timestep=model.opt.timestep,
        geoms=geoms,
        frame_geoms=frame_geoms,
        camera=camera,
    )


def build_frame(data: mujoco.MjData, frame_geoms: np.ndarray) -> FrameMessage:
    """Current world pose of each dynamic geom (MuJoCo's geom_xpos and geom_xmat)."""
    return FrameMessage(
        time=round(float(data.time), 6),
        xpos=np.round(data.geom_xpos[frame_geoms], _FRAME_DECIMALS).ravel().tolist(),
        xmat=np.round(data.geom_xmat[frame_geoms], _FRAME_DECIMALS).ravel().tolist(),
    )


def _geom_rgba(model: mujoco.MjModel, g: int) -> tuple[float, float, float, float]:
    # Colors are stored as float32; round after converting so 0.35 stays 0.35.
    rgba = tuple(round(float(c), 4) for c in model.geom_rgba[g])
    mat = model.geom_matid[g]
    if mat >= 0 and np.allclose(rgba, _DEFAULT_RGBA):
        rgba = tuple(round(float(c), 4) for c in model.mat_rgba[mat])
    return rgba
