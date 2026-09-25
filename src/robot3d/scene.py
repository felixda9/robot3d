"""Turn MuJoCo's model and state into protocol messages: what the browser needs
to draw and control the robot."""

import math

import mujoco
import numpy as np

from robot3d.protocol import (
    ActuatorInfo,
    CameraInfo,
    FrameMessage,
    GeomInfo,
    GeomType,
    KeyframeInfo,
    SceneMessage,
)

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

# Frames round to 1e-5 (10 micrometers, 0.0006 degrees): invisible on screen,
# and the JSON gets much shorter than full float64 precision.
_FRAME_DECIMALS = 5


class SceneEncoder:
    """Builds SceneMessage / FrameMessage for one model.

    The index arrays (which geoms move, which qpos entry belongs to which
    motor's joint) are computed once here, so building a frame 60 times per
    second is a few numpy lookups.
    """

    def __init__(self, model: mujoco.MjModel, robot: str):
        self.model = model
        self.robot = robot
        self.frame_geoms = np.array(frame_geom_ids(model), dtype=int)
        self.actuators = [_actuator_info(model, i) for i in range(model.nu)]
        # qpos address of each motor's joint (hinge joints use one qpos entry).
        self._joint_qpos = np.array(
            [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)], dtype=int
        )

    def scene(self, data: mujoco.MjData) -> SceneMessage:
        """The static scene. `data` must be up to date (mj_forward or mj_step
        already called); it provides the initial pose of every geom."""
        model = self.model
        dynamic = set(self.frame_geoms.tolist())
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

        # The same default viewpoint MuJoCo's viewer starts with (from the
        # MJCF's <statistic> and <visual><global azimuth/elevation>).
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        camera = CameraInfo(
            lookat=tuple(cam.lookat.tolist()),
            distance=cam.distance,
            azimuth=cam.azimuth,
            elevation=cam.elevation,
            fovy=float(model.vis.global_.fovy),
        )
        keyframes = [
            KeyframeInfo(name=model.key(k).name or f"key{k}", ctrl=model.key_ctrl[k].tolist())
            for k in range(model.nkey)
        ]
        return SceneMessage(
            robot=self.robot,
            timestep=model.opt.timestep,
            geoms=geoms,
            frame_geoms=self.frame_geoms.tolist(),
            camera=camera,
            actuators=self.actuators,
            keyframes=keyframes,
        )

    def frame(self, data: mujoco.MjData) -> FrameMessage:
        """Current state: world pose of each dynamic geom (MuJoCo's geom_xpos
        and geom_xmat) plus each motor's target, actual angle, and torque."""
        r = _FRAME_DECIMALS
        return FrameMessage(
            time=round(float(data.time), 6),
            xpos=np.round(data.geom_xpos[self.frame_geoms], r).ravel().tolist(),
            xmat=np.round(data.geom_xmat[self.frame_geoms], r).ravel().tolist(),
            ctrl=np.round(data.ctrl, r).tolist(),
            joint_pos=np.round(data.qpos[self._joint_qpos], r).tolist(),
            torque=np.round(data.actuator_force, 3).tolist(),
        )


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


def _actuator_info(model: mujoco.MjModel, i: int) -> ActuatorInfo:
    if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
        raise ValueError(f"Actuator {model.actuator(i).name!r} doesn't drive a joint; only joint motors are supported.")
    ctrl_range = model.actuator_ctrlrange[i] if model.actuator_ctrllimited[i] else (-math.pi, math.pi)
    force_range = model.actuator_forcerange[i] if model.actuator_forcelimited[i] else (0.0, 0.0)
    return ActuatorInfo(
        name=model.actuator(i).name,
        joint=model.joint(model.actuator_trnid[i, 0]).name,
        ctrl_range=(round(float(ctrl_range[0]), 6), round(float(ctrl_range[1]), 6)),
        force_range=(round(float(force_range[0]), 6), round(float(force_range[1]), 6)),
    )


def _geom_rgba(model: mujoco.MjModel, g: int) -> tuple[float, float, float, float]:
    # Colors are stored as float32; round after converting so 0.35 stays 0.35.
    rgba = tuple(round(float(c), 4) for c in model.geom_rgba[g])
    mat = model.geom_matid[g]
    if mat >= 0 and np.allclose(rgba, _DEFAULT_RGBA):
        rgba = tuple(round(float(c), 4) for c in model.mat_rgba[mat])
    return rgba
