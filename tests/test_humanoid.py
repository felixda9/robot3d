"""The humanoid robot (milestone 8): structure, standing, and its pose presets."""

import mujoco
import numpy as np
import pytest

from robot3d.robots import load_model, reset_to_keyframe
from robot3d.simulation import Simulation

LEG = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
ARM = ["shoulder_pitch", "shoulder_roll", "elbow"]
MOTORS = ([f"L_{j}" for j in LEG] + [f"R_{j}" for j in LEG] + ["waist_yaw", "waist_roll", "waist_pitch"]
          + [f"L_{j}" for j in ARM] + [f"R_{j}" for j in ARM])


@pytest.fixture(scope="module")
def model():
    return load_model("humanoid")


def test_structure(model):
    assert [model.actuator(i).name for i in range(model.nu)] == MOTORS  # 21 motors, in this order
    for i in range(model.nu):
        assert model.actuator(i).name == model.joint(model.actuator_trnid[i, 0]).name  # each drives its joint
    assert model.nq == 7 + 21
    assert 18.0 < model.body_subtreemass[1] < 20.0  # ~19 kg
    assert {mujoco.mjtGeom(t) for t in model.geom_type} <= {
        mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_BOX,
    }


def test_stands_on_its_feet(model):
    data = mujoco.MjData(model)
    reset_to_keyframe(model, data, "home")
    for _ in range(round(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert all(w.number == 0 for w in data.warning)
    assert 0.5 < data.qpos[2] < 0.56 and data.xmat[1][8] > 0.99  # pelvis ~0.53 m, level
    assert np.abs(data.qvel).max() < 0.05  # settled
    touching = {int(g) for c in data.contact[: data.ncon] for g in (c.geom1, c.geom2)} - {model.geom("floor").id}
    assert touching == {model.geom("L_foot").id, model.geom("R_foot").id}
    head_top = data.geom_xpos[model.geom("head").id][2] + model.geom_size[model.geom("head").id][0]
    assert 0.95 < head_top < 1.05  # child-size: ~1.0 m


@pytest.mark.parametrize("pose", ["crouch", "arms_out"])
def test_pose_presets_stay_upright(pose):
    sim = Simulation("humanoid")
    for _ in range(round(1.5 / sim.model.opt.timestep)):
        sim.step()
    lowest_up = 1.0
    for preset, seconds in ((pose, 3.0), ("home", 5.0)):
        sim.set_ctrl(dict(zip(sim.actuator_names, sim.model.key(preset).ctrl)), duration=0.8)
        for _ in range(round(seconds / sim.model.opt.timestep)):
            sim.step()
            lowest_up = min(lowest_up, sim.data.xmat[1][8])
    assert lowest_up > 0.9, "tipped over on the way"
    assert np.abs(sim.data.qvel).max() < 0.05
