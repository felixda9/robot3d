"""Headless physics sanity checks for the quadrupeds (8 and 12 motors).

Milestone 1 success criterion, as code: dropped onto the floor, the robot
lands on its feet and settles without jittering or exploding.
"""

import mujoco
import numpy as np
import pytest

from robot3d.robots import load_model, reset_to_keyframe
from robot3d.simulation import Simulation

LEGS = ["FL", "FR", "RL", "RR"]
# Robot -> its joints per leg (each driven by a motor of the same name).
ROBOTS = {"quadruped": ("hip", "knee"), "quadruped12": ("roll", "hip", "knee")}
PRIMITIVE_GEOMS = {
    mujoco.mjtGeom.mjGEOM_PLANE,
    mujoco.mjtGeom.mjGEOM_SPHERE,
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_BOX,
}


@pytest.fixture(scope="module", params=list(ROBOTS))
def robot(request):
    return request.param


@pytest.fixture(scope="module")
def model(robot):
    return load_model(robot)


def test_structure(robot, model):
    parts = ROBOTS[robot]
    n = 4 * len(parts)
    # 1 free joint (torso) + one hinge per motor: 7 qpos for the torso (xyz + quaternion) + the angles.
    assert model.njnt == 1 + n
    assert model.nq == 7 + n
    assert model.nu == n
    for leg in LEGS:
        for part in parts:
            name = f"{leg}_{part}"
            jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_HINGE
            assert model.actuator_trnid[act, 0] == jnt, f"motor {name} drives the wrong joint"
    # Only primitive shapes, so the web viewer never needs meshes.
    assert {mujoco.mjtGeom(t) for t in model.geom_type} <= PRIMITIVE_GEOMS


def simulate(model, data, seconds):
    """Step the physics; record torso height and the fastest joint speed each step."""
    n = int(round(seconds / model.opt.timestep))
    torso_z = np.empty(n)
    max_speed = np.empty(n)
    for i in range(n):
        mujoco.mj_step(model, data)
        torso_z[i] = data.qpos[2]
        max_speed[i] = np.abs(data.qvel).max()
    return torso_z, max_speed


def reset_home(model, data):
    reset_to_keyframe(model, data, "home")


def reset_straight_legs(model, data):
    # qpos0 = the pose as written in the MJCF: legs straight, motors targeting 0.
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


@pytest.mark.parametrize(
    "reset, height_range",
    [
        (reset_home, (0.24, 0.28)),  # crouched stance
        (reset_straight_legs, (0.32, 0.35)),  # standing on stilts
    ],
    ids=["home", "straight_legs"],
)
def test_drop_and_settle(model, reset, height_range):
    data = mujoco.MjData(model)
    reset(model, data)

    torso_z, max_speed = simulate(model, data, seconds=2.0)
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot") for leg in LEGS]
    feet_start = data.geom_xpos[feet].copy()
    torso_z2, max_speed2 = simulate(model, data, seconds=1.0)

    # Not exploding: all numbers finite, no solver warnings, speeds stay sane.
    assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    assert all(w.number == 0 for w in data.warning), "MuJoCo raised a warning"
    assert max(max_speed.max(), max_speed2.max()) < 20.0

    # Landed upright at the expected height.
    lo, hi = height_range
    assert lo < torso_z2[-1] < hi
    torso_up = data.xmat[model.body("torso").id].reshape(3, 3)[:, 2]
    assert torso_up[2] > 0.99, "torso is tilted"

    # Standing on its feet: every floor contact is a foot.
    touching = {int(g) for c in data.contact[: data.ncon] for g in (c.geom1, c.geom2)}
    touching.discard(model.geom("floor").id)
    assert touching == set(feet)

    # Settled, no jitter: over the final second almost nothing moves.
    assert max_speed2.max() < 1e-2, "still moving / jittering"
    assert np.ptp(torso_z2) < 1e-3, "torso height still changing"
    assert np.abs(data.geom_xpos[feet] - feet_start).max() < 2e-3, "feet are sliding"


PRESETS = ["home", "crouch", "tall", "sit"]
PRESET_GLIDE = 0.8  # seconds; same as PRESET_DURATION in web/src/motors.ts


PRESETS12 = [*PRESETS, "wide"]


@pytest.mark.parametrize(
    "robot, start, goal",
    [("quadruped", a, b) for a in PRESETS for b in PRESETS if a != b]
    + [("quadruped12", a, b) for a in PRESETS12 for b in PRESETS12 if a != b],
    ids=lambda p: p,
)
def test_pose_preset_transition_is_safe(robot, start, goal):
    """The UI's pose buttons glide the motor targets to a keyframe's ctrl.
    Every preset-to-preset move must stay upright the whole way and end at
    rest, without any motor maxing out. (Jumping the targets instantly
    instead flips the robot on crouch -> tall.)"""
    sim = Simulation(robot)
    torso = sim.model.body("torso").id

    def steps(seconds):
        return range(round(seconds / sim.model.opt.timestep))

    for _ in steps(1.5):  # stand
        sim.step()
    for preset, seconds in [(start, 3.0), (goal, 5.0)]:
        targets = dict(zip(sim.actuator_names, sim.model.key(preset).ctrl))
        sim.set_ctrl(targets, duration=PRESET_GLIDE)
        lowest_up, speeds = 1.0, []
        for _ in steps(seconds):
            sim.step()
            lowest_up = min(lowest_up, sim.data.xmat[torso][8])  # [8] = z of torso's up axis
            speeds.append(np.abs(sim.data.qvel).max())

    assert all(w.number == 0 for w in sim.data.warning), "MuJoCo raised a warning"
    assert lowest_up > 0.8, "tipped over on the way"  # sit leans back ~32 degrees: cos = 0.85
    # 5 s: "tall" rocks back and forth for a while after extending (nearly
    # straight legs damp that sway weakly); it fades smoothly and is still by ~4 s.
    assert max(speeds[-500:]) < 1e-2, "not at rest"
    torque_limit = sim.model.actuator_forcerange[:, 1]
    assert np.all(np.abs(sim.data.actuator_force) < 0.95 * torque_limit), "a motor is at its torque limit"
