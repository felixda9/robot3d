"""The shared real-time loop (used by the MuJoCo viewer script and the server)."""

import time

import numpy as np
import pytest

from robot3d.simulation import Simulation

FRAME = 1 / 60


def run_frames(sim: Simulation, start: float, seconds: float) -> None:
    """Call advance() once per 60 fps frame with a fake clock."""
    for k in range(1, round(seconds / FRAME) + 1):
        sim.advance(start + k * FRAME)


@pytest.fixture
def sim():
    return Simulation("quadruped")


def test_sim_time_follows_wall_clock(sim):
    start = time.perf_counter()
    run_frames(sim, start, 1.0)
    assert sim.data.time == pytest.approx(1.0, abs=0.005)


def test_pause_stops_time_and_play_does_not_catch_up(sim):
    start = time.perf_counter()
    run_frames(sim, start, 0.5)
    sim.pause()
    assert sim.advance(start + 3.0) == 0
    t_paused = sim.data.time

    sim.play()
    resume = time.perf_counter()
    run_frames(sim, resume, 0.5)
    # Only the 0.5 s after resuming counts, not the 2.5 s spent paused.
    assert sim.data.time - t_paused == pytest.approx(0.5, abs=0.005)


def test_falling_behind_slows_down_instead_of_bursting(sim):
    start = time.perf_counter()
    steps = sim.advance(start + 10.0)  # a 10 s hiccup
    assert steps == sim.max_steps_per_advance  # capped, not 5000 steps
    assert sim.advance(start + 10.0 + FRAME) <= 9  # then back to ~8 steps per frame


def test_set_ctrl_by_name_with_clamping(sim):
    knee = sim.actuator_names.index("FL_knee")
    sim.set_ctrl({"FL_knee": -0.5})
    assert sim.data.ctrl[knee] == -0.5
    sim.set_ctrl({"FL_knee": 3.0})  # knee range is [-2.6, 0]
    assert sim.data.ctrl[knee] == 0.0
    with pytest.raises(ValueError):
        sim.set_ctrl({"tail": 1.0})


def test_set_ctrl_while_paused_updates_torque(sim):
    run_frames(sim, time.perf_counter(), 1.0)  # standing
    sim.pause()
    knee = sim.actuator_names.index("FL_knee")
    sim.set_ctrl({"FL_knee": -0.3})  # far from the current ~-1.4: motor pulls hard
    assert abs(sim.data.actuator_force[knee]) > 5.0


def test_glide_moves_named_motors_together(sim):
    hip, knee = sim.actuator_names.index("FL_hip"), sim.actuator_names.index("FL_knee")
    start = sim.data.ctrl.copy()
    goal = {"FL_hip": 0.2, "FL_knee": -0.4}
    sim.set_ctrl(goal, duration=0.5)
    assert np.array_equal(sim.data.ctrl, start)  # nothing jumps

    for _ in range(round(0.25 / sim.model.opt.timestep)):
        sim.step()
    # Halfway in time = halfway in value (smoothstep(0.5) = 0.5), for both motors.
    for i, name in [(hip, "FL_hip"), (knee, "FL_knee")]:
        assert sim.data.ctrl[i] == pytest.approx((start[i] + goal[name]) / 2, abs=0.02)

    for _ in range(round(0.3 / sim.model.opt.timestep)):
        sim.step()
    assert sim.data.ctrl[hip] == pytest.approx(0.2) and sim.data.ctrl[knee] == pytest.approx(-0.4)
    others = [i for i in range(sim.model.nu) if i not in (hip, knee)]
    assert np.array_equal(sim.data.ctrl[others], start[others])


def test_instant_set_cancels_a_glide(sim):
    knee = sim.actuator_names.index("FL_knee")
    sim.set_ctrl({"FL_knee": -0.4}, duration=1.0)
    sim.step()
    sim.set_ctrl({"FL_knee": -2.0})  # e.g. the user grabs the slider mid-glide
    for _ in range(100):
        sim.step()
    assert sim.data.ctrl[knee] == -2.0


def test_reset_keeps_paused_state(sim):
    run_frames(sim, time.perf_counter(), 0.2)
    sim.pause()
    sim.reset()
    assert sim.paused and sim.data.time == 0.0
    assert sim.data.qpos[2] == pytest.approx(0.4)  # torso back at the drop height


# ------------------------------------------------------ grab and push (mouse)


def steps(sim: Simulation, seconds: float) -> None:
    for _ in range(round(seconds / sim.model.opt.timestep)):
        sim.step()


def torso_geom(sim: Simulation) -> int:
    return next(g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == 1)


def test_grab_lifts_the_robot_and_release_drops_it(sim):
    steps(sim, 1.0)  # settle standing
    geom = torso_geom(sim)
    start = sim.data.qpos[:3].copy()
    sim.grab(geom, point=[0, 0, 0], target=sim.data.geom_xpos[geom] + [0, 0, 0.3])
    steps(sim, 1.5)
    lifted = sim.data.qpos[2] - start[2]
    assert lifted > 0.2  # pulled up ~0.3 m (a spring sags a little under the weight)
    assert np.linalg.norm(sim.data.qpos[:2] - start[:2]) < 0.05  # straight up, not sideways

    sim.release()
    steps(sim, 1.0)
    assert not sim.data.xfrc_applied.any()
    assert sim.data.qpos[2] < start[2] + 0.05  # back on the floor


def test_push_shoves_briefly(sim):
    steps(sim, 1.0)
    geom = torso_geom(sim)
    sim.push(geom, point=[0, 0, 0], direction=[0, 2, 0], force=60.0)  # direction length doesn't matter
    steps(sim, Simulation.PUSH_SECONDS)
    # Impulse 60 N x 0.1 s = 6 N s; the robot resists with its feet on the
    # floor, so it moves sideways, but slower than 6 / mass.
    vy = sim.data.qvel[1]
    assert 0.2 < vy < 6.0 / sim.robot_mass + 0.05
    steps(sim, 0.01)
    assert not sim.data.xfrc_applied.any()  # the push is over


def test_grab_and_push_only_move_the_robot(sim):
    floor = next(g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == 0)
    with pytest.raises(ValueError, match="static world"):
        sim.grab(floor, [0, 0, 0], [0, 0, 1])
    with pytest.raises(ValueError, match="zero"):
        sim.push(torso_geom(sim), [0, 0, 0], [0, 0, 0], 50.0)


def test_reset_lets_go(sim):
    sim.grab(torso_geom(sim), [0, 0, 0], [0, 0, 2])
    steps(sim, 0.1)
    sim.reset()
    steps(sim, 0.1)
    assert not sim.data.xfrc_applied.any()
