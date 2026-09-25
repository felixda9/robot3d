"""The shared real-time loop (used by the MuJoCo viewer script and the server)."""

import time

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


def test_reset_keeps_paused_state(sim):
    run_frames(sim, time.perf_counter(), 0.2)
    sim.pause()
    sim.reset()
    assert sim.paused and sim.data.time == 0.0
    assert sim.data.qpos[2] == pytest.approx(0.4)  # torso back at the drop height
