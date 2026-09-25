"""Training pipeline end to end: a tiny run, its files, and replaying its checkpoints."""

import json

import numpy as np
import pytest

from robot3d.policy import PolicyController, evaluate, find_checkpoint, list_checkpoints
from robot3d.simulation import Simulation
from robot3d.training import default_num_envs


def test_default_env_count_follows_cpu():
    import psutil

    assert default_num_envs() == max(1, psutil.cpu_count(logical=True) - 2)


def test_run_folder_contents(tiny_run):
    info = json.loads((tiny_run / "run.json").read_text())
    assert info["robot"] == "quadruped" and info["n_envs"] == 2
    assert info["steps_done"] >= 256 and not info["interrupted"]
    assert info["walk_config"]["control_dt"] == 0.02
    assert list((tiny_run / "tb").rglob("events.out.tfevents.*")), "no TensorBoard log"

    checkpoints = list_checkpoints(tiny_run)
    assert len(checkpoints) >= 2
    assert [c.steps for c in checkpoints] == sorted(c.steps for c in checkpoints)
    for checkpoint in checkpoints:
        assert checkpoint.normalizer_path.is_file()


def test_find_checkpoint(tiny_run):
    newest = list_checkpoints(tiny_run)[-1]
    assert find_checkpoint(tiny_run) == newest
    assert find_checkpoint(tiny_run / "checkpoints") == newest
    first = list_checkpoints(tiny_run)[0]
    assert find_checkpoint(first.model_path) == first
    assert first.label == f"tiny @ {first.steps:,} steps"
    with pytest.raises(FileNotFoundError):
        find_checkpoint(tiny_run / "tb")


def test_evaluate_runs_full_episodes(tiny_run):
    results = evaluate(find_checkpoint(tiny_run), episodes=2)
    assert len(results) == 2
    for r in results:
        assert 0 < r["seconds"] <= 20.0
        assert set(r) == {
            "return", "seconds", "distance", "speed", "fell", "upright",
            "duty_factor", "airborne", "diagonal_sync", "cadence",
        }


def test_gait_numbers_tell_a_trot_from_a_run():
    from robot3d.envs import WalkEnv
    from robot3d.policy import gait_numbers

    task = WalkEnv().task
    (a1, a2), (b1, b2) = task.diagonal_pairs
    steps = 100  # 2 s at 50 Hz
    # Trot-walk: one diagonal pair down for 12 steps, overlapping 2 steps with
    # the other pair (both down), period 20 steps = 0.4 s.
    trot = np.zeros((steps, 4), dtype=bool)
    phase = np.arange(steps) % 20
    trot[:, [a1, a2]] = (phase < 12)[:, None]
    trot[:, [b1, b2]] = ((phase >= 10) | (phase < 2))[:, None]
    g = gait_numbers(trot, task)
    assert g["duty_factor"] == pytest.approx(0.6)
    assert g["airborne"] == 0.0
    assert g["diagonal_sync"] == 1.0
    assert g["cadence"] == pytest.approx(2.5, abs=0.3)  # 1 touchdown per 0.4 s

    run = trot.copy()
    run[phase >= 16] = False  # a flight phase: all four feet up 20% of the time
    g = gait_numbers(run, task)
    assert g["airborne"] == pytest.approx(0.2)
    assert g["duty_factor"] < 0.6

    assert gait_numbers(trot[:1], task)["duty_factor"] is None  # fell right away: no gait


def test_policy_drives_a_live_simulation(tiny_run):
    sim = Simulation("quadruped")
    controller = PolicyController(find_checkpoint(tiny_run), sim.model)
    sim.set_controller(controller)

    # Starts standing, like a training episode (not dropped from 0.4 m).
    assert sim.data.qpos[2] == pytest.approx(controller.task.standing_height, abs=0.01)
    home = controller.task.home_ctrl

    targets = []
    for _ in range(40):  # 4 control steps
        sim.step()
        targets.append(sim.data.ctrl.copy())
    # The policy sets new targets every 10 physics steps (50 Hz), holding them in between.
    assert all(np.array_equal(targets[0], t) for t in targets[:10])
    assert not np.allclose(targets[0], home)  # it's acting, not just holding home

    with pytest.raises(RuntimeError):
        sim.set_ctrl({"FL_knee": -1.0})  # manual control is locked while the policy drives
    sim.use_controller(False)
    sim.set_ctrl({"FL_knee": -1.0})
    assert sim.data.ctrl[1] == -1.0
