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


def test_the_getup_policy_takes_over_after_a_fall(tiny_run, tiny_stand_run, tiny_getup_run):
    import mujoco

    from robot3d.policy import Behaviors

    sim = Simulation("quadruped")
    walker, stander, getup = (PolicyController(find_checkpoint(r), sim.model)
                              for r in (tiny_run, tiny_stand_run, tiny_getup_run))
    behaviors = Behaviors()
    behaviors.install(walker, "walker")
    assert behaviors.mode == "walk"
    behaviors.install(stander, "stander")
    assert behaviors.mode == "stand"  # loading a walk or stand policy switches to its mode
    behaviors.install(getup, "getup")
    assert behaviors.mode == "stand"  # a get-up policy doesn't
    sim.set_controller(behaviors)  # restarts standing

    def knock_over():
        sim.data.qpos[3:7] = [0, 1, 0, 0]  # upside down
        mujoco.mj_forward(sim.model, sim.data)

    for mode, policy in (("walk", walker), ("stand", stander)):
        behaviors.set_mode(mode)
        sim.reset()
        behaviors.act(sim.data)
        assert behaviors.active is policy and not behaviors.recovering
        knock_over()
        behaviors.act(sim.data)
        assert behaviors.recovering and behaviors.active is getup, mode  # the get-up policy takes over
        policy.reset_state(sim.data)  # (placed back on its feet)
        for _ in range(24):  # 0.48 s standing steady: not yet
            behaviors.act(sim.data)
        assert behaviors.recovering
        behaviors.act(sim.data)  # 0.5 s: back to the mode's policy
        assert behaviors.active is policy and not behaviors.recovering

    only_getup = Behaviors()
    with pytest.raises(ValueError, match="no stand policy"):
        only_getup.set_mode("stand")
    only_getup.install(getup, "getup")
    assert only_getup.mode == "stand" and only_getup.active is getup  # it can stand too
    with pytest.raises(ValueError, match="no walk policy"):
        only_getup.set_mode("walk")


def test_jump_on_command(tiny_run, tiny_jump_run):
    from robot3d.policy import Behaviors

    sim = Simulation("quadruped")
    walker = PolicyController(find_checkpoint(tiny_run), sim.model)
    jumper = PolicyController(find_checkpoint(tiny_jump_run), sim.model)
    behaviors = Behaviors()
    with pytest.raises(ValueError, match="no jump policy"):
        behaviors.jump()
    behaviors.install(walker, "walker")
    behaviors.install(jumper, "jumper")
    assert behaviors.mode == "walk"  # loading a jump policy keeps the mode
    sim.set_controller(behaviors)
    behaviors.jump()
    assert behaviors.jumping and behaviors.active is jumper and jumper.steps == 0  # its clock starts
    for _ in range(jumper.task.max_steps - 1):
        behaviors.act(sim.data)  # (the tiny policy doesn't really jump, so it never lands)
    assert behaviors.jumping
    behaviors.act(sim.data)  # its episode length is over: back to walking
    assert not behaviors.jumping and behaviors.active is walker


def test_steering_a_policy(tiny_run, tiny_steer_run):
    from robot3d.policy import Behaviors

    sim = Simulation("quadruped")
    behaviors = Behaviors()
    behaviors.install(PolicyController(find_checkpoint(tiny_run), sim.model), "plain")
    assert not behaviors.steerable and behaviors.command_limits() == (0, 0, 0, 0)
    with pytest.raises(ValueError, match="steering"):
        behaviors.set_command(0.3, 0, 0)
    steer = PolicyController(find_checkpoint(tiny_steer_run), sim.model)
    assert steer.steerable and np.allclose(steer.command, 0)  # starts standing still
    behaviors.install(steer, "steer")
    behaviors.set_command(5.0, 0.1, -9.0)
    c = steer.task.config
    assert np.allclose(steer.command, [c.command_max_forward, 0.1, -c.command_max_turn])  # clamped
    sim.set_controller(behaviors)
    for _ in range(20):
        sim.step()  # its observation carries the command
    assert np.isfinite(sim.data.qpos).all()


def test_skill_tests(tiny_run, tiny_getup_run):
    from robot3d.policy import skill_test

    walk = skill_test(find_checkpoint(tiny_run))
    assert 0.0 <= walk["skill"] <= 1.0 and walk["skill_test"].startswith("v2: pushes of")
    getup = skill_test(find_checkpoint(tiny_getup_run))
    assert 0.0 <= getup["skill"] <= 1.0 and "24 fallen starts" in getup["skill_test"]
    assert walk == skill_test(find_checkpoint(tiny_run))  # deterministic: comparable across checkpoints


def test_jump_skill_test(tiny_jump_run):
    from robot3d.policy import skill_test

    jump = skill_test(find_checkpoint(tiny_jump_run))
    assert 0.0 <= jump["skill"] <= 1.0 and "8 jumps" in jump["skill_test"] and "median height" in jump["skill_test"]


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
