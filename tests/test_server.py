"""End-to-end tests of the WebSocket server: a real app with its simulation
thread, driven through FastAPI's TestClient (no browser needed)."""

import time

import pytest
from fastapi.testclient import TestClient

from robot3d.protocol import server_message_adapter
from robot3d.server import create_app

N_GEOMS = 15  # floor + torso + head + 4 legs x (thigh, shin, foot)
N_FRAME_GEOMS = 14  # everything except the static floor
ACTUATORS = ["FL_hip", "FL_knee", "FR_hip", "FR_knee", "RL_hip", "RL_knee", "RR_hip", "RR_knee"]
HOME_CTRL = [0.7, -1.4] * 4
FL_KNEE = ACTUATORS.index("FL_knee")


@pytest.fixture
def client():
    with TestClient(create_app("quadruped")) as client:  # starts the sim thread
        yield client


def receive(ws):
    """Next message, validated against the protocol models."""
    return server_message_adapter.validate_json(ws.receive_text())


def receive_until(ws, msg_type, predicate=lambda m: True, limit=1000):
    for _ in range(limit):
        message = receive(ws)
        if message.type == msg_type and predicate(message):
            return message
    raise AssertionError(f"no matching '{msg_type}' message within {limit} messages")


def test_connect_sends_scene_then_status_then_frame(client):
    with client.websocket_connect("/ws") as ws:
        scene = receive(ws)
        assert scene.type == "scene"
        assert scene.robot == "quadruped"
        assert len(scene.geoms) == N_GEOMS
        assert len(scene.frame_geoms) == N_FRAME_GEOMS
        floor = scene.geoms[0]
        assert (floor.name, floor.type, floor.dynamic) == ("floor", "plane", False)
        assert {g.type for g in scene.geoms} == {"plane", "box", "capsule", "sphere"}

        assert [a.name for a in scene.actuators] == ACTUATORS
        assert all(a.name == a.joint for a in scene.actuators)
        knee = scene.actuators[1]
        assert knee.ctrl_range == (-2.6, 0.0) and knee.force_range == (-10.0, 10.0)
        assert [k.name for k in scene.keyframes] == ["home", "crouch", "tall", "sit"]
        assert scene.keyframes[0].ctrl == HOME_CTRL

        status = receive(ws)
        assert status.type == "status" and status.paused is False

        frame = receive(ws)
        assert frame.type == "frame"
        assert len(frame.xpos) == 3 * N_FRAME_GEOMS
        assert len(frame.xmat) == 9 * N_FRAME_GEOMS
        assert frame.ctrl == HOME_CTRL
        assert len(frame.joint_pos) == len(frame.torque) == len(ACTUATORS)


def test_frames_stream_at_60fps_in_real_time(client):
    with client.websocket_connect("/ws") as ws:
        first = receive_until(ws, "frame", lambda f: f.time > 0)
        wall_start = time.perf_counter()
        last = first
        for _ in range(60):
            last = receive_until(ws, "frame")
        wall = time.perf_counter() - wall_start
        sim = last.time - first.time
        assert 0.7 < wall < 1.5, f"60 frames took {wall:.2f} s (expected ~1 s at 60 fps)"
        assert abs(sim - wall) < 0.15, f"sim advanced {sim:.2f} s in {wall:.2f} s of wall time"


def test_pause_reset_play(client):
    with client.websocket_connect("/ws") as ws:
        receive_until(ws, "frame", lambda f: f.time > 0.2)

        ws.send_json({"type": "pause"})
        assert receive_until(ws, "status").paused is True

        # Reset while paused: exactly one new frame, back at t = 0.
        ws.send_json({"type": "reset"})
        frame = receive_until(ws, "frame", lambda f: f.time == 0.0)
        torso_z = frame.xpos[2]  # first frame geom is the torso
        assert torso_z == pytest.approx(0.4)

        ws.send_json({"type": "play"})
        assert receive_until(ws, "status").paused is False
        assert receive_until(ws, "frame", lambda f: f.time > 0.1).time > 0.1


def test_set_ctrl_moves_the_joint(client):
    with client.websocket_connect("/ws") as ws:
        start = receive_until(ws, "frame", lambda f: f.time > 1.0)  # standing
        ws.send_json({"type": "set_ctrl", "ctrl": {"FL_knee": -0.5}, "duration": 0})
        # One second later the motor has pulled the knee to (nearly) the target.
        frame = receive_until(ws, "frame", lambda f: f.time > start.time + 1.0)
        assert frame.ctrl[FL_KNEE] == -0.5
        assert frame.joint_pos[FL_KNEE] == pytest.approx(-0.5, abs=0.1)
        others = [i for i in range(len(ACTUATORS)) if i != FL_KNEE]
        assert [frame.ctrl[i] for i in others] == [HOME_CTRL[i] for i in others]


def test_set_ctrl_while_paused_clamps_and_reset_restores(client):
    with client.websocket_connect("/ws") as ws:
        receive_until(ws, "frame", lambda f: f.time > 0.2)
        ws.send_json({"type": "pause"})
        receive_until(ws, "status")

        # Out of range (knee max is 0): clamped. A frame arrives even though paused.
        ws.send_json({"type": "set_ctrl", "ctrl": {"FL_knee": 5.0, "FL_hip": 0.1}, "duration": 0})
        first = receive_until(ws, "frame", lambda f: f.ctrl[FL_KNEE] == 0.0)
        assert first.ctrl[0] == pytest.approx(0.1)

        # Still paused: new targets produce frames, but time doesn't move.
        ws.send_json({"type": "set_ctrl", "ctrl": {"FL_knee": -1.0}, "duration": 0})
        second = receive_until(ws, "frame", lambda f: f.ctrl[FL_KNEE] == -1.0)
        assert second.time == first.time

        ws.send_json({"type": "reset"})
        assert receive_until(ws, "frame", lambda f: f.time == 0.0).ctrl == HOME_CTRL


def test_bad_messages_get_an_error_reply(client):
    with client.websocket_connect("/ws") as ws:
        for bad in [
            "not json",
            '{"type": "fly"}',
            '{"type": "play", "speed": 2}',
            '{"type": "set_ctrl", "ctrl": {"FL_knee": NaN}, "duration": 0}',  # would poison the physics
            '{"type": "set_ctrl", "ctrl": {"FL_knee": "high"}, "duration": 0}',
            '{"type": "set_ctrl", "ctrl": {"FL_knee": -1.0}}',  # duration is required
            '{"type": "set_ctrl", "ctrl": {"FL_knee": -1.0}, "duration": -1}',
        ]:
            ws.send_text(bad)
            error = receive_until(ws, "error")
            assert error.message.startswith("invalid message"), bad
        ws.send_json({"type": "set_ctrl", "ctrl": {"tail": 1.0}, "duration": 0})
        assert receive_until(ws, "error").message == "unknown actuator(s): tail"
        ws.send_bytes(b"\x00\x01")
        assert "JSON text" in receive_until(ws, "error").message


def test_without_a_policy(client):
    with client.websocket_connect("/ws") as ws:
        status = receive_until(ws, "status")
        assert status.walk_policy == "" and status.policy_active is False
        ws.send_json({"type": "use_policy", "active": True})
        assert "no policy loaded" in receive_until(ws, "error").message


def test_policy_drives_and_can_be_switched_off(tiny_run):
    with TestClient(create_app("quadruped", policy=tiny_run)) as client:
        with client.websocket_connect("/ws") as ws:
            scene = receive(ws)
            status = receive(ws)
            assert status.walk_policy.startswith("tiny @ ") and status.policy_active is True
            # Starts standing (as in training), not dropped from 0.4 m.
            torso_z = receive_until(ws, "frame").xpos[2]
            assert torso_z == pytest.approx(0.26, abs=0.02)

            ws.send_json({"type": "set_ctrl", "ctrl": {"FL_knee": -1.0}, "duration": 0})
            assert "policy is driving" in receive_until(ws, "error").message

            ws.send_json({"type": "use_policy", "active": False})
            assert receive_until(ws, "status").policy_active is False
            ws.send_json({"type": "set_ctrl", "ctrl": {"FL_knee": -1.0}, "duration": 0})
            frame = receive_until(ws, "frame", lambda f: f.ctrl[FL_KNEE] == -1.0)
            assert len(frame.ctrl) == len(scene.actuators)


def test_dashboard_api(tiny_run):
    with TestClient(create_app("quadruped", runs_dir=tiny_run.parent)) as client:
        runs = client.get("/api/runs").json()
        assert [r["name"] for r in runs] == ["tiny"] and runs[0]["status"] == "finished"

        detail = client.get("/api/runs/tiny").json()
        assert len(detail["checkpoints"]) >= 2 and detail["evaluating"] == 0 and detail["evaluation_error"] == ""
        for checkpoint in detail["checkpoints"]:
            assert checkpoint["evaluation"] is None or checkpoint["evaluation"]["episodes"] > 0

        scalars = client.get("/api/runs/tiny/scalars", params={"tags": "train/std,rollout/ep_len_mean"}).json()
        assert [s["tag"] for s in scalars["series"]] == ["train/std", "rollout/ep_len_mean"]

        assert client.get("/api/runs/nope").status_code == 404
        assert client.get("/api/runs/..%2F..%2Fsecret").status_code == 404

        # Evaluate every checkpoint in the background, then they all have results.
        client.post("/api/runs/tiny/evaluate")
        deadline = time.time() + 60
        while client.get("/api/runs/tiny").json()["evaluating"] > 0 and time.time() < deadline:
            time.sleep(0.2)
        detail = client.get("/api/runs/tiny").json()
        assert all(c["evaluation"] is not None for c in detail["checkpoints"])
        assert client.post("/api/runs/tiny/evaluate").json() == {"queued": 0}  # nothing left to do


def test_load_policy_over_websocket(tiny_run):
    with TestClient(create_app("quadruped", runs_dir=tiny_run.parent)) as client:
        with client.websocket_connect("/ws") as ws:
            assert receive_until(ws, "status").walk_policy == ""
            first = client.get("/api/runs/tiny").json()["checkpoints"][0]["name"]

            ws.send_json({"type": "load_policy", "run": "tiny", "checkpoint": first})
            status = receive_until(ws, "status", lambda s: s.walk_policy != "")
            assert status.walk_policy.startswith("tiny @ ") and status.policy_active is True

            ws.send_json({"type": "load_policy", "run": "tiny", "checkpoint": "step_999999999"})
            assert "no checkpoint" in receive_until(ws, "error").message
            ws.send_json({"type": "load_policy", "run": "../tiny", "checkpoint": first})
            assert receive_until(ws, "error").message.startswith("invalid message")


def test_run_from_newer_code_explains_itself(tiny_run, tmp_path):
    """A server started before a code update can't read runs the new code
    trained. Watch and Evaluate must say so, not fail silently."""
    import json
    import shutil

    run_dir = tmp_path / "future"
    shutil.copytree(tiny_run, run_dir)
    for cached in run_dir.glob("checkpoints/*_eval.json"):
        cached.unlink()
    info = json.loads((run_dir / "run.json").read_text())
    info["walk_config"]["some_new_setting"] = 1.0
    (run_dir / "run.json").write_text(json.dumps(info))

    with TestClient(create_app("quadruped", runs_dir=tmp_path)) as client:
        assert client.post("/api/runs/future/evaluate").json()["queued"] >= 2
        deadline = time.time() + 30
        while client.get("/api/runs/future").json()["evaluating"] > 0 and time.time() < deadline:
            time.sleep(0.1)
        error = client.get("/api/runs/future").json()["evaluation_error"]
        assert "some_new_setting" in error and "restart" in error

        with client.websocket_connect("/ws") as ws:
            first = client.get("/api/runs/future").json()["checkpoints"][0]["name"]
            ws.send_json({"type": "load_policy", "run": "future", "checkpoint": first})
            message = receive_until(ws, "error").message
            assert "some_new_setting" in message and "restart" in message


def test_grab_and_push_over_websocket(client):
    runner = client.app.state.runner
    sim = runner.sim
    torso = next(g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == 1)
    floor = next(g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == 0)
    with client.websocket_connect("/ws") as ws:
        receive_until(ws, "frame")
        ws.send_json({"type": "grab", "geom": floor, "point": [0, 0, 0], "target": [0, 0, 1]})
        assert "static world" in receive_until(ws, "error").message
        ws.send_json({"type": "push", "geom": torso, "point": [0, 0, 0], "direction": [1, 0, 0], "force": -5})
        assert receive_until(ws, "error").message.startswith("invalid message")
        ws.send_json({"type": "push", "geom": torso, "point": [0, 0, 0], "direction": [0, 0, 0], "force": 50})
        assert "zero" in receive_until(ws, "error").message

        ws.send_json({"type": "grab", "geom": torso, "point": [0, 0, 0], "target": [0, 0, 1.0]})
        deadline = time.time() + 5
        while sim._grab is None and time.time() < deadline:
            time.sleep(0.02)
        assert sim._grab is not None
    # The browser disconnected mid-drag: the server lets go.
    deadline = time.time() + 5
    while sim._grab is not None and time.time() < deadline:
        time.sleep(0.02)
    assert sim._grab is None


@pytest.fixture(scope="module")
def tiny_run12(tmp_path_factory):
    """A tiny real training run of the 12-motor robot."""
    from robot3d.training import PPOConfig, train

    return train(
        robot="quadruped12", total_steps=128, n_envs=2, name="tiny12", checkpoint_every=64,
        ppo=PPOConfig(n_steps=64, minibatches=2, n_epochs=1), runs_dir=tmp_path_factory.mktemp("runs12"), verbose=0,
    )


def test_watching_another_robots_policy_switches_the_robot(tiny_run12):
    with TestClient(create_app("quadruped", runs_dir=tiny_run12.parent)) as client:
        with client.websocket_connect("/ws") as ws:
            assert receive_until(ws, "scene").robot == "quadruped"
            checkpoint = client.get("/api/runs/tiny12").json()["checkpoints"][0]["name"]
            ws.send_json({"type": "load_policy", "run": "tiny12", "checkpoint": checkpoint})
            scene = receive_until(ws, "scene")
            assert scene.robot == "quadruped12" and len(scene.actuators) == 12
            status = receive_until(ws, "status", lambda s: s.walk_policy != "")
            assert status.walk_policy.startswith("tiny12 @ ") and status.policy_active
            frame = receive_until(ws, "frame")
            assert len(frame.ctrl) == 12
        # A browser connecting now gets the new robot straight away.
        with client.websocket_connect("/ws") as ws:
            assert receive_until(ws, "scene").robot == "quadruped12"


def test_walk_and_stand_modes(tiny_run, tiny_stand_run, tiny_getup_run, tmp_path):
    import shutil

    for run in (tiny_run, tiny_stand_run, tiny_getup_run):
        shutil.copytree(run, tmp_path / run.name)
    with TestClient(create_app("quadruped", runs_dir=tmp_path)) as client:
        with client.websocket_connect("/ws") as ws:
            status = receive_until(ws, "status")
            assert (status.walk_policy, status.stand_policy, status.recovering) == ("", "", False)
            ws.send_json({"type": "set_mode", "mode": "stand"})
            assert "no stand policy" in receive_until(ws, "error").message

            def load(run):
                checkpoint = client.get(f"/api/runs/{run}").json()["checkpoints"][0]["name"]
                ws.send_json({"type": "load_policy", "run": run, "checkpoint": checkpoint})

            load("tiny")
            status = receive_until(ws, "status", lambda s: s.walk_policy != "")
            assert status.mode == "walk" and status.stand_policy == "" and status.policy_active
            load("tiny_stand")
            status = receive_until(ws, "status", lambda s: s.stand_policy != "")
            assert status.mode == "stand" and status.walk_policy.startswith("tiny @ ")

            ws.send_json({"type": "use_policy", "active": False})  # manual control ...
            receive_until(ws, "status", lambda s: not s.policy_active)
            ws.send_json({"type": "set_mode", "mode": "walk"})  # ... picking a mode hands it back
            status = receive_until(ws, "status", lambda s: s.mode == "walk")
            assert status.policy_active and status.getup_policy == ""

            load("tiny_getup")
            status = receive_until(ws, "status", lambda s: s.getup_policy != "")
            assert status.mode == "walk"  # loading a get-up policy keeps the mode


def test_root_page_responds(client):
    assert client.get("/").status_code == 200
