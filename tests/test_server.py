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


def test_root_page_responds(client):
    assert client.get("/").status_code == 200
