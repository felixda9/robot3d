"""End-to-end tests of the WebSocket server: a real app with its simulation
thread, driven through FastAPI's TestClient (no browser needed)."""

import time

import pytest
from fastapi.testclient import TestClient

from robot3d.protocol import server_message_adapter
from robot3d.server import create_app

N_GEOMS = 15  # floor + torso + head + 4 legs x (thigh, shin, foot)
N_FRAME_GEOMS = 14  # everything except the static floor


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

        status = receive(ws)
        assert status.type == "status" and status.paused is False

        frame = receive(ws)
        assert frame.type == "frame"
        assert len(frame.xpos) == 3 * N_FRAME_GEOMS
        assert len(frame.xmat) == 9 * N_FRAME_GEOMS


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


def test_bad_messages_get_an_error_reply(client):
    with client.websocket_connect("/ws") as ws:
        for bad in ['not json', '{"type": "fly"}', '{"type": "play", "speed": 2}']:
            ws.send_text(bad)
            error = receive_until(ws, "error")
            assert error.message.startswith("invalid message")
        ws.send_bytes(b"\x00\x01")
        assert "JSON text" in receive_until(ws, "error").message


def test_root_page_responds(client):
    assert client.get("/").status_code == 200
