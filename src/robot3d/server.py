"""FastAPI server: runs one shared simulation in real time and streams it to
browsers over a WebSocket.

    WS   /ws   the protocol in web/src/protocol.ts (mirrored in protocol.py)
    GET  /     the built frontend (web/dist), once `npm run build` has been run

Every connected browser sees the same simulation, like one real robot seen
from several screens. Any browser's play/pause/reset affects all of them.

Threads:
  * The simulation runs on its own thread (SimRunner) at a steady 60 fps,
    independent of the network.
  * Networking runs on asyncio's event loop. The sim thread hands each frame,
    already serialized to JSON once, to the loop with call_soon_threadsafe.
    Commands go the other way through a thread-safe queue.
"""

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from robot3d.protocol import ClientMessage, ErrorMessage, StatusMessage, client_message_adapter
from robot3d.scene import build_frame, build_scene
from robot3d.simulation import Simulation

log = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
FPS = 60

# publish(json_text, is_frame): called from the sim thread.
PublishFn = Callable[[str, bool], None]


class SimRunner:
    """Owns the Simulation and advances it on a dedicated thread."""

    def __init__(self, sim: Simulation, fps: int = FPS):
        self.sim = sim
        self.frame_dt = 1.0 / fps
        scene = build_scene(sim.model, sim.data, sim.robot)
        self._frame_geoms = np.array(scene.frame_geoms, dtype=int)
        # Plain str attributes: reading them from another thread is safe, since
        # Python swaps the reference in one step.
        self.scene_json = scene.model_dump_json()
        self.status_json = self._status_json()
        self.latest_frame_json = self._frame_json()
        self._commands: queue.SimpleQueue[ClientMessage] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, publish: PublishFn) -> None:
        self._thread = threading.Thread(target=self._run, args=(publish,), name="sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def submit(self, command: ClientMessage) -> None:
        """Queue a command. Called from the event loop; applied on the sim thread."""
        self._commands.put(command)

    def _run(self, publish: PublishFn) -> None:
        next_frame = time.perf_counter()
        last_time = self.sim.data.time
        while not self._stop.is_set():
            try:
                if self._apply_commands():
                    self.status_json = self._status_json()
                    publish(self.status_json, False)
                self.sim.advance()
                # Only send a frame if something changed (not while paused).
                if self.sim.data.time != last_time:
                    last_time = self.sim.data.time
                    self.latest_frame_json = self._frame_json()
                    publish(self.latest_frame_json, True)
            except Exception:
                log.exception("simulation thread error")
                self.sim.pause()

            # Sleep until the next frame. time.sleep is precise to ~1 ms on
            # Python 3.11+ (also on Windows).
            next_frame += self.frame_dt
            delay = next_frame - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame = time.perf_counter()  # fell behind; don't burst to catch up

    def _apply_commands(self) -> bool:
        """Apply queued commands; return True if the paused state changed."""
        was_paused = self.sim.paused
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            match command.type:
                case "play":
                    self.sim.play()
                case "pause":
                    self.sim.pause()
                case "reset":
                    self.sim.reset()
        return self.sim.paused != was_paused

    def _frame_json(self) -> str:
        return build_frame(self.sim.data, self._frame_geoms).model_dump_json()

    def _status_json(self) -> str:
        return StatusMessage(paused=self.sim.paused).model_dump_json()


class Client:
    """One browser connection, with its own sender task.

    Frames: only the newest is kept, so a slow client skips frames instead of
    falling further and further behind. Other messages are queued, never dropped.
    """

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self._messages: deque[str] = deque()
        self._frame: str | None = None
        self._wake = asyncio.Event()

    def send_message(self, text: str) -> None:
        self._messages.append(text)
        self._wake.set()

    def send_frame(self, text: str) -> None:
        self._frame = text
        self._wake.set()

    async def run_sender(self) -> None:
        try:
            while True:
                await self._wake.wait()
                self._wake.clear()
                while self._messages:
                    await self.ws.send_text(self._messages.popleft())
                if self._frame is not None:
                    frame, self._frame = self._frame, None
                    await self.ws.send_text(frame)
        except Exception:
            pass  # connection closed; the receive loop cleans up


def create_app(robot: str = "quadruped", keyframe: str = "home") -> FastAPI:
    runner = SimRunner(Simulation(robot, keyframe))
    clients: set[Client] = set()  # only touched on the event loop thread

    def broadcast(text: str, is_frame: bool) -> None:
        for client in clients:
            if is_frame:
                client.send_frame(text)
            else:
                client.send_message(text)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()

        def publish(text: str, is_frame: bool) -> None:  # runs on the sim thread
            try:
                loop.call_soon_threadsafe(broadcast, text, is_frame)
            except RuntimeError:
                pass  # event loop already closed (shutting down)

        runner.start(publish)
        yield
        runner.stop()

    app = FastAPI(title="robot3d", lifespan=lifespan)
    app.state.runner = runner

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        client = Client(ws)
        client.send_message(runner.scene_json)
        client.send_message(runner.status_json)
        client.send_frame(runner.latest_frame_json)
        clients.add(client)
        sender = asyncio.create_task(client.run_sender())
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is None:
                    client.send_message(ErrorMessage(message="expected a JSON text message").model_dump_json())
                    continue
                try:
                    command = client_message_adapter.validate_json(text)
                except ValidationError as e:
                    error = e.errors(include_url=False)[0]
                    detail = f"{error['msg']} at {'.'.join(map(str, error['loc'])) or 'top level'}"
                    client.send_message(ErrorMessage(message=f"invalid message: {detail}").model_dump_json())
                else:
                    runner.submit(command)
        finally:
            clients.discard(client)
            sender.cancel()

    if WEB_DIST.is_dir():
        # Production: serve the built frontend. Mounted last so /ws wins.
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    else:

        @app.get("/", response_class=HTMLResponse)
        def frontend_not_built() -> str:
            return (
                "<h3>robot3d backend is running.</h3>"
                "<p>Development: run <code>npm run dev</code> in <code>web/</code> and open "
                '<a href="http://localhost:5173">http://localhost:5173</a>.</p>'
                "<p>Or build the frontend once with <code>npm run build</code> in <code>web/</code> "
                "and reload this page.</p>"
            )

    return app
