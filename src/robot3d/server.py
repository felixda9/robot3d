"""FastAPI server: runs one shared simulation in real time and streams it to
browsers over a WebSocket, and serves training runs to the dashboard.

    WS   /ws        the protocol in web/src/protocol.ts (mirrored in protocol.py)
    GET  /api/...   training runs, curves, checkpoints (see protocol.ts, HTTP API)
    GET  /          the built frontend (web/dist), once `npm run build` has been run

Every connected browser sees the same simulation, like one real robot seen
from several screens. Any browser's commands (play/pause/reset, motor
targets) affect all of them. Loading a policy trained on another robot
switches the simulation to that robot, and every browser gets the new scene.

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
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from robot3d.protocol import (
    ClientMessage,
    ErrorMessage,
    EvaluateResponse,
    GrabCommand,
    LoadPolicyCommand,
    PauseCommand,
    PlayCommand,
    PushCommand,
    ReleaseCommand,
    ResetCommand,
    RunDetail,
    RunSummary,
    ScalarsResponse,
    SetCtrlCommand,
    StatusMessage,
    UsePolicyCommand,
    client_message_adapter,
)
from robot3d.runs import (
    RUNS_DIR,
    Checkpoint,
    ScalarReader,
    find_checkpoint,
    list_checkpoints,
    list_run_dirs,
    resolve_checkpoint,
    resolve_run,
    run_detail,
    run_summary,
    save_evaluation,
)
from robot3d.scene import SceneEncoder
from robot3d.simulation import Controller, Simulation

log = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
FPS = 60

# publish(json_text, is_frame): called from the sim thread.
PublishFn = Callable[[str, bool], None]


@dataclass(frozen=True)
class InstallPolicy:
    """Internal command (not from the protocol): swap in a loaded policy,
    and with `sim`, a new simulation first (the policy is for another robot)."""

    controller: Controller
    label: str
    sim: Simulation | None = None


def load_policy(checkpoint: Checkpoint, sim: Simulation) -> InstallPolicy:
    """Load a checkpoint to drive the simulation, or, if it was trained on
    another robot, a new simulation of that robot (slow: loads PyTorch and
    the network, so call it off the sim thread)."""
    from robot3d.policy import PolicyController  # PyTorch: import only when needed

    trained_on = checkpoint.run_info()["robot"]
    new_sim = None if trained_on == sim.robot else Simulation(trained_on)
    return InstallPolicy(PolicyController(checkpoint, (new_sim or sim).model), checkpoint.label, new_sim)


class EvaluationQueue:
    """Evaluates checkpoints one at a time on a background thread (a few
    seconds of CPU each) and caches results next to the checkpoint files."""

    def __init__(self, episodes: int = 5):
        self.episodes = episodes
        self._queue: queue.SimpleQueue[Checkpoint | None] = queue.SimpleQueue()
        self._pending: dict[str, set[str]] = defaultdict(set)  # run name -> checkpoint names
        self._errors: dict[str, str] = {}  # run name -> why its last failed evaluation failed
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def submit_missing(self, run_dir: Path) -> int:
        """Queue every checkpoint of the run that has no evaluation yet."""
        queued = 0
        with self._lock:
            self._errors.pop(run_dir.name, None)  # a new attempt
            pending = self._pending[run_dir.name]
            for checkpoint in list_checkpoints(run_dir):
                if checkpoint.name not in pending and checkpoint.evaluation() is None:
                    pending.add(checkpoint.name)
                    self._queue.put(checkpoint)
                    queued += 1
            if queued and self._thread is None:
                self._thread = threading.Thread(target=self._work, name="evaluate", daemon=True)
                self._thread.start()
        return queued

    def pending(self, run_name: str) -> int:
        with self._lock:
            return len(self._pending[run_name])

    def error(self, run_name: str) -> str:
        with self._lock:
            return self._errors.get(run_name, "")

    def stop(self) -> None:
        self._queue.put(None)

    def _work(self) -> None:
        from robot3d.policy import evaluate  # PyTorch: import only when needed

        while (checkpoint := self._queue.get()) is not None:
            try:
                save_evaluation(checkpoint, evaluate(checkpoint, episodes=self.episodes))
            except Exception as e:
                log.exception("evaluating %s failed", checkpoint.label)
                with self._lock:  # shown in the dashboard, not only in this log
                    self._errors[checkpoint.run_dir.name] = f"{checkpoint.name}: {e}"
            finally:
                with self._lock:
                    self._pending[checkpoint.run_dir.name].discard(checkpoint.name)


class SimRunner:
    """Owns the Simulation and advances it on a dedicated thread."""

    def __init__(self, sim: Simulation, fps: int = FPS, policy_label: str = ""):
        self.sim = sim
        self.policy_label = policy_label
        self.frame_dt = 1.0 / fps
        self._encoder = SceneEncoder(sim.model, sim.robot)
        # Plain str attributes: reading them from another thread is safe, since
        # Python swaps the reference in one step.
        self.scene_json = self._encoder.scene(sim.data).model_dump_json()
        self.status_json = self._status_json()
        self.latest_frame_json = self._frame_json()
        self._commands: queue.SimpleQueue[ClientMessage | InstallPolicy] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._publish: PublishFn | None = None

    def start(self, publish: PublishFn) -> None:
        self._publish = publish
        self._thread = threading.Thread(target=self._run, args=(publish,), name="sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def submit(self, command: ClientMessage | InstallPolicy) -> None:
        """Queue a command. Called from the event loop; applied on the sim thread."""
        self._commands.put(command)

    def _run(self, publish: PublishFn) -> None:
        next_frame = time.perf_counter()
        last_time = self.sim.data.time
        while not self._stop.is_set():
            try:
                status_before = self.status_json
                state_changed = self._apply_commands()
                self.status_json = self._status_json()
                if self.status_json != status_before:
                    publish(self.status_json, False)
                self.sim.advance()
                # Only send a frame if something changed: time moved on, or a
                # command changed the state (e.g. new motor targets while paused).
                if state_changed or self.sim.data.time != last_time:
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
        """Apply all queued commands (a slider drag can queue several per
        frame); return True if the robot's state changed."""
        state_changed = False
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            try:
                match command:
                    case PlayCommand():
                        self.sim.play()
                    case PauseCommand():
                        self.sim.pause()
                    case ResetCommand():
                        self.sim.reset()
                        state_changed = True
                    case SetCtrlCommand(ctrl=targets, duration=duration):
                        self.sim.set_ctrl(targets, duration)
                        state_changed = True
                    case UsePolicyCommand(active=active):
                        self.sim.use_controller(active)
                        state_changed = True
                    case GrabCommand(geom=geom, point=point, target=target):
                        self.sim.grab(geom, point, target)
                    case ReleaseCommand():
                        self.sim.release()
                    case PushCommand(geom=geom, point=point, direction=direction, force=force):
                        self.sim.push(geom, point, direction, force)
                    case InstallPolicy(controller=controller, label=label, sim=new_sim):
                        if new_sim is not None:
                            self._switch_robot(new_sim)
                        self.sim.set_controller(controller)  # drives now; robot restarts standing
                        self.policy_label = label
                        state_changed = True
            except (ValueError, RuntimeError) as e:
                # Checked on the event loop already; this only catches races,
                # e.g. a slider command queued just after "let the policy drive".
                log.warning("ignored %s: %s", type(command).__name__, e)
        return state_changed

    def _switch_robot(self, sim: Simulation) -> None:
        """Simulate another robot from now on; every browser gets its scene."""
        sim.paused = self.sim.paused
        self.sim = sim
        self._encoder = SceneEncoder(sim.model, sim.robot)
        self.scene_json = self._encoder.scene(sim.data).model_dump_json()
        if self._publish is not None:
            self._publish(self.scene_json, False)

    def _frame_json(self) -> str:
        return self._encoder.frame(self.sim.data).model_dump_json()

    def _status_json(self) -> str:
        return StatusMessage(
            paused=self.sim.paused,
            policy=self.policy_label,
            policy_active=self.sim.controller_active,
        ).model_dump_json()


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


def _refusal(command: ClientMessage, sim: Simulation) -> str | None:
    """Why a well-formed command can't be applied, or None. Checked on the
    event loop, where we can still answer the sender (the sim thread doesn't
    know who sent a command)."""
    match command:
        case SetCtrlCommand():
            if sim.controller_active:
                return "a policy is driving the motors; switch to manual control first"
            unknown = sorted(set(command.ctrl) - set(sim.actuator_names))
            if unknown:
                return f"unknown actuator(s): {', '.join(unknown)}"
        case UsePolicyCommand(active=True) if sim.controller is None:
            return "no policy loaded (start the server with --policy <run folder or checkpoint>)"
        case GrabCommand() | PushCommand():
            model = sim.model
            if command.geom >= model.ngeom:
                return f"no geom {command.geom}"
            if model.body_rootid[model.geom_bodyid[command.geom]] == 0:
                return "that's part of the static world; only the robot can be grabbed or pushed"
            if isinstance(command, PushCommand) and not any(command.direction):
                return "push direction is zero"
    return None


def create_app(
    robot: str = "quadruped",
    keyframe: str = "home",
    policy: str | Path | None = None,
    runs_dir: Path = RUNS_DIR,
) -> FastAPI:
    """`policy`: a run folder (-> newest checkpoint) or checkpoint .zip to drive
    the robot. `runs_dir`: where the dashboard finds training runs."""
    sim = Simulation(robot, keyframe)
    policy_label = ""
    if policy is not None:
        install = load_policy(find_checkpoint(policy), sim)
        sim = install.sim or sim
        sim.set_controller(install.controller)
        policy_label = install.label
    runner = SimRunner(sim, policy_label=policy_label)
    evaluations = EvaluationQueue()
    scalars = ScalarReader()
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
        evaluations.stop()

    app = FastAPI(title="robot3d", lifespan=lifespan)
    app.state.runner = runner

    # ------------------------------------------------ HTTP API: training runs
    # Plain `def` endpoints: FastAPI runs them on worker threads, so reading
    # event files never blocks the WebSocket traffic on the event loop.

    def _run_dir(run: str) -> Path:
        try:
            return resolve_run(run, runs_dir)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=404, detail=str(e)) from None

    @app.get("/api/runs")
    def api_runs() -> list[RunSummary]:
        return [run_summary(d) for d in list_run_dirs(runs_dir)]

    @app.get("/api/runs/{run}")
    def api_run(run: str) -> RunDetail:
        run_dir = _run_dir(run)
        return run_detail(
            run_dir,
            evaluating=evaluations.pending(run_dir.name),
            evaluation_error=evaluations.error(run_dir.name),
        )

    @app.get("/api/runs/{run}/scalars")
    def api_scalars(run: str, tags: str) -> ScalarsResponse:
        """tags: comma-separated TensorBoard tags, e.g. rollout/ep_rew_mean,train/std"""
        run_dir = _run_dir(run)
        wanted = [t for t in tags.split(",") if t]
        return ScalarsResponse(run=run_dir.name, series=scalars.read(run_dir, wanted))

    @app.post("/api/runs/{run}/evaluate")
    def api_evaluate(run: str) -> EvaluateResponse:
        return EvaluateResponse(queued=evaluations.submit_missing(_run_dir(run)))

    # ------------------------------------------------------------- WebSocket

    async def load_policy_command(command: LoadPolicyCommand) -> str | None:
        """Load a checkpoint off the event loop and hand it to the sim thread.
        Returns an error message, or None on success."""

        def load() -> InstallPolicy:
            checkpoint = resolve_checkpoint(resolve_run(command.run, runs_dir), command.checkpoint)
            return load_policy(checkpoint, runner.sim)

        try:
            install = await asyncio.to_thread(load)
        except (ValueError, FileNotFoundError) as e:
            return str(e)
        except Exception as e:
            log.exception("loading policy failed")
            return f"could not load the policy: {e}"
        runner.submit(install)
        return None

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        client = Client(ws)
        client.send_message(runner.scene_json)
        client.send_message(runner.status_json)
        client.send_frame(runner.latest_frame_json)
        clients.add(client)
        sender = asyncio.create_task(client.run_sender())
        grabbing = False  # did this browser grab a part without letting go yet?
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
                    continue
                problem = _refusal(command, runner.sim)
                if problem is None and isinstance(command, LoadPolicyCommand):
                    problem = await load_policy_command(command)
                elif problem is None:
                    runner.submit(command)
                    if isinstance(command, (GrabCommand, ReleaseCommand)):
                        grabbing = isinstance(command, GrabCommand)
                if problem is not None:
                    client.send_message(ErrorMessage(message=problem).model_dump_json())
        finally:
            clients.discard(client)
            sender.cancel()
            if grabbing:  # closed the tab mid-drag: don't keep pulling forever
                runner.submit(ReleaseCommand())

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
