"""Training runs on disk: listing, status, checkpoints, TensorBoard curves and
cached evaluations. Deliberately light on imports (no PyTorch), so the web
server can browse runs quickly.

A run folder (written by training.py):
    runs/<name>/run.json
    runs/<name>/tb/<sub>/events.out.tfevents.*
    runs/<name>/checkpoints/step_000500000.zip                (policy network)
    runs/<name>/checkpoints/step_000500000_vecnormalize.pkl   (its input scaling)
    runs/<name>/checkpoints/step_000500000_eval.json          (evaluation cache, optional)
"""

import json
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from robot3d.protocol import (
    CheckpointInfo,
    EvaluationInfo,
    RunDetail,
    RunStatus,
    RunSummary,
    ScalarSeries,
    SettingInfo,
)

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

# step_<N>.zip = Stable-Baselines3 (CPU) checkpoint, needs step_<N>_vecnormalize.pkl;
# step_<N>.pt = GPU-PPO checkpoint (network + normalizer in one file).
_STEP_FILE = re.compile(r"^step_(\d+)\.(zip|pt)$")
_SAFE_NAME = re.compile(r"^[\w][\w.-]*$")
# A run without a "finished" mark counts as running while its files keep
# changing (training rewrites run.json every ~30 s); otherwise it crashed or
# was killed.
_RUNNING_IF_ACTIVE_WITHIN = 120.0  # seconds


# ------------------------------------------------------------------ checkpoints


@dataclass(frozen=True)
class Checkpoint:
    model_path: Path  # checkpoints/step_XXXXXXXXX.zip (the neural network)
    run_dir: Path

    @property
    def name(self) -> str:
        return self.model_path.stem  # "step_009000012"

    @property
    def steps(self) -> int:
        return int(_STEP_FILE.match(self.model_path.name).group(1))

    @property
    def format(self) -> str:
        """"sb3" (CPU training) or "torch" (GPU training)."""
        return "torch" if self.model_path.suffix == ".pt" else "sb3"

    @property
    def normalizer_path(self) -> Path:
        return self.model_path.with_name(self.name + "_vecnormalize.pkl")

    @property
    def eval_path(self) -> Path:
        return self.model_path.with_name(self.name + "_eval.json")

    @property
    def label(self) -> str:
        return f"{self.run_dir.name} @ {self.steps:,} steps"

    def run_info(self) -> dict:
        return read_run_info(self.run_dir)

    def evaluation(self) -> EvaluationInfo | None:
        """Cached evaluation (see save_evaluation), or None."""
        try:
            return EvaluationInfo.model_validate_json(self.eval_path.read_text())
        except (OSError, ValueError):
            return None


def list_checkpoints(run_dir: Path) -> list[Checkpoint]:
    """A run's complete checkpoints, oldest first (an SB3 .zip counts once its
    normalizer file is saved too)."""
    paths = sorted((run_dir / "checkpoints").glob("step_*"))
    checkpoints = [Checkpoint(p, run_dir) for p in paths if _STEP_FILE.match(p.name)]
    return [c for c in checkpoints if c.format == "torch" or c.normalizer_path.is_file()]


def find_checkpoint(path: str | Path) -> Checkpoint:
    """A checkpoint file (.zip or .pt), or a run folder (-> its newest checkpoint)."""
    path = Path(path)
    if path.is_file():
        if not _STEP_FILE.match(path.name):
            raise ValueError(f"Not a checkpoint file (expected step_<N>.zip or .pt): {path}")
        return Checkpoint(path, path.parent.parent)
    if path.name == "checkpoints":
        path = path.parent
    checkpoints = list_checkpoints(path)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {path} (expected {path / 'checkpoints' / 'step_<N>.zip or .pt'})")
    return checkpoints[-1]


def save_evaluation(checkpoint: Checkpoint, results: list[dict], skill: dict | None = None) -> EvaluationInfo:
    """Store evaluate() results (and a skill_test() result) next to the
    checkpoint (the dashboard shows them)."""
    n = len(results)

    def mean_of(key: str) -> float | None:
        values = [r[key] for r in results if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    info = EvaluationInfo(
        episodes=n,
        distance=mean_of("distance"),
        speed=mean_of("speed"),
        falls=sum(bool(r["fell"]) for r in results),
        mean_return=mean_of("return"),
        duty_factor=mean_of("duty_factor"),
        airborne=mean_of("airborne"),
        diagonal_sync=mean_of("diagonal_sync"),
        cadence=mean_of("cadence"),
        upright=mean_of("upright"),
        **(skill or {}),
    )
    checkpoint.eval_path.write_text(info.model_dump_json(indent=2) + "\n")
    return info


# ------------------------------------------------------------------------- runs


def read_run_info(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text())


def write_run_info(run_dir: Path, info: dict) -> None:
    """Write run.json atomically: a temp file, then swapped in, so a reader
    (the dashboard) never sees a half-written file. On Windows the swap fails
    while another process has the file open for that instant, so retry."""
    path = run_dir / "run.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=2) + "\n")
    for _ in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.05)
    tmp.replace(path)  # last try: let the error surface


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def default_run_name(robot: str, suffix: str = "walk") -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}_{robot}_{suffix}"


def list_run_dirs(runs_dir: Path = RUNS_DIR) -> list[Path]:
    """Run folders, newest first."""
    if not runs_dir.is_dir():
        return []
    dirs = [d for d in runs_dir.iterdir() if (d / "run.json").is_file()]
    return sorted(dirs, key=lambda d: read_run_info(d).get("started", ""), reverse=True)


def resolve_run(name: str, runs_dir: Path = RUNS_DIR) -> Path:
    """A run folder by name. Names only, never paths (no "..", "/", "\\")."""
    if not _SAFE_NAME.match(name):
        raise ValueError(f"invalid run name: {name!r}")
    run_dir = runs_dir / name
    if not (run_dir / "run.json").is_file():
        raise FileNotFoundError(f"no run named {name!r}")
    return run_dir


def resolve_checkpoint(run_dir: Path, name: str) -> Checkpoint:
    if not _SAFE_NAME.match(name):
        raise ValueError(f"invalid checkpoint name: {name!r}")
    for checkpoint in list_checkpoints(run_dir):
        if checkpoint.name == name:
            return checkpoint
    raise FileNotFoundError(f"run {run_dir.name!r} has no checkpoint {name!r}")


def run_status(run_dir: Path, info: dict) -> RunStatus:
    if info.get("finished"):
        return "stopped" if info.get("interrupted") else "finished"
    files = [run_dir / "run.json", *(run_dir / "tb").rglob("events.out.tfevents.*")]
    last_activity = max(f.stat().st_mtime for f in files if f.exists())
    return "running" if time.time() - last_activity < _RUNNING_IF_ACTIVE_WITHIN else "stopped"


def run_task(info: dict) -> str:
    """"walk", "stand" or "getup" (runs before the task setting: getup if
    falls didn't end its episodes, as WalkConfig.from_run)."""
    walk = info.get("walk_config", {})
    return walk.get("task") or ("getup" if walk.get("terminate_on_fall") is False else "walk")


def needs_evaluation(checkpoint: Checkpoint) -> bool:
    """Not evaluated yet, or evaluated before the skill test existed."""
    evaluation = checkpoint.evaluation()
    return evaluation is None or evaluation.skill is None


def run_summary(run_dir: Path) -> RunSummary:
    info = read_run_info(run_dir)
    return RunSummary(
        name=run_dir.name,
        robot=info.get("robot", "?"),
        task=run_task(info),
        backend=info.get("backend", "cpu"),  # runs from before GPU training existed were CPU
        status=run_status(run_dir, info),
        started=info.get("started", ""),
        finished=info.get("finished", ""),
        steps_done=int(info.get("steps_done", 0)),
        total_steps=int(info.get("total_steps", 0)),
        n_envs=int(info.get("n_envs", 0)),
        checkpoints=len(list_checkpoints(run_dir)),
    )


def run_detail(run_dir: Path, evaluating: int = 0, evaluation_error: str = "") -> RunDetail:
    info = read_run_info(run_dir)
    checkpoints = [
        CheckpointInfo(name=c.name, steps=c.steps, evaluation=c.evaluation()) for c in list_checkpoints(run_dir)
    ]
    settings = [SettingInfo(group="run", key=k, value=str(info[k])) for k in ("seed", "n_envs", "command") if k in info]
    for group in ("walk_config", "ppo_config"):
        for key, value in info.get(group, {}).items():
            settings.append(SettingInfo(group=group.removesuffix("_config"), key=key, value=str(value)))
    return RunDetail(
        summary=run_summary(run_dir),
        checkpoints=checkpoints,
        settings=settings,
        evaluating=evaluating,
        evaluation_error=evaluation_error,
    )


# ---------------------------------------------------------- TensorBoard curves


class ScalarReader:
    """Reads a run's TensorBoard curves. Keeps one EventAccumulator per event
    folder and reloads incrementally, so polling a live run only reads the
    new events. Thread-safe (the web server calls it from worker threads)."""

    def __init__(self, max_points: int = 1500):
        self.max_points = max_points
        self._accumulators: dict[Path, object] = {}
        self._lock = threading.Lock()

    def read(self, run_dir: Path, tags: list[str]) -> list[ScalarSeries]:
        # Imported here: TensorBoard's reader is slowish to import and only
        # needed by the dashboard.
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        points: dict[str, list[tuple[int, float]]] = {tag: [] for tag in tags}
        with self._lock:
            for folder in sorted({p.parent for p in (run_dir / "tb").rglob("events.out.tfevents.*")}):
                accumulator = self._accumulators.get(folder)
                if accumulator is None:
                    accumulator = EventAccumulator(str(folder), size_guidance={"scalars": 0})  # 0 = keep all
                    self._accumulators[folder] = accumulator
                accumulator.Reload()
                available = set(accumulator.Tags()["scalars"])
                for tag in tags:
                    if tag in available:
                        points[tag].extend((e.step, e.value) for e in accumulator.Scalars(tag))

        series = []
        for tag, pts in points.items():
            pts = sorted((s, v) for s, v in pts if math.isfinite(v))  # JSON can't carry NaN/inf
            stride = max(1, math.ceil(len(pts) / self.max_points))
            pts = pts[::stride]
            series.append(
                ScalarSeries(tag=tag, steps=[s for s, _ in pts], values=[float(f"{v:.6g}") for _, v in pts])
            )
        return series
