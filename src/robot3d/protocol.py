"""WebSocket message types: the Python mirror of web/src/protocol.ts.

web/src/protocol.ts is the single source of truth and documents every field.
These Pydantic models must describe exactly the same JSON; tests/test_protocol.py
generates a JSON Schema from the TypeScript file and fails if they drift apart.
Field names are snake_case on the wire, identical in both languages.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, TypeAdapter

Vec3 = tuple[float, float, float]
Rgba = tuple[float, float, float, float]
Range = tuple[float, float]  # [min, max]
Mat3 = tuple[float, float, float, float, float, float, float, float, float]  # row-major 3x3

GeomType = Literal["plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"]


class _Message(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # unknown fields are an error, not silently dropped
        # Fields with defaults (like `type`) are always present in the JSON we
        # send, so list them as required in the schema, as TypeScript does.
        json_schema_serialization_defaults_required=True,
    )


class GeomInfo(_Message):
    id: int
    name: str
    type: GeomType
    size: Vec3
    rgba: Rgba
    body: str
    group: int
    dynamic: bool
    pos: Vec3
    mat: Mat3


class CameraInfo(_Message):
    lookat: Vec3
    distance: float
    azimuth: float
    elevation: float
    fovy: float


class ActuatorInfo(_Message):
    name: str
    joint: str
    ctrl_range: Range
    force_range: Range


class KeyframeInfo(_Message):
    name: str
    ctrl: list[float]


# ---------------------------------------------------------------- server -> client


class SceneMessage(_Message):
    type: Literal["scene"] = "scene"
    robot: str
    timestep: float
    geoms: list[GeomInfo]
    frame_geoms: list[int]
    camera: CameraInfo
    actuators: list[ActuatorInfo]
    keyframes: list[KeyframeInfo]


class FrameMessage(_Message):
    type: Literal["frame"] = "frame"
    time: float
    xpos: list[float]
    xmat: list[float]
    ctrl: list[float]
    joint_pos: list[float]
    torque: list[float]


class StatusMessage(_Message):
    type: Literal["status"] = "status"
    paused: bool
    policy: str
    policy_active: bool


class ErrorMessage(_Message):
    type: Literal["error"] = "error"
    message: str


# `discriminator="type"` tells Pydantic to pick the model by the "type" field,
# the same way TypeScript narrows a union with `switch (msg.type)`.
ServerMessage = Annotated[
    SceneMessage | FrameMessage | StatusMessage | ErrorMessage,
    Field(discriminator="type"),
]

# ---------------------------------------------------------------- client -> server


class PlayCommand(_Message):
    type: Literal["play"] = "play"


class PauseCommand(_Message):
    type: Literal["pause"] = "pause"


class ResetCommand(_Message):
    type: Literal["reset"] = "reset"


class SetCtrlCommand(_Message):
    type: Literal["set_ctrl"] = "set_ctrl"
    # FiniteFloat: reject NaN/Infinity (Python's JSON parser accepts them, and
    # one NaN motor target would turn the whole simulation into NaNs).
    ctrl: dict[str, FiniteFloat]
    duration: Annotated[FiniteFloat, Field(ge=0.0, le=10.0)]


class UsePolicyCommand(_Message):
    type: Literal["use_policy"] = "use_policy"
    active: bool


# Run/checkpoint names are plain names, never paths (no "/", "\" or ".."),
# so a client can't make the server open files outside the runs folder.
SafeName = Annotated[str, Field(pattern=r"^[\w][\w.-]*$", max_length=200)]


class LoadPolicyCommand(_Message):
    type: Literal["load_policy"] = "load_policy"
    run: SafeName
    checkpoint: SafeName


FiniteVec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
GeomId = Annotated[int, Field(ge=0)]


class GrabCommand(_Message):
    type: Literal["grab"] = "grab"
    geom: GeomId
    point: FiniteVec3
    target: FiniteVec3


class ReleaseCommand(_Message):
    type: Literal["release"] = "release"


class PushCommand(_Message):
    type: Literal["push"] = "push"
    geom: GeomId
    point: FiniteVec3
    direction: FiniteVec3
    force: Annotated[FiniteFloat, Field(ge=0.0, le=1000.0)]


ClientMessage = Annotated[
    PlayCommand
    | PauseCommand
    | ResetCommand
    | SetCtrlCommand
    | UsePolicyCommand
    | LoadPolicyCommand
    | GrabCommand
    | ReleaseCommand
    | PushCommand,
    Field(discriminator="type"),
]

# TypeAdapters validate/serialize the union types (plain BaseModels have methods for this).
server_message_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)
client_message_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)

# ================================================================ HTTP API (training dashboard)

RunStatus = Literal["running", "finished", "stopped"]


class RunSummary(_Message):
    name: str
    robot: str
    backend: str
    status: RunStatus
    started: str
    finished: str
    steps_done: int
    total_steps: int
    n_envs: int
    checkpoints: int


class EvaluationInfo(_Message):
    episodes: int
    distance: float
    speed: float
    falls: int
    mean_return: float
    # Gait numbers (None in evaluations saved before they existed)
    duty_factor: float | None = None
    airborne: float | None = None
    diagonal_sync: float | None = None
    cadence: float | None = None


class CheckpointInfo(_Message):
    name: str
    steps: int
    evaluation: EvaluationInfo | None


class SettingInfo(_Message):
    group: str
    key: str
    value: str


class RunDetail(_Message):
    summary: RunSummary
    checkpoints: list[CheckpointInfo]
    settings: list[SettingInfo]
    evaluating: int
    evaluation_error: str


class ScalarSeries(_Message):
    tag: str
    steps: list[int]
    values: list[float]


class ScalarsResponse(_Message):
    run: str
    series: list[ScalarSeries]


class EvaluateResponse(_Message):
    queued: int
