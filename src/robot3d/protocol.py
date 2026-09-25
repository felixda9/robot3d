"""WebSocket message types: the Python mirror of web/src/protocol.ts.

web/src/protocol.ts is the single source of truth and documents every field.
These Pydantic models must describe exactly the same JSON; tests/test_protocol.py
generates a JSON Schema from the TypeScript file and fails if they drift apart.
Field names are snake_case on the wire, identical in both languages.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

Vec3 = tuple[float, float, float]
Rgba = tuple[float, float, float, float]
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


# ---------------------------------------------------------------- server -> client


class SceneMessage(_Message):
    type: Literal["scene"] = "scene"
    robot: str
    timestep: float
    geoms: list[GeomInfo]
    frame_geoms: list[int]
    camera: CameraInfo


class FrameMessage(_Message):
    type: Literal["frame"] = "frame"
    time: float
    xpos: list[float]
    xmat: list[float]


class StatusMessage(_Message):
    type: Literal["status"] = "status"
    paused: bool


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


ClientMessage = Annotated[
    PlayCommand | PauseCommand | ResetCommand,
    Field(discriminator="type"),
]

# TypeAdapters validate/serialize the union types (plain BaseModels have methods for this).
server_message_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)
client_message_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)
