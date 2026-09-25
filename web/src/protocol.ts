/*
 * protocol.ts: every message sent over the WebSocket between the Python
 * backend and the browser. THIS FILE IS THE SINGLE SOURCE OF TRUTH.
 *
 * Python mirror: src/robot3d/protocol.py (Pydantic models with the same names
 * and fields). tests/test_protocol.py generates a JSON Schema from this file
 * and fails if the Python models don't match it, so change both together.
 *
 * Conventions:
 *   - JSON text messages, snake_case field names (identical in both languages).
 *   - SI units (meters, radians, seconds) in MuJoCo's world frame:
 *     x forward, y left, z UP.
 *   - Rotation matrices are 3x3, row-major, flattened to 9 numbers
 *     (the layout of MuJoCo's data.geom_xmat).
 *
 * Flow:
 *   connect -> server sends SceneMessage, StatusMessage, latest FrameMessage
 *           -> then a FrameMessage ~60 times per second while the sim runs
 *   client  -> sends ClientMessage commands (play / pause / reset)
 */

export type Vec3 = [number, number, number];
export type Rgba = [number, number, number, number];
/** Row-major 3x3 rotation matrix: [r00, r01, r02, r10, r11, r12, r20, r21, r22]. */
export type Mat3 = [number, number, number, number, number, number, number, number, number];

/** Primitive shapes only; robots never use meshes. */
export type GeomType = "plane" | "sphere" | "capsule" | "ellipsoid" | "cylinder" | "box";

/**
 * One MuJoCo geom (a collision/visual shape attached to a body).
 *
 * `size` follows MuJoCo's geom_size convention (all HALF sizes):
 *   sphere:            [radius, 0, 0]
 *   capsule, cylinder: [radius, half_length, 0], along the geom's local z axis
 *   ellipsoid:         [semi_axis_x, semi_axis_y, semi_axis_z]
 *   box:               [half_x, half_y, half_z]
 *   plane:             [half_x, half_y, grid_spacing]; half sizes of 0 = infinite
 */
export interface GeomInfo {
  /** MuJoCo geom id (index into model.geom_*). */
  id: number;
  /** Geom name, "" if unnamed. */
  name: string;
  type: GeomType;
  size: Vec3;
  /** Color and opacity, each 0..1. */
  rgba: Rgba;
  /** Name of the body this geom is attached to ("world" for static scenery). */
  body: string;
  /** Visibility group 0..5. MuJoCo's viewer shows groups 0-2 by default. */
  group: number;
  /** True if the geom can move, i.e. it is included in every FrameMessage. */
  dynamic: boolean;
  /** Initial world position. For static geoms this never changes. */
  pos: Vec3;
  /** Initial world orientation. */
  mat: Mat3;
}

/** MuJoCo's default free camera, so the web view starts from the same viewpoint. */
export interface CameraInfo {
  /** Point the camera looks at. */
  lookat: Vec3;
  /** Distance from lookat to the camera. */
  distance: number;
  /** Degrees; rotation about world z (0 = camera looks along +x). */
  azimuth: number;
  /** Degrees; negative = camera above the lookat point, looking down. */
  elevation: number;
  /** Vertical field of view in degrees. */
  fovy: number;
}

// ---------------------------------------------------------------- server -> client

/** Sent once per connection: everything that does not change while the sim runs. */
export interface SceneMessage {
  type: "scene";
  /** Robot name (the MJCF file name in robots/). */
  robot: string;
  /** Physics step in seconds. */
  timestep: number;
  geoms: GeomInfo[];
  /** Ids of the dynamic geoms, in the order their poses appear in FrameMessage. */
  frame_geoms: number[];
  camera: CameraInfo;
}

/** The poses of all dynamic geoms at one instant (MuJoCo's geom_xpos / geom_xmat). */
export interface FrameMessage {
  type: "frame";
  /** Simulation time in seconds. */
  time: number;
  /** Flat [x, y, z] per geom in scene.frame_geoms order: length 3 * frame_geoms.length. */
  xpos: number[];
  /** Flat row-major 3x3 matrix per geom: length 9 * frame_geoms.length. */
  xmat: number[];
}

/** Simulation run state. Sent on connect and whenever it changes. */
export interface StatusMessage {
  type: "status";
  paused: boolean;
}

/** The server rejected a message from this client. */
export interface ErrorMessage {
  type: "error";
  message: string;
}

export type ServerMessage = SceneMessage | FrameMessage | StatusMessage | ErrorMessage;

// ---------------------------------------------------------------- client -> server

/** Resume the simulation. */
export interface PlayCommand {
  type: "play";
}

/** Freeze the simulation (physics stops stepping). */
export interface PauseCommand {
  type: "pause";
}

/** Reset the robot to its starting keyframe (keeps the paused/playing state). */
export interface ResetCommand {
  type: "reset";
}

export type ClientMessage = PlayCommand | PauseCommand | ResetCommand;
