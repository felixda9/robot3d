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
 *              (and whenever motor targets change)
 *   client  -> sends ClientMessage commands (play / pause / reset / set_ctrl /
 *              use_policy / load_policy / grab / release / push)
 *
 * The HTTP API for the training dashboard is at the end of this file.
 *
 * Per-motor arrays (FrameMessage.ctrl etc., KeyframeInfo.ctrl) are in
 * actuator order, the order of SceneMessage.actuators.
 */

export type Vec3 = [number, number, number];
export type Rgba = [number, number, number, number];
/** [min, max] */
export type Range = [number, number];
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

/**
 * One motor. All our motors are MuJoCo position actuators: ctrl is the TARGET
 * joint angle (radians), and a built-in PD controller produces the torque.
 */
export interface ActuatorInfo {
  /** Actuator name (for our robots, the same as its joint's name). */
  name: string;
  /** Name of the joint it drives. */
  joint: string;
  /** Allowed targets in radians; the server clamps set_ctrl values to this. */
  ctrl_range: Range;
  /** Torque limits in N·m (the motor's strength); [0, 0] = unlimited. */
  force_range: Range;
}

/** A named pose from the robot's MJCF <keyframe> list, used as a pose preset. */
export interface KeyframeInfo {
  name: string;
  /** Motor targets in actuator order (same order as SceneMessage.actuators). */
  ctrl: number[];
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
  /** The motors, in actuator order (the order of every per-motor array). */
  actuators: ActuatorInfo[];
  /** Pose presets; the first is the pose the robot starts from and resets to. */
  keyframes: KeyframeInfo[];
}

/**
 * Everything that changes, at one instant. Sent ~60 times per second while
 * the sim runs, and also when the motor targets change (even while paused).
 */
export interface FrameMessage {
  type: "frame";
  /** Simulation time in seconds. */
  time: number;
  /** Flat [x, y, z] per geom in scene.frame_geoms order: length 3 * frame_geoms.length. */
  xpos: number[];
  /** Flat row-major 3x3 matrix per geom: length 9 * frame_geoms.length. */
  xmat: number[];
  /** Per motor: target angle in radians (MuJoCo's data.ctrl). */
  ctrl: number[];
  /** Per motor: actual angle of its joint in radians. Differs from ctrl while moving or under load. */
  joint_pos: number[];
  /** Per motor: torque it is applying right now, N·m (MuJoCo's data.actuator_force). */
  torque: number[];
}

/**
 * What the robot is doing while a policy drives. walk: the walk policy.
 * stand: the stand policy (stands still, catches shoves), or the get-up
 * policy if that's all there is. In both, a loaded get-up policy takes over
 * after a fall until the robot stands steady again.
 */
export type Mode = "walk" | "stand";

/** Simulation run state. Sent on connect and whenever it changes. */
export interface StatusMessage {
  type: "status";
  paused: boolean;
  /** The loaded walking policy ("<run> @ <steps> steps"), or "" if none. */
  walk_policy: string;
  /** The loaded standing policy, or "" if none. */
  stand_policy: string;
  /** The loaded get-up policy (drives after falls), or "" if none. */
  getup_policy: string;
  /** Which policy drives while a policy drives. */
  mode: Mode;
  /** True while a policy drives the motors; set_ctrl is refused then. */
  policy_active: boolean;
  /** The robot fell and the get-up policy is getting it back up. */
  recovering: boolean;
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

/**
 * Set motor targets. Only the named motors change, so two browsers can move
 * different sliders at once. Values in radians; unknown names are an error;
 * out-of-range values are clamped to ctrl_range.
 */
export interface SetCtrlCommand {
  type: "set_ctrl";
  /** Actuator name -> target angle in radians, e.g. { "FL_knee": -1.2 }. */
  ctrl: Record<string, number>;
  /**
   * Seconds (sim time) to get there, 0..10. 0 = set the targets at once
   * (sliders). > 0 = the named motors' targets glide from where they are to
   * the new values together, all arriving at the same moment (pose presets).
   * Jumping at once between very different poses can throw the robot over.
   */
  duration: number;
}

/**
 * Let the loaded policy drive (true) or take over manually (false). Manual
 * control starts from the policy's last motor targets. Error if no policy.
 */
export interface UsePolicyCommand {
  type: "use_policy";
  active: boolean;
}

/**
 * Load a checkpoint from the runs folder and let it drive; the robot
 * restarts standing. Names, not paths, e.g.
 * { run: "walk_10m", checkpoint: "step_009000012" }. It goes in the slot of
 * its task (walk, stand or getup; replacing what was there); walk and stand
 * policies switch the mode to theirs. If the run was trained on another
 * robot, the server switches to that robot (forgetting the other slots) and
 * sends every browser a new SceneMessage.
 */
export interface LoadPolicyCommand {
  type: "load_policy";
  run: string;
  checkpoint: string;
}

/**
 * Grab a robot part with the mouse and pull it; sent again whenever the
 * mouse moves, until ReleaseCommand. A spring pulls the grabbed point toward
 * `target`, sized to the robot's mass (strong enough to lift it by the
 * torso, capped at 3 g). It acts in the physics, so it works while a policy
 * drives too. One grab at a time per server; a new one replaces the old.
 */
export interface GrabCommand {
  type: "grab";
  /** The clicked geom (id as in SceneMessage.geoms); must be part of the robot. */
  geom: number;
  /** The grabbed spot in the geom's own frame, so it stays on the part as it moves. */
  point: Vec3;
  /** Where to pull it: the mouse position in the world, m. */
  target: Vec3;
}

/** Let go of the grabbed part (also happens when the grabbing browser disconnects). */
export interface ReleaseCommand {
  type: "release";
}

/**
 * Shove a robot part: `force` newtons at `point`, along `direction`, for
 * 0.1 s of sim time. The velocity change is force x 0.1 s / mass: 60 N on
 * the 6.6 kg quadruped is a ~0.9 m/s shove. Waits while paused.
 */
export interface PushCommand {
  type: "push";
  /** The clicked geom; must be part of the robot. */
  geom: number;
  /** Where it's hit, in the geom's own frame. */
  point: Vec3;
  /** World direction (any length; the server normalizes it; not zero). */
  direction: Vec3;
  /** Newtons, 0..1000. */
  force: number;
}

/** Switch between walking and standing (and let the policies drive). */
export interface SetModeCommand {
  type: "set_mode";
  mode: Mode;
}

export type ClientMessage =
  | PlayCommand
  | PauseCommand
  | ResetCommand
  | SetCtrlCommand
  | UsePolicyCommand
  | LoadPolicyCommand
  | GrabCommand
  | ReleaseCommand
  | PushCommand
  | SetModeCommand;

// ================================================================ HTTP API
// The training dashboard reads runs over plain HTTP (JSON), not the
// WebSocket: it's request/response data, not a live stream.
//   GET  /api/runs                      -> RunSummary[]
//   GET  /api/runs/{run}                -> RunDetail
//   GET  /api/runs/{run}/scalars?tags=a,b -> ScalarsResponse
//   POST /api/runs/{run}/evaluate       -> EvaluateResponse

/** running: still writing; finished: completed; stopped: interrupted or crashed. */
export type RunStatus = "running" | "finished" | "stopped";

export interface RunSummary {
  /** Folder name under runs/. */
  name: string;
  robot: string;
  /** What it was trained for: "walk", "stand" or "getup". */
  task: string;
  /** Where it trained: "cpu" (Stable-Baselines3) or "gpu" (MuJoCo Warp + GPU PPO). */
  backend: string;
  status: RunStatus;
  /** ISO timestamps; finished is "" while running or if it crashed. */
  started: string;
  finished: string;
  steps_done: number;
  total_steps: number;
  n_envs: number;
  checkpoints: number;
}

/** A checkpoint measured headless without exploration noise (mean over episodes). */
export interface EvaluationInfo {
  episodes: number;
  /** Meters walked forward per episode. */
  distance: number;
  /** m/s. */
  speed: number;
  /** Episodes that ended by falling. */
  falls: number;
  /** Mean total reward per episode. */
  mean_return: number;
  /** Gait (null in evaluations saved before these existed): share of time each foot is on the ground (walk > 0.5, run < 0.5). */
  duty_factor: number | null;
  /** Share of time all feet are in the air (a walk: 0). */
  airborne: number | null;
  /** Share of time diagonal feet are both down or both up (trot: ~1). */
  diagonal_sync: number | null;
  /** Touchdowns per foot per second. */
  cadence: number | null;
  /** Share of the time not fallen (null in older evaluations). For standing policies: includes getting up from fallen starts. */
  upright: number | null;
  /**
   * The task's skill test, share passed 0..1 (null until tested): walk and
   * stand survive sudden shoves; getup gets up from hard fallen starts. The
   * best checkpoint is the one with the highest skill.
   */
  skill: number | null;
  /** What the skill test tested, e.g. "shoves of 1 and 2 m/s from 16 directions". */
  skill_test: string;
}

export interface CheckpointInfo {
  /** e.g. "step_009000012" (load it with LoadPolicyCommand). */
  name: string;
  steps: number;
  /** null until evaluated (POST /api/runs/{run}/evaluate). */
  evaluation: EvaluationInfo | null;
}

/** One training setting, preformatted for display. */
export interface SettingInfo {
  group: string;
  key: string;
  value: string;
}

export interface RunDetail {
  summary: RunSummary;
  checkpoints: CheckpointInfo[];
  settings: SettingInfo[];
  /** Checkpoints of this run waiting for or in evaluation. */
  evaluating: number;
  /** Why the last evaluation of this run failed, "" if none did (cleared when evaluation is requested again). */
  evaluation_error: string;
}

/** One TensorBoard curve: value at each logged training step. */
export interface ScalarSeries {
  tag: string;
  steps: number[];
  values: number[];
}

export interface ScalarsResponse {
  run: string;
  series: ScalarSeries[];
}

export interface EvaluateResponse {
  /** Checkpoints newly queued for evaluation. */
  queued: number;
}
