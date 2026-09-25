# robot3d — Robot simulation & training platform

## Vision

A system to **design legged robots** (limbs, joints, motors), **watch them in 3D in
the browser**, **control them manually**, and **train them with reinforcement
learning** to walk and do other tasks. Later: a robot designer, a training
dashboard, custom environments (obstacle courses, terrain, stairs), and gamepad
control of trained robots.

MuJoCo is the physics engine; everything around it is built here.

## Architecture

```
┌──────────────────────────── Python backend ────────────────────────────┐
│  robots/*.xml (MJCF)  ──►  MuJoCo sim  ◄──  Gymnasium envs  ◄──  SB3 PPO │
│                               │                                        │
│                     FastAPI WebSocket server                           │
└───────────────────────────────┼────────────────────────────────────────┘
                                │  1× static scene on connect
                                │  ~60 fps: geom_xpos + geom_xmat per frame
                                │  ◄── commands (play/pause/reset, motor targets)
┌───────────────────────────────┴──── Browser (three.js) ───────────────────┐
│  Viewer + controller only. NEVER simulates physics.                       │
└───────────────────────────────────────────────────────────────────────────┘
```

- **The Python backend owns everything physical:** MuJoCo simulation, Gymnasium
  environments, and training (Stable-Baselines3, starting with PPO).
- **The browser frontend (three.js) is only a viewer and controller.** It never
  simulates physics.
- **Protocol (WebSocket at `/ws`, FastAPI):**
  - On connect, the backend sends the static scene once: for each geom, its type,
    size, color (rgba), and parent body.
  - Then it streams each frame's geom positions and rotation matrices
    (MuJoCo's `data.geom_xpos` and `data.geom_xmat`) at ~60 fps.
  - The frontend sends commands back (play/pause/reset, later motor targets).
  - **`web/src/protocol.ts` is the single source of truth for every message.**
    `src/robot3d/protocol.py` mirrors it with Pydantic models (same names and
    fields, snake_case on the wire). `tests/test_protocol.py` generates a JSON
    Schema from the TS file and fails if the Python models differ. **Adding or
    changing a message: edit both files, then run `uv run pytest`.**
- **Dev setup:** the Vite dev server (port 5173) serves the page and proxies
  `/ws` to the backend (port 8000), so the page always connects to
  `ws://<its own host>/ws`. In production, `npm run build` writes `web/dist/`,
  which the backend serves itself at port 8000.
- **Robots are data, not code:** each robot is an MJCF file in `robots/` (later
  generated from a simpler YAML/JSON spec). **Only primitive shapes** (capsule,
  box, sphere; plane for the ground). No meshes, so rendering stays simple.
- **Environments stay Gymnasium-compatible** so standard RL libraries work.

## Conventions

- SI units everywhere: meters, kilograms, seconds, **radians** (MJCF files set
  `<compiler angle="radian"/>`, because MJCF's default unit is degrees).
- World frame: **+x forward, +y left, +z up**.
- Joint angle 0 = the "drawn" pose in the MJCF (for the quadruped: legs straight
  down). Useful poses live in named `<keyframe>`s; `home` = standing pose.
- Robot parts collide with the world but not with each other (see Decisions).
- Leg naming: `FL`, `FR`, `RL`, `RR` (front/rear, left/right). Joints are
  `<leg>_hip`, `<leg>_knee`; actuators use the same names as their joints.
- Explain physics/RL concepts briefly in code comments where they matter (the
  user is learning as we go).

## Environment

- Windows 10, NVIDIA RTX 3090 (24 GB), Python 3.12 managed with **uv**;
  Node 24 + npm for the frontend (Vite 8, TypeScript 7, three.js 0.186).
- Commands (from the repo root unless noted):
  - `uv sync` and `cd web; npm install`: install everything
  - `uv run pytest`: all tests (physics, real-time loop, server over a real
    WebSocket, TS↔Python protocol contract; the contract test needs `web/node_modules`)
  - `uv run scripts/view_mujoco.py`: MuJoCo's built-in viewer
  - `uv run scripts/serve.py`: simulation server on http://localhost:8000
  - `cd web; npm run dev`: frontend dev server on http://localhost:5173
    (start the backend too)
  - `cd web; npm run build`: type-check (`tsc`) + production build to `web/dist/`
  - `cd web; npm run typecheck`: type-check only

## Layout

```
robots/                 MJCF robot files (data, not code)
src/robot3d/
  robots.py             load_model(), reset_to_keyframe()
  simulation.py         Simulation: model + state + real-time pacing (shared loop)
  scene.py              MuJoCo model/state -> SceneMessage / FrameMessage
  protocol.py           Pydantic mirror of web/src/protocol.ts
  server.py             FastAPI app: sim thread, WebSocket, static files
scripts/                view_mujoco.py, serve.py (later: training/eval)
tests/                  pytest
web/                    Vite + TypeScript + three.js frontend
  src/protocol.ts       ALL WebSocket message types (single source of truth)
  src/connection.ts     WebSocket with auto-reconnect
  src/viewer.ts         three.js scene: camera, lights, ground, sky, geoms
  src/geoms.ts          MuJoCo geom -> three.js mesh, pose helpers
  src/main.ts           wiring + UI (buttons, stats)
  vite.config.ts        dev server + /ws proxy
```

## Milestones

- [x] **1. Foundation:** Python project (uv), MuJoCo installed, a simple
  quadruped in MJCF (torso, 4 legs, 2 hinge joints per leg, position-controlled
  motors) on a ground plane. A script that shows it in MuJoCo's built-in viewer.
  *Success = the robot drops onto the floor and settles without jittering or
  exploding.*
- [x] **2. Web viewer:** FastAPI WebSocket server runs the simulation in real
  time; a three.js page renders it with an orbit camera, shadows, a ground grid,
  and play/pause/reset buttons. *Success = the web view matches MuJoCo's viewer.*
- [ ] **3. Manual control:** one slider per motor in the web UI plus a few
  keyboard shortcuts. Commands go to the backend and move the robot.
- [ ] **4. First training:** Gymnasium env for the quadruped (reward forward
  velocity and staying upright; penalize energy use and falling). PPO training
  script with parallel envs, checkpoints, TensorBoard logs. Load any checkpoint
  and watch it in the web viewer.
- [ ] **5. Training dashboard:** training curves, checkpoint list, and replay of
  any checkpoint in the web UI.
- [ ] **6. Robot designer:** simple YAML/JSON robot spec (body parts, joints,
  motors) → generated MJCF; then a visual editor in the browser.
- [ ] **7. Environments & commands:** terrain, stairs, obstacles. Train a policy
  that follows direction + speed commands, controllable by keyboard or gamepad.

## How we work

- **One milestone at a time.** At the end of each: explain exactly how to run and
  test it, then **wait for the user**.
- Keep this file updated with decisions and current status.
- **Commit after each working milestone.**
- If a design choice is significant or something is ambiguous, **ask** instead of
  guessing.

## Decisions log

- **2026-09-24: Tooling:** uv for env and dependency management, Python 3.12,
  src layout (`src/robot3d`), pytest for headless checks.
- **2026-09-24: Quadruped leg layout: dog-style** (user's choice). Each leg has a
  hip-pitch and a knee-pitch hinge, both rotating about the sideways (y) axis, so
  legs swing in vertical planes like Petoi Bittle or Ghost Minitaur (both
  8-joint robots). It can walk forward and back and turn with uneven strides,
  but it can't step sideways, and the policy has to learn to balance.
- **2026-09-24: Quadruped dimensions:** 6.6 kg total. Torso box 0.40 × 0.16 ×
  0.08 m (4 kg) plus a white "head" box marking +x. Thigh and shin 0.16 m each.
  Hips at (±0.16, ±0.11). Standing torso height ≈ 0.26 m.
- **2026-09-24: Joint zero = legs straight down; standing pose = `home` keyframe**
  (hip +0.7, knee −1.4, which puts each foot under its hip, knees back). Positive
  angles swing a segment backward. Knee range [−2.6, 0], so it can't
  hyperextend. Hip range [−1.0, 1.8]. The RL env (M4) should read the default
  pose from the keyframe, not hard-code it.
- **2026-09-24: Motors = MuJoCo `<position>` actuators** (a PD controller:
  kp=40, kv=1, torque limit ±10 N·m, target range = joint range via
  `inheritrange`). Joints have armature 0.01 (rotor inertia) and damping 0.1.
  `data.ctrl` = target angles in radians, in actuator order FL_hip, FL_knee,
  FR_hip, FR_knee, RL_hip, RL_knee, RR_hip, RR_knee.
- **2026-09-24: Physics options:** timestep 2 ms, `implicitfast` integrator
  (stable with stiff PD damping), `cone="elliptic" impratio="100"`. The last
  one cut foot creep ~35× (7 mm → 0.2 mm over 5 s); MuJoCo Menagerie's robot
  dogs use the same.
- **2026-09-24: No self-collision:** robot geoms use contype=1/conaffinity=0,
  so they collide with world geoms (default 1/1) but not each other. Revisit
  if the legs visibly pass through each other in gaits.
- **2026-09-24: Real-time loop:** `Simulation.advance()` (src/robot3d/simulation.py),
  shared by the MuJoCo viewer script and the server. Physics is anchored to a
  (wall time, sim time) pair; each 60 fps frame steps until sim time catches
  up, capped at 50 steps per frame, then resyncs. `play()` resyncs so nothing
  "catches up" after a pause. The MuJoCo *passive* viewer only draws. It
  applies its own Backspace/Reset inside `viewer.sync()` (reset to qpos0), so
  the script detects time-going-backwards after `sync()` and resets to the
  keyframe instead.
- **2026-09-24: Frontend = Vite + TypeScript + three.js** (user's choice). All
  WebSocket message types live in `web/src/protocol.ts`, mirrored by Pydantic
  models, with the Vite dev server proxying `/ws` (user's spec).
  - Sync is enforced by `tests/test_protocol.py` (ts-json-schema-generator vs.
    Pydantic's `json_schema(mode="serialization")`, both reduced to a common
    "shape": field names, required fields, types, and fixed array lengths).
  - Pydantic models use `extra="forbid"`: unknown fields are rejected.
- **2026-09-24: JSON frames, not binary:** every message is JSON so it can be
  typed in protocol.ts and inspected in the browser devtools. Frames round
  poses to 1e-5 (~1 KB per frame, ~60 KB/s). Only dynamic geoms (body not
  welded to the world) are streamed; static scenery is sent once in the
  scene. Switch to binary Float32Array only if big scenes need it.
- **2026-09-24: One shared simulation per server**, like one real robot seen
  from several screens; any tab's play/pause/reset affects all of them.
- **2026-09-24: Server threading:** the sim runs on its own thread at a steady
  60 fps (time.sleep is ~1 ms precise on Python 3.11+ Windows); asyncio
  handles the network. Frames are serialized once and broadcast with
  `loop.call_soon_threadsafe`; commands go through a `queue.SimpleQueue`.
  Per client, only the newest frame is kept (slow clients skip frames);
  scene/status/error messages are queued and never dropped. Frames are sent
  only when sim time changed (none while paused). New clients get scene,
  status, then the latest frame.
- **2026-09-24: three.js runs z-up** (`Object3D.DEFAULT_UP = (0,0,1)` before
  creating the camera), so MuJoCo poses go in unchanged. Capsules and cylinders
  are rotated from three's y axis to MuJoCo's z axis. Colors are sRGB. Geom
  groups > 2 are hidden (as in MuJoCo's viewer). The start camera = MuJoCo's
  default free camera (`mjv_defaultFreeCamera`, sent in the scene). An infinite
  plane is drawn as our own floor + grid (0.2 m / 1 m lines). The sun,
  its shadow box, and the grid follow the center of the moving geoms.
- **2026-09-24: Backend binds 127.0.0.1 by default** (`--host 0.0.0.0` to allow
  other devices, e.g. a phone/tablet later). The Vite proxy targets
  127.0.0.1:8000 (env `ROBOT3D_BACKEND` overrides), because Node may resolve
  `localhost` to IPv6.

## Current status

**Milestone 2 done and confirmed by the user (2026-09-24). Next: Milestone 3
(manual control).**
- Verified in headless Edge (DevTools protocol), both via Vite (5173) and the
  production build served by the backend (8000):
  - streams at 60 fps; pause/reset/play buttons work (reset while paused
    shows the robot hanging at 0.40 m); play runs sim time at real time.
  - page reload works; auto-reconnect after a backend restart works.
  - no console errors in the production build.
- Parity: the web screenshot and MuJoCo's own offscreen render (same default
  camera, settled robot) match in framing, position, and every leg angle.
  Only the styling differs (MuJoCo: reflective checker floor, overhead light).
- Tests: 15 passing (`uv run pytest`), `tsc` clean.
- Deferred on purpose: keyboard shortcuts (M3), camera "follow robot" toggle
  (useful once it walks, M4).
- Git remote: `origin` = https://github.com/felixda9/robot3d.git, `main` pushed
  (M1 + M2). Push after each milestone commit.

## Notes for later milestones

- M4: Stable-Baselines3 PPO with small MLP policies usually trains *faster on CPU*
  than GPU; the bottleneck is stepping many envs in parallel. Benchmark both.
- M4: On Windows, `SubprocVecEnv` uses the `spawn` start method, so training
  scripts need an `if __name__ == "__main__":` guard.
- M4: To use CUDA, PyTorch has to come from the CUDA wheel index (configure it
  in `pyproject.toml` under `[tool.uv.sources]`).
