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
- CPU: i7-11700K, 8 cores / 16 threads, 32 GB RAM. Training uses **CPU-only
  PyTorch** (user's choice; pyproject pins torch to the pytorch-cpu index).
  Stable-Baselines3 2.9, Gymnasium 1.3.
- Commands (from the repo root unless noted):
  - `uv sync` and `cd web; npm install`: install everything
  - `uv run pytest`: all tests (physics, real-time loop, server over a real
    WebSocket, TS↔Python protocol contract; the contract test needs `web/node_modules`)
  - `uv run scripts/view_mujoco.py`: MuJoCo's built-in viewer
  - `uv run scripts/serve.py`: simulation server on http://localhost:8000
    (`--policy runs/<name>` = a trained policy drives, newest checkpoint)
  - `cd web; npm run dev`: frontend dev server on http://localhost:5173
    (start the backend too)
  - `uv run scripts/train.py`: PPO training (10M steps, ~30 min; `--steps`,
    `--envs`, `--name`, `--seed`); Ctrl+C stops and still saves a checkpoint
  - `uv run tensorboard --logdir runs`: training curves on http://localhost:6006
  - `uv run scripts/evaluate.py runs/<name> [--all]`: headless distance/speed/falls
    (also caches results for the dashboard)
  - Training dashboard: the **Training** tab (`#training`) of the web UI
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
  server.py             FastAPI app: sim thread, WebSocket, static files, --policy
  walk.py               WalkTask/WalkConfig: observation, action, reward (shared!)
  envs.py               WalkEnv (Gymnasium), registered as robot3d/Walk-v0
  training.py           train(): PPO, run folders, checkpoints, TensorBoard metrics
  runs.py               run folders for the dashboard: list/status/checkpoints/curves/
                        eval cache; Checkpoint + find_checkpoint (no PyTorch import)
  policy.py             PolicyController (plays a policy), evaluate()
scripts/                view_mujoco.py, serve.py, train.py, evaluate.py
tests/                  pytest (conftest.py: a tiny real training run, shared)
runs/<name>/            training output (gitignored): run.json, tb/, checkpoints/
web/                    Vite + TypeScript + three.js frontend
  src/protocol.ts       ALL WebSocket message types (single source of truth)
  src/connection.ts     WebSocket with auto-reconnect
  src/viewer.ts         three.js scene: camera, lights, ground, sky, geoms
  src/geoms.ts          MuJoCo geom -> three.js mesh, pose helpers
  src/motors.ts         motor panel: sliders, pose presets, torque bars
  src/api.ts            typed fetch client for the HTTP API
  src/dashboard/        training dashboard (dashboard.ts, linechart.ts SVG charts, css)
  src/main.ts           wiring + UI (views/tabs, buttons, stats, shortcuts, toast)
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
- [x] **3. Manual control:** one slider per motor in the web UI plus a few
  keyboard shortcuts. Commands go to the backend and move the robot.
- [x] **4. First training:** Gymnasium env for the quadruped (reward forward
  velocity and staying upright; penalize energy use and falling). PPO training
  script with parallel envs, checkpoints, TensorBoard logs. Load any checkpoint
  and watch it in the web viewer.
- [x] **5. Training dashboard:** training curves, checkpoint list, and replay of
  any checkpoint in the web UI.
- [ ] **6. Robot designer:** simple YAML/JSON robot spec (body parts, joints,
  motors) → generated MJCF; then a visual editor in the browser.
- [ ] **7. Environments & commands:** terrain, stairs, obstacles. Train a policy
  that follows direction + speed commands, controllable by keyboard or gamepad.
- [ ] **Future: GPU-parallel training on the RTX 3090** (evaluate MJX / MuJoCo
  Playground or Isaac Lab) once the CPU-based pipeline works end to end.

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
  from several screens; any tab's commands (play/pause/reset, motor targets)
  affect all of them.
- **2026-09-24: Server threading:** the sim runs on its own thread at a steady
  60 fps (time.sleep is ~1 ms precise on Python 3.11+ Windows); asyncio
  handles the network. Frames are serialized once and broadcast with
  `loop.call_soon_threadsafe`; commands go through a `queue.SimpleQueue`.
  Per client, only the newest frame is kept (slow clients skip frames);
  scene/status/error messages are queued and never dropped. Frames are sent
  when sim time changed or a command changed the state (e.g. new targets
  while paused). New clients get scene, status, then the latest frame.
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
- **2026-09-24: Manual control protocol (M3):**
  - The scene lists `actuators` (name, joint, ctrl_range, force_range) and
    `keyframes` (name, ctrl).
  - Every frame carries per-motor `ctrl` (current target), `joint_pos`
    (actual angle) and `torque`, all in actuator order.
  - `set_ctrl` is partial and addressed by actuator name
    (`{ctrl: {FL_knee: -1.2}, duration}`), so two tabs can move different
    motors at once.
  - The server clamps values to ctrl_range and rejects NaN/Inf (Pydantic
    `FiniteFloat`) and unknown names (checked on the event loop so the error
    goes back to the sender).
  - Radians on the wire; the UI shows degrees.
- **2026-09-24: Pose presets = the robot's MJCF keyframes** (robots are data).
  The UI shows one button per keyframe; number keys 1–9 pick them in file
  order. Only a keyframe's ctrl is applied. The quadruped has home, crouch,
  tall, and sit.
- **2026-09-24: Presets glide; sliders jump.** `set_ctrl.duration` > 0 moves
  the named motors' targets from current to new *together* (synchronized
  joint-space interpolation with smoothstep, in sim time, so pausing freezes
  a glide). Presets use 0.8 s; sliders use 0, so the user sets the speed.
  - Why: jumping targets flipped the robot on crouch → tall.
  - A per-joint *rate limit* made it worse (sit → tall flipped even at
    1.5 rad/s): joints with little distance to travel arrive early, and the
    lopsided in-between poses tip the robot backward.
  - Synchronized glides of ≥ 0.5 s were safe for all 12 transitions, and
    tests/test_quadruped.py checks them all at 0.8 s.
  - `Simulation.step()` = apply glides + mj_step.
- **2026-09-24: Keyboard shortcuts:** Space = play/pause, R = reset, 1–9 = pose
  presets (ignored with Ctrl/Alt/Meta so browser shortcuts still work).
  Buttons blur after a click so Space doesn't press them again.
- **2026-09-24: Motor panel UI:**
  - Slider thumb = target, white marker = actual angle, centered bar =
    torque (turns red at ≥ 95% of the limit).
  - While you drag, or for 300 ms after, frames don't overwrite that slider,
    so it doesn't jump back while your command is in flight.
  - The preset whose targets match the current ones is highlighted.
- **2026-09-24: One walk task, shared** (`walk.py` `WalkTask`): the Gymnasium
  env (training) and `PolicyController` (web playback, evaluate.py) use the
  same observation / action / settings, read from the run's `run.json`, so
  a policy sees and acts identically in both. The task is generic for our
  robot conventions (body 1 = free torso, position motors, `home` keyframe).
- **2026-09-24: Walk task design:**
  - Policy acts at 50 Hz (every 10 physics steps).
  - Action = target offsets in [−1, 1] × 0.5 rad around `home`, clipped to
    ctrl_range; action 0 = stand still.
  - Observation (34): torso height, gravity direction in the torso frame
    (tilt without heading), torso linear and angular velocity in the torso
    frame, joint angles relative to home, joint speeds, previous action.
    It contains no x/y position or heading, so walking works the same
    anywhere, in any direction (a test checks this).
  - Reward per step: +1.0 × forward speed (capped at 1 m/s, so reckless
    sprinting doesn't pay) + 0.5 × upright (torso up · world up)
    − 0.002 × motor power (W) − 0.05 × Σ(action change)² − 10 on falling.
  - Falling (ends the episode) = tilt > 60° or torso below half its
    standing height. Episodes are 20 s (1000 steps).
  - Episodes start from the *settled standing* state (simulated once at
    init) plus noise of ±0.1 rad on joints and ±0.1 on velocities.
- **2026-09-24: PPO setup (SB3):** SubprocVecEnv + VecNormalize (obs +
  reward). The normalizer stats are saved with **every** checkpoint, because a
  policy only works with the input scaling it was trained on. n_steps 1024 per
  env, 4 minibatches, 10 epochs, lr 3e-4, γ 0.99, λ 0.95, clip 0.2, [256, 256]
  ELU nets, log_std_init −1. Checkpoints every 500k steps + final (also on
  Ctrl+C). Custom TensorBoard scalars: `episode/distance_m`, `speed_mps`,
  `fell`, and `reward/<term>` (per-step mean of each reward term).
- **2026-09-24: Parallel envs default = logical CPU threads − 2** (14 here;
  user asked for a core-count-based default). Measured PPO throughput:
  4 envs 2.7k, 8 envs 4.5k, 12 envs 5.1k, 16 envs 5.6k steps/s. Hyperthreads
  help (workers are Python-overhead bound); 4 torch threads was slower than 8.
  With 14 envs: ~5.05k steps/s, so 10M steps ≈ 33 min.
- **2026-09-24: Policy playback:** `serve.py --policy <run|checkpoint>` →
  `Simulation.set_controller(PolicyController)`.
  - The controller acts every `decimation` physics steps in `Simulation.step()`.
  - While a policy drives, reset starts from the standing state (as in
    training), not from the keyframe drop.
  - Status carries `policy` (label) and `policy_active`.
  - `use_policy` toggles policy/manual. Manual starts from the policy's last
    targets; the P key and a button do the same.
  - `set_ctrl` is refused while the policy drives. The sliders are locked
    but still show its targets moving live.
- **2026-09-24: Follow camera** (checkbox + F, on by default): camera and orbit
  target move with the robot's center, horizontally only (no bobbing).
- **2026-09-24: Dashboard data over HTTP, not the WebSocket** (request/response
  data, not a stream):
  - `GET /api/runs`, `GET /api/runs/{run}`, `GET /api/runs/{run}/scalars?tags=`,
    `POST /api/runs/{run}/evaluate`.
  - Its types live in the same `protocol.ts` (section "HTTP API"), mirrored in
    protocol.py, and the contract test covers them.
  - Run/checkpoint names are validated as plain names (no paths) on both
    the HTTP and WS sides.
  - Vite proxies `/api` too; `ROBOT3D_BACKEND` is now `host:port`.
- **2026-09-24: Run status:**
  - `finished` / `stopped` come from run.json.
  - An unfinished run counts as `running` while run.json or its TB events
    changed within 120 s; training rewrites run.json every ~30 s
    (Heartbeat callback, atomic write with retries for Windows).
- **2026-09-24: Curves:** ScalarReader keeps one EventAccumulator per event
  folder and reloads incrementally (walk_10m: 440 ms cold, 7 ms warm).
  NaN/inf values are dropped; capped at 1500 points per series.
- **2026-09-24: Checkpoint evaluations cached** as
  `checkpoints/step_N_eval.json`. The dashboard's "Evaluate" queues missing
  ones on a single background thread; evaluate.py writes the cache too. The
  best checkpoint = highest mean return.
- **2026-09-24: load_policy** WS command `{run, checkpoint}`: loads on a
  worker thread (PyTorch), then swaps the controller in on the sim thread
  (robot restarts standing). The dashboard's "Watch" = load_policy + switch
  to the Simulator tab.
- **2026-09-24: Charts** are a small in-house SVG component (`linechart.ts`),
  following the dataviz skill:
  - 2px lines, hairline solid grid, snapping crosshair with one tooltip
    listing every series, legend only for 2+ series, keyboard arrows,
    "Show data" table view, and a log y-axis for KL.
  - Palette: the reference dark categorical slots, validated against our
    chart surface #141a22 (all checks pass).
  - A run keeps its color slot while checked; up to 8 runs overlaid.
  - The reward-terms chart shows the selected run only.

## Current status

**Milestone 5 done (2026-09-24), waiting for the user to test.** (M4
confirmed by the user, who then asked me to run the full training.)
- **First full run `runs/walk_10m`** (10M steps, 14 envs, 33m49s, ~4.9k steps/s):
  - The robot learned a **canter** by ~4M steps: 3-beat, RR → FR+RL together
    → FL, airborne ~30–34% of the cycle, ~4.2 steps/s per foot.
  - No falls at any checkpoint in deterministic evaluation (5 episodes each).
  - Return plateaus ~1420 from 4.5M on.
  - Best by return: 6.0M (1425, 1.36 m/s); 9M is a tie within noise
    (1423.5, 1.43 m/s, 31 W mean motor power vs 58 W at 1M).
  - The newest checkpoint (10.01M) is weaker (1379).
  - Gait at 1M: front legs hopped together and RL skittered (19
    touchdowns/s, 0.41 m/s slip). Gone by 4M.
  - Remaining flaw: feet slide ~0.25 m/s while touching the floor.
- **PPO instability (full-run numbers, from TensorBoard):**
  - approx_kl above 0.05 in **114** of 697 updates, **max 50.5** (at 9.48M);
    spikes grew through the whole second half. (An earlier mid-run count said
    14 / max 6.5; that only covered the log up to 7.1M.) Clip fraction ~0.3.
  - Cause: the policy std collapsed to ~0.04–0.06 (from 0.37) with lr fixed
    at 3e-4, so small action changes = huge KL.
  - Snapshots right after spikes were bad: 2.5M (0.13 m/s), 7.0M (0.56 m/s).
  - **Suggested fixes for the next run** (not applied yet; user to decide):
    `target_kl≈0.02`, linear lr decay, possibly a std floor or a small
    ent_coef, and a foot-slip penalty in the reward.
- Dashboard verified in headless Edge:
  - runs list + overlay (walk_10m vs smoke);
  - 7 curve charts + reward terms;
  - evaluated all 21 walk_10m checkpoints in 63 s, best marked;
  - tooltip, keyboard reading and table view work;
  - Watch loads the checkpoint in the simulator;
  - no console errors.
- Tests: 63 passing, `tsc` clean.
- GPU research done (see Notes); not acted on yet. User asked whether to do
  GPU-parallel training sooner.
- Git remote: `origin` = https://github.com/felixda9/robot3d.git. Push after
  each milestone commit.

## Notes for later milestones

- GPU research (2026-09-24, via a research agent, primary sources):
  - JAX has **no CUDA on native Windows** (WSL2 is "experimental"), so MJX and
    MuJoCo Playground need WSL2.
  - **MuJoCo Warp** (`mujoco-warp` 3.14.0 on PyPI, versioned with MuJoCo)
    runs on NVIDIA Warp, which has Windows CUDA wheels. The community reports
    its test suite passing on native Windows. It uses float32.
  - mjlab (MJWarp + RSL-RL, Isaac-Lab-style API, Go1 velocity task, terrain):
    Linux-first, Windows "preliminary"; pins mujoco~=3.11; ignores MJCF
    `<option>` (set it in code).
  - Isaac Lab/Sim 6.1 needs Windows 11 + an RTX 4080 minimum and uses PhysX,
    not MuJoCo. Not a fit.
  - Recommended path: MJWarp + PyTorch (own batched env or mjlab), with
    ONNX/weights export to run in CPU MuJoCo. Verify sim-to-sim with
    identical options.
- M6+: switching robots at runtime (load_policy for another robot) needs the
  server to rebuild the simulation and resend the scene.
