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
- CPU: i7-11700K, 8 cores / 16 threads, 32 GB RAM. PyTorch 2.14 **CUDA 13.0
  build** (pyproject pins torch to the pytorch-cu130 index; it started CPU-only,
  switched for GPU training). Stable-Baselines3 2.9, Gymnasium 1.3,
  mujoco-warp 3.14 (Warp 1.17), rsl-rl-lib 5.5.
- Commands (from the repo root unless noted):
  - `uv sync` and `cd web; npm install`: install everything
  - `uv run pytest`: all tests (physics, real-time loop, server over a real
    WebSocket, TS↔Python protocol contract; the contract test needs `web/node_modules`)
  - `uv run scripts/view_mujoco.py`: MuJoCo's built-in viewer
  - `uv run scripts/serve.py`: simulation server on http://localhost:8000
    (`--policy runs/<name>` = a trained policy drives, newest checkpoint)
  - `cd web; npm run dev`: frontend dev server on http://localhost:5173
    (start the backend too)
  - `uv run scripts/train.py`: PPO training on the CPU (10M steps, ~30 min;
    `--steps`, `--envs`, `--name`, `--seed`); Ctrl+C stops and still saves a checkpoint
  - `uv run scripts/train_gpu.py`: training on the GPU (MuJoCo Warp, 4096 robots;
    `--trainer ppo|rsl`, `--steps`, `--envs`, `--name`, `--seed`,
    `--checkpoint-every`, `--epochs`, `--minibatches`, `--gait-hz`)
  - `uv run tensorboard --logdir runs`: training curves on http://localhost:6006
  - `uv run scripts/evaluate.py runs/<name> [--all]`: headless distance/speed/falls
    (also caches results for the dashboard)
  - Training dashboard: the **Training** tab (`#training`) of the web UI
  - `cd web; npm run build`: type-check (`tsc`) + production build to `web/dist/`
  - `cd web; npm run typecheck`: type-check only

## Layout

```
robots/                 MJCF robot files (data, not code): quadruped.xml (8 motors),
                        quadruped12.xml (12 motors: + a sideways "roll" joint per hip)
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
  policy.py             PolicyController (plays a policy), evaluate() + gait numbers
  gpu/                  GPU training (MuJoCo Warp + PyTorch):
    task.py             BatchedWalkTask: walk.py as batched torch math (parity-tested)
    env.py              GpuWalkEnv: N robots, CUDA graph physics, auto-reset
    ppo.py              own PPO (train_gpu), ActorCritic, TorchPolicy (plays .pt on CPU)
    rsl.py              RSL-RL's PPO driven by our env (train_rsl), TorchScript export
    common.py           run folder + TensorBoard logging shared by both trainers
scripts/                view_mujoco.py, serve.py, train.py, train_gpu.py, evaluate.py
tests/                  pytest (conftest.py: a tiny real training run, shared)
runs/<name>/            training output (gitignored): run.json, tb/, checkpoints/
web/                    Vite + TypeScript + three.js frontend
  src/protocol.ts       ALL WebSocket message types (single source of truth)
  src/connection.ts     WebSocket with auto-reconnect
  src/viewer.ts         three.js scene: camera, lights, ground, sky, geoms
  src/geoms.ts          MuJoCo geom -> three.js mesh, pose helpers
  src/motors.ts         motor panel: sliders, pose presets, torque bars
  src/interaction.ts    mouse grab (drag a part) and push (double-click) on the robot
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
- [ ] **5b. GPU-parallel training** (moved up from "Future" on 2026-09-24 at the
  user's request): MuJoCo Warp physics + batched walk task + PPO in PyTorch on
  the RTX 3090; same metrics/dashboard as CPU runs; GPU checkpoints replay
  in CPU MuJoCo. **Pipeline done and verified. Since separate gradient
  clipping, our GPU PPO is stable** (trot_ppo: 18/18 checkpoints good);
  now used for the trot-walk (see Decisions, 2026-09-25).
- [ ] **5c. Robustness** (added 2026-09-25 at the user's request: "always
  stand upright and not fall in any circumstance, even get back up if it
  fell", tested by grabbing and pushing the robot with the mouse):
  1. [x] Mouse grab + push in the viewer, push force adjustable.
  2. [x] A 12-motor robot (a sideways hip joint per leg, like real robot
     dogs; user's choice, since the 8-motor legs can't roll it back over
     from its side), trained to walk while being shoved at random.
  3. [ ] Three policies with an automatic switch (Walk / Stand modes in
     the viewer): **walk** (trained with shoves), **stand** (the user: "its
     goal should be to not move and stay in a stable position": home pose,
     still, back to it after a shove, try not to fall; on uneven ground
     "stabilise itself to a point where it's not moving and is stable" →
     needs terrain in training, M7), and **getup** (after a fall, in either
     mode, until standing steady; user's original choice of a separate
     get-up policy, like ANYmal's recovery controller).
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
- After changing `protocol.ts`/`protocol.py`, **restart any running backend**
  (`uv run scripts/serve.py`). Vite hot-reloads the page at once, and a page
  speaking the new protocol to an old server breaks (this happened once:
  the dashboard showed "No data yet" until the server was restarted).

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

- **2026-09-24: CPU training fixes** (after walk_10m's KL spikes):
  - PPOConfig: `lr_decay` (linear to 0, `LinearDecay` class so checkpoints
    pickle) and `target_kl=0.02`.
  - WalkConfig: `slip_weight=0.5` penalizes the sum of squared sliding speeds
    of feet on the ground for the whole control step. Feet = geoms named
    `*_foot`; on the ground = lowest point within 5 mm of the floor.
  - `WalkConfig.from_run()` fills settings missing from old runs with their
    original values (old runs: slip_weight 0).
- **2026-09-24: GPU training architecture (milestone 5b):**
  - `gpu/task.py` BatchedWalkTask: the walk task as batched torch math.
    `tests/test_gpu_task.py` checks it equals WalkTask on the same states.
  - `gpu/env.py` GpuWalkEnv:
    - MuJoCo Warp (`mujoco-warp` 3.14, native Windows) with N worlds.
      PyTorch reads and writes the Warp arrays through `wp.to_torch` views,
      with no copies.
    - The 10 physics steps per policy step, plus a Warp kernel adding up
      motor power, are recorded once as a CUDA graph and replayed each step.
    - Warp runs on PyTorch's CUDA stream.
    - Robots that fall or time out restart standing, using
      `mjw.reset_data(mask)` + `kinematics`.
    - Stale-pose parity: after mj_step, MuJoCo's xmat/geom_xpos trail qpos
      by one physics step. Observations and foot positions are taken before
      the reset's kinematics call, so every robot sees what the CPU env sees.
  - `gpu/ppo.py`: own compact PPO in the RSL-RL style (Stable-Baselines3
    isn't built for GPU-batched envs):
    - 4096 envs × 24 steps per rollout, 5 epochs × 4 minibatches;
    - adaptive learning rate from the exact Gaussian KL (target 0.01),
      clipped value loss, entropy 0.005;
    - time-outs bootstrapped with V(s);
    - same [256, 256] ELU networks and observation normalization as the CPU
      runs;
    - logs the same TensorBoard tags as SB3 (approx_kl uses SB3's estimator),
      so the dashboard compares runs directly;
    - checkpoints are `step_N.pt` (network + normalizer, format
      "robot3d-ppo-v1").
  - `runs.py` accepts `.zip` (SB3) and `.pt` (GPU) checkpoints.
    PolicyController plays both on the CPU, so GPU-trained policies are
    evaluated in regular float64 MuJoCo (the sim-to-sim check).
    RunSummary has `backend` ("cpu"/"gpu"; old runs = cpu), shown as a pill
    in the dashboard. The throughput chart uses a log axis.
  - Checked on Warp's CPU backend: GpuWalkEnv vs WalkEnv from the same state
    with the same actions give the same observations (~1e-4) and rewards
    (1e-4) over 10 steps.
  - PyTorch must be the CUDA build for GPU training (cu130 wheels exist for
    torch 2.14 on Windows; driver 581.57 supports CUDA 13). A CUDA build also
    runs everything on the CPU. It can't be swapped while a CPU training run
    has PyTorch loaded (Windows file locks).
  - CUDA graphs can't be recorded on PyTorch's legacy default stream, so
    GpuWalkEnv creates its own torch.cuda.Stream and makes it current
    (`torch.cuda.set_stream`).
  - GPU env throughput (random policy, no learning): 4096 robots 112k,
    8192 151k, 16384 186k policy steps/s (vs ~5k/s CPU training).
  - Robots start at a random point of their first episode (as legged_gym
    does). Otherwise all of them hit the 20 s limit together, and until then
    the only finished episodes are falls: early "falls" read 100%.
- **2026-09-25: GPU tuning experiments** (30M steps each, ~6.5–7.5 min;
  evaluated in CPU MuJoCo):
  - v1 `walk_gpu_30m` (legged_gym-like: init std 0.5, entropy 0.005,
    RSL-RL adaptive lr): best return 1238.
  - v2 `walk_gpu_v2` (init std 0.37, entropy 0.001): best 1283 at 17.6M
    (3.8 min). Checkpoints alternated between good and barely moving;
    adaptive lr swung between 1e-5 and 2e-3.
  - v3 `walk_gpu_v3` (+ linear lr decay from 1e-3 + target_kl 0.02 early
    stop + random episode starts): KL tame (max 0.014), best 1313 at 10M,
    but checkpoints still alternate (15M: −0.18 m/s). Diagnosis: far fewer
    gradient steps than CPU (~5k total vs ~28k), so std stayed 0.25 (CPU
    0.05) and the noise-free "mean" policy is unreliable.
  - v4 `walk_gpu_v4`: 10 epochs × 8 minibatches (up to 80 gradient steps
    per rollout). std settled more (0.36 → 0.14), best 1326 at 12.6M
    (2.9 min), but checkpoints still alternate (4/8 good after the first
    good one); training reward also dips to ~500, which is "stand still".
    **Current defaults = v4.**
  - v5 `walk_gpu_v5` (v4 + SB3-style reward normalization + no value
    clipping): stuck standing still for all 30M steps. The options remain,
    off by default. One seed each, so no config comparison is conclusive.
  - Summary, time to first checkpoint with return ≥ 1200 (CPU-MuJoCo eval):
    - CPU fixed: 4.0 min; best 1421; 20/20 later checkpoints good.
    - GPU v3/v4: 2.2–2.9 min; best 1313–1326; 3/9 and 4/8 good.
    - GPU runs 15–18× more steps/s but need 10–20× more steps, so the
      wall-clock to a good walker is about equal for this simple task, and
      CPU is more reliable.
  - Candidate next steps (user to choose):
    - run a reference GPU PPO (RSL-RL) on the same GpuWalkEnv, to tell an
      issue in our PPO from the massive-parallel regime;
    - multi-seed sweeps (cheap on GPU: ~7 min per 30M run) over lr (3e-4 vs
      1e-3), reward normalization, and fewer envs (1024–2048) with more steps
      per env.
    - GPU should matter most for harder tasks (terrain, commands), where
      100M+ steps would take hours on the CPU.
  - Transfer works throughout: every GPU checkpoint runs in float64 CPU
    MuJoCo without falls.
- **2026-09-25: Gait feedback → a trot-walk task** (user: both walkers "run";
  GPU gait more natural with longer strides, CPU tiny rapid steps; both ~45%
  of the time airborne). User chose: **trot-walk at 0.4 m/s, trained on the
  GPU, check RSL-RL first**. The first version of the task:
  - `tracking` = 2 × exp(−speed error² / σ) around target_speed 0.4 m/s
    (replaces raw forward speed; `forward_weight` 0 keeps the old term for
    old runs);
  - `support`: −1 per step with fewer than 2 feet down (no flight phases);
  - `trot`: diagonal feet (FL+RR, FR+RL) in the same state;
  - `air_time`: per landing, 2 × (swing time − 0.25 s), legged_gym style.
  - `WalkConfig.from_run` fills settings missing from older runs with the
    values that reproduce them, and raises a clear error for settings it
    doesn't know (a run trained by newer code than the running server).
- **2026-09-25: RSL-RL** (`gpu/rsl.py`, rsl-rl-lib 5.5): its PPO class driven
  from our own loop (our GpuWalkEnv, not its VecEnv/runner), legged_gym's
  recipe (4096 × 24, 5 epochs × 4 minibatches, adaptive lr on KL 0.01,
  entropy 0.01, [512, 256, 128] ELU, init std 1.0, rewards × 0.02 as in
  legged_gym's dt scaling). Checkpoints store a TorchScript actor (with its
  observation normalizer) in the `.pt` ("robot3d-jit-v1"); TorchPolicy plays
  both .pt formats. (torch.jit prints deprecation warnings; still works in
  2.14. Move to torch.export when it stops.)
- **2026-09-25: Our PPO clips actor and critic gradients separately** (as
  RSL-RL does). Clipped together, the critic's gradients (value errors in
  reward units) dominate the shared norm, which shrinks every policy update
  by an arbitrary factor: a likely cause of the v1–v4 good/bad alternation.
- **2026-09-25: Gait numbers in evaluations** (`policy.gait_numbers`, after
  the first second): duty factor (share of time each foot is down; walk
  > 50%), airborne share (all four up), diagonal sync, cadence (touchdowns
  per foot per second). Optional (null) fields of EvaluationInfo, shown as
  columns in the dashboard's checkpoint table. Reward terms are split into
  "Rewards (+)" and "Penalties (−)" charts (≤ 8 lines each).
- **2026-09-25: Evaluation failures are shown**, not just logged: RunDetail
  has `evaluation_error` (cleared when evaluation is requested again),
  shown under the checkpoint table. (Happened: a server started before the
  walk-task change couldn't read trot_rsl's settings; Watch and Evaluate
  failed silently.)
- **2026-09-25: `trot_rsl` result** (RSL-RL, 50M steps, 13 min): stable
  (0 falls in the last 6 checkpoints), 0.31 m/s, never airborne, but
  **it scoots**: feet down 94% of the time, back feet never lift (max
  1.7 mm in an episode), front feet ~1 cm. The trot term paid 0.44 of 0.5
  because "all four down" counts as in sync, and air_time only penalizes
  short swings (never lifting costs nothing). A reward flaw, not a trainer
  problem.
- **2026-09-25: `trot_ppo` result** (our PPO with separate gradient
  clipping, same reward as trot_rsl, 50M steps, ~17 min): **a real
  trot-walk.** From 7.6M steps on, 18/18 checkpoints walk without falls, and
  the return rises steadily (no good/bad alternation any more). At 50M:
  0.39 m/s (target 0.4), feet down 55%, never airborne, diagonal sync
  1.00, all four feet lift 3–4.5 cm, planted feet barely slide (slip
  −0.002 per step vs −0.132 for trot_rsl). So the trainer matters: on the
  same flawed reward, RSL-RL found the scoot and ours didn't. **Our PPO is
  the GPU trainer from here** (user's rule: "whichever GPU trainer is
  stable"); RSL-RL stays as a reference (`--trainer rsl`).
- **2026-09-25: Gait clock** (user's choice over reward fixes alone): a
  2 Hz rhythm, the standard method for gaits in legged RL (periodic reward
  composition, e.g. Siekmann et al. 2021; walk-these-ways):
  - phase = step × control_dt × gait_frequency mod 1, from the integer step
    count in float64 on CPU and GPU (bit-identical, even at cycle
    boundaries); the observation gets (sin, cos) of it: 34 → 36 numbers.
    GPU robots with random episode starts get random phases.
  - Schedule: FL+RR down for phase [0, 0.6), FR+RL for [0.5, 1.1): each foot
    down 60% of the cycle (a walk), both pairs down twice per cycle for
    0.05 cycle.
  - `gait` = share of feet whose contact matches the schedule (weight 1.0);
    `clearance` = mean over scheduled-swing feet of min(height / 4 cm, 1)
    (weight 0.5). `trot` and `air_time` default to 0 now.
  - tracking_sigma 0.25 → 0.1: at 0.25, 0.31 m/s already earned 97%.
  - Runs from before the clock load with gait_frequency 0 (34 observations,
    same rewards); checked: trot_rsl and walk_cpu_fixed evaluate to exactly
    their cached returns.
- **2026-09-25: `trot_clock` result** (our PPO + gait clock, 50M steps,
  ~16 min): 19/19 checkpoints from 5M steps on walk without falls, return
  rising steadily. At 50M: 0.40 m/s, feet down 59% (schedule: 60%),
  exactly 2.0 steps per foot per second, diagonal sync 0.98, contact
  matches the schedule 98.5% of the time, feet lift 5–6 cm (above the
  4 cm the clearance reward pays for), slip −0.009 per step.
  - vs trot_ppo (no clock): same speed, but quicker, shorter steps
    (2.0 vs 1.3 per second, ~20 vs ~30 cm per cycle), higher lifts, and
    twice the motor power (~25 vs ~12 W). A lower gait_frequency (e.g.
    1.5 Hz: ~27 cm) would give longer strides with the clock.
  - The user watched both: the clock walk "looks a lot faster and more
    consistent" (same 0.4 m/s; it looks faster from the higher cadence).
    **The gait clock is the default gait.** User chose to try it slower:
    `trot_clock_15` = 1.5 Hz (`train_gpu.py --gait-hz 1.5`, ~27 cm
    strides) for comparison with 2 Hz; the winner sets the default
    gait_frequency.
  - `trot_clock_15` result (1.5 Hz, 50M steps): 0 falls from 7.6M on;
    at 50M 0.40 m/s, 1.47 steps per foot per second (~27 cm per cycle),
    feet down 59%, diagonal sync 0.99, feet lift 6.5–8 cm (higher than
    at 2 Hz: the longer swing leaves more time), motor power ~25 W (same).
  - **The user preferred 1.5 Hz: default gait_frequency = 1.5** (runs
    keep the frequency saved in their run.json).
- **2026-09-25: Mouse grab and push (5c step 1):**
  - Protocol: `grab {geom, point, target}` (resent as the mouse moves),
    `release`, `push {geom, point, direction, force}`. Points are in the
    geom's own frame (three.js mesh frame = MuJoCo geom frame), so a grabbed
    spot stays on its part. The server refuses static-world geoms and zero
    directions, and releases a grab when its browser disconnects.
  - Physics in `Simulation` via `data.xfrc_applied` (set before every
    mj_step, cleared when nothing acts): the policy feels it like any
    external force and is never told.
    - Grab = critically damped spring, stiffness scaled by the whole
      robot's mass (2 Hz; sags ~6 cm when lifting the robot), force capped
      at 3 g × mass. Grabbing off-center tilts the robot, as it should.
    - Push = force held for 0.1 s of sim time (60 N on 6.6 kg ≈ 0.9 m/s),
      horizontal, away from the camera through the clicked spot (along the
      view when looking straight down). Waits while paused.
  - UI: left-drag on a robot part grabs it (after 4 px, so clicks and
    double-clicks don't), anywhere else rotates the camera (a capture-phase
    listener disables OrbitControls before it sees the press); double-click
    pushes; "Push force" slider 10–300 N (default 60, remembered per
    browser). An orange line + dot shows the grab, an arrow flashes on a
    push.
  - Checked with real mouse events in headless Edge: lifting the torso,
    release, a 150 N push (knocked the standing robot over, away from the
    camera), and a drag on empty space (camera rotates, robot untouched).
- **2026-09-25: 12-motor robot `quadruped12`** (5c step 2, user's choice):
  - quadruped.xml plus a roll joint per hip (`<leg>_roll`, hinge about x,
    ±0.8 rad like Spot/ANYmal's ~0.75; positive = foot swings to the
    robot's left). A short hip link (0.15 kg capsule) carries the thigh from
    the roll axis (y = ±0.08, the torso's edge) out to y = ±0.11, where
    quadruped.xml has its hips, so both stand alike (torso 0.26 m).
    7.2 kg total. Motors ordered leg by leg: roll, hip, knee.
  - **Legs collide with the torso** (torso contype 1/conaffinity 2, leg
    parts contype 3/conaffinity 0; legs still pass through each other).
    quadruped.xml's legs pass through its body; a get-up policy must not
    learn to cheat that way.
  - Presets: home, crouch, tall, sit as before, plus "wide" (roll 0.3
    outward). tests/test_quadruped.py runs every check for both robots
    (drop and settle, all 20 preset transitions).
  - The walk task needed no changes (48 observations, 12 actions); 4096
    GPU robots flailing at random: no contact-buffer overflow.
- **2026-09-25: Random shoves in training** (5c step 2): WalkConfig
  `push_interval` 4 s (each gap random in 2–6 s) and `push_max_speed`
  1 m/s: the torso's horizontal velocity jumps by up to ±1 m/s in x and y
  (legged_gym's method; it pushes every 15 s). On by default for new runs,
  also during evaluation (it's part of the task); off for older runs
  (from_run). CPU env: np_random; GPU env: per-robot schedule, applied
  without a GPU→CPU sync. Tests that compare exact physics turn it off.
- **2026-09-25: Watching another robot's policy switches the robot**:
  load_policy for a run trained on another robot builds a new Simulation
  of that robot (off the sim thread), the sim thread swaps it in and
  broadcasts the new SceneMessage; the viewer and motor panel rebuild.
  (Was: an error "trained on robot X, but this server simulates Y".)
- **2026-09-25: `walk12_push` walked in circles** (user watching it
  mid-training): at 30M steps −28 °/s, a full circle every ~13 s, in every
  episode; 0.28 m/s along its nose, 0.19 m/s along world +x. Cause: the
  speed reward was along world +x, but the policy never observes its
  heading, so after shoves (and with roll joints that make turning easy)
  its reward changed for reasons it couldn't sense. (The 8-motor walkers
  turn with difficulty and went straight on their own: trot_clock_15
  drifts < 1° in 20 s.) Stopped at ~30M; kept as a record.
- **2026-09-25: Speed in the robot's heading frame + a turn reward** (the
  legged_gym standard, and the interface M7's commands will use):
  - `velocity_frame` "body": forward/sideways speed relative to the torso's
    heading (its x axis flattened onto the ground); "world" for older runs.
  - `turn` = 0.5 × exp(−turn_rate² / 0.25), turn_rate = torso angular
    velocity about its own z (qvel[5]): 10 °/s keeps 88%, 30 °/s 33%.
  - Consequence: after a shove turns it, the robot walks straight in its
    new direction; steering back needs a heading target (M7 commands).
  - Episode distance/speed stay along world +x (robots start facing +x).
- **2026-09-25: `walk12_straight`** (heading-frame speed + turn reward,
  stopped at ~25M): went straight (0.8 °/s, 0.38 m/s along its nose), but
  the user saw it "using its roll too much": three legs held 16–19° out,
  6–9° roll swing per step, lopsided hips (one 25° back). trot_clock_15 is
  symmetric. → `roll_weight` 1.0: penalty per rad² of roll angle (joints
  named `<leg>_roll`), legged_gym's "hip_pos"; 0 for the stand task, which
  needs roll to get up, and for older runs.
- **2026-09-25: Episode distance** = straight-line distance from the start
  for heading-frame runs (a robot shoved onto a new heading walks on that
  way; world-x progress read 0.18 m/s for a 0.38 m/s walker); progress
  along +x for older runs, so their numbers stay comparable.
- **2026-09-25: The stand task** (`WalkConfig.stand()`, `train_gpu.py
  --task stand`): the walk task with standing settings, so the same robot
  interface, GPU pipeline and viewer work:
  - target speed 0 (tracking weight 1), no gait clock, turn reward;
  - new posture terms: `height` (torso height / standing height, capped
    at 1), `pose` (exp(−Σ(joint − home)² / 1 rad²), weight 0.5), `down`
    (−1 per step spent fallen); upright weight 1.0;
  - a fall doesn't end the episode (`terminate_on_fall` off; no one-off
    fall penalty then); shoves up to 1.5 m/s;
  - half the episodes start fallen (`fallen_start_fraction` 0.5), from a
    bank of 128 fallen states made once on first use (~1 s): dropped from
    0.5 m at a uniformly random orientation with random joint angles,
    settled 1 s; >70% land fallen. The viewer always resets it standing
    (`allow_fallen=False`).
  - All new settings default to "off", so walkers old and new are
    unchanged. Evaluations get `upright` (share of time not fallen; a new
    dashboard column). Reward charts: "moving", "posture", "penalties";
    all-zero terms are hidden.
- **2026-09-25: Walk / Stand modes** (user: a stand-only mode to test
  stability, and "the two mixed so that while it's walking and I shove it,
  it keeps upright while walking"):
  - `policy.Behaviors` holds a walk and a stand policy and drives the sim
    (Controller protocol). Stand mode: the stand policy. Walk mode: the
    walker; if the robot falls (WalkTask.fell) and a stand policy is
    loaded, the stand policy takes over (`recovering`) until the torso is
    level (up_z > 0.9) and high (> 80% standing height) for 0.5 s, then
    the walker resumes. The walker itself trains with shoves.
  - Protocol: StatusMessage has `walk_policy`, `stand_policy`, `mode`,
    `recovering` (replacing `policy`); new `set_mode {mode}` (also hands
    control back to the policy). load_policy puts a run in its slot by
    its task (`WalkConfig.is_stand`) and switches to that mode; switching
    robots forgets the other slot.
  - UI: Walk / Stand buttons in the policy box (disabled until such a
    policy is loaded), M switches, "fell, getting up…" while recovering.
- **2026-09-25: `stand12` (first stand run) learned to lie still**: time
  fallen rose from 44% to 77% in the first 3M steps (10M: still 66%),
  because "stand still" (tracking) and "standing pose" paid while lying
  too: motionless on its back with legs in the standing pose earned 1.3
  per step. → `posture_gating` (stand task): tracking, turn and pose are
  multiplied by clip(up_z, 0, 1), so lying earns nothing for keeping still
  and only getting up pays. Stopped at 10M; `stand12_gated` restarted.
- **2026-09-25: `stand12_gated` learned to balance, not to get up**
  (checked at 10M steps from 64 fallen poses, no shoves): on its back
  0/46 up after 10 s, on its side 1/15, belly/feet 2/3; from standing
  8/8 stayed up 20 s. (The fallen bank is ~72% "on its back".) Two
  physical limits, not rewards:
  1. actions reached only ±0.5 rad around the standing pose (the walk
     setting): getting up needs big movements → the stand task uses
     `action_scale` 2.0 (≈ each joint's full range);
  2. on its back the legs couldn't reach the floor: with hip pitch up to
     1.8 rad, swinging straight legs over lifts the torso 1 cm and no
     more. → quadruped12 hip range −2.0..2.8 (was −1.0..1.8, like
     ANYmal's far-swinging legs; they sit outside the torso's width). At
     2.8 the same motion flips it onto its belly (checked in simulation).
     Walkers stay far from the limits (±0.5 rad around home).
  → `stand12_reach` (80M steps).
- **2026-09-25: `walk12_tidy` result** (quadruped12, 1.5 Hz clock, shoves,
  heading-frame speed + turn reward + roll penalty; 50M steps): every
  checkpoint from 10M on walks with 0 falls (evaluated with shoves on); at
  50M 0.39 m/s, feet down 56%, diagonal sync 0.96, 1.58 steps/s,
  straight (0.1 °/s). Roll: 0–8° held out, 3–4° swing (walk12_straight:
  16–19°, 6–9°); slightly asymmetric still (left legs 6–8° out, rear
  hips differ). **Shove survival** (scratchpad push_survival.py: walk 3 s,
  one shove from each of 16 directions, still up 3 s later):
  | shove (m/s) | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
  | trot_clock_15 (8 motors, never shoved) | 100% | 81% | 31% | 12% | 0% | 0% |
  | walk12_tidy | 100% | 100% | 93% | 81% | 62% | 18% |
  (1 m/s ≈ 70 N on the viewer's push slider.)
- **2026-09-25: Stand and get-up split** (the user watched stand12_reach:
  "just moving way too much"). One policy for calm standing and for
  getting up couldn't work well: getting up needs ±2 rad actions, which
  make every twitch while standing 4× bigger, and its reward never asked
  the legs to be still. Now `WalkConfig.task` = "walk" | "stand" |
  "getup" (runs without it: getup if falls didn't end episodes, else
  walk):
  - `stand()`: ±0.5 rad actions; a fall ends the episode (fall penalty);
    pose weight 1.0 with a tight σ 0.1 rad² (3° off on every joint keeps
    74%); new `joint_speed` (−0.01 per (rad/s)²) and `wobble` (−0.5 per
    (rad/s)² of torso tipping) terms; smoothness 0.1; shoves ≤ 1.5 m/s.
  - `getup()`: see the next entry.
  - Behaviors: slots walk / stand / getup. Loading walk or stand switches
    to that mode; a get-up policy takes over after a fall in either mode
    until `WalkTask.steady` (level within ~25°, > 80% height) for 0.5 s.
    Stand mode can use the get-up policy when no stand policy is loaded.
    StatusMessage gains `getup_policy`; the policy box shows it.
- **2026-09-25: `stand12_reach` collapsed into lying on its belly** at
  24–36M steps (time fallen 36% → 96%, energy −0.40 → −0.02, std stuck
  at 0.15): lying level still paid "upright", "still", "don't turn"
  (the gate only looked at tilt), and it was safe from the shoves. At 80M:
  stays up from standing 1/8 (8/8 at 20M). → `getup()` redesigned:
  - every episode starts fallen (fraction 1.0), 10 s, no shoves;
  - it ends on success: `steady` for 0.5 s → `success_bonus` 10;
  - only progress terms (upright 1.0, height 1.0) and `down` 2.5 per
    fallen step, so every fallen step scores < 0 (fallen ⇒ upright +
    height ≤ 1.5): lying still never pays, getting up sooner is better;
  - no tracking / turn / pose / support terms (standing still is the
    stand policy's job); ±2 rad actions.
  Runs: `stand12_calm` (stand, 50M) and `getup12` (getup, 80M).
- **2026-09-25: `stand12_calm` results and a PPO stall** (stopped at ~36M):
  - Stillness (no shoves, 10 s): at 17.6M joint speed 0.006 rad/s, 3.6°
    off home (the motors alone holding home: 2.1°, legs sag), torso
    wobble 0.3 °/s, 5 cm drift: essentially the home pose, still.
  - Shoves (16 directions): 20M 87 / 68 / 50 / 50 / 25 / 12% at
    0.5 … 3.0 m/s; 30M 100 / 68 / 62 / 43 / 31 / 31%. Too stiff: it
    doesn't step to catch itself (walk12_tidy: 100 / 100 / 93 / 81 / 62 / 18%).
  - **Stall:** from 21M, `train/update_fraction` 0.013 = 1 of 80 planned
    minibatch steps per update. Its std had shrunk to 0.10, and the exact
    Gaussian KL grows as Δmean²/std², so one Adam step already passed the
    early-stop threshold (1.5 × target_kl = 0.03); the logged KL is from
    before that step (0). Learning stopped while the observation
    normalizer kept updating, so behavior drifted: 26° off home at 32M.
    (walk12_tidy reached std 0.097 but kept update_fraction 1.0: smaller
    policy gradients.) → the KL early stop needs replacing (e.g. RSL-RL's
    adaptive learning rate) or a floor; decided with the open-source survey.
- **2026-09-25: Open-source survey → new stand/getup recipe** (user: "check
  open sourced stuff and find how other people have done it"; details in
  Notes for later milestones). Changes:
  - PPO: `lr_schedule` "adaptive" (RSL-RL: after each minibatch lr /1.5 if
    KL > 2 × 0.01, ×1.5 if < 0.005, within [1e-5, 1e-2]) is the new
    default; "linear_kl_stop" stays selectable (the walk runs used it).
  - `reward_floor`: each step's total clipped at 0 (legged_gym's
    only_positive_rewards, Playground) for stand and getup. stand12_calm's
    penalties made early steps negative, so falling (ending the episode)
    looked good: ~85% of its training episodes ended in falls.
  - Stand: `stillness_speed_gate` 0.5 m/s: pose, joint_speed and wobble
    only while the torso moves slower (Isaac Lab's Spot): calm alone,
    free to step when shoved. Shoves ≤ 1.0 m/s.
  - Getup, MuJoCo Playground's Go1 getup: `action_mode` "relative" (target
    = current angle + 0.5 × action; Playground found home-based worse);
    60% of episodes from the fallen bank, 40% standing; fixed 6 s
    episodes, no success ending (with positive standing rewards, ending
    at success threw away the reward after it: getup12 had that flaw);
    rewards `orientation` exp(−4(1 − up_z)), height, pose exp(−0.5|Δq|²)
    once upright within ~10° (`upright_gate` 0.985), `hold`
    exp(−0.5|a|²) once also at 95% height (`height_gate`); small energy
    and smoothness penalties. (Relative actions: action 0 = no holding
    force, so a robot doing nothing sags; the policy learns the offset.)
  - Hand-off (WalkTask.steady): tilt < 20° and ≥ 90% height for 0.5 s
    (was 25° / 80%), Lee et al.'s FSM scaled to this robot.
  Runs: `stand12_v3`, `getup12_v3` (50M each).
- **2026-09-25: Walking survey → push-robust walking recipe** (user: "do a
  similar research but for the walking … while walking and getting
  pushed"; details in Notes). Our walker already trains with more pushes
  than most references (they use ≤ 0.5–1 m/s every 10–15 s; Isaac Lab's
  Go1/A1 and walk-these-ways none). Physics limit (capture point): a shove
  of Δv needs the feet to move ~Δv·√(h/g) = 0.16 m per m/s, so 3 m/s needs
  ~0.5 m, more than a leg. Likely limits, and the changes (new defaults;
  runs without these settings load with them off):
  1. training shoves were per-axis U(±1): straight shoves never passed
     1 m/s, every test shove from 1.5 m/s up was new → `push_direction`
     "circle" (random direction, size U(0, max)), `push_interval` 2 s
     (1–3 s, mjlab), `push_max_spin` 0.5 rad/s, and a per-robot
     **curriculum** (`push_curriculum_max` 3.0 m/s, +0.25 per survived
     episode, −0.25 per fall, logged as `curriculum/push_max_speed`);
  2. episodes ended at 60° tilt / 50% height, so it never practised deep
     recoveries → falling = tilt > ~80° (up_z < 0.17) or torso < 30%
     height (get-up keeps 60° / 50% for its statistics);
  3. rewards punished recovery steps → `constraint_gate_speed_error` 0.5:
     gait, clearance, support and roll terms pause while the speed is off
     target by > 0.5 m/s (being off speed still costs tracking, so there's
     no point triggering it on purpose).
  Also: the get-up hand-over triggers on the *driving* policy's own fall
  measure (a walker allowed 80° isn't cut off at 60°), and `train_gpu.py
  --init <run>` fine-tunes from a checkpoint of our PPO. Deferred: a phase
  that adapts (policy-set clock frequency or ground-force feedback: the
  ORC paper roughly halved failures vs a fixed clock), privileged critic
  inputs, randomization/noise/latency (mostly for a real robot).
- **2026-09-25: Results with the surveyed recipes** (50M steps each):
  - `stand12_v3`: stillness (no shoves, 10 s) joint speed 0.001 rad/s,
    2.9° off home (motors alone 2.1°), wobble 0.1 °/s, 1.4 cm drift:
    as still as the bare motors. Shoves 100 / 100 / 93 / 43 / 31 / 18% at
    0.5 … 3.0 m/s (stand12_calm: 87 / 68 / 50 / 50 / 25 / 12%). Training
    falls dropped from > 80% to ~5% (reward floor).
  - `getup12_v3`: up within 10 s from its back 35/35, side 18/18,
    belly/feet 11/11 (median ~0.5 s); from standing 8/8 stay up. (Every
    earlier attempt: 0 from its back.)
  - The whole chain in the real-time Simulation (Behaviors with
    walk12_tidy + stand12_v3 + getup12_v3): knocked upside down, onto a
    side or its nose, in Stand and Walk mode: the get-up policy takes over
    at once, hands back after 1.3–2.0 s, and the robot stands still (Stand)
    or walks on (Walk, ~0.4 m/s).
  - Adaptive learning rate: both runs trained without stalling.
- **2026-09-25: Skill tests pick the best checkpoint** (user: "why does it
  say 40M is the best"). The dashboard's best = highest mean return over
  5 episodes, too noisy for these tasks: getup12_v3 40M (1000) beat 50M
  (979) by luck of easy starts, yet failed 7/12 upside-down starts with
  random legs that 50M passed; stand runs lost "best" to one unlucky
  shove. → `policy.skill_test()`, deterministic and task-specific:
  walk/stand: 32 shoves (1 and 2 m/s, 16 directions) after 3 s, survived
  if still up 3 s later; getup: 24 hard starts (16 from the fallen bank,
  8 upside down with random legs), passed if steady for 0.5 s within
  10 s. EvaluationInfo gets `skill` + `skill_test`, RunSummary `task`;
  a "Shove test"/"Get-up test" column; best = highest skill (ties: the
  newer). Evaluations without a skill count as not evaluated, so the
  Evaluate button fills them in.
- **2026-09-25: `walk12_robust` result** (walk12_tidy fine-tuned 60M steps
  with the push-robust recipe; curriculum level 1.0 → 2.63 m/s on
  average; training falls ~40% of episodes under shoves every 1–3 s):
  | shove (m/s) | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
  | walk12_tidy | 100% | 100% | 93% | 81% | 62% | 18% |
  | walk12_robust | 100% | 100% | 93% | 81% | 68% | 56% |
  (the same under tidy's stricter 60° fall rule, so not an artifact of
  its looser 80° one). Gait: 0.39 m/s, −1.2 °/s, a bit wider and lower
  at the back (roll 9° out on the left legs, rear hips 20° back vs
  6–8° / 8–14°): the braced stance robust walkers tend to take.
  Skill-test "best": getup12_v3 50M (100%), stand12_v3 47.6M (78%).
- **2026-09-25: Stand policy vs the viewer's pushes** (user: "not in the home
  configuration, one of its legs is a bit in the air, and it still falls
  when I push it with 60 N from the side"; also "a lot more stable when
  walking than when standing"). Measured on stand12_v3 47.6M:
  - at rest the front-left foot hovers 49 mm up: a three-legged stance
    (my stillness check averaged joint offsets and missed it);
  - a viewer push is a force at a point: 60 N on the upper side edge for
    0.1 s → 0.8 m/s sideways *and* 143 °/s of roll. The training kicks
    (velocity at the center of mass, no spin) were far easier. Viewer-
    style pushes at the upper edge, 8 directions: 40 N 8/8, 60 N 5/8,
    80 N 4/8, 100 N 3/8, 150 N 0/8. (walk12_robust trained with harder
    kicks + spin, hence "more stable walking".)
  Changes:
  - `push_kind` "force" (stand default): training pushes = viewer pushes
    (WalkTask.push_wrench / BatchedWalkTask.push_wrench, xfrc_applied on
    the torso for 0.1 s at a point on its side facing the pusher, 0–100%
    of its half-height up). Size still in m/s of impulse for the whole
    robot (1 m/s ≈ 72 N), so the curriculum (to 3 m/s ≈ 216 N) is the same.
  - `stance` reward (stand 1.0): share of feet down, while calm.
  - Skill test v2 (`runs.SKILL_TEST_VERSION`): walk/stand are pushed like
    the viewer (72 and 144 N, halfway up the side, 16 directions);
    evaluations with another version count as not evaluated.
  → `stand12_push`: stand12_v3 47.6M fine-tuned 60M steps.
- MuJoCo Warp occasionally prints "linesearch iterations limit reached"
  (~5 times per 50M-step run, i.e. per ~500M robot-physics-steps): some
  world's contact solve stopped at ls_iterations 50, slightly less
  converged. Harmless at that rate; not worth a slower solver.

## Current status

**2026-09-25: Working on 5b: a calm trot-walk trained on the GPU.**
- The user's gait feedback (both walkers ran) led to the trot-walk task, gait
  numbers in evaluations, RSL-RL as a reference trainer, and a gait clock
  (see Decisions, 2026-09-25).
- Two walkers, both trained on the GPU in ~16 min with our PPO, both
  0.4 m/s trot-walks without falls: `trot_ppo` (no clock: longer, calmer
  strides) and `trot_clock` (gait clock: 2 steps/s, higher lifts). The
  user prefers the clock (more consistent), and of the clock walkers the
  1.5 Hz one (`trot_clock_15`, ~27 cm strides): now the default gait.
- Milestone 5c (robustness): step 1 (mouse grab + push) done. Step 2: the
  12-motor robot, random shoves in training and robot switching are built
  and tested. `walk12_push` walked in circles (world-frame speed reward);
  `walk12_straight` went straight but splayed its legs (roll). Walk /
  Stand modes with the automatic switch are built and tested. Training now,
  `walk12_tidy` (+ roll penalty) is done: straight, mostly tidy, and far
  more shove-proof (see Decisions); waiting for the user's verdict on its
  look. After an open-source survey (user's request) the stand and getup
  recipes follow Isaac Lab's Spot and MuJoCo Playground's Go1 getup:
  `stand12_v3` stands as still as the bare motors, `getup12_v3` gets up
  from every fallen pose tested, and the Walk/Stand/get-up chain works.
  Waiting for the user to try them. A second survey (walking under
  pushes) led to `walk12_robust`: walk12_tidy fine-tuned with a shove
  curriculum; 3× the survival at 3 m/s shoves (56% vs 18%), same below.
  The dashboard now picks "best" checkpoints by task-specific skill tests.
  Waiting for the user to try walk12_robust + stand12_v3 + getup12_v3.
- `walk_cpu_fixed` (target_kl + lr decay + slip penalty):
  - 0 KL spikes (walk_10m: 114, max 50.5);
  - steady 1.0–1.28 m/s after 3M steps;
  - feet slide half as much (0.10–0.13 vs 0.22–0.28 m/s) with higher steps
    (3–5 cm);
  - best return 1421 at 8.5M.
- PyTorch switched to the CUDA 13.0 build (torch 2.14.0+cu130).
- GPU pipeline done and tested:
  - batched task parity;
  - GPU vs CPU physics parity on the real GPU;
  - fallen robots restart;
  - a tiny GPU training run plays in CPU MuJoCo.
- GPU runs v1–v5: see Decisions (GPU tuning). Transfer to CPU MuJoCo is fine;
  sample efficiency and stability are not yet at CPU level.
- Tests: 142 passing (GPU tests skip without CUDA), `tsc` clean.
- The dashboard shows a CPU/GPU pill; the throughput chart uses a log axis;
  errors show a red banner instead of blank charts.
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
  - **Measured 2026-09-24 on this PC (native Windows, no WSL):**
    `uv run --with mujoco-warp` (mujoco-warp 3.14.0, Warp 1.17, CUDA 12.9)
    sees the RTX 3090. `mjwarp-testspeed robots/quadruped.xml`:
    - 4096 worlds: **2.38M physics steps/s** (33 s one-time kernel compile);
    - 16384 worlds: **4.0M physics steps/s** (compile cached: 0.7 s);
    - all worlds converged; solver ~1.2–2.9 iterations.
    - For comparison, CPU training runs ~50k physics steps/s effective
      (5k env steps/s × 10 substeps).
- **Open-source survey (2026-09-25), how others train standing and getting up**
  (sources saved in the session scratchpad `research/`):
  - MuJoCo Playground Go1 getup (`go1/getup.py`): relative actions
    (q + 0.5a), Kp 35; 60% drops from 0.5 m with random orientation and
    joints, 40% home; 0.5 s settle; 6 s episodes, no early end; reward
    clip(Σ × dt, 0, ·): orientation 1, torso height 1 (capped), posture 1
    (gated upright ~6°), stand_still exp(−0.5|a|²) 1 (gated upright and at
    height), tiny penalties; 50M steps, ~9 min on an A100. Paper
    curriculum: power cutoff 400 W, then finetune with a joint-velocity cost.
  - Playground Go1 joystick: home + 0.5a; ends only upside down; standing
    at zero command: stand_still −1 × Σ|q − home| and pose exp; pushes
    are force pulses (off by default), Δv up to ~1 m/s.
  - legged_gym: pushes overwrite base xy velocity U(±1) every 15 s;
    stand_still exists but weight 0 in shipped configs; only positive
    rewards; A1 action scale 0.25.
  - Isaac Lab: additive velocity pushes ±0.5 m/s every 10–15 s;
    rel_standing_envs 2–10%; Spot gates its standing penalties on
    "command 0 and body speed < 0.5 m/s" (stand_still_scale 5).
  - mjlab (MuJoCo Warp + RSL-RL, our stack): pushes every 1–3 s (xy ±0.5,
    roll/pitch ±0.52); standing posture std 0.05 rad hip/thigh, 0.1 calf.
  - walk-these-ways: no pushes; its contact schedule isn't gated at zero
    command, so it marches in place (don't do that when standing).
  - Lee, Hwangbo, Hutter 2019 (ANYmal, no code): three policies
    (self-right, stand up, walk) + selector; bounded costs (unbounded ones
    make terminating attractive); curriculum on smoothness costs; FSM:
    recover if tilt > 35° or low, hand back after tilt < 20° for 0.5 s.
    >97% success in 100+ real falls; one multi-skill policy gave
    "frequent slippages and highly conservative postures".
  - Smith et al. 2022 (A1, code public), AFR 2024 (Go1), HoST (humanoid,
    upward assist force curriculum) for other get-up variants.
- **Walking-under-pushes survey (2026-09-25)** (sources in the session
  scratchpad `src/`):
  - legged_gym: pushes *overwrite* base xy velocity U(±1) every 15 s;
    friction U(0.5, 1.25); obs noise; termination on base contact;
    only_positive_rewards; ~147M steps with a terrain curriculum.
  - Isaac Lab: pushes *add* ±0.5 m/s every 10–15 s (Go1/Go2/A1: none);
    mass ±5 kg, COM ±5 cm; Spot's clock-free GaitReward (10) only while
    commanded or moving > 0.5 m/s; base orientation −3.
  - MuJoCo Playground Go1 joystick: kicks as force pulses (Δv ≲ 0.95 m/s),
    off in training by default, used for evaluation; friction 0.4–1.0,
    masses ±10%, COM ±5 cm; ends only upside down; 200M steps staged.
  - mjlab: pushes every 1–3 s, xy ±0.5, z ±0.4 m/s, roll/pitch ±0.52,
    yaw ±0.78 rad/s; ends at 70°; critic sees contacts/forces; ~1B steps.
  - Papers: Lee 2020 (per-leg phase with policy-set frequency offsets,
    f0 1.25 Hz for disturbance rejection), RMA (no pushes; terrain instead),
    DreamWaQ (largest survived push 0.51 → 1.12 m/s with its estimator),
    walk-these-ways ("low footswing and wide stance… robust to shoves"),
    PA-LOCO (force curriculum up to ±60 N), Shi 2024 (adversarial pushes on
    5% of envs; directions matter), ORC 2024 (fixed clock + phase reward
    beat gait-free under pushes; ground-force-fed phase halved failures).
- **Jumping** (user asked, 2026-09-25, after the stand policy works): a
  third behavior next to Walk and Stand, triggered by a key (J) in the
  viewer: crouch, jump, land on its feet, return to the mode it was in.
  Reward height/air time and a steady upright landing. Rough estimate from
  the motors (±10 N·m, 0.32 m legs, 7.2 kg): 10–20 cm clearance. Ask the
  user which kind (straight up / forward / on command) when starting.
