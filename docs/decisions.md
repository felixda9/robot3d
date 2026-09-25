# robot3d: decisions log

The full, dated record of design decisions, experiments and results, moved out of
CLAUDE.md on 2026-09-25 to keep that file short. Newest entries at the bottom.
CLAUDE.md keeps the current state and the lessons; this file keeps the why.

## Log

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
  hips differ). **Shove survival** (walk 3 s,
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
- **2026-09-25: `stand12_push` result** (stand12_v3 47.6M fine-tuned 60M
  steps with force pushes + stance reward): all four feet on the floor at
  rest (was one 49 mm up); a 60 N upper-edge side push now rolls it at
  35 °/s (was 143: it braces); viewer pushes at the upper edge, 8
  directions: 40 N 8/8, 60 N 8/8, 80 N 7/8, 100 N 7/8, 150 N 2/8 (was
  8/5/4/3/0); stillness 2.4° off home, joints still, 1.2 cm drift.
- **2026-09-25: Jump task + steering** (user: "do jump training and start
  building the next milestones … a new human like robot"; choices: jump on
  command, steering first, humanoid legs + arms, child-size):
  - Jump (`WalkConfig.jump()`, `--task jump`): 3 s episodes from standing;
    the clock input runs once over the jump; `jump` = 50 × (torso height −
    standing height) while all feet are up before the first landing
    ("landed": touchdown after ≥ 3 airborne steps); `settle` after landing
    (feet down × standing pose); `rejump` −1 per airborne step after
    landing; orientation 1.0; ±1 rad actions; falls end it (60°/50%).
    Behaviors: jump slot; `jump()` drives until landed + steady 0.5 s or
    3 s; JumpCommand, StatusMessage jump_policy / jumping; J key + button.
    `jump12` training: by 17M steps jump/settle rising, re-hops fading,
    0 falls.
  - Steering (`WalkConfig.steer()`, `--task steer`, task "walk"):
    commands (forward −0.3..0.6, sideways ±0.3 m/s, turn ±1 rad/s) observed
    (+3 inputs) and tracked (tracking and turn follow the command);
    re-drawn every 5 s (forward 90%, sideways 30%, turn 50%, 10% all zero);
    at zero: gait/clearance off, `still` (feet down × home pose, σ 0.1).
    `--init` widens first layers for appended inputs. SetCommandCommand,
    StatusMessage steerable / command_limits; viewer steering.ts (WASD/
    arrows/QE, gamepad sticks). `steer12` training (from walk12_robust, 80M).
- **2026-09-25: The humanoid (M8)**, `robots/humanoid.xml`: ~1.0 m (head
  top 0.999 m), 19.0 kg, 21 motors (I had quoted "~21" for 6 + 6 + 3 + 3 +
  1 = 19; the waist got roll and pitch too, as on Unitree G1):
  - legs: hip yaw/roll/pitch, knee, ankle pitch/roll (thigh and shin
    0.22 m, box feet 18 × 9 cm); waist yaw/roll/pitch; arms: shoulder
    pitch/roll, elbow. Pelvis 3 kg (the free root, body 1), torso 6 kg,
    head 1 kg with a dark visor marking the face.
  - motors: hips/knees kp 150 ±80 N·m, hip yaw/roll kp 100 ±50, ankles
    kp 80 ±40, waist kp 100 ±40, arms kp 30 ±15.
  - collisions: parts hit the floor, not each other, except lower legs and
    feet, which hit each other (feet can't pass through each other).
  - home: hip −0.25, knee 0.5, ankle −0.25 (feet flat), arms slightly out
    and bent; presets crouch, arms_out. It stands on its own (pelvis
    0.53 m, only the feet touching); tests/test_humanoid.py.
  - Task generalized for bipeds: soles of box feet (half-height), left and
    right half a cycle apart, "support" = at least half the feet down,
    pushes on the root body's box, higher drops for the fallen bank.
  - `WalkConfig.for_robot` + `ROBOT_SETTINGS`: the humanoid's walk has no
    roll penalty (hip/ankle roll balance a biped), falls at 60° tilt or
    pelvis < 60% height, gentler shoves (0.5 → 1.5 m/s). train_gpu.py
    applies it. `humanoid_walk` training (100M steps).
- **2026-09-25: Jump skill test**: 8 jumps from standing, passed if landed
  (≥ 60 ms airborne) and steady within 3 s; reports the median height.
- **2026-09-25: `jump12` result** (50M steps): jump test 8/8 landed and
  steady, median torso rise 14 cm (11 cm at 15M). In the viewer's chain
  (stand12_push + getup12_v3 + jump12): three jumps in a row, 12–13 cm,
  each done after ~0.9 s, back to standing level at 0.26 m, no falls.
- **2026-09-25: `steer12` at 15M steps** (the user tried it: "walks forward
  even when I'm not pressing W"): partly learned. Command → measured:
  stand still → 0.17 m/s forward; forward 0.5 → 0.39; back 0.3 → +0.05;
  left 0.3 → 0.05 sideways (+0.19 forward); turn left 0.8 → 0.54 rad/s;
  forward + turn right → 0.37, −0.42. Turning and forward come first;
  stopping/back/sideways not yet (the walk12_robust start walks forward
  regardless). Check again at 40M and 80M.
- **2026-09-25: Higher jumps → 20 N·m motors on quadruped12** (user: "I want
  the jump to be higher, right now it's pretty bugged and not high"):
  - "bugged": jump12 from walking did nothing (torso +0–1 cm, 0–40 ms in
    the air, then 3 s wasted): it had only trained from standing. From
    standing: +13 cm, feet up to 12–14 cm, 280–300 ms airborne, ≤ 9° tilt.
  - Height was near the motors' limit: a scripted crouch-and-extend
    reached +16 cm with 10 N·m (policy: 13–14 cm); 15 N·m +25 cm; 20 N·m
    +37–39 cm (damping kv made ~2 cm of difference).
  - User chose stronger motors: quadruped12 now 20 N·m, kv 1.0 (0.5–0.7
    left it swaying after the drop test). Existing policies with the new
    motors (same kp, so unsaturated torques are unchanged): stand12_push
    viewer pushes 8/8/8/7/2 of 8 at 40/60/80/100/150 N; getup12_v3 48/48
    upside-down cases, ~1.1 s; walk12_robust shoves 100/100/100/100/93/87%
    at 0.5…3 m/s (was …/93/81/68/56): no retraining needed. Older runs of
    quadruped12 now replay with the stronger motors.
  - Jump: `start_speed_max` 0.5 m/s + joint noise ±0.3 rad (running
    starts); `jump12_m20` fine-tunes jump12 on the new motors (50M), and
    `steer12_m20` restarts steering (steer12 was on the old motors).
- MuJoCo Warp occasionally prints "linesearch iterations limit reached"
  (~5 times per 50M-step run, i.e. per ~500M robot-physics-steps): some
  world's contact solve stopped at ls_iterations 50, slightly less
  converged. Harmless at that rate; not worth a slower solver.

## Status as of 2026-09-25 (before the move)

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
- Tests: 156 passing (GPU tests skip without CUDA), `tsc` clean.
- The dashboard shows a CPU/GPU pill; the throughput chart uses a log axis;
  errors show a red banner instead of blank charts.
- Git remote: `origin` = https://github.com/felixda9/robot3d.git. Push after
  each milestone commit.


## 2026-09-25: Terrain (milestone 7, part 2)

- User's choices: all four terrain types (rough ground, slopes, stairs,
  obstacles); a height map for perception; quadruped12 first. Asked
  mid-build whether to train per obstacle: no. One policy on a random mix,
  as legged_gym, Lee 2020, RMA and Miki 2022 do (docs/research.md). The plan
  gained randomized tiles, randomized physics, a noisy height map and a
  held-out test course.
- **Everything is boxes** (terrain.py). A height field was as slow as boxes
  on the GPU and gives one contact per pair. Boxes can be turned (yaw) and
  tilted (pitch/roll), for ramps and tilted rubble slabs.
- **Speed on the GPU (4096 quadruped12s, shared GPU):**
  - flat: 2.34M physics steps/s;
  - the whole 736-box park in every world: 0.38M. The broadphase checks
    every robot geom against every box (~13k pairs per robot); the
    sweep-and-prune broadphases were slower still (0.11–0.12M);
  - **per-world tiles: 1.66M (71% of flat).** MuJoCo Warp lets model
    fields differ per world (`worldid % rows`), and static geoms get their
    pose once at `put_data` and never again. So the GPU model has 25
    placeholder boxes, and each world writes its own tile into them
    (`d.geom_xpos`/`geom_xmat`, plus per-world `m.geom_size`/`geom_aabb`/
    `geom_rbound`). `reset_data` doesn't touch them.
- **Tiles** (3 m, ≤ 25 boxes, centered at 0,0):
  - rough: 5 x 5 slabs, each at its own height and tilt (up to 8 cm, 8°);
  - slope: a ridge along x with ramps to ±y (up to 25°);
  - stairs: a pyramid of nested boxes (2–12 cm steps, 25–35 cm treads);
  - obstacles: 10 turned blocks (up to 12 cm).
  - Slopes and stairs have a 0.25 m floor border. Robots start on top
    (walk down) or just outside the tile at the foot (walk up); starting
    on the border straddled the first step, and one robot tipped over.
  - Difficulty = (level + U(0,1)) / 10: every level is a band, never one
    fixed value.
- **GPU pool:** 10 levels x 4 types x 5 variants = 200 tiles. Each robot
  keeps a terrain type (like legged_gym's columns). At every reset it gets a
  random tile of its type and level.
  - Curriculum: up a level after walking 1.2 m from its start (which ends
    the episode like a time-out, so PPO bootstraps); down after a fall, or
    a time-out having walked less than half of what the commands asked
    (capped at 1 m).
  - Past the top level: a random level, so easy ones aren't forgotten.
    Robots start at levels 0–3.
- **Heights from the ground below:** a 2 cm height grid, rasterized once
  (matches MuJoCo's rays: median 0 mm, 99% within 3 mm away from edges).
  - Torso height (obs[0]), falls, steady and the rewards use bilinear
    lookups.
  - Feet use the highest of the 4 grid points around them (a foot on a
    stair's edge is on the step).
  - On flat ground everything is 0, so old runs are unchanged (tests pass).
- **Height map:** 13 x 7 points, 8 cm apart, from 32 cm behind to 64 cm
  ahead and ±24 cm sideways, in the heading frame. Value = torso height
  above that point − standing height (0 on flat ground), clipped at ±1 m,
  plus 2 cm noise in training. Appended at the end of the observation, so
  `--init` can widen a flat walker onto it.
- **Fine-tuning onto new inputs:** the new inputs' normalizer statistics
  start from the first batch. The old normalizer has counted ~10^11 samples,
  so otherwise they'd stay at mean 0, variance 1 forever.
- **Randomized physics** (`on_terrain()`), per robot and episode:
  friction 0.4–1.25 (every geom, since a contact uses the larger value),
  torso payload −0.5..+1.5 kg, motor kp/kv ±15%. On the GPU these are
  per-world model rows; on the CPU the env sets the model at reset.
  Evaluation and skill tests use nominal physics and an exact map.
- **Bug found by the CPU/GPU parity test:** the settled standing state was
  made at the origin of the terrain model. On a single-tile layout that is
  on top of the stairs, so the standing height came out 22 cm high. States
  made at the origin now use `WalkTask.flat_model` (terrain boxes don't
  collide).
- **Viewer:** a Ground selector (Flat / Park / Course, key G) rebuilds the
  simulation on a worker thread. Loaded policies stay (retargeted: same
  network, new task). Loading a terrain-trained policy while on flat floor
  switches to the park. The park starts 1.5 m ahead of the origin, levels
  along +x, so walking forward means harder ground.
- **Skill test for terrain runs:** the test course, 25 m, six sections
  with 1.5 m of floor before each: turned rubble, 12°/20° ramps with a
  plateau, a 9 cm step, 6 cm narrow-tread (22 cm) stairs, a stepping field,
  a 10° cross-slope.
  - Each section is tried twice on its own: start 1 m before it; a pilot
    steers at 0.4 m/s, turning and stepping sideways back to the center
    line. Passed = 0.5 m past its end upright, within twice the time that
    takes. Skill = the share passed; the description lists what it missed.
  - The first version scored the share of the whole course walked. Both
    walkers stopped at the 9 cm step, so nothing after it was ever tested.
  - Cross-slope: a sideways-tilted face can't meet a level ramp edge
    everywhere, so off the center line the junctions have lips. A rear foot
    caught on a 3.5 cm lip stalled terrain12. Its test starts on the slope.
    With a heading-only pilot the robot also drifted 0.6 m downhill; the
    pilot now corrects sideways too.
  - Results so far: steer12_m20 (flat, no map) 33%: ramps and cross-slope
    only. terrain12 at 25M: 67%: also rubble and the stepping field; not yet
    the 9 cm step or the narrow stairs (its curriculum is around level 2–3,
    with 3–5 cm steps).
- jump12_m20 (20 N·m, running starts, 50M): from standing +16–17 cm, 0.3 s
  airborne, feet up to 21–22 cm, landing tilt 24–30°; while walking +12–21
  cm, sometimes with a small second hop.
- steer12_m20 at 62.5M: forward 0.43 (asked 0.5), back 0.17 (0.3),
  sideways 0.18 (0.3), turn 0.84 (0.8) rad/s, stands still at zero.
- humanoid_walk (100M): no falls in 5 x 20 s, but only 0.16 m/s (target
  0.4); push test 53% at 190/380 N.
- **terrain12 result** (steer12_m20 fine-tuned on the park, 150M steps,
  ~65 min):
  - Level curriculum: rose from 1.5 to ~6 by 40M, then flat; falls ~5%;
    shoves at the 3 m/s maximum.
  - Test course: 83% at 150M (5/6; only the narrow 6 cm stairs missed); the
    flat steer12_m20 gets 33%.
  - Going down works at every level. Climbing (6 tries per level): stairs
    up to 7–8 cm steps (level 5: 4/6, level 6+: 0/6); slopes up to 20°
    (level 7: 6/6, level 8+: 0/6). That's why the mean level stalled at 6.
  - Stairs: **the back feet get caught** (the user saw it in the viewer
    too). Front feet lift 11–16 cm above the ground below, back feet only
    6–8 cm. On 9–10 cm steps the back toes press against the riser for
    3–5 s, then the episode times out.
  - Steep ramps: nothing gets caught. Front feet on the ramp, back feet on
    the floor, and it creeps up at 2 cm/s: a skill it rarely practiced,
    since failed climbs demote it.
- **Stumble penalty** (`stumble_weight`, on in `on_terrain()`: 0.5 per
  foot, legged_gym's feet_stumble): a foot touching the static world with a
  contact normal more than 60° from vertical. On the GPU it comes from MuJoCo
  Warp's contact list (geom, frame, worldid, nacon), scattered to (world,
  foot) without a CPU sync. `terrain12_stumble`: terrain12 + this, 100M.
- **terrain12_stumble result** (terrain12 + stumble 0.5 per foot, 100M):
  the back feet stopped catching (highest lift 9–14 cm, was 6–8; riser
  contacts 2–30 control steps, was 124–273). But climbing got **worse**:
  stairs 7–8 cm 1/6 (was 4/6), still 0/6 from 8 cm; slopes 17.5–20° 2/6
  (was 6/6); test course 4/6 (lost the 9 cm step). It now hesitates in
  front of steps, its front feet tapping the riser: the penalty taught it
  that touching an edge costs, so it stopped approaching them.
  **terrain12 stays the best terrain walker.**
