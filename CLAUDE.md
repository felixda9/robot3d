# robot3d — Robot simulation & training platform

A system to **design legged robots**, **watch them in 3D in the browser**,
**control them**, and **train them with reinforcement learning** to walk, stand,
get up, jump and follow steering commands. MuJoCo is the physics engine;
everything around it is built here.

Details that used to live here: `docs/decisions.md` (the dated log: every
decision, experiment and result, with the why) and `docs/research.md` (what
other projects do). Keep this file short; put long stories there.

## Architecture

```
┌──────────────────────────── Python backend ─────────────────────────────┐
│ robots/*.xml (MJCF) ─► MuJoCo sim ◄─ tasks (walk.py) ◄─ PPO (CPU SB3 / GPU) │
│                            │          GPU: MuJoCo Warp, 4096 robots at once  │
│                  FastAPI server: WebSocket /ws + HTTP /api                  │
└────────────────────────────┼───────────────────────────────────────────────┘
          scene once, ~60 fps poses ▼   ▲ commands (play/pause, motors, grab/push,
┌────────────────────────────┴──── Browser (three.js) ─────── policies, steering)
│ Viewer + controller + training dashboard. NEVER simulates physics.         │
└────────────────────────────────────────────────────────────────────────────┘
```

- The backend owns everything physical: simulation, tasks, training.
- **`web/src/protocol.ts` is the single source of truth** for every WebSocket
  message and HTTP API type; `src/robot3d/protocol.py` mirrors it (Pydantic,
  `extra="forbid"`), and `tests/test_protocol.py` fails if they differ. Change
  both, then run `uv run pytest`.
- One shared simulation per server (every browser sees the same robot). The sim
  runs on its own thread at 60 fps; commands arrive through a queue.
- Dev: Vite (5173) serves the page and proxies `/ws` and `/api` to the backend
  (8000). Production: `npm run build` → `web/dist/`, served by the backend.
- Robots are data (MJCF in `robots/`), **primitive shapes only**.
- Tasks stay Gymnasium-compatible (`robot3d/Walk-v0`); the same task code
  (`walk.py`) runs in training (CPU and batched on the GPU) and in the viewer.

## Conventions

- SI units, **radians**; world **+x forward, +y left, +z up**.
- Robot MJCF conventions the code relies on: body 1 = the free root (torso or
  pelvis); every motor a `<position>` actuator named like its joint; a `home`
  keyframe = the standing pose (more keyframes = pose presets in the UI); feet
  are geoms named `*_foot`; quadruped legs `FL FR RL RR`, humanoid sides `L R`;
  sideways joints named `*_roll`.
- Joint signs (right-hand rule): pitch about +y, positive = a hanging segment
  swings backward; roll about +x, positive = toward the robot's left.
- Explain physics/RL concepts briefly in code comments (the user is learning).

## Environment and commands

Windows 10, RTX 3090, i7-11700K; Python 3.12 via **uv**; PyTorch 2.14 CUDA 13.0
build; MuJoCo 3.14 + mujoco-warp 3.14; SB3 2.9; rsl-rl-lib 5.5 (reference
only); Node 24, Vite 8, TypeScript 7, three.js 0.186.

- `uv sync`; `cd web; npm install` — install
- `uv run pytest` — all tests (GPU tests skip without CUDA; the protocol test
  needs `web/node_modules`)
- `uv run scripts/serve.py` — server on :8000 (`--robot`, `--policy runs/<name>`)
- `cd web; npm run dev` — page on http://localhost:5173 (Simulator + Training tabs)
- `uv run scripts/train_gpu.py` — GPU training: `--robot`, `--task
  walk|steer|stand|getup|jump`, `--steps`, `--name`, `--init <run>` (fine-tune,
  also onto appended inputs), `--gait-hz`, `--trainer ppo|rsl`
- `uv run scripts/train.py` — CPU training (SB3; the original walker)
- `uv run scripts/evaluate.py runs/<name> [--all]` — headless evaluation + skill test
- `uv run scripts/view_mujoco.py` — MuJoCo's own viewer
- `cd web; npm run build` — type-check + production build

## Layout

```
robots/            quadruped.xml (8 motors), quadruped12.xml (12, 20 N·m), humanoid.xml (21)
src/robot3d/
  walk.py          WalkConfig (all task settings; walk/steer/stand/getup/jump presets,
                   ROBOT_SETTINGS) + WalkTask (observation, action, reward, pushes)
  envs.py          WalkEnv (Gymnasium)          gpu/task.py  BatchedWalkTask (walk.py in torch)
  gpu/env.py       GpuWalkEnv (MuJoCo Warp)     gpu/ppo.py   our PPO + TorchPolicy
  gpu/rsl.py       RSL-RL reference trainer     gpu/common.py  run folders, logging
  policy.py        PolicyController, Behaviors (walk/stand/getup/jump switch),
                   evaluate(), skill_test()
  simulation.py    real-time sim (glides, grab/push forces)   scene.py  scene/frame messages
  server.py        FastAPI app                  runs.py  run folders, evaluations (no torch)
  training.py      CPU (SB3) training           protocol.py  Pydantic mirror
scripts/           serve, train, train_gpu, evaluate, view_mujoco
tests/             pytest (conftest: tiny real training runs per task)
web/src/           protocol.ts, main.ts, viewer.ts, geoms.ts, motors.ts, interaction.ts
                   (grab/push), steering.ts (keyboard/gamepad), dashboard/, api.ts
docs/              decisions.md (history), research.md (surveys)
runs/<name>/       training output (gitignored): run.json, tb/, checkpoints/
```

## How we work

- **One milestone (or step) at a time**; at the end explain how to test it,
  then **wait for the user**. **Ask** about significant or ambiguous choices.
- Commit after each working step and push to `origin`
  (https://github.com/felixda9/robot3d.git).
- **Restart the backend** (`uv run scripts/serve.py`) whenever the protocol or
  `WalkConfig` fields change, right away: an old server can't read new runs
  ("uses task settings this code doesn't know") and a new page can't talk to
  it. A restart clears the loaded policies; tell the user.
- Don't tell the user a policy is ready before testing it the way they will
  use it (the viewer's pushes, jumping mid-walk, steering keys...).
- Keep this file current and short; record the story in `docs/decisions.md`.

## Milestones

- [x] 1 Foundation · [x] 2 Web viewer · [x] 3 Manual control · [x] 4 First
  training (CPU PPO) · [x] 5 Training dashboard
- [x] 5b GPU training: MuJoCo Warp + our PPO, ~50k steps/s, parity-tested
  against CPU MuJoCo; GPU policies play in the CPU viewer.
- [x] 5c Robustness: mouse grab/push; 12-motor robot; shove-robust walking;
  Walk/Stand modes with automatic get-up; jump on J. (Higher jump with the
  20 N·m motors: `jump12_m20` training.)
- [ ] 6 Robot designer — **postponed by the user**.
- [ ] 7 Environments & commands. Part 1, **steering** (keyboard/gamepad): built,
  `steer12_m20` training. Part 2, **terrain**: next.
- [ ] 8 Humanoid (child-size, 21 motors): robot built, `humanoid_walk` training.

## The robots

| robot | motors | mass | notes |
|---|---|---|---|
| quadruped | 8 (hip, knee per leg) | 6.6 kg | the original; can't step sideways or right itself |
| quadruped12 | 12 (+ hip roll) | 7.2 kg | 20 N·m motors (were 10: jump limit 16 → 37 cm); hip pitch −2.0..2.8 so it can flip itself over; legs collide with the torso |
| humanoid | 21 (legs 6 each, waist 3, arms 3 each) | 19 kg | ~1.0 m, box feet, lower legs collide with each other; stands on its own |

## Tasks and best policies (quadruped12 unless noted)

All tasks share `WalkTask`; `WalkConfig` presets set the differences. Runs load
their saved settings (`WalkConfig.from_run`), so old runs replay as trained.

| task | preset | best run | how good |
|---|---|---|---|
| walk | `WalkConfig()` | `walk12_robust` | 0.39 m/s trot-walk, 1.5 Hz gait clock; survives 3 m/s shoves 87% (20 N·m) |
| steer | `.steer()` | `steer12_m20` (training) | commands: forward −0.3..0.6, sideways ±0.3 m/s, turn ±1 rad/s; zero = stand |
| stand | `.stand()` | `stand12_push` | as still as the bare motors in the home pose, all feet down; viewer pushes 8/8 at 80 N |
| getup | `.getup()` | `getup12_v3` | up from every tested fallen pose (upside down with legs anywhere), ~1.1 s |
| jump | `.jump()` | `jump12` (10 N·m); `jump12_m20` training | 8/8 jumps, torso +14 cm; new one: higher, and mid-walk |
| walk (humanoid) | `.for_robot("humanoid")` | `humanoid_walk` (training) | — |

Key recipe facts (details and numbers in `docs/decisions.md`):
- GPU PPO: 4096 robots × 24 steps, adaptive learning rate (RSL-RL's), actor and
  critic gradients clipped separately, `--init` fine-tuning.
- Walk: heading-frame speed tracking + turn reward, 1.5 Hz trot clock
  (user's choice), roll penalty, shoves every 1–3 s with a per-robot
  curriculum up to 3 m/s, falls at 80° tilt, gait rules paused while knocked
  off speed.
- Stand: stillness terms only below 0.5 m/s, viewer-style force pushes,
  stance reward, reward floor at 0. Getup: MuJoCo Playground's recipe
  (relative actions, 60% fallen starts, fixed 6 s episodes).
- Viewer behaviors (`policy.Behaviors`): Walk/Stand modes; the get-up policy
  takes over after a fall until level (<20°) and high (≥90%) for 0.5 s; J
  starts a jump; steering keys drive a steerable walker.
- Dashboard "best" = highest **skill test** (v3): walk/stand 32 viewer-style
  pushes (72/144 N); getup 24 hard fallen starts; jump 8 jumps (+ height).

## Lessons (each cost a training run)

- Reward only what the policy can observe: world-frame speed → it walked in circles.
- Check the loopholes: "diagonal feet in sync" paid for all four down (it
  scooted); "stand still" and "pose" paid while lying down (it lay still).
- Clip each step's reward at 0 when there are penalties, or falling looks good.
- Don't end episodes on success when staying up keeps paying.
- A KL early stop stalls once exploration noise is small: use an adaptive rate.
- Test with the user's inputs: my center-of-mass shove test missed that the
  viewer's side pushes also tip the robot; my pose average missed a raised foot.
- A policy only knows its training starts (the jump failed mid-walk).
- Check physical limits before tuning rewards (the jump was motor-limited).
- Pick checkpoints by a skill test, not by a 5-episode mean return.

## Current status

2026-09-25:
- **Training on the GPU:**
  - `jump12_m20` (higher jump, 20 N·m, running starts)
  - `steer12_m20` (steering)
  - `humanoid_walk` (100M steps)
- **Next:** milestone 7 part 2, terrain.
- **Tests:** 156 passing, `tsc` clean.
