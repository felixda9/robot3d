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
- **Protocol (WebSocket, FastAPI):**
  - On connect, the backend sends the static scene once: for each geom, its type,
    size, color (rgba), and parent body.
  - Then it streams each frame's geom positions and rotation matrices
    (MuJoCo's `data.geom_xpos` and `data.geom_xmat`) at ~60 fps.
  - The frontend sends commands back (play/pause/reset, later motor targets).
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

- Windows 10, NVIDIA RTX 3090 (24 GB), Python 3.12, managed with **uv**.
- Commands:
  - `uv sync`: create `.venv` and install everything
  - `uv run pytest`: run tests (headless physics checks)
  - `uv run scripts/view_mujoco.py`: MuJoCo's built-in viewer

## Layout

```
robots/            MJCF robot files (data, not code)
src/robot3d/       Python package (sim helpers; later server, envs, training)
scripts/           Entry-point scripts (viewer, later training/eval)
tests/             pytest, headless physics sanity checks
```

## Milestones

- [x] **1. Foundation:** Python project (uv), MuJoCo installed, a simple
  quadruped in MJCF (torso, 4 legs, 2 hinge joints per leg, position-controlled
  motors) on a ground plane. A script that shows it in MuJoCo's built-in viewer.
  *Success = the robot drops onto the floor and settles without jittering or
  exploding.*
- [ ] **2. Web viewer:** FastAPI WebSocket server runs the simulation in real
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
- **2026-09-24: Real-time loop (reused in M2):** the script steps physics
  itself, anchored to a (wall time, sim time) pair; each 60 fps frame steps
  until sim time catches up, capped at 50 steps per frame, then resyncs. The
  MuJoCo *passive* viewer only draws. It applies its own Backspace/Reset
  inside `viewer.sync()` (reset to qpos0), so the script detects
  time-going-backwards after `sync()` and resets to the keyframe instead.

## Current status

**Milestone 1 done (2026-09-24), waiting for the user to test.**
- `robots/quadruped.xml`: the robot, commented for learning.
- `src/robot3d/robots.py`: `load_model(name)`, `reset_to_keyframe(...)`.
- `scripts/view_mujoco.py`: passive-viewer script with real-time loop;
  Space = pause, Backspace = drop again; prints status lines.
- `tests/test_quadruped.py`: structure check, plus drop-and-settle from `home`
  and from straight legs. Checks: no warnings/NaN, upright, only feet touching,
  final-second max joint speed < 0.01, feet drift < 2 mm.
- Measured: lands from 0.40 m, settles at 0.261 m torso height within ~1.5 s,
  ~1° pitch. Real-time pacing verified (4.0 s sim over 4.0 s wall).

## Notes for later milestones

- M4: Stable-Baselines3 PPO with small MLP policies usually trains *faster on CPU*
  than GPU; the bottleneck is stepping many envs in parallel. Benchmark both.
- M4: On Windows, `SubprocVecEnv` uses the `spawn` start method, so training
  scripts need an `if __name__ == "__main__":` guard.
- M4: To use CUDA, PyTorch has to come from the CUDA wheel index (configure it
  in `pyproject.toml` under `[tool.uv.sources]`).
