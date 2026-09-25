# robot3d: research notes

What other projects and papers do, gathered by research agents from primary sources
(moved out of CLAUDE.md on 2026-09-25).


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
- **Open-source survey (2026-09-25), how others train standing and getting up:**
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
- **Walking-under-pushes survey (2026-09-25):**
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
- **Walking on any ground (2026-09-25)**. The user asked whether to train
  on each obstacle, or whether something more general lets a robot go
  anywhere. (From known papers, not a fresh web survey.)
  - Nobody trains one skill per obstacle: **one policy is trained on a
    broad random mix of ground**, with a curriculum that makes it harder
    as the policy improves. The generalization comes from the diversity,
    randomized physics, and feeling the ground.
  - legged_gym (Rudin et al. 2021, "Learning to walk in minutes"): 4096
    ANYmals on a 10 x 20 grid of tiles: smooth and rough slopes, stairs up
    and down, discrete obstacles. Rows are difficulty levels; a robot moves
    up after walking more than half its tile, down after falling short. A
    height scan of 187 points. ~20 min on one GPU.
  - Lee et al. 2020 (ANYmal, Science Robotics): a **blind** policy
    (proprioception history, a teacher with privileged terrain info
    distilled into it). Trained on procedural hills, steps and stairs with
    an adaptive curriculum. It then walked on mud, snow, rubble, vegetation
    and running water it had never seen.
  - RMA (Kumar et al. 2021): trained only on fractal bumpy ground with
    randomized mass, friction and motor strength, plus an adaptation
    module (it infers the ground and physics from recent history). It
    worked on sand, mud, grass, hiking trails and stairs.
  - Miki et al. 2022 (ANYmal hiking in the Alps): a height map **and**
    proprioception, trained with deliberately corrupted maps, so the
    robot learns when to trust what it sees and when to rely on what it
    feels. The current standard.
  - What we took from them (terrain.py): one mixed park with a per-robot
    level curriculum; every tile random within its level; a noisy height
    map (2 cm); per-robot random friction (0.4–1.25), payload (−0.5..+1.5
    kg) and motor strength (±15%) on top of the pushes; a held-out test
    course of shapes the park never has, to measure generalization.
    Possible next steps: height-map dropout or corruption (Miki), and a
    history input for "feeling" the ground (RMA, Lee).
