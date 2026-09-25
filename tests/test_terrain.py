"""Terrain (M7): tiles, layouts, heights above the ground, the height map,
and terrain + randomized physics on the GPU (parity with the CPU env)."""

import dataclasses

import mujoco
import numpy as np
import pytest
import torch

from robot3d.envs import WalkEnv
from robot3d.robots import load_model
from robot3d.terrain import LEAVE_DISTANCE, LEVELS, MAX_TILE_BOXES, TYPES, Terrain, make_tile, park_tiles
from robot3d.walk import WalkConfig

TERRAIN_STEER = dataclasses.replace(WalkConfig.steer().on_terrain(), push_interval=0.0)
# Exact inputs, for comparisons: no noise in the height map, nominal physics.
EXACT = dataclasses.replace(TERRAIN_STEER, height_map_noise=0.0, friction_min=1.0, friction_max=1.0,
                            added_mass_min=0.0, added_mass_max=0.0, motor_strength_min=1.0, motor_strength_max=1.0)


def ray_height(model, data, x, y):
    """Height of the first world geom straight below (x, y), by MuJoCo's own ray cast."""
    world_only = np.array([1, 0, 0, 0, 0, 0], np.uint8)  # geom group 0
    dist = mujoco.mj_ray(model, data, np.array([x, y, 3.0]), np.array([0.0, 0.0, -1.0]), world_only, 1, -1,
                         np.zeros(1, np.int32))
    return 3.0 - dist if dist >= 0 else 0.0


@pytest.mark.parametrize("terrain", [Terrain.park(0), Terrain.course(0), Terrain.single(make_tile(
    "rough", LEVELS - 1, np.random.default_rng(0)))], ids=["park", "course", "rough tile"])
def test_height_grid_matches_mujoco(terrain):
    model = load_model("quadruped12", terrain)
    data = mujoco.MjData(model)
    data.qpos[2] = -5.0  # the robot out of the rays' way
    mujoco.mj_forward(model, data)
    heights, x0, y0 = terrain.heights
    rng = np.random.default_rng(1)
    errors = []
    while len(errors) < 500:
        x = x0 + rng.uniform(0.5, (heights.shape[0] - 1) * 0.02 - 0.5)
        y = y0 + rng.uniform(0.5, (heights.shape[1] - 1) * 0.02 - 0.5)
        around = [terrain.ground_height(x + dx, y + dy) for dx in (-0.04, 0, 0.04) for dy in (-0.04, 0, 0.04)]
        if np.ptp(around) > 0.03:
            continue  # at an edge the 2 cm grid can't be exact
        errors.append(abs(ray_height(model, data, x, y) - float(terrain.ground_height(x, y))))
    assert np.percentile(errors, 99) < 0.006


def test_tiles_fit_the_gpu_slots_and_start_flat():
    tiles = park_tiles(seed=0, variants=2)
    assert len(tiles) == LEVELS * len(TYPES) * 2
    assert max(len(t.boxes) for t in tiles) <= MAX_TILE_BOXES
    for tile in tiles[:len(TYPES) * 2]:  # level 0: small bumps and steps (stairs: 3 steps of 2-3 cm)
        assert tile.heights.max() < 0.12
    # the hardest stairs are much taller than the easiest
    stairs = [t for t in tiles if t.kind == "stairs"]
    assert stairs[-1].heights.max() > 3 * stairs[0].heights.max()


def test_park_starts_ahead_of_the_origin():
    park = Terrain.park(0)
    assert park.ground_height(0.0, 0.0) == 0.0  # the viewer's robot starts on flat floor
    assert len(park.tiles) == LEVELS * len(TYPES)
    assert Terrain.course(0).length > 15.0


def test_height_map_sees_the_stairs():
    """Standing on a stairs landing, the ground drops away ahead (positive
    values), and on flat floor the height map is ~0 everywhere."""
    tile = make_tile("stairs", 5, np.random.default_rng(0))
    env = WalkEnv("quadruped12", EXACT, terrain=Terrain.single(tile))
    env.task.reset_state(env.data, np.random.default_rng(0), spawn=(0.0, 0.0, 0.0))
    heights = env.task.height_map(env.data)
    grid = heights.reshape(13, 7)  # (x ahead: -0.32..0.64, y: -0.24..0.24)
    assert abs(grid[4, 3]) < 0.02  # under the torso: the landing
    assert grid[-1].min() > 0.03  # 64 cm ahead: at least a step down (landing: +-40 cm)
    flat = WalkEnv("quadruped12", dataclasses.replace(EXACT, terrain=""))
    flat.reset(seed=0)
    assert np.abs(flat.task.height_map(flat.data)).max() < 0.02


def test_episodes_start_on_the_park_standing():
    env = WalkEnv("quadruped12", TERRAIN_STEER)
    assert env.observation_space.shape == (1 + 3 + 3 + 3 + 3 * 12 + 2 + 3 + 91,)
    frictions = set()
    for seed in range(8):
        obs, _ = env.reset(seed=seed)
        frictions.add(round(float(env.model.geom_friction[0, 0]), 3))
        assert env.task.standing_height - 0.01 < obs[0] < env.task.standing_height + 0.15  # on (or just above) the ground
        for _ in range(25):  # half a second holding the standing pose
            obs, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape))
        assert not terminated
        assert obs[0] == pytest.approx(env.task.standing_height, abs=0.04)
    assert len(frictions) == 8  # new physics every episode


# ------------------------------------------------------------------ GPU

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU and CUDA PyTorch")


@pytest.fixture(scope="module")
def terrain_gpu_env():
    from robot3d.gpu.env import GpuWalkEnv

    env = GpuWalkEnv(num_envs=64, robot="quadruped12", config=EXACT, device="cuda:0", seed=0)
    env.reset()
    return env


@gpu
@pytest.mark.parametrize("kind", ["stairs", "slope", "rough"])
def test_gpu_terrain_matches_cpu(terrain_gpu_env, kind):
    """A robot on the same tile, same state, same actions: the same
    observations (height map included) and rewards on the GPU as on the CPU."""
    from test_gpu_env import put_cpu_state_into_world  # (pytest puts tests/ on sys.path)

    env = terrain_gpu_env
    env.reset()
    world = int(np.nonzero([env.tiles[t].kind == kind for t in env.task.tile.tolist()])[0][0])
    tile = env.tiles[env.task.tile[world].item()]
    cpu = WalkEnv("quadruped12", EXACT, terrain=Terrain.single(tile))
    cpu.reset(seed=4)
    cpu._command = np.array([0.3, 0.0, 0.2])
    env.command[world] = torch.tensor(cpu._command, dtype=torch.float32, device=env.device)
    env.next_command[world] = 10**6
    put_cpu_state_into_world(env, world, cpu)
    cpu_obs = cpu._observe()
    torch.testing.assert_close(env.observe()[world].cpu(), torch.tensor(cpu_obs), rtol=1e-4, atol=2e-4)

    rng = np.random.default_rng(0)
    for step in range(15):
        action = rng.uniform(-0.5, 0.5, cpu.task.num_actions)
        cpu_obs, cpu_reward, *_ = cpu.step(action)
        actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        actions[world] = torch.as_tensor(action, dtype=torch.float32)
        result = env.step(actions)
        gpu_obs = result.obs[world].cpu().numpy()
        tol = 2e-3 * (1 + step)
        np.testing.assert_allclose(gpu_obs[:10], cpu_obs[:10], atol=tol, err_msg=f"torso state, step {step}")
        np.testing.assert_allclose(gpu_obs[-91:], cpu_obs[-91:], atol=tol + 0.01, err_msg=f"height map, step {step}")
        assert result.reward[world].item() == pytest.approx(cpu_reward, abs=0.05 + 0.02 * step), f"reward, step {step}"


@gpu
def test_gpu_robots_stand_on_their_own_tiles(terrain_gpu_env):
    env = terrain_gpu_env
    env.reset(randomize_episode_start=False)
    for world in range(3):  # each world's box slots hold its own tile's boxes
        tile = env.tiles[env.task.tile[world].item()]
        slots = env.geom_xpos[world, env.slot_geoms].cpu().numpy()
        np.testing.assert_allclose(slots[:len(tile.boxes)], [b.center for b in tile.boxes], atol=1e-5)
        assert (slots[len(tile.boxes):, 2] < -0.5).all()  # the rest: hidden under the floor
    zero = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    for _ in range(50):  # a second holding the standing pose
        result = env.step(zero)
    height = env.task.height(env.qpos)
    assert (height - env.task.standing_height).abs().median() < 0.01
    assert result.obs.shape == (64, env.num_obs)


@gpu
def test_terrain_curriculum_on_the_gpu(terrain_gpu_env):
    """Walking off its tile ends the episode like a time-out and moves the
    robot a level up (onto a new tile); falling moves it a level down."""
    env = terrain_gpu_env
    env.reset(randomize_episode_start=False)
    env.level[:] = 4
    zero = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    env.start_xy[0, 0] -= LEAVE_DISTANCE + 0.2  # as if it had walked this far
    env.qpos[1, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device)  # fell over
    result = env.step(zero)
    assert result.done[0] and result.time_out[0]
    assert result.done[1] and not result.time_out[1]
    assert env.level[0].item() == 5 and env.level[1].item() == 3
    assert env.level[2:].eq(4).all()
    level_of = lambda world: env.task.tile[world].item() // (len(TYPES) * env.task.config.terrain_variants)  # noqa: E731
    assert level_of(0) == 5 and level_of(1) == 3  # new tiles at the new levels
    assert result.stats["curriculum/terrain_level"].item() == pytest.approx(4.0, abs=0.05)


@gpu
def test_randomized_physics_on_the_gpu():
    from robot3d.gpu.env import GpuWalkEnv

    env = GpuWalkEnv(num_envs=32, robot="quadruped12", config=TERRAIN_STEER, device="cuda:0", seed=1)
    env.reset()
    friction = env.friction[:, 0, 0]
    assert friction.min() >= 0.4 and friction.max() <= 1.25 and friction.std() > 0.1
    mass = env.body_mass[:, 1] - env._nominal_mass
    assert mass.min() >= -0.5 and mass.max() <= 1.5 and mass.std() > 0.2
    kp = env.gainprm[:, 0, 0] / env._nominal_gain[0]
    np.testing.assert_allclose(kp.cpu(), (-env.biasprm[:, 0, 1] / env._nominal_gain[0]).cpu(), rtol=1e-5)
    zero = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    for _ in range(25):
        result = env.step(zero)
    assert not result.done.any() or result.time_out[result.done].all()  # nobody falls just standing


def test_terrain_runs_are_tested_on_the_course(tiny_terrain_run):
    from robot3d.policy import evaluate, find_checkpoint, skill_test

    checkpoint = find_checkpoint(tiny_terrain_run)
    result = skill_test(checkpoint)
    assert "test course" in result["skill_test"] and 0.0 <= result["skill"] <= 1.0
    episodes = evaluate(checkpoint, episodes=1)
    assert episodes[0]["seconds"] > 0


def _feet_against_the_first_step(env) -> list[bool]:
    """Put env's robot at the foot of its stairs tile, the front foot that
    reaches furthest pressed 2 mm into the first riser. Returns which feet
    touch the riser."""
    tile = env.terrain.tiles[0][0]
    env.task.reset_state(env.data, np.random.default_rng(0), spawn=(-1.6, 0.0, 0.0))
    riser = -tile.boxes[0].half[0]
    reach = env.data.geom_xpos[env.task.feet, 0] + env.task.foot_radius
    env.data.qpos[0] += riser - reach.max() + 0.002
    mujoco.mj_forward(env.model, env.data)
    reach = env.data.geom_xpos[env.task.feet, 0] + env.task.foot_radius
    return (reach > riser).tolist()


def test_stumbling_feet_are_feet_against_a_riser():
    tile = make_tile("stairs", 7, np.random.default_rng(0))
    env = WalkEnv("quadruped12", EXACT, terrain=Terrain.single(tile))
    env.reset(seed=0, options={"spawn": (-1.6, 0.0, 0.0)})
    assert not env.task.stumbling_feet(env.data).any()  # standing on the floor: normals point up
    touching = _feet_against_the_first_step(env)
    assert sum(touching) >= 1 and not any(touching[2:])  # a front foot, not the back ones
    assert env.task.stumbling_feet(env.data).tolist() == touching


@gpu
def test_stumbling_feet_on_the_gpu(terrain_gpu_env):
    import mujoco_warp as mjw
    import warp as wp
    from test_gpu_env import put_cpu_state_into_world  # (pytest puts tests/ on sys.path)

    env = terrain_gpu_env
    env.reset()
    world = int(np.nonzero([env.tiles[t].kind == "stairs" for t in env.task.tile.tolist()])[0][0])
    cpu = WalkEnv("quadruped12", EXACT, terrain=Terrain.single(env.tiles[env.task.tile[world].item()]))
    cpu.reset(seed=0)
    touching = _feet_against_the_first_step(cpu)
    put_cpu_state_into_world(env, world, cpu)
    with wp.ScopedDevice(env.wp_device), wp.ScopedStream(wp.stream_from_torch(env.device)):
        mjw.collision(env.m, env.d)  # the contact list for this state
    stumbling = env.task.stumbling_feet(env.contact_geom, env.contact_frame, env.contact_world, env.nacon,
                                        env.num_envs)
    assert stumbling[world].tolist() == cpu.task.stumbling_feet(cpu.data).tolist() == touching
