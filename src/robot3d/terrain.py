"""Terrain (milestone 7): uneven ground built from boxes, for training and testing.

How others make a walker that copes with any ground (legged_gym, ANYmal,
RMA; see docs/research.md): one policy trained on a broad random mix of
ground that gets harder as it improves, not one skill per obstacle. Ground
in the real world is mostly combinations of what a foot can meet here:
bumps and tilted patches (rough), slopes, and edges (stairs, obstacles).

The unit is a TILE x TILE m **tile** of one type at one difficulty (0..1),
centered on (0, 0) in its own coordinates, made of at most MAX_TILE_BOXES
boxes. Every tile is drawn at random within its level: the same level is
never the same twice, so a policy can't memorize it.

  * GPU training (gpu/env.py): every robot gets its own tile from a pool
    (park_tiles(): LEVELS x TYPES x variants) and swaps it for another at
    each reset. MuJoCo Warp can give each simulated world different box
    positions, so each robot only collides with its own ~25 boxes (with all
    tiles in every world, 4096 robots were 6x slower than on flat ground).
  * The viewer and CPU runs use layouts of tiles (Terrain):
      - "park": one tile per level and type, levels along +x (the robot
        starts on flat floor and walks forward into harder ground), types
        side by side along y;
      - "course" (testing): a lane along +x of six sections of shapes the
        park never has (turned rubble, long ramps, a single tall step,
        narrow-tread stairs, a stepping field, a cross-slope). Which of them
        a policy gets across measures how well it generalizes.

Everything is boxes (primitive shapes, like the robots), added to a robot's
model when it's loaded (robots.load_model(robot, terrain=...)), so robot
files stay unchanged. Boxes can be turned (yaw) and tilted (pitch, roll).

Heights: the top surface of the boxes is rasterized into a GRID m height
grid once, so the task can look up the ground height anywhere cheaply, on
the CPU and batched on the GPU: for the height map the policy sees, and for
"height above the ground" (falls, rewards, feet on the ground).
"""

import math
from dataclasses import dataclass, field
from functools import cached_property

import mujoco
import numpy as np

TYPES = ("rough", "slope", "stairs", "obstacles")
LEVELS = 10  # difficulty rows of the park and the tile pool
TILE = 3.0  # m
MAX_TILE_BOXES = 25
GRID = 0.02  # m: height grid resolution
PARK_START_X = 1.5  # m: the park's near edge (the robot starts at the origin)
BORDER = 0.25  # m: flat floor around stairs and slopes, to start at their foot

# Difficulty: each value goes from its first number (difficulty 0) to the
# second (1). For quadruped12 (legs 2 x 16 cm, torso 26 cm up): steps up to
# 12 cm, a bit under half its leg length.
ROUGH_HEIGHT = (0.01, 0.08)  # m: slab tops up to this high ...
ROUGH_TILT_DEG = (0.0, 8.0)  # ... and tilted up to this much
SLOPE_DEG = (0.0, 25.0)
STEP_HEIGHT = (0.02, 0.12)  # m
OBSTACLE_HEIGHT = (0.02, 0.12)  # m


@dataclass(frozen=True)
class Box:
    center: tuple[float, float, float]
    half: tuple[float, float, float]  # half-sizes along the box's own x, y, z
    yaw: float = 0.0  # rad, about world z
    pitch: float = 0.0  # rad, about its own y axis: positive lowers its +x end
    roll: float = 0.0  # rad, about its own x axis: positive raises its +y side

    @property
    def rotation(self) -> np.ndarray:
        """Box frame -> world frame: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return rz @ ry @ rx

    @staticmethod
    def with_top(top_center, half_xy, thickness, **angles) -> "Box":
        """A box whose top face is centered at `top_center`, `thickness` m thick."""
        box = Box((0.0, 0.0, 0.0), (*half_xy, thickness / 2), **angles)
        center = np.asarray(top_center) - box.rotation[:, 2] * thickness / 2
        return Box(tuple(center), box.half, **angles)

    def moved(self, dx: float, dy: float) -> "Box":
        return Box((self.center[0] + dx, self.center[1] + dy, self.center[2]), self.half,
                   self.yaw, self.pitch, self.roll)


def _lerp(spec: tuple[float, float], difficulty: float) -> float:
    return spec[0] + (spec[1] - spec[0]) * difficulty


# ------------------------------------------------------------------ tiles


@dataclass
class Tile:
    kind: str
    level: int
    difficulty: float  # 0..1
    boxes: list[Box]  # in tile coordinates: centered on (0, 0)
    # Where robots start, in tile coordinates. spawns[0]: the middle (on top
    # of stairs and slopes: walk down); the rest: at the foot (walk up).
    spawns: np.ndarray = field(default_factory=lambda: np.zeros((1, 2)))

    @cached_property
    def heights(self) -> np.ndarray:
        """This tile's height grid, TILE_GRID_POINTS square, from -TILE_GRID_HALF."""
        return rasterize(self.boxes, -TILE_GRID_HALF, -TILE_GRID_HALF, TILE_GRID_POINTS, TILE_GRID_POINTS)


TILE_GRID_HALF = TILE / 2 + 0.1  # m: a tile's grid reaches a little past its edge (tilted boxes)
TILE_GRID_POINTS = round(2 * TILE_GRID_HALF / GRID) + 1


def make_tile(kind: str, level: int, rng: np.random.Generator) -> Tile:
    """A random tile of this type within this level's band of difficulty."""
    difficulty = (level + rng.uniform()) / LEVELS
    return _TILE_MAKERS[kind](level, difficulty, rng)


def park_tiles(seed: int = 0, variants: int = 5) -> list[Tile]:
    """The GPU tile pool: `variants` random tiles per level and type,
    ordered level, type, variant (index = (level * len(TYPES) + type) * variants + variant)."""
    rng = np.random.default_rng(seed)
    return [make_tile(kind, level, rng) for level in range(LEVELS) for kind in TYPES for _ in range(variants)]


SPAWN_JITTER = 0.1  # m: start this far (at most) from a spawn point, each way
LEAVE_DISTANCE = 1.2  # m from its start: a robot has walked off its tile (GPU: next tile, a level up)


def spawn_point(tile: Tile, rng: np.random.Generator) -> tuple[float, float, float]:
    """(x, y, heading) in tile coordinates: half the time the middle (on top
    of stairs and slopes), otherwise one of its other spawn points (at their
    feet), facing any way. The GPU env draws the same way (gpu/env.py)."""
    spawns = tile.spawns
    i = 0 if len(spawns) == 1 or rng.uniform() < 0.5 else rng.integers(1, len(spawns))
    x, y = spawns[i] + rng.uniform(-SPAWN_JITTER, SPAWN_JITTER, 2)
    return float(x), float(y), float(rng.uniform(-math.pi, math.pi))


def _border_spawns(rng, sides="xy", n=8) -> np.ndarray:
    """Points at the foot of stairs or slopes, on random sides: just outside
    the tile, so the robot (+-0.2 m) stands clear of the first step (GPU:
    flat floor there; park: the next tile)."""
    points = []
    foot = TILE / 2 + 0.1
    for _ in range(n):
        along = rng.uniform(-1.0, 1.0)
        sign = rng.choice([-1.0, 1.0])
        axis = rng.choice(list(sides))
        points.append((sign * foot, along) if axis == "x" else (along, sign * foot))
    return np.array(points)


def _rough(level, difficulty, rng) -> Tile:
    """5 x 5 slabs of 0.6 m, each at its own height and tilted its own way."""
    top = _lerp(ROUGH_HEIGHT, difficulty)
    max_tilt = math.radians(_lerp(ROUGH_TILT_DEG, difficulty))
    cell = TILE / 5
    boxes = []
    for i in range(5):
        for j in range(5):
            pitch, roll = rng.uniform(-max_tilt, max_tilt, 2)
            # high enough that its lowest top corner is still above the floor
            drop = cell / 2 * (abs(math.tan(pitch)) + abs(math.tan(roll)))
            h = rng.uniform(0.0, top) + drop
            x, y = -TILE / 2 + (i + 0.5) * cell, -TILE / 2 + (j + 0.5) * cell
            boxes.append(Box.with_top((x, y, h), (cell / 2 + 0.01, cell / 2 + 0.01), h + 0.1, pitch=pitch, roll=roll))
    return Tile("rough", level, difficulty, boxes, np.array([(0.0, 0.0)]))


def _slope(level, difficulty, rng) -> Tile:
    """A ridge along x with ramps down to both sides (y), a flat border at
    their feet. Starts on the crest (walk down) or at a foot (walk up)."""
    angle = math.radians(_lerp(SLOPE_DEG, difficulty))
    crest = 0.3  # half-width of the flat top
    run = TILE / 2 - BORDER - crest
    rise = run * math.tan(angle)
    length = run / math.cos(angle)
    boxes = []
    if rise > 0.005:
        boxes.append(Box((0.0, 0.0, rise / 2), (TILE / 2, crest, rise / 2)))
    for side in (1, -1):
        # roll > 0 raises +y; the +y ramp comes down toward +y, the -y ramp rises toward +y
        boxes.append(Box.with_top((0.0, side * (crest + run / 2), rise / 2), (TILE / 2, length / 2), 0.06,
                                  roll=-side * angle))
    spawns = np.vstack([[(0.0, 0.0)], _border_spawns(rng, sides="y")])
    return Tile("slope", level, difficulty, boxes, spawns)


def _stairs(level, difficulty, rng) -> Tile:
    """A stepped pyramid (nested boxes) with a landing on top and a flat
    border around it. Starts on the landing (walk down) or on the border (walk up)."""
    step = _lerp(STEP_HEIGHT, difficulty)
    tread = rng.uniform(0.25, 0.35)
    landing = 0.4  # half-width
    base = TILE / 2 - BORDER
    rings = int((base - landing) / tread)
    boxes = []
    for k in range(rings + 1):  # k = 0: the bottom step, k = rings: the landing
        half = base - k * tread if k < rings else landing
        h = step * (k + 1)
        boxes.append(Box((0.0, 0.0, h / 2), (half, half, h / 2)))
    spawns = np.vstack([[(0.0, 0.0)], _border_spawns(rng)])
    return Tile("stairs", level, difficulty, boxes, spawns)


def _obstacles(level, difficulty, rng) -> Tile:
    """Scattered blocks (legged_gym's "discrete obstacles"); the middle stays clear to start on."""
    top = _lerp(OBSTACLE_HEIGHT, difficulty)
    boxes = []
    while len(boxes) < 10:
        x, y = rng.uniform(-1.3, 1.3, 2)
        hx, hy = rng.uniform(0.1, 0.35, 2)
        if abs(x) - hx < 0.35 and abs(y) - hy < 0.35:
            continue
        h = rng.uniform(0.5, 1.0) * top
        boxes.append(Box((x, y, h / 2), (hx, hy, h / 2), yaw=rng.uniform(0, math.pi / 2)))
    return Tile("obstacles", level, difficulty, boxes, np.array([(0.0, 0.0)]))


_TILE_MAKERS = {"rough": _rough, "slope": _slope, "stairs": _stairs, "obstacles": _obstacles}


# ---------------------------------------------------------------- layouts


class Terrain:
    """Boxes in world coordinates (a layout of tiles, or one tile at the
    origin), their height grid, and how to add them to a model."""

    def __init__(self, name: str, boxes: list[Box], tiles: list[tuple[Tile, float, float]] | None = None):
        self.name = name
        self.boxes = boxes
        self.tiles = tiles or []  # (tile, x, y of its center): where robots can start on it

    @classmethod
    def make(cls, name: str, seed: int = 0) -> "Terrain":
        """A named layout: "park" or "course"."""
        if name == "park":
            return cls.park(seed)
        if name == "course":
            return cls.course(seed)
        raise ValueError(f"unknown terrain {name!r} (known: park, course)")

    def random_spawn(self, rng: np.random.Generator) -> tuple[float, float, float]:
        """(x, y, heading) to start a robot at: a random tile of this layout (spawn_point)."""
        tile, cx, cy = self.tiles[rng.integers(len(self.tiles))]
        x, y, heading = spawn_point(tile, rng)
        return cx + x, cy + y, heading

    @staticmethod
    def park_tile_center(level: int, type_index: int) -> tuple[float, float]:
        return PARK_START_X + (level + 0.5) * TILE, (type_index - (len(TYPES) - 1) / 2) * TILE

    @classmethod
    def park(cls, seed: int = 0) -> "Terrain":
        """One random tile per level and type: levels along +x, types along y."""
        rng = np.random.default_rng(seed)
        boxes, tiles = [], []
        for level in range(LEVELS):
            for t, kind in enumerate(TYPES):
                cx, cy = cls.park_tile_center(level, t)
                tile = make_tile(kind, level, rng)
                boxes += [b.moved(cx, cy) for b in tile.boxes]
                tiles.append((tile, cx, cy))
        return cls("park", boxes, tiles)

    @classmethod
    def single(cls, tile: Tile) -> "Terrain":
        """One tile at the origin (as a GPU world has it)."""
        terrain = cls("tile", list(tile.boxes), [(tile, 0.0, 0.0)])
        # the tile's own grid, exactly as a GPU world looks it up
        terrain.__dict__["heights"] = (tile.heights, -TILE_GRID_HALF, -TILE_GRID_HALF)
        return terrain

    @classmethod
    def course(cls, seed: int = 0) -> "Terrain":
        """The test course: a 1.6 m lane along +x with six sections of shapes
        the park never has, 1.5 m of flat floor before each, so each can be
        tried on its own (policy.course_test). `sections`: (name, start x, end
        x); a test starts 1 m before start and passes 0.5 m after end."""
        rng = np.random.default_rng(seed + 1000)
        boxes: list[Box] = []
        sections: list[tuple[str, float, float]] = []
        half_w = 0.8
        x = 0.0

        def section(name: str, length: float) -> float:
            """Start a section after 1.5 m of floor; returns its start x."""
            nonlocal x
            start = x + 1.5
            sections.append((name, start, start + length))
            x = start + length
            return start

        # 1. Turned rubble: slabs at random angles (the park's are axis-aligned).
        s = section("turned rubble", 3.0)
        for _ in range(30):
            h = rng.uniform(0.01, 0.05)
            boxes.append(Box((s + rng.uniform(0.2, 2.8), rng.uniform(-0.7, 0.7), h / 2),
                             (rng.uniform(0.05, 0.2), rng.uniform(0.05, 0.2), h / 2), yaw=rng.uniform(0, math.pi)))
        # 2. A long ramp up at 12 deg, a 1 m plateau, a steeper one down at 20 deg.
        rise = 0.25
        up, down = rise / math.tan(math.radians(12)), rise / math.tan(math.radians(20))
        s = section("ramps 12/20 deg", up + 1.0 + down)
        boxes.append(_ramp(s, s + up, 0.0, rise, half_w))
        boxes.append(Box((s + up + 0.5, 0.0, rise / 2), (0.5, half_w, rise / 2)))
        boxes.append(_ramp(s + up + 1.0, s + up + 1.0 + down, rise, 0.0, half_w))
        # 3. One tall step (9 cm) up and, 1 m later, down.
        s = section("9 cm step", 1.0)
        boxes.append(Box((s + 0.5, 0.0, 0.045), (0.5, half_w, 0.045)))
        # 4. Narrow stairs: 6 steps of 6 cm on 22 cm treads up, a landing, 6 down.
        tread, step, n = 0.22, 0.06, 6
        s = section("narrow stairs", 2 * n * tread + 0.8)
        for k in range(n):
            h = step * (k + 1)
            boxes.append(Box((s + (k + 0.5) * tread, 0.0, h / 2), (tread / 2, half_w, h / 2)))
        top = s + n * tread
        boxes.append(Box((top + 0.4, 0.0, step * n / 2), (0.4, half_w, step * n / 2)))
        for k in range(n):
            h = step * (n - k)
            boxes.append(Box((top + 0.8 + (k + 0.5) * tread, 0.0, h / 2), (tread / 2, half_w, h / 2)))
        # 5. Stepping field: 25 cm blocks, each 0-7 cm high.
        s = section("stepping field", 3.0)
        for i in range(12):
            for j in range(6):
                h = rng.uniform(0.0, 0.07)
                if h > 0.005:
                    boxes.append(Box((s + (i + 0.5) * 0.25, -0.75 + (j + 0.5) * 0.25, h / 2), (0.125, 0.125, h / 2)))
        # 6. Cross-slope: the lane tilted 10 deg sideways (left side up) for 3 m,
        # raised so its center line is level with the 10 deg ramps up and down
        # to it (at the lane's edges, the ramps meet it 14 cm off).
        tilt = math.radians(10)
        lift = half_w * math.tan(tilt) + 0.01  # center height: the low edge just above the floor
        ramp = lift / math.tan(tilt)
        s = section("10 deg cross-slope", 2 * ramp + 3.0)
        boxes.append(_ramp(s, s + ramp, 0.0, lift, half_w))
        boxes.append(Box.with_top((s + ramp + 1.5, 0.0, lift), (1.5, half_w / math.cos(tilt)), 0.3, roll=tilt))
        boxes.append(_ramp(s + ramp + 3.0, s + 2 * ramp + 3.0, lift, 0.0, half_w))
        # Its test starts on the slope (0.3 m in) and ends 0.5 m before its end:
        # off the center line, the junctions with the flat ramps have lips (a
        # sideways-tilted face can't meet a level edge everywhere), and a rear
        # foot caught on one isn't what this section tests.
        sections[-1] = (sections[-1][0], s + ramp + 1.3, s + ramp + 2.0)
        terrain = cls("course", boxes)
        terrain.sections = sections
        terrain.length = x
        return terrain

    # -------------------------------------------------------------- model

    def add_to(self, spec: mujoco.MjSpec) -> None:
        """Add the boxes to a robot's MjSpec as static world geoms (colliding
        with the robot like the floor does)."""
        for k, b in enumerate(self.boxes):
            g = spec.worldbody.add_geom()
            g.name = f"terrain{k}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.pos = b.center
            g.size = b.half
            g.quat = box_quat(b)
            g.rgba = box_rgba(k)

    # ------------------------------------------------------------ heights

    @cached_property
    def heights(self) -> tuple[np.ndarray, float, float]:
        """(heights[ix, iy], x0, y0): the top of the ground on a GRID m grid
        whose first point is at (x0, y0); 0 = the floor."""
        if not self.boxes:
            return np.zeros((2, 2)), 0.0, 0.0
        lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
        for b in self.boxes:
            reach = np.abs(b.rotation[:2, :]) @ np.array(b.half)
            lo = np.minimum(lo, np.array(b.center[:2]) - reach)
            hi = np.maximum(hi, np.array(b.center[:2]) + reach)
        x0, y0 = lo - 0.5
        nx, ny = (np.ceil((hi + 0.5 - (lo - 0.5)) / GRID) + 1).astype(int)
        return rasterize(self.boxes, x0, y0, nx, ny), float(x0), float(y0)

    def ground_height(self, x, y) -> np.ndarray:
        """Ground height at world (x, y) (arrays ok): bilinear in the grid; 0 off it."""
        heights, x0, y0 = self.heights
        return bilinear(heights, x0, y0, x, y)

    def ground_height_max(self, x, y) -> np.ndarray:
        """The highest of the 4 grid points around (x, y): for feet, which rest
        on the higher side of an edge (bilinear would put a foot standing on a
        stair's edge half a step up in the air)."""
        heights, x0, y0 = self.heights
        return highest(heights, x0, y0, x, y)


def _ramp(x0, x1, z0, z1, half_w, thick=0.06) -> Box:
    """A ramp along x whose top face goes from height z0 at x0 to z1 at x1."""
    length = math.hypot(x1 - x0, z1 - z0)
    pitch = -math.atan2(z1 - z0, x1 - x0)  # positive pitch lowers +x
    return Box.with_top(((x0 + x1) / 2, 0.0, (z0 + z1) / 2), (length / 2, half_w), thick, pitch=pitch)


def box_quat(b: Box) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, b.rotation.flatten())
    return quat


def box_rgba(k: int) -> list[float]:
    """A slightly different gray per box, so edges and steps stand out."""
    shade = 0.45 + 0.1 * ((k * 0.618) % 1.0)
    return [shade, shade * 1.02, shade * 1.08, 1.0]


# ---------------------------------------------------------------- heights


def rasterize(boxes: list[Box], x0: float, y0: float, nx: int, ny: int) -> np.ndarray:
    """(nx, ny) heights of the boxes' top faces on the grid x0 + i GRID, y0 + j GRID (0 where none)."""
    xs = x0 + GRID * np.arange(nx)
    ys = y0 + GRID * np.arange(ny)
    heights = np.zeros((nx, ny))
    for b in boxes:
        rot = b.rotation
        reach = np.abs(rot[:2, :]) @ np.array(b.half)
        ix = np.nonzero(np.abs(xs - b.center[0]) <= reach[0])[0]
        iy = np.nonzero(np.abs(ys - b.center[1]) <= reach[1])[0]
        if len(ix) == 0 or len(iy) == 0:
            continue
        normal = rot[:, 2]  # the top face's normal
        gx, gy = np.meshgrid(xs[ix] - b.center[0], ys[iy] - b.center[1], indexing="ij")
        # The top face's plane: normal . (p - center) = half_z
        gz = (b.half[2] - normal[0] * gx - normal[1] * gy) / normal[2]
        local = np.stack([gx, gy, gz], axis=-1) @ rot  # into the box's frame (R^T p, row-wise)
        inside = (np.abs(local[..., 0]) <= b.half[0] + 1e-9) & (np.abs(local[..., 1]) <= b.half[1] + 1e-9)
        top = np.where(inside, b.center[2] + gz, 0.0)
        heights[np.ix_(ix, iy)] = np.maximum(heights[np.ix_(ix, iy)], top)
    return heights


def _cells(heights, x0, y0, x, y):
    fx = (np.asarray(x, dtype=np.float64) - x0) / GRID
    fy = (np.asarray(y, dtype=np.float64) - y0) / GRID
    nx, ny = heights.shape
    i = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    j = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    outside = (fx < 0) | (fy < 0) | (fx > nx - 1) | (fy > ny - 1)
    return i, j, np.clip(fx - i, 0, 1), np.clip(fy - j, 0, 1), outside


def bilinear(heights, x0, y0, x, y) -> np.ndarray:
    """Height at (x, y), bilinear between grid points; 0 off the grid."""
    i, j, tx, ty, outside = _cells(heights, x0, y0, x, y)
    value = (heights[i, j] * (1 - tx) * (1 - ty) + heights[i + 1, j] * tx * (1 - ty)
             + heights[i, j + 1] * (1 - tx) * ty + heights[i + 1, j + 1] * tx * ty)
    return np.where(outside, 0.0, value)


def highest(heights, x0, y0, x, y) -> np.ndarray:
    """The highest of the 4 grid points around (x, y); 0 off the grid."""
    i, j, _, _, outside = _cells(heights, x0, y0, x, y)
    value = np.maximum(np.maximum(heights[i, j], heights[i + 1, j]), np.maximum(heights[i, j + 1], heights[i + 1, j + 1]))
    return np.where(outside, 0.0, value)


# The height map the policy sees: ground heights at these points around the
# torso, in its heading frame (x forward, y left), more ahead than behind.
# 13 x 7 = 91 points, 8 cm apart (legged_gym: 17 x 11 at 10 cm for ANYmal,
# ~1.7x our size).
HEIGHT_MAP_X = np.round(np.arange(-0.32, 0.641, 0.08), 6)
HEIGHT_MAP_Y = np.round(np.arange(-0.24, 0.241, 0.08), 6)
HEIGHT_MAP_POINTS = np.array([(x, y) for x in HEIGHT_MAP_X for y in HEIGHT_MAP_Y])  # (91, 2)
