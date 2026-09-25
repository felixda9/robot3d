"""A MuJoCo simulation that advances in real time.

Shared by the MuJoCo viewer script and the web server so both pace time the
same way.
"""

import time

import mujoco

from robot3d.robots import load_model, reset_to_keyframe


class Simulation:
    """One robot's model + state, stepped so sim time keeps pace with the wall clock.

    Not thread-safe: use it from one thread (the web server gives it its own).
    """

    def __init__(self, robot: str = "quadruped", keyframe: str = "home", max_steps_per_advance: int = 50):
        self.robot = robot
        self.keyframe = keyframe
        self.model = load_model(robot)
        self.data = mujoco.MjData(self.model)
        # If the computer can't keep up (or a frame was delayed), take at most
        # this many physics steps per advance() and let the sim fall behind,
        # instead of trying ever harder to catch up and freezing.
        self.max_steps_per_advance = max_steps_per_advance
        self.paused = False
        self.reset()

    def reset(self) -> None:
        """Back to the start keyframe. Keeps the paused/playing state."""
        reset_to_keyframe(self.model, self.data, self.keyframe)
        self._resync()

    def pause(self) -> None:
        self.paused = True

    def play(self) -> None:
        if self.paused:
            self.paused = False
            self._resync()  # don't try to "catch up" on the time spent paused

    def advance(self, now: float | None = None) -> int:
        """Step the physics until sim time catches up with the wall clock.

        Real-time pacing: we remember one (wall time, sim time) pair, the
        "anchor". At wall time `now` the sim should be at
        anchor_sim + (now - anchor_wall). Each mj_step advances data.time by one
        timestep (2 ms), so at 60 fps that's about 8 steps per call.

        `now` is a time.perf_counter() value (pass one in tests to fake the
        clock). Returns the number of physics steps taken.
        """
        if self.paused:
            return 0
        now = time.perf_counter() if now is None else now
        target_time = self._anchor_sim + (now - self._anchor_wall)
        steps = 0
        while self.data.time < target_time and steps < self.max_steps_per_advance:
            mujoco.mj_step(self.model, self.data)
            steps += 1
        if steps == self.max_steps_per_advance:
            self._resync(now)  # fell behind: continue from here, slower than real time
        return steps

    def _resync(self, now: float | None = None) -> None:
        self._anchor_wall = time.perf_counter() if now is None else now
        self._anchor_sim = self.data.time
