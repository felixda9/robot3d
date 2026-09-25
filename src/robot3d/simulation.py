"""A MuJoCo simulation that advances in real time.

Shared by the MuJoCo viewer script and the web server so both pace time the
same way.
"""

import time
from collections.abc import Mapping
from typing import Protocol

import mujoco
import numpy as np

from robot3d.robots import load_model, reset_to_keyframe


class Controller(Protocol):
    """Something that drives the motors, e.g. a trained policy (policy.py)."""

    decimation: int  # act every this many physics steps

    def reset(self) -> None: ...  # forget any memory (e.g. the previous action)

    def reset_state(self, data: mujoco.MjData) -> None: ...  # start state it expects (e.g. standing)

    def act(self, data: mujoco.MjData) -> None: ...  # sets data.ctrl


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
        self.actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        # Per-motor target glides (see set_ctrl): from, to, start time, duration.
        # A duration of 0 means "no glide in progress".
        nu = self.model.nu
        self._glide_from = np.zeros(nu)
        self._glide_to = np.zeros(nu)
        self._glide_start = np.zeros(nu)
        self._glide_duration = np.zeros(nu)
        # An optional controller (policy) that sets the motor targets itself.
        self.controller: Controller | None = None
        self.controller_active = False
        self._physics_steps = 0  # since reset; decides when the controller acts
        self.reset()

    def set_controller(self, controller: Controller | None, active: bool = True) -> None:
        self.controller = controller
        self.use_controller(active and controller is not None)
        if self.controller_active:
            self.reset()  # start the way the controller expects

    def use_controller(self, active: bool) -> None:
        """Hand the motors to the controller (True) or back to manual targets (False).
        Switching to manual keeps the controller's last targets."""
        if active and self.controller is None:
            raise ValueError("no controller loaded")
        self.controller_active = active
        if active:
            self._glide_duration[:] = 0.0
            self.controller.reset()
            self._physics_steps = 0

    def set_ctrl(self, targets: Mapping[str, float], duration: float = 0.0) -> None:
        """Set motor targets by actuator name, clamped to each motor's range.

        Our motors are position actuators, so a target is a joint angle: the
        motor's PD controller then pushes the joint toward it (limited by its
        torque range); the joint doesn't jump there.

        duration > 0: instead of jumping, the targets of all named motors glide
        from their current values to the new ones over `duration` seconds of
        sim time, together, so they arrive at the same moment. The robot then
        passes only through in-between poses. A jump between very different
        poses makes some joints finish long before others, and the lopsided
        poses in between can tip the robot over (e.g. sit -> tall).
        """
        if self.controller_active:
            raise RuntimeError("a controller (policy) is driving the motors; switch to manual first")
        for name, value in targets.items():
            i = self.actuator_names.index(name)  # ValueError for unknown names
            if self.model.actuator_ctrllimited[i]:
                low, high = self.model.actuator_ctrlrange[i]
                value = min(max(value, low), high)
            if duration > 0:
                self._glide_from[i] = self.data.ctrl[i]
                self._glide_to[i] = value
                self._glide_start[i] = self.data.time
                self._glide_duration[i] = duration
            else:
                self._glide_duration[i] = 0.0  # cancels a glide on this motor
                self.data.ctrl[i] = value
        if self.paused:
            # Physics isn't stepping, but recompute derived quantities (like the
            # torque each motor would now apply) so the UI can show them.
            mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        """Back to the start: the start keyframe, or, while a controller
        drives, the state it expects (a policy: standing, as in training).
        Keeps the paused/playing state."""
        self._glide_duration[:] = 0.0
        if self.controller_active:
            self.controller.reset_state(self.data)
        else:
            reset_to_keyframe(self.model, self.data, self.keyframe)
        self._physics_steps = 0
        if self.controller is not None:
            self.controller.reset()
        self._resync()

    def pause(self) -> None:
        self.paused = True

    def play(self) -> None:
        if self.paused:
            self.paused = False
            self._resync()  # don't try to "catch up" on the time spent paused

    def step(self) -> None:
        """One physics step (2 ms). First the controller (if active, every
        `decimation` steps) or the motor-target glides update data.ctrl."""
        if self.controller_active and self._physics_steps % self.controller.decimation == 0:
            self.controller.act(self.data)
        self._update_glides()
        mujoco.mj_step(self.model, self.data)
        self._physics_steps += 1

    def advance(self, now: float | None = None) -> int:
        """Step the physics until sim time catches up with the wall clock.

        Real-time pacing: we remember one (wall time, sim time) pair, the
        "anchor". At wall time `now` the sim should be at
        anchor_sim + (now - anchor_wall). Each step advances data.time by one
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
            self.step()
            steps += 1
        if steps == self.max_steps_per_advance:
            self._resync(now)  # fell behind: continue from here, slower than real time
        return steps

    def _update_glides(self) -> None:
        active = self._glide_duration > 0
        if not active.any():
            return
        progress = (self.data.time - self._glide_start[active]) / self._glide_duration[active]
        s = np.clip(progress, 0.0, 1.0)
        s = s * s * (3 - 2 * s)  # "smoothstep": starts and stops gently instead of lurching
        start, end = self._glide_from[active], self._glide_to[active]
        self.data.ctrl[active] = start + (end - start) * s
        finished = np.flatnonzero(active)[progress >= 1.0]
        self._glide_duration[finished] = 0.0

    def _resync(self, now: float | None = None) -> None:
        self._anchor_wall = time.perf_counter() if now is None else now
        self._anchor_sim = self.data.time
