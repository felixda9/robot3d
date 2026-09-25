"""Watch a robot in MuJoCo's built-in viewer (Milestone 1).

    uv run scripts/view_mujoco.py                     # quadruped, dropped from "home"
    uv run scripts/view_mujoco.py --robot quadruped --keyframe home
    uv run scripts/view_mujoco.py --seconds 5         # auto-close after 5 s

We run the simulation loop ourselves (MuJoCo's "passive" viewer only draws).
The FastAPI server in Milestone 2 will use the same loop.

Keys (with the viewer window focused):
    Space       pause / resume
    Backspace   reset to the keyframe (drop the robot again)
Mouse: left-drag = rotate, right-drag = pan, scroll = zoom.
Push the robot: double-click a body part, then Ctrl + right-drag.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from robot3d.robots import available_robots, load_model, reset_to_keyframe

# GLFW key codes (the windowing library the MuJoCo viewer uses).
KEY_SPACE = 32
KEY_BACKSPACE = 259

FRAME_DT = 1 / 60  # redraw the viewer at 60 fps
MAX_STEPS_PER_FRAME = 50  # if the PC can't keep up, slow down instead of freezing
SETTLED_SPEED = 1e-2  # "settled" = no joint moving faster than this (m/s or rad/s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="quadruped", choices=available_robots())
    parser.add_argument("--keyframe", default="home", help="keyframe to start from and reset to")
    parser.add_argument("--seconds", type=float, default=None, help="close automatically after this many seconds")
    args = parser.parse_args()

    model = load_model(args.robot)
    data = mujoco.MjData(model)
    reset_to_keyframe(model, data, args.keyframe)

    # The key callback runs on the viewer's thread, so it only sets flags;
    # the main loop below acts on them while it holds the viewer lock.
    state = {"paused": False, "reset": False}

    def on_key(keycode: int) -> None:
        if keycode == KEY_SPACE:
            state["paused"] = not state["paused"]
        elif keycode == KEY_BACKSPACE:
            state["reset"] = True

    print(
        f"Robot '{args.robot}': {model.nbody - 1} bodies, {model.njnt} joints, {model.nu} motors, "
        f"mass {model.body_subtreemass[1]:.2f} kg, physics step {model.opt.timestep * 1000:.0f} ms"
    )
    print("Space = pause, Backspace = drop again. Close the window to quit.\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        wall_start = time.perf_counter()
        # Real-time pacing: sim time should advance as fast as the wall clock.
        # We remember one (wall time, sim time) pair and step the physics until
        # data.time catches up with the wall clock.
        sync_wall, sync_sim = wall_start, data.time
        next_report, last_status = 0.5, None
        print(f"Starting from keyframe '{args.keyframe}' (torso at {data.qpos[2]:.2f} m)")

        while viewer.is_running():
            frame_start = time.perf_counter()
            if args.seconds is not None and frame_start - wall_start > args.seconds:
                break

            with viewer.lock():
                if state["paused"]:
                    sync_wall, sync_sim = frame_start, data.time  # don't try to "catch up" after a pause
                else:
                    target_sim_time = sync_sim + (frame_start - sync_wall)
                    steps = 0
                    while data.time < target_sim_time and steps < MAX_STEPS_PER_FRAME:
                        mujoco.mj_step(model, data)
                        steps += 1
                    if steps == MAX_STEPS_PER_FRAME:  # fell behind: resync rather than spiral
                        sync_wall, sync_sim = frame_start, data.time

                # Status line every 0.5 s of sim time while moving, once when settled.
                if data.time >= next_report:
                    next_report = data.time + 0.5
                    status = "settled" if np.abs(data.qvel).max() < SETTLED_SPEED else "moving"
                    if status == "moving" or last_status != "settled":
                        report(model, data, status)
                    last_status = status
                time_before_sync = data.time

            # Hand the new state to the viewer to draw. sync() also applies the
            # viewer's own UI actions, including its Backspace/"Reset" handling,
            # which resets to qpos0 (legs straight, motors targeting 0) and
            # sets time back to 0.
            viewer.sync()

            # Swap that reset for our keyframe. We check the time as well as our
            # key flag, which also catches the viewer's "Reset" button.
            if state["reset"] or data.time < time_before_sync:
                state["reset"] = False
                with viewer.lock():
                    reset_to_keyframe(model, data, args.keyframe)
                sync_wall, sync_sim = time.perf_counter(), data.time
                next_report, last_status = 0.5, None
                print(f"\nReset to keyframe '{args.keyframe}' (torso at {data.qpos[2]:.2f} m)")

            sleep = FRAME_DT - (time.perf_counter() - frame_start)
            if sleep > 0:
                time.sleep(sleep)

    print(f"\nViewer closed after {time.perf_counter() - wall_start:.1f} s (sim clock: {data.time:.1f} s).")


def report(model: mujoco.MjModel, data: mujoco.MjData, status: str) -> None:
    """Print a one-line status: torso height, tilt, fastest joint.

    Assumes the robot's root (body 1) is free-floating with a free joint first
    in qpos, which is true for every legged robot in this project."""
    torso_z = data.qpos[2]  # free joint qpos = [x, y, z, qw, qx, qy, qz]
    max_speed = np.abs(data.qvel).max()
    # Element [2, 2] of the torso's rotation matrix = z-component of its "up"
    # axis in world coordinates: 1 when level, 0 when on its side.
    up_z = data.xmat[1].reshape(3, 3)[2, 2]
    tilt_deg = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))
    print(
        f"t={data.time:5.1f} s | torso height {torso_z:.3f} m | tilt {tilt_deg:4.1f} deg | "
        f"fastest joint {max_speed:7.4f} | {status}"
    )


if __name__ == "__main__":
    main()
