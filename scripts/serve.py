"""Run the simulation server.

    uv run scripts/serve.py                         # quadruped at http://localhost:8000
    uv run scripts/serve.py --policy runs/<name>    # a trained policy drives (newest checkpoint)
    uv run scripts/serve.py --policy runs/<name>/checkpoints/step_002000000.zip

Development: also run `npm run dev` in web/ and open http://localhost:5173.
Vite serves the page and forwards the /ws WebSocket to this server.
After `npm run build` in web/, this server serves the built page itself at
http://localhost:8000.
"""

import argparse

import uvicorn

from robot3d.robots import available_robots
from robot3d.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=available_robots(),
                        help="default: quadruped, or the robot the --policy was trained on")
    parser.add_argument("--keyframe", default="home", help="keyframe to start from and reset to")
    parser.add_argument("--policy", help="run folder (uses its newest checkpoint) or checkpoint .zip")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices on your network")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    robot = args.robot
    if robot is None:
        robot = "quadruped"
        if args.policy:
            from robot3d.runs import find_checkpoint

            robot = find_checkpoint(args.policy).run_info()["robot"]

    app = create_app(robot, args.keyframe, policy=args.policy)
    if args.policy:
        behaviors = app.state.runner.behaviors
        print(f"Policy: {behaviors.walk_label or behaviors.stand_label} ({behaviors.mode})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
