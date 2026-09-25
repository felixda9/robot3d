"""Run the simulation server (Milestone 2).

    uv run scripts/serve.py                   # quadruped at http://localhost:8000
    uv run scripts/serve.py --robot quadruped --port 8000

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
    parser.add_argument("--robot", default="quadruped", choices=available_robots())
    parser.add_argument("--keyframe", default="home", help="keyframe to start from and reset to")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices on your network")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(create_app(args.robot, args.keyframe), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
