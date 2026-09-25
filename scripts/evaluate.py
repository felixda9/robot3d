"""Measure a trained policy headless: distance, speed, and falls per episode.

    uv run scripts/evaluate.py runs/<name>                       # newest checkpoint
    uv run scripts/evaluate.py runs/<name>/checkpoints/step_002000000.zip
    uv run scripts/evaluate.py runs/<name> --all                 # every checkpoint (learning progress)

Results are also saved next to each checkpoint (step_..._eval.json), where the
training dashboard shows them.
"""

import argparse

from robot3d.policy import evaluate
from robot3d.runs import find_checkpoint, list_checkpoints, save_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="run folder or checkpoint .zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--all", action="store_true", help="evaluate every checkpoint of the run")
    args = parser.parse_args()

    checkpoint = find_checkpoint(args.path)
    checkpoints = list_checkpoints(checkpoint.run_dir) if args.all else [checkpoint]
    print(f"{'checkpoint':>34} | {'distance':>9} | {'speed':>9} | {'falls':>5} | {'return':>7}")
    for cp in checkpoints:
        info = save_evaluation(cp, evaluate(cp, episodes=args.episodes))
        print(
            f"{cp.label:>34} | {info.distance:7.2f} m | {info.speed:5.2f} m/s | "
            f"{info.falls:2d}/{info.episodes:<2d} | {info.mean_return:7.1f}"
        )


if __name__ == "__main__":
    main()
