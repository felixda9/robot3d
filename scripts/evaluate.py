"""Measure a trained policy headless: distance, speed, and falls per episode.

    uv run scripts/evaluate.py runs/<name>                       # newest checkpoint
    uv run scripts/evaluate.py runs/<name>/checkpoints/step_002000000.zip
    uv run scripts/evaluate.py runs/<name> --all                 # every checkpoint (learning progress)
"""

import argparse

from robot3d.policy import evaluate, find_checkpoint, list_checkpoints


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
        results = evaluate(cp, episodes=args.episodes)
        n = len(results)
        distance = sum(r["distance"] for r in results) / n
        speed = sum(r["speed"] for r in results) / n
        falls = sum(r["fell"] for r in results)
        ret = sum(r["return"] for r in results) / n
        print(f"{cp.label:>34} | {distance:7.2f} m | {speed:5.2f} m/s | {falls:2d}/{n:<2d} | {ret:7.1f}")


if __name__ == "__main__":
    main()
