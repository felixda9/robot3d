"""Train a walking policy with PPO.

    uv run scripts/train.py                          # 10M steps (~30 min on 8 cores/16 threads)
    uv run scripts/train.py --steps 2e6 --name first_try
    uv run scripts/train.py --envs 8 --seed 1

Watch it learn:     uv run tensorboard --logdir runs     (then http://localhost:6006)
Watch it walk:      uv run scripts/serve.py --policy runs/<name>
Measure it:         uv run scripts/evaluate.py runs/<name>

Ctrl+C stops early and still saves a final checkpoint.
"""

import argparse

from robot3d.robots import available_robots
from robot3d.training import RUNS_DIR, default_num_envs, default_run_name, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="quadruped", choices=available_robots())
    parser.add_argument("--steps", type=float, default=10e6, help="total environment steps (e.g. 1e7)")
    parser.add_argument("--envs", type=int, default=default_num_envs(),
                        help="parallel environments (default: logical CPU threads - 2)")
    parser.add_argument("--name", help="run folder name under runs/ (default: date_robot_walk)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=float, default=500_000, help="steps between checkpoints")
    args = parser.parse_args()

    name = args.name or default_run_name(args.robot)
    print(f"Run folder: {RUNS_DIR / name}")
    print(f"Training {args.robot} for {int(args.steps):,} steps with {args.envs} parallel environments.")
    print("TensorBoard: uv run tensorboard --logdir runs   (open http://localhost:6006)\n")

    run_dir = train(
        robot=args.robot,
        total_steps=int(args.steps),
        n_envs=args.envs,
        name=name,
        seed=args.seed,
        checkpoint_every=int(args.checkpoint_every),
    )
    print(f"\nDone. Checkpoints in {run_dir / 'checkpoints'}")
    print(f"Watch it:   uv run scripts/serve.py --policy {run_dir.relative_to(RUNS_DIR.parent)}")
    print(f"Measure it: uv run scripts/evaluate.py {run_dir.relative_to(RUNS_DIR.parent)}")


# The guard matters on Windows: each parallel environment runs in a new Python
# process that re-imports this file, and must not start another training.
if __name__ == "__main__":
    main()
