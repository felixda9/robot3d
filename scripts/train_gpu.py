"""Train a walking policy on the GPU: MuJoCo Warp physics + PPO in PyTorch.

    uv run scripts/train_gpu.py                          # 50M steps, 4096 robots in parallel
    uv run scripts/train_gpu.py --steps 1e8 --envs 8192 --name fast_walk

Needs an NVIDIA GPU and the CUDA build of PyTorch. Watch it in the web UI's
Training tab (same metrics as CPU runs); replay any checkpoint from there, or
    uv run scripts/serve.py --policy runs/<name>
Ctrl+C stops early and still saves a final checkpoint.
"""

import argparse

import torch

from robot3d.gpu.ppo import GpuPPOConfig, train_gpu
from robot3d.robots import available_robots
from robot3d.runs import RUNS_DIR, default_run_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="quadruped", choices=available_robots())
    parser.add_argument("--steps", type=float, default=50e6, help="total environment steps (e.g. 5e7)")
    parser.add_argument("--envs", type=int, default=GpuPPOConfig.num_envs, help="robots simulated in parallel")
    parser.add_argument("--name", help="run folder name under runs/ (default: date_robot_walk_gpu)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=float, default=5e6, help="steps between checkpoints")
    parser.add_argument("--epochs", type=int, default=GpuPPOConfig.epochs, help="passes over each rollout")
    parser.add_argument("--minibatches", type=int, default=GpuPPOConfig.minibatches, help="minibatches per epoch")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "PyTorch can't see a CUDA GPU. GPU training needs an NVIDIA GPU and the CUDA build of "
            "PyTorch (see pyproject.toml, [tool.uv.sources])."
        )
    name = args.name or default_run_name(args.robot, "walk_gpu")
    print(f"Run folder: {RUNS_DIR / name}")
    print(f"Training {args.robot} for {int(args.steps):,} steps with {args.envs:,} robots on "
          f"{torch.cuda.get_device_name(0)}. First steps compile GPU kernels (~30 s, cached after).\n")
    run_dir = train_gpu(
        robot=args.robot,
        total_steps=int(args.steps),
        name=name,
        seed=args.seed,
        checkpoint_every=int(args.checkpoint_every),
        ppo=GpuPPOConfig(num_envs=args.envs, epochs=args.epochs, minibatches=args.minibatches),
    )
    print(f"\nDone. Checkpoints in {run_dir / 'checkpoints'}")
    print(f"Compare it with other runs in the web UI's Training tab, or watch it:")
    print(f"  uv run scripts/serve.py --policy {run_dir.relative_to(RUNS_DIR.parent)}")


if __name__ == "__main__":
    main()
