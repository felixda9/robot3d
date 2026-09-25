"""Train a walking policy on the GPU: MuJoCo Warp physics + PPO in PyTorch.

    uv run scripts/train_gpu.py                          # our PPO, 50M steps, 4096 robots
    uv run scripts/train_gpu.py --trainer rsl            # RSL-RL's PPO (legged_gym recipe)
    uv run scripts/train_gpu.py --steps 1e8 --envs 8192 --name fast_walk
    uv run scripts/train_gpu.py --gait-hz 1.5            # slower stepping rhythm (0 = no gait clock)
    uv run scripts/train_gpu.py --task stand --robot quadruped12   # stand still, catch shoves
    uv run scripts/train_gpu.py --task getup --robot quadruped12   # get up after falls
    uv run scripts/train_gpu.py --robot quadruped12 --init runs/walk12_tidy   # fine-tune a walker

Needs an NVIDIA GPU and the CUDA build of PyTorch. Watch it in the web UI's
Training tab (same metrics as CPU runs); replay any checkpoint from there, or
    uv run scripts/serve.py --policy runs/<name>
Ctrl+C stops early and still saves a final checkpoint.
"""

import argparse
import dataclasses

import torch

from robot3d.gpu.ppo import GpuPPOConfig, train_gpu
from robot3d.gpu.rsl import RslConfig, train_rsl
from robot3d.robots import available_robots
from robot3d.runs import RUNS_DIR, default_run_name
from robot3d.walk import WalkConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trainer", choices=["ppo", "rsl"], default="ppo",
                        help="ppo = our PPO (gpu/ppo.py); rsl = RSL-RL's reference PPO (gpu/rsl.py)")
    parser.add_argument("--task", choices=["walk", "steer", "stand", "getup", "jump"], default="walk",
                        help="walk: the trot-walk; steer: walking that follows steering commands; stand: stand "
                             "still, catch shoves; getup: get up after falls; jump: one high jump, then stand")
    parser.add_argument("--robot", default="quadruped", choices=available_robots())
    parser.add_argument("--steps", type=float, default=50e6, help="total environment steps (e.g. 5e7)")
    parser.add_argument("--envs", type=int, default=4096, help="robots simulated in parallel")
    parser.add_argument("--name", help="run folder name under runs/ (default: date_robot_walk_gpu)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=float, default=5e6, help="steps between checkpoints")
    parser.add_argument("--epochs", type=int, help="passes over each rollout (default: trainer's own)")
    parser.add_argument("--minibatches", type=int, help="minibatches per epoch (default: trainer's own)")
    parser.add_argument("--init", help="fine-tune from this run (newest checkpoint) or .pt checkpoint (our PPO)")
    parser.add_argument("--gait-hz", type=float,
                        help=f"gait clock: steps per foot per second (default {WalkConfig.gait_frequency}; 0 = no clock)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "PyTorch can't see a CUDA GPU. GPU training needs an NVIDIA GPU and the CUDA build of "
            "PyTorch (see pyproject.toml, [tool.uv.sources])."
        )
    suffix = f"{args.task}_rsl" if args.trainer == "rsl" else f"{args.task}_gpu"  # (steer runs are "walk" tasks)
    name = args.name or default_run_name(args.robot, suffix)
    print(f"Run folder: {RUNS_DIR / name}")
    print(f"Training {args.robot} with {args.trainer} for {int(args.steps):,} steps, {args.envs:,} robots on "
          f"{torch.cuda.get_device_name(0)}. First steps compile GPU kernels (~30 s, cached after).\n")
    overrides = {k: v for k, v in (("epochs", args.epochs), ("minibatches", args.minibatches)) if v is not None}
    walk = {"walk": WalkConfig, "steer": WalkConfig.steer, "stand": WalkConfig.stand, "getup": WalkConfig.getup,
            "jump": WalkConfig.jump}[args.task]()
    walk = walk.for_robot(args.robot)  # e.g. the humanoid's own settings (walk.ROBOT_SETTINGS)
    if args.gait_hz is not None:
        walk = dataclasses.replace(walk, gait_frequency=args.gait_hz)
    common = dict(robot=args.robot, total_steps=int(args.steps), name=name, seed=args.seed,
                  checkpoint_every=int(args.checkpoint_every), walk=walk)
    if args.trainer == "rsl":
        if args.init:
            raise SystemExit("--init works with --trainer ppo only")
        run_dir = train_rsl(**common, rsl=RslConfig(num_envs=args.envs, **overrides))
    else:
        init_from = None
        if args.init:
            from robot3d.runs import find_checkpoint

            init_from = find_checkpoint(args.init).model_path
            print(f"Starting from {init_from}")
        run_dir = train_gpu(**common, ppo=GpuPPOConfig(num_envs=args.envs, **overrides), init_from=init_from)
    print(f"\nDone. Checkpoints in {run_dir / 'checkpoints'}")
    print("Compare it with other runs in the web UI's Training tab, or watch it:")
    print(f"  uv run scripts/serve.py --policy {run_dir.relative_to(RUNS_DIR.parent)}")


if __name__ == "__main__":
    main()
