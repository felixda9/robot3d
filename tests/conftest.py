import pytest

from robot3d.training import PPOConfig, train


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """A real (but tiny) training run: 2 parallel envs, a few hundred steps,
    checkpoints along the way. Shared by every test that needs a checkpoint."""
    runs_dir = tmp_path_factory.mktemp("runs")
    return train(
        total_steps=256,
        n_envs=2,
        name="tiny",
        checkpoint_every=128,
        ppo=PPOConfig(n_steps=64, minibatches=2, n_epochs=1),
        runs_dir=runs_dir,
        verbose=0,
    )


def _tiny_task_run(tmp_path_factory, task: str):
    """A tiny real training run of the stand or get-up task (8-motor robot, for speed)."""
    from robot3d.walk import WalkConfig

    return train(
        total_steps=128,
        n_envs=2,
        name=f"tiny_{task}",
        checkpoint_every=64,
        walk=getattr(WalkConfig, task)(),
        ppo=PPOConfig(n_steps=64, minibatches=2, n_epochs=1),
        runs_dir=tmp_path_factory.mktemp(f"{task}_runs"),
        verbose=0,
    )


@pytest.fixture(scope="session")
def tiny_stand_run(tmp_path_factory):
    return _tiny_task_run(tmp_path_factory, "stand")


@pytest.fixture(scope="session")
def tiny_getup_run(tmp_path_factory):
    return _tiny_task_run(tmp_path_factory, "getup")
