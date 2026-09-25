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
