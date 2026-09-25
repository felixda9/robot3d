"""Run folders as the dashboard sees them (runs.py)."""

import json
import os
import time

import pytest

from robot3d.runs import (
    ScalarReader,
    list_checkpoints,
    list_run_dirs,
    resolve_checkpoint,
    resolve_run,
    run_detail,
    run_summary,
    save_evaluation,
)


def test_summary_and_detail(tiny_run):
    summary = run_summary(tiny_run)
    assert summary.name == "tiny" and summary.status == "finished"
    assert summary.checkpoints == len(list_checkpoints(tiny_run)) >= 2
    assert list_run_dirs(tiny_run.parent) == [tiny_run]

    detail = run_detail(tiny_run)
    keys = {(s.group, s.key) for s in detail.settings}
    assert ("walk", "action_scale") in keys and ("ppo", "learning_rate") in keys and ("run", "n_envs") in keys


def test_crashed_run_counts_as_stopped(tiny_run, tmp_path):
    info = json.loads((tiny_run / "run.json").read_text())
    for key in ("finished", "interrupted"):
        info.pop(key)
    crashed = tmp_path / "crashed"
    (crashed / "tb").mkdir(parents=True)
    (crashed / "run.json").write_text(json.dumps(info))
    assert run_summary(crashed).status == "running"  # just written: looks alive
    old = time.time() - 600
    os.utime(crashed / "run.json", (old, old))
    assert run_summary(crashed).status == "stopped"  # silent for 10 minutes: crashed


def test_scalars_from_tensorboard(tiny_run):
    series = ScalarReader().read(tiny_run, ["rollout/ep_len_mean", "train/std", "no/such_tag"])
    by_tag = {s.tag: s for s in series}
    assert len(by_tag["train/std"].steps) >= 1
    assert by_tag["train/std"].steps == sorted(by_tag["train/std"].steps)
    assert by_tag["no/such_tag"].steps == []


def test_evaluation_cache(tiny_run):
    checkpoint = list_checkpoints(tiny_run)[0]
    results = [
        {"distance": 1.0, "speed": 0.05, "fell": False, "return": 10.0},
        {"distance": 3.0, "speed": 0.15, "fell": True, "return": 30.0},
    ]
    info = save_evaluation(checkpoint, results)
    assert (info.episodes, info.distance, info.falls, info.mean_return) == (2, 2.0, 1, 20.0)
    assert checkpoint.evaluation() == info
    checkpoint.eval_path.unlink()
    assert checkpoint.evaluation() is None


@pytest.mark.parametrize("name", ["..", "../secret", "a/b", "a\\b", "", ".hidden"])
def test_names_never_escape_the_runs_folder(tiny_run, name):
    with pytest.raises(ValueError):
        resolve_run(name, tiny_run.parent)
    with pytest.raises(ValueError):
        resolve_checkpoint(tiny_run, name)
