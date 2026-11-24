"""Checkpoint schemas must match actual scheduling semantics before inference."""
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from dual_arm_conveyor import cli
from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.learning import PPOConfig, PPOTrainer, load_policy
from dual_arm_conveyor.scheduler import SchedulerEnv, joint_action_ids


@pytest.fixture
def env():
    base = ConveyorEnv(Config(
        num_arms=4, strategy="adaptive", use_dispatch_preferences=False,
        sensor_mode="oracle", total_objects=8, pool_size=8,
        deep_bins=True, auto_unload=False, bin_capacity=180,
    ))
    yield SchedulerEnv(base)
    base.close()


def test_saved_scheduler_round_trips_through_playback_and_rejects_bad_layout(env, tmp_path):
    trainer = PPOTrainer(env, PPOConfig(steps=0, hidden_dim=8), asdict(env.config))
    path = tmp_path / "scheduler.pt"
    trainer.save(path)
    learned = load_policy(path)
    assert learned.metadata["scheduler_action_metadata"] == env.action_metadata()
    native_env = env.env
    action = cli._policy_action(str(path), native_env)(native_env.observe())
    assert action in joint_action_ids()
    assert native_env.observe()["mask"][action]
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["scheduler_action_metadata"]["schema"] = "obsolete_encoding"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="compatible four-arm"):
        cli._policy_action(str(path), native_env)


def test_soft_mask_configuration_cannot_silently_change_at_deployment(env, tmp_path):
    trainer = PPOTrainer(env, PPOConfig(steps=0, hidden_dim=8), asdict(env.config))
    path = tmp_path / "scheduler.pt"
    trainer.save(path)
    env.env.config = replace(env.env.config, use_dispatch_preferences=True)
    with pytest.raises(ValueError, match="dispatch preferences differ"):
        cli._policy_action(str(path), env.env)


def test_same_native_mask_survives_cli_scene_resolution():
    args = cli.parser().parse_args([
        "evaluate", "--num-arms", "4", "--strategy", "adaptive",
        "--no-dispatch-preferences", "--objects", "8", "--episodes", "1",
    ])
    config = cli.config_from_args(args)
    assert not config.use_dispatch_preferences
    assert np.isclose(config.decision_dt, .1)
