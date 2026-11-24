"""Real-physics deployment/training mapping must have identical semantics."""
from dataclasses import asdict, replace

import numpy as np
import pytest

from dual_arm_conveyor import cli
from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.learning import PPOConfig, PPOTrainer, load_policy
from dual_arm_conveyor.scheduler import SchedulerEnv


def test_future_checkpoint_deployment_matches_training_through_physical_steps(tmp_path):
    config = Config(num_arms=4, strategy='adaptive', use_dispatch_preferences=False,
        sensor_mode='oracle', total_objects=8, pool_size=8, deep_bins=True,
        auto_unload=False, bin_capacity=180, future_planning=True,
        safe_preposition=True, belt_speed=.2)
    training_base, deployment = ConveyorEnv(config), ConveyorEnv(config)
    try:
        adapter = SchedulerEnv(training_base, teacher_mode='predictive')
        trainer = PPOTrainer(adapter, PPOConfig(steps=0, hidden_dim=8), asdict(config))
        path = tmp_path / 'future.pt'
        trainer.save(path)
        trained = load_policy(path)
        policy = cli._policy_action(str(path), deployment)
        adapter.reset(seed=701)
        deployment.reset(seed=701)
        for _ in range(20):
            encoded = adapter.observe()
            training_action = int(trained(encoded))
            native_action = policy(deployment.observe())
            expected = adapter.action_to_native(training_action)
            assert native_action == expected
            assert deployment.observe()['mask'][native_action]
            training_base.step(expected)
            deployment.step(native_action)
            np.testing.assert_array_equal(training_base.target, deployment.target)
            np.testing.assert_allclose(training_base.data.qpos, deployment.data.qpos, atol=1e-12)
            assert training_base.phase == deployment.phase
        deployment.config = replace(config, guarded_descent=True)
        with pytest.raises(ValueError, match='guarded_descent'):
            cli._policy_action(str(path), deployment)
        deployment.config = replace(config, future_planning=False, safe_preposition=False)
        with pytest.raises(ValueError, match='compatible four-arm'):
            cli._policy_action(str(path), deployment)
    finally:
        training_base.close()
        deployment.close()
