"""Scheduler feedback includes new environment contacts, not historical ones."""

import pytest
from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv


@pytest.mark.parametrize("robot_steps,arm_steps,penalty", [(1, 0, 0.25), (5, 0, 0.25), (1, 1, 0.35)])
def test_new_collision_feedback_is_once_per_decision(monkeypatch, robot_steps, arm_steps, penalty):
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=2))
    try:
        monkeypatch.setattr(env, "_collisions", lambda: None)
        env.reset(seed=73)
        baseline = [env.step(env.wait_action)[1] for _ in range(2)]
        env.reset(seed=73)
        injected = False

        def contact_counter():
            nonlocal injected
            if not injected:
                env.robot_collision_steps += robot_steps
                env.collision_steps += arm_steps
                injected = True

        monkeypatch.setattr(env, "_collisions", contact_counter)
        first = env.step(env.wait_action)[1]
        second = env.step(env.wait_action)[1]
        assert first == pytest.approx(baseline[0] - penalty)
        assert second == pytest.approx(baseline[1])
        assert env.robot_collision_steps == robot_steps
    finally:
        env.close()
