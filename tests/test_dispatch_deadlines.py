"""Acceleration-aware dispatch and last-station deadline coordination."""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.vision import Track


@pytest.mark.parametrize("distance", [0.01, 0.10, 0.50])
def test_stationary_interception_accounts_for_start_and_braking(distance):
    env = SimpleNamespace(
        config=SimpleNamespace(control_speed=0.65),
        ik=SimpleNamespace(max_cartesian_acceleration=2.0),
    )
    duration = ConveyorEnv._intercept_motion_time(
        env, [0, 0, 0], [distance, 0, 0], [0, 0, 0]
    )
    speed = 0.85 * 0.65
    expected = (
        2 * math.sqrt(distance / 2.0)
        if distance <= speed**2 / 2.0
        else distance / speed + speed / 2.0
    )
    assert duration == pytest.approx(expected)
    assert duration > distance / speed


def test_faster_receding_target_is_not_dispatched():
    env = SimpleNamespace(
        config=SimpleNamespace(control_speed=0.65),
        ik=SimpleNamespace(max_cartesian_acceleration=2.0),
    )
    assert math.isinf(
        ConveyorEnv._intercept_motion_time(env, [0, 0, 0], [0.2, 0, 0], [0.6, 0, 0])
    )


@pytest.fixture
def env():
    instance = ConveyorEnv(
        Config(
            num_arms=4,
            strategy="adaptive",
            sensor_mode="oracle",
            belt_speed=0.15,
            control_speed=0.65,
            supply_lanes=2,
            spacing=0.60,
            spacing_jitter=0.18,
            min_object_gap=0.08,
            bin_capacity=24,
        )
    )
    instance.tracks = [None] * 10
    instance.belief[:] = 0
    yield instance
    instance.close()


def target(env, slot, color, x, y=0.065):
    item = Track(
        9000 + slot,
        np.array([x, y, 0.23]),
        np.array([0.15, 0, 0]),
        np.pi / 2,
        color,
        1.0,
        0.0,
        0.0,
        10,
        True,
    )
    env.tracks[slot] = item
    env.belief[slot] = 1
    return item


def test_last_station_defers_fresh_work_for_returning_partner_deadline(env):
    env.phase[3] = "return"
    target(env, 0, 1, 0.28)
    assert env._returning_partner_has_urgent_work(2, 1.8, 6.0)
    env.phase[3] = "idle"
    assert not env._returning_partner_has_urgent_work(2, 1.8, 6.0)


def test_upstream_station_does_not_sacrifice_throughput_for_partner_return(env):
    env.phase[1] = "return"
    target(env, 0, 1, -0.6)
    assert not env._returning_partner_has_urgent_work(0, 1.0, 6.0)


@pytest.mark.parametrize(
    "condition", ["stale", "expired", "full_bin", "wrong_color", "reserved"]
)
def test_unpickable_partner_target_does_not_starve_idle_hand(env, condition):
    env.phase[3] = "return"
    item = target(env, 0, 1, 0.28)
    if condition == "stale":
        env.tracks[0] = replace(item, last_seen=-1.0)
    elif condition == "expired":
        item.position[0] = 0.81
    elif condition == "full_bin":
        env.bin_occupied[1] = True
    elif condition == "wrong_color":
        env.tracks[0] = replace(item, kind=0)
    elif condition == "reserved":
        env.belief[0] = 2
    assert not env._returning_partner_has_urgent_work(2, 1.8, 6.0)


def test_urgent_current_task_keeps_station_access(env):
    env.phase[3] = "return"
    target(env, 0, 1, 0.28)
    assert not env._returning_partner_has_urgent_work(2, 1.8, 2.4)


def test_learned_scheduler_can_ignore_soft_priority_but_not_physical_limits(env):
    target(env, 0, 0, 0.10, y=-0.065)
    env._returning_partner_has_urgent_work = lambda *_: True
    assert not env._candidate(2, 0)[0]
    env.config = replace(env.config, use_dispatch_preferences=False)
    assert env._candidate(2, 0)[0]
    env.phase[3] = "descend"
    assert not env._candidate(2, 0)[0]
    env.phase[3] = "idle"
    env.tracks[0].position[0] = 0.81
    assert not env._candidate(2, 0)[0]
