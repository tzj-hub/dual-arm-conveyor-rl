"""Clearance-aware ordering must preserve the calibrated bin grid and layers.

These are numerical scheduling tests: no MuJoCo model or privileged object
state is present. Full trajectory/contact validation belongs to paired runs.
"""

from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.vision import Track


def make_env(arm):
    env = SimpleNamespace()
    env._deposit_position = MethodType(ConveyorEnv._deposit_position, env)
    env._choose_drop_slot = MethodType(ConveyorEnv._choose_drop_slot, env)
    env.num_arms = 4
    env.config = SimpleNamespace(
        nearest_bin_slot=True,
        cooperative_motion=False,
        cooperative_drop_cost=True,
        bin_capacity=40,
        deep_bins=True,
        belt_top=0.20,
        track_max_age=0.6,
        control_speed=0.65,
        direct_retask=True,
    )
    bases = np.array([[-0.6, -0.5, 0.0], [-0.6, 0.5, 0.0],
                      [0.6, -0.5, 0.0], [0.6, 0.5, 0.0]])
    homes = np.array([[x, -0.18 if i % 2 == 0 else 0.18, 0.38]
                      for i, x in enumerate(bases[:, 0])])
    env.ids = SimpleNamespace(
        bin_centers=np.array([[0.0, -0.28, 0.255], [0.0, 0.28, 0.255]]),
        arm_base_positions=bases,
        home_ee=homes,
        ee_sites=np.arange(4),
        bin_rim_z=0.255,
    )
    env.data = SimpleNamespace(time=1.0, site_xpos=homes.copy())
    env.approach_clearance = 0.13
    env.descent_time = 0.45
    env.closure_timeout = 0.40
    env.lift_height = 0.38
    env.transport_height = 0.325
    env.windows = np.array([[x - 0.3, x + 0.3] for x in bases[:, 0]])
    env.phase = ["idle"] * 4
    env.bin_service = ["closed"] * 2
    env.bin_drain_requested = [False] * 2
    env.bin_occupied = np.zeros((2, 40), dtype=bool)
    env.retry_after = np.zeros((4, 10))
    env.belief = np.zeros(10, dtype=int)
    env.tracks = [None] * 10
    for slot, color in enumerate((arm % 2, (arm ^ 1) % 2)):
        env.tracks[slot] = Track(
            track_id=100 + slot,
            position=np.array([bases[arm, 0] - 0.21, 0.0, 0.23]),
            velocity=np.array([0.2, 0.0, 0.0]),
            yaw=0.0,
            kind=color,
            confidence=1.0,
            timestamp=1.0,
            last_seen=1.0,
        )
        env.belief[slot] = 1
    env._position = lambda slot: env.tracks[slot].predict(env.data.time)
    env._grasp_offset = lambda track: 0.0
    env._intercept_motion_time = lambda start, target, velocity: 0.0
    return env


def layer(arm):
    return (arm // 2) * 20 + np.arange(10)


def chosen_y(env, arm, free=None):
    rank = env._choose_drop_slot(arm, 0, layer(arm) if free is None else free)
    return abs(env._deposit_position(arm, arm % 2, rank)[1])


@pytest.mark.parametrize("arm", range(4))
def test_waiting_partner_prefers_existing_outer_row_over_inner_settling_tail(arm):
    env = make_env(arm)
    # Own travel alone favors the inner .235 row. With actual visual demand
    # for the other hand, the existing .325 row avoids that settling tail.
    env.config.cooperative_drop_cost = False
    assert chosen_y(env, arm) == pytest.approx(0.235)
    env.config.cooperative_drop_cost = True
    assert chosen_y(env, arm) == pytest.approx(0.325)


@pytest.mark.parametrize("reason", [
    "missing", "reserved", "stale", "uncertain", "same_color", "other_station",
    "off_belt", "retry_wait", "busy_partner", "full_bin", "open_bin", "draining",
])
def test_no_usable_partner_demand_preserves_original_nearest_row(reason):
    arm = 2
    env = make_env(arm)
    pending = env.tracks[1]
    if reason == "missing":
        env.tracks[1] = None
    elif reason == "reserved":
        env.belief[1] = 2
    elif reason == "stale":
        env.tracks[1] = replace(pending, last_seen=0.0)
    elif reason == "uncertain":
        env.tracks[1] = replace(pending, confidence=0.1)
    elif reason == "same_color":
        env.tracks[1] = replace(pending, kind=0)
    elif reason == "other_station":
        env.tracks[1] = replace(pending, position=np.array([-0.6, 0.0, 0.23]))
    elif reason == "off_belt":
        env.tracks[1] = replace(pending, position=np.array([0.6, 0.2, 0.23]))
    elif reason == "retry_wait":
        env.retry_after[3, 1] = 2.0
    elif reason == "busy_partner":
        env.phase[3] = "transport"
    elif reason == "full_bin":
        env.bin_occupied[1, 20:] = True
    elif reason == "open_bin":
        env.bin_service[1] = "open"
    elif reason == "draining":
        env.bin_drain_requested[1] = True
    assert chosen_y(env, arm) == pytest.approx(0.235)


@pytest.mark.parametrize("phase", ["idle", "retreat", "return"])
def test_ready_or_returning_partner_can_benefit(phase):
    env = make_env(0)
    env.phase[1] = phase
    assert chosen_y(env, 0) == pytest.approx(0.325)


@pytest.mark.parametrize("arm", range(4))
def test_ordering_preserves_all_calibrated_slots_and_exhausts_layer_before_reuse(arm):
    env = make_env(arm)
    initial = (arm // 2) * 20 + np.arange(20)
    free = initial.copy()
    chosen = []
    for _ in initial:
        previous_free = free.copy()
        rank = env._choose_drop_slot(arm, 0, free)
        np.testing.assert_array_equal(free, previous_free)
        chosen.append(rank)
        free = free[free != rank]
    assert set(chosen[:10]) == set(initial[:10])
    assert set(chosen[10:]) == set(initial[10:])
    assert len(set(chosen)) == 20
    positions = np.array([env._deposit_position(arm, arm % 2, rank) for rank in chosen[:10]])
    assert len(set(map(tuple, positions))) == 10
    np.testing.assert_allclose(np.unique(np.abs(positions[:, 1])), [0.235, 0.325])
    np.testing.assert_allclose(
        np.unique(positions[:, 0] - env.ids.arm_base_positions[arm, 0]),
        [-0.18, -0.09, 0.0, 0.09, 0.18], atol=1e-15,
    )


def test_outer_slots_in_later_layer_do_not_displace_unfinished_inner_slots():
    env = make_env(0)
    free = np.array([7, 12])  # inner row of layer 0; outer row of layer 1
    assert env._choose_drop_slot(0, 0, free) == 7


def test_disabled_nearest_slot_keeps_fifo_even_with_partner_demand():
    env = make_env(0)
    env.config.nearest_bin_slot = False
    assert env._choose_drop_slot(0, 0, np.array([8, 2])) == 8
