"""Partial unloading drains admitted work before physically reopening capacity."""

import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.vision import Track


@pytest.fixture
def env():
    simulation = ConveyorEnv(
        Config(
            num_arms=4,
            strategy="adaptive",
            sensor_mode="oracle",
            gripper_mode="vacuum",
            bin_capacity=24,
            physics_dt=0.0025,
            belt_speed=0.3,
            control_speed=0.65,
        ),
        seed=95000,
    )
    yield simulation
    simulation.close()


def reserve_partial_bin(env, color=0, count=18):
    # Scheduler reservations represent completed slots and admitted tasks;
    # no physical objects are fabricated or removed by this unit fixture.
    env.bin_occupied[color, :count] = True
    env.bin_slots[color] = count


def track(env, slot, color, x):
    env.tracks[slot] = Track(
        95000 + slot,
        np.array([x, 0.0, 0.23]),
        np.array([0.3, 0.0, 0.0]),
        0.0,
        color,
        1.0,
        float(env.data.time),
        float(env.data.time),
        10,
        True,
    )
    env.belief[slot] = 1


def test_request_stops_new_color_dispatch_but_admitted_task_can_place(env, monkeypatch):
    reserve_partial_bin(env)
    track(env, 0, 0, -0.55)
    track(env, 1, 0, 0.15)
    track(env, 2, 1, 0.15)
    env.target[0] = 0
    env.drop_kind[0] = 0
    env.belief[0] = 3
    env.phase[0] = "transfer_wait"
    env.holding[0] = True
    env.idle_anchors[0] = env.data.site_xpos[env.ids.ee_sites[0]].copy()
    assert env._candidate(2, 1)[0]
    assert env._candidate(3, 2)[0]
    monkeypatch.setattr(env.ik, "command", lambda *args, **kwargs: None)
    monkeypatch.setattr(env, "_gripper_has_object", lambda arm: True)

    env._control()

    assert env.bin_drain_requested[0]
    assert env.bin_service[0] == "closed"
    assert not env._candidate(2, 1)[0]
    assert env._candidate(3, 2)[0]  # The other bin continues accepting work.
    assert env.phase[0] == "transport"
    assert env.bin_zone_owner[0, 0] == 0
    assert env.target[0] == 0  # Existing load retains its placement reservation.


@pytest.mark.parametrize("arm", range(4))
def test_every_pending_task_blocks_opening_until_finished(env, arm):
    reserve_partial_bin(env)
    env.target[arm] = 0
    env.drop_kind[0] = 0
    env._service_bins()
    assert env.bin_drain_requested[0]
    assert env.bin_service[0] == "closed"
    env.target[arm] = -1
    env._service_bins()
    assert env.bin_service[0] == "opening"


@pytest.mark.parametrize("zone", range(2))
def test_every_zone_owner_blocks_opening_even_if_shared_owner_was_released(env, zone):
    reserve_partial_bin(env)
    env.bin_zone_owner[zone, 0] = 2 * zone
    env.bin_owner[0] = -1
    env._service_bins()
    assert env.bin_drain_requested[0]
    assert env.bin_service[0] == "closed"
    env.bin_zone_owner[zone, 0] = -1
    env._service_bins()
    assert env.bin_service[0] == "opening"


def test_partial_drain_keeps_last_release_settling_time(env):
    reserve_partial_bin(env)
    env.bin_last_release[0] = env.data.time
    env._service_bins()
    assert env.bin_drain_requested[0] and env.bin_service[0] == "closed"
    env.data.time = 1.5
    env._service_bins()
    assert env.bin_service[0] == "closed"
    env.data.time += env.config.physics_dt
    env._service_bins()
    assert env.bin_service[0] == "opening"


def test_capacity_and_request_stay_locked_until_physical_gate_closes(env):
    reserve_partial_bin(env)
    before_poses = env.data.qpos[env.ids.cube_qpos].copy()
    env._service_bins()
    assert env.bin_service[0] == "opening"
    qpos = env.ids.bin_gate_qpos[0]
    actuator = env.ids.bin_gate_actuators[0]
    env.data.qpos[qpos] = env.ids.bin_gate_open_position
    env._service_bins()
    assert env.bin_service[0] == "draining"
    env.data.time += 2.01
    env._service_bins()
    assert env.bin_service[0] == "closing"
    env.data.ctrl[actuator] = 0.0
    env.data.qpos[qpos] = 0.02  # Close command alone must never free capacity.
    env._service_bins()
    assert env.bin_service[0] == "closing"
    assert env.bin_drain_requested[0] and env.bin_slots[0] == 18
    assert env.bin_occupied[0].sum() == 18 and env.bin_unloads[0] == 0
    env.data.qpos[qpos] = 0.0
    env._service_bins()
    assert env.bin_service[0] == "closed"
    assert not env.bin_drain_requested[0]
    assert env.bin_slots[0] == 0 and not env.bin_occupied[0].any()
    assert env.bin_unloads[0] == 1
    np.testing.assert_array_equal(env.data.qpos[env.ids.cube_qpos], before_poses)


@pytest.mark.parametrize("occupied", [12, 17])
def test_full_station_below_three_quarters_does_not_request_drain(env, occupied):
    reserve_partial_bin(env, count=occupied)
    env._service_bins()
    assert not env.bin_drain_requested.any()
    assert env.bin_service == ["closed", "closed"]


def test_three_quarters_without_full_station_does_not_request_drain(env):
    env.bin_occupied[0, :9] = True
    env.bin_occupied[0, 12:21] = True
    env.bin_slots[0] = 18
    env._service_bins()
    assert not env.bin_drain_requested.any()
    assert env.bin_service == ["closed", "closed"]


@pytest.mark.parametrize("arms,auto_unload", [(4, False), (2, True), (2, False)])
def test_partial_unload_is_opt_in_to_four_arm_automatic_service(arms, auto_unload):
    env = ConveyorEnv(
        Config(
            num_arms=arms,
            strategy="adaptive",
            sensor_mode="oracle",
            bin_capacity=12,
            auto_unload=auto_unload,
        ),
        seed=95001,
    )
    try:
        reserve_partial_bin(env, count=9)
        env._service_bins()
        assert not env.bin_drain_requested.any()
        assert env.bin_service == ["closed", "closed"]
        env.bin_occupied[0] = True
        env.bin_slots[0] = 12
        env._service_bins()
        assert env.bin_service[0] == ("opening" if auto_unload else "closed")
    finally:
        env.close()
