"""Real two-lane births, deterministic demand and non-overlapping admission."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv, State
from dual_arm_conveyor.supply import LaneSupply


def configuration(**changes):
    values = dict(
        num_arms=4, strategy="adaptive", gripper_mode="fingers", sensor_mode="oracle",
        belt_speed=0.15, arrival_mode="distance", supply_lanes=2,
        spacing=0.60, spacing_jitter=0.18, lane_offset=0.065, lane_jitter=0.012,
        physics_dt=0.0025, bin_capacity=24,
    )
    values.update(changes)
    return Config(**values)


@pytest.mark.parametrize("changes", [
    {"supply_lanes": 0}, {"supply_lanes": 3}, {"supply_lanes": True},
    {"supply_lanes": 1.0}, {"lane_offset": 0}, {"lane_jitter": -0.01},
    {"lane_jitter": float("nan")}, {"lane_jitter": .065},
    {"lane_offset": .14}, {"arrival_mode": "time"},
])
def test_invalid_lane_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        configuration(**changes)


def test_initial_birth_is_genuinely_side_by_side_and_has_no_object_overlap():
    env = ConveyorEnv(configuration(), seed=173)
    try:
        assert env.spawned == 2 and env.vacuum is None
        positions = np.array(env.spawn_position_log)
        assert positions[0, 0] == positions[1, 0]
        assert positions[0, 1] < -.05 and positions[1, 1] > .05
        assert list(env.spawn_lane_log) == [0, 1]
        assert env.arrivals[0] == env.arrivals[1] == 0
        objects = set(map(int, env.ids.cube_geoms))
        for contact in env.data.contact[:env.data.ncon]:
            assert not (int(contact.geom1) in objects and int(contact.geom2) in objects)
        assert env.metrics()["supply"]["admitted_per_lane"] == [1, 1]
    finally:
        env.close()


def test_spatial_schedule_is_independent_of_poll_rate_and_backpressure():
    c = configuration()
    dense, sparse = LaneSupply(c, 13), LaneSupply(c, 13)
    seen = [[], []]
    for distance in np.linspace(0, 12, 1201):
        dense.advance(distance)
        for lane in range(2):
            while dense.pending[lane]:
                seen[lane].append(dense.accepted(lane))
    sparse.advance(12)
    for lane in range(2):
        assert list(sparse.pending[lane]) == seen[lane]
        gaps = np.diff([a.distance for a in seen[lane]])
        assert len(gaps) > 15 and np.ptp(gaps) > .20
        assert gaps.min() >= .42 and gaps.max() <= .78
    assert not np.allclose(
        [a.distance for a in seen[0]][:10], [a.distance for a in seen[1]][:10]
    )
    np.testing.assert_array_equal(dense.offered, sparse.offered)


def test_blocked_birth_is_queued_without_resampling_or_overlapping_existing_objects():
    env = ConveyorEnv(configuration(), seed=31)
    try:
        env.data.time = .81 / env.config.belt_speed
        env.belt_sample = env.belt_motion.sample(env.data.time)
        before = env.data.qpos.copy()
        env._spawn()
        assert env.spawned == 2
        np.testing.assert_array_equal(before, env.data.qpos)
        source = env.supply_source
        queued = [tuple(q) for q in source.pending]
        assert all(queued)
        env._spawn()
        assert queued == [tuple(q) for q in source.pending]
        assert env.metrics()["supply"]["inlet_blocked_lane_steps"]["object"] == 4
        for q in env.ids.cube_qpos[:2]:
            env.data.qpos[q] += .35
        mujoco.mj_forward(env.model, env.data)
        env._spawn()
        assert env.spawned == 4
        m = env.metrics()["supply"]
        assert m["offered"] == m["admitted"] + m["queued_upstream"] + m["overflow"]
        assert m["max_wait_distance_m"] > 0
        assert m["queued_upstream"] == 0 and m["overflow"] == 0
        for slot, lane in enumerate(list(env.spawn_lane_log)[-2:], start=2):
            arrival = queued[lane][0]
            assert env.yaws[slot] == arrival.yaw
            assert env.lateral[slot] == arrival.lateral
            assert env.types[slot] == arrival.kind
    finally:
        env.close()


def test_queue_overflow_is_explicit_and_finite_requests_do_not_offer_extra_objects():
    c = configuration()
    blocked = LaneSupply(c, 99)
    blocked.advance(100)
    report = blocked.diagnostics(100 / .15)
    assert report["queued_upstream"] == report["queue_capacity"]
    assert report["overflow"] > 0
    assert report["offered"] == report["queued_upstream"] + report["overflow"]
    finite = LaneSupply(replace(c, total_objects=5), 99)
    finite.advance(100)
    assert finite.offered.sum() == 5
    assert sum(map(len, finite.pending)) == 5
    finite.advance(200)
    assert finite.offered.sum() == 5


def test_single_requested_object_does_not_force_a_second_lane_birth():
    env = ConveyorEnv(configuration(total_objects=1), seed=73)
    try:
        assert env.spawned == env.supply_source.offered.sum() == 1
        assert env.metrics()["supply"]["queued_upstream"] == 0
        assert env.metrics()["supply"]["offered_per_minute"] == 0
        env.data.time = 10
        env.belt_sample = env.belt_motion.sample(10)
        env._spawn()
        assert env.spawned == env.supply_source.offered.sum() == 1
    finally:
        env.close()


def test_finite_batch_larger_than_pool_retains_all_requests_under_backpressure():
    supply = LaneSupply(configuration(total_objects=150), 101)
    supply.advance(100)
    report = supply.diagnostics(100 / .15)
    assert report["offered"] == report["queued_upstream"] == 150
    assert report["admitted"] == report["overflow"] == 0
    for lane in range(2):
        while supply.pending[lane]:
            supply.accepted(lane)
    supply.advance(200)
    report = supply.diagnostics(200 / .15)
    assert report["offered"] == report["admitted"] == 150
    assert report["queued_upstream"] == report["overflow"] == 0


def test_admission_envelope_respects_arbitrary_yaw_and_existing_roll():
    env = ConveyorEnv(configuration(), seed=81)
    try:
        # A still-present birth blocks its own lane across all proposed yaws.
        lateral = env.lateral[0]
        for yaw in np.linspace(-np.pi, np.pi, 19):
            assert env._inlet_obstruction(2, yaw, lateral) == "object"
        assert env._inlet_obstruction(2, np.pi / 4, .15) == "edge"
        q = env.ids.cube_qpos[0]
        env.data.qpos[q + 3:q + 7] = [np.cos(.4), np.sin(.4), 0, 0]
        assert env._inlet_obstruction(2, 0, lateral) == "object"
    finally:
        env.close()


def test_wait_policy_stream_has_dense_irregular_lanes_without_blocked_births():
    env = ConveyorEnv(configuration(max_time=20), seed=91)
    try:
        while not env._done:
            env.step(env.wait_action)
        m = env.metrics()
        source = m["supply"]
        assert source["offered"] == source["admitted"] == m["total_objects"]
        assert source["queued_upstream"] == source["overflow"] == 0
        assert min(source["admitted_per_lane"]) >= 4
        assert m["supply_blocked_steps"] == 0
        assert set(m["spawn_lanes"]) == {0, 1}
        assert all(abs(p[1]) < .08 for p in m["spawn_positions"])
        assert m["total_objects"] == sum(m[k] for k in ("sorted", "missed", "dropped", "unfinished"))
        assert m["sorted"] == 0 and m["dropped"] == 0
        assert m["total_objects"] > 20 * .15 / .45  # denser than old single lane
        assert np.count_nonzero(env.states == State.BELT) > 0
    finally:
        env.close()
