"""Lifetime supply, physical hopper service, and bounded streaming state.

The WAIT-policy integration cases exercise the actual belt and recycling;
they deliberately make no claim about sorting-policy success. Sorting and
robot collision regressions live in test_environment.py.
"""

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv, State
from dual_arm_conveyor.ik import downward_quaternion


def _assert_accounting(env, metrics):
    assert metrics["total_objects"] == sum(
        metrics[key] for key in ("sorted", "missed", "dropped", "unfinished")
    )
    assert metrics["total_objects"] == env.recycled + np.count_nonzero(
        env.states != State.PENDING
    )
    assert metrics["sorted"] == sum(metrics["arm_sorted"])
    resolved = metrics["sorted"] + metrics["missed"] + metrics["dropped"]
    assert metrics["completed_success_rate"] == pytest.approx(
        metrics["sorted"] / max(resolved, 1)
    )


def test_only_present_objects_enter_fixed_controller_slots():
    env = ConveyorEnv(Config(sensor_mode="oracle"))
    try:
        assert env.spawned == 1
        assert len(env.ids.cube_bodies) == 64
        assert len(env.tracks) == len(env.belief) == 10
        assert len(env.track_slots) == 1
        assert list(env.track_slots) == [0]
        assert np.count_nonzero(env.states == State.PENDING) == 63
        assert len(env.arrivals) == 1
        obs = env.observe()
        assert obs["features"].shape == (121, 22)
        assert obs["global"].shape == (20,)
        assert obs["mask"].shape == (121,)
    finally:
        env.close()


def test_random_color_supply_is_not_forced_to_balance_each_batch():
    env = ConveyorEnv(Config(sensor_mode="oracle"))
    try:
        red_counts = []
        for seed in range(8):
            env.reset(seed=seed)
            for _ in range(9):
                env.data.time = env.next_arrival
                env._spawn()
            red_counts.append(np.count_nonzero(env.types[:10] == 0))
        assert len(set(red_counts)) > 1
        assert any(count != 5 for count in red_counts)
    finally:
        env.close()


@pytest.mark.parametrize(
    "state", [State.BELT, State.RESERVED, State.HELD, State.RELEASED]
)
def test_recycling_cannot_remove_an_unresolved_object(state):
    env = ConveyorEnv(Config(sensor_mode="oracle"))
    try:
        env.states[0] = state
        q = env.ids.cube_qpos[0]
        env.data.qpos[q : q + 3] = [env.config.belt_half_length + 0.1, 0, 0.01]
        mujoco.mj_forward(env.model, env.data)
        before = env.data.qpos[q : q + 7].copy()
        env._recycle()
        assert env.states[0] == state and env.serial[0] == 0
        assert env.recycled == 0
        np.testing.assert_array_equal(env.data.qpos[q : q + 7], before)
    finally:
        env.close()


def test_continuous_supply_passes_entire_pool_without_reset():
    # Dense supply speeds up this no-pick lifecycle test without changing dt,
    # contact physics, or the belt speed. New objects still arrive >=1.2 s apart.
    env = ConveyorEnv(Config(sensor_mode="oracle", spacing=0.12, arrival_jitter=0.1))
    try:
        initial_model = env.model
        initial_data = env.data
        for step in range(1200):
            obs, _, terminated, truncated, metrics = env.step(120)
            assert not terminated and not truncated
            _assert_accounting(env, metrics)
            assert len(env.track_slots) <= 10
            assert len(env.retired_tracks) < 20
            if env.spawned > env.config.pool_size and env.generations.max() > 1:
                break
        else:
            pytest.fail("Continuous supply failed to reuse its physical pool")
        assert env.model is initial_model and env.data is initial_data
        assert metrics["continuous"] and metrics["requested_objects"] is None
        assert metrics["termination_reason"] == "running"
        assert metrics["recycled_objects"] > 0
        assert metrics["missed"] > 0 and metrics["sorted"] == 0
        assert metrics["supply_blocked_steps"] == 0
        assert obs["features"].shape == (121, 22)
        serials = env.serial[env.serial >= 0]
        assert len(serials) == len(set(serials))
        assert serials.max() >= 64
        assert all(track.track_id >= 0 for track in env.tracks if track is not None)
    finally:
        env.close()


def test_finite_fifty_object_supply_finishes_with_exact_lifetime_counts():
    env = ConveyorEnv(
        Config(sensor_mode="oracle", total_objects=50, spacing=0.12, arrival_jitter=0.1)
    )
    try:
        for _ in range(1100):
            _, _, terminated, truncated, metrics = env.step(120)
            _assert_accounting(env, metrics)
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert metrics["total_objects"] == metrics["requested_objects"] == 50
        assert metrics["missed"] == 50
        assert metrics["sorted"] == metrics["dropped"] == metrics["unfinished"] == 0
        assert metrics["recycled_objects"] > 0
        assert metrics["termination_reason"] == "finished"
        assert not metrics["episode_success"]
        with pytest.raises(RuntimeError, match="ended"):
            env.step(120)
    finally:
        env.close()


def test_duration_reports_inflight_without_claiming_completed_episode():
    env = ConveyorEnv(Config(sensor_mode="oracle", max_time=0.5))
    try:
        for _ in range(6):
            _, _, terminated, truncated, metrics = env.step(120)
            if terminated or truncated:
                break
        assert truncated and not terminated
        assert metrics["termination_reason"] == "time_limit"
        assert metrics["total_objects"] == metrics["unfinished"] == 1
        assert metrics["requested_objects"] is None
        assert not metrics["episode_success"]
        _assert_accounting(env, metrics)
    finally:
        env.close()


def test_gate_physically_unloads_settled_object_and_preserves_success(monkeypatch):
    env = ConveyorEnv(Config(sensor_mode="oracle", bin_capacity=1))
    try:
        color = env.types[0]
        q, v = env.ids.cube_qpos[0], env.ids.cube_dofs[0]
        # Controlled post-release initial condition. Thereafter the object
        # settles, falls through the sliding gate, and is retired by physics.
        env.data.qpos[q : q + 3] = env.ids.bin_centers[color] + [
            0,
            0,
            env.ids.object_specs[0].half_height + 0.002,
        ]
        env.data.qvel[v : v + 6] = 0
        env.owner[0] = 0
        env.states[0] = State.RELEASED
        mujoco.mj_forward(env.model, env.data)
        for _ in range(12):
            env.step(120)
            if env.states[0] == State.SORTED:
                break
        assert env.states[0] == State.SORTED
        assert env._count(State.SORTED) == 1
        initial_height = env.cube_position(0)[2]
        physical_heights = [initial_height]
        original_recycle = env._recycle

        def observe_physics_before_recycle():
            # Sample each physics step before the collection service retires
            # the body. A 100 ms decision sample can miss the whole final fall.
            if env.serial[0] == 0 and env.states[0] != State.PENDING:
                physical_heights.append(float(env.data.qpos[q + 2]))
            original_recycle()

        monkeypatch.setattr(env, "_recycle", observe_physics_before_recycle)
        env.bin_occupied[color, 0] = True
        env.bin_slots[color] = 1
        maximum_gate_opening = 0.0
        for _ in range(80):
            _, _, terminated, truncated, metrics = env.step(120)
            assert not terminated and not truncated
            maximum_gate_opening = max(
                maximum_gate_opening, env.data.qpos[env.ids.bin_gate_qpos[color]]
            )
            assert metrics["sorted"] == 1
            assert metrics["dropped"] == 0
            _assert_accounting(env, metrics)
            if env.bin_unloads[color] == 1:
                break
        assert maximum_gate_opening > env.ids.bin_gate_open_position - 0.02
        assert min(physical_heights) < env.ids.bin_floor_z
        assert min(physical_heights) < env.ids.bin_outlet_z
        assert env.archived[int(State.SORTED)] == 1
        assert env.states[0] == State.PENDING
        assert env.bin_unloads[color] == 1
        assert env.bin_service[color] == "closed"
        assert not env.bin_occupied[color].any()
        assert env.data.qpos[env.ids.bin_gate_qpos[color]] < 0.005
        assert metrics["arm_sorted"] == [1, 0]
    finally:
        env.close()


@pytest.mark.parametrize("gate_position", [0.0, 0.004, 0.019])
def test_unload_opening_preserves_completed_count_before_gate_threshold(gate_position):
    env = ConveyorEnv(Config(sensor_mode="oracle", bin_capacity=1))
    try:
        color = env.types[0]
        q, v = env.ids.cube_qpos[0], env.ids.cube_dofs[0]
        env.data.qpos[q : q + 3] = env.ids.bin_centers[color] + [
            0,
            0,
            env.ids.object_specs[0].half_height,
        ]
        # A previously settled object can accelerate as the gate starts. It
        # must retain its completed outcome throughout authorized unloading,
        # including the first motion before the old 0.02 m gate threshold.
        env.data.qvel[v : v + 3] = [0.35, 0, 0]
        env.data.qpos[env.ids.bin_gate_qpos[color]] = gate_position
        env.states[0] = State.SORTED
        env.owner[0] = 0
        env.cycle_times[0] = 1.0
        env.arm_completions[0] = 1
        env.shape_sorted[env.ids.object_specs[0].shape] = 1
        env.bin_service[color] = "opening"
        mujoco.mj_forward(env.model, env.data)
        env._outcomes()
        assert env.states[0] == State.SORTED
        assert env.metrics()["sorted"] == 1
        assert env.metrics()["arm_sorted"] == [1, 0]
        # Unloading does not excuse an object escaping the receiver sideways.
        env.data.qpos[q] = env.ids.bin_centers[color, 0] + 0.20
        mujoco.mj_forward(env.model, env.data)
        env._outcomes()
        assert env.states[0] == State.DROPPED
        assert env.metrics()["sorted"] == 0
        assert env.metrics()["arm_sorted"] == [0, 0]
    finally:
        env.close()


def test_full_bin_waits_for_last_release_to_settle_before_opening():
    env = ConveyorEnv(Config(sensor_mode="oracle", bin_capacity=1))
    try:
        color = env.types[0]
        env.bin_occupied[color, 0] = True
        env.bin_slots[color] = 1
        env.bin_last_release[color] = env.data.time
        env._service_bins()
        assert env.bin_service[color] == "closed"
        env.data.time = 1.49
        env._service_bins()
        assert env.bin_service[color] == "closed"
        env.data.time = 1.51
        env._service_bins()
        assert env.bin_service[color] == "opening"
    finally:
        env.close()


def test_six_mixed_objects_unload_without_losing_lifetime_successes():
    env = ConveyorEnv(Config(sensor_mode="oracle", bin_capacity=6))
    try:
        # Six independently released rigid bodies occupy the calibrated grid.
        # This isolates hopper physics and accounting from robot path tuning.
        for _ in range(5):
            env.next_arrival = env.data.time
            env._spawn()
        env.next_arrival = 1000.0
        for slot in range(6):
            q, v = env.ids.cube_qpos[slot], env.ids.cube_dofs[slot]
            env.types[slot] = 0
            env.owner[slot] = slot % 2
            env.states[slot] = State.RELEASED
            env.data.qpos[q : q + 3] = env.ids.bin_centers[0] + [
                (-0.09, 0.0, 0.09)[slot % 3],
                (-0.045, 0.045)[slot // 3],
                env.ids.object_specs[slot].half_height + 0.002,
            ]
            env.data.qvel[v : v + 6] = 0
        mujoco.mj_forward(env.model, env.data)
        for _ in range(15):
            env.step(120)
            if env._count(State.SORTED) == 6:
                break
        assert env._count(State.SORTED) == 6
        assert len({spec.shape for spec in env.ids.object_specs[:6]}) >= 2
        env.bin_occupied[0] = True
        env.bin_slots[0] = 6
        env.bin_last_release[0] = env.data.time
        for _ in range(100):
            _, _, terminated, truncated, metrics = env.step(120)
            assert not terminated and not truncated
            assert metrics["sorted"] == 6 and metrics["dropped"] == 0
            _assert_accounting(env, metrics)
            if env.bin_unloads[0] == 1:
                break
        assert env.bin_unloads[0] == 1
        assert env.archived[int(State.SORTED)] == 6
        assert np.all(env.states[:6] == State.PENDING)
        assert metrics["arm_sorted"] == [3, 3]
        assert metrics["completed_success_rate"] == 1.0
        assert len(env.object_records) == 6
        assert all(record["state"] == "SORTED" for record in env.object_records)
    finally:
        env.close()


def test_lifetime_history_and_controller_caches_remain_bounded():
    env = ConveyorEnv(Config(sensor_mode="oracle"))
    try:
        for lifetime_id in range(700):
            slot = int(np.flatnonzero(env.serial == lifetime_id)[0])
            # Unit fixture for a missed object already outside the work cell;
            # the separate integration case covers its actual belt journey.
            env.states[slot] = State.MISSED
            env.data.qpos[env.ids.cube_qpos[slot]] = env.config.belt_half_length + 0.1
            mujoco.mj_forward(env.model, env.data)
            env._recycle()
            assert env._count(State.MISSED) == lifetime_id + 1
            env.data.time = env.next_arrival
            env._spawn()
            env._sense()
        metrics = env.metrics()
        _assert_accounting(env, metrics)
        assert env.recycled == 700 and env.spawned == 701
        assert (
            len(env.arrivals)
            == len(env.spawn_yaw_log)
            == len(env.object_records)
            == 512
        )
        assert len(env.events) <= 2000
        assert len(env.track_slots) <= 10 and len(env.retired_tracks) <= 10
        assert len(env.vision_id_matches) <= env.config.pool_size
        assert env.states.shape == env.serial.shape == (64,)
        assert metrics["max_slot_generations"] > 10
        assert metrics["log_window"] == 512
        assert env.object_records[-1]["id"] == 699
    finally:
        env.close()


@pytest.mark.parametrize(
    "mode, expected_phase", [("optimized", "idle"), ("legacy", "return")]
)
def test_local_placement_optimized_skips_unnecessary_home_return(mode, expected_phase):
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1, motion_mode=mode))
    try:
        env._assign(0, 0)
        goal = env.drop_targets[0].copy()
        goal[2] = env.ids.bin_rim_z + 0.07
        env.data.qpos[env.ids.arm_qpos[0]] = env.ik.solve_pose(
            env.data, 0, goal, downward_quaternion(env.grasp_yaw[0])
        )
        mujoco.mj_forward(env.model, env.data)
        assert np.linalg.norm(env.data.site_xpos[env.ids.ee_sites[0]] - goal) < 0.02
        # A completed local deposit is clear of the shared crossing corridor.
        env.transfer_owner = -1
        env.belief[0] = 4
        env._set_phase(0, "retreat")
        env._control()
        assert env.phase[0] == expected_phase
        assert env.home_returns[0] == int(mode == "legacy")
        if mode == "optimized":
            assert env.target[0] == -1
            np.testing.assert_allclose(
                env.idle_anchors[0], env.data.site_xpos[env.ids.ee_sites[0]]
            )
            assert np.linalg.norm(env.idle_anchors[0] - env.ids.home_ee[0]) > 0.05
    finally:
        env.close()


def test_path_metrics_accumulate_actual_motion_and_phase_time():
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1))
    try:
        env._assign(0, 0)
        before = env.metrics()
        for _ in range(5):
            _, _, _, _, metrics = env.step(120)
        assert sum(metrics["ee_path_m"]) > sum(before["ee_path_m"])
        assert sum(metrics["joint_travel_rad"]) > 0
        assert all(value >= 0 for value in metrics["ee_path_m"])
        assert len(metrics["ee_path_m"]) == len(metrics["home_returns"]) == 2
        assert sum(metrics["phase_seconds"].values()) == pytest.approx(
            2 * metrics["time"]
        )
        assert metrics["motion_mode"] == "optimized"
        assert metrics["route_failures"] >= 0 and metrics["route_shortcuts"] >= 0
    finally:
        env.close()
