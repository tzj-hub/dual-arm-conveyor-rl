"""Throughput options retain release, braking, reservation and grid contracts."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.ik import downward_quaternion
from dual_arm_conveyor.vision import Track


def make_config(**changes):
    return Config(
        **dict(
            num_arms=4,
            strategy="adaptive",
            motion_mode="optimized",
            sensor_mode="oracle",
            deep_bins=True,
            auto_unload=False,
            total_objects=2,
            pool_size=64,
            bin_capacity=24,
            belt_speed=0.2,
            control_speed=0.65,
            max_time=2,
            **changes,
        )
    )


@pytest.fixture
def env():
    simulation = ConveyorEnv(make_config(direct_retask=True, nearest_bin_slot=True), seed=55000)
    yield simulation
    simulation.close()


def put_track(env, slot=0, arm=0, x=-0.75):
    env.tracks[slot] = Track(
        9900 + slot,
        np.array([x, 0.0, 0.23]),
        np.array([0.2, 0, 0]),
        np.pi / 2 if arm % 2 == 0 else -np.pi / 2,
        arm % 2,
        1.0,
        float(env.data.time),
        float(env.data.time),
        10,
        True,
    )
    env.belief[slot] = 1


def released_target(env):
    put_track(env)
    env.target[0] = 0
    env.belief[0] = 4
    env.retired_tracks[9900] = float(env.data.time)
    env.drop_kind[0] = 0
    env.drop_targets[0] = env._deposit_position(0, 0, 0)
    env.bin_owner[0] = env.bin_zone_owner[0, 0] = 0
    env.bin_occupied[0, 0] = True
    env.return_goals[0] = env._normal_return_goal(0)
    env.phase[0] = "return"


def put_tool(env, position):
    q = env.ik.solve_pose(env.data, 0, position, downward_quaternion(env.grasp_yaw[0]))
    env.data.qpos[env.ids.arm_qpos[0]] = q
    mujoco.mj_forward(env.model, env.data)
    assert np.linalg.norm(env.data.site_xpos[env.ids.ee_sites[0]] - position) < 0.001


@pytest.mark.parametrize("name", ["direct_retask", "nearest_bin_slot", "responsive_release", "direct_bin_release"])
def test_options_default_off_and_reject_incompatible_configuration(name):
    assert getattr(Config(), name) is False
    with pytest.raises(ValueError, match=f"{name} must be bool"):
        make_config(**{name: 1})
    with pytest.raises(ValueError, match=name):
        Config(**{name: True})
    with pytest.raises(ValueError, match=name):
        replace(make_config(**{name: True}), gripper_mode="vacuum")


def test_short_return_is_empty_only_and_stays_above_bin(env, monkeypatch):
    released_target(env)
    short = env._normal_return_goal(0)
    assert short[0] == pytest.approx(env.drop_targets[0, 0])
    assert short[1] == pytest.approx(env.ids.home_ee[0, 1])
    assert short[2] >= env.ids.bin_rim_z + 0.12
    assert np.linalg.norm(short - env.ids.home_ee[0]) > 0.1
    env.holding[0] = True
    np.testing.assert_array_equal(env._normal_return_goal(0), env.ids.home_ee[0])
    env.holding[0] = False
    monkeypatch.setattr(env, "_gripper_released", lambda arm: False)
    np.testing.assert_array_equal(env._normal_return_goal(0), env.ids.home_ee[0])


def test_short_return_keeps_ownership_until_braked_and_released(env, monkeypatch):
    released_target(env)
    end = env.return_goals[0].copy()
    put_tool(env, end)
    accepted = [False]
    env.routes[0] = SimpleNamespace(
        verified=True, waypoints=(end,), can_advance=lambda *args, **kwargs: accepted[0]
    )
    monkeypatch.setattr(env.ik, "command", lambda *args, **kwargs: None)
    monkeypatch.setattr(env, "_route_goal", lambda arm, goal, loaded, **kwargs: (np.asarray(goal), True))
    env._control()
    assert env.phase[0] == "return" and env.target[0] == 0
    assert env.bin_zone_owner[0, 0] == env.bin_owner[0] == 0
    accepted[0] = True
    released = [False]
    monkeypatch.setattr(env, "_gripper_released", lambda arm: released[0])
    env._control()
    assert env.phase[0] == "return" and env.bin_zone_owner[0, 0] == 0
    released[0] = True
    env._control()
    assert env.phase[0] == "idle" and env.target[0] == -1
    assert env.bin_zone_owner[0, 0] == env.bin_owner[0] == -1
    assert env.bin_occupied[0, 0] and env.belief[0] == 4
    assert 9900 in env.retired_tracks
    assert env.direct_retask_returns[0] == 1
    np.testing.assert_allclose(env.idle_anchors[0], end, atol=0.001)


def test_return_cannot_receive_task_or_unlock_partner_at_low_height(env):
    released_target(env)
    put_track(env, slot=1, arm=0)
    env.phase[0] = "idle"
    assert env._candidate(0, 1)[0]
    env.phase[0] = "return"
    assert not env._candidate(0, 1)[0]
    put_track(env, slot=1, arm=1)
    env.phase[0] = "idle"
    assert env._candidate(1, 1)[0]
    env.phase[0] = "return"
    assert not env._candidate(1, 1)[0]


def test_candidate_preview_does_not_clear_old_resources_or_change_return_path(env):
    released_target(env)
    put_track(env, slot=1, arm=0)
    before = {
        name: getattr(env, name).copy()
        for name in ("target", "belief", "bin_owner", "bin_zone_owner", "return_goals")
    }
    for _ in range(3):
        observation = env.observe()
        assert observation["mask"][env.wait_action]
        assert not env._candidate(0, 1)[0]
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(env, name), value)


def test_rejected_short_route_falls_back_without_releasing_reservations(env, monkeypatch):
    released_target(env)
    calls = []
    monkeypatch.setattr(env.ik, "command", lambda *args, **kwargs: None)
    def reject(arm, goal, loaded, **kwargs):
        calls.append(np.asarray(goal).copy())
        return np.asarray(goal), False
    monkeypatch.setattr(env, "_route_goal", reject)
    env._control()
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[-1], env.ids.home_ee[0])
    assert env.phase[0] == "return" and env.target[0] == 0
    assert env.bin_zone_owner[0, 0] == 0
    assert env.direct_retask_fallbacks[0] == 1


def test_nearest_slot_keeps_each_layer_distinct_and_inside_own_zone(env):
    for arm in (0, 2):
        put_track(env, slot=0, arm=arm, x=(-0.7 if arm == 0 else 0.1))
        per_zone = env.config.bin_capacity // 2
        free = np.arange((arm // 2) * per_zone, (arm // 2 + 1) * per_zone)
        original = free.copy()
        chosen = []
        for _ in range(per_zone):
            rank = env._choose_drop_slot(arm, 0, free)
            chosen.append(rank)
            free = free[free != rank]
        assert set(chosen[:10]) == set(original[:10])
        assert set(chosen[10:]) == set(original[10:])
        assert len(set(chosen)) == per_zone
        positions = [env._deposit_position(arm, arm % 2, rank) for rank in chosen[:10]]
        assert len({tuple(p[:2]) for p in positions}) == 10
        assert all(abs(p[0] - env.ids.arm_base_positions[arm, 0]) <= 0.180001 for p in positions)


def test_disabled_options_keep_original_slot_order_and_home(env):
    env.config = replace(env.config, direct_retask=False, nearest_bin_slot=False)
    released_target(env)
    np.testing.assert_array_equal(env._normal_return_goal(0), env.ids.home_ee[0])
    assert env._choose_drop_slot(0, 0, np.array([3, 7, 10])) == 3
    env.reset(55001)
    assert not env.direct_retask_returns.any()
    assert not env.direct_retask_fallbacks.any()


def start_responsive_release(env, aperture=0.04):
    env.config = replace(env.config, responsive_release=True)
    released_target(env)
    env.belief[0] = 3
    env.holding[0] = True
    env.data.qpos[env.ids.finger_qpos[0]] = aperture / 2
    env.data.sensordata[env.ids.finger_touch_adr[0]] = 2.0
    env.ik.gripper_targets[0] = 0.002
    env.data.ctrl[env.ids.gripper_actuators[0]] = 0.002
    env._set_phase(0, "release")


def test_responsive_release_preserves_controller_reference_and_opening_slew(env):
    start_responsive_release(env)
    assert env.release_start_aperture[0] == pytest.approx(0.04)
    assert env.ik.gripper_targets[0] == 0.002
    assert env.data.ctrl[env.ids.gripper_actuators[0]] == 0.002
    before_qpos, before_qvel = env.data.qpos.copy(), env.data.qvel.copy()
    env.ik.command(env.data, 0, env.ids.home_ee[0], gripper=0.04, hold=True)
    assert env.ik.gripper_targets[0] == pytest.approx(0.002 + 0.10 * env.config.physics_dt)
    np.testing.assert_array_equal(env.data.qpos, before_qpos)
    np.testing.assert_array_equal(env.data.qvel, before_qvel)


def test_responsive_release_requires_clearance_and_stable_no_contact(env, monkeypatch):
    start_responsive_release(env)
    monkeypatch.setattr(env.ik, "command", lambda *args, **kwargs: None)
    env.data.time = 0.20
    env.data.qpos[env.ids.finger_qpos[0]] = 0.049 / 2
    env.data.sensordata[env.ids.finger_touch_adr[0]] = 0.0
    env._control()
    assert env.phase[0] == "release" and np.isnan(env.release_clear_since[0])
    env.data.time = 0.21
    env.data.qpos[env.ids.finger_qpos[0]] = 0.051 / 2
    env._control()
    assert env.phase[0] == "release" and env.release_clear_since[0] == 0.21
    env.data.time = 0.26
    env.data.sensordata[env.ids.finger_touch_adr[0]] = [0.11, 0.0]
    env._control()
    assert env.phase[0] == "release" and np.isnan(env.release_clear_since[0])
    env.data.time = 0.27
    env.data.sensordata[env.ids.finger_touch_adr[0]] = 0.0
    env._control()
    env.data.time = 0.365
    env._control()
    assert env.phase[0] == "release"
    env.data.time = 0.375
    env._control()
    assert env.phase[0] == "retreat" and not env.holding[0]
    assert env.responsive_release_count == 1 and env.arm_releases[0] == 1
    assert env.target[0] == 0 and env.bin_zone_owner[0, 0] == env.bin_owner[0] == 0
    assert env._gripper_released(0)  # Same criterion during retreat.
    env._set_phase(0, "return")
    assert env._gripper_released(0)
    assert env.bin_zone_owner[0, 0] == 0


def test_responsive_release_caps_wide_object_clearance_at_original_threshold(env):
    start_responsive_release(env, aperture=0.070)
    assert env._release_aperture_threshold(0) == pytest.approx(0.075)
    env.data.sensordata[env.ids.finger_touch_adr[0]] = 0.0
    env.data.qpos[env.ids.finger_qpos[0]] = 0.0749 / 2
    assert not env._gripper_released(0)
    env.data.qpos[env.ids.finger_qpos[0]] = 0.0751 / 2
    assert env._gripper_released(0)


def test_recovery_keeps_original_full_open_guard_and_clears_release_reference(env):
    start_responsive_release(env)
    env.data.qpos[env.ids.finger_qpos[0]] = 0.051 / 2
    env.data.sensordata[env.ids.finger_touch_adr[0]] = 0.0
    assert env._gripper_released(0)
    env._set_phase(0, "recover")
    assert np.isnan(env.release_start_aperture[0])
    assert not env._gripper_released(0)
    assert env._release_aperture_threshold(0) == pytest.approx(0.075)
    _, ready = env._route_goal(0, env.ids.home_ee[0], False)
    assert not ready and env.hold_requested[0]
    assert env.bin_zone_owner[0, 0] == 0
    env._set_phase(0, "approach")
    assert not env._responsive_release_active(0)
    env.reset(55001)
    assert np.isnan(env.release_start_aperture).all()
    assert env.responsive_release_count == 0


def test_disabled_responsive_release_retains_full_open_requirement(env):
    start_responsive_release(env)
    env.config = replace(env.config, responsive_release=False)
    env.data.qpos[env.ids.finger_qpos[0]] = 0.051 / 2
    env.data.sensordata[env.ids.finger_touch_adr[0]] = 0.0
    assert not env._gripper_released(0)
    assert env._release_aperture_threshold(0) == pytest.approx(0.075)


def test_direct_bin_release_changes_only_the_two_centimetre_deposit_stroke(env):
    original_deposit = env.deposit_height
    original_transport = env.transport_height
    assert original_transport - original_deposit == pytest.approx(0.020)
    env.config = replace(env.config, direct_bin_release=True)
    assert env.deposit_height == env.transport_height == original_transport
    metrics = env.metrics()
    assert metrics["deposit_tool_height_m"] == pytest.approx(original_deposit + 0.020)
    assert metrics["deposit_tool_clearance_above_rim_m"] == pytest.approx(0.070)
    env.config = replace(env.config, direct_bin_release=False)
    assert env.deposit_height == original_deposit


def test_direct_bin_release_retains_place_velocity_path_and_release_guards(env, monkeypatch):
    env.config = replace(env.config, direct_bin_release=True)
    released_target(env)
    env.belief[0] = 3
    env.holding[0] = True
    end = env.drop_targets[0].copy()
    end[2] = env.transport_height
    put_tool(env, end)
    env._set_phase(0, "transport")
    ready = [True]
    monkeypatch.setattr(env.ik, "command", lambda *args, **kwargs: None)
    monkeypatch.setattr(env, "_gripper_has_object", lambda arm: True)
    def route(arm, goal, loaded, **kwargs):
        env.routes[arm] = SimpleNamespace(
            verified=ready[0], waypoints=(np.asarray(goal),),
            can_advance=lambda *args, **kwargs: True,
        )
        return np.asarray(goal), ready[0]
    monkeypatch.setattr(env, "_route_goal", route)
    env._control()
    assert env.phase[0] == "place"  # No phase is skipped.
    env.data.qvel[env.ids.arm_dofs[0]] = 0.4
    env._control()
    assert env.phase[0] == "place"
    env.data.qvel[env.ids.arm_dofs[0]] = 0.0
    ready[0] = False
    env._control()
    assert env.phase[0] == "place"
    ready[0] = True
    env._control()
    assert env.phase[0] == "release"
    env.data.time = 1.0
    env.data.sensordata[env.ids.finger_touch_adr[0]] = 2.0
    env._control()
    assert env.phase[0] == "release" and env.holding[0]
    env.data.time = 1.1
    env.data.sensordata[env.ids.finger_touch_adr[0]] = 0.0
    env._control()
    assert env.phase[0] == "release"
    env.data.time = 1.21
    env._control()
    assert env.phase[0] == "retreat"
    assert env.bin_zone_owner[0, 0] == 0
    env._control()
    assert env.phase[0] == "return" and env.bin_zone_owner[0, 0] == 0


def test_direct_bin_release_clears_station_strip_before_holding_empty_hand(env, monkeypatch):
    env.config = replace(env.config, direct_bin_release=True)
    released_target(env)
    env.belief[0], env.holding[0] = 3, True
    env.drop_targets[0] = env._deposit_position(0, 0, 5)
    end = env.drop_targets[0].copy()
    end[2] = env.deposit_height
    assert end[1] == pytest.approx(-0.235)
    near = end.copy()
    near[1] = -0.229
    put_tool(env, near)
    env._set_phase(0, "place")
    accepted = [True]
    def route(arm, goal, loaded, **kwargs):
        env.routes[arm] = SimpleNamespace(
            verified=True, waypoints=(np.asarray(goal),),
            can_advance=lambda *args, **kwargs: accepted[0],
        )
        return np.asarray(goal), True
    monkeypatch.setattr(env, "_route_goal", route)
    monkeypatch.setattr(env.ik, "command", lambda *args, **kwargs: None)
    monkeypatch.setattr(env, "_gripper_has_object", lambda arm: True)
    assert np.linalg.norm(env.data.site_xpos[env.ids.ee_sites[0]] - end) < 0.012
    env._control()
    assert env.phase[0] == "place" and env.bin_zone_owner[0, 0] == 0
    near[1] = -0.234
    put_tool(env, near)
    accepted[0] = False
    env._control()
    assert env.phase[0] == "place"  # Geometry alone does not prove braking.
    accepted[0] = True
    env._control()
    assert env.phase[0] == "release" and env.bin_zone_owner[0, 0] == 0
