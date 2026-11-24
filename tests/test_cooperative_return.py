"""Off-belt returns keep release, checked-path and forecast contracts.

Most tests inspect single controller commands. The short physical return test
also checks actual settling under actuators; no policy training is performed.
"""
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.forecast import controller_readiness, phase_budgets, predicted_ready_poses
from dual_arm_conveyor.ik import downward_quaternion
from dual_arm_conveyor.motion import MotionPlan


@pytest.fixture
def env():
    simulation = ConveyorEnv(Config(
        num_arms=4, strategy="adaptive", sensor_mode="oracle",
        deep_bins=True, auto_unload=False, total_objects=5, pool_size=16,
        bin_capacity=20, direct_retask=True, nearest_bin_slot=True,
        cooperative_motion=True, cooperative_drop_cost=False, control_speed=0.65,
    ))
    yield simulation
    simulation.close()


def released_at_bin(env, arm=0):
    side = -1 if arm % 2 == 0 else 1
    point = np.array([env.ids.arm_base_positions[arm, 0] + 0.06,
                      side * 0.305, env.transport_height])
    yaw = np.pi / 2 if arm % 2 == 0 else -np.pi / 2
    joints = env.ik.solve_pose(env.data, arm, point, downward_quaternion(yaw))
    env.data.qpos[env.ids.arm_qpos[arm]] = joints
    env.data.ctrl[env.ids.arm_actuators[arm]] = joints
    env.data.qpos[env.ids.finger_qpos[arm]] = 0.04
    env.data.qvel[env.ids.arm_dofs[arm]] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.ik.reset(env.data)
    env.target[arm] = 0
    env.belief[0] = 4
    env.drop_kind[0] = arm % 2
    env.drop_targets[0] = point
    env.drop_slot[0] = (arm // 2) * (env.config.bin_capacity // 2)
    env.bin_occupied[arm % 2, env.drop_slot[0]] = True
    env.bin_owner[arm % 2] = arm
    env.bin_zone_owner[arm // 2, arm % 2] = arm
    env._set_phase(arm, "retreat")
    assert env._gripper_released(arm)


@pytest.mark.parametrize("arm", range(4))
def test_confirmed_empty_return_stays_outside_station_strip_and_is_opt_in(env, arm):
    released_at_bin(env, arm)
    endpoint = env._baseline_return_goal(arm)
    assert endpoint[1] == pytest.approx((-1 if arm % 2 == 0 else 1) * 0.26)
    assert endpoint[0] == pytest.approx(env.drop_targets[0, 0])
    assert endpoint[2] >= env.ids.bin_rim_z + 0.12
    env.config = replace(env.config, cooperative_motion=False)
    legacy = env._baseline_return_goal(arm)
    np.testing.assert_array_equal(legacy[[0, 2]], endpoint[[0, 2]])
    assert legacy[1] == env.ids.home_ee[arm, 1]


@pytest.mark.parametrize("reason", ["holding", "belief", "unreleased", "wrong_bin", "missing_target"])
def test_off_belt_endpoint_requires_actual_empty_hand_release(env, reason):
    released_at_bin(env)
    if reason == "holding":
        env.holding[0] = True
    elif reason == "belief":
        env.belief[0] = 3
    elif reason == "unreleased":
        env.data.sensordata[env.ids.finger_touch_adr[0]] = [0.5, 0.0]
    elif reason == "wrong_bin":
        env.drop_kind[0] = 1
    else:
        env.target[0] = -1
    np.testing.assert_array_equal(env._baseline_return_goal(0), env.ids.home_ee[0])


def test_return_locks_endpoint_until_completed_instead_of_following_new_advice(env):
    released_at_bin(env)
    env._control()
    assert env.phase[0] == "return"
    locked = env.return_goals[0].copy()
    env.drop_targets[0, 0] += 0.1
    env.future_preposition_goals[0] = [-0.9, -0.26, 0.4]
    env._control()
    np.testing.assert_array_equal(env.return_goals[0], locked)
    np.testing.assert_array_equal(predicted_ready_poses(env)[0], locked)
    assert env.bin_zone_owner[0, 0] == 0 and env.target[0] == 0
    assert env.phase[0] == "return"


@pytest.mark.parametrize("arm", range(4))
def test_physical_empty_return_settles_outside_strip_before_releasing_resources(env, arm):
    released_at_bin(env, arm)
    env._control()
    endpoint = env.return_goals[arm].copy()
    for _ in range(35):
        if env.phase[arm] == "idle":
            break
        assert env.target[arm] == 0
        assert env.bin_zone_owner[arm // 2, arm % 2] == arm
        _, _, _, _, info = env.step(env.wait_action)
        assert info['robot_collision_steps'] == info['inter_arm_collision_steps'] == 0
    assert env.phase[arm] == 'idle'
    assert env.target[arm] == -1
    assert env.bin_zone_owner[arm // 2, arm % 2] == -1
    np.testing.assert_allclose(env.data.site_xpos[env.ids.ee_sites[arm]], endpoint, atol=.004)
    assert abs(env.data.site_xpos[env.ids.ee_sites[arm], 1]) > .25


def test_rejected_off_belt_route_falls_back_to_checked_home_without_unlocking(env, monkeypatch):
    released_at_bin(env)
    env._control()
    endpoint = env.return_goals[0].copy()
    original = env.planner.plan
    checked = []

    def obstacle(data, arm, goal, **kwargs):
        checked.append(np.asarray(goal).copy())
        if arm == 0 and np.allclose(goal, endpoint):
            return MotionPlan((), False, "test off-belt obstacle", 0, 0, 1, 0, 0)
        return original(data, arm, goal, **kwargs)

    monkeypatch.setattr(env.planner, "plan", obstacle)
    env._control()
    np.testing.assert_array_equal(env.return_goals[0], env.ids.home_ee[0])
    assert any(np.allclose(goal, env.ids.home_ee[0]) for goal in checked)
    assert env.routes[0].verified
    assert env.direct_retask_fallbacks[0] == 1
    assert env.phase[0] == "return" and env.bin_zone_owner[0, 0] == 0
    np.testing.assert_array_equal(predicted_ready_poses(env)[0], env.ids.home_ee[0])


def test_new_tactile_load_aborts_empty_return_before_any_transit_command(env):
    released_at_bin(env)
    env._control()
    env.data.sensordata[env.ids.finger_touch_adr[0]] = [0.5, 0.0]
    env._control()
    assert env.phase[0] == "recover"
    assert env.hold_requested[0] and env.active_motion_plans[0] is None
    assert env.target[0] == 0 and env.bin_zone_owner[0, 0] == 0
    assert any(event["event"] == "cooperative_return_release_lost" for event in env.events)


def test_future_ready_pose_predicts_post_release_endpoint_while_still_holding(env):
    released_at_bin(env)
    endpoint = env._baseline_return_goal(0)
    env.phase[0] = "transport"
    env.holding[0] = True
    env.belief[0] = 3
    # Prediction concerns the eventual empty endpoint, not authorization now.
    np.testing.assert_array_equal(predicted_ready_poses(env)[0], endpoint)
    np.testing.assert_array_equal(env._baseline_return_goal(0), env.ids.home_ee[0])


def test_return_budget_uses_selected_off_belt_endpoint(env):
    released_at_bin(env)
    env.phase[0] = "release"
    budget = phase_budgets(env)
    endpoint = predicted_ready_poses(env)[0]
    origin = env.drop_targets[0].copy()
    origin[2] = env.transport_height
    speed = 0.85 * env.config.control_speed
    expected = budget["release"] + budget["retreat"] + max(0.12, np.linalg.norm(endpoint-origin)/speed + 0.20)
    ready, _ = controller_readiness(env)
    assert ready[0] == pytest.approx(expected)


@pytest.mark.parametrize("phase", ["recover", "yield"])
def test_recovery_forecast_uses_home_even_with_cached_off_belt_return(env, phase):
    released_at_bin(env)
    env.return_goals[0] = env._baseline_return_goal(0)
    env.phase[0] = phase
    np.testing.assert_array_equal(predicted_ready_poses(env)[0], env.ids.home_ee[0])
    ready, _ = controller_readiness(env)
    distance = np.linalg.norm(env.ids.home_ee[0] - env.data.site_xpos[env.ids.ee_sites[0]])
    assert ready[0] == pytest.approx(max(0.12, distance / (0.85*env.config.control_speed) + 0.20))


def test_inside_strip_forecast_never_announces_immediate_clearance(env):
    released_at_bin(env)
    env.phase[0] = "return"
    env.return_goals[0] = env._baseline_return_goal(0)
    # Only a forecast fixture; never advance physical state in this test.
    env.data.site_xpos[env.ids.ee_sites[0], 1] = -0.20
    off_belt_ready, off_belt_clear = controller_readiness(env)
    assert 0 < off_belt_clear[0] <= off_belt_ready[0]
    env.return_goals[0] = env.ids.home_ee[0]
    fallback_ready, fallback_clear = controller_readiness(env)
    assert fallback_clear[0] == fallback_ready[0]


def test_disabled_cooperation_keeps_original_forecast_endpoint(env):
    released_at_bin(env)
    env.config = replace(env.config, cooperative_motion=False, future_planning=True)
    env.phase[0] = "transport"
    endpoint = predicted_ready_poses(env)[0]
    assert endpoint[1] == env.ids.home_ee[0, 1]
