"""A released hand returns directly to a checked, off-belt ready pose."""
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.ik import downward_quaternion
from dual_arm_conveyor.motion import MotionPlan


@pytest.fixture
def env():
    e = ConveyorEnv(Config(sensor_mode="oracle", total_objects=5, pool_size=16,
                           num_arms=4, strategy="adaptive", motion_mode="optimized",
                           deep_bins=True, auto_unload=False, bin_capacity=24,
                           direct_retask=True, future_planning=True, planned_return=True))
    yield e
    e.close()


def released_at_bin(e, arm=0, x_offset=0.0):
    side = -1 if arm % 2 == 0 else 1
    color = arm % 2
    yaw = np.pi / 2 if color == 0 else -np.pi / 2
    point = np.array([e.ids.arm_base_positions[arm, 0] + x_offset, side * 0.305, e.transport_height])
    joints = e.ik.solve_pose(e.data, arm, point, downward_quaternion(yaw))
    e.data.qpos[e.ids.arm_qpos[arm]] = joints
    e.data.ctrl[e.ids.arm_actuators[arm]] = joints
    e.data.qpos[e.ids.finger_qpos[arm]] = 0.04
    e.data.qvel[e.ids.arm_dofs[arm]] = 0
    mujoco.mj_forward(e.model, e.data)
    e.ik.reset(e.data)
    e.target[arm] = 0
    e.belief[0] = 4
    e.drop_kind[0] = color
    e.drop_targets[0] = point
    e.bin_owner[color] = arm
    e.bin_zone_owner[0,color] = arm
    e.drop_slot[0] = 0
    e.bin_occupied[color,0] = True
    e._set_phase(arm, "retreat")
    assert e._gripper_released(arm)


def test_planned_pose_uses_own_advice_clips_reach_and_stays_outside_station_strip(env):
    released_at_bin(env)
    before = (env.target.copy(), env.belief.copy(), env.data.qpos.copy())
    goal = env._planned_ready_goal(0)
    assert goal[0] == pytest.approx(env.drop_targets[0,0])
    assert goal[1] == -0.26
    assert goal[2] == max(env.lift_height, env.ids.bin_rim_z + 0.12)
    env.future_preposition_goals[1] = [10, 0.26, 0.4]
    np.testing.assert_array_equal(env._planned_ready_goal(0), goal)
    env.future_preposition_goals[0] = [10, -0.26, 0.4]
    advised = env._planned_ready_goal(0)
    assert advised[0] == pytest.approx(env.ids.arm_base_positions[0,0] + 0.18)
    for actual, initial in zip((env.target, env.belief, env.data.qpos), before):
        np.testing.assert_array_equal(actual, initial)


def test_planned_return_is_opt_in_and_requires_actual_release(env):
    released_at_bin(env)
    assert env._planned_return_available(0)
    assert abs(env._normal_return_goal(0)[1]) >= 0.26
    env.config = replace(env.config, planned_return=False)
    np.testing.assert_array_equal(env._normal_return_goal(0), env._baseline_return_goal(0))
    assert abs(env._normal_return_goal(0)[1]) < 0.23
    env.config = replace(env.config, planned_return=True)
    env.data.sensordata[env.ids.finger_touch_adr[0]] = [0.4, 0.0]
    assert not env._planned_return_available(0)
    np.testing.assert_array_equal(env._normal_return_goal(0), env.ids.home_ee[0])


def test_real_planned_return_locks_endpoint_and_retains_resources_until_settled(env):
    released_at_bin(env)
    env.future_preposition_goals[0] = [env.ids.arm_base_positions[0,0] + 0.06, -0.26, 0.4]
    env._control()
    assert env.phase[0] == 'return'
    assert env.planned_return_active[0]
    endpoint = env.return_goals[0].copy()
    env.future_preposition_goals[0,0] -= 0.15
    for _ in range(30):
        if env.phase[0] == 'idle':
            break
        assert env.bin_zone_owner[0,0] == 0
        _, _, _, _, metrics = env.step(env.wait_action)
        np.testing.assert_array_equal(env.return_goals[0], endpoint)
        assert metrics['robot_collision_steps'] == metrics['inter_arm_collision_steps'] == 0
    assert env.phase[0] == 'idle'
    assert env.planned_return_completed[0] == 1
    assert env.planned_return_fallbacks[0] == 0
    assert np.linalg.norm(env.data.site_xpos[env.ids.ee_sites[0]] - endpoint) < 0.004
    assert abs(env.data.site_xpos[env.ids.ee_sites[0],1]) > 0.25
    assert env.target[0] == -1 and env.bin_zone_owner[0,0] == -1
    assert env._gripper_released(0)


def test_rejected_planned_return_checks_original_fallback_before_releasing_resources(env, monkeypatch):
    released_at_bin(env)
    env._control()
    endpoint = env.return_goals[0].copy()
    fallback = env._baseline_return_goal(0)
    original = env.planner.plan
    requests = []

    def selective_obstacle(data, arm, goal, **kwargs):
        requests.append(np.asarray(goal).copy())
        if arm == 0 and np.allclose(goal, endpoint):
            return MotionPlan((), False, 'test planned endpoint obstacle', 0, 0, 1, 0, 0)
        return original(data, arm, goal, **kwargs)

    monkeypatch.setattr(env.planner, 'plan', selective_obstacle)
    env._control()
    assert env.planned_return_fallbacks[0] == 1
    assert not env.planned_return_active[0]
    np.testing.assert_array_equal(env.return_goals[0], fallback)
    assert env.phase[0] == 'return' and env.bin_zone_owner[0,0] == 0
    assert any(np.allclose(goal, fallback) for goal in requests)
    assert env.routes[0].verified


def test_new_tactile_load_during_planned_return_recovers_without_empty_transit(env):
    released_at_bin(env)
    env._control()
    env.data.sensordata[env.ids.finger_touch_adr[0]] = [0.5, 0.0]
    env._control()
    assert env.phase[0] == 'recover'
    assert env.hold_requested[0] and env.active_motion_plans[0] is None
    assert env.target[0] == 0 and env.bin_zone_owner[0,0] == 0
    assert env.planned_return_completed[0] == 0


def test_planned_return_does_not_finish_at_position_with_unsettled_velocity(env):
    released_at_bin(env)
    env._control()
    goal = env.return_goals[0]
    joints = env.ik.solve_pose(env.data, 0, goal, downward_quaternion(np.pi / 2))
    env.data.qpos[env.ids.arm_qpos[0]] = joints
    env.data.ctrl[env.ids.arm_actuators[0]] = joints
    mujoco.mj_forward(env.model, env.data)
    env.ik.reset(env.data)
    env.ik.cart_velocities[0] = [0.20, 0, 0]
    env._control()
    assert env.phase[0] == 'return'
    assert env.target[0] == 0 and env.bin_zone_owner[0,0] == 0
    assert env.planned_return_completed[0] == 0


@pytest.mark.parametrize('returning_arm', [0, 1])
def test_physical_planned_return_overlaps_opposing_grasp_without_robot_contacts(env, returning_arm):
    # Controlled initial conditions only: one genuinely open, empty hand at
    # the outer deposit point and a box physically moving on the belt.
    # All subsequent robot/object motion uses ordinary env.step actuators.
    released_at_bin(env, arm=returning_arm, x_offset=-0.18)
    partner = returning_arm ^ 1
    side = -1 if returning_arm % 2 == 0 else 1
    env.future_preposition_goals[returning_arm] = [
        env.ids.arm_base_positions[returning_arm, 0] + 0.18, side * 0.26, 0.4
    ]
    body_slot = next(i for i, spec in enumerate(env.ids.object_specs)
                     if i > 0 and spec.shape == 'box')
    env._admit_object(body_slot, 0.0, (-1 if partner % 2 == 0 else 1) * 0.065,
                      partner % 2)
    q = env.ids.cube_qpos[body_slot]
    env.data.qpos[q] = -0.64
    mujoco.mj_forward(env.model, env.data)
    env._sense()
    target_slot = env.track_slots[int(env.serial[body_slot])]
    env._control()
    assert env.phase[returning_arm] == 'return'
    assert env.planned_return_active[returning_arm]
    choices = [10] * 4
    choices[partner] = target_slot
    action = int(np.ravel_multi_index(tuple(choices), (11,) * 4))
    assert env.observe()['mask'][action]
    overlap_steps = 0
    partner_phases = set()

    def audit(e):
        nonlocal overlap_steps
        partner_phases.add(e.phase[partner])
        if e.phase[returning_arm] == 'return' and e.phase[partner] in ('approach','descend','close','lift'):
            overlap_steps += 1
        assert e.robot_collision_steps == 0
        assert e.collision_steps == 0

    _, _, _, _, metrics = env.step(action, physics_callback=audit)
    for _ in range(79):
        if env.phase[partner] == 'idle' and env.phase[returning_arm] == 'idle':
            break
        _, _, _, _, metrics = env.step(env.wait_action, physics_callback=audit)
    assert overlap_steps * env.config.physics_dt > 0.1
    assert {'approach','descend','close','lift'} <= partner_phases
    assert env.planned_return_completed[returning_arm] == 1
    assert env.phase[returning_arm] == env.phase[partner] == 'idle'
    assert metrics['robot_collision_steps'] == metrics['inter_arm_collision_steps'] == 0
