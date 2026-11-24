"""Physical jaw admission, contact-width evidence and carried-object bounds."""

from dataclasses import replace

import numpy as np
import pytest
import mujoco

from dual_arm_conveyor.ik import downward_quaternion

from test_finger_approach import env as _environment_fixture, target

env = _environment_fixture


def test_rotated_rectangle_projection_keeps_full_observed_extent(env):
    track = replace(target(env), size=np.array([0.060, 0.040, 0.05]), yaw=0.0)
    np.testing.assert_allclose(
        env._grasp_size_at_yaw(track, np.pi / 2), [0.04, 0.06, 0.05]
    )
    np.testing.assert_allclose(
        env._grasp_size_at_yaw(track, np.pi / 4),
        [0.1 / np.sqrt(2), 0.1 / np.sqrt(2), 0.05],
    )


def test_large_diagonal_selects_nearby_wrist_angle_that_fits(env):
    track = replace(target(env), size=np.array([0.060, 0.055, 0.05]), yaw=np.pi / 4)
    angles = env._grasp_angles(0, track)
    assert np.pi / 2 not in angles  # 81 mm diagonal cannot enter 80 mm jaws.
    assert angles
    assert min(abs(yaw - np.pi / 2) for yaw in angles) <= np.pi / 4
    assert all(env._grasp_size_at_yaw(track, yaw)[1] <= 0.072 for yaw in angles)


def test_rotated_contact_grasp_uses_minimum_width_and_both_touch_sensors(env):
    env.grasp_sizes[0] = [0.09, 0.09, 0.05]  # Conservative orientation-free proxy.
    env.grasp_min_widths[0] = 0.040
    env.data.qpos[env.ids.finger_qpos[0]] = 0.021
    env.data.sensordata[env.ids.finger_touch_adr[0]] = [4.0, 5.0]
    assert env._gripper_has_object(0)
    env.data.sensordata[env.ids.finger_touch_adr[0]] = [4.0, 0.0]
    assert not env._gripper_has_object(0)
    env.data.sensordata[env.ids.finger_touch_adr[0]] = [4.0, 5.0]
    env.data.qpos[env.ids.finger_qpos[0]] = 0.003
    assert not env._gripper_has_object(0)


def test_assignment_uses_orientation_free_payload_and_observed_min_width(env):
    track = replace(
        target(env),
        size=np.array([0.055, 0.050, 0.050]),
    )
    env.tracks[0] = track
    env._assign(0, 0)
    assert env.grasp_yaw[0] == pytest.approx(np.pi / 2)
    assert env.grasp_min_widths[0] == min(track.size[:2])
    assert min(env.grasp_sizes[0, :2]) >= np.linalg.norm(track.size[:2])


def test_ungraspable_visual_footprint_is_rejected_before_assignment(env):
    env.tracks[0] = replace(target(env), size=np.array([0.09, 0.09, 0.05]))
    assert env._grasp_angles(0, env.tracks[0]) == []
    assert not env._candidate(0, 0)[0]
    assert env.heuristic_action() == env.wait_action


def test_descent_predicts_remaining_rendezvous_and_uses_actual_contact_center(
    env, monkeypatch
):
    track = target(env)
    env.target[0] = 0
    env.belief[0] = 2
    env.grasp_yaw[0] = np.pi / 2
    captured = []
    monkeypatch.setattr(env.ik, "command", lambda *args, **kwargs: None)

    def route(arm, goal, loaded, **kwargs):
        if arm == 0:
            captured.append(goal.copy())
        return goal, True

    monkeypatch.setattr(env, "_route_goal", route)
    low = track.position.copy()
    low[2] = env.config.belt_top + track.half_height + env.grasp_offsets[0]
    for height in (0.10, 0.002):
        above = low + [0, 0, height]
        env.data.qpos[env.ids.arm_qpos[0]] = env.ik.solve_pose(
            env.data, 0, above, downward_quaternion(env.grasp_yaw[0])
        )
        mujoco.mj_forward(env.model, env.data)
        actual_height = env.data.site_xpos[env.ids.ee_sites[0], 2] - low[2]
        env._set_phase(0, "descend")
        env._control()
        assert captured[-1][0] == pytest.approx(
            low[0] + 0.15 * min(0.35, max(0, actual_height) / 0.40)
        )
    assert env.phase[0] == "close"
    np.testing.assert_allclose(env.close_position[0], track.position)


@pytest.mark.parametrize("arm,x,expected_lead", [(0, -0.65, False), (2, 0.10, True)])
def test_approach_lead_is_bounded_and_vanishes_at_the_observed_target(
    env, monkeypatch, arm, x, expected_lead
):
    track = target(env)
    track.position[0] = x
    env.target[arm] = 0
    env.belief[0] = 2
    env.grasp_yaw[arm] = np.pi / 2
    env._set_phase(arm, "approach")
    captured = []

    def route(requested_arm, goal, loaded, **kwargs):
        if requested_arm == arm:
            captured.append(goal.copy())
        return goal, False

    monkeypatch.setattr(env, "_route_goal", route)
    monkeypatch.setattr(env, "_descent_feasible", lambda *args: True)
    monkeypatch.setattr(env.ik, "command", lambda *args, **kwargs: None)
    env._control()
    if expected_lead:
        assert 0 < captured[-1][0] - track.position[0] <= 0.15
    else:
        assert captured[-1][0] == track.position[0]
    assert env.phase[arm] == "approach"
    above = track.position.copy()
    above[2] = (
        env.config.belt_top
        + track.half_height
        + env.grasp_offsets[arm]
        + env.approach_clearance
    )
    env.data.qpos[env.ids.arm_qpos[arm]] = env.ik.solve_pose(
        env.data, arm, above, downward_quaternion(env.grasp_yaw[arm])
    )
    mujoco.mj_forward(env.model, env.data)
    env._control()
    np.testing.assert_allclose(captured[-1], above, atol=0.0001)
