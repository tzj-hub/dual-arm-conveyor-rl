"""Reject a low grasp whose moving closure cannot be followed by a safe lift."""

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.ik import downward_quaternion
from dual_arm_conveyor.model import build_scene, reset_robot_pose
from dual_arm_conveyor.motion import MotionPlanner


@pytest.fixture
def scene():
    model, ids = build_scene(
        Config(num_arms=4, strategy="adaptive", sensor_mode="oracle")
    )
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    return model, data, ids, MotionPlanner(model, ids)


def early_preview(planner, data, **kwargs):
    return planner.grasp_exit_feasible(
        data,
        1,
        [-0.40, 0.0, 0.225],
        [-0.36, 0.0, 0.38],
        grasp_position=[-0.52, 0.0, 0.225],
        quaternion=downward_quaternion(np.pi / 2),
        grasp_offset=-0.009,
        **kwargs,
    )


def test_early_grasp_keeps_one_checked_branch_through_closure_and_lift(scene):
    model, data, ids, planner = scene
    preview = early_preview(planner, data)
    assert preview.verified, preview.reason
    assert preview.execution_kind == "preview"
    assert preview.checked_samples > 20
    assert preview.minimum_clearance >= planner.margin
    assert len(preview.waypoints) == len(preview.joint_paths) == 3
    np.testing.assert_array_equal(preview.joint_paths[0][0], data.qpos[ids.arm_qpos[1]])
    for before, after in zip(preview.joint_paths, preview.joint_paths[1:]):
        np.testing.assert_array_equal(before[-1], after[0])
    # A preview is an admission check; no caller may accidentally execute it
    # as a static route after the object or the other robots have moved.
    with pytest.raises(ValueError, match="not an executable"):
        preview.joint_reference_at(0, preview.start_position)
    with pytest.raises(ValueError, match="not an executable"):
        preview.orientation_at(0, preview.start_position)
    assert not preview.can_advance(0, preview.waypoints[0], np.zeros(3))


def test_reachable_late_closure_is_rejected_when_following_lift_is_unreachable(scene):
    _, data, _, planner = scene
    # Regression from the four-arm variable-speed failure: the upstream arm
    # can close near x=+.02, but cannot then lift at the downstream continuation.
    # Checking just the low endpoint used to leave the hand stationary on the
    # moving belt, where the next workpiece subsequently hit its palm.
    preview = planner.grasp_exit_feasible(
        data,
        1,
        [0.02, 0.0, 0.225],
        [0.06, 0.0, 0.38],
        grasp_position=[-0.10, 0.0, 0.225],
        quaternion=downward_quaternion(np.pi / 2),
    )
    assert not preview.verified
    assert preview.reason.startswith("grasp exit lift: IK residual")
    assert preview.checked_samples > 20
    assert preview.waypoints == preview.joint_paths == ()


def test_exit_preview_does_not_read_hidden_objects_or_modify_live_state(scene):
    model, data, ids, planner = scene
    ordinary = early_preview(planner, data)
    for address in ids.cube_qpos:
        data.qpos[address : address + 7] = np.nan
    for address in ids.cube_dofs:
        data.qvel[address : address + 6] = np.nan
    before = (
        data.qpos.copy(),
        data.qvel.copy(),
        data.ctrl.copy(),
        data.qfrc_applied.copy(),
        model.geom_size.copy(),
        model.geom_type.copy(),
    )
    poisoned = early_preview(planner, data)
    assert ordinary.verified and poisoned.verified
    assert poisoned.checked_samples == ordinary.checked_samples
    for expected, actual in zip(ordinary.joint_paths, poisoned.joint_paths):
        np.testing.assert_array_equal(actual, expected)
    for expected, actual in zip(
        before,
        (
            data.qpos,
            data.qvel,
            data.ctrl,
            data.qfrc_applied,
            model.geom_size,
            model.geom_type,
        ),
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("position", [[np.nan, 0, 0.38], [0, 0], [0, 0, 0.20]])
def test_exit_preview_rejects_invalid_lift_requests(scene, position):
    _, data, _, planner = scene
    with pytest.raises(ValueError):
        planner.grasp_exit_feasible(data, 1, [-0.4, 0, 0.225], position)
