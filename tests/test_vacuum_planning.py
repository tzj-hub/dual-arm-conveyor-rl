"""Keep intentional cup contact distinct from stem and other-arm clearance."""

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.model import build_scene, reset_robot_pose
from dual_arm_conveyor.motion import MotionPlanner


@pytest.fixture
def vacuum_scene():
    model, ids = build_scene(
        Config(
            num_arms=4, strategy="adaptive", sensor_mode="oracle", gripper_mode="vacuum"
        )
    )
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    return model, data, ids


@pytest.mark.parametrize("arm", range(4))
def test_loaded_vacuum_permits_only_calibrated_own_cup_contact(vacuum_scene, arm):
    model, data, ids = vacuum_scene
    planner = MotionPlanner(model, ids)
    result = planner.plan(
        data,
        arm,
        ids.home_ee[arm] + [0.02, 0.0, 0.02],
        loaded=True,
        object_size=[0.06, 0.06, 0.04],
        grasp_offset=0.02 + 0.039,
    )
    assert result.verified, result.reason
    # 1 mm conservative box inflation plus 1 mm commanded seal compression.
    assert result.minimum_clearance == pytest.approx(-0.002, abs=1e-7)
    cup = int(ids.vacuum_geoms[arm])
    stem = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"r{arm}/vacuum_stem")
    assert planner._pair_margin(planner.proxy_geom, cup, arm, True) == -0.0021
    assert planner._pair_margin(planner.proxy_geom, stem, arm, True) == 0.0018
    assert planner._pair_margin(planner.proxy_geom, cup, arm, False) == planner.margin
    for other in range(4):
        if other == arm:
            continue
        assert (
            planner._pair_margin(
                planner.proxy_geom, int(ids.vacuum_geoms[other]), arm, True
            )
            == planner.margin
        )


def test_payload_penetrating_beyond_seal_allowance_is_rejected(vacuum_scene):
    model, data, ids = vacuum_scene
    # Leave extra room around the stem so this regression specifically tests
    # the cup's contact allowance rather than failing first on the stem.
    stem = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r0/vacuum_stem")
    model.geom_pos[stem, 2] -= 0.01
    mujoco.mj_forward(model, data)
    planner = MotionPlanner(model, ids)
    result = planner.plan(
        data,
        0,
        ids.home_ee[0],
        loaded=True,
        object_size=[0.06, 0.06, 0.046],
        grasp_offset=0.059,
    )
    assert not result.verified
    assert "clearance" in result.reason
    assert "vacuum_cup" in result.reason
    assert result.minimum_clearance < -0.0021


def test_rigid_stem_is_not_exempted_with_the_contacting_cup(vacuum_scene):
    model, data, ids = vacuum_scene
    stem = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r0/vacuum_stem")
    # Move only the rigid stem's tip 1 mm toward the object. The actual cup
    # face and its permitted seal contact stay unchanged.
    model.geom_pos[stem, 2] += 0.001
    mujoco.mj_forward(model, data)
    planner = MotionPlanner(model, ids)
    result = planner.plan(
        data,
        0,
        ids.home_ee[0],
        loaded=True,
        object_size=[0.06, 0.06, 0.04],
        grasp_offset=0.059,
    )
    assert not result.verified
    assert "vacuum_stem" in result.reason
    assert "clearance 0.0010" in result.reason


def test_vacuum_proxy_still_rejects_known_bin_collision(vacuum_scene):
    model, data, ids = vacuum_scene
    source_geometry = model.geom_size.copy()
    controls = data.ctrl.copy()
    planner = MotionPlanner(model, ids)
    oversized = planner.plan(
        data,
        0,
        ids.home_ee[0],
        loaded=True,
        object_size=[0.50, 0.50, 0.30],
        grasp_offset=0.15 + 0.039,
    )
    assert not oversized.verified
    assert "clearance" in oversized.reason
    assert oversized.minimum_clearance < planner.margin
    # Neither the source robot/scene geometry nor actual controls were changed
    # by configuring the carried-object proxy in the copied planning model.
    np.testing.assert_array_equal(model.geom_size, source_geometry)
    np.testing.assert_array_equal(data.ctrl, controls)
