"""Camera coverage, calibration and occlusion checks in the unchanged scene."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.ik import PandaIK, downward_quaternion
from dual_arm_conveyor.model import build_scene, reset_robot_pose, scene_xml
from dual_arm_conveyor.vision import CameraCalibration


CAMERAS = ("overhead", "oblique_south", "oblique_north")


@pytest.fixture(scope="module")
def scene():
    return build_scene(Config())


def camera_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)


@pytest.mark.parametrize("rig", ["overhead", "dual", "triple"])
def test_rig_selection_preserves_physics_and_environment_schema(rig):
    baseline = Config()
    config = replace(baseline, camera_rig=rig)
    assert config.schema_version == baseline.schema_version == 4
    assert config.to_dict()["camera_rig"] == rig
    assert scene_xml(config)[0] == scene_xml(baseline)[0]


@pytest.mark.parametrize("invalid", ["", "single", "stereo", "DUAL", None])
def test_unknown_camera_rig_is_rejected(invalid):
    with pytest.raises(ValueError, match="camera_rig"):
        Config(camera_rig=invalid)


def test_overhead_calibration_is_preserved_and_oblique_views_are_opposed(scene):
    model, ids = scene
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    overhead = camera_id(model, "overhead")
    assert overhead == ids.overhead_camera
    np.testing.assert_array_equal(model.cam_pos[overhead], [0, 0, 1.8])
    np.testing.assert_array_equal(model.cam_quat[overhead], [1, 0, 0, 0])
    assert model.cam_fovy[overhead] == 60
    for name, side in (("oblique_south", -1), ("oblique_north", 1)):
        camera = camera_id(model, name)
        assert camera >= 0
        position = data.cam_xpos[camera]
        rotation = data.cam_xmat[camera].reshape(3, 3)
        np.testing.assert_allclose(position, [-0.10, side * 0.50, 1.35])
        target = np.array([-0.10, 0.0, 0.23])
        direction = target - position
        np.testing.assert_allclose(
            -rotation[:, 2], direction / np.linalg.norm(direction), atol=1e-9
        )
        np.testing.assert_allclose(rotation[:, 0], [1, 0, 0])
        assert model.cam_fovy[camera] == 60
        assert 23 < np.degrees(np.arccos(rotation[2, 2])) < 25


@pytest.mark.parametrize("name", CAMERAS)
def test_grasp_window_and_complete_upstream_objects_fit_calibrated_image(scene, name):
    model, ids = scene
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    calibration = CameraCalibration.from_camera(
        model, data, camera_id(model, name), 480, 360
    )
    points = np.array(
        [
            [x, y, z]
            for x in np.linspace(-0.92, 0.69, 33)
            for y in np.linspace(-0.12, 0.12, 7)
            for z in (0.23, 0.275)
        ]
    )
    pixels = calibration.project(points)
    assert np.all(pixels >= 5)
    assert np.all(pixels <= [474, 354])
    optical_depth = -((points - calibration.position) @ calibration.rotation)[:, 2]
    assert np.all(optical_depth > 0)
    restored = calibration.unproject(pixels[:, 1], pixels[:, 0], optical_depth)
    np.testing.assert_allclose(restored, points, atol=1e-12)


def test_opposing_views_recover_workspace_hidden_by_both_lifted_arms(scene):
    model, ids = scene
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    ik = PandaIK(model, ids)
    for arm, x in ((0, -0.35), (1, 0.35)):
        target = np.array([x, 0.0, 0.40])
        orientation = downward_quaternion(0 if arm == 0 else np.pi)
        joints = ik.solve_pose(data, arm, target, orientation, max_iterations=400)
        assert ik.pose_error(data, arm, joints, target, orientation)[0] < 0.0001
        data.qpos[ids.arm_qpos[arm]] = joints
    mujoco.mj_forward(model, data)
    samples = np.array(
        [
            [x, y, 0.255]
            for x in np.linspace(-0.88, 0.65, 52)
            for y in np.linspace(-0.10, 0.10, 11)
        ]
    )
    visible = []
    for name in CAMERAS:
        origin = data.cam_xpos[camera_id(model, name)]
        mask = []
        for target in samples:
            ray = target - origin
            length = np.linalg.norm(ray)
            distance = mujoco.mj_ray(
                model,
                data,
                origin,
                ray / length,
                mujoco.MjvOption().geomgroup,
                True,
                -1,
                None,
            )
            mask.append(distance < 0 or distance >= length - 0.001)
        visible.append(mask)
    visible = np.asarray(visible)
    opposed = np.any(visible[1:], axis=0)
    assert opposed.mean() > 0.95
    assert np.sum(opposed & ~visible[0]) > 100


@pytest.mark.parametrize("name", ["oblique_south", "oblique_north"])
def test_rendered_oblique_depth_recovers_upstream_object_top(scene, name):
    model, ids = scene
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    index = next(i for i, spec in enumerate(ids.object_specs) if spec.shape == "box")
    spec = ids.object_specs[index]
    joint = ids.cube_qpos[index]
    yaw = 0.30
    data.qpos[joint : joint + 7] = [
        -0.88,
        0.06,
        0.205 + spec.half_height,
        np.cos(yaw / 2),
        0,
        0,
        np.sin(yaw / 2),
    ]
    mujoco.mj_forward(model, data)
    calibration = CameraCalibration.from_camera(
        model, data, camera_id(model, name), 480, 360
    )
    top_z = 0.205 + spec.size[2]
    u, v = np.rint(calibration.project([-0.88, 0.06, top_z])).astype(int)
    with mujoco.Renderer(model, height=360, width=480) as renderer:
        renderer.update_scene(data, camera=name)
        renderer.enable_depth_rendering()
        depth = renderer.render()
    columns, rows = np.meshgrid(np.arange(u - 2, u + 3), np.arange(v - 2, v + 3))
    recovered = calibration.unproject(rows, columns, depth[rows, columns])
    np.testing.assert_allclose(recovered[:, :, 2], top_z, atol=2e-6)
