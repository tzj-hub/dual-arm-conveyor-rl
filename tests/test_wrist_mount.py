"""Mounted sensor calibration, real visibility and unchanged robot dynamics."""

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.ik import PandaIK, downward_quaternion
from dual_arm_conveyor.model import build_scene, reset_robot_pose
from dual_arm_conveyor.vision import CameraCalibration


@pytest.fixture(scope="module")
def scenes():
    return {
        mode: build_scene(Config(wrist_camera_mode=mode))
        for mode in ("off", "observe", "refine")
    }


def object_id(model, kind, name):
    return mujoco.mj_name2id(model, getattr(mujoco.mjtObj, f"mjOBJ_{kind}"), name)


def test_off_keeps_wrist_camera_and_mount_absent(scenes):
    model, _ = scenes["off"]
    for arm in range(2):
        assert object_id(model, "CAMERA", f"r{arm}/wrist_rgbd") == -1
        assert object_id(model, "BODY", f"r{arm}/wrist_camera_mount") == -1
        for part in ("housing", "lens", "bracket"):
            assert object_id(model, "GEOM", f"r{arm}/wrist_camera_{part}") == -1


@pytest.mark.parametrize("mode", ["observe", "refine"])
def test_enabled_visual_mount_adds_no_mass_contacts_or_dynamics(scenes, mode):
    baseline, baseline_ids = scenes["off"]
    model, ids = scenes[mode]
    assert (model.nq, model.nv, model.nu) == (
        baseline.nq,
        baseline.nv,
        baseline.nu,
    )
    for arm in range(2):
        mount = object_id(model, "BODY", f"r{arm}/wrist_camera_mount")
        hand = object_id(model, "BODY", f"r{arm}/hand")
        assert model.body_parentid[mount] == hand
        assert model.body_mass[mount] == 0
        np.testing.assert_array_equal(model.body_inertia[mount], 0)
        for part in ("housing", "lens", "bracket"):
            geom = object_id(model, "GEOM", f"r{arm}/wrist_camera_{part}")
            assert model.geom_contype[geom] == model.geom_conaffinity[geom] == 0
    before = mujoco.MjData(baseline)
    after = mujoco.MjData(model)
    reset_robot_pose(baseline, before, baseline_ids)
    reset_robot_pose(model, after, ids)
    before_mass = before.M if hasattr(before, "M") else before.qM
    after_mass = after.M if hasattr(after, "M") else after.qM
    np.testing.assert_allclose(after_mass, before_mass, atol=1e-12, rtol=0)
    np.testing.assert_allclose(after.qfrc_bias, before.qfrc_bias, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(model.actuator_gainprm, baseline.actuator_gainprm)
    np.testing.assert_array_equal(model.actuator_biasprm, baseline.actuator_biasprm)
    assert model.vis.map.znear * model.stat.extent < 0.012
    assert baseline.vis.map.znear == pytest.approx(0.01)


@pytest.mark.parametrize("arm", [0, 1])
def test_camera_extrinsics_follow_hand_and_lens_is_in_front_of_housing(scenes, arm):
    model, ids = scenes["observe"]
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    ik = PandaIK(model, ids)
    hand = object_id(model, "BODY", f"r{arm}/hand")
    camera = object_id(model, "CAMERA", f"r{arm}/wrist_rgbd")
    mount = object_id(model, "BODY", f"r{arm}/wrist_camera_mount")
    assert model.cam_bodyid[camera] == mount
    assert model.cam_fovy[camera] == 70
    poses = []
    for yaw, height in ((-0.6, 0.36), (0.8, 0.26)):
        target = np.array([-0.4 if arm == 0 else 0.4, 0.0, height])
        quat = downward_quaternion(yaw + arm * np.pi)
        joints = ik.solve_pose(data, arm, target, quat, max_iterations=400)
        assert ik.pose_error(data, arm, joints, target, quat)[0] < 0.0001
        data.qpos[ids.arm_qpos[arm]] = joints
        mujoco.mj_forward(model, data)
        hand_rotation = data.xmat[hand].reshape(3, 3)
        position = data.cam_xpos[camera].copy()
        np.testing.assert_allclose(
            (position - data.xpos[hand]) @ hand_rotation,
            [0.11, 0.0, 0.0],
            atol=1e-9,
        )
        rotation = data.cam_xmat[camera].reshape(3, 3)
        np.testing.assert_allclose(
            hand_rotation.T @ (-rotation[:, 2]),
            np.array([-0.11, 0, 0.12]) / np.linalg.norm([0.11, 0, -0.12]),
            atol=1e-9,
        )
        # Lens/housing are entirely behind the optical origin in camera space.
        for part in ("housing", "lens"):
            geom = object_id(model, "GEOM", f"r{arm}/wrist_camera_{part}")
            local_centre = (data.geom_xpos[geom] - position) @ rotation
            assert local_centre[2] > model.geom_size[geom, 2]
        poses.append(position)
    assert np.linalg.norm(poses[1] - poses[0]) > 0.08


@pytest.mark.parametrize("arm,yaw", [(0, -0.7), (1, 0.8)])
@pytest.mark.parametrize("height_above_final", [0.13, 0.05, 0.02])
def test_real_rgbd_keeps_box_top_visible_until_two_cm_above_final(
    scenes, arm, yaw, height_above_final
):
    model, ids = scenes["observe"]
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    index = next(i for i, spec in enumerate(ids.object_specs) if spec.shape == "box")
    spec = ids.object_specs[index]
    position = np.array([-0.4 if arm == 0 else 0.4, 0.0, 0.205 + spec.half_height])
    qpos = ids.cube_qpos[index]
    data.qpos[qpos : qpos + 7] = [
        *position,
        np.cos(yaw / 2),
        0,
        0,
        np.sin(yaw / 2),
    ]
    target = position + [
        0,
        0,
        max(-0.009, spec.half_height - 0.030) + height_above_final,
    ]
    ik = PandaIK(model, ids)
    joints = ik.solve_pose(
        data, arm, target, downward_quaternion(yaw + arm * np.pi), max_iterations=400
    )
    data.qpos[ids.arm_qpos[arm]] = joints
    mujoco.mj_forward(model, data)
    name = f"r{arm}/wrist_rgbd"
    camera = object_id(model, "CAMERA", name)
    calibration = CameraCalibration.from_camera(model, data, camera, 320, 240)
    top = np.r_[position[:2], 0.205 + spec.height]
    u, v = np.rint(calibration.project(top)).astype(int)
    assert 3 <= u < 317 and 3 <= v < 237
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0  # Simulator debug markers are not physical occluders.
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.update_scene(data, camera=name, scene_option=option)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()  # Evaluator-only visibility check.
        segmentation = renderer.render()
    rows, columns = np.meshgrid(np.arange(v - 2, v + 3), np.arange(u - 2, u + 3))
    assert np.all(segmentation[rows, columns, 0] == ids.cube_geoms[index])
    assert np.any(rgb[rows, columns].max(axis=-1) > 60)
    recovered = calibration.unproject(rows, columns, depth[rows, columns])
    np.testing.assert_allclose(recovered[..., 2], top[2], atol=2e-6)
