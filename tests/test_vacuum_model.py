"""Optional suction hardware has real mass, collision, and calibrated face pose."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor import cli
from dual_arm_conveyor.config import Config
from dual_arm_conveyor.model import build_scene, reset_robot_pose


@pytest.mark.parametrize("num_arms", [2, 4])
def test_vacuum_tools_add_mass_and_collision_without_extra_joints(num_arms):
    config = Config(strategy="adaptive", num_arms=num_arms, sensor_mode="oracle")
    fingers, original = build_scene(config)
    vacuum, ids = build_scene(replace(config, gripper_mode="vacuum"))
    assert (
        original.vacuum_sites is original.vacuum_geoms is original.vacuum_bodies is None
    )
    assert vacuum.nq == fingers.nq and vacuum.nv == fingers.nv
    assert vacuum.neq == fingers.neq  # No hidden attachment/weld equality.
    assert vacuum.body_mass.sum() - fingers.body_mass.sum() == pytest.approx(
        0.03 * num_arms
    )
    np.testing.assert_allclose(vacuum.body_mass[ids.vacuum_bodies], 0.03)
    assert np.all(vacuum.body_inertia[ids.vacuum_bodies] > 0)
    assert np.all(vacuum.geom_contype[ids.vacuum_geoms] == 1)
    assert np.all(vacuum.geom_conaffinity[ids.vacuum_geoms] == 1)
    assert np.all(vacuum.geom_type[ids.vacuum_geoms] == mujoco.mjtGeom.mjGEOM_CYLINDER)
    np.testing.assert_allclose(
        vacuum.geom_size[ids.vacuum_geoms, :2], np.tile([0.012, 0.002], (num_arms, 1))
    )
    assert set(vacuum.geom_bodyid[ids.vacuum_geoms]) == set(ids.vacuum_bodies)


@pytest.mark.parametrize("num_arms", [2, 4])
def test_cup_faces_follow_hand_fk_and_extend_below_open_fingers(num_arms):
    model, ids = build_scene(
        Config(strategy="adaptive", num_arms=num_arms, gripper_mode="vacuum")
    )
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    for arm in range(num_arms):
        site = ids.ee_sites[arm]
        face = ids.vacuum_sites[arm]
        normal = data.site_xmat[face].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(
            data.site_xpos[face] - data.site_xpos[site], normal * 0.04, atol=1e-9
        )
        np.testing.assert_allclose(
            data.site_xmat[face], data.site_xmat[site], atol=1e-9
        )
        np.testing.assert_allclose(normal, [0, 0, -1], atol=1e-5)
        stem = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"r{arm}/vacuum_stem")
        assert stem >= 0 and stem != ids.vacuum_geoms[arm]
        assert model.geom_contype[stem] and model.geom_conaffinity[stem]
    assert not any(
        contact.dist < -0.0005
        for contact in data.contact
        if ids.robot_geom_arm[contact.geom1] >= 0
        or ids.robot_geom_arm[contact.geom2] >= 0
    )


def test_flat_object_physically_contacts_cup_before_any_open_finger():
    model, ids = build_scene(
        Config(strategy="adaptive", num_arms=4, gripper_mode="vacuum")
    )
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    index = next(i for i, spec in enumerate(ids.object_specs) if spec.shape == "box")
    spec = ids.object_specs[index]
    face = data.site_xpos[ids.vacuum_sites[0]].copy()
    position = face - [0, 0, spec.half_height - 0.0002]
    qpos = ids.cube_qpos[index]
    data.qpos[qpos : qpos + 7] = [*position, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    object_geom = ids.cube_geoms[index]
    other_geoms = [
        contact.geom2 if contact.geom1 == object_geom else contact.geom1
        for contact in data.contact
        if object_geom in (contact.geom1, contact.geom2)
    ]
    assert ids.vacuum_geoms[0] in other_geoms
    assert not any(
        model.geom_bodyid[geom] in ids.finger_bodies[0] for geom in other_geoms
    )
    assert not any(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) == "r0/vacuum_stem"
        for geom in other_geoms
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"gripper_mode": "magnet"},
        {"gripper_mode": "vacuum"},
        {
            "gripper_mode": "vacuum",
            "strategy": "adaptive",
            "wrist_camera_mode": "observe",
        },
        {"gripper_mode": "vacuum", "strategy": "adaptive", "motion_mode": "legacy"},
    ],
)
def test_uncalibrated_vacuum_combinations_are_explicitly_rejected(changes):
    with pytest.raises(ValueError):
        Config(**changes)


def test_vacuum_cli_is_explicit_and_finger_default_unchanged():
    assert (
        cli.config_from_args(cli.parser().parse_args(["play"])).gripper_mode
        == "fingers"
    )
    config = cli.config_from_args(
        cli.parser().parse_args(
            [
                "play",
                "--strategy",
                "adaptive",
                "--num-arms",
                "4",
                "--gripper-mode",
                "vacuum",
            ]
        )
    )
    assert config.gripper_mode == "vacuum" and config.num_arms == 4
