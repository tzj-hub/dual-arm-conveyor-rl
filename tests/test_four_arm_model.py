"""Four physical Panda chains use two longitudinal bins and the same object pool."""

import sys
from itertools import product
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor import cli
from dual_arm_conveyor.config import Config
from dual_arm_conveyor.model import (
    BASE_POSITIONS,
    HOME_Q,
    arm_layout,
    build_scene,
    reset_robot_pose,
)


def test_four_arms_have_independent_joints_actuators_and_two_physical_bins():
    config = Config(strategy="adaptive", num_arms=4, sensor_mode="oracle")
    model, ids = build_scene(config)
    assert (
        ids.arm_joints.shape == ids.arm_qpos.shape == ids.arm_actuators.shape == (4, 7)
    )
    assert ids.finger_joints.shape == (4, 2) and len(ids.gripper_actuators) == 4
    assert len(np.unique(ids.arm_qpos)) == 28
    assert len(np.unique(ids.arm_actuators)) == 28
    assert ids.bin_centers.shape == (2, 3) and len(ids.bin_gate_actuators) == 2
    np.testing.assert_allclose(ids.bin_centers[:, :2], [[0, -0.28], [0, 0.28]])
    np.testing.assert_allclose(ids.bin_half_sizes, [0.8, 0.1])
    np.testing.assert_allclose(
        model.geom_size[ids.bin_geoms, :2], [[0.8, 0.1], [0.8, 0.1]]
    )
    np.testing.assert_array_equal(ids.arm_base_positions, arm_layout(4)[0])
    assert set(np.unique(ids.robot_geom_arm)) == {-1, 0, 1, 2, 3}
    counts = [np.count_nonzero(ids.robot_geom_arm == arm) for arm in range(4)]
    assert min(counts) > 0 and len(set(counts)) == 1
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    np.testing.assert_allclose(data.qpos[ids.arm_qpos], ids.home_qpos)
    np.testing.assert_allclose(data.site_xpos[ids.ee_sites], ids.home_ee)
    np.testing.assert_allclose(
        ids.home_ee,
        [
            [-0.55, -0.20, 0.42],
            [-0.55, 0.20, 0.42],
            [0.55, -0.20, 0.42],
            [0.55, 0.20, 0.42],
        ],
        atol=1e-5,
    )
    robot_contacts = [
        contact
        for contact in data.contact
        if contact.dist < -0.0005
        and (
            ids.robot_geom_arm[contact.geom1] >= 0
            or ids.robot_geom_arm[contact.geom2] >= 0
        )
    ]
    assert not robot_contacts
    assert model.nq == 4 * 9 + config.n_objects * 7 + 2


def test_original_two_arm_geometry_is_preserved():
    model, ids = build_scene(Config(sensor_mode="oracle"))
    np.testing.assert_array_equal(ids.arm_base_positions, BASE_POSITIONS)
    assert ids.arm_joints.shape == (2, 7)
    np.testing.assert_allclose(ids.home_qpos, np.tile(HOME_Q, (2, 1)))
    assert set(np.unique(ids.robot_geom_arm)) == {-1, 0, 1}
    np.testing.assert_allclose(ids.bin_centers[:, :2], [[0, -0.33], [0, 0.33]])
    np.testing.assert_allclose(ids.bin_half_sizes, [0.15, 0.1])


def test_long_bin_gates_unload_into_receivers_without_hitting_bases_or_tunneling():
    model, ids = build_scene(
        Config(strategy="adaptive", num_arms=4, sensor_mode="oracle")
    )
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    for index in range(8):
        qpos = ids.cube_qpos[index]
        data.qpos[qpos : qpos + 7] = [
            (-0.55, -0.4, 0.4, 0.55)[index // 2],
            ids.bin_centers[index % 2, 1],
            ids.bin_floor_z + ids.object_specs[index].half_height + 0.005,
            1,
            0,
            0,
            0,
        ]
        model.body_gravcomp[ids.cube_bodies[index]] = 0
    mujoco.mj_forward(model, data)
    opened = False
    for step in range(2000):
        target = ids.bin_gate_open_position if 150 <= step < 1000 else 0
        command = data.ctrl[ids.bin_gate_actuators]
        data.ctrl[ids.bin_gate_actuators] += np.clip(
            target - command, -0.6 * model.opt.timestep, 0.6 * model.opt.timestep
        )
        mujoco.mj_step(model, data)
        opened |= bool(
            np.all(data.qpos[ids.bin_gate_qpos] > ids.bin_gate_open_position - 0.02)
        )
        assert not any(
            contact.dist < -0.0005
            and (
                ids.robot_geom_arm[contact.geom1] >= 0
                or ids.robot_geom_arm[contact.geom2] >= 0
            )
            for contact in data.contact
        )
    assert opened and np.max(data.qpos[ids.bin_gate_qpos]) < 0.005
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    for index in range(8):
        body, spec = ids.cube_bodies[index], ids.object_specs[index]
        position = data.xpos[body]
        assert abs(position[0]) < ids.bin_half_sizes[0]
        assert abs(position[1] - ids.bin_centers[index % 2, 1]) < ids.bin_half_sizes[1]
        vertices = (
            spec.prism_mesh()[0]
            if spec.prism_sides
            else np.array(list(product((-1, 1), repeat=3)))
            * np.asarray(spec.half_sizes)
        )
        world_vertices = vertices @ data.xmat[body].reshape(3, 3).T + position
        assert world_vertices[:, 2].min() >= ids.bin_receiver_floor_z - 0.001
        assert world_vertices[:, 2].max() < ids.bin_floor_z - 0.012


@pytest.mark.parametrize(
    "changes",
    [
        {"num_arms": 3},
        {"num_arms": 4.0},
        {"num_arms": True},
        {"num_arms": 4},
        {"num_arms": 4, "strategy": "adaptive", "motion_mode": "legacy"},
        {"num_arms": 4, "strategy": "adaptive", "wrist_camera_mode": "observe"},
    ],
)
def test_unsupported_arm_counts_and_experimental_combinations_rejected(changes):
    with pytest.raises(ValueError):
        Config(**changes)


def test_four_arm_and_catalogue_seed_options_reach_config(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "dual_arm_conveyor.env",
        SimpleNamespace(ConveyorEnv=lambda config: config),
    )
    args = cli.parser().parse_args(
        [
            "evaluate",
            "--num-arms",
            "4",
            "--strategy",
            "adaptive",
            "--catalog-seed",
            "75",
        ]
    )
    config = cli._make_env(args)
    assert config.num_arms == 4 and config.seed == 75


def test_four_arm_mode_rejects_incompatible_learned_checkpoint_metadata(monkeypatch):
    from dual_arm_conveyor import learning

    monkeypatch.setattr(
        learning, "load_policy", lambda _: SimpleNamespace(metadata={"env_config": {"num_arms": 2}})
    )
    env = SimpleNamespace(config=Config(strategy="adaptive", num_arms=4))
    with pytest.raises(ValueError, match="different number of arms"):
        cli._policy_action("does-not-exist.pt", env)


def test_metal_bin_priority_is_local_and_effective_at_actual_contacts():
    model, ids = build_scene(
        Config(strategy="adaptive", num_arms=4, sensor_mode="oracle", physics_dt=0.0025)
    )
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    slot = 2  # Soft-contact catalogue slot, independent of its sampled shape.
    obj = int(ids.cube_geoms[slot])
    gate = int(ids.bin_geoms[0])
    walls = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"binwall_{c}_{k}")
        for c in range(2)
        for k in range(4)
    ]
    np.testing.assert_array_equal(model.geom_priority[list(ids.bin_geoms) + walls], 1)
    np.testing.assert_array_equal(model.geom_priority[ids.cube_geoms], 0)
    assert model.geom_priority[ids.belt_geom] == 0
    for geom, spec in zip(ids.cube_geoms, ids.object_specs):
        np.testing.assert_allclose(model.geom_solref[geom], spec.solref)
        np.testing.assert_allclose(model.geom_friction[geom], spec.friction)

    def object_contact(surface, xy, top):
        q = ids.cube_qpos[slot]
        data.qpos[q : q + 7] = (
            *xy,
            top + ids.object_specs[slot].half_height - 0.0001,
            1,
            0,
            0,
            0,
        )
        mujoco.mj_forward(model, data)
        return next(
            contact
            for contact in data.contact
            if {int(contact.geom1), int(contact.geom2)} == {obj, int(surface)}
        )

    floor_contact = object_contact(
        gate, (-0.55, ids.bin_centers[0, 1]), ids.bin_floor_z
    )
    np.testing.assert_allclose(floor_contact.solref, [0.008, 1])
    np.testing.assert_allclose(floor_contact.solimp, model.geom_solimp[gate])
    np.testing.assert_allclose(floor_contact.friction[:2], model.geom_friction[gate, 0])
    # Moving this same workpiece back to the belt restores the ordinary mixed
    # material contact rather than propagating the bin's metal override.
    belt_contact = object_contact(ids.belt_geom, (0, 0), 0.205)
    expected = (model.geom_solref[obj] + model.geom_solref[ids.belt_geom]) / 2
    np.testing.assert_allclose(belt_contact.solref, expected)
    assert belt_contact.solref[0] > 0.008


def test_two_arm_bin_contact_and_default_timestep_remain_unchanged():
    config = Config(sensor_mode="oracle")
    assert config.physics_dt == 0.005
    model, ids = build_scene(config)
    np.testing.assert_array_equal(model.geom_priority[ids.bin_geoms], 0)
    np.testing.assert_allclose(model.geom_solref[ids.bin_geoms], [[0.02, 1], [0.02, 1]])
