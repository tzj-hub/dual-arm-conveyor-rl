"""Fixed deep bins retain real colliding contents below the original floor."""

from dataclasses import replace
import time

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.model import build_scene, reset_robot_pose


def config(**changes):
    settings = dict(
        deep_bins=True,
        auto_unload=False,
        num_arms=4,
        strategy="adaptive",
        sensor_mode="oracle",
        bin_capacity=180,
        pool_size=384,
        physics_dt=0.0025,
    )
    settings.update(changes)
    return Config(**settings)


@pytest.mark.parametrize(
    "changes",
    [
        {"deep_bins": 1},
        {"auto_unload": True},
        {"num_arms": 2},
        {"pool_size": 367},
        {"pool_size": 1025},
        {"bin_capacity": 513},
        {"bin_capacity": 181},
        {"total_objects": 600, "pool_size": 599},
    ],
)
def test_invalid_deep_capacity_cannot_hide_resident_parts(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_deep_capacity_supports_finite_and_continuous_resident_pools():
    assert config().pool_size == 384
    batch = config(total_objects=600, pool_size=600, bin_capacity=384)
    assert batch.n_objects == 600 and not batch.auto_unload
    assert config(total_objects=10, pool_size=10).n_objects == 10
    assert (
        config(total_objects=600, pool_size=1024, bin_capacity=512).bin_capacity == 512
    )
    assert not Config().deep_bins
    with pytest.raises(ValueError):
        replace(Config(), pool_size=384)


def test_deep_bin_floor_is_world_fixed_with_compatible_locked_gate_ids():
    model, ids = build_scene(config(total_objects=8, pool_size=8))
    assert ids.bin_static_floor
    assert (
        len(ids.bin_geoms)
        == len(ids.bin_gate_joints)
        == len(ids.bin_gate_actuators)
        == 2
    )
    assert ids.bin_rim_z == pytest.approx(0.255)
    assert ids.bin_floor_z == pytest.approx(-0.545)
    assert ids.bin_rim_z - ids.bin_floor_z == pytest.approx(0.8)
    assert ids.ground_z == pytest.approx(-0.625)
    assert ids.bin_gate_open_position == 0
    np.testing.assert_array_equal(model.geom_bodyid[ids.bin_geoms], [0, 0])
    np.testing.assert_array_equal(model.actuator_gear[ids.bin_gate_actuators], 0)
    np.testing.assert_allclose(model.geom_size[ids.bin_geoms], [[0.8, 0.1, 0.01]] * 2)
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    before = data.geom_xpos[ids.bin_geoms].copy()
    data.ctrl[ids.bin_gate_actuators] = 10000
    for _ in range(100):
        mujoco.mj_step(model, data)
    np.testing.assert_array_equal(data.geom_xpos[ids.bin_geoms], before)
    np.testing.assert_allclose(data.qpos[ids.bin_gate_qpos], 0, atol=1e-12)
    assert not any(
        contact.dist < -.0005
        and (ids.robot_geom_arm[contact.geom1] >= 0 or ids.robot_geom_arm[contact.geom2] >= 0)
        for contact in data.contact
    )
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        for g in range(model.ngeom)
    ]
    assert not any(name.startswith("receiver_") for name in names)


def test_deep_walls_and_supports_are_collidable_without_changing_robot_layout():
    model, ids = build_scene(config(total_objects=8, pool_size=8))
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert model.geom_pos[ground, 2] == pytest.approx(ids.ground_z)
    for color in range(2):
        for wall in range(4):
            geom = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"binwall_{color}_{wall}"
            )
            assert model.geom_contype[geom] and model.geom_conaffinity[geom]
            assert model.geom_pos[geom, 2] - model.geom_size[geom, 2] == pytest.approx(
                ids.bin_floor_z
            )
            assert model.geom_pos[geom, 2] + model.geom_size[geom, 2] == pytest.approx(
                ids.bin_rim_z
            )
    for arm in range(4):
        geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"robot_pedestal_{arm}"
        )
        assert model.geom_contype[geom] and model.geom_conaffinity[geom]
        assert model.geom_pos[geom, 2] - model.geom_size[geom, 2] == pytest.approx(
            ids.ground_z
        )
        assert model.geom_pos[geom, 2] + model.geom_size[geom, 2] == pytest.approx(0)
    np.testing.assert_allclose(ids.home_ee[:, 2], 0.42, atol=1e-5)
    assert not any(
        c.dist < -0.0005
        and (ids.robot_geom_arm[c.geom1] >= 0 or ids.robot_geom_arm[c.geom2] >= 0)
        for c in data.contact
    )


def place(model, data, ids, index, xyz):
    address = ids.cube_qpos[index]
    data.qpos[address : address + 7] = [*xyz, 1, 0, 0, 0]
    model.body_gravcomp[ids.cube_bodies[index]] = 0


def test_real_drop_passes_old_ground_and_settles_on_fixed_deep_floor():
    model, ids = build_scene(config(total_objects=24, pool_size=24))
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    for index in range(24):
        color = index % 2
        rank = index // 2
        place(
            model,
            data,
            ids,
            index,
            [
                -0.65 + 0.26 * (rank % 6),
                ids.bin_centers[color, 1] + (-0.035 if rank < 6 else 0.035),
                0.32,
            ],
        )
    mujoco.mj_forward(model, data)
    for _ in range(1200):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert max(data.xpos[ids.cube_bodies, 2]) < -0.45
    for index, spec in enumerate(ids.object_specs):
        body = ids.cube_bodies[index]
        half = np.abs(data.xmat[body].reshape(3, 3)) @ spec.half_sizes
        assert data.xpos[body, 2] - half[2] >= ids.bin_floor_z - 0.002
        assert (
            np.linalg.norm(data.qvel[ids.cube_dofs[index] : ids.cube_dofs[index] + 3])
            < 0.04
        )
    assert np.all(model.geom_contype[ids.cube_geoms] == 1)


def test_six_hundred_retained_parts_keep_contacts_and_finite_dynamics():
    model, ids = build_scene(config(total_objects=600, pool_size=600, bin_capacity=384))
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    for index in range(600):
        color = index % 2
        rank = index // 2
        level, cell = divmod(rank, 60)
        row, column = divmod(cell, 20)
        place(
            model,
            data,
            ids,
            index,
            [
                -0.7125 + 0.075 * column,
                ids.bin_centers[color, 1] + 0.065 * (row - 1),
                ids.bin_floor_z + 0.032 + 0.075 * level,
            ],
        )
    mujoco.mj_forward(model, data)
    peak_contacts = 0
    started = time.perf_counter()
    for _ in range(240):
        mujoco.mj_step(model, data)
        peak_contacts = max(peak_contacts, data.ncon)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert peak_contacts > 600
    assert data.maxuse_arena < model.narena
    assert not any(w.number for w in data.warning)
    assert np.all(model.geom_contype[ids.cube_geoms] == 1)
    assert np.all(model.geom_conaffinity[ids.cube_geoms] == 1)
    assert len(ids.cube_geoms) == 600
    assert np.min(data.xpos[ids.cube_bodies, 2]) > ids.bin_floor_z
    assert np.max(data.xpos[ids.cube_bodies, 2]) < ids.bin_rim_z
    print(
        {
            "parts": 600,
            "simulation_seconds": 240 * model.opt.timestep,
            "wall_seconds": time.perf_counter() - started,
            "peak_contacts": peak_contacts,
            "peak_arena_bytes": int(data.maxuse_arena),
            "arena_bytes": int(model.narena),
        }
    )
