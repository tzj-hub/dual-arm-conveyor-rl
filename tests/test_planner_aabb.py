"""The additional broad phase must only discard provably separated pairs."""
from __future__ import annotations

import itertools

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.motion import MotionPlanner


def toy_planner():
    model = mujoco.MjModel.from_xml_string('''
    <mujoco>
      <asset><mesh name="asymmetric" vertex="-.3 -.1 -.1 .6 -.1 -.1 -.3 .2 -.1 -.3 -.1 .4 .6 .2 .4"/></asset>
      <worldbody>
        <geom name="plane" type="plane" size="3 3 .1"/>
        <body><freejoint/><geom name="sphere" type="sphere" size=".12"/></body>
        <body><freejoint/><geom name="capsule" type="capsule" size=".07 .25"/></body>
        <body><freejoint/><geom name="ellipsoid" type="ellipsoid" size=".12 .27 .18"/></body>
        <body><freejoint/><geom name="cylinder" type="cylinder" size=".15 .24"/></body>
        <body><freejoint/><geom name="box" type="box" size=".80 .10 .06"/></body>
        <body><freejoint/><geom name="mesh" type="mesh" mesh="asymmetric"/></body>
      </worldbody>
    </mujoco>''')
    planner = object.__new__(MotionPlanner)
    planner.model = model
    planner.scratch = mujoco.MjData(model)
    planner._initialize_aabb_bounds()
    return planner


def test_rotated_geometry_pair_bounds_are_conservative():
    planner = toy_planner()
    pairs = np.asarray(list(itertools.combinations(range(1, 7), 2)))
    rng = np.random.default_rng(87002)
    checked = 0
    for _ in range(80):
        for j in range(planner.model.njnt):
            q = planner.model.jnt_qposadr[j]
            planner.scratch.qpos[q:q + 3] = rng.uniform(-.7, .7, 3)
            rotation = rng.normal(size=4)
            planner.scratch.qpos[q + 3:q + 7] = rotation / np.linalg.norm(rotation)
        mujoco.mj_kinematics(planner.model, planner.scratch)
        bounds = planner._aabb_lower_bounds(*pairs.T)
        for (first, second), lower in zip(pairs, bounds):
            if lower >= .008:
                exact = mujoco.mj_geomDistance(planner.model, planner.scratch, int(first), int(second), 100, None)
                assert exact + 1e-8 >= lower
                assert exact >= .008
                checked += 1
    assert checked > 300


def test_changed_box_proxy_uses_current_type_size_not_compiled_aabb():
    planner = toy_planner()
    model, data = planner.model, planner.scratch
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    sphere = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
    sphere_joint = model.body_jntadr[model.geom_bodyid[sphere]]
    q = model.jnt_qposadr[sphere_joint]
    data.qpos[q:q + 3] = [1.2, 0, 0]
    model.geom_size[box] = [1.5, .25, .25]
    mujoco.mj_kinematics(model, data)
    bound = planner._aabb_lower_bounds(np.array([box]), np.array([sphere]))[0]
    assert bound < .008
    assert mujoco.mj_geomDistance(model, data, box, sphere, .008, None) < .008


def test_planes_never_pruned_by_finite_aabb():
    planner = toy_planner()
    mujoco.mj_kinematics(planner.model, planner.scratch)
    result = planner._aabb_lower_bounds(np.array([0]), np.array([1]))
    assert np.isneginf(result[0])


@pytest.mark.parametrize("distance", [.008 - 1e-8, .008, .008 + 1e-8])
def test_near_margin_has_conservative_roundoff(distance):
    planner = toy_planner()
    model, data = planner.model, planner.scratch
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    sphere = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
    q = model.jnt_qposadr[model.body_jntadr[model.geom_bodyid[sphere]]]
    data.qpos[q:q + 3] = [.8 + .12 + distance, 0, 0]
    mujoco.mj_kinematics(model, data)
    lower = planner._aabb_lower_bounds(np.array([box]), np.array([sphere]))[0]
    exact = mujoco.mj_geomDistance(model, data, box, sphere, .1, None)
    assert lower <= exact + 1e-12
    if distance <= .008:
        assert lower < .008
