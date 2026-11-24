"""The display camera must include the full four-arm unloading stroke."""

from itertools import product

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.cli import _camera
from dual_arm_conveyor.config import Config
from dual_arm_conveyor.model import build_scene, reset_robot_pose


def test_default_observer_retains_two_arm_framing():
    default = _camera(mujoco)
    explicit = _camera(mujoco, num_arms=2)
    np.testing.assert_array_equal(default.lookat, [0, 0, .45])
    np.testing.assert_array_equal(default.lookat, explicit.lookat)
    assert default.distance == explicit.distance == 2.8
    assert default.azimuth == explicit.azimuth == 135
    assert default.elevation == explicit.elevation == -42


@pytest.fixture(scope="module")
def fully_open_gates():
    model, ids = build_scene(
        Config(strategy="adaptive", num_arms=4, sensor_mode="oracle")
    )
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    data.qpos[ids.bin_gate_qpos] = ids.bin_gate_open_position
    mujoco.mj_forward(model, data)
    corners = np.asarray(list(product((-1, 1), repeat=3)))
    points = np.concatenate(
        [
            (corners * model.geom_size[geom])
            @ data.geom_xmat[geom].reshape(3, 3).T
            + data.geom_xpos[geom]
            for geom in ids.bin_geoms
        ]
    )
    return model, data, points


@pytest.mark.parametrize("inset", [False, True])
@pytest.mark.parametrize("aspect", [960 / 720, 1280 / 800])
def test_four_arm_observer_includes_both_fully_open_gates(
    fully_open_gates, inset, aspect
):
    model, data, points = fully_open_gates
    scene = mujoco.MjvScene(model, maxgeom=1000)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        _camera(mujoco, num_arms=4, inset=inset),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    camera = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    relative = points - camera.pos
    depth = relative @ camera.forward
    assert np.all(depth > camera.frustum_near)
    right = np.cross(camera.forward, camera.up)
    height = camera.frustum_top - camera.frustum_bottom
    horizontal = (
        (relative @ right) * camera.frustum_near / depth - camera.frustum_center
    ) / (height * aspect) + .5
    vertical = (
        (relative @ camera.up) * camera.frustum_near / depth - camera.frustum_bottom
    ) / height
    # Check actual projected gate corners, rather than only literal camera
    # parameters. The old two-arm view clips the full-length open gates.
    assert np.all((horizontal > .02) & (horizontal < .98))
    assert np.all((vertical > .02) & (vertical < .98))
