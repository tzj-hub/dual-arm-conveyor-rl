"""Moving visual stripes preserve belt physics and obey the calibrated speed."""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.belt_visual import (
    BeltMarkers,
    MARKER_HALF_WIDTH,
    append_belt_markers,
)


def _config():
    return SimpleNamespace(
        belt_half_length=1.15, belt_half_width=0.16, belt_top=0.205, belt_speed=0.1
    )


def _model(markers=True):
    config = _config()
    root = ET.fromstring("""<mujoco>
      <option timestep="0.005"/>
      <worldbody>
        <geom name="belt" type="box" pos="0 0 0.15" size="1.15 .16 .055"/>
        <body name="object" pos="0 0 .3">
          <freejoint/><geom name="object_geom" type="box" size=".025 .025 .025" mass=".04"/>
        </body>
      </worldbody>
    </mujoco>""")
    if markers:
        append_belt_markers(root.find("worldbody"), config)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, config


def test_marker_geometry_is_massless_gray_noncolliding_and_fits_the_belt():
    model, config = _model()
    markers = BeltMarkers(model, config)
    data = mujoco.MjData(model)
    assert len(markers.mocap_ids) == 12
    geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"belt_marker_geom_{i}")
        for i in range(12)
    ]
    bodies = model.geom_bodyid[geom_ids]
    assert np.all(model.geom_contype[geom_ids] == 0)
    assert np.all(model.geom_conaffinity[geom_ids] == 0)
    assert np.all(model.geom_group[geom_ids] == 2)
    assert np.all(model.body_mass[bodies] == 0)
    assert np.all(np.ptp(model.geom_rgba[geom_ids, :3], axis=1) < 0.1)
    np.testing.assert_allclose(model.geom_size[geom_ids, 1], config.belt_half_width)
    centers = data.mocap_pos[markers.mocap_ids, 0]
    assert np.all(centers - MARKER_HALF_WIDTH >= -config.belt_half_length - 1e-10)
    assert np.all(centers + MARKER_HALF_WIDTH <= config.belt_half_length + 1e-10)
    assert np.all((np.diff(centers) >= 0.18) & (np.diff(centers) <= 0.20))


def test_stripes_move_at_belt_speed_and_wrap_without_endpoint_overflow():
    model, config = _model()
    markers = BeltMarkers(model, config)
    data = mujoco.MjData(model)
    markers.update(data)
    initial = data.mocap_pos.copy()
    data.time = 0.5
    markers.update(data)
    np.testing.assert_allclose(data.mocap_pos[:, 0] - initial[:, 0], 0.05, atol=1e-12)
    np.testing.assert_array_equal(data.mocap_pos[:, 1:], initial[:, 1:])
    for now in (22.9, 22.96, 23.0, 1000.0, 100000.0):
        data.time = now
        markers.update(data)
        x = data.mocap_pos[markers.mocap_ids, 0]
        assert np.all(x - MARKER_HALF_WIDTH >= -config.belt_half_length - 1e-10)
        assert np.all(x + MARKER_HALF_WIDTH <= config.belt_half_length + 1e-10)
        circular_gaps = np.diff(np.r_[np.sort(x), np.min(x) + markers.span])
        np.testing.assert_allclose(circular_gaps, markers.span / len(x), atol=1e-10)
    data.time = markers.span / config.belt_speed
    markers.update(data)
    np.testing.assert_allclose(data.mocap_pos, initial, atol=1e-12)
    data.time = 0
    markers.update(data)
    np.testing.assert_array_equal(data.mocap_pos, initial)


def test_update_reads_only_time_and_mocap_positions_without_forward(monkeypatch):
    model, config = _model()
    markers = BeltMarkers(model, config)

    class MinimalData:
        time = 0.5
        mocap_pos = np.zeros((model.nmocap, 3))

        def __getattr__(self, name):
            raise AssertionError(f"Stripe animation accessed {name}")

    def forbidden(*args, **kwargs):
        raise AssertionError("Stripe update must not run an extra physics/forward pass")

    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    data = MinimalData()
    markers.update(data)
    assert np.count_nonzero(data.mocap_pos[:, 0]) == model.nmocap


def test_moving_stripes_do_not_change_contact_dynamics_or_degrees_of_freedom():
    plain, _ = _model(markers=False)
    marked, config = _model(markers=True)
    assert plain.nq == marked.nq and plain.nv == marked.nv
    assert marked.body_mass.sum() == plain.body_mass.sum()
    left, right = mujoco.MjData(plain), mujoco.MjData(marked)
    markers = BeltMarkers(marked, config)
    for _ in range(160):
        markers.update(right)
        mujoco.mj_step(plain, left)
        mujoco.mj_step(marked, right)
        np.testing.assert_array_equal(left.qpos, right.qpos)
        np.testing.assert_array_equal(left.qvel, right.qvel)
        assert left.ncon == right.ncon
    assert left.ncon > 0  # the falling object actually reached the physical belt


def test_missing_marker_bodies_have_an_actionable_error():
    model, config = _model(markers=False)
    with pytest.raises(ValueError, match="append_belt_markers"):
        BeltMarkers(model, config)
