"""World-space fusion and real rendered oblique RGB-D grasp geometry."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from dataclasses import replace
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.objects import ObjectSpec
from dual_arm_conveyor.vision import (
    CameraCalibration,
    Detection,
    SensorPipeline,
    fit_prism_outline,
    fuse_camera_detections,
    symmetric_angle,
)


def measured(x=0.0, y=0.0, kind=0, yaw=0.0, size=(0.06, 0.04, 0.05)):
    return Detection(
        np.array([x, y, 0.23]),
        yaw,
        kind,
        0.9,
        (0, 0, 10, 10),
        np.array(size),
        "prism",
        math.pi,
    )


def test_duplicate_views_keep_one_measured_grasp_without_averaging_equivalent_yaws():
    a = measured(yaw=0.0)
    b = replace(a, yaw=math.pi / 2, uncertainty=0.003)
    result = fuse_camera_detections({"south": [a], "north": [b]})
    assert len(result) == 1
    assert result[0].yaw == b.yaw
    assert result[0].source_cameras == ("south", "north")


def test_occluded_fragment_does_not_drag_complete_view_centre():
    whole = measured()
    fragment = replace(
        whole,
        position=whole.position + [0.018, 0, 0],
        size=np.array([0.032, 0.04, 0.05]),
        uncertainty=0.001,
    )
    result = fuse_camera_detections({"south": [fragment], "north": [whole]})
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].position, whole.position)
    np.testing.assert_array_equal(result[0].size, whole.size)


def test_complete_overhead_retains_geometry_despite_sharper_oblique_view():
    overhead = replace(measured(yaw=0.35), footprint_pixel_size=0.005)
    oblique = replace(
        overhead,
        position=overhead.position + [0.001, -0.001, 0],
        yaw=overhead.yaw + math.pi / 2,
        size=overhead.size[[1, 0, 2]],
        uncertainty=0.001,
        footprint_pixel_size=0.003,
    )
    result = fuse_camera_detections({"overhead": [overhead], "south": [oblique]})
    assert len(result) == 1
    assert result[0].yaw == overhead.yaw
    np.testing.assert_array_equal(result[0].position, overhead.position)
    np.testing.assert_array_equal(result[0].size, overhead.size)
    assert result[0].uncertainty == overhead.uncertainty


@pytest.mark.parametrize("visible_fraction", [0.89, 0.91])
def test_overhead_priority_requires_ninety_percent_of_largest_measured_area(
    visible_fraction,
):
    oblique = replace(measured(), uncertainty=0.001)
    overhead = replace(
        measured(x=0.003),
        size=oblique.size * [visible_fraction, 1, 1],
    )
    result = fuse_camera_detections({"overhead": [overhead], "south": [oblique]})
    assert len(result) == 1
    expected = overhead if visible_fraction >= 0.90 else oblique
    np.testing.assert_array_equal(result[0].position, expected.position)
    np.testing.assert_array_equal(result[0].size, expected.size)


@pytest.mark.parametrize("size", [(0.08, 0.03, 0.05), (0.06, 0.04, 0.058)])
def test_overhead_priority_rejects_inconsistent_dimensions_or_height(size):
    oblique = replace(measured(), uncertainty=0.001)
    overhead = replace(
        measured(x=0.002, size=size),
        uncertainty=0.0025,
    )
    result = fuse_camera_detections({"overhead": [overhead], "south": [oblique]})
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].position, oblique.position)
    np.testing.assert_array_equal(result[0].size, oblique.size)


def test_missing_overhead_preserves_existing_oblique_selection():
    south = measured(x=0.001, yaw=-0.3)
    north = replace(south, position=south.position + [0.001, 0, 0], uncertainty=0.001)
    dual = fuse_camera_detections({"south": [south], "north": [north]})
    triple = fuse_camera_detections(
        {"overhead": [], "south": [south], "north": [north]}
    )
    assert len(dual) == len(triple) == 1
    np.testing.assert_array_equal(dual[0].position, north.position)
    np.testing.assert_array_equal(triple[0].position, dual[0].position)
    assert dual[0].yaw == triple[0].yaw == north.yaw


def test_close_same_colour_objects_remain_distinct_with_one_detection_per_camera():
    south = [measured(x=0.0), measured(x=0.025)]
    north = [measured(x=0.004), measured(x=0.029)]
    result = fuse_camera_detections({"south": south, "north": north})
    assert len(result) == 2
    assert all(item.source_cameras == ("south", "north") for item in result)
    np.testing.assert_allclose([item.position[0] for item in result], [0, 0.025])


def test_association_is_complete_link_and_cannot_chain_neighbours_together():
    result = fuse_camera_detections(
        {
            "overhead": [measured(x=0)],
            "south": [measured(x=0.025)],
            "north": [measured(x=0.05)],
        }
    )
    assert len(result) == 2
    assert all(len(item.source_cameras) <= 2 for item in result)


def test_colour_and_measured_height_block_incompatible_cross_view_merges():
    red = measured()
    blue = replace(red, kind=1)
    taller = replace(
        red, position=red.position + [0, 0, 0.02], size=red.size + [0, 0, 0.04]
    )
    result = fuse_camera_detections({"south": [red], "north": [blue, taller]})
    assert len(result) == 3


def test_tilted_plane_pixel_support_uses_both_projected_image_axes():
    corners = np.array([[-0.03, -0.02], [0.03, -0.02], [0.03, 0.02], [-0.03, 0.02]])
    axes = ((0.003, 0.001), (0.001, 0.005))
    centre, yaw, size, _ = fit_prism_outline(corners, 0.005, pixel_axes=axes)
    np.testing.assert_allclose(centre, 0, atol=1e-12)
    assert yaw == 0
    np.testing.assert_allclose(size, [0.064, 0.046], atol=1e-12)


def fixture(shape="box", blocker=False, second=False):
    size = np.array([0.055, 0.045, 0.05])
    if shape == "box":
        asset = ""
        geom = 'type="box" size=".0275 .0225 .025"'
        outline = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size[:2] / 2
    else:
        spec = ObjectSpec(
            shape, tuple(size), 0.05, (1, 0.01, 0.001), (0.012, 1), "fixture"
        )
        vertices, faces = spec.prism_mesh()
        outline = vertices[: spec.prism_sides, :2]
        asset = (
            '<asset><mesh name="measured_poly" vertex="'
            + " ".join(map(str, vertices.ravel()))
            + '" face="'
            + " ".join(map(str, faces.ravel()))
            + '"/></asset>'
        )
        geom = 'type="mesh" mesh="measured_poly"'
    cameras = '<camera name="overhead" pos="0 0 1.8" fovy="60"/>'
    for name, side in (("oblique_south", -1), ("oblique_north", 1)):
        cameras += (
            f'<camera name="{name}" pos="-.10 {side * 0.50} 1.35" '
            f'xyaxes="1 0 0 0 1.12 {-side * 0.50}" fovy="60"/>'
        )
    obstruction = (
        (
            '<geom type="box" pos="-.295 .025 .50" size=".04 .04 .015" rgba=".5 .5 .5 1"/>'
        )
        if blocker
        else ""
    )
    extra = (
        f'<body pos="-.275 .03 .23"><geom {geom} rgba=".87 .13 .08 1"/></body>'
        if second
        else ""
    )
    model = mujoco.MjModel.from_xml_string(f"""<mujoco>
        <visual><global offwidth="640" offheight="480"/><quality offsamples="0"/></visual>
        {asset}<worldbody><light pos="0 0 3"/>{cameras}
        <geom type="box" pos="0 0 .15" size="1.15 .16 .055" rgba=".1 .15 .2 1"/>
        <body pos="-.35 .03 .23"><geom {geom} rgba=".87 .13 .08 1"/></body>
        {extra}{obstruction}</worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, outline


def config(rig):
    return SimpleNamespace(
        belt_top=0.205,
        belt_half_length=1.15,
        belt_half_width=0.16,
        belt_speed=0.1,
        camera_width=480,
        camera_height=360,
        camera_hz=10.0,
        camera_latency=0.0,
        camera_dropout=0.0,
        depth_noise=0.001,
        camera_rig=rig,
        seed=17,
        object_family="flat_prisms",
    )


@pytest.mark.parametrize("name", ["oblique_south", "oblique_north"])
def test_pixel_plane_intersections_and_depth_range_are_calibration_only(name):
    model, data, _ = fixture()
    sensor = SensorPipeline(model, config("dual"))
    try:
        before = next(view for view in sensor.camera_views() if view["name"] == name)
        assert before["rgb"] is None
        assert before["calibration_metadata"] is not None
        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        calibration = CameraCalibration.from_camera(model, data, camera, 480, 360)
        world = np.array([[-0.8, -0.1, 0.255], [0.5, 0.1, 0.255]])
        pixels = calibration.project(world)
        restored = calibration.on_horizontal_plane(pixels[:, 1], pixels[:, 0], 0.255)
        np.testing.assert_allclose(restored, world, atol=1e-12)
        sensor.update(data, 0)
        after = next(view for view in sensor.camera_views() if view["name"] == name)
        assert after["depth_range_m"] == before["depth_range_m"]
        near, far = after["depth_range_m"]
        assert 0.9 < near < 1.1
        assert 1.5 < far < 1.7
    finally:
        sensor.close()


@pytest.mark.parametrize("rig", ["dual", "triple"])
@pytest.mark.parametrize("shape", ["box", "pentagonal_prism", "hexagonal_prism"])
@pytest.mark.parametrize("x,yaw", [(-0.85, 0.2), (0.50, 0.9)])
def test_real_oblique_prism_tracks_measure_current_grasp_support(rig, shape, x, yaw):
    model, data, outline = fixture(shape)
    sensor = SensorPipeline(model, config(rig))
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    )
    previous = None
    try:
        for frame in range(6):
            centre_x = x + 0.003 * frame
            model.body_pos[1] = [centre_x, 0.03, 0.23]
            model.body_quat[1] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            mujoco.mj_forward(model, data)
            tracks = sensor.update(data, frame * 0.1)
            assert len(tracks) == 1
            track = tracks[0]
            assert track.track_id == 1
            assert track.hits == frame + 1  # one correction for the camera batch
            axes = np.array(
                [
                    [math.cos(track.yaw), -math.sin(track.yaw)],
                    [math.sin(track.yaw), math.cos(track.yaw)],
                ]
            )
            world = outline @ rotation.T + [centre_x, 0.03]
            projected = world @ axes
            expected = ((projected.min(0) + projected.max(0)) / 2) @ axes.T
            np.testing.assert_allclose(track.position[:2], expected, atol=0.004)
            assert abs(track.grasp_width - np.ptp(projected[:, 1])) < 0.009
            assert abs(track.size[2] - 0.05) < 0.003
            if frame >= 4:
                assert abs(symmetric_angle(track.yaw - previous, math.pi)) < 0.12
            previous = track.yaw
        assert (
            sensor.diagnostics()["raw_detections"]
            > sensor.diagnostics()["fused_detections"]
        )
    finally:
        sensor.close()


def test_real_opposed_views_recover_object_fully_hidden_from_overhead():
    model, data, _ = fixture(blocker=True)
    sensors = [
        SensorPipeline(model, config(rig)) for rig in ("overhead", "dual", "triple")
    ]
    try:
        counts = [len(sensor.update(data, 0)) for sensor in sensors]
        assert counts == [0, 1, 1]
        for sensor in sensors[1:]:
            np.testing.assert_allclose(
                sensor.tracker.snapshots(0)[0].position, [-0.35, 0.03, 0.23], atol=0.004
            )
    finally:
        for sensor in sensors:
            sensor.close()


def test_real_close_same_colour_objects_do_not_get_deduplicated_together():
    model, data, _ = fixture(second=True)
    sensor = SensorPipeline(model, config("triple"))
    try:
        tracks = sensor.update(data, 0)
        assert len(tracks) == 2
        np.testing.assert_allclose(
            [track.position[0] for track in tracks], [-0.35, -0.275], atol=0.004
        )
        assert sensor.diagnostics()["raw_detections"] == 6
        assert sensor.diagnostics()["fused_detections"] == 2
    finally:
        sensor.close()
