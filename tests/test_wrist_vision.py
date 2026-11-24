"""Target-conditioned wrist perception: delayed poses, identity and abstention."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from dataclasses import replace
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.vision import Detection, Track
from dual_arm_conveyor.wrist import WRIST_MAX_AGE, WristSensorPipeline


def configuration(**changes):
    values = dict(
        wrist_camera_mode="refine",
        wrist_camera_hz=6.0,
        wrist_camera_width=320,
        wrist_camera_height=240,
        wrist_max_correction=0.008,
        camera_latency=0.08,
        depth_noise=0.001,
        camera_dropout=0.0,
        seed=31,
        visual_quality="balanced",
        belt_top=0.205,
        belt_half_length=1.15,
        belt_half_width=0.16,
        object_family="flat_prisms",
    )
    values.update(changes)
    return SimpleNamespace(**values)


def scene(shape="box", size=(0.055, 0.045, 0.05), yaw=0.0):
    asset = ""
    if shape == "box":
        geom = 'type="box" size="' + " ".join(str(value / 2) for value in size) + '"'
    else:
        from dual_arm_conveyor.objects import ObjectSpec

        spec = ObjectSpec(shape, size, 0.05, (1, 0.01, 0.001), (0.012, 1), "fixture")
        vertices, faces = spec.prism_mesh()
        asset = (
            '<asset><mesh name="anonymous" vertex="'
            + " ".join(map(str, vertices.ravel()))
            + '" face="'
            + " ".join(map(str, faces.ravel()))
            + '"/></asset>'
        )
        geom = 'type="mesh" mesh="anonymous"'
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
        <visual><global offwidth="640" offheight="480"/><quality offsamples="0"/>
        <map znear=".001"/></visual>{asset}
        <worldbody><light pos="0 0 3"/>
        <geom type="box" pos="0 0 .15" size="1.15 .16 .055" rgba=".1 .15 .2 1"/>
        <body name="hand0" pos="-.20 0 .50">
          <camera name="r0/wrist_rgbd" pos=".05 0 0" xyaxes="1 0 -.204 0 1 0" fovy="80"/>
        </body>
        <body name="hand1" pos=".30 0 .50">
          <camera name="r1/wrist_rgbd" pos=".05 0 0" xyaxes="1 0 -.204 0 1 0" fovy="80"/>
        </body>
        <body name="unrelated_object" pos="-.20 0 {0.205 + size[2] / 2}"
              quat="{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}">
          <geom {geom} rgba=".87 .13 .08 1"/>
        </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def target(
    track_id=17, x=-0.2, yaw=0.0, size=(0.055, 0.045, 0.05), velocity=(0.1, 0.0, 0.0)
):
    return Track(
        track_id,
        np.array([x, 0.0, 0.205 + size[2] / 2]),
        np.array(velocity),
        yaw,
        0,
        0.95,
        0.0,
        0.0,
        hits=5,
        size=np.array(size),
        shape="prism",
        yaw_period=math.pi,
        uncertainty=0.002,
    )


def detection(x=-0.198, y=0.0, size=(0.055, 0.045, 0.05), kind=0, yaw=0.0):
    points = np.array(
        [
            [px, py]
            for px in np.linspace(-size[0] / 2, size[0] / 2, 8)
            for py in np.linspace(-size[1] / 2, size[1] / 2, 8)
        ]
    )
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    )
    points = points @ rotation.T + [x, y]
    return Detection(
        np.array([x, y, 0.205 + size[2] / 2]),
        yaw,
        kind,
        0.92,
        (30, 30, 100, 100),
        np.array(size),
        "prism",
        math.pi,
        0.0015,
        footprint_points=tuple(map(tuple, points)),
    )


class FakeRenderer:
    def __init__(self):
        self.scene = SimpleNamespace(flags=np.ones(32, dtype=bool))
        self.is_depth = False
        self.calls = []
        self.closed = False

    def disable_depth_rendering(self):
        self.is_depth = False

    def enable_depth_rendering(self):
        self.is_depth = True

    def update_scene(self, data, camera, **kwargs):
        self.calls.append(camera)
        assert np.all(kwargs["scene_option"].sitegroup == 0)

    def render(self):
        return (
            np.full((240, 320), 0.2, dtype=np.float32)
            if self.is_depth
            else np.zeros((240, 320, 3), dtype=np.uint8)
        )

    def close(self):
        self.closed = True


def fake_pipeline(**changes):
    model, data = scene()
    sensor = WristSensorPipeline(model, configuration(**changes))
    sensor._renderer = FakeRenderer()
    sensor.detector.detect = lambda rgb, depth, calibration: [detection()]
    return sensor, data


def test_only_assigned_arm_captures_at_its_own_cadence_and_never_creates_ids():
    sensor, data = fake_pipeline()
    try:
        sensor.update(data, 0.0)
        assert sensor.diagnostics()["captured_frames"] == 0
        sensor.set_targets([target(), None])
        sensor.update(data, 0.01)
        sensor.update(data, 0.09)
        assert sensor._renderer.calls == ["r0/wrist_rgbd"]
        measured = sensor.refinement(0, 17, 0.09)
        assert measured is not None and measured.track_id == 17
        assert sensor.refinement(1, 17, 0.09) is None
        assert not hasattr(sensor, "tracker")
        sensor.update(data, 0.176)
        assert len(sensor._renderer.calls) == 1
        sensor.update(data, 0.177)
        assert len(sensor._renderer.calls) == 2
    finally:
        sensor.close()


def test_real_delayed_frame_uses_capture_camera_extrinsics_after_hand_moves():
    model, data = scene()
    sensor = WristSensorPipeline(model, configuration())
    try:
        sensor.set_targets([target(x=-0.197), None])
        sensor.update(data, 0.0)
        assert sensor.refinement(0, 17, 0.079) is None
        capture_position = data.cam_xpos[0].copy()
        model.body_pos[1] += [0.20, -0.04, 0.05]
        model.body_quat[1] = [math.cos(0.3), 0, 0, math.sin(0.3)]
        mujoco.mj_forward(model, data)
        sensor.update(data, 0.08)
        measured = sensor.refinement(0, 17, 0.08)
        assert measured is not None, sensor.diagnostics()
        assert measured.timestamp == 0.0
        np.testing.assert_allclose(measured.position, [-0.20, 0, 0.23], atol=0.002)
        np.testing.assert_allclose(
            measured.predict(0.08)[:2], measured.position[:2] + [0.008, 0]
        )
        np.testing.assert_allclose(
            sensor.camera_views()[0]["calibration_metadata"]["position"],
            capture_position,
        )
        assert np.linalg.norm(data.cam_xpos[0] - capture_position) > 0.19
    finally:
        sensor.close()


def test_capture_target_arrays_are_copied_and_same_id_latest_track_cannot_relabel_frame():
    sensor, data = fake_pipeline()
    original = target()
    try:
        sensor.set_targets([original, None])
        sensor.update(data, 0.0)
        original.position[:] += 1
        original.velocity[:] = 7
        original.size[:] = 2
        sensor.set_targets([target(x=-0.19), None])
        sensor.update(data, 0.08)
        measured = sensor.refinement(0, 17, 0.08)
        assert measured is not None
        np.testing.assert_allclose(measured.source_target_position, [-0.2, 0, 0.23])
        np.testing.assert_allclose(measured.velocity, [0.1, 0, 0])
        with pytest.raises(ValueError):
            measured.position[0] = 0
    finally:
        sensor.close()


@pytest.mark.parametrize("replacement", [target(track_id=18), None])
def test_changed_target_rejects_in_flight_frame_and_wrong_id_queries(replacement):
    sensor, data = fake_pipeline()
    try:
        sensor.set_targets([target(), None])
        sensor.update(data, 0.0)
        sensor.set_targets([replacement, None])
        sensor.update(data, 0.08)
        assert sensor.refinement(0, 17, 0.08) is None
        assert sensor.refinement(0, 18, 0.08) is None
        assert (
            sensor.diagnostics()["per_camera"]["r0/wrist_rgbd"]["rejection_reasons"][
                "changed_target"
            ]
            == 1
        )
        assert sensor.camera_views()[0]["frame_timestamp"] is None
    finally:
        sensor.close()


def test_clear_then_reassign_same_id_invalidates_old_capture_generation():
    sensor, data = fake_pipeline()
    try:
        sensor.set_targets([target(), None])
        sensor.update(data, 0.0)
        sensor.set_targets([None, None])
        sensor.set_targets([target(), None])
        sensor.update(data, 0.08)
        assert sensor.refinement(0, 17, 0.08) is None
    finally:
        sensor.close()


def test_new_target_clears_previously_delivered_image_and_measurement():
    sensor, data = fake_pipeline(camera_latency=0.0)
    try:
        sensor.set_targets([target(), None])
        sensor.update(data, 0.0)
        assert sensor.camera_views()[0]["rgb"] is not None
        sensor.set_targets([target(track_id=20), None])
        view = sensor.camera_views()[0]
        for key in (
            "rgb",
            "depth",
            "overlay",
            "frame_timestamp",
            "calibration_metadata",
            "frame_target_track_id",
        ):
            assert view[key] is None
        assert view["target_track_id"] == 20
        assert sensor.refinement(0, 17, 0.0) is None
    finally:
        sensor.close()


@pytest.mark.parametrize(
    "observations,reason",
    [
        ([detection(kind=1)], "no_target_detection"),
        ([detection(x=-0.198), detection(x=-0.205)], "ambiguous_detection"),
        (
            [detection(size=(0.030, 0.045, 0.05))],
            "incomplete_or_inconsistent_footprint",
        ),
        ([detection(size=(0.055, 0.045, 0.060))], "inconsistent_height"),
        ([detection(x=-0.188)], "excessive_correction"),
        ([replace(detection(), bbox=(0, 30, 100, 100))], "image_boundary"),
        ([replace(detection(), confidence=0.7)], "insufficient_footprint"),
    ],
)
def test_unreliable_local_measurement_abstains_instead_of_selecting_wrong_object(
    observations, reason
):
    sensor, data = fake_pipeline(camera_latency=0.0)
    sensor.detector.detect = lambda rgb, depth, calibration: observations
    try:
        sensor.set_targets([target(), None])
        sensor.update(data, 0.0)
        assert sensor.refinement(0, 17, 0.0) is None
        assert sensor.camera_views()[0]["status"] == reason
        assert sensor.diagnostics()["accepted_refinements"] == 0
    finally:
        sensor.close()


def test_local_refit_uses_assigned_grasp_direction_instead_of_detector_equivalent_yaw():
    sensor, data = fake_pipeline(camera_latency=0.0)
    observed = replace(
        detection(), yaw=math.pi / 2, size=np.array([0.045, 0.055, 0.05])
    )
    sensor.detector.detect = lambda rgb, depth, calibration: [observed]
    try:
        sensor.set_targets([target(yaw=0.0), None])
        sensor.update(data, 0.0)
        measured = sensor.refinement(0, 17, 0.0)
        assert measured is not None
        np.testing.assert_allclose(measured.observed_size, [0.055, 0.045, 0.05])
        np.testing.assert_allclose(measured.correction_xy, [0.002, 0])
    finally:
        sensor.close()


def test_stale_refinement_and_delayed_stale_frame_cannot_reach_controller():
    sensor, data = fake_pipeline(camera_latency=0.0)
    try:
        sensor.set_targets([target(), None])
        sensor.update(data, 0.0)
        assert sensor.refinement(0, 17, WRIST_MAX_AGE - 0.001) is not None
        assert sensor.refinement(0, 17, WRIST_MAX_AGE + 0.001) is None
        assert sensor.refinement(0, 18, 0.01) is None
        sensor.reset(31)
        sensor.latency = 0.4
        sensor.set_targets([target(), None])
        sensor.update(data, 0.0)
        sensor.update(data, 0.4)
        assert sensor.refinement(0, 17, 0.4) is None
        assert sensor.camera_views()[0]["status"] == "stale_frame"
    finally:
        sensor.close()


def test_observe_records_the_same_quality_measurement_but_returns_no_control_correction():
    sensor, data = fake_pipeline(wrist_camera_mode="observe", camera_latency=0.0)
    try:
        sensor.set_targets([target(), None])
        sensor.update(data, 0.0)
        assert sensor.diagnostics()["accepted_refinements"] == 1
        assert sensor.refinement(0, 17, 0.0) is None
    finally:
        sensor.close()


def test_off_mode_requires_no_wrist_cameras_and_allocates_no_renderer():
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    sensor = WristSensorPipeline(model, configuration(wrist_camera_mode="off"))
    try:
        sensor.set_targets([target(), None])
        sensor.update(mujoco.MjData(model), 0.0)
        assert sensor.diagnostics()["captured_frames"] == 0
        assert sensor._renderer is None
        assert sensor.refinement(0, 17, 0.0) is None
    finally:
        sensor.close()


def test_each_wrist_rng_is_independent_of_other_active_camera_and_global_numpy_rng():
    a, data_a = fake_pipeline(camera_latency=0.0, camera_dropout=0.35)
    b, data_b = fake_pipeline(camera_latency=0.0, camera_dropout=0.35)
    np.random.seed(47)
    global_state = np.random.get_state()
    try:
        a.set_targets([target(), None])
        b.set_targets([target(), target(track_id=18)])
        for step in range(12):
            now = step * 0.17
            a.update(data_a, now)
            b.update(data_b, now)
            av, bv = a.camera_views()[0], b.camera_views()[0]
            assert av["frame_timestamp"] == bv["frame_timestamp"]
            if av["depth"] is not None:
                np.testing.assert_array_equal(av["depth"], bv["depth"])
        after = np.random.get_state()
        assert global_state[0] == after[0]
        np.testing.assert_array_equal(global_state[1], after[1])
        assert global_state[2:] == after[2:]
    finally:
        a.close()
        b.close()


def test_pending_frames_and_images_are_bounded_and_reset_close_clear_resources():
    sensor, data = fake_pipeline()
    renderer = sensor._renderer
    try:
        sensor.set_targets([target(), target(track_id=18)])
        for step in range(20):
            for arm in range(2):
                sensor.capture(arm, data, step * 0.001)
        diagnostics = sensor.diagnostics()
        assert diagnostics["queued_frames"] == diagnostics["queue_capacity"]
        assert diagnostics["queue_overflows"] > 0
        sensor.reset(31)
        assert sensor.diagnostics()["queued_frames"] == 0
        assert sensor.diagnostics()["active_targets"] == [None, None]
        assert all(view["rgb"] is None for view in sensor.camera_views())
    finally:
        sensor.close()
    assert renderer.closed
    assert sensor._renderer is None


@pytest.mark.parametrize("shape", ["box", "pentagonal_prism", "hexagonal_prism"])
@pytest.mark.parametrize("yaw", [0.2, 0.9])
def test_real_rgbd_prisms_accept_small_known_target_offset_with_one_mm_noise(
    shape, yaw
):
    from dual_arm_conveyor.vision import RGBDDetector

    model, data = scene(shape=shape, yaw=yaw)
    sensor = WristSensorPipeline(model, configuration(camera_latency=0.0))
    try:
        # Establish an image-only target grasp frame. This intentionally does
        # not substitute the polygon body origin for its valid grasp centre.
        sensor.set_targets([target(yaw=yaw), None])
        sensor.capture(0, data, 0.0)
        frame = sensor._queue[-1]
        visual = RGBDDetector(configuration()).detect(
            frame.rgb, frame.depth, frame.calibration
        )
        assert len(visual) == 1
        initial = visual[0]
        known = replace(
            target(),
            position=initial.position + [0.003, -0.002, 0.0],
            yaw=initial.yaw,
            size=initial.size,
        )
        sensor.reset(32)
        sensor.set_targets([known, None])
        sensor.update(data, 0.0)
        measured = sensor.refinement(0, 17, 0.0)
        assert measured is not None, sensor.diagnostics()
        np.testing.assert_allclose(measured.position, initial.position, atol=0.001)
        np.testing.assert_allclose(measured.correction_xy, [-0.003, 0.002], atol=0.001)
        assert sensor.camera_views()[0]["role"] == "local refinement"
    finally:
        sensor.close()


@pytest.mark.parametrize(
    "clearance,accepted", [(0.02, False), (0.074, False), (0.075, True), (0.09, True)]
)
def test_local_measurement_requires_75_mm_vertical_clearance_above_estimated_top(
    clearance, accepted
):
    sensor, data = fake_pipeline(camera_latency=0.0)
    try:
        known = target(velocity=(0, 0, 0))
        sensor.set_targets([known, None])
        sensor.model.body_pos[1, 2] = known.position[2] + known.half_height + clearance
        mujoco.mj_forward(sensor.model, data)
        sensor.update(data, 0.0)
        assert (sensor.refinement(0, 17, 0.0) is not None) == accepted
        if not accepted:
            assert sensor.camera_views()[0]["status"] == "low_view_clearance"
    finally:
        sensor.close()


@pytest.mark.parametrize(
    "rejection",
    [
        "low_view_clearance",
        "no_target_detection",
        "incomplete_or_inconsistent_footprint",
    ],
)
def test_quality_rejection_keeps_previous_measurement_without_extending_original_deadline(
    rejection,
):
    sensor, data = fake_pipeline(camera_latency=0.0)
    try:
        sensor.set_targets([target(velocity=(0, 0, 0)), None])
        sensor.update(data, 0.0)
        original = sensor.refinement(0, 17, 0.0)
        assert original is not None
        if rejection == "low_view_clearance":
            sensor.model.body_pos[1, 2] = 0.255 + 0.04
            mujoco.mj_forward(sensor.model, data)
        else:
            observed = (
                []
                if rejection == "no_target_detection"
                else [detection(size=(0.030, 0.045, 0.05))]
            )
            sensor.detector.detect = lambda rgb, depth, calibration: observed
        sensor.update(data, 0.17)
        assert sensor.camera_views()[0]["status"] == rejection
        retained = sensor.refinement(0, 17, 0.17)
        assert retained is original
        assert retained.timestamp == 0.0
        assert retained.valid_until == WRIST_MAX_AGE
        sensor.update(data, 0.34)
        assert sensor.refinement(0, 17, 0.34) is None
        assert sensor.diagnostics()["accepted_refinements"] == 1
        assert sensor.camera_views()[0]["diagnostics"]["accepted_timestamp"] == 0.0
    finally:
        sensor.close()


@pytest.mark.parametrize(
    "reason", ["excessive_correction", "ambiguous_detection", "stale_target"]
)
def test_contradictory_or_expired_target_clears_previously_accepted_measurement(reason):
    sensor, data = fake_pipeline(camera_latency=0.0)
    known = target(velocity=(0, 0, 0))
    try:
        sensor.set_targets([known, None])
        sensor.update(data, 0.0)
        assert sensor.refinement(0, 17, 0.0) is not None
        if reason == "stale_target":
            sensor.set_targets([replace(known, last_seen=-2), None])
        else:
            observed = (
                [detection(x=-0.188)]
                if reason == "excessive_correction"
                else [detection(x=-0.198), detection(x=-0.203)]
            )
            sensor.detector.detect = lambda rgb, depth, calibration: observed
        sensor.update(data, 0.17)
        assert sensor.camera_views()[0]["status"] == reason
        assert sensor.refinement(0, 17, 0.17) is None
    finally:
        sensor.close()
