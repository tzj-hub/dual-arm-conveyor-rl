"""Camera-image, geometric calibration, association, and delay regression tests."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import math
from dataclasses import replace
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from dual_arm_conveyor.vision import (
    CameraCalibration,
    Detection,
    MultiObjectTracker,
    SensorPipeline,
    fit_prism_outline,
    square_angle,
    symmetric_angle,
)


def sensor_config(**kwargs):
    values = dict(
        belt_top=0.205,
        belt_half_length=1.15,
        belt_half_width=0.16,
        cube_size=0.05,
        belt_speed=0.1,
        camera_width=320,
        camera_height=240,
        camera_hz=10.0,
        camera_latency=0.08,
        depth_noise=0.0,
        camera_dropout=0.0,
        seed=3,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def image_scene(yaw=0.2, occluded=False):
    # Deliberately unrelated body/geom names: detection cannot use object IDs.
    blocker = (
        '<geom type="box" pos="-.35 -.03 .34" size=".04 .04 .015" rgba=".5 .5 .5 1"/>'
        if occluded
        else ""
    )
    xml = f'''<mujoco><visual><global offwidth="640" offheight="480"/></visual>
    <worldbody><light pos="0 0 3" diffuse=".9 .9 .9"/>
    <camera name="overhead" pos="0 0 1.8" fovy="60"/>
    <geom type="box" pos="0 0 .15" size="1.15 .16 .055" rgba=".1 .15 .2 1"/>
    <body name="anonymous_a" pos="-.35 -.03 .23" quat="{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}">
    <geom type="box" size=".025 .025 .025" rgba=".87 .13 .08 1"/></body>
    <body name="anonymous_b" pos=".3 .045 .23" quat="{math.cos(-0.3 / 2)} 0 0 {math.sin(-0.3 / 2)}">
    <geom type="box" size=".025 .025 .025" rgba=".08 .28 .92 1"/></body>
    <geom type="box" pos="0 .3 .23" size=".1 .06 .025" rgba=".87 .13 .08 1"/>
    <geom type="box" pos=".7 0 .207" size=".03 .03 .002" rgba=".87 .13 .08 1"/>
    {blocker}</worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def detection(x, y=0.0, kind=0, yaw=0.0):
    return Detection(np.array([x, y, 0.23]), yaw, kind, 0.95, (0, 0, 6, 6))


def varied_shape_scene(shape, dimensions, x=-0.4, yaw=0.37):
    half = np.asarray(dimensions) / 2
    geom_size = (
        f"{half[0]} {half[2]}" if shape == "cylinder" else " ".join(map(str, half))
    )
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <light pos="0 0 3"/><camera name="overhead" pos="0 0 1.8" fovy="60"/>
      <geom type="box" pos="0 0 .15" size="1.15 .16 .055" rgba=".1 .15 .2 1"/>
      <body name="unrelated_render_name" pos="{x} .03 {0.205 + half[2]}"
            quat="{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}">
        <geom type="{shape}" size="{geom_size}" rgba=".87 .13 .08 1"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def flat_prism_scene(shape):
    from dual_arm_conveyor.objects import ObjectSpec

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
            '<asset><mesh name="anonymous_mesh" vertex="'
            + " ".join(map(str, vertices.ravel()))
            + '" face="'
            + " ".join(map(str, faces.ravel()))
            + '"/></asset>'
        )
        geom = 'type="mesh" mesh="anonymous_mesh"'
    model = mujoco.MjModel.from_xml_string(f"""<mujoco>
      <visual><global offwidth="640" offheight="480"/><quality offsamples="0"/></visual>{asset}
      <worldbody><light pos="0 0 3"/><camera name="overhead" pos="0 0 1.8" fovy="60"/>
      <geom type="box" pos="0 0 .15" size="1.15 .16 .055" rgba=".1 .15 .2 1"/>
      <body pos="0 .03 .23"><geom {geom} rgba=".87 .13 .08 1"/></body>
      </worldbody></mujoco>""")
    return model, mujoco.MjData(model), outline


class CalibrationTests(unittest.TestCase):
    def test_prism_outline_minimum_width_and_asymmetric_grasp_centre(self):
        for sides in (5, 6):
            angles = np.arange(sides) * 2 * math.pi / sides
            points = np.column_stack((np.cos(angles), np.sin(angles))) * 0.03
            centre, yaw, dimensions, candidates = fit_prism_outline(points)
            expected_width = (
                0.03 * (1 + math.cos(math.pi / sides))
                if sides == 5
                else 0.06 * math.cos(math.pi / 6)
            )
            self.assertAlmostEqual(dimensions[1], expected_width)
            self.assertEqual(len(candidates), 5 if sides == 5 else 3)
            axes = np.array(
                [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
            )
            local = (points - centre) @ axes
            self.assertTrue(np.all(np.abs(local) <= dimensions / 2 + 1e-12))
            if sides == 5:
                self.assertGreater(np.linalg.norm(centre), 0.002)
            else:
                np.testing.assert_allclose(centre, 0, atol=1e-12)

    def test_rectangle_grasp_quaternion_closes_across_short_axis(self):
        from dual_arm_conveyor.ik import downward_quaternion

        for yaw in (-1.2, -0.3, 0.6, 1.3):
            long_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            for equivalent_yaw in (yaw, yaw + math.pi):
                matrix = np.empty(9)
                mujoco.mju_quat2Mat(matrix, downward_quaternion(equivalent_yaw))
                closing_axis = matrix.reshape(3, 3)[:, 1]
                self.assertAlmostEqual(float(closing_axis @ long_axis), 0.0, places=7)
                self.assertAlmostEqual(
                    symmetric_angle(equivalent_yaw - yaw, math.pi), 0.0
                )

    def test_project_unproject_roundtrip_with_rotated_camera(self):
        angle = 0.32
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        calibration = CameraCalibration(
            320, 240, 207.846, np.array([0.1, 0.2, 1.8]), rotation
        )
        rows, columns = np.array([20.0, 120.0, 210.0]), np.array([50.0, 160.0, 290.0])
        world = calibration.unproject(rows, columns, np.array([1.5, 1.6, 1.4]))
        np.testing.assert_allclose(
            calibration.project(world), np.column_stack([columns, rows]), atol=1e-10
        )

    def test_square_yaw_symmetry(self):
        for angle in (-0.7, -0.2, 0.0, 0.3, 0.7):
            for turn in (-2, -1, 0, 1, 2):
                self.assertAlmostEqual(
                    square_angle(angle), square_angle(angle + turn * math.pi / 2)
                )


class TrackingTests(unittest.TestCase):
    def test_polygon_equivalent_edge_changes_do_not_invent_rotation(self):
        tracker = MultiObjectTracker()
        points = tuple(
            map(
                tuple,
                np.array(
                    [[-0.025, -0.025], [0.025, -0.025], [0.025, 0.025], [-0.025, 0.025]]
                ),
            )
        )
        first = replace(
            detection(0.0),
            shape="prism",
            yaw_period=math.pi,
            footprint_candidates=((0.0, 0.0, 0.0, 0.05, 0.05),),
            footprint_points=points,
            footprint_pixel_size=0.004,
        )
        for frame in range(3):
            tracker.update([first], frame * 0.1)
        alternate = replace(
            first,
            yaw=math.pi / 2,
            footprint_candidates=((math.pi / 2, 0.0, 0.0, 0.05, 0.05),),
        )
        tracker.update([alternate], 0.3)
        track = tracker.snapshots(0.3)[0]
        self.assertAlmostEqual(track.yaw, 0.0)
        self.assertEqual(track.hits, 4)
        self.assertEqual(track.shape, "prism")

    def test_seventy_percent_prism_fragment_cannot_drag_the_track(self):
        tracker = MultiObjectTracker()
        whole = replace(
            detection(0.0),
            shape="prism",
            yaw_period=math.pi,
            size=np.array([0.06, 0.04, 0.05]),
        )
        tracker.update([whole], 0.0)
        fragment = replace(
            whole,
            position=np.array([0.02, 0.01, 0.23]),
            size=np.array([0.042, 0.04, 0.05]),
        )
        tracker.update([fragment], 0.1)
        track = tracker.snapshots(0.1)[0]
        self.assertEqual(track.last_seen, 0.0)
        self.assertFalse(track.visible)
        np.testing.assert_allclose(track.size, whole.size)
        np.testing.assert_allclose(track.predict(0.1), [0.01, 0, 0.23])

    def test_continuous_stream_ids_and_filter_storage_are_bounded(self):
        tracker = MultiObjectTracker(max_tracks=8)
        for sequence in range(2000):
            now = sequence * 2.0
            tracker.update([detection(0.0)], now)
            tracks = tracker.snapshots(now)
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].track_id, sequence + 1)
            self.assertLessEqual(len(tracker._filters), 8)
        tracker.update([detection(i * 0.15) for i in range(30)], 4002.0)
        self.assertEqual(len(tracker.snapshots(4002.0)), 8)

    def test_partial_occlusion_does_not_shrink_grasp_measurement(self):
        tracker = MultiObjectTracker()
        original = replace(
            detection(0.0), size=np.array([0.06, 0.04, 0.05]), yaw_period=math.pi
        )
        tracker.update([original], 0.0)
        fragment = replace(
            original,
            position=np.array([0.02, 0.01, 0.23]),
            size=np.array([0.025, 0.025, 0.05]),
        )
        tracker.update([fragment], 0.1)
        track = tracker.snapshots(0.1)[0]
        self.assertEqual(track.last_seen, 0.0)
        self.assertFalse(track.visible)
        np.testing.assert_allclose(track.size, original.size)
        self.assertFalse(track.size.flags.writeable)
        self.assertAlmostEqual(track.half_height, 0.025)
        self.assertAlmostEqual(track.grasp_width, 0.04)

    def test_association_velocity_and_occlusion_keep_independent_ids(self):
        tracker = MultiObjectTracker(belt_speed=0.1)
        tracker.update([detection(-0.5), detection(0.1)], 0.0)
        initial = tracker.snapshots(0.0)
        self.assertEqual([t.track_id for t in initial], [1, 2])
        for frame in range(1, 11):
            now = frame * 0.1
            # Deliberately reverse detector order to test association, not indices.
            tracker.update(
                [detection(0.1 + 0.1 * now), detection(-0.5 + 0.1 * now)], now
            )
        for track, x in zip(tracker.snapshots(1.0), (-0.4, 0.2)):
            self.assertAlmostEqual(track.position[0], x, places=5)
            self.assertAlmostEqual(track.velocity[0], 0.1, places=5)
            self.assertFalse(track.position.flags.writeable)
        tracker.update([], 1.3)
        coasted = tracker.snapshots(1.5)
        np.testing.assert_allclose(coasted[0].predict(1.5), [-0.35, 0, 0.23], atol=1e-5)
        self.assertFalse(coasted[0].visible)
        self.assertLess(coasted[0].confidence, initial[0].confidence)
        tracker.update([detection(-0.34), detection(0.26)], 1.6)
        self.assertEqual([t.track_id for t in tracker.snapshots(1.6)], [1, 2])

    def test_stale_tracks_expire_and_prediction_is_bounded(self):
        tracker = MultiObjectTracker(max_age=1.5)
        tracker.update([detection(0.0)], 0.0)
        snapshot = tracker.snapshots(0.0)[0]
        np.testing.assert_allclose(snapshot.predict(20.0), [0.15, 0, 0.23])
        self.assertEqual(tracker.snapshots(1.51), [])
        tracker.update([detection(0.3)], 3.0)
        self.assertEqual(tracker.snapshots(3.0)[0].track_id, 2)

    def test_colour_gate_and_square_angle_filter(self):
        tracker = MultiObjectTracker()
        tracker.update([detection(0, kind=0, yaw=0.77)], 0.0)
        tracker.update([detection(0.01, kind=0, yaw=-0.77)], 0.1)
        self.assertLess(abs(square_angle(tracker.snapshots(0.1)[0].yaw - 0.77)), 0.02)
        tracker.update([detection(0.02, kind=1)], 0.2)
        self.assertEqual(len(tracker.snapshots(0.2)), 2)
        self.assertEqual(tracker.snapshots(0.2)[1].kind, 1)

    def test_reject_out_of_order_frames(self):
        tracker = MultiObjectTracker()
        tracker.update([], 1.0)
        with self.assertRaises(ValueError):
            tracker.update([], 0.9)


class RenderedVisionTests(unittest.TestCase):
    def test_balanced_lighting_preserves_camera_depth_and_grasp_geometry(self):
        model, data, _ = flat_prism_scene("pentagonal_prism")
        model.body_pos[1] = [-0.4, 0.03, 0.23]
        mujoco.mj_forward(model, data)
        config = sensor_config(
            camera_width=480,
            camera_height=360,
            camera_latency=0.0,
            object_family="flat_prisms",
        )
        high = SensorPipeline(
            model, SimpleNamespace(**vars(config), visual_quality="high")
        )
        balanced = SensorPipeline(
            model, SimpleNamespace(**vars(config), visual_quality="balanced")
        )
        try:
            original = high.update(data, 0.0)
            faster = balanced.update(data, 0.0)
            np.testing.assert_array_equal(high.depth, balanced.depth)
            self.assertEqual(len(original), 1)
            self.assertEqual(len(faster), 1)
            np.testing.assert_allclose(original[0].position, faster[0].position)
            np.testing.assert_allclose(original[0].size, faster[0].size)
            self.assertAlmostEqual(original[0].yaw, faster[0].yaw)
        finally:
            high.close()
            balanced.close()

    def test_flat_prism_camera_tracks_current_grasp_support_and_stable_orientation(
        self,
    ):
        for shape in ("box", "pentagonal_prism", "hexagonal_prism"):
            for yaw in (0.2, 0.9):
                with self.subTest(shape=shape, yaw=yaw):
                    model, data, outline = flat_prism_scene(shape)
                    config = sensor_config(
                        camera_width=480,
                        camera_height=360,
                        camera_latency=0.0,
                        depth_noise=0.001,
                        object_family="flat_prisms",
                        seed=0,
                    )
                    del config.cube_size
                    pipeline = SensorPipeline(model, config)
                    rotation = np.array(
                        [
                            [math.cos(yaw), -math.sin(yaw)],
                            [math.sin(yaw), math.cos(yaw)],
                        ]
                    )
                    previous = None
                    try:
                        for frame in range(16):
                            x, now = -0.45 + 0.01 * frame, 0.1 * frame
                            model.body_pos[1] = [x, 0.03, 0.23]
                            model.body_quat[1] = [
                                math.cos(yaw / 2),
                                0,
                                0,
                                math.sin(yaw / 2),
                            ]
                            mujoco.mj_forward(model, data)
                            tracks = pipeline.update(data, now)
                            self.assertEqual(len(tracks), 1)
                            track = tracks[0]
                            self.assertEqual(track.shape, "prism")
                            self.assertEqual(track.track_id, 1)
                            if frame >= 4:
                                axes = np.array(
                                    [
                                        [math.cos(track.yaw), -math.sin(track.yaw)],
                                        [math.sin(track.yaw), math.cos(track.yaw)],
                                    ]
                                )
                                world = outline @ rotation.T + [x, 0.03]
                                projected = world @ axes
                                expected_centre = (
                                    (projected.min(0) + projected.max(0)) / 2
                                ) @ axes.T
                                true_width = float(np.ptp(projected[:, 1]))
                                self.assertLess(
                                    np.linalg.norm(
                                        track.predict(now)[:2] - expected_centre
                                    ),
                                    0.004,
                                )
                                self.assertLess(abs(track.size[1] - true_width), 0.009)
                                self.assertLess(abs(track.size[2] - 0.05), 0.003)
                                self.assertLess(
                                    abs(symmetric_angle(track.yaw - previous, math.pi)),
                                    0.08,
                                )
                            previous = track.yaw
                    finally:
                        pipeline.close()

    def test_variable_shape_size_and_height_are_measured_without_catalogue(self):
        cases = (
            ("box", (0.06, 0.035, 0.035)),
            ("box", (0.045, 0.04, 0.06)),
            ("cylinder", (0.045, 0.045, 0.06)),
            ("ellipsoid", (0.06, 0.045, 0.05)),
        )
        for shape, dimensions in cases:
            for x, yaw in ((-0.4, 0.37), (0.5, 1.1)):
                with self.subTest(shape=shape, x=x):
                    model, data = varied_shape_scene(shape, dimensions, x, yaw)
                    config = sensor_config(
                        camera_latency=0.0, depth_noise=0.001, seed=0
                    )
                    # No nominal object dimensions or shape/material catalogue
                    # are available to perception, including through config.
                    del config.cube_size
                    pipeline = SensorPipeline(model, config)
                    try:
                        tracks = pipeline.update(data, 0.0)
                        self.assertEqual(len(tracks), 1)
                        track = tracks[0]
                        self.assertEqual(track.shape, shape)
                        np.testing.assert_allclose(track.size, dimensions, atol=0.004)
                        np.testing.assert_allclose(
                            track.position,
                            [x, 0.03, 0.205 + dimensions[2] / 2],
                            atol=0.003,
                        )
                        if shape == "cylinder":
                            self.assertEqual(track.yaw_period, 0.0)
                            self.assertEqual(track.yaw, 0.0)
                        else:
                            self.assertLess(
                                abs(symmetric_angle(track.yaw - yaw, track.yaw_period)),
                                0.16,
                            )
                        self.assertGreater(track.uncertainty, 0.0)
                    finally:
                        pipeline.close()

    def test_capture_queue_has_explicit_memory_bound(self):
        model, data = image_scene()
        pipeline = SensorPipeline(model, sensor_config(camera_latency=0.15))

        class FakeRenderer:
            depth = False
            scene = SimpleNamespace(flags=np.ones(32, dtype=bool))

            def disable_depth_rendering(self):
                self.depth = False

            def enable_depth_rendering(self):
                self.depth = True

            def update_scene(self, *args, **kwargs):
                pass

            def render(self):
                return (
                    np.ones((240, 320), dtype=np.float32)
                    if self.depth
                    else np.zeros((240, 320, 3), dtype=np.uint8)
                )

            def close(self):
                pass

        pipeline._renderer = FakeRenderer()
        try:
            for frame in range(100):
                pipeline.capture(data, frame * 0.1)
            diagnostics = pipeline.diagnostics()
            self.assertEqual(
                diagnostics["queued_frames"], diagnostics["queue_capacity"]
            )
            self.assertLessEqual(diagnostics["queued_frames"], 5)
            self.assertEqual(
                diagnostics["queue_overflows"], 100 - diagnostics["queue_capacity"]
            )
        finally:
            pipeline.close()

    def test_rendered_square_yaw_near_ninety_degree_symmetry_boundary(self):
        model, data = image_scene()
        pipeline = SensorPipeline(model, sensor_config(camera_latency=0.0))
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "anonymous_a")
        try:
            for yaw in (-0.78, -0.73, 0.73, 0.78, 1.05, 1.9):
                model.body_quat[body] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
                mujoco.mj_forward(model, data)
                pipeline.reset(3)
                tracks = pipeline.update(data, 0.0)
                red = next(track for track in tracks if track.kind == 0)
                self.assertLess(
                    abs(square_angle(red.yaw - yaw)),
                    0.15,
                    f"yaw={yaw}, estimate={red.yaw}",
                )
                np.testing.assert_allclose(
                    red.position, [-0.35, -0.03, 0.23], atol=0.004
                )
        finally:
            pipeline.close()

    def test_rgb_depth_detects_location_class_yaw_and_rejects_background(self):
        model, data = image_scene()
        pipeline = SensorPipeline(model, sensor_config(camera_latency=0.0))
        try:
            tracks = pipeline.update(data, 0.0)
            self.assertEqual(len(tracks), 2)
            self.assertEqual([track.kind for track in tracks], [0, 1])
            for track, expected, yaw in zip(
                tracks, ([-0.35, -0.03, 0.23], [0.3, 0.045, 0.23]), (0.2, -0.3)
            ):
                np.testing.assert_allclose(track.position, expected, atol=0.004)
                self.assertLess(abs(square_angle(track.yaw - yaw)), 0.15)
            self.assertEqual(pipeline.rgb.shape, (240, 320, 3))
            self.assertEqual(pipeline.overlay.shape, pipeline.rgb.shape)
            self.assertFalse(np.array_equal(pipeline.rgb, pipeline.overlay))
        finally:
            pipeline.close()

    def test_latency_cadence_and_encoder_prediction(self):
        model, data = image_scene()
        pipeline = SensorPipeline(model, sensor_config())
        try:
            self.assertEqual(pipeline.update(data, 0.0), [])
            self.assertEqual(pipeline.update(data, 0.04), [])
            self.assertIsNone(pipeline.rgb)
            tracks = pipeline.update(data, 0.08)
            self.assertEqual(len(tracks), 2)
            self.assertEqual(tracks[0].timestamp, 0.0)
            self.assertAlmostEqual(
                tracks[0].predict(0.08)[0] - tracks[0].position[0], 0.008
            )
            self.assertEqual(pipeline.diagnostics()["captured_frames"], 1)
            self.assertAlmostEqual(pipeline.diagnostics()["frame_age_seconds"], 0.08)
            pipeline.update(data, 0.1)
            self.assertEqual(pipeline.diagnostics()["captured_frames"], 2)
            self.assertEqual(pipeline.diagnostics()["delivered_frames"], 1)
        finally:
            pipeline.close()

    def test_actual_image_occlusion_and_dropout_do_not_leak_truth(self):
        model, data = image_scene(occluded=True)
        pipeline = SensorPipeline(model, sensor_config(camera_latency=0.0))
        try:
            tracks = pipeline.update(data, 0.0)
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].kind, 1)
            pipeline.dropout = 1.0
            pipeline.update(data, 0.1)
            self.assertEqual(pipeline.diagnostics()["dropped_frames"], 1)
            self.assertEqual(pipeline.update(data, 1.6), [])
            pipeline.reset(4)
            self.assertEqual(pipeline.diagnostics()["delivered_frames"], 0)
            self.assertIsNone(pipeline.rgb)
        finally:
            pipeline.close()

    def test_depth_noise_is_applied_to_images(self):
        model, data = image_scene()
        clean = SensorPipeline(
            model, sensor_config(camera_latency=0.0, depth_noise=0.0)
        )
        noisy = SensorPipeline(
            model, sensor_config(camera_latency=0.0, depth_noise=0.001)
        )
        try:
            clean.update(data, 0.0)
            tracks = noisy.update(data, 0.0)
            self.assertEqual(len(tracks), 2)
            self.assertGreater(np.std(noisy.depth - clean.depth), 0.0008)
            self.assertLess(np.std(noisy.depth - clean.depth), 0.0012)
        finally:
            clean.close()
            noisy.close()


if __name__ == "__main__":
    unittest.main()
