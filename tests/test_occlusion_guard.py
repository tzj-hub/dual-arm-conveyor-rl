"""Depth-only foreground-edge guard for wrist-equipped visual sorting."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from dataclasses import replace
import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.ik import PandaIK, downward_quaternion
from dual_arm_conveyor.model import build_scene, reset_robot_pose
from dual_arm_conveyor.vision import CameraCalibration, MultiObjectTracker, RGBDDetector


def config(mode="observe", noise=0.001):
    return SimpleNamespace(
        belt_top=0.205,
        belt_half_length=1.15,
        belt_half_width=0.16,
        object_family="flat_prisms",
        depth_noise=noise,
        wrist_camera_mode=mode,
    )


def image_fixture(foreground_height=None, row=34, pixels=(47, 48, 49)):
    calibration = CameraCalibration(
        100, 80, 200.0, np.array([0.0, 0.0, 1.0]), np.eye(3)
    )
    rgb = np.full((80, 100, 3), 60, dtype=np.uint8)
    depth = np.full((80, 100), 1.0 - 0.205, dtype=np.float32)
    rgb[35:47, 44:57] = [220, 30, 20]
    depth[35:47, 44:57] = 1.0 - 0.255
    if foreground_height is not None:
        depth[row, list(pixels)] = 1.0 - foreground_height
        rgb[row, list(pixels)] = [110, 110, 110]
    return rgb, depth, calibration


def detect_fixture(detector, **kwargs):
    result = detector.detect(*image_fixture(**kwargs))
    assert len(result) == 1
    return result[0]


@pytest.mark.parametrize("mode", ["observe", "refine"])
@pytest.mark.parametrize("row", [33, 34])
def test_three_connected_high_foreground_pixels_in_two_pixel_ring_flag_contour(
    mode, row
):
    detection = detect_fixture(
        RGBDDetector(config(mode)), foreground_height=0.28, row=row
    )
    assert detection.occluded


@pytest.mark.parametrize(
    "height,noise,flag",
    [
        (0.259, 0.001, False),
        (0.261, 0.001, True),
        (0.261, 0.002, False),
        (0.265, 0.002, True),
    ],
)
def test_height_threshold_is_five_mm_or_four_depth_noise_sigmas(height, noise, flag):
    observed = detect_fixture(
        RGBDDetector(config(noise=noise)), foreground_height=height
    )
    assert observed.occluded == flag


@pytest.mark.parametrize(
    "kwargs",
    [
        {"foreground_height": 0.24},
        {"foreground_height": 0.28, "pixels": (47, 48)},
        {"foreground_height": 0.28, "pixels": (45, 49, 53)},
        {"foreground_height": 0.28, "row": 32},
    ],
)
def test_background_small_isolated_outliers_and_nonadjacent_foreground_do_not_flag(
    kwargs,
):
    assert not detect_fixture(RGBDDetector(config()), **kwargs).occluded


def test_ordinary_one_mm_depth_noise_does_not_invent_foreground_boundary():
    detector = RGBDDetector(config())
    for seed in range(12):
        rgb, depth, calibration = image_fixture(foreground_height=0.255)
        depth += (
            np.random.default_rng(seed)
            .normal(0.0, 0.001, depth.shape)
            .astype(depth.dtype)
        )
        observations = detector.detect(rgb, depth, calibration)
        assert len(observations) == 1 and not observations[0].occluded


def test_off_mode_bypasses_guard_and_retains_identical_measured_geometry(monkeypatch):
    image = image_fixture(foreground_height=0.28)
    guarded = RGBDDetector(config("observe")).detect(*image)[0]
    detector = RGBDDetector(config("off"))
    monkeypatch.setattr(
        detector, "_foreground_occlusion", lambda *args: pytest.fail("off guard ran")
    )
    original = detector.detect(*image)[0]
    assert guarded.occluded and not original.occluded
    np.testing.assert_array_equal(original.position, guarded.position)
    np.testing.assert_array_equal(original.size, guarded.size)
    assert original.yaw == guarded.yaw
    assert original.footprint_points == guarded.footprint_points
    assert original.footprint_candidates == guarded.footprint_candidates


def test_visible_hole_cut_by_nearer_foreground_also_marks_occlusion():
    rgb, depth, calibration = image_fixture()
    rgb[39:42, 49:51] = [110, 110, 110]
    depth[39:42, 49:51] = 1.0 - 0.28
    observations = RGBDDetector(config()).detect(rgb, depth, calibration)
    assert len(observations) == 1 and observations[0].occluded


def test_associated_occluded_observation_is_consumed_without_geometry_or_ttl_update():
    whole = detect_fixture(RGBDDetector(config()))
    tracker = MultiObjectTracker(belt_speed=0.1, max_age=0.30)
    tracker.update([whole], 0.0)
    initial = tracker.snapshots(0.0)[0]
    fragment = replace(
        whole,
        position=whole.position + [0.013, 0.002, 0],
        size=whole.size * [0.95, 0.95, 1.0],
        yaw=0.2,
        occluded=True,
    )
    for now in (0.1, 0.2):
        tracker.update([fragment], now)
        tracks = tracker.snapshots(now)
        assert len(tracks) == 1
        track = tracks[0]
        assert track.track_id == initial.track_id
        assert track.last_seen == 0.0 and track.hits == 1 and not track.visible
        assert track.yaw == initial.yaw
        np.testing.assert_array_equal(track.size, initial.size)
        np.testing.assert_allclose(track.position, initial.position + [0.1 * now, 0, 0])
    tracker.update([fragment], 0.31)
    assert tracker.snapshots(0.31) == []
    tracker.update([whole], 0.32)
    assert tracker.snapshots(0.32)[0].track_id == 2


def test_unassociated_occluded_detections_do_not_create_global_identities():
    observed = detect_fixture(RGBDDetector(config()), foreground_height=0.28)
    tracker = MultiObjectTracker()
    tracker.update(
        [observed, replace(observed, position=observed.position + [0.2, 0, 0])], 0.0
    )
    assert tracker.snapshots(0.0) == []
    tracker.update([replace(observed, occluded=False)], 0.1)
    assert tracker.snapshots(0.1)[0].track_id == 1


def test_wrist_foreground_rejection_preserves_only_original_measurement_deadline():
    from test_wrist_vision import fake_pipeline, target, detection

    sensor, data = fake_pipeline(camera_latency=0.0)
    try:
        sensor.set_targets([target(velocity=(0, 0, 0)), None])
        sensor.update(data, 0.0)
        original = sensor.refinement(0, 17, 0.0)
        assert original is not None
        sensor.detector.detect = lambda *args: [replace(detection(), occluded=True)]
        sensor.update(data, 0.17)
        assert sensor.camera_views()[0]["status"] == "occluded_foreground"
        assert sensor.refinement(0, 17, 0.17) is original
        assert original.timestamp == 0.0 and original.valid_until == 0.3
        sensor.update(data, 0.34)
        assert sensor.refinement(0, 17, 0.34) is None
        assert sensor.diagnostics()["accepted_refinements"] == 1
    finally:
        sensor.close()


@pytest.fixture(scope="module")
def rendered_views():
    config = Config(wrist_camera_mode="observe", camera_dropout=0.0)
    model, ids = build_scene(config)
    data = mujoco.MjData(model)
    ik = PandaIK(model, ids)
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
    views = {}
    with mujoco.Renderer(model, height=360, width=480) as renderer:
        for pose in ("home", "left_approach"):
            reset_robot_pose(model, data, ids)
            for index, joint in enumerate(ids.cube_qpos):
                data.qpos[joint : joint + 7] = [3 + index * 0.1, 0, -1, 1, 0, 0, 0]
            objects = []
            for shape, x, y, yaw in [
                ("box", -0.4, -0.015, 0.37),
                ("pentagonal_prism", 0.4, 0.025, 0.9),
            ]:
                index = next(
                    i for i, spec in enumerate(ids.object_specs) if spec.shape == shape
                )
                spec = ids.object_specs[index]
                joint = ids.cube_qpos[index]
                data.qpos[joint : joint + 7] = [
                    x,
                    y,
                    0.205 + spec.half_height,
                    math.cos(yaw / 2),
                    0,
                    0,
                    math.sin(yaw / 2),
                ]
                objects.append((x, y, yaw, spec))
            mujoco.mj_forward(model, data)
            if pose == "left_approach":
                x, y, yaw, spec = objects[0]
                goal = np.array(
                    [
                        x,
                        y,
                        0.205
                        + spec.half_height
                        + max(-0.009, spec.half_height - 0.03)
                        + 0.13,
                    ]
                )
                joints = ik.solve_pose(
                    data, 0, goal, downward_quaternion(yaw), max_iterations=400
                )
                data.qpos[ids.arm_qpos[0]] = joints
                mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="overhead")
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            depth += (
                np.random.default_rng(912)
                .normal(0, 0.001, depth.shape)
                .astype(depth.dtype)
            )
            calibration = CameraCalibration.from_camera(model, data, camera, 480, 360)
            views[pose] = (rgb, depth, calibration)
    return config, views


def test_real_unobstructed_home_observations_are_not_flagged(rendered_views):
    config, views = rendered_views
    observations = RGBDDetector(config).detect(*views["home"])
    assert len(observations) == 2
    assert not any(item.occluded for item in observations)


def test_real_ten_pixel_mount_occlusion_flags_fragment_without_using_object_identity(
    rendered_views,
):
    config, views = rendered_views
    observations = RGBDDetector(config).detect(*views["left_approach"])
    assert len(observations) == 2
    left, right = sorted(observations, key=lambda item: item.position[0])
    assert left.occluded
    assert not right.occluded
    tracker = MultiObjectTracker()
    whole = RGBDDetector(config).detect(*views["home"])
    tracker.update(whole, 0.0)
    previous = tracker.snapshots(0.0)
    tracker.update(observations, 0.1)
    current = tracker.snapshots(0.1)
    assert len(current) == 2
    assert current[0].last_seen == 0.0
    assert not current[0].visible
    np.testing.assert_array_equal(current[0].size, previous[0].size)
    assert current[1].last_seen == 0.1


@pytest.mark.parametrize(
    "mode,expected_occluded", [("off", 0), ("observe", 4), ("refine", 4)]
)
def test_sensor_counts_occluded_detections_per_camera_and_globally_not_frames(
    mode, expected_occluded
):
    from dual_arm_conveyor.vision import SensorPipeline

    rgb, depth, _ = image_fixture(foreground_height=0.28)
    rgb[35:47, 65:78] = [220, 30, 20]
    depth[35:47, 65:78] = 1.0 - 0.255
    rgb[34, 68:71] = [110, 110, 110]
    depth[34, 68:71] = 1.0 - 0.28
    fovy = math.degrees(2 * math.atan(80 / (2 * 200)))
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><camera name="overhead" pos="0 0 1" fovy="{fovy}"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cfg = config(mode, noise=0.0)
    cfg.belt_speed = 0.1
    cfg.camera_width, cfg.camera_height = 100, 80
    cfg.camera_hz, cfg.camera_latency, cfg.camera_dropout = 10.0, 0.0, 0.0
    sensor = SensorPipeline(model, cfg)

    class Renderer:
        scene = SimpleNamespace(flags=np.ones(32, dtype=bool))
        is_depth = False

        def disable_depth_rendering(self):
            self.is_depth = False

        def enable_depth_rendering(self):
            self.is_depth = True

        def update_scene(self, *args, **kwargs):
            pass

        def render(self):
            return depth if self.is_depth else rgb

        def close(self):
            pass

    sensor._renderer = Renderer()
    try:
        sensor.update(data, 0.0)
        sensor.update(data, 0.1)
        diagnostics = sensor.diagnostics()
        assert diagnostics["delivered_frames"] == 2
        assert diagnostics["total_detections"] == 4
        assert diagnostics["occluded_detections"] == expected_occluded
        assert (
            diagnostics["per_camera"]["overhead"]["occluded_detections"]
            == expected_occluded
        )
        assert diagnostics["per_camera"]["overhead"]["total_detections"] == 4
        sensor.reset(5)
        assert sensor.diagnostics()["occluded_detections"] == 0
        assert (
            sensor.diagnostics()["per_camera"]["overhead"]["occluded_detections"] == 0
        )
    finally:
        sensor.close()
