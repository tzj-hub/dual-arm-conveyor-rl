"""Independent checks of synchronized delivery, image age and camera RNGs."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor import vision
from dual_arm_conveyor.config import Config


class FakeRenderer:
    instances = []

    def __init__(self, model, height, width):
        self.height, self.width = height, width
        self.scene = SimpleNamespace(flags=np.ones(32, dtype=bool))
        self.is_depth = False
        self.renders = []
        self.close_calls = 0
        self.instances.append(self)

    def disable_depth_rendering(self):
        self.is_depth = False

    def enable_depth_rendering(self):
        self.is_depth = True

    def update_scene(self, data, camera):
        self.camera = camera
        self.timestamp = data.time

    def render(self):
        camera_index = ("overhead", "oblique_south", "oblique_north").index(self.camera)
        self.renders.append((self.camera, self.timestamp, self.is_depth))
        if self.is_depth:
            return np.full(
                (self.height, self.width), 1.2 + 0.01 * camera_index, dtype=np.float32
            )
        return np.full(
            (self.height, self.width, 3),
            [camera_index, round(self.timestamp * 100) % 256, 20],
            dtype=np.uint8,
        )

    def close(self):
        self.close_calls += 1


class FixedDropout:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


@pytest.fixture
def pipeline_factory(monkeypatch):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="2 2 .1"/>'
        '<camera name="overhead" pos="0 0 1.8"/>'
        '<camera name="oblique_south" pos="-.1 -.5 1.35"/>'
        '<camera name="oblique_north" pos="-.1 .5 1.35"/>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    FakeRenderer.instances = []
    monkeypatch.setattr(vision.mujoco, "Renderer", FakeRenderer)
    pipelines = []

    def create(**kwargs):
        config = replace(
            Config(camera_width=64, camera_height=64, depth_noise=0, camera_dropout=0),
            **kwargs,
        )
        pipeline = vision.SensorPipeline(model, config)

        def detect(rgb, depth, calibration):
            # All views observe the same synthetic moving object, using only
            # pixels. Distinct camera RGB stamps expose accidental view mixing.
            timestamp = float(rgb[0, 0, 1]) / 100
            return [
                vision.Detection(
                    np.array([-0.2 + 0.1 * timestamp, 0.0, 0.23]),
                    0.0,
                    0,
                    0.9,
                    (5, 5, 10, 10),
                )
            ]

        monkeypatch.setattr(pipeline.detector, "detect", detect)
        pipelines.append(pipeline)
        return pipeline, data

    yield create
    for pipeline in pipelines:
        pipeline.close()


def update(pipeline, data, timestamp):
    data.time = timestamp
    return pipeline.update(data)


def latest_capture(pipeline, timestamp):
    if pipeline._queue and pipeline._queue[-1].timestamp == timestamp:
        return pipeline._queue[-1].frames
    return {}


def test_synchronized_batch_is_delayed_and_corrects_tracker_once(
    pipeline_factory, monkeypatch
):
    pipeline, data = pipeline_factory(camera_rig="triple", camera_latency=0.08)
    corrections = []
    correct = pipeline.tracker.update

    def record(detections, timestamp):
        corrections.append((timestamp, len(detections)))
        correct(detections, timestamp)

    monkeypatch.setattr(pipeline.tracker, "update", record)
    assert update(pipeline, data, 0.0) == []
    batch = pipeline._queue[0]
    assert len(batch.frames) == 3
    assert {frame.timestamp for frame in batch.frames.values()} == {0.0}
    assert {time for _, time, _ in pipeline._renderer.renders} == {0.0}
    assert update(pipeline, data, 0.04) == []
    first = update(pipeline, data, 0.08)
    assert corrections == [(0.0, 1)]
    assert len(first) == 1 and first[0].hits == 1
    update(pipeline, data, 0.10)
    assert corrections == [(0.0, 1)]
    second = update(pipeline, data, 0.18)
    assert corrections == [(0.0, 1), (0.1, 1)]
    assert len(second) == 1 and second[0].hits == 2
    diagnostics = pipeline.diagnostics()
    assert diagnostics["raw_detections"] == 6
    assert diagnostics["fused_detections"] == 2
    assert diagnostics["frame_age_seconds"] == pytest.approx(0.08)
    assert all(view["frame_timestamp"] == 0.1 for view in pipeline.camera_views())


def test_all_queued_batches_keep_capture_timestamps_when_delivered_together(
    pipeline_factory, monkeypatch
):
    pipeline, data = pipeline_factory(camera_rig="triple", camera_latency=0.15)
    timestamps = []
    correct = pipeline.tracker.update

    def record(detections, timestamp):
        timestamps.append(timestamp)
        correct(detections, timestamp)

    monkeypatch.setattr(pipeline.tracker, "update", record)
    for timestamp in (0.0, 0.1, 0.2):
        data.time = timestamp
        pipeline.capture(data)
    update(pipeline, data, 0.4)
    assert timestamps == [0.0, 0.1, 0.2]
    assert pipeline.diagnostics()["delivered_frames"] == 3
    assert pipeline._queue[0].timestamp == 0.4


def test_camera_rng_and_dropout_are_independent_of_active_rig(pipeline_factory):
    single, data = pipeline_factory(
        camera_rig="overhead", depth_noise=0.001, camera_dropout=0.4, seed=91
    )
    dual, _ = pipeline_factory(
        camera_rig="dual", depth_noise=0.001, camera_dropout=0.4, seed=91
    )
    triple, _ = pipeline_factory(
        camera_rig="triple", depth_noise=0.001, camera_dropout=0.4, seed=91
    )
    histories = {name: [] for name in triple.camera_names}
    for index in range(20):
        timestamp = index * 0.1
        data.time = timestamp
        for pipeline in (single, dual, triple):
            pipeline.capture(data)
        frames = latest_capture(triple, timestamp)
        for other in (single, dual):
            comparison = latest_capture(other, timestamp)
            for name in other.camera_names:
                assert (name in comparison) == (name in frames)
                if name in frames:
                    np.testing.assert_array_equal(
                        frames[name].rgb, comparison[name].rgb
                    )
                    np.testing.assert_array_equal(
                        frames[name].depth, comparison[name].depth
                    )
        for name in histories:
            histories[name].append(name in frames)
    assert len({tuple(history) for history in histories.values()}) == 3
    assert all(any(history) and not all(history) for history in histories.values())


@pytest.mark.parametrize("initial_primary_frame", [False, True])
def test_primary_dropout_preserves_its_actual_image_age(
    pipeline_factory, initial_primary_frame
):
    pipeline, data = pipeline_factory(camera_rig="triple", camera_latency=0)
    if initial_primary_frame:
        update(pipeline, data, 0.0)
    original = pipeline.overlay
    pipeline.dropout = 0.5
    pipeline._rngs = {name: FixedDropout(0.9) for name in pipeline.camera_names}
    pipeline._rngs["overhead"] = FixedDropout(0.1)
    tracks = update(pipeline, data, 0.1)
    assert len(tracks) == 1
    assert tracks[0].timestamp == 0.1
    assert pipeline.overlay is original
    views = {view["name"]: view for view in pipeline.camera_views()}
    primary = views["overhead"]
    assert primary["frame_timestamp"] == (0.0 if initial_primary_frame else None)
    age = pipeline.diagnostics()["frame_age_seconds"]
    assert age == (0.1 if initial_primary_frame else None)
    assert views["oblique_south"]["frame_timestamp"] == 0.1
    assert views["oblique_north"]["frame_timestamp"] == 0.1
    assert pipeline.diagnostics()["partial_batches"] == 1


def test_all_camera_dropout_expires_tracks_without_fabricating_frames(pipeline_factory):
    pipeline, data = pipeline_factory(camera_rig="triple", camera_latency=0)
    assert len(update(pipeline, data, 0.0)) == 1
    pipeline.dropout = 1.0
    assert update(pipeline, data, 1.6) == []
    diagnostics = pipeline.diagnostics()
    assert diagnostics["captured_frames"] == 2
    assert diagnostics["dropped_frames"] == 1
    assert diagnostics["delivered_frames"] == 1
    assert diagnostics["queued_frames"] == 0
    assert diagnostics["frame_age_seconds"] == 1.6
    assert all(v["frame_timestamp"] == 0.0 for v in pipeline.camera_views())


def test_queue_bound_reset_and_close_release_all_view_caches(pipeline_factory):
    pipeline, data = pipeline_factory(camera_rig="triple", camera_latency=0.15)
    for index in range(20):
        data.time = index * 0.1
        pipeline.capture(data)
    diagnostics = pipeline.diagnostics()
    capacity = diagnostics["queue_capacity"]
    assert capacity == 5
    assert diagnostics["queued_frames"] == capacity
    assert diagnostics["queued_images"] == diagnostics["image_queue_capacity"] == 15
    assert diagnostics["queue_overflows"] == 20 - capacity
    assert all(
        counts["queue_overflows"] == 20 - capacity
        for counts in diagnostics["per_camera"].values()
    )
    pipeline.reset(91)
    assert pipeline.diagnostics()["captured_frames"] == 0
    assert pipeline.diagnostics()["queued_images"] == 0
    assert all(view["frame_timestamp"] is None for view in pipeline.camera_views())
    update(pipeline, data, 0.0)
    update(pipeline, data, 0.15)
    assert pipeline.overlay is not None
    renderer = pipeline._renderer
    pipeline.close()
    pipeline.close()
    assert renderer.close_calls == 1
    assert pipeline._renderer is None
    assert pipeline.overlay is pipeline.rgb is pipeline.depth is None
    assert pipeline.diagnostics()["queued_images"] == 0
    assert pipeline.tracker.snapshots(0.15) == []
    assert all(
        all(
            view[field] is None
            for field in ("rgb", "depth", "overlay", "frame_timestamp")
        )
        for view in pipeline.camera_views()
    )


def test_capture_requires_only_time_and_camera_calibration_from_simulator(
    pipeline_factory,
):
    pipeline, data = pipeline_factory(camera_rig="triple", camera_latency=0)

    class CameraOnlyData:
        time = 0.0
        cam_xpos = data.cam_xpos
        cam_xmat = data.cam_xmat

        def __getattr__(self, name):
            raise AssertionError(
                f"Perception accessed unexpected simulator state: {name}"
            )

    tracks = pipeline.update(CameraOnlyData())
    assert len(tracks) == 1
    assert pipeline.diagnostics()["raw_detections"] == 3
