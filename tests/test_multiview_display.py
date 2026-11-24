"""Delivered multi-camera images remain distinct in live and recorded panels."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import ImageDraw

from dual_arm_conveyor import live_view
from dual_arm_conveyor.live_view import CameraPanel, compose_camera_views


COLORS = [(181, 27, 53), (22, 149, 69), (39, 72, 203)]
NAMES = ["overhead", "oblique_south", "oblique_north"]


def _views(count=3):
    views = []
    for index in range(count):
        overlay = np.full((60, 80, 3), COLORS[index], dtype=np.uint8)
        depth = np.full((60, 80), 1.3 + index * 0.25, dtype=np.float32)
        overlay.setflags(write=False)
        depth.setflags(write=False)
        views.append(
            {
                "name": NAMES[index],
                "rgb": np.zeros_like(overlay),
                "overlay": overlay,
                "depth": depth,
                "frame_timestamp": 2.0,
                "diagnostics": {"detections": index + 1, "tracks": index + 1},
            }
        )
    return views


class _MultiSensor:
    def __init__(self, count=3):
        self.views = _views(count)
        self.reads = 0

    def camera_views(self):
        self.reads += 1
        return [dict(view) for view in self.views]

    def __getattr__(self, name):
        raise AssertionError(f"Panel must read delivered camera_views only, not {name}")


class _Viewer:
    def __init__(self, width=1000, height=700, left=120, bottom=30):
        self.viewport = SimpleNamespace(
            width=width, height=height, left=left, bottom=bottom
        )
        self.uploads = []
        self.clears = 0

    def set_images(self, rect_image):
        self.uploads.append(rect_image)

    def clear_images(self):
        self.clears += 1


@pytest.fixture
def labels(monkeypatch):
    labels = []
    original = ImageDraw.ImageDraw.text

    def text(draw, position, label, *args, **kwargs):
        labels.append(label)
        return original(draw, position, label, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text)
    return labels


@pytest.mark.parametrize("count", [2, 3])
def test_rgbd_shows_all_detection_overlays_and_only_primary_depth(
    count, monkeypatch, labels
):
    views = _views(count)
    depth_inputs = []
    original = live_view.depth_rgb

    def depth_rgb(depth, **kwargs):
        depth_inputs.append(depth)
        return original(depth, **kwargs)

    monkeypatch.setattr(live_view, "depth_rgb", depth_rgb)
    image = compose_camera_views(views, 480, 600, 2.1)
    assert image.dtype == np.uint8 and image.shape[1:] == (480, 3)
    assert image.shape[0] <= 600
    for index, view in enumerate(views):
        assert f"{view['name']} | RGB" in labels
        locations = np.argwhere(np.all(image == COLORS[index], axis=-1))
        assert len(locations) > 100
        # Resizing must letterbox, not stretch a camera's projection.
        height, width = locations.max(axis=0) - locations.min(axis=0) + 1
        assert width / height == pytest.approx(80 / 60, abs=0.03)
        np.testing.assert_array_equal(view["overlay"][0, 0], COLORS[index])
        assert not view["overlay"].flags.writeable
        assert not view["depth"].flags.writeable
    assert len(depth_inputs) == 1 and depth_inputs[0] is views[0]["depth"]
    assert f"{views[0]['name']} | DEPTH" in labels
    assert all(f"{view['name']} | DEPTH" not in labels for view in views[1:])


def test_capture_time_and_staleness_belong_to_each_view(labels):
    views = _views()
    for view, timestamp in zip(views, [3.8, 3.5, 3.99]):
        view["frame_timestamp"] = timestamp
        # A global diagnostic timestamp cannot override the delivered view time.
        view["diagnostics"]["delivered_frame_time"] = 4.0
    compose_camera_views(views, 480, 600, 4.0)
    expected = ["t=3.80s", "t=3.50s | STALE", "t=3.99s"]
    for view, status in zip(views, expected):
        label_index = labels.index(f"{view['name']} | RGB")
        assert labels[label_index + 1] == status
    depth_index = labels.index("overhead | DEPTH")
    assert labels[depth_index + 1] == expected[0]


def test_waiting_camera_keeps_its_slot_without_borrowing_other_camera_buffers(labels):
    views = _views()
    views[1].update(rgb=None, overlay=None, depth=None, frame_timestamp=None)
    views[1]["diagnostics"]["delivered_frame_time"] = 2.0
    image = compose_camera_views(views, 480, 600, 2.1)
    index = labels.index("oblique_south | RGB")
    assert labels[index + 1] == "Waiting for delayed frame"
    assert "No delivered image" in labels
    assert not np.any(np.all(image == COLORS[1], axis=-1))
    assert np.any(np.all(image == COLORS[0], axis=-1))
    assert np.any(np.all(image == COLORS[2], axis=-1))


def test_rgb_mode_shows_all_views_without_colorizing_any_depth(monkeypatch, labels):
    def forbidden(*args):
        raise AssertionError("RGB display must not process a depth map")

    monkeypatch.setattr(live_view, "depth_rgb", forbidden)
    image = CameraPanel("rgb").compose(_MultiSensor(), 2.1, 480, 600)
    for color in COLORS:
        assert np.any(np.all(image == color, axis=-1))
    assert all("DEPTH" not in label and "Depth:" not in label for label in labels)


def test_rgb_buffer_is_shown_when_one_view_has_no_overlay():
    views = _views(2)
    views[1]["rgb"] = views[1]["overlay"]
    views[1]["overlay"] = None
    image = compose_camera_views(views, 480, 600, 2.1, "rgb")
    assert np.any(np.all(image == COLORS[1], axis=-1))


def test_off_mode_never_reads_inputs():
    class Forbidden:
        def __getattribute__(self, name):
            raise AssertionError(f"Off mode accessed {name}")

    panel = CameraPanel("off")
    assert panel.compose(Forbidden(), 1.0, 480, 600) is None
    panel.update(Forbidden(), Forbidden(), 1.0)
    assert compose_camera_views(Forbidden(), 480, 600, 1.0, "off") is None
    assert panel.uploads == 0


def test_live_cache_updates_for_secondary_delivery_and_independent_stale_transition():
    viewer = _Viewer()
    sensor = _MultiSensor()
    panel = CameraPanel()
    panel.update(viewer, sensor, 2.1)
    panel.update(viewer, sensor, 2.2)
    assert panel.uploads == 1
    # The primary did not change, but another camera delivered a new frame.
    sensor.views[1]["frame_timestamp"] = 2.2
    panel.update(viewer, sensor, 2.25)
    assert panel.uploads == 2
    # Only primary/third become stale, then the second becomes stale later.
    panel.update(viewer, sensor, 2.31)
    assert panel.uploads == 3
    panel.update(viewer, sensor, 2.4)
    assert panel.uploads == 3
    panel.update(viewer, sensor, 2.51)
    assert panel.uploads == 4
    panel.update(viewer, sensor, 2.8)
    assert panel.uploads == 4


def test_recording_composer_reuses_unchanged_image_and_invalidates_on_layout_change():
    sensor = _MultiSensor()
    panel = CameraPanel()
    first = panel.compose(sensor, 2.1, 480, 600)
    assert panel.compose(sensor, 2.2, 480, 600) is first
    resized = panel.compose(sensor, 2.2, 320, 400)
    assert resized is not first and resized.shape[1] == 320
    sensor.views.reverse()
    reordered = panel.compose(sensor, 2.2, 320, 400)
    assert not np.array_equal(reordered, resized)


@pytest.mark.parametrize("size", [(160, 120), (500, 300), (1000, 700)])
def test_multiview_inset_stays_within_offset_viewport_and_recovers_from_minimize(size):
    viewer = _Viewer(*size)
    sensor = _MultiSensor()
    panel = CameraPanel()
    panel.update(viewer, sensor, 2.1)
    rect, pixels = viewer.uploads[-1]
    viewport = viewer.viewport
    assert pixels.shape == (rect.height, rect.width, 3)
    assert rect.left >= viewport.left and rect.bottom >= viewport.bottom
    assert rect.left + rect.width <= viewport.left + viewport.width
    assert rect.bottom + rect.height <= viewport.bottom + viewport.height
    width = viewport.width
    viewport.width = 80
    panel.update(viewer, sensor, 2.1)
    panel.update(viewer, sensor, 2.1)
    assert viewer.clears == 1
    viewport.width = width
    panel.update(viewer, sensor, 2.1)
    assert panel.uploads == 2


def test_pure_compositor_rejects_invalid_dimensions_and_modes():
    with pytest.raises(ValueError, match="positive"):
        compose_camera_views(_views(), 0, 400, 2.1)
    with pytest.raises(ValueError, match="camera view"):
        compose_camera_views(_views(), 400, 400, 2.1, "stereo")


def test_primary_calibrated_depth_range_sets_colors_legend_and_cache(labels):
    sensor = _MultiSensor(2)
    sensor.views[0]["depth_range_m"] = (0.9, 1.5)
    sensor.views[0]["depth"] = np.full((60, 80), 1.2, dtype=np.float32)
    panel = CameraPanel()
    first = panel.compose(sensor, 2.1, 480, 600)
    midpoint_color = live_view.depth_rgb(np.array([[1.2]]), near=0.9, far=1.5)[0, 0]
    assert np.any(np.all(np.abs(first.astype(float) - midpoint_color) <= 1, axis=-1))
    assert "Depth: 0.9m yellow / 1.5m blue" in labels
    sensor.views[0]["depth_range_m"] = (1.3, 1.8)
    second = panel.compose(sensor, 2.1, 480, 600)
    assert not np.array_equal(second, first)
    assert "Depth: 1.3m yellow / 1.8m blue" in labels


@pytest.mark.parametrize("near,far", [(1.0, 1.0), (1.0, 0.5), (0, 1), (1, np.inf)])
def test_depth_scale_rejects_invalid_calibration(near, far):
    with pytest.raises(ValueError, match="depth range"):
        live_view.depth_rgb(np.array([[1.2]]), near=near, far=far)
