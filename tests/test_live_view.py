"""Exercise the real MuJoCo image-overlay adapter without opening a window."""

from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
import pytest
from PIL import ImageDraw

from dual_arm_conveyor import cli
from dual_arm_conveyor.live_view import CameraPanel, depth_rgb


class _Sim:
    """Only the public adapter's weak-reference target is replaced."""

    def __init__(self, viewport):
        self.viewport = viewport
        self.images = []
        self.upload_calls = 0
        self.clear_calls = 0

    def set_images(self, images):
        self.upload_calls += 1
        self.images = [(rect, pixels.copy()) for rect, pixels in images]
        assert all(pixels.flags.c_contiguous for _, pixels in images)

    def clear_images(self):
        self.clear_calls += 1
        self.images = []


def _viewer(left=180, bottom=25, width=900, height=640):
    sim = _Sim(mujoco.MjrRect(left, bottom, width, height))
    handle = mujoco.viewer.Handle(
        sim, mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvPerturb(), None
    )
    return handle, sim


class _Sensor:
    def __init__(self, timestamp=2.0, delivered=True):
        self.timestamp = timestamp if delivered else None
        self.overlay = np.empty((48, 64, 3), dtype=np.uint8) if delivered else None
        self.depth = np.full((48, 64), 1.55, dtype=np.float32) if delivered else None
        if delivered:
            self.overlay[:24] = [193, 17, 43]
            self.overlay[24:] = [19, 71, 211]
            self.overlay.setflags(write=False)
            self.depth.setflags(write=False)

    def diagnostics(self):
        return {
            "delivered_frame_time": self.timestamp,
            "detections": 2,
            "tracks": 2,
        }

    def __getattr__(self, name):
        raise AssertionError(
            f"Live display must not access {name}; only delivered images/diagnostics"
        )


def _inside_viewport(sim):
    assert sim.images
    viewport = sim.viewport
    for rect, pixels in sim.images:
        assert isinstance(rect, mujoco.MjrRect)
        assert pixels.shape == (rect.height, rect.width, 3)
        assert pixels.dtype == np.uint8
        assert rect.width > 0 and rect.height > 0
        assert viewport.left <= rect.left
        assert viewport.bottom <= rect.bottom
        assert rect.left + rect.width <= viewport.left + viewport.width
        assert rect.bottom + rect.height <= viewport.bottom + viewport.height


@pytest.mark.parametrize("mode", ["rgb", "rgbd"])
def test_real_handle_accepts_panel_shape_and_does_the_single_vertical_flip(mode):
    viewer, sim = _viewer()
    sensor = _Sensor()
    panel = CameraPanel(mode)
    panel.update(viewer, sensor, simulation_time=2.1)
    assert sim.upload_calls == panel.uploads == 1
    _inside_viewport(sim)
    # Handle.set_images performs the flip for OpenGL. Undo that adapter flip
    # to inspect normal screen rows; a caller-side flip would reverse the image.
    screen = sim.images[0][1][::-1]
    red_rows = np.where(np.all(screen == [193, 17, 43], axis=-1))[0]
    blue_rows = np.where(np.all(screen == [19, 71, 211], axis=-1))[0]
    assert len(red_rows) > 100 and len(blue_rows) > 100
    assert red_rows.mean() < blue_rows.mean()
    np.testing.assert_array_equal(
        sensor.overlay[:24], np.tile([193, 17, 43], (24, 64, 1))
    )
    np.testing.assert_array_equal(
        sensor.overlay[24:], np.tile([19, 71, 211], (24, 64, 1))
    )


def test_delivered_frame_and_stale_status_cache_avoid_repeated_uploads():
    viewer, sim = _viewer()
    sensor = _Sensor()
    panel = CameraPanel()
    panel.update(viewer, sensor, 2.1)
    initial_image = sim.images[0][1].copy()
    for now in (2.12, 2.15, 2.25):
        panel.update(viewer, sensor, now)
    assert sim.upload_calls == 1
    panel.update(viewer, sensor, 2.31)
    assert sim.upload_calls == 2  # the stale status becomes visible once
    assert not np.array_equal(sim.images[0][1], initial_image)
    panel.update(viewer, sensor, 3.0)
    assert sim.upload_calls == 2
    sensor.timestamp = 2.95
    sensor.overlay = np.full((48, 64, 3), [33, 144, 71], dtype=np.uint8)
    panel.update(viewer, sensor, 3.05)
    assert sim.upload_calls == 3
    assert np.any(np.all(sim.images[0][1] == [33, 144, 71], axis=-1))


@pytest.mark.parametrize(
    "viewport", [(0, 0, 160, 120), (240, 50, 500, 300), (100, 25, 1400, 900)]
)
def test_resizing_keeps_uploaded_panel_inside_scene_viewport(viewport):
    viewer, sim = _viewer()
    sensor = _Sensor()
    panel = CameraPanel()
    panel.update(viewer, sensor, 2.1)
    first = sim.images[0][0]
    sim.viewport = mujoco.MjrRect(*viewport)
    panel.update(viewer, sensor, 2.1)
    assert sim.upload_calls == 2
    _inside_viewport(sim)
    rect = sim.images[0][0]
    assert (rect.left, rect.bottom, rect.width, rect.height) != (
        first.left,
        first.bottom,
        first.width,
        first.height,
    )
    panel.update(viewer, sensor, 2.1)
    assert sim.upload_calls == 2


def test_minimizing_clears_overlay_once_and_restoring_uploads_same_frame_again():
    viewer, sim = _viewer()
    sensor = _Sensor()
    panel = CameraPanel()
    panel.update(viewer, sensor, 2.1)
    sim.viewport = mujoco.MjrRect(0, 0, 80, 60)
    panel.update(viewer, sensor, 2.1)
    assert sim.clear_calls == 1 and not sim.images
    panel.update(viewer, sensor, 2.1)
    assert sim.clear_calls == 1
    sim.viewport = mujoco.MjrRect(200, 30, 800, 600)
    panel.update(viewer, sensor, 2.1)
    assert sim.upload_calls == 2
    _inside_viewport(sim)


def test_off_mode_does_not_read_sensor_or_require_viewer_overlay_support():
    class Forbidden:
        def __getattribute__(self, name):
            raise AssertionError(f"Off mode accessed {name}")

    panel = CameraPanel("off")
    panel.update(Forbidden(), Forbidden(), 4.0)
    assert panel.uploads == 0


def test_waiting_and_oracle_states_are_visible_without_capturing_a_frame(monkeypatch):
    labels = []
    original = ImageDraw.ImageDraw.text

    def text(draw, position, label, *args, **kwargs):
        labels.append(label)
        return original(draw, position, label, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text)
    viewer, sim = _viewer()
    panel = CameraPanel()
    panel.update(viewer, _Sensor(delivered=False), 0.05)
    assert "Waiting for delayed frame" in labels
    assert "No delivered image" in labels
    _inside_viewport(sim)
    labels.clear()
    panel.update(viewer, None, 1.0)
    assert "ORACLE: no sensor input" in labels
    assert sim.upload_calls == 2


def test_depth_coloring_is_finite_fixed_scale_and_does_not_modify_readonly_input():
    depth = np.array(
        [[np.nan, np.inf, -np.inf, 0.0, -1.0, 1.3, 1.8, 1.55]], dtype=np.float32
    )
    original = depth.copy()
    depth.setflags(write=False)
    colored = depth_rgb(depth)
    assert colored.shape == (1, 8, 3) and colored.dtype == np.uint8
    assert np.isfinite(colored).all()
    np.testing.assert_array_equal(colored[0, :5], np.zeros((5, 3), dtype=np.uint8))
    np.testing.assert_allclose(colored[0, 5], [250, 220, 65], atol=1)
    np.testing.assert_allclose(colored[0, 6], [35, 55, 150], atol=1)
    assert not np.array_equal(colored[0, 7], colored[0, 5])
    np.testing.assert_array_equal(depth, original)
    sensor = _Sensor()
    sensor.depth = np.tile(depth, (48, 8))
    sensor.depth.setflags(write=False)
    before_rgb, before_depth = sensor.overlay.copy(), sensor.depth.copy()
    viewer, sim = _viewer()
    CameraPanel().update(viewer, sensor, 2.1)
    _inside_viewport(sim)
    np.testing.assert_array_equal(sensor.overlay, before_rgb)
    np.testing.assert_array_equal(sensor.depth, before_depth)


def test_rgb_mode_does_not_colorize_depth(monkeypatch):
    from dual_arm_conveyor import live_view

    def forbidden(*args):
        raise AssertionError("RGB-only mode should not create a depth image")

    monkeypatch.setattr(live_view, "depth_rgb", forbidden)
    viewer, sim = _viewer()
    CameraPanel("rgb").update(viewer, _Sensor(), 2.1)
    _inside_viewport(sim)


def test_released_viewer_weak_reference_is_safe():
    viewer, sim = _viewer()
    del sim
    assert viewer.viewport is None
    CameraPanel().update(viewer, _Sensor(), 2.1)


def test_missing_image_overlay_api_has_an_actionable_error():
    with pytest.raises(RuntimeError, match="camera-view off"):
        CameraPanel().update(SimpleNamespace(), _Sensor(), 2.1)


def test_cli_camera_panel_defaults_to_rgbd_and_accepts_rgb_or_off():
    assert cli.parser().parse_args(["play"]).camera_view == "rgbd"
    for mode in ("rgb", "off"):
        assert (
            cli.parser().parse_args(["play", "--camera-view", mode]).camera_view == mode
        )
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["play", "--camera-view", "stereo"])
    with pytest.raises(ValueError):
        CameraPanel("stereo")
