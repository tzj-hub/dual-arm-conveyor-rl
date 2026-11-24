"""Task-triggered wrist observations are distinct from global tracking input."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import ImageDraw

from dual_arm_conveyor.live_view import (
    CameraPanel,
    CameraViewGroup,
    compose_camera_views,
)


def _view(name, color, **fields):
    rgb = np.full((60, 80, 3), color, dtype=np.uint8)
    rgb.setflags(write=False)
    return dict(
        name=name,
        rgb=rgb,
        overlay=rgb,
        depth=np.full((60, 80), 1.5),
        frame_timestamp=2.0,
        diagnostics={},
        **fields,
    )


def _views():
    return [
        _view("overhead", [173, 21, 39]),
        _view(
            "r0/wrist_rgbd",
            [22, 137, 65],
            role="local refinement",
            active=False,
            target_track_id=None,
        ),
        _view(
            "r1/wrist_rgbd",
            [39, 77, 187],
            role="local refinement",
            active=False,
            target_track_id=7,
        ),
    ]


class _Views:
    def __init__(self, views):
        self.views = views

    def camera_views(self):
        return list(self.views)

    def __getattr__(self, name):
        raise AssertionError(f"Display unexpectedly accessed {name}")


@pytest.fixture
def labels(monkeypatch):
    output = []
    original = ImageDraw.ImageDraw.text

    def draw_text(draw, position, label, *args, **kwargs):
        output.append(label)
        return original(draw, position, label, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", draw_text)
    return output


def test_wrist_idle_and_paused_keep_capture_times_without_false_stale(labels):
    views = _views()
    image = compose_camera_views(views, 480, 600, 8.0)
    assert image.shape == (538, 480, 3)
    assert "IDLE | no target" in labels
    assert "PAUSED | T7" in labels
    assert labels.count("Last t=2.00s") == 2
    # Global RGB/depth remain stale even while the local captures are paused.
    assert labels.count("t=2.00s | STALE") == 2
    assert labels.count("LOCAL REFINEMENT") == 2
    for view in views:
        assert np.any(np.all(image == view["rgb"][0, 0], axis=-1))


def test_active_wrist_waits_for_first_delivery_and_then_reports_its_own_staleness(
    labels,
):
    views = _views()
    local = views[1]
    local.update(
        active=True,
        target_track_id=0,
        frame_timestamp=None,
        rgb=None,
        overlay=None,
        depth=None,
    )
    compose_camera_views(views, 480, 600, 8.0)
    assert "WAITING | T0" in labels
    assert "No delivered frame" in labels
    labels.clear()
    local.update(frame_timestamp=7.9)
    compose_camera_views(views, 480, 600, 8.0)
    assert "ACTIVE | T0" in labels and "t=7.90s" in labels
    labels.clear()
    compose_camera_views(views, 480, 600, 8.3)
    assert "t=7.90s | STALE" in labels


def test_state_and_target_changes_refresh_same_frame_and_paused_age_does_not():
    views = _views()
    sensor = _Views(views)
    panel = CameraPanel()
    first = panel.compose(sensor, 8.0, 480, 600)
    assert panel.compose(sensor, 8.2, 480, 600) is first
    views[1]["target_track_id"] = 13
    assigned = panel.compose(sensor, 8.2, 480, 600)
    assert not np.array_equal(assigned, first)
    views[1]["active"] = True
    activated = panel.compose(sensor, 8.2, 480, 600)
    assert not np.array_equal(activated, assigned)
    views[1]["role"] = "global tracking"
    changed_role = panel.compose(sensor, 8.2, 480, 600)
    assert not np.array_equal(changed_role, activated)


def test_idle_without_a_delivered_image_is_not_a_capture_failure(labels):
    views = _views()
    views[1].update(rgb=None, overlay=None, depth=None, frame_timestamp=None)
    compose_camera_views(views, 480, 600, 8.0)
    index = labels.index("r0/wrist_rgbd | RGB")
    assert labels[index + 1 : index + 4] == [
        "LOCAL REFINEMENT",
        "IDLE | no target",
        "No delivered frame",
    ]


def test_single_local_view_also_has_task_status(labels):
    image = compose_camera_views([_views()[2]], 320, 500, 8.0, "rgb")
    assert image.shape[1] == 320
    assert "LOCAL REFINEMENT" in labels
    assert "PAUSED | T7" in labels
    assert "Last t=2.00s" in labels
    assert not any("STALE" in label for label in labels)


@pytest.mark.parametrize(
    "mode,caption", [("observe", "OBSERVE ONLY"), ("refine", "EXPERIMENTAL REFINE")]
)
def test_control_mode_is_explicit_without_changing_the_internal_role(
    mode, caption, labels
):
    views = _views()
    for view in views[1:]:
        view["control_mode"] = mode
    compose_camera_views(views, 480, 600, 8.0)
    assert labels.count(caption) == 2
    assert "LOCAL REFINEMENT" not in labels
    assert all(view["role"] == "local refinement" for view in views[1:])
    labels.clear()
    compose_camera_views(views[1:2], 320, 500, 8.0)
    assert caption in labels


def test_control_mode_label_invalidates_the_cached_panel_without_a_new_image():
    views = _views()
    sensor = _Views(views)
    for view in views[1:]:
        view["control_mode"] = "observe"
    panel = CameraPanel()
    observed = panel.compose(sensor, 8.0, 480, 600)
    for view in views[1:]:
        view["control_mode"] = "refine"
    experimental = panel.compose(sensor, 8.0, 480, 600)
    assert not np.array_equal(observed, experimental)


def test_reassigned_target_does_not_relabel_the_retained_previous_image(labels):
    views = _views()
    views[1].update(active=True, target_track_id=12, frame_target_track_id=4)
    sensor = _Views(views)
    panel = CameraPanel()
    prior = panel.compose(sensor, 8.0, 480, 600)
    assert "WAITING | T12" in labels
    assert "Last T4 t=2.00s" in labels
    assert "ACTIVE | T12" not in labels
    # Provenance alone is display state and must invalidate the cached panel.
    labels.clear()
    views[1]["frame_target_track_id"] = 12
    current = panel.compose(sensor, 8.0, 480, 600)
    assert not np.array_equal(current, prior)
    assert "ACTIVE | T12" in labels
    assert "Last T4 t=2.00s" not in labels


def test_display_group_preserves_primary_api_without_merging_controller_data():
    views = _views()
    diag = {"tracks": 3, "track_ids": [1, 2, 3]}

    class Primary(_Views):
        def diagnostics(self):
            return diag

    primary = Primary(views[:1])
    primary.rgb = views[0]["rgb"]
    primary.overlay = views[0]["overlay"]
    primary.depth = views[0]["depth"]
    group = CameraViewGroup(primary, _Views(views[1:]))
    combined = group.camera_views()
    assert [view["name"] for view in combined] == [view["name"] for view in views]
    assert combined[0]["rgb"] is primary.rgb
    assert group.diagnostics() is diag
    assert group.rgb is primary.rgb
    assert group.overlay is primary.overlay
    assert group.depth is primary.depth
    combined.pop()
    assert len(group.camera_views()) == 3
    assert len(primary.views) == 1
    # Both sensors expose only their delivered view API; composition cannot
    # call capture, render, object state, or a local/global tracker.
    assert CameraPanel().compose(group, 8.0, 480, 600).shape == (538, 480, 3)


def test_long_camera_names_fit_their_own_tile_without_crossing_into_neighbor(labels):
    views = _views()
    views[1]["name"] = "r0/" + "very_long_camera_name_" * 5
    compose_camera_views(views, 324, 500, 8.0)
    shortened = next(label for label in labels if label.startswith("r0/"))
    assert shortened.endswith("... | RGB")
    from PIL import Image

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    assert draw.textlength(shortened) <= (324 - 24) // 2


@pytest.mark.parametrize(
    "width,height,expected",
    [(900, 640, (420, 324)), (1280, 720, (524, 460)), (1920, 1080, (538, 480))],
)
def test_wrist_panel_dimensions_and_viewport_bounds(width, height, expected):
    images = []
    viewer = SimpleNamespace(
        viewport=SimpleNamespace(left=130, bottom=20, width=width, height=height),
        set_images=lambda image: images.append(image),
    )
    CameraPanel().update(viewer, _Views(_views()), 8.0)
    rect, pixels = images[0]
    assert pixels.shape[:2] == expected
    assert rect.left >= 130 and rect.bottom >= 20
    assert rect.left + rect.width <= 130 + width
    assert rect.bottom + rect.height <= 20 + height
