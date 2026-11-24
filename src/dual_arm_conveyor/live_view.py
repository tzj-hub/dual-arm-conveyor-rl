"""Display delivered camera observations without another sensor or render pass."""

from __future__ import annotations

from inspect import getattr_static

import numpy as np


DEPTH_NEAR = 1.3
DEPTH_FAR = 1.8
STALE_SECONDS = 0.3


def depth_rgb(depth, near=DEPTH_NEAR, far=DEPTH_FAR):
    """Fixed metre scale: near is yellow, far is blue; invalid pixels are black."""
    if not np.isfinite(near) or not np.isfinite(far) or not 0 < near < far:
        raise ValueError("depth range must be finite and satisfy 0 < near < far")
    depth = np.asarray(depth)
    valid = np.isfinite(depth) & (depth > 0)
    clean = np.where(valid, depth, far)
    fraction = np.clip((clean - near) / (far - near), 0, 1)
    near_color = np.array([250, 220, 65], dtype=float)
    far_color = np.array([35, 55, 150], dtype=float)
    rgb = (near_color + fraction[..., None] * (far_color - near_color)).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _delivered_views(sensor):
    if sensor is None:
        return []
    # Check the optional public API without triggering legacy sensors' dynamic
    # attribute lookups. A multi-camera sensor returns cached delivered buffers.
    if getattr_static(sensor, "camera_views", None) is not None:
        return list(sensor.camera_views())
    diagnostics = sensor.diagnostics()
    return [
        {
            "name": "overhead",
            "overlay": getattr(sensor, "overlay", None),
            "depth": getattr(sensor, "depth", None),
            "frame_timestamp": diagnostics.get("delivered_frame_time"),
            "diagnostics": diagnostics,
        }
    ]


def _timestamp(view):
    return view.get(
        "frame_timestamp", view.get("diagnostics", {}).get("delivered_frame_time")
    )


def _local_view(view):
    return view.get("role") == "local refinement"


def _local_caption(view):
    return {
        "observe": "OBSERVE ONLY",
        "refine": "EXPERIMENTAL REFINE",
    }.get(view.get("control_mode"), "LOCAL REFINEMENT")


def _frame_matches_target(view):
    return "frame_target_track_id" not in view or view[
        "frame_target_track_id"
    ] == view.get("target_track_id")


def _stale(view, simulation_time):
    # Task-triggered cameras intentionally retain their last observation while
    # idle or paused. Only an actively requested stream can become stale.
    if not view.get("active", True):
        return False
    if _local_view(view) and (
        view.get("target_track_id") is None or not _frame_matches_target(view)
    ):
        return False
    timestamp = _timestamp(view)
    return timestamp is not None and simulation_time - timestamp > STALE_SECONDS


def _local_status(view, simulation_time):
    target = view.get("target_track_id")
    timestamp = _timestamp(view)
    if target is None:
        state = "IDLE | no target"
    elif not view.get("active", True):
        state = f"PAUSED | T{target}"
    elif timestamp is None or not _frame_matches_target(view):
        state = f"WAITING | T{target}"
    else:
        state = f"ACTIVE | T{target}"
    if timestamp is None:
        capture = "No delivered frame"
    else:
        retained = (
            target is None
            or not view.get("active", True)
            or not _frame_matches_target(view)
        )
        prior_target = view.get("frame_target_track_id")
        if not _frame_matches_target(view) and prior_target is not None:
            prefix = f"Last T{prior_target} "
        else:
            prefix = "Last " if retained else ""
        capture = prefix + f"t={timestamp:.2f}s"
        if _stale(view, simulation_time):
            capture += " | STALE"
    return state, capture


def _rgb(view):
    overlay = view.get("overlay")
    return view.get("rgb") if overlay is None else overlay


def _depth_range(view):
    return tuple(view.get("depth_range_m", (DEPTH_NEAR, DEPTH_FAR)))


def _depth_image(view):
    depth = view.get("depth")
    near, far = _depth_range(view)
    return None if depth is None else depth_rgb(depth, near=near, far=far)


def _depth_legend(view):
    near, far = _depth_range(view)
    return f"{near:g}m yellow / {far:g}m blue"


def _views_key(views, simulation_time):
    return tuple(
        (
            view.get("name", "camera"),
            _timestamp(view),
            _stale(view, simulation_time),
            _rgb(view) is not None,
            view.get("depth") is not None,
            _depth_range(view),
            view.get("diagnostics", {}).get("detections", 0),
            view.get("diagnostics", {}).get("tracks", 0),
            view.get("role"),
            view.get("control_mode"),
            view.get("active", True),
            view.get("target_track_id"),
            "frame_target_track_id" in view,
            view.get("frame_target_track_id"),
        )
        for view in views
    )


def _pillow():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "Live camera images need Pillow: install '.[video]' or use "
            "--camera-view off."
        ) from exc
    return Image, ImageDraw


def _fit_text(draw, text, width):
    """Keep labels inside their own tile, including unusually long names."""
    if draw.textlength(text) <= width:
        return text
    while text and draw.textlength(text + "...") > width:
        text = text[:-1]
    return text + "..." if draw.textlength("...") <= width else ""


def _camera_label(draw, name, kind, width):
    suffix = f" | {kind}"
    name = _fit_text(draw, name, max(0, width - draw.textlength(suffix)))
    return _fit_text(draw, name + suffix, width)


def _paste_frame(canvas, draw, pixels, box):
    """Letterbox delivered pixels without changing their camera geometry."""
    left, top, width, height = box
    if pixels is None:
        draw.text((left, top + 8), "No delivered image", fill=(180, 180, 180))
        return
    if width < 1 or height < 1:
        return
    Image, _ = _pillow()
    frame = Image.fromarray(np.asarray(pixels, dtype=np.uint8))
    frame.thumbnail((width, height), Image.Resampling.NEAREST)
    canvas.paste(frame, (left + (width - frame.width) // 2, top))


def _compose_single(view, width, height_limit, simulation_time, mode, oracle):
    """Preserve the familiar full-width layout for one active camera."""
    Image, ImageDraw = _pillow()
    diagnostics = view.get("diagnostics", {})
    rgb = _rgb(view)
    stale = _stale(view, simulation_time)
    local = _local_view(view)
    header, section_header = (80 if local else 64), 20
    image_width = max(1, width - 16)
    sections = 2 if mode == "rgbd" else 1
    aspect = rgb.shape[0] / rgb.shape[1] if rgb is not None else 0.75
    image_height = max(
        1,
        min(
            round(image_width * aspect),
            (height_limit - header - sections * section_header - 8) // sections,
        ),
    )
    height = min(height_limit, header + sections * (section_header + image_height) + 8)
    canvas = Image.new("RGB", (width, height), (18, 24, 32))
    draw = ImageDraw.Draw(canvas)
    name = view.get("name", "overhead")
    title = _fit_text(draw, f"{name.upper()} | 1 RGB-D CAMERA", width - 16)
    draw.text((8, 7), title, fill=(235, 240, 245))
    timestamp = _timestamp(view)
    if local:
        state, capture = _local_status(view, simulation_time)
        for y, line in ((26, _local_caption(view)), (44, state), (62, capture)):
            draw.text(
                (8, y),
                _fit_text(draw, line, width - 16),
                fill=(245, 190, 90) if y == 62 and stale else (190, 205, 220),
            )
    elif oracle:
        status = "ORACLE: no sensor input"
    elif timestamp is None:
        status = "Waiting for delayed frame"
    else:
        status = f"Capture t={timestamp:.2f}s" + (" | STALE" if stale else "")
    if not local:
        draw.text((8, 26), status, fill=(245, 190, 90) if stale else (190, 205, 220))
    if not oracle and not local:
        draw.text(
            (8, 44),
            f"Detections {diagnostics.get('detections', 0)}  "
            f"Tracks {diagnostics.get('tracks', 0)}",
            fill=(190, 205, 220),
        )
    for index in range(sections):
        y = header + index * (section_header + image_height)
        label = (
            "RGB + DETECTOR TRACKS" if index == 0 else "DEPTH | " + _depth_legend(view)
        )
        draw.text((8, y), label, fill=(225, 230, 240))
        pixels = rgb if index == 0 else _depth_image(view)
        _paste_frame(
            canvas, draw, pixels, (8, y + section_header, image_width, image_height)
        )
    return np.asarray(canvas).copy()


def compose_camera_views(
    views, width, height_limit, simulation_time, mode="rgbd", *, oracle=False
):
    """Compose cached delivered views into a normal top-down RGB image.

    ``views`` follows ``sensor.camera_views()``: each dict contains ``name``,
    ``rgb``, ``overlay``, ``depth``, ``frame_timestamp`` and ``diagnostics``.
    Optional ``depth_range_m=(near, far)`` sets its fixed depth color scale.
    Task-triggered views can set ``role='local refinement'``, ``active`` and
    ``target_track_id``. No target displays IDLE; inactive capture with a target
    displays PAUSED; an active target without a delivered frame displays
    WAITING. Idle/paused images retain their capture time without STALE alarms.
    Optional ``frame_target_track_id`` identifies the target in a retained old
    image when a new target has been assigned but its frame has not arrived.
    ``control_mode='observe'/'refine'`` labels observation-only versus
    experimental control; it never changes perception, targets, or control.
    The first view is primary. RGB-D mode shows every RGB overlay plus only
    primary depth; RGB mode shows every RGB overlay. Capture times and stale
    flags belong to each individual view, never a synthetic shared timestamp.
    This pure compositor does not capture/render, access a sensor, or mutate
    supplied images. ``off`` returns None.
    """
    if mode not in ("rgb", "rgbd", "off"):
        raise ValueError("camera view must be rgb, rgbd or off")
    if mode == "off":
        return None
    if width < 1 or height_limit < 1:
        raise ValueError("camera panel dimensions must be positive")
    views = list(views)
    if len(views) <= 1:
        return _compose_single(
            views[0] if views else {},
            width,
            height_limit,
            simulation_time,
            mode,
            oracle,
        )

    Image, ImageDraw = _pillow()
    tiles = [(view, "RGB", _rgb(view)) for view in views]
    if mode == "rgbd":
        tiles.append((views[0], "DEPTH", _depth_image(views[0])))
    # Two columns keep 2/3 camera rigs compact instead of stacking six images.
    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    padding, gap, header = 8, 8, 28
    tile_header = 66 if any(_local_view(view) for view in views) else 34
    footer = 20 if mode == "rgbd" else 0
    tile_width = max(1, (width - 2 * padding - gap) // columns)
    preferred_height = (
        max(
            round(tile_width * (pixels.shape[0] / pixels.shape[1]))
            for _, _, pixels in tiles
            if pixels is not None
        )
        if any(pixels is not None for _, _, pixels in tiles)
        else round(tile_width * 0.75)
    )
    image_height = max(
        1,
        min(
            preferred_height,
            (height_limit - header - footer - padding - (rows - 1) * gap) // rows
            - tile_header,
        ),
    )
    row_height = tile_header + image_height
    height = min(
        height_limit, header + rows * row_height + (rows - 1) * gap + padding + footer
    )
    canvas = Image.new("RGB", (width, height), (18, 24, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (padding, 7), f"{len(views)} CAMERAS | DELIVERED FRAMES", fill=(235, 240, 245)
    )
    for index, (view, kind, pixels) in enumerate(tiles):
        x = padding + (index % columns) * (tile_width + gap)
        y = header + (index // columns) * (row_height + gap)
        # Clip each tile independently so labels on narrow windows cannot run
        # into the neighboring camera and appear to label the wrong image.
        tile = Image.new("RGB", (tile_width, row_height), (18, 24, 32))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text(
            (0, 0),
            _camera_label(tile_draw, view.get("name", "camera"), kind, tile_width),
            fill=(225, 230, 240),
        )
        timestamp = _timestamp(view)
        stale = _stale(view, simulation_time)
        status = (
            "Waiting for delayed frame"
            if timestamp is None
            else f"t={timestamp:.2f}s" + (" | STALE" if stale else "")
        )
        if _local_view(view):
            state, capture = _local_status(view, simulation_time)
            for text_y, line in (
                (16, _local_caption(view)),
                (32, state),
                (48, capture),
            ):
                tile_draw.text(
                    (0, text_y),
                    _fit_text(tile_draw, line, tile_width),
                    fill=(245, 190, 90) if text_y == 48 and stale else (190, 205, 220),
                )
        else:
            tile_draw.text(
                (0, 16),
                _fit_text(tile_draw, status, tile_width),
                fill=(245, 190, 90) if stale else (190, 205, 220),
            )
        _paste_frame(
            tile, tile_draw, pixels, (0, tile_header, tile_width, image_height)
        )
        canvas.paste(tile, (x, y))
    if mode == "rgbd":
        draw.text(
            (padding, height - 18),
            "Depth: " + _depth_legend(views[0]),
            fill=(190, 205, 220),
        )
    return np.asarray(canvas).copy()


class CameraViewGroup:
    """Display-only combination of independent primary and auxiliary sensors.

    Camera buffers are concatenated without capture, fusion, or tracker access.
    Legacy image/diagnostic access still refers exclusively to the primary.
    Control and evaluation should keep using the original primary sensor.
    """

    def __init__(self, primary, auxiliary):
        self.primary = primary
        self.auxiliary = auxiliary

    def camera_views(self):
        return _delivered_views(self.primary) + _delivered_views(self.auxiliary)

    def diagnostics(self):
        return {} if self.primary is None else self.primary.diagnostics()

    @property
    def overlay(self):
        return getattr(self.primary, "overlay", None)

    @property
    def rgb(self):
        return getattr(self.primary, "rgb", None)

    @property
    def depth(self):
        return getattr(self.primary, "depth", None)


class CameraPanel:
    """Right-hand viewer inset, uploaded only for new frames/layout/status.

    Only public, delivered sensor buffers and diagnostics are read. No model,
    object catalogue, immediate capture, or renderer is used here. ``set_images``
    handles the OpenGL vertical flip, so our images retain normal top-down rows.
    """

    def __init__(self, mode="rgbd"):
        if mode not in ("rgb", "rgbd", "off"):
            raise ValueError("camera view must be rgb, rgbd or off")
        self.mode = mode
        self._key = None
        self._image_key = None
        self._image = None
        self.uploads = 0

    def _compose_delivered(self, views, simulation_time, width, height_limit, oracle):
        key = (
            self.mode,
            width,
            height_limit,
            oracle,
            _views_key(views, simulation_time),
        )
        if key != self._image_key:
            self._image = compose_camera_views(
                views, width, height_limit, simulation_time, self.mode, oracle=oracle
            )
            self._image_key = key
        return self._image

    def compose(self, sensor, simulation_time, width, height_limit):
        """Reusable cached compositor for recordings; ``off`` returns None.

        The returned array belongs to the panel cache and should not be mutated.
        No GUI or MuJoCo imports are needed for composition.
        """
        if self.mode == "off":
            return None
        return self._compose_delivered(
            _delivered_views(sensor),
            simulation_time,
            width,
            height_limit,
            sensor is None,
        )

    def update(self, viewer, sensor, simulation_time):
        if self.mode == "off":
            return
        if not callable(getattr(viewer, "set_images", None)):
            raise RuntimeError(
                "This MuJoCo viewer lacks image overlays. Use the tested MuJoCo "
                "3.17 or newer, or pass --camera-view off."
            )
        viewport = viewer.viewport
        if viewport is None or viewport.width < 160 or viewport.height < 120:
            # A minimized/tiny viewport cannot show a legible panel.
            if self._key is not None:
                viewer.clear_images()
                self._key = None
            return
        views = _delivered_views(sensor)
        key = (
            self.mode,
            viewport.left,
            viewport.bottom,
            viewport.width,
            viewport.height,
            _views_key(views, simulation_time),
            sensor is None,
        )
        if key == self._key:
            return
        # Keep the inset inside the scene viewport, including when UI sidebars
        # move its origin. Multi-view grids cap the inset at 36%/480px, with a
        # 140px minimum on narrow windows. Single-view retains its 38% layout.
        share = 0.36 if len(views) > 1 else 0.38
        width = min(480, max(140, int(viewport.width * share)))
        image = self._compose_delivered(
            views, simulation_time, width, viewport.height - 16, sensor is None
        )
        import mujoco

        rect = mujoco.MjrRect(
            viewport.left + viewport.width - image.shape[1] - 8,
            viewport.bottom + viewport.height - image.shape[0] - 8,
            image.shape[1],
            image.shape[0],
        )
        viewer.set_images((rect, image))
        self.uploads += 1
        self._key = key
