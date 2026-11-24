"""Task-conditioned eye-in-hand RGB-D measurements for existing global targets.

Wrist cameras never create identities or update the global tracker. Each frame
freezes the camera calibration and assigned target at capture time; delayed
processing uses those snapshots, not the moving hand's delivery-time pose.
Only rendered RGB-D, calibrated camera poses and supplied visual Tracks enter
this module. No object IDs, body poses, spawn schedules or catalogue are read.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math

import mujoco
import numpy as np

from .vision import CameraCalibration, RGBDDetector, Track, _rectangle_at_yaw


WRIST_MAX_AGE = 0.30
WRIST_MIN_VIEW_CLEARANCE = 0.075


def _frozen_array(value):
    array = np.asarray(value, dtype=float).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class WristRefinement:
    arm: int
    track_id: int
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    confidence: float
    uncertainty: float
    correction_xy: np.ndarray
    source_target_position: np.ndarray
    observed_size: np.ndarray
    valid_until: float

    def predict(self, now):
        """Advance XY with the frozen global velocity, within validity only."""
        elapsed = max(0.0, min(float(now), self.valid_until) - self.timestamp)
        position = self.position.copy()
        position[:2] += self.velocity[:2] * elapsed
        return position


@dataclass(frozen=True)
class _TargetSnapshot:
    track: Track
    position: np.ndarray
    generation: int


@dataclass(frozen=True)
class _WristFrame:
    arm: int
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    calibration: CameraCalibration
    target: _TargetSnapshot


class WristSensorPipeline:
    """Low-rate local measurement with strict target identity and quality gates.

    Call set_targets with two Track-or-None entries before each update. Targets
    should only be active during approach/descent. ``observe`` executes the
    same sensing and diagnostics but refinement() deliberately returns None.
    ``off`` allocates no renderer and needs no wrist cameras in the model.
    """

    camera_names = ("r0/wrist_rgbd", "r1/wrist_rgbd")
    max_age = WRIST_MAX_AGE
    min_view_clearance = WRIST_MIN_VIEW_CLEARANCE

    def __init__(self, model, config):
        self.model = model
        self.mode = getattr(config, "wrist_camera_mode", "off")
        if self.mode not in ("off", "observe", "refine"):
            raise ValueError("wrist_camera_mode must be off, observe or refine")
        self.width = int(getattr(config, "wrist_camera_width", 320))
        self.height = int(getattr(config, "wrist_camera_height", 240))
        self.period = 1.0 / float(getattr(config, "wrist_camera_hz", 6.0))
        self.latency = float(getattr(config, "camera_latency", 0.08))
        self.depth_noise = float(getattr(config, "depth_noise", 0.001))
        self.dropout = float(getattr(config, "camera_dropout", 0.02))
        self.max_correction = float(getattr(config, "wrist_max_correction", 0.008))
        self.visual_quality = getattr(config, "visual_quality", "balanced")
        self.belt_top = float(config.belt_top)
        self.camera_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in self.camera_names
        )
        if self.mode != "off" and min(self.camera_ids) < 0:
            raise ValueError(
                "Eye-in-hand sensing requires r0/wrist_rgbd and r1/wrist_rgbd"
            )
        self.detector = RGBDDetector(config)
        self._queue_capacity = 2 * max(3, math.ceil(self.latency / self.period) + 3)
        self._renderer = None
        # Sites are simulator debug markers, including the green TCP sphere.
        # Hide only those markers; all physical robot/object geoms remain visible.
        self._render_option = mujoco.MjvOption()
        self._render_option.sitegroup[:] = 0
        self.reset(int(getattr(config, "seed", 0)))

    def reset(self, seed=None):
        self._rngs = [
            np.random.default_rng(
                np.random.SeedSequence(seed, spawn_key=(0x57524953, arm))
            )
            for arm in range(2)
        ]
        self._targets = [None, None]
        self._generation = [0, 0]
        self._next_capture = [0.0, 0.0]
        self._queue = deque(maxlen=self._queue_capacity)
        self._refinements = [None, None]
        self._now = 0.0
        self._counts = {
            "captured_frames": 0,
            "dropped_frames": 0,
            "delivered_frames": 0,
            "accepted_refinements": 0,
            "rejected_frames": 0,
            "queue_overflows": 0,
        }
        self._views = [
            {
                "rgb": None,
                "depth": None,
                "overlay": None,
                "frame_timestamp": None,
                "calibration": None,
                "frame_target_track_id": None,
                "status": "inactive",
                "counts": dict(self._counts),
                "rejection_reasons": {},
                "detections": 0,
                "accepted_timestamp": None,
            }
            for _ in range(2)
        ]

    def set_targets(self, targets):
        if len(targets) != 2:
            raise ValueError("Supply exactly two wrist targets")
        for arm, target in enumerate(targets):
            if target is not None and not isinstance(target, Track):
                raise TypeError(
                    "A wrist target must be an existing visual Track or None"
                )
            old = self._targets[arm]
            old_id = None if old is None else old.track_id
            new_id = None if target is None else target.track_id
            if old_id != new_id:
                self._generation[arm] += 1
                self._refinements[arm] = None
                self._views[arm]["status"] = "inactive" if target is None else "waiting"
                self._views[arm]["accepted_timestamp"] = None
                if target is not None:
                    self._views[arm].update(
                        rgb=None,
                        depth=None,
                        overlay=None,
                        calibration=None,
                        frame_timestamp=None,
                        frame_target_track_id=None,
                        detections=0,
                    )
            self._targets[arm] = target

    def _increment(self, arm, key):
        self._counts[key] += 1
        self._views[arm]["counts"][key] += 1

    def _reject(self, arm, reason):
        self._increment(arm, "rejected_frames")
        view = self._views[arm]
        view["rejection_reasons"][reason] = view["rejection_reasons"].get(reason, 0) + 1
        view["status"] = reason
        # An obscured/low-angle frame is an absence of new information. Keep
        # the last sound observation only until its ORIGINAL deadline. Identity
        # changes, expired targets and contradictory observations invalidate it.
        if reason in {
            "changed_target",
            "stale_target",
            "stale_frame",
            "excessive_correction",
            "ambiguous_detection",
        }:
            self._refinements[arm] = None

    def capture(self, arm, data, now=None):
        """Capture only an assigned target; hold calibration/target across latency."""
        if self.mode == "off" or self._targets[arm] is None:
            return False
        timestamp = float(data.time if now is None else now)
        target = self._targets[arm]
        if timestamp - target.last_seen > target.max_prediction or target.hits < 2:
            self._reject(arm, "stale_target")
            return False
        self._increment(arm, "captured_frames")
        rng = self._rngs[arm]
        if rng.random() < self.dropout:
            self._increment(arm, "dropped_frames")
            self._views[arm]["status"] = "dropout"
            return False
        calibration = CameraCalibration.from_camera(
            self.model, data, self.camera_ids[arm], self.width, self.height
        )
        track = replace(
            target,
            position=_frozen_array(target.position),
            velocity=_frozen_array(target.velocity),
            size=_frozen_array(target.size),
        )
        snapshot = _TargetSnapshot(
            track, _frozen_array(target.predict(timestamp)), self._generation[arm]
        )
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(
            data, camera=self.camera_names[arm], scene_option=self._render_option
        )
        if self.visual_quality == "balanced":
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        if self.depth_noise:
            depth += rng.normal(0.0, self.depth_noise, depth.shape).astype(depth.dtype)
        if len(self._queue) == self._queue.maxlen:
            self._increment(self._queue[0].arm, "queue_overflows")
        self._queue.append(
            _WristFrame(arm, timestamp, rgb, depth, calibration, snapshot)
        )
        return True

    def _measure(self, frame, detections):
        """Select one complete observation and retain the global grasp direction."""
        target = frame.target.track
        expected = frame.target.position
        top_height = expected[2] + target.half_height
        if frame.calibration.position[2] - top_height + 1e-9 < self.min_view_clearance:
            return None, "low_view_clearance"
        if frame.calibration.rotation[2, 2] < 0.25:
            return None, "grazing_view"
        nearby = [
            detection
            for detection in detections
            if detection.kind == target.kind
            and np.linalg.norm(detection.position[:2] - expected[:2]) <= 0.035
            and abs(detection.position[2] - expected[2]) <= 0.015
        ]
        if not nearby:
            return None, "no_target_detection"
        # Multiple components close to one expected target are ambiguous even
        # when one fragment happens to have a slightly closer centre.
        if len(nearby) != 1:
            return None, "ambiguous_detection"
        detection = nearby[0]
        if detection.occluded:
            return None, "occluded_foreground"
        left, top, right, bottom = detection.bbox
        if min(left, top, self.width - 1 - right, self.height - 1 - bottom) < 3:
            return None, "image_boundary"
        if detection.confidence < 0.8 or len(detection.footprint_points) < 24:
            return None, "insufficient_footprint"
        try:
            centre, dimensions = _rectangle_at_yaw(
                np.asarray(detection.footprint_points),
                target.yaw,
                detection.footprint_pixel_size,
                detection.footprint_pixel_axes,
            )
        except (ValueError, FloatingPointError):
            return None, "invalid_footprint"
        observed = np.r_[dimensions, detection.size[2]]
        if not np.isfinite(observed).all() or not np.isfinite(centre).all():
            return None, "invalid_footprint"
        planar_tolerance = np.maximum(0.006, 0.18 * target.size[:2])
        area_ratio = float(np.prod(dimensions) / max(np.prod(target.size[:2]), 1e-12))
        if (
            area_ratio < 0.80
            or area_ratio > 1.35
            or np.any(np.abs(dimensions - target.size[:2]) > planar_tolerance)
        ):
            return None, "incomplete_or_inconsistent_footprint"
        if abs(observed[2] - target.size[2]) > max(0.006, 3 * self.depth_noise):
            return None, "inconsistent_height"
        correction = centre - expected[:2]
        if np.linalg.norm(correction) > self.max_correction:
            return None, "excessive_correction"
        position = expected.copy()
        position[:2] = centre
        valid_until = min(
            frame.timestamp + self.max_age, target.last_seen + target.max_prediction
        )
        return WristRefinement(
            frame.arm,
            target.track_id,
            frame.timestamp,
            _frozen_array(position),
            _frozen_array(target.velocity),
            min(detection.confidence, target.confidence),
            detection.uncertainty,
            _frozen_array(correction),
            _frozen_array(expected),
            _frozen_array(observed),
            valid_until,
        ), "accepted"

    def update(self, data, now=None):
        now = float(data.time if now is None else now)
        if now < self._now - 1e-9:
            raise ValueError("Wrist sensor time went backwards; call reset")
        self._now = now
        if self.mode == "off":
            return
        for arm in range(2):
            if self._targets[arm] is not None and now + 1e-9 >= self._next_capture[arm]:
                self.capture(arm, data, now)
                self._next_capture[arm] = now + self.period
        while self._queue and self._queue[0].timestamp + self.latency <= now + 1e-9:
            frame = self._queue.popleft()
            arm = frame.arm
            self._increment(arm, "delivered_frames")
            target = self._targets[arm]
            if (
                target is None
                or target.track_id != frame.target.track.track_id
                or self._generation[arm] != frame.target.generation
            ):
                self._reject(arm, "changed_target")
                continue
            detections = self.detector.detect(frame.rgb, frame.depth, frame.calibration)
            view = self._views[arm]
            view.update(
                rgb=frame.rgb,
                depth=frame.depth,
                frame_timestamp=frame.timestamp,
                calibration=frame.calibration,
                frame_target_track_id=frame.target.track.track_id,
                detections=len(detections),
            )
            measurement = None
            if now - frame.timestamp > self.max_age:
                reason = "stale_frame"
            else:
                measurement, reason = self._measure(frame, detections)
            if measurement is None:
                self._reject(arm, reason)
            else:
                self._increment(arm, "accepted_refinements")
                self._refinements[arm] = measurement
                view["status"] = "accepted"
                view["accepted_timestamp"] = frame.timestamp
            view["overlay"] = self._make_overlay(frame, detections, measurement)

    def refinement(self, arm, track_id, now):
        """Return only a fresh local observation of the currently assigned ID."""
        if self.mode != "refine":
            return None
        target = self._targets[arm]
        measurement = self._refinements[arm]
        if (
            target is None
            or target.track_id != track_id
            or measurement is None
            or measurement.track_id != track_id
            or now < measurement.timestamp
            or now > measurement.valid_until + 1e-9
        ):
            return None
        return measurement

    def _make_overlay(self, frame, detections, measurement):
        overlay = frame.rgb.copy()
        for detection in detections:
            left, top, right, bottom = detection.bbox
            colour = (255, 140, 0) if detection.occluded else (255, 210, 30)
            overlay[top : bottom + 1, (left, right)] = colour
            overlay[(top, bottom), left : right + 1] = colour
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return overlay
        image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(image)
        point = (
            frame.target.position.copy()
            if measurement is None
            else measurement.position.copy()
        )
        point[2] += frame.target.track.half_height
        camera_point = (point - frame.calibration.position) @ frame.calibration.rotation
        if camera_point[2] < 0:
            u, v = frame.calibration.project(point)
            if 0 <= u < self.width and 0 <= v < self.height:
                colour = (40, 255, 80) if measurement is not None else (255, 180, 20)
                draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline=colour)
                draw.text(
                    (u + 5, v - 12),
                    f"Target T{frame.target.track.track_id}",
                    fill=colour,
                )
        return np.asarray(image).copy()

    def camera_views(self):
        views = []
        for arm, name in enumerate(self.camera_names):
            view = self._views[arm]
            target = self._targets[arm]
            calibration = view["calibration"]
            metadata = None
            depth_range = (0.03, 0.40)
            if calibration is not None:
                metadata = {
                    "name": name,
                    "position": calibration.position.tolist(),
                    "rotation": calibration.rotation.tolist(),
                    "focal": calibration.focal,
                    "width": calibration.width,
                    "height": calibration.height,
                }
            timestamp = view["frame_timestamp"]
            diagnostics = dict(view["counts"])
            diagnostics.update(
                detections=view["detections"],
                delivered_frame_time=timestamp,
                frame_age_seconds=None if timestamp is None else self._now - timestamp,
                accepted_timestamp=view["accepted_timestamp"],
                rejection_reasons=dict(view["rejection_reasons"]),
                status=view["status"],
            )
            views.append(
                {
                    "name": name,
                    "rgb": view["rgb"],
                    "depth": view["depth"],
                    "overlay": view["overlay"],
                    "frame_timestamp": timestamp,
                    "calibration_metadata": metadata,
                    "depth_range_m": depth_range,
                    "diagnostics": diagnostics,
                    "role": "local refinement",
                    "control_mode": self.mode,
                    "active": self.mode != "off" and target is not None,
                    "target_track_id": None if target is None else target.track_id,
                    "frame_target_track_id": view["frame_target_track_id"],
                    "status": view["status"],
                }
            )
        return views

    def diagnostics(self):
        result = dict(self._counts)
        result.update(
            mode=self.mode,
            max_age_seconds=self.max_age,
            max_correction_m=self.max_correction,
            min_view_clearance_m=self.min_view_clearance,
            queued_frames=len(self._queue),
            queue_capacity=self._queue_capacity,
            active_targets=[
                None if target is None else target.track_id for target in self._targets
            ],
            per_camera={
                view["name"]: view["diagnostics"] for view in self.camera_views()
            },
        )
        return result

    def close(self):
        if self._renderer is not None:
            # Free this renderer's resources in its own GL context, even when
            # the overhead or another environment's camera was used last.
            context = getattr(self._renderer, "_gl_context", None)
            if context is not None:
                context.make_current()
            self._renderer.close()
            self._renderer = None
        self._queue.clear()
        self._targets = [None, None]
        self._refinements = [None, None]
        for view in self._views:
            view.update(
                rgb=None,
                depth=None,
                overlay=None,
                calibration=None,
                frame_timestamp=None,
                frame_target_track_id=None,
                accepted_timestamp=None,
                detections=0,
                status="closed",
            )
