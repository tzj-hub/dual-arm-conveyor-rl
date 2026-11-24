"""RGB-D perception and timestamped tracking without simulator object state.

The only MuJoCo data consumed here are rendered RGB/depth and the calibrated
camera pose. Detection uses colour and measured top-face geometry, not instance
segmentation, object names, body poses, spawn schedules, or object identities.
Track IDs are assigned by image association and are unrelated to simulator IDs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import math

import mujoco
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError

from .rgbd import render_rgbd


def square_angle(angle: float) -> float:
    """Canonical square yaw in [-pi/4, pi/4), with pi/2 symmetry."""
    return float((angle + math.pi / 4) % (math.pi / 2) - math.pi / 4)


def symmetric_angle(angle: float, period: float) -> float:
    """Canonical orientation of a footprint, with zero for circular footprints."""
    return 0.0 if period == 0 else float((angle + period / 2) % period - period / 2)


def _default_size():
    # Backwards compatibility for manually supplied oracle/test observations.
    # The visual detector always replaces this with measured RGB-D dimensions.
    return np.array([0.05, 0.05, 0.05])


def _pixel_padding(yaw, pixel_size, pixel_axes=None):
    cs, sn = math.cos(yaw), math.sin(yaw)
    if pixel_axes is None or len(pixel_axes) == 0:
        return pixel_size * (abs(cs) + abs(sn))
    axes = np.array([[cs, -sn], [sn, cs]])
    return np.abs(np.asarray(pixel_axes) @ axes).sum(axis=0)


def _rectangle_at_yaw(points, yaw, pixel_size=0.0, pixel_axes=None):
    reference = points.mean(axis=0)
    cs, sn = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cs, -sn], [sn, cs]])
    aligned = (points - reference) @ rotation
    low, high = aligned.min(axis=0), aligned.max(axis=0)
    centre = reference + ((low + high) / 2) @ rotation.T
    size = high - low + _pixel_padding(yaw, pixel_size, pixel_axes)
    return centre, size


def fit_prism_outline(xy, pixel_size=0.0, reference_yaw=None, pixel_axes=None):
    """Fit a grasp rectangle to a visible convex, flat-topped footprint.

    Only planar points are used. The minimum Feret width chooses parallel jaw
    faces; nearby alternatives let a tracker avoid switching between equivalent
    polygon edges as the pixel grid changes. ``pixel_size`` supplies the missing
    boundary between measured pixel centres, and is zero for exact geometry.
    Returns centre, yaw, full rectangle dimensions, and candidates encoded as
    ``(yaw, centre_x, centre_y, length, width)``. No polygon side count or object
    catalogue is required.
    """
    points = np.asarray(xy, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 3
        or not np.isfinite(points).all()
        or not np.isfinite(pixel_size)
        or pixel_size < 0
    ):
        raise ValueError(
            "A footprint needs finite 2D points and nonnegative pixel size"
        )
    reference = points.mean(axis=0)
    local = points - reference
    try:
        hull = local[ConvexHull(local).vertices]
    except QhullError as exc:
        raise ValueError("Footprint points must span a plane") from exc
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    choices = []
    for angle in angles:
        cs, sn = math.cos(angle), math.sin(angle)
        rotation = np.array([[cs, -sn], [sn, cs]])
        aligned = local @ rotation
        low, high = aligned.min(axis=0), aligned.max(axis=0)
        centre = reference + ((low + high) / 2) @ rotation.T
        padding = _pixel_padding(angle, pixel_size, pixel_axes)
        size = high - low + padding
        angle = symmetric_angle(float(angle), math.pi)
        candidate = (
            angle,
            float(centre[0]),
            float(centre[1]),
            float(size[0]),
            float(size[1]),
        )
        if not any(
            abs(symmetric_angle(angle - previous[0], math.pi)) < 1e-4
            for previous in choices
        ):
            choices.append(candidate)
    narrowest = min(value[4] for value in choices)
    tolerance = max(1e-8, 0.35 * pixel_size)
    choices = [value for value in choices if value[4] <= narrowest + tolerance]
    reference_angle = 0.0 if reference_yaw is None else float(reference_yaw)
    choices.sort(
        key=lambda value: (
            abs(symmetric_angle(value[0] - reference_angle, math.pi)),
            value[3] * value[4],
            value[0],
        )
    )
    chosen = choices[0]
    return np.asarray(chosen[1:3]), chosen[0], np.asarray(chosen[3:5]), tuple(choices)


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    focal: float
    position: np.ndarray
    rotation: np.ndarray

    @classmethod
    def from_camera(cls, model, data, camera_id, width, height):
        focal = (
            0.5 * height / math.tan(math.radians(float(model.cam_fovy[camera_id])) / 2)
        )
        return cls(
            width,
            height,
            focal,
            data.cam_xpos[camera_id].copy(),
            data.cam_xmat[camera_id].reshape(3, 3).copy(),
        )

    def unproject(self, rows, columns, depth):
        """MuJoCo depth is optical-axis distance, not Euclidean ray length."""
        depth = np.asarray(depth)
        camera_points = np.stack(
            (
                (np.asarray(columns) - (self.width - 1) / 2) * depth / self.focal,
                -(np.asarray(rows) - (self.height - 1) / 2) * depth / self.focal,
                -depth,
            ),
            axis=-1,
        )
        return camera_points @ self.rotation.T + self.position

    def project(self, points):
        points = (np.asarray(points) - self.position) @ self.rotation
        distance = -points[..., 2]
        return np.stack(
            (
                (self.width - 1) / 2 + self.focal * points[..., 0] / distance,
                (self.height - 1) / 2 - self.focal * points[..., 1] / distance,
            ),
            axis=-1,
        )

    def on_horizontal_plane(self, rows, columns, height):
        """Intersect measured pixel rays with a horizontal surface plane."""
        rays = self.unproject(rows, columns, np.ones_like(rows, dtype=float))
        rays -= self.position
        distance = (height - self.position[2]) / rays[..., 2]
        return self.position + rays * distance[..., None]

    def plane_pixel_axes(self, row, column, height):
        points = self.on_horizontal_plane(
            np.array([row, row, row + 1]),
            np.array([column, column + 1, column]),
            height,
        )
        return points[1:, :2] - points[0, :2]


@dataclass(frozen=True)
class Detection:
    position: np.ndarray
    yaw: float
    kind: int
    confidence: float
    # Image bounding box is (left, top, right, bottom), in inclusive pixels.
    bbox: tuple[int, int, int, int]
    size: np.ndarray = field(default_factory=_default_size)
    shape: str = "box"
    yaw_period: float = math.pi / 2
    uncertainty: float = 0.004
    footprint_candidates: tuple[tuple[float, float, float, float, float], ...] = ()
    footprint_points: tuple[tuple[float, float], ...] = ()
    footprint_pixel_size: float = 0.0
    footprint_pixel_axes: tuple[tuple[float, float], ...] = ()
    source_cameras: tuple[str, ...] = ()
    occluded: bool = False


@dataclass(frozen=True)
class Track:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    yaw: float
    kind: int
    confidence: float
    timestamp: float
    last_seen: float
    hits: int = 1
    visible: bool = True
    max_prediction: float = 1.5
    size: np.ndarray = field(default_factory=_default_size)
    shape: str = "box"
    yaw_period: float = math.pi / 2
    uncertainty: float = 0.004
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def half_height(self) -> float:
        return float(self.size[2] / 2)

    @property
    def grasp_width(self) -> float:
        """Short footprint dimension, perpendicular to a non-square object's yaw."""
        return float(self.size[1])

    def predict(self, now: float) -> np.ndarray:
        """Predict from the filter timestamp, bounded by the occlusion budget."""
        until = min(
            max(float(now), self.timestamp), self.last_seen + self.max_prediction
        )
        dt = max(0.0, until - self.timestamp)
        # A locally estimated acceleration is useful across camera latency, but
        # extending it through a long occlusion invents future belt commands.
        horizon = min(dt, 0.25)
        return (
            self.position
            + self.velocity * dt
            + self.acceleration * (horizon * dt - 0.5 * horizon**2)
        )

    def predict_velocity(self, now: float) -> np.ndarray:
        until = min(
            max(float(now), self.timestamp), self.last_seen + self.max_prediction
        )
        return self.velocity + self.acceleration * min(0.25, until - self.timestamp)


class RGBDDetector:
    """Measure coloured objects above the belt without a size/shape catalogue.

    Height is measured relative to the calibrated belt plane. Planar top faces
    use rectangle/circle moments; curved surfaces use an elliptical silhouette.
    At low resolution near-circular and nearly square shapes are ambiguous, so
    the geometric shape label is an estimate and carries limited confidence.
    Dimensions are full lengths: long-axis, short-axis, observed upright height.
    """

    def __init__(self, config):
        self.belt_top = float(config.belt_top)
        self.half_length = float(config.belt_half_length)
        self.half_width = float(config.belt_half_width)
        self.depth_noise = float(getattr(config, "depth_noise", 0.001))
        self.flat_prisms = getattr(config, "object_family", None) == "flat_prisms"
        # Adaptive tracking uses the same image rejection with every camera rig.
        # The baseline remains reproducible for before/after measurements.
        self.occlusion_guard = (
            getattr(config, "strategy", "baseline") == "adaptive"
            or getattr(config, "wrist_camera_mode", "off") != "off"
        )
        self._pixel_grid_shape = None
        self._pixel_offsets = None

    def _unproject_image(self, depth, calibration):
        """Reuse pixel coordinates while applying the current camera calibration.

        Keep the multiply/divide order of ``CameraCalibration.unproject`` so
        fitting and threshold decisions receive the same floating-point values.
        Only image coordinates are cached; wrist motion and changing intrinsics
        always use the supplied calibration.
        """
        if self._pixel_grid_shape != depth.shape:
            rows, columns = np.indices(depth.shape)
            self._pixel_offsets = (
                columns - (calibration.width - 1) / 2,
                -(rows - (calibration.height - 1) / 2),
            )
            self._pixel_grid_shape = depth.shape
        columns, rows = self._pixel_offsets
        points = np.empty((*depth.shape, 3), dtype=float)
        points[..., 0] = columns * depth / calibration.focal
        points[..., 1] = rows * depth / calibration.focal
        points[..., 2] = -depth
        if np.array_equal(calibration.rotation, np.eye(3)):
            # The overhead camera needs only translation. Invalid depths still
            # produce the all-NaN points that the general matrix product yields.
            points[~np.isfinite(depth)] = np.nan
            points += calibration.position
            return points
        return points @ calibration.rotation.T + calibration.position

    def _foreground_occlusion(
        self, labels, label, rows, columns, depth, points, top_height, calibration
    ):
        """Flag a contour bordered by a connected, measurably nearer surface.

        Only rendered depth and calibration are used. A two-pixel exterior
        ring includes holes cut into a coloured component by an occluder. Three
        connected foreground pixels avoid treating isolated depth noise as a
        robot edge. Heights are compared in world coordinates, including for
        tilted wrist views; no geometry names or instance segmentation enter.
        """
        y = slice(max(0, int(rows.min()) - 2), min(depth.shape[0], int(rows.max()) + 3))
        x = slice(
            max(0, int(columns.min()) - 2), min(depth.shape[1], int(columns.max()) + 3)
        )
        component = labels[y, x] == label
        ring = ndimage.binary_dilation(
            component, structure=np.ones((3, 3)), iterations=2
        )
        ring &= ~component
        local_depth = depth[y, x]
        heights = points[y, x, 2]
        foreground = (
            ring
            & np.isfinite(local_depth)
            & (local_depth > 0)
            & np.isfinite(heights)
            & (heights > top_height + max(0.005, 4 * self.depth_noise))
            # A downward ray reaches this higher surface before the measured
            # horizontal object plane. Upward rays cannot occlude that plane.
            & (heights < calibration.position[2])
        )
        if np.count_nonzero(foreground) < 3:
            return False
        components, count = ndimage.label(foreground, structure=np.ones((3, 3)))
        return bool(count and np.max(np.bincount(components.ravel())[1:]) >= 3)

    def _fit_curved_surface(self, cloud, centre, size, yaw):
        """Fit the visible ellipsoid surface to remove perspective centroid bias.

        The support plane supplies the bottom contact height. Bounds are local
        to the image estimate, not a list of known simulator object parameters.
        """
        initial = np.r_[centre[:2], np.asarray(size) / 2, yaw]
        lower = np.r_[initial[:2] - 0.015, initial[2:5] * 0.65, yaw - 0.45]
        upper = np.r_[initial[:2] + 0.015, initial[2:5] * 1.45, yaw + 0.45]

        def residual(parameters):
            cx, cy, rx, ry, rz, angle = parameters
            cs, sn = math.cos(angle), math.sin(angle)
            xy = (cloud[:, :2] - [cx, cy]) @ np.array([[cs, -sn], [sn, cs]])
            z = cloud[:, 2] - self.belt_top - rz
            return (
                (xy[:, 0] / rx) ** 2 + (xy[:, 1] / ry) ** 2 + (z / rz) ** 2 - 1
            ) * rz

        fit = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=max(self.depth_noise, 0.0007),
            max_nfev=30,
        )
        if not fit.success or np.sqrt(np.mean(residual(fit.x) ** 2)) > 0.003:
            return centre, np.asarray(size), yaw
        cx, cy, rx, ry, rz, angle = fit.x
        if rx < ry:
            rx, ry, angle = ry, rx, angle + math.pi / 2
        return (
            np.array([cx, cy, self.belt_top + rz]),
            np.array([2 * rx, 2 * ry, 2 * rz]),
            symmetric_angle(angle, math.pi),
        )

    def detect(self, rgb, depth, calibration: CameraCalibration) -> list[Detection]:
        rgb, depth = np.asarray(rgb), np.asarray(depth)
        if (
            rgb.shape != (calibration.height, calibration.width, 3)
            or depth.shape != rgb.shape[:2]
        ):
            raise ValueError("RGB-D shape does not match camera calibration")
        points = self._unproject_image(depth, calibration)
        tolerance = max(0.0035, 3 * self.depth_noise)
        valid = (
            np.isfinite(depth)
            & (depth > 0)
            & (np.abs(points[..., 0]) < self.half_length)
            & (np.abs(points[..., 1]) < self.half_width - 0.002)
            # Broad workspace limits reject the belt itself and lifted objects;
            # they do not encode an individual object's height or footprint.
            & (points[..., 2] > self.belt_top + 0.012)
            & (points[..., 2] < self.belt_top + 0.11)
        )
        r, g, b = rgb.astype(np.float32).transpose(2, 0, 1)
        colours = (
            (r > 60) & (r > 1.4 * g) & (r > 1.5 * b),
            (b > 60) & (b > 1.4 * r) & (b > 1.15 * g),
        )
        detections = []
        for kind, colour in enumerate(colours):
            labels, count = ndimage.label(valid & colour, structure=np.ones((3, 3)))
            for label in range(1, count + 1):
                yy, xx = np.nonzero(labels == label)
                if len(xx) < 8:
                    continue
                cloud = points[yy, xx]
                distance = float(np.median(depth[yy, xx]))
                pixel_size = distance / calibration.focal
                top = float(np.quantile(cloud[:, 2], 0.90))
                top_face = cloud[:, 2] > top - tolerance
                planar = float(np.mean(top_face)) > 0.66
                if planar or self.flat_prisms:
                    # Exclude visible vertical sides before estimating the
                    # footprint. Otherwise a perspective view biases its centre.
                    footprint = cloud[top_face]
                    height = float(np.median(footprint[:, 2])) - self.belt_top
                else:
                    footprint = cloud
                    height = float(np.quantile(cloud[:, 2], 0.97)) - self.belt_top
                if len(footprint) < 8 or not 0.022 < height < 0.105:
                    continue
                occluded = self.occlusion_guard and self._foreground_occlusion(
                    labels,
                    label,
                    yy,
                    xx,
                    depth,
                    points,
                    self.belt_top + height,
                    calibration,
                )
                if self.flat_prisms:
                    pixel_axes = ()
                    if abs(calibration.rotation[2, 2]) < 0.999999:
                        # Tilted pixels cover an anisotropic parallelogram on
                        # the top plane. Reproject noisy depth samples onto the
                        # measured plane before fitting its grasp rectangle.
                        top_height = self.belt_top + height
                        footprint = calibration.on_horizontal_plane(
                            yy[top_face], xx[top_face], top_height
                        )
                        pixel_axes = tuple(
                            map(
                                tuple,
                                calibration.plane_pixel_axes(
                                    float(np.median(yy[top_face])),
                                    float(np.median(xx[top_face])),
                                    top_height,
                                ),
                            )
                        )
                        pixel_size = max(np.linalg.norm(pixel_axes, axis=1))
                    try:
                        centre_xy, yaw, dimensions, candidates = fit_prism_outline(
                            footprint[:, :2], pixel_size, pixel_axes=pixel_axes
                        )
                    except ValueError:
                        continue
                    if not (np.min(dimensions) > 0.021 and np.max(dimensions) < 0.095):
                        continue
                    centre = np.r_[centre_xy, self.belt_top + height / 2]
                    size = np.r_[dimensions, height]
                    confidence = 0.62 + 0.30 * min(1.0, len(footprint) / 45.0)
                    detections.append(
                        Detection(
                            centre,
                            yaw,
                            kind,
                            confidence,
                            (
                                int(xx.min()),
                                int(yy.min()),
                                int(xx.max()),
                                int(yy.max()),
                            ),
                            size,
                            "prism",
                            math.pi,
                            max(0.0015, pixel_size / 2, self.depth_noise),
                            candidates,
                            tuple(map(tuple, footprint[:, :2])),
                            pixel_size,
                            pixel_axes,
                            occluded=occluded,
                        )
                    )
                    continue
                centre = np.mean(footprint, axis=0)
                xy = footprint[:, :2] - centre[:2]
                covariance = xy.T @ xy / len(xy)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                aspect = math.sqrt(
                    max(eigenvalues[1], 1e-10) / max(eigenvalues[0], 1e-10)
                )
                z = xy[:, 0] + 1j * xy[:, 1]
                fourth = np.mean(z**4)
                harmonic = abs(fourth) / max(float(np.mean(np.abs(z) ** 4)), 1e-12)
                circular = planar and aspect < 1.22 and harmonic < 0.23
                if circular:
                    shape, period, yaw = "cylinder", 0.0, 0.0
                    diameter = math.sqrt(
                        8 * np.trace(covariance) + 4 * pixel_size**2 / 3
                    )
                    length = width = diameter
                elif planar and aspect < 1.15:
                    shape, period = "box", math.pi / 2
                    yaw = square_angle(float(np.angle(-fourth)) / 4)
                    cs, sn = math.cos(yaw), math.sin(yaw)
                    aligned = xy @ np.array([[cs, -sn], [sn, cs]])
                    dimensions = np.sqrt(
                        12 * np.mean(aligned**2, axis=0) + pixel_size**2
                    )
                    length = width = float(np.mean(dimensions))
                else:
                    shape = "box" if planar else "ellipsoid"
                    period = math.pi
                    yaw = symmetric_angle(
                        math.atan2(eigenvectors[1, 1], eigenvectors[0, 1]), period
                    )
                    if planar and aspect < 1.45:
                        # Near-square rectangles have poorly separated PCA
                        # eigenvalues. Fourth moments locate their edges; PCA
                        # only resolves which of those edges is the long one.
                        edge_yaw = square_angle(float(np.angle(-fourth)) / 4)
                        candidates = (edge_yaw, edge_yaw + math.pi / 2)
                        yaw = symmetric_angle(
                            min(
                                candidates,
                                key=lambda angle: abs(
                                    symmetric_angle(angle - yaw, math.pi)
                                ),
                            ),
                            math.pi,
                        )
                    factor = 12 if planar else 16
                    length, width = np.sqrt(
                        factor * eigenvalues[::-1] + factor * pixel_size**2 / 12
                    )
                    if not planar and aspect < 1.10:
                        # A sphere-like silhouette has no reliable grasp yaw.
                        period, yaw = 0.0, 0.0
                        length = width = float((length + width) / 2)
                if not (0.021 < width <= length < 0.095):
                    continue
                centre[2] = self.belt_top + height / 2
                size = np.array([length, width, height])
                if shape == "ellipsoid":
                    centre, size, yaw = self._fit_curved_surface(
                        cloud, centre, size, yaw
                    )
                    if period == 0:
                        yaw = 0.0
                resolution_quality = min(1.0, len(footprint) / 45.0)
                confidence = 0.62 + 0.30 * resolution_quality
                uncertainty = max(0.0015, pixel_size / 2, self.depth_noise)
                detections.append(
                    Detection(
                        centre,
                        yaw,
                        kind,
                        confidence,
                        (int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())),
                        size,
                        shape,
                        period,
                        uncertainty,
                        occluded=occluded,
                    )
                )
        return sorted(
            detections, key=lambda d: (float(d.position[0]), float(d.position[1]))
        )


class VisualBeltMotion:
    """Estimate a common belt velocity from accepted RGB-D displacements.

    The inputs are timestamped visual measurements, not the speed generator or
    simulator state. A two-state velocity/acceleration Kalman filter accounts
    for the fact that a displacement measures *interval-average* velocity.
    Robust aggregation rejects one independently moving object when at least
    three usable objects are visible. With only one usable object this ambiguity
    is unavoidable and is reported through the contributor count.
    """

    max_speed = 0.6
    acceleration_limit = 0.25
    extrapolation_horizon = 0.25

    def __init__(self, initial_speed=0.1):
        self.initial_speed = float(initial_speed)
        self.reset()

    def reset(self):
        self.state = np.array([self.initial_speed, 0.0])
        self.covariance = np.diag([0.06**2, 0.12**2])
        self.timestamp = None
        self.last_measurement = None
        self.contributors = 0
        self.updates = 0
        self._previous = {}
        self._history = deque(maxlen=512)

    def update(self, observations, timestamp):
        """Observations are ``(visual_id, measured_xyz, measured_size)`` tuples."""
        timestamp = float(timestamp)
        if self.timestamp is not None and timestamp < self.timestamp - 1e-9:
            raise ValueError("Visual belt estimates need increasing timestamps")
        dt = 0.0 if self.timestamp is None else timestamp - self.timestamp
        remaining = self.extrapolation_horizon
        if self.last_measurement is not None:
            remaining = max(
                0.0,
                remaining - max(0.0, self.timestamp - self.last_measurement),
            )
        horizon = min(dt, remaining)
        transition = np.array([[1.0, horizon], [0.0, 1.0]])
        self.state = transition @ self.state
        noise = np.array([0.5 * horizon**2, horizon])
        self.covariance = (
            transition @ self.covariance @ transition.T
            + np.outer(noise, noise) * 0.35**2
        )
        samples = []
        for identity, position, size in observations:
            position, size = np.asarray(position), np.asarray(size)
            previous = self._previous.get(identity)
            if previous is not None:
                old_time, old_position, old_size = previous
                interval = timestamp - old_time
                delta = position - old_position
                if (
                    0.04 <= interval <= 0.25
                    and abs(delta[1]) < 0.008
                    and abs(delta[2]) < 0.005
                    and np.all(
                        np.abs(size - old_size) < np.maximum(0.004, 0.12 * old_size)
                    )
                    and -0.03 <= delta[0] / interval <= self.max_speed + 0.03
                ):
                    # The observed mean speed is timestamped at interval centre.
                    samples.append((float(delta[0] / interval), interval))
            self._previous[identity] = (timestamp, position.copy(), size.copy())
        self._previous = {
            key: value
            for key, value in self._previous.items()
            if timestamp - value[0] <= 0.5
        }
        self.contributors = 0
        if samples:
            values = np.asarray(samples)
            middle = float(np.median(values[:, 0]))
            mad = float(np.median(np.abs(values[:, 0] - middle)))
            inliers = values[np.abs(values[:, 0] - middle) <= max(0.025, 3.0 * mad)]
            observation = float(np.median(inliers[:, 0]))
            interval = float(np.median(inliers[:, 1]))
            measurement = np.array([1.0, -0.5 * interval])
            # Shared contours are correlated; cap the gain from many objects.
            variance = (0.0025 * math.sqrt(2) / interval) ** 2 / min(len(inliers), 4)
            variance += (1.4826 * mad) ** 2
            innovation = float(measurement @ self.covariance @ measurement + variance)
            gain = self.covariance @ measurement / innovation
            self.state += gain * (observation - measurement @ self.state)
            identity = np.eye(2) - np.outer(gain, measurement)
            self.covariance = (
                identity @ self.covariance @ identity.T
                + np.outer(gain, gain) * variance
            )
            self.last_measurement = timestamp
            self.contributors = len(inliers)
            self.updates += 1
        elif (
            self.last_measurement is not None
            and timestamp - self.last_measurement > 0.25
        ):
            # No image evidence of continuing acceleration: hold the last speed.
            self.state[1] = 0.0
        self.state[0] = float(np.clip(self.state[0], 0.0, self.max_speed))
        self.state[1] = float(
            np.clip(self.state[1], -self.acceleration_limit, self.acceleration_limit)
        )
        self.timestamp = timestamp
        entry = (timestamp, float(self.state[0]), float(self.state[1]))
        if self._history and self._history[-1][0] == timestamp:
            self._history[-1] = entry
        else:
            self._history.append(entry)

    def velocity(self, now):
        dt = 0.0 if self.timestamp is None else max(0.0, float(now) - self.timestamp)
        return float(
            np.clip(
                self.state[0] + self.state[1] * min(dt, self.extrapolation_horizon),
                0.0,
                self.max_speed,
            )
        )

    def acceleration(self, now):
        if (
            self.timestamp is None
            or float(now) - self.timestamp > self.extrapolation_horizon
        ):
            return 0.0
        return float(self.state[1])

    def displacement(self, start, end):
        """Integrate measured speed history, bounded extrapolation after last image."""
        start, end = float(start), float(end)
        if end < start:
            return -self.displacement(end, start)
        if not self._history:
            return self.initial_speed * (end - start)
        history = np.asarray(self._history)
        # Piecewise linear interpolation integrates the visual estimate exactly.
        inside = history[(history[:, 0] > start) & (history[:, 0] < end), 0].tolist()
        latest = float(history[-1, 0])
        limit = latest + self.extrapolation_horizon
        if start < limit < end:
            inside.append(limit)
        points = sorted([start, *inside, end])

        def speed(time):
            if time > latest:
                return self.velocity(time)
            return float(np.interp(time, history[:, 0], history[:, 1]))

        return sum(
            0.5 * (speed(a) + speed(b)) * (b - a)
            for a, b in zip(points[:-1], points[1:])
        )

    def diagnostics(self, now):
        return {
            "source": "accepted_rgbd_displacements",
            "speed_mps": self.velocity(now),
            "acceleration_mps2": self.acceleration(now),
            "contributors": self.contributors,
            "measurement_age_seconds": None
            if self.last_measurement is None
            else max(0.0, now - self.last_measurement),
            "updates": self.updates,
            "extrapolation_horizon_seconds": self.extrapolation_horizon,
        }


class _Filter:
    def __init__(self, track_id, detection, timestamp, belt_speed):
        self.track_id = track_id
        self.x = np.r_[detection.position, belt_speed, 0.0, 0.0]
        self.covariance = np.diag([0.004**2] * 3 + [0.04**2] * 3)
        self.timestamp = self.last_seen = timestamp
        self.yaw = detection.yaw
        self.size = np.asarray(detection.size).copy()
        self.shape = detection.shape
        self.yaw_period = detection.yaw_period
        self.uncertainty = detection.uncertainty
        self.kind = detection.kind
        self.confidence = detection.confidence
        self.hits = 1
        self.visible = True
        self.measurement = np.asarray(detection.position).copy()
        self.adaptive = False

    def advance(self, timestamp):
        dt = max(0.0, timestamp - self.timestamp)
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        noise = np.vstack((np.eye(3) * dt**2 / 2, np.eye(3) * dt))
        self.x = transition @ self.x
        self.covariance = (
            transition @ self.covariance @ transition.T
            + noise @ noise.T * (0.20 if self.adaptive else 0.08) ** 2
        )
        self.timestamp = timestamp
        self.visible = False

    def correct(self, detection, timestamp):
        if detection.footprint_candidates:
            candidate = min(
                detection.footprint_candidates,
                key=lambda value: abs(symmetric_angle(value[0] - self.yaw, math.pi)),
            )
            detection = replace(
                detection,
                position=np.r_[candidate[1:3], detection.position[2]],
                yaw=candidate[0],
                size=np.r_[candidate[3:5], detection.size[2]],
            )
            if (
                self.hits >= 3
                and detection.footprint_points
                and abs(symmetric_angle(detection.yaw - self.yaw, math.pi)) > 0.20
            ):
                # Several polygon faces can have indistinguishable measured
                # widths. Interpolating between their yaws invents a diagonal
                # grasp and makes a stationary prism appear to rotate. Keep the
                # observed previous direction while its current measured width
                # is still within one pixel of the narrow alternative. Refit
                # centre and dimensions to the *current* outline at that yaw.
                centre, dimensions = _rectangle_at_yaw(
                    np.asarray(detection.footprint_points),
                    self.yaw,
                    detection.footprint_pixel_size,
                    detection.footprint_pixel_axes,
                )
                if dimensions[1] <= detection.size[1] + max(
                    0.004, detection.footprint_pixel_size
                ):
                    detection = replace(
                        detection,
                        position=np.r_[centre, detection.position[2]],
                        yaw=self.yaw,
                        size=np.r_[dimensions, detection.size[2]],
                    )
        # A hand can hide most of a previously measured footprint. Do not turn
        # that exposed fragment into a new centre/height estimate during grasp.
        minimum_area = 0.80 if detection.shape == self.shape == "prism" else 0.60
        if np.prod(detection.size[:2]) < minimum_area * np.prod(self.size[:2]):
            return
        residual = detection.position - self.x[:3]
        uncertainty = max(0.0025, detection.uncertainty)
        innovation = self.covariance[:3, :3] + np.eye(3) * uncertainty**2
        gain = np.linalg.solve(innovation, self.covariance[:3, :]).T
        self.x += gain @ residual
        self.covariance -= gain @ self.covariance[:3, :]
        self.covariance = (self.covariance + self.covariance.T) / 2
        # Only belt top faces are measured. Vertical velocity is not a useful
        # grasp signal and otherwise amplifies depth quantisation/noise.
        limit = VisualBeltMotion.max_speed if self.adaptive else 0.35
        self.x[3:5] = np.clip(self.x[3:5], -limit, limit)
        self.x[5] = 0.0
        if detection.yaw_period != self.yaw_period:
            self.yaw = detection.yaw
        else:
            delta = symmetric_angle(detection.yaw - self.yaw, detection.yaw_period)
            self.yaw = symmetric_angle(self.yaw + 0.35 * delta, detection.yaw_period)
        self.size = 0.65 * self.size + 0.35 * detection.size
        self.shape = detection.shape
        self.yaw_period = detection.yaw_period
        self.uncertainty = 0.65 * self.uncertainty + 0.35 * detection.uncertainty
        self.confidence = 0.5 * self.confidence + 0.5 * detection.confidence
        self.last_seen = timestamp
        self.hits += 1
        self.visible = True
        self.measurement = np.asarray(detection.position).copy()


class MultiObjectTracker:
    """Constant-velocity Kalman filters with gated colour-aware association."""

    def __init__(self, belt_speed=0.1, max_age=1.5, max_tracks=64, adaptive=False):
        self.belt_speed = float(belt_speed)
        self.max_age = float(max_age)
        self.max_tracks = int(max_tracks)
        self.adaptive = bool(adaptive)
        self.belt_motion = VisualBeltMotion(belt_speed)
        self.reset()

    def reset(self):
        self._filters = []
        self._next_id = 1
        self._last_timestamp = -math.inf
        self.belt_motion.reset()

    def update(self, detections: list[Detection], timestamp: float):
        timestamp = float(timestamp)
        if timestamp < self._last_timestamp - 1e-9:
            raise ValueError("Tracking frames must have increasing capture timestamps")
        self._last_timestamp = timestamp
        self._filters = [
            f for f in self._filters if timestamp - f.last_seen <= self.max_age
        ]
        previous = {}
        for track in self._filters:
            previous[track.track_id] = (track.timestamp, track.x[:3].copy())
            track.advance(timestamp)
            if self.adaptive:
                old_time, old_position = previous[track.track_id]
                track.x[0] = old_position[0] + self.belt_motion.displacement(
                    old_time, timestamp
                )
        assigned = set()
        if self._filters and detections:
            cost = np.full((len(self._filters), len(detections)), 1e6)
            for i, track in enumerate(self._filters):
                for j, detection in enumerate(detections):
                    error = detection.position - track.x[:3]
                    gate = min(0.11, 0.04 + 0.05 * (timestamp - track.last_seen))
                    if (
                        track.kind == detection.kind
                        and np.linalg.norm(error[:2]) < gate
                        and abs(error[2]) < 0.025
                    ):
                        cost[i, j] = np.linalg.norm(error[:2]) + 0.5 * abs(error[2])
            ii, jj = linear_sum_assignment(cost)
            for i, j in zip(ii, jj):
                if cost[i, j] < 1e5:
                    if not detections[j].occluded:
                        self._filters[i].correct(detections[j], timestamp)
                    assigned.add(int(j))
        for j, detection in enumerate(detections):
            if (
                j not in assigned
                and not detection.occluded
                and len(self._filters) < self.max_tracks
            ):
                state = _Filter(self._next_id, detection, timestamp, self.belt_speed)
                state.adaptive = self.adaptive
                self._filters.append(state)
                self._next_id += 1
        if self.adaptive:
            self.belt_motion.update(
                [
                    (f.track_id, f.measurement, f.size)
                    for f in self._filters
                    if f.visible
                ],
                timestamp,
            )
            for state in self._filters:
                if not state.visible:
                    old_time, old_position = previous[state.track_id]
                    state.x[0] = old_position[0] + self.belt_motion.displacement(
                        old_time, timestamp
                    )
                state.x[3] = self.belt_motion.velocity(timestamp)

    def snapshots(self, now: float) -> list[Track]:
        snapshots = []
        for state in self._filters:
            age = max(0.0, now - state.last_seen)
            if age > self.max_age:
                continue
            position, velocity, size = (
                state.x[:3].copy(),
                state.x[3:].copy(),
                state.size.copy(),
            )
            position.setflags(write=False)
            velocity.setflags(write=False)
            size.setflags(write=False)
            acceleration = np.zeros(3)
            if self.adaptive:
                acceleration[0] = self.belt_motion.acceleration(state.timestamp)
            acceleration.setflags(write=False)
            snapshots.append(
                Track(
                    state.track_id,
                    position,
                    velocity,
                    state.yaw,
                    state.kind,
                    state.confidence * math.exp(-age / 0.8),
                    state.timestamp,
                    state.last_seen,
                    state.hits,
                    state.visible,
                    self.max_age,
                    size,
                    state.shape,
                    state.yaw_period,
                    state.uncertainty + 0.01 * age,
                    acceleration,
                )
            )
        return snapshots


def _footprint_area(detection):
    if len(detection.footprint_points) >= 3:
        try:
            return float(ConvexHull(detection.footprint_points).volume)
        except QhullError:
            pass
    return float(np.prod(detection.size[:2]))


def fuse_camera_detections(camera_detections):
    """Associate synchronized views, choosing complete measured grasp geometry.

    The association is one-to-one for each camera and uses only measured colour,
    height and world-space proximity. A cluster never contains two detections
    from one camera. Partial contours cannot drag a complete view's centre, and
    equivalent polygon grasp yaws are never averaged into a diagonal grasp.
    A complete, dimensionally consistent overhead view keeps its original grasp
    geometry; oblique views fill missing or visibly incomplete observations.
    """
    groups = []
    for name, detections in camera_detections.items():
        detections = list(detections)
        cost = np.full((len(groups), len(detections)), 1e6)
        for i, group in enumerate(groups):
            for j, detection in enumerate(detections):
                distances = []
                for _, other in group:
                    delta = detection.position - other.position
                    # At most half a footprint diagonal: distinct neighbouring
                    # objects should not merge simply because they share colour.
                    gate = min(
                        0.032,
                        0.5
                        * min(
                            np.linalg.norm(detection.size[:2]),
                            np.linalg.norm(other.size[:2]),
                        ),
                    )
                    distance = float(np.linalg.norm(delta[:2]))
                    if (
                        detection.kind != other.kind
                        or distance > gate
                        or abs(delta[2]) > 0.010
                        or abs(detection.size[2] - other.size[2]) > 0.015
                    ):
                        break
                    distances.append(distance)
                else:
                    cost[i, j] = max(distances)
        assigned = set()
        if groups and detections:
            ii, jj = linear_sum_assignment(cost)
            for i, j in zip(ii, jj):
                if cost[i, j] < 1e5:
                    groups[i].append((name, detections[j]))
                    assigned.add(int(j))
        groups.extend(
            [(name, detection)]
            for j, detection in enumerate(detections)
            if j not in assigned
        )
    fused = []
    for group in groups:
        areas = [_footprint_area(detection) for _, detection in group]
        largest = max(areas)
        complete = [
            detection
            for (_, detection), area in zip(group, areas)
            if area >= 0.90 * largest
        ]
        chosen = min(
            complete,
            key=lambda detection: (
                detection.uncertainty,
                detection.footprint_pixel_size,
                -detection.confidence,
            ),
        )
        for (name, detection), area in zip(group, areas):
            if name != "overhead" or area < 0.90 * largest:
                continue
            # Preserve the established overhead grasp when its footprint is
            # complete. Compare unordered dimensions because equivalent prism
            # grasp edges can exchange long/short axes across views. The finite
            # tolerances cover differing pixel footprints, not object catalogues.
            dimensions = np.sort(detection.size[:2])
            reference = np.sort(chosen.size[:2])
            same_footprint = np.all(
                np.abs(dimensions - reference) <= np.maximum(0.010, 0.20 * reference)
            )
            same_height = abs(detection.size[2] - chosen.size[2]) <= max(
                0.004, 2 * max(detection.uncertainty, chosen.uncertainty)
            )
            if same_footprint and same_height:
                chosen = detection
            break
        fused.append(replace(chosen, source_cameras=tuple(name for name, _ in group)))
    return sorted(fused, key=lambda d: (float(d.position[0]), float(d.position[1])))


@dataclass(frozen=True)
class _Frame:
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    calibration: CameraCalibration


@dataclass(frozen=True)
class _CaptureBatch:
    timestamp: float
    frames: dict[str, _Frame]


class SensorPipeline:
    """Synchronized RGB-D cameras, bounded delivery latency, one shared tracker.

    Frame counters count capture batches; per_camera counters count images.
    Each camera has an independent random stream. The overhead stream preserves
    the original single-camera draws irrespective of other active cameras.
    Compatible rgb/depth/overlay attributes always refer to the first camera.
    """

    CAMERA_RIGS = {
        "overhead": ("overhead",),
        "dual": ("oblique_south", "oblique_north"),
        "triple": ("overhead", "oblique_south", "oblique_north"),
    }

    def __init__(self, model, config):
        self.model = model
        self.width = int(getattr(config, "camera_width", 320))
        self.height = int(getattr(config, "camera_height", 240))
        self.period = 1 / float(getattr(config, "camera_hz", 10.0))
        self.latency = float(getattr(config, "camera_latency", 0.08))
        self.depth_noise = float(getattr(config, "depth_noise", 0.001))
        self.dropout = float(getattr(config, "camera_dropout", 0.02))
        self.visual_quality = getattr(config, "visual_quality", "balanced")
        self.belt_top = float(config.belt_top)
        self.rig = getattr(config, "camera_rig", "overhead")
        if self.rig not in self.CAMERA_RIGS:
            raise ValueError(f"Unknown camera rig: {self.rig!r}")
        self.camera_names = self.CAMERA_RIGS[self.rig]
        self.camera_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in self.camera_names
        }
        for name, camera_id in self.camera_ids.items():
            if camera_id < 0:
                raise ValueError(
                    f"RGB-D perception requires a calibrated '{name}' camera"
                )
        self.camera_id = self.camera_ids[self.camera_names[0]]
        self.detector = RGBDDetector(config)
        self.tracker = MultiObjectTracker(
            float(config.belt_speed),
            max_age=float(getattr(config, "grasp_coast_time", 1.5)),
            adaptive=getattr(config, "strategy", "baseline") == "adaptive",
        )
        self._queue_capacity = max(3, math.ceil(self.latency / self.period) + 3)
        self._renderer = None
        self.reset(int(getattr(config, "seed", 0)))

    def _fixed_calibration(self, camera_id):
        # Only world-fixed cameras can be calibrated before the first capture.
        if self.model.cam_bodyid[camera_id] != 0:
            return None
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.model.cam_quat[camera_id])
        focal = (
            0.5
            * self.height
            / math.tan(math.radians(float(self.model.cam_fovy[camera_id])) / 2)
        )
        return CameraCalibration(
            self.width,
            self.height,
            focal,
            self.model.cam_pos[camera_id].copy(),
            rotation.reshape(3, 3),
        )

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        # Fixed spawn keys make the secondary cameras independent of rig/order.
        self._rngs = {"overhead": self.rng}
        for name, key in (("oblique_south", 1), ("oblique_north", 2)):
            self._rngs[name] = np.random.default_rng(
                np.random.SeedSequence(seed, spawn_key=(key,))
            )
        self.tracker.reset()
        self._queue = deque(maxlen=self._queue_capacity)
        self._next_capture = self._now = 0.0
        self._frame_timestamp = None
        self.rgb = self.depth = self.overlay = None
        self._detections = []
        self._calibration = None
        self._views = {
            name: {
                "rgb": None,
                "depth": None,
                "overlay": None,
                "frame_timestamp": None,
                "calibration": self._fixed_calibration(camera_id),
                "detections": [],
                "track_ids": [],
            }
            for name, camera_id in self.camera_ids.items()
        }
        self._camera_counts = {
            name: {
                "captured_frames": 0,
                "dropped_frames": 0,
                "delivered_frames": 0,
                "total_detections": 0,
                "occluded_detections": 0,
                "queue_overflows": 0,
                "latest_capture_attempt_time": None,
                "latest_capture_time": None,
            }
            for name in self.camera_names
        }
        self._counts = {
            "captured_frames": 0,
            "dropped_frames": 0,
            "delivered_frames": 0,
            "total_detections": 0,
            "occluded_detections": 0,
            "queue_overflows": 0,
            "partial_batches": 0,
            "raw_detections": 0,
        }

    def capture(self, data, now=None):
        """Render every active camera at the same simulation state and timestamp."""
        timestamp = float(data.time if now is None else now)
        self._counts["captured_frames"] += 1
        frames = {}
        for name, camera_id in self.camera_ids.items():
            counts = self._camera_counts[name]
            counts["captured_frames"] += 1
            counts["latest_capture_attempt_time"] = timestamp
            rng = self._rngs[name]
            if rng.random() < self.dropout:
                counts["dropped_frames"] += 1
                continue
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model, height=self.height, width=self.width
                )
            self._renderer.disable_depth_rendering()
            self._renderer.update_scene(data, camera=name)
            if self.visual_quality == "balanced":
                self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            rgb, depth = render_rgbd(self._renderer)
            if self.depth_noise:
                depth += rng.normal(0.0, self.depth_noise, depth.shape).astype(
                    depth.dtype
                )
            calibration = CameraCalibration.from_camera(
                self.model, data, camera_id, self.width, self.height
            )
            frames[name] = _Frame(timestamp, rgb, depth, calibration)
            counts["latest_capture_time"] = timestamp
        if not frames:
            self._counts["dropped_frames"] += 1
            return
        if len(frames) != len(self.camera_names):
            self._counts["partial_batches"] += 1
        if len(self._queue) == self._queue.maxlen:
            self._counts["queue_overflows"] += 1
            for name in self._queue[0].frames:
                self._camera_counts[name]["queue_overflows"] += 1
        self._queue.append(_CaptureBatch(timestamp, frames))

    def update(self, data, now=None) -> list[Track]:
        now = float(data.time if now is None else now)
        if now < self._now - 1e-9:
            raise ValueError("Sensor time went backwards; call reset for a new episode")
        self._now = now
        if now + 1e-9 >= self._next_capture:
            self.capture(data, now)
            self._next_capture = now + self.period
        while self._queue and self._queue[0].timestamp + self.latency <= now + 1e-9:
            batch = self._queue.popleft()
            by_camera = {
                name: self.detector.detect(frame.rgb, frame.depth, frame.calibration)
                for name, frame in batch.frames.items()
            }
            detections = (
                next(iter(by_camera.values()))
                if len(self.camera_names) == 1
                else fuse_camera_detections(by_camera)
            )
            # Exactly one correction per synchronized capture, including when
            # only a secondary camera survives. All-drop batches never queue.
            self.tracker.update(detections, batch.timestamp)
            self._detections = detections
            self._counts["delivered_frames"] += 1
            self._counts["total_detections"] += len(detections)
            self._counts["occluded_detections"] += sum(d.occluded for d in detections)
            self._counts["raw_detections"] += sum(map(len, by_camera.values()))
            for name, frame in batch.frames.items():
                observed = by_camera[name]
                projected = self._projected_tracks(frame)
                self._camera_counts[name]["delivered_frames"] += 1
                self._camera_counts[name]["total_detections"] += len(observed)
                self._camera_counts[name]["occluded_detections"] += sum(
                    d.occluded for d in observed
                )
                self._views[name].update(
                    rgb=frame.rgb,
                    depth=frame.depth,
                    calibration=frame.calibration,
                    frame_timestamp=frame.timestamp,
                    detections=observed,
                    overlay=self._make_overlay(frame, observed, projected),
                    track_ids=[track.track_id for track, _, _ in projected],
                )
            primary = self._views[self.camera_names[0]]
            self.rgb, self.depth, self.overlay = (
                primary["rgb"],
                primary["depth"],
                primary["overlay"],
            )
            self._calibration = primary["calibration"]
            self._frame_timestamp = primary["frame_timestamp"]
        return self.tracker.snapshots(now)

    def _projected_tracks(self, frame):
        projected = []
        for track in self.tracker.snapshots(frame.timestamp):
            top = track.predict(frame.timestamp) + np.array(
                [0.0, 0.0, track.half_height]
            )
            distance = -(
                (top - frame.calibration.position) @ frame.calibration.rotation
            )[2]
            if distance <= 0:
                continue
            u, v = frame.calibration.project(top)
            if 0 <= u < self.width and 0 <= v < self.height:
                projected.append((track, float(u), float(v)))
        return projected

    def _make_overlay(self, frame, detections, projected):
        overlay = frame.rgb.copy()
        for detection in detections:
            left, top, right, bottom = detection.bbox
            colour = (255, 140, 0) if detection.occluded else (255, 210, 30)
            overlay[top : bottom + 1, (left, right)] = colour
            overlay[(top, bottom), left : right + 1] = colour
        # Pillow is optional, used only for readable debug labels.
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return overlay
        image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(image)
        for track, u, v in projected:
            colour = (40, 255, 80) if track.visible else (255, 180, 20)
            draw.ellipse((u - 2, v - 2, u + 2, v + 2), outline=colour)
            draw.text(
                (u + 4, v - 12), f"T{track.track_id} {track.shape[:3]}", fill=colour
            )
        return np.asarray(image).copy()

    def debug_overlay(self):
        return None if self.overlay is None else self.overlay.copy()

    def save_debug(self, path):
        if self.overlay is None:
            raise RuntimeError("No camera frame has been delivered yet")
        from PIL import Image

        Image.fromarray(self.overlay).save(path)

    def _camera_diagnostics(self, name):
        result = dict(self._camera_counts[name])
        view = self._views[name]
        timestamp = view["frame_timestamp"]
        result.update(
            delivered_frame_time=timestamp,
            frame_age_seconds=None if timestamp is None else self._now - timestamp,
            detections=len(view["detections"]),
            tracks=len(view["track_ids"]),
            track_ids=list(view["track_ids"]),
        )
        return result

    def camera_views(self):
        """Read latest-delivered caches without rendering. Do not mutate arrays.

        Dropped cameras retain their old image, overlay and actual timestamp,
        even while other views advance the shared tracker.
        """
        result = []
        for name in self.camera_names:
            view = self._views[name]
            calibration = view["calibration"]
            metadata = None
            depth_range = (1.3, 1.8)
            if calibration is not None:
                metadata = {
                    "name": name,
                    "position": calibration.position.tolist(),
                    "rotation": calibration.rotation.tolist(),
                    "focal": calibration.focal,
                    "width": calibration.width,
                    "height": calibration.height,
                }
                if name != "overhead":
                    vertical = float(calibration.rotation[2, 2])
                    centre = (
                        (calibration.position[2] - self.belt_top) / vertical
                        if vertical > 1e-6
                        else 1.5
                    )
                    depth_range = (max(0.05, centre - 0.3), max(0.65, centre + 0.3))
            result.append(
                {
                    "name": name,
                    "rgb": view["rgb"],
                    "depth": view["depth"],
                    "overlay": view["overlay"],
                    "frame_timestamp": view["frame_timestamp"],
                    "diagnostics": self._camera_diagnostics(name),
                    "calibration_metadata": metadata,
                    "depth_range_m": depth_range,
                }
            )
        return result

    def diagnostics(self):
        result = dict(self._counts)
        tracks = self.tracker.snapshots(self._now)
        result.update(
            rig=self.rig,
            camera_count=len(self.camera_names),
            per_camera={
                name: self._camera_diagnostics(name) for name in self.camera_names
            },
            fused_detections=self._counts["total_detections"],
            delivered_frame_time=self._frame_timestamp,
            frame_age_seconds=None
            if self._frame_timestamp is None
            else self._now - self._frame_timestamp,
            detections=len(self._detections),
            tracks=len(tracks),
            track_ids=[track.track_id for track in tracks],
            queued_frames=len(self._queue),
            queue_capacity=self._queue_capacity,
            queued_images=sum(len(batch.frames) for batch in self._queue),
            image_queue_capacity=self._queue_capacity * len(self.camera_names),
            track_capacity=self.tracker.max_tracks,
            max_track_age=max(
                (self._now - track.last_seen for track in tracks), default=0.0
            ),
            tracking_model="shared_visual_acceleration"
            if self.tracker.adaptive
            else "independent_constant_velocity",
            belt_motion=self.tracker.belt_motion.diagnostics(self._now)
            if self.tracker.adaptive
            else None,
        )
        return result

    def belt_velocity(self, now):
        """Current sensor estimate; never reads the configured speed profile."""
        speed = (
            self.tracker.belt_motion.velocity(now)
            if self.tracker.adaptive
            else self.tracker.belt_speed
        )
        return np.array([speed, 0.0, 0.0])

    def belt_displacement(self, start, end):
        """Visual odometry for a grasp reference frozen at ``start``."""
        distance = (
            self.tracker.belt_motion.displacement(start, end)
            if self.tracker.adaptive
            else self.tracker.belt_speed * (end - start)
        )
        return np.array([distance, 0.0, 0.0])

    def close(self):
        if self._renderer is not None:
            # MuJoCo 3.17 frees GL resources in the current context. Another
            # sensor may have rendered since this one was last used.
            context = getattr(self._renderer, "_gl_context", None)
            if context is not None:
                context.make_current()
            self._renderer.close()
            self._renderer = None
        self._queue.clear()
        self.rgb = self.depth = self.overlay = None
        self._frame_timestamp = self._calibration = None
        self._detections = []
        self.tracker.reset()
        for view in self._views.values():
            view.update(
                rgb=None,
                depth=None,
                overlay=None,
                frame_timestamp=None,
                detections=[],
                track_ids=[],
            )
