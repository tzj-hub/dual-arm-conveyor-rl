"""Projection caching preserves measurements and follows camera calibration."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dual_arm_conveyor.vision import CameraCalibration, RGBDDetector


def detector():
    return RGBDDetector(
        SimpleNamespace(
            belt_top=0.205,
            belt_half_length=1.15,
            belt_half_width=0.16,
            object_family="flat_prisms",
            strategy="adaptive",
        )
    )


def calibration(width=80, height=60, angle=0.0):
    cs, sn = np.cos(angle), np.sin(angle)
    return CameraCalibration(
        width,
        height,
        120.0,
        np.array([0.02, -0.03, 1.8]),
        np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]]),
    )


def assert_reference_projection(subject, camera, depth):
    rows, columns = np.indices(depth.shape)
    with np.errstate(invalid="ignore"):
        expected = camera.unproject(rows, columns, depth)
        observed = subject._unproject_image(depth, camera)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("angle", [0.0, -0.7, 1.1])
def test_cached_projection_matches_reference_exactly(dtype, angle):
    camera = calibration(angle=angle)
    depth = np.random.default_rng(89).uniform(0.2, 2.0, (60, 80)).astype(dtype)
    depth[0, :4] = [0, np.nan, np.inf, -np.inf]
    subject = detector()
    assert_reference_projection(subject, camera, depth)
    # A new frame reuses pixel offsets, never old depth or camera pose.
    pixels = subject._pixel_offsets
    assert_reference_projection(subject, camera, depth * 0.8)
    assert subject._pixel_offsets is pixels


def test_cache_follows_intrinsics_extrinsics_and_resolution_changes():
    subject = detector()
    first = calibration()
    cameras = [
        first,
        replace(first, focal=140.0),
        replace(first, position=np.array([0.1, 0.2, 0.6])),
        calibration(angle=0.63),
        calibration(width=96, height=72),
        first,
    ]
    rng = np.random.default_rng(91)
    for camera in cameras:
        depth = rng.uniform(0.4, 1.5, (camera.height, camera.width))
        assert_reference_projection(subject, camera, depth)


def test_detector_outputs_are_unchanged_including_occlusion_guard():
    camera = calibration(width=100, height=80)
    camera = replace(camera, focal=200.0, position=np.array([0.0, 0.0, 1.0]))
    rng = np.random.default_rng(92)
    for occlusion in (False, True):
        rgb = np.full((80, 100, 3), 60, dtype=np.uint8)
        depth = np.full((80, 100), 0.795, dtype=np.float32)
        rgb[35:47, 44:57] = [220, 30, 20]
        depth[35:47, 44:57] = 0.745
        if occlusion:
            rgb[39:42, 49:51] = [110, 110, 110]
            depth[39:42, 49:51] = 0.72
        depth += rng.normal(0, 0.001, depth.shape).astype(depth.dtype)
        optimized, original = detector(), detector()
        original._unproject_image = lambda value, cal: cal.unproject(
            *np.indices(value.shape), value
        )
        actual = optimized.detect(rgb, depth, camera)
        expected = original.detect(rgb, depth, camera)
        assert len(actual) == len(expected) == 1
        for name in actual[0].__dataclass_fields__:
            first, second = getattr(actual[0], name), getattr(expected[0], name)
            if isinstance(first, np.ndarray):
                np.testing.assert_array_equal(first, second)
            else:
                assert first == second
