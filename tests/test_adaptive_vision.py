"""Causal RGB-D motion estimation under unseen belt changes and occlusion."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dual_arm_conveyor.vision import (
    Detection,
    MultiObjectTracker,
    RGBDDetector,
    VisualBeltMotion,
)


def detection(x, y=0.0, kind=0):
    return Detection(np.array([x, y, 0.23]), 0.0, kind, 0.95, (0, 0, 6, 6))


def run_sine(adaptive, seed=251):
    """Only image observations are passed to the tracker, never this trajectory."""
    tracker = MultiObjectTracker(belt_speed=0.3, adaptive=adaptive)
    rng = np.random.default_rng(seed)
    errors, velocity_errors = [], []
    for index in range(241):
        now = index * 0.1
        position = -0.8 + 0.3 * now + 0.1 * (1 - np.cos(0.8 * now)) / 0.8
        observations = []
        for obj in range(3):
            if obj == 0 and 12 <= index % 30 < 20:
                continue
            measured = np.array([position + 0.3 * obj, 0.03 * (obj - 1), 0.23])
            measured += rng.normal(0.0, 0.0008, 3)
            observations.append(
                replace(detection(0.0, kind=obj % 2), position=measured)
            )
        # Detection ordering must not select the velocity source or change IDs.
        rng.shuffle(observations)
        tracker.update(observations, now)
        candidates = [track for track in tracker.snapshots(now) if track.kind == 0]
        track = min(candidates, key=lambda item: item.position[0])
        if index > 10:
            future = now + 0.08  # image delivery latency, not an input future command
            true_position = -0.8 + 0.3 * future + 0.1 * (1 - np.cos(0.8 * future)) / 0.8
            true_velocity = 0.3 + 0.1 * np.sin(0.8 * future)
            errors.append(track.predict(future)[0] - true_position)
            velocity_errors.append(track.predict_velocity(future)[0] - true_velocity)
    return np.asarray(errors), np.asarray(velocity_errors), tracker


@pytest.mark.parametrize("seed", [251, 903])
def test_heldout_sinusoid_prediction_through_occlusion_beats_constant_velocity(seed):
    old_errors, _, _ = run_sine(False, seed)
    errors, velocity_errors, tracker = run_sine(True, seed)
    assert np.percentile(np.abs(errors), 95) < 0.005
    assert np.percentile(np.abs(errors), 95) < 0.25 * np.percentile(
        np.abs(old_errors), 95
    )
    assert np.percentile(np.abs(velocity_errors), 95) < 0.02
    assert len(tracker.snapshots(24.0)) == 3
    assert (
        tracker.belt_motion.diagnostics(24.0)["source"] == "accepted_rgbd_displacements"
    )


def test_constant_point_four_is_not_clipped_to_old_point_three_five_limit():
    tracker = MultiObjectTracker(belt_speed=0.3, adaptive=True)
    for index in range(41):
        now = index * 0.1
        tracker.update([detection(-0.8 + 0.4 * now)], now)
    track = tracker.snapshots(4.0)[0]
    assert track.velocity[0] == pytest.approx(0.4, abs=0.002)
    assert track.predict(4.08)[0] == pytest.approx(-0.8 + 0.4 * 4.08, abs=0.003)
    assert not track.acceleration.flags.writeable


def test_common_motion_rejects_one_independently_moving_outlier():
    motion = VisualBeltMotion(initial_speed=0.3)
    size = np.array([0.05, 0.05, 0.05])
    for index in range(31):
        now = index * 0.1
        observations = [
            (identity, np.array([offset + velocity * now, 0.0, 0.23]), size)
            for identity, offset, velocity in [
                (1, -0.6, 0.3),
                (2, 0.0, 0.3),
                (3, 0.6, 0.03),
            ]
        ]
        motion.update(observations, now)
    assert motion.velocity(3.0) == pytest.approx(0.3, abs=0.002)
    assert motion.contributors == 2
    assert motion.displacement(2.0, 3.0) == pytest.approx(0.3, abs=0.003)


def test_shared_motion_does_not_refresh_hidden_object_measurement_deadline():
    tracker = MultiObjectTracker(belt_speed=0.3, adaptive=True, max_age=0.5)
    tracker.update([detection(-0.5), detection(0.2, kind=1)], 0.0)
    for index in range(1, 6):
        now = index * 0.1
        tracker.update([detection(0.2 + 0.3 * now, kind=1)], now)
        hidden = next(track for track in tracker.snapshots(now) if track.track_id == 1)
        assert hidden.last_seen == 0.0 and not hidden.visible
        assert hidden.hits == 1
    tracker.update([detection(0.38, kind=1)], 0.6)
    assert [track.track_id for track in tracker.snapshots(0.6)] == [2]


def test_all_objects_occluded_stops_acceleration_and_expires_tracks():
    tracker = MultiObjectTracker(belt_speed=0.2, adaptive=True, max_age=0.5)
    for index in range(21):
        now = index * 0.1
        tracker.update([detection(-0.8 + 0.2 * now + 0.04 * now**2)], now)
    assert tracker.belt_motion.acceleration(2.0) > 0.04
    previous_speed = tracker.belt_motion.velocity(2.0)
    previous_acceleration = tracker.belt_motion.acceleration(2.0)
    for now in (2.1, 2.2, 2.3, 2.4):
        tracker.update([], now)
    assert (
        tracker.belt_motion.velocity(2.4)
        <= previous_speed + 0.25 * previous_acceleration + 1e-9
    )
    assert tracker.belt_motion.acceleration(2.4) == 0.0
    assert tracker.belt_motion.diagnostics(2.4)[
        "measurement_age_seconds"
    ] == pytest.approx(0.4)
    tracker.update([], 2.6)
    assert tracker.snapshots(2.6) == []


def test_visual_displacement_uses_history_when_speed_changes_during_grasp():
    motion = VisualBeltMotion(initial_speed=0.2)
    size = np.array([0.05, 0.05, 0.05])
    for index in range(41):
        now = index * 0.1
        x = -0.8 + 0.2 * now + 0.025 * now**2
        motion.update([(1, np.array([x, 0.0, 0.23]), size)], now)
    expected = 0.2 * 0.8 + 0.025 * (4.0**2 - 3.2**2)
    assert motion.displacement(3.2, 4.0) == pytest.approx(expected, abs=0.003)
    assert motion.displacement(4.0, 3.2) == pytest.approx(-expected, abs=0.003)
    # Extrapolation does not continue estimated acceleration indefinitely.
    assert motion.velocity(40.0) == motion.velocity(4.25)
    assert motion.acceleration(40.0) == 0.0


def test_adaptive_guard_is_identical_across_camera_modes_without_changing_baseline():
    base = dict(belt_top=0.205, belt_half_length=1.15, belt_half_width=0.16)
    for mode in ("off", "observe", "refine"):
        detector = RGBDDetector(
            SimpleNamespace(**base, strategy="adaptive", wrist_camera_mode=mode)
        )
        assert detector.occlusion_guard
    assert not RGBDDetector(
        SimpleNamespace(**base, wrist_camera_mode="off")
    ).occlusion_guard


def test_motion_reset_clears_all_prior_episode_evidence():
    motion = VisualBeltMotion(initial_speed=0.3)
    size = np.ones(3) * 0.05
    motion.update([(1, np.array([0, 0, 0.23]), size)], 0.0)
    motion.update([(1, np.array([0.04, 0, 0.23]), size)], 0.1)
    assert motion.updates == 1
    motion.reset()
    assert motion.velocity(10.0) == 0.3
    assert motion.displacement(0.0, 1.0) == 0.3
    assert motion.last_measurement is None and motion.updates == 0
    assert not motion._previous and not motion._history
