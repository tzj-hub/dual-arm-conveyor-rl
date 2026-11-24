"""Station-exclusion diagnostics use fake controller/vision state only."""

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from dual_arm_conveyor.cooperation import OpportunityMonitor


@dataclass(frozen=True)
class Track:
    track_id: int = 10
    kind: int = 1
    position: tuple = (-0.5, 0.065, 0.23)
    velocity: tuple = (0.2, 0.0, 0.0)
    timestamp: float = 4.0
    last_seen: float = 3.9
    confidence: float = 0.9
    visible: bool = True

    def predict(self, now):
        return np.asarray(self.position) + np.asarray(self.velocity) * (now - self.timestamp)


class Env:
    def __init__(self):
        self.num_arms = 4
        self.phase = ["transport", "idle", "idle", "idle"]
        self.ids = SimpleNamespace(ee_sites=np.arange(4))
        self.data = SimpleNamespace(
            time=4.0,
            site_xpos=np.array([
                [-0.5, -0.20, 0.4], [-0.5, 0.25, 0.4],
                [0.5, -0.25, 0.4], [0.5, 0.25, 0.4],
            ]),
        )
        self.config = SimpleNamespace(track_max_age=0.5)
        self.windows = np.array([[-0.95, -0.1], [-0.95, -0.1], [-0.15, 0.82], [-0.15, 0.82]])
        self.tracks = [Track()]
        self.belief = np.array([1])

    def __getattr__(self, name):
        if name in {"qpos", "states", "types", "catalog", "objects", "_candidate", "_position"}:
            raise AssertionError(f"diagnostic attempted forbidden access: {name}")
        raise AttributeError(name)


def test_disabled_monitor_does_not_read_environment():
    class Unreadable:
        def __getattribute__(self, name):
            raise AssertionError(name)

    monitor = OpportunityMonitor()
    monitor.sample(Unreadable(), 0.1)
    assert not monitor.metrics()["cooperation_monitor_enabled"]
    assert monitor.metrics()["cooperation_samples"] == 0


def test_return_reentry_and_partner_blocking_count_only_measured_strip_occupancy():
    env = Env()
    env.phase[0] = "return"
    monitor = OpportunityMonitor(enabled=True)
    monitor.sample(env, 0.1)
    env.data.site_xpos[0, 1] = -0.26
    monitor.sample(env, 0.2)
    report = monitor.metrics()
    assert report["cooperation_arm_return_strip_seconds"] == [0.1, 0, 0, 0]
    assert report["cooperation_arm_blocked_by_return_seconds"] == [0, 0.1, 0, 0]
    assert report["cooperation_arm_exit_tail_seconds"] == [0, 0, 0, 0]
    monitor.reset()
    assert monitor.metrics()["cooperation_arm_return_strip_seconds"] == [0, 0, 0, 0]


def test_strip_blocking_and_exit_tail_have_different_arm_attribution():
    env = Env()
    before = deepcopy(env)
    monitor = OpportunityMonitor(enabled=True)
    monitor.sample(env, 0.1)
    monitor.sample(env, 0.2)
    report = monitor.metrics()
    assert report["cooperation_timing"] == "decision_snapshot_estimate"
    assert report["cooperation_sampled_seconds"] == pytest.approx(0.3)
    np.testing.assert_allclose(report["cooperation_arm_blocked_seconds"], [0, 0.3, 0, 0])
    np.testing.assert_allclose(report["cooperation_arm_exit_tail_seconds"], [0.3, 0, 0, 0])
    assert report["cooperation_arm_blocked_visible_ticks"] == [0, 2, 0, 0]
    assert report["cooperation_arm_blocked_visible_unique_tracks"] == [0, 1, 0, 0]
    json.dumps(report, allow_nan=False)
    assert env.phase == before.phase and env.tracks == before.tracks
    np.testing.assert_array_equal(env.belief, before.belief)
    np.testing.assert_array_equal(env.data.site_xpos, before.data.site_xpos)


@pytest.mark.parametrize("phase", ["approach", "descend", "close", "lift", "reorient"])
def test_exclusive_phase_blocks_outside_strip_and_reasons_do_not_double_count(phase):
    env = Env()
    env.phase[0] = phase
    env.data.site_xpos[0, 1] = -0.25
    monitor = OpportunityMonitor(enabled=True)
    monitor.sample(env, 0.1)
    env.data.site_xpos[0, 1] = -0.20
    monitor.sample(env, 0.1)
    report = monitor.metrics()
    assert report["cooperation_arm_blocked_seconds"][1] == pytest.approx(0.2)
    assert report["cooperation_arm_phase_blocked_seconds"][1] == pytest.approx(0.2)
    assert report["cooperation_arm_strip_blocked_seconds"][1] == pytest.approx(0.1)
    assert report["cooperation_arm_exit_tail_seconds"] == [0, 0, 0, 0]


@pytest.mark.parametrize("phase,y", [("idle", -0.20), ("transport", -0.23), ("place", -0.25)])
def test_idle_partner_and_cleared_strip_are_not_blocking(phase, y):
    env = Env()
    env.phase[0] = phase
    env.data.site_xpos[0, 1] = y
    monitor = OpportunityMonitor(enabled=True)
    monitor.sample(env, 0.1)
    assert monitor.metrics()["cooperation_arm_blocked_seconds"] == [0, 0, 0, 0]


def test_exit_tail_does_not_require_idle_partner_and_busy_arms_are_not_blocked():
    env = Env()
    env.phase = ["place", "approach", "release", "idle"]
    env.data.site_xpos[2, 1] = -0.20
    monitor = OpportunityMonitor(enabled=True)
    monitor.sample(env, 0.1)
    report = monitor.metrics()
    assert report["cooperation_arm_exit_tail_seconds"] == [0.1, 0, 0, 0]
    assert report["cooperation_arm_blocked_seconds"] == [0, 0, 0, 0.1]


def test_coarse_candidates_filter_visibility_reservation_class_freshness_and_window():
    env = Env()
    valid = Track()
    env.tracks = [
        valid, replace(valid, track_id=11),
        replace(valid, track_id=12, kind=0),
        replace(valid, track_id=13, last_seen=3.4),
        replace(valid, track_id=14, confidence=0.34),
        replace(valid, track_id=15, visible=False),
        replace(valid, track_id=16, position=(0.5, 0.065, 0.23)),
        replace(valid, track_id=17, position=(-0.5, 0.111, 0.23)),
        replace(valid, track_id=18), None,
        replace(valid, track_id=19, position=(float("nan"), 0.065, 0.23)),
        replace(valid, track_id=20, confidence=float("nan")),
    ]
    env.belief = np.ones(len(env.tracks), dtype=int)
    env.belief[8] = 2
    monitor = OpportunityMonitor(enabled=True)
    monitor.sample(env, 0.1)
    monitor.sample(env, 0.1)
    report = monitor.metrics()
    assert report["cooperation_arm_blocked_visible_track_ticks"] == [0, 4, 0, 0]
    assert report["cooperation_arm_blocked_visible_unique_tracks"] == [0, 2, 0, 0]
    assert report["cooperation_arm_blocked_visible_track_seconds"][1] == pytest.approx(0.4)
    assert report["cooperation_arm_blocked_visible_seconds"][1] == pytest.approx(0.2)


def test_candidates_use_predicted_visual_position_and_matching_station_window():
    env = Env()
    env.phase = ["transport", "idle", "transport", "idle"]
    env.data.site_xpos[2, 1] = -0.20
    # The reported position is outside downstream's window. Visual velocity
    # advances the track to x=-0.1, inside both inclusive station windows.
    env.tracks = [replace(Track(), position=(-0.2, 0.065, 0.23), timestamp=3.5)]
    monitor = OpportunityMonitor(enabled=True)
    monitor.sample(env, 0.1)
    assert monitor.metrics()["cooperation_arm_blocked_visible_ticks"] == [0, 1, 0, 1]


def test_blocked_seconds_do_not_claim_a_visible_opportunity_when_no_track_qualifies():
    env = Env()
    env.belief[:] = 2
    monitor = OpportunityMonitor(enabled=True)
    monitor.sample(env, 0.1)
    assert monitor.metrics()["cooperation_arm_blocked_seconds"][1] == pytest.approx(0.1)
    assert monitor.metrics()["cooperation_arm_blocked_visible_ticks"] == [0, 0, 0, 0]


def test_metrics_are_detached_and_reset_starts_an_independent_episode():
    monitor = OpportunityMonitor(enabled=True)
    monitor.sample(Env(), 0.1)
    report = monitor.metrics()
    report["cooperation_arm_blocked_seconds"][1] = 999
    assert monitor.metrics()["cooperation_arm_blocked_seconds"][1] == pytest.approx(0.1)
    monitor.reset()
    assert monitor.metrics()["cooperation_samples"] == 0
    assert monitor.metrics()["cooperation_arm_blocked_visible_unique_tracks"] == [0, 0, 0, 0]


@pytest.mark.parametrize("dt", [-0.1, float("nan"), float("inf")])
def test_invalid_duration_rejected_without_counting_a_sample(dt):
    monitor = OpportunityMonitor(enabled=True)
    with pytest.raises(ValueError, match="dt"):
        monitor.sample(Env(), dt)
    assert monitor.metrics()["cooperation_samples"] == 0


def test_zero_duration_is_noop_and_wrong_arm_count_fails_explicitly():
    monitor = OpportunityMonitor(enabled=True)
    monitor.sample(object(), 0)
    env = Env()
    env.num_arms = 2
    with pytest.raises(ValueError, match="four arms"):
        monitor.sample(env, 0.1)
    assert monitor.metrics()["cooperation_samples"] == 0
