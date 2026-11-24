"""Physical variable-speed profiles and deployment flags, without a heavy rollout."""

from dataclasses import replace
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dual_arm_conveyor import cli
from dual_arm_conveyor.config import Config
from dual_arm_conveyor.conveyor_speed import BeltMotion


def test_legacy_default_stays_constant_and_has_exact_distance():
    config = Config()
    assert config.strategy == "baseline" and config.belt_profile == "constant"
    motion = BeltMotion(config, 50000)
    for time in (0, 0.005, 3.2, 300):
        sample = motion.sample(time)
        assert sample.speed == 0.1 and sample.acceleration == 0
        assert sample.distance == 0.1 * time


@pytest.mark.parametrize("profile", ["sine", "random"])
@pytest.mark.parametrize("period", [0.1, 12.0])
def test_variable_speed_and_acceleration_bounds_with_analytical_integral(
    profile, period
):
    config = Config(belt_profile=profile, belt_speed=0.3, belt_speed_period=period)
    motion = BeltMotion(config, 741)
    times = np.linspace(0, 60, 2401)
    samples = [motion.sample(t) for t in times]
    velocities = np.array([s.speed for s in samples])
    accelerations = np.array([s.acceleration for s in samples])
    assert velocities.min() >= 0.2 - 1e-12
    assert velocities.max() <= 0.4 + 1e-12
    assert abs(accelerations).max() <= 0.08 + 1e-12
    assert np.ptp(velocities) > 0.06
    for time in times[1:-1:71]:
        before, after = motion.sample(time - 1e-5), motion.sample(time + 1e-5)
        center = motion.sample(time)
        assert (after.distance - before.distance) / 2e-5 == pytest.approx(
            center.speed, abs=1e-8
        )
        assert (after.speed - before.speed) / 2e-5 == pytest.approx(
            center.acceleration, abs=1e-7
        )


def test_random_drive_reproducible_under_different_sampling_and_seed_changes():
    config = Config(belt_profile="random", belt_speed=0.3)
    dense, sparse, other = (BeltMotion(config, seed) for seed in (17, 17, 18))
    for t in np.arange(0, 100, 0.03):
        dense.sample(t)
    for t in (87.1, 0, 35, 0.05, 100):
        assert dense.sample(t) == sparse.sample(t)
    assert sparse.sample(10).speed != other.sample(10).speed


def test_random_ramps_have_continuous_velocity_acceleration_and_distance_at_knots():
    motion = BeltMotion(Config(belt_profile="random", belt_speed=0.3), 11)
    for knot in range(1, 6):
        time = knot * motion.ramp_duration
        before, after = motion.sample(time - 1e-8), motion.sample(time + 1e-8)
        assert before.speed == pytest.approx(after.speed, abs=1e-10)
        assert before.acceleration == pytest.approx(after.acceleration, abs=1e-8)
        assert before.distance == pytest.approx(after.distance, abs=1e-8)


@pytest.mark.parametrize(
    "changes",
    [
        {"belt_profile": "unknown"},
        {"belt_speed_amplitude": -0.1},
        {"belt_speed_amplitude": math.nan},
        {"belt_speed_period": 0},
        {"belt_max_acceleration": math.inf},
        {"belt_profile": "random"},
        {"belt_profile": "sine", "belt_speed": 0.3, "belt_speed_amplitude": 0.3},
        {"strategy": "oracle_future"},
        {"spacing_jitter": -0.1},
        {"arrival_mode": "unknown"},
    ],
)
def test_invalid_profiles_and_strategy_rejected(changes):
    with pytest.raises(ValueError):
        Config(**changes)


def test_variable_profile_cli_values_reach_environment(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "dual_arm_conveyor.env",
        SimpleNamespace(ConveyorEnv=lambda config: config),
    )
    args = cli.parser().parse_args(
        [
            "evaluate",
            "--continuous",
            "--strategy",
            "adaptive",
            "--belt-profile",
            "random",
            "--belt-speed",
            ".3",
            "--belt-speed-amplitude",
            ".1",
            "--belt-speed-period",
            "9",
            "--belt-max-acceleration",
            ".12",
            "--spacing-jitter",
            ".08",
            "--control-speed",
            ".7",
            "--arrival-mode",
            "distance",
        ]
    )
    config = cli._make_env(args)
    assert config.belt_profile == "random" and config.strategy == "adaptive"
    assert config.belt_speed == 0.3 and config.belt_speed_amplitude == 0.1
    assert config.belt_speed_period == 9 and config.belt_max_acceleration == 0.12
    assert config.spacing_jitter == 0.08 and config.control_speed == 0.7
    assert config.arrival_mode == "distance" and config.spatial_supply
    assert config.total_objects is None and config.max_time == 300


def test_finite_variable_speed_horizon_covers_slowest_arrivals():
    finite = Config(total_objects=100, belt_speed=0.3, arrival_mode="distance")
    variable = replace(finite, belt_profile="random")
    assert variable.horizon > finite.horizon


def test_supply_mode_can_match_baseline_and_adaptive_without_changing_legacy():
    assert not Config().spatial_supply
    assert Config(strategy="adaptive").spatial_supply
    assert Config(belt_speed=0.3, belt_profile="sine").spatial_supply
    assert Config(arrival_mode="distance").spatial_supply
    assert not Config(strategy="adaptive", arrival_mode="time").spatial_supply


def test_variable_stripes_follow_integrated_distance_not_current_speed_times_time():
    from dual_arm_conveyor.belt_visual import BeltMarkers

    marker = BeltMarkers.__new__(BeltMarkers)
    marker.low, marker.span, marker.offsets, marker.speed = -1, 2, np.array([0.0]), 0.3
    marker.mocap_ids = np.array([0])
    data = SimpleNamespace(time=4.0, mocap_pos=np.zeros((1, 3)))
    motion = BeltMotion(Config(belt_profile="sine", belt_speed=0.3), 1)
    sample = motion.sample(data.time)
    marker.update(data, distance=sample.distance)
    assert data.mocap_pos[0, 0] == pytest.approx(-1 + sample.distance % 2)
    assert abs(sample.distance - sample.speed * data.time) > 0.01
