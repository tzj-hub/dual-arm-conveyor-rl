"""Capacity validation protects calibrated station stacks and the recycling pool."""

import pytest

from dual_arm_conveyor import cli
from dual_arm_conveyor.config import Config


@pytest.mark.parametrize("capacity", [2, 4, 12, 14, 22, 24])
def test_four_arm_capacity_splits_evenly_into_at_most_twelve_slots_per_station(
    capacity,
):
    config = Config(num_arms=4, strategy="adaptive", bin_capacity=capacity)
    assert config.bin_capacity // 2 <= 12
    assert config.pool_size >= 2 * capacity + 8


@pytest.mark.parametrize("capacity", [0, 1, 3, 13, 23, 25, 26, 24.0, True])
def test_four_arm_capacity_rejects_odd_oversize_or_noninteger_values(capacity):
    with pytest.raises(ValueError):
        Config(num_arms=4, strategy="adaptive", bin_capacity=capacity)


def test_pool_must_hold_both_full_long_bins_and_eight_in_flight_objects():
    with pytest.raises(ValueError, match="both bins"):
        Config(num_arms=4, strategy="adaptive", bin_capacity=24, pool_size=55)
    assert (
        Config(num_arms=4, strategy="adaptive", bin_capacity=24, pool_size=56).n_objects
        == 56
    )


def test_original_capacity_defaults_and_two_arm_limit_stay_unchanged():
    assert Config().bin_capacity == 12
    assert Config(num_arms=4, strategy="adaptive").bin_capacity == 12
    assert Config(bin_capacity=11).bin_capacity == 11
    with pytest.raises(ValueError):
        Config(bin_capacity=14)


def test_larger_capacity_must_be_explicit_in_deployment_arguments():
    config = cli.config_from_args(
        cli.parser().parse_args(
            [
                "evaluate",
                "--num-arms",
                "4",
                "--strategy",
                "adaptive",
                "--gripper-mode",
                "vacuum",
                "--bin-capacity",
                "24",
                "--pool-size",
                "64",
            ]
        )
    )
    assert config.bin_capacity == 24 and config.pool_size == 64
