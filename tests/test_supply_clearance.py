"""Admission checks actual edge clearance, including cross-lane diagonals."""

from dataclasses import replace

import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv, State


@pytest.fixture
def env():
    instance = ConveyorEnv(Config(
        num_arms=4, strategy="adaptive", sensor_mode="oracle",
        belt_speed=.15, supply_lanes=2, spacing=.60, min_object_gap=.08,
        physics_dt=.0025, bin_capacity=24,
    ), seed=31)
    yield instance
    instance.close()


def place_neighbour(env, dx, dy, lift=0):
    env.states[:] = State.PENDING
    env.states[0] = State.BELT
    q = env.ids.cube_qpos[0]
    env.data.qpos[q:q+7] = [
        -.98 + dx, dy,
        env.config.belt_top + env.ids.object_specs[0].half_height + .001 + lift,
        1, 0, 0, 0,
    ]


@pytest.mark.parametrize("gap,reason", [(.05, "gap"), (.082, "gap"), (.09, None)])
def test_cross_lane_edge_gap_is_not_centre_spacing(env, gap, reason):
    first, proposed = env.ids.object_specs[0], env.ids.object_specs[2]
    place_neighbour(env, 0, first.half_sizes[1] + proposed.half_sizes[1] + gap)
    assert env._inlet_obstruction(2, 0, 0) == reason


def test_diagonal_uses_euclidean_clearance_instead_of_square_padding(env):
    half = np.array(env.ids.object_specs[0].half_sizes) + env.ids.object_specs[2].half_sizes
    # Two 60 mm axis gaps give an 84.85 mm planar gap: sufficient even
    # though neither individual axis exceeds 80 mm.
    place_neighbour(env, half[0] + .060, half[1] + .060)
    assert env._inlet_obstruction(2, 0, 0) is None
    place_neighbour(env, half[0] + .050, half[1] + .050)
    assert env._inlet_obstruction(2, 0, 0) == "gap"


def test_random_yaw_expands_admission_footprint(env):
    half = env.ids.object_specs[2].half_sizes
    rotated_x = (half[0] + half[1]) / np.sqrt(2)
    expansion = rotated_x - half[0]
    assert expansion > 0
    dx = env.ids.object_specs[0].half_sizes[0] + half[0] + .083 + expansion / 2
    place_neighbour(env, dx, 0)
    assert env._inlet_obstruction(2, 0, 0) is None
    assert env._inlet_obstruction(2, np.pi / 4, 0) == "gap"


def test_carried_object_above_belt_does_not_block_spacing(env):
    place_neighbour(env, 0, 0, lift=.4)
    env.states[0] = State.HELD
    assert env._inlet_obstruction(2, 0, 0) is None
    place_neighbour(env, 0, 0)
    # 3D overlap checks remain active independently of the new spacing rule.
    assert env._inlet_obstruction(2, 0, 0) == "object"


def test_zero_gap_retains_original_birth_clearance(env):
    env.config = replace(env.config, min_object_gap=0)
    half = np.array(env.ids.object_specs[0].half_sizes) + env.ids.object_specs[2].half_sizes
    place_neighbour(env, half[0] + .02, 0)
    assert env._inlet_obstruction(2, 0, 0) is None


@pytest.mark.parametrize("gap", [-.01, float("nan"), float("inf")])
def test_invalid_gap_is_rejected(gap):
    with pytest.raises(ValueError, match="min_object_gap"):
        Config(supply_lanes=2, strategy="adaptive", min_object_gap=gap)


def test_single_lane_cannot_silently_ignore_requested_gap():
    with pytest.raises(ValueError, match="two-lane"):
        Config(min_object_gap=.08)
