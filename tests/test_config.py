"""Reject inputs which can bypass settlement or make task timing undefined."""

import pytest

from dual_arm_conveyor.config import Config


@pytest.mark.parametrize(
    "changes",
    [
        {"settle_time": 0},
        {"settle_time": -1},
        {"close_time": float("nan")},
        {"max_time": float("nan")},
        {"max_time": float("inf")},
        {"total_objects": 2.5},
        {"total_objects": True},
        {"lateral_range": -1},
        {"yaw_range": float("nan")},
        {"arrival_jitter": -1},
        {"belt_half_width": float("inf")},
        {"physics_dt": 0.003},
        {"seed": -1},
        {"schema_version": 2},
        {"schema_version": 3},
        {"object_family": "rounded"},
        {"visual_quality": "unknown"},
        {"pool_size": 31},
        {"pool_size": 257},
        {"pool_size": 64.5},
        {"pool_size": True},
        {"bin_capacity": 0},
        {"bin_capacity": 13},
        {"bin_capacity": 1.5},
        {"bin_capacity": True},
        {"auto_unload": 1},
        {"motion_mode": "teleport"},
        {"grasp_offset": float("nan")},
        {"grasp_offset": -0.013},
        {"grasp_offset": 0.013},
        {"guarded_descent": 1},
        {"guarded_descent": True, "strategy": "baseline"},
        {"guarded_descent": True, "strategy": "adaptive", "gripper_mode": "vacuum"},
    ],
)
def test_invalid_configuration_cannot_bypass_physics_or_success(changes):
    with pytest.raises(ValueError):
        Config(**changes)


def test_default_is_unlimited_supply_with_bounded_physical_pool():
    config = Config()
    assert config.schema_version == 4
    assert config.object_family == "flat_prisms"
    assert config.visual_quality == "balanced"
    assert config.camera_rig == "overhead"
    assert config.total_objects is None
    assert config.horizon == float("inf")
    assert config.n_objects == config.pool_size == 64
    assert config.bin_capacity == 12
    assert config.auto_unload
    assert config.motion_mode == "optimized"


def test_finite_supply_can_exceed_physical_pool_and_duration_is_explicit():
    config = Config(total_objects=1000, max_time=300.0)
    assert config.total_objects > config.n_objects
    assert config.horizon == 300.0
    assert Config(total_objects=50).horizon < float("inf")
