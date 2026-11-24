"""Target-local visual servo corrections must remain bounded and isolated."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dual_arm_conveyor import cli
from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.vision import Track


@pytest.mark.parametrize(
    "changes",
    [
        {"wrist_camera_mode": "all_targets"},
        {"wrist_camera_mode": "refine", "sensor_mode": "oracle"},
        {"wrist_camera_mode": "observe", "camera_rig": "triple"},
        {"wrist_camera_mode": "refine", "motion_mode": "legacy"},
        {"wrist_camera_hz": 0},
        {"wrist_camera_width": 63},
        {"wrist_max_correction": 0.013},
        {"wrist_max_correction": float("nan")},
    ],
)
def test_invalid_wrist_configuration(changes):
    with pytest.raises(ValueError):
        Config(**changes)


def test_cli_exposes_local_wrist_options():
    args = cli.parser().parse_args(
        ["play", "--wrist-camera-mode", "refine", "--wrist-camera-hz", "5"]
    )
    config = Config(**cli.sensor_config_kwargs(args))
    assert config.wrist_camera_mode == "refine"
    assert config.wrist_camera_hz == 5
    assert config.camera_rig == "overhead"


def local_controller(mode="refine"):
    env = ConveyorEnv.__new__(ConveyorEnv)
    env.config = Config(wrist_camera_mode=mode)
    env.data = SimpleNamespace(time=1.0)
    track = Track(
        7,
        np.array([0.1, 0.02, 0.23]),
        np.array([0.1, 0, 0]),
        0.0,
        0,
        0.9,
        1.0,
        0.5,
    )
    env.tracks = [track]
    env.wrist_offsets = np.zeros((2, 2))
    env.wrist_control_seconds = np.zeros(2)
    env.wrist_max_applied = np.zeros(2)
    env.wrist_used_frames = np.zeros(2, dtype=int)
    env.wrist_last_used_timestamp = np.full(2, -np.inf)
    return env, track


def test_correction_slews_then_decays_without_changing_global_track():
    env, track = local_controller()
    observation = replace(track, position=track.position + [0.008, 0, 0])
    env.wrist = SimpleNamespace(refinement=lambda arm, track_id, now: observation)
    original = track.position.copy()
    previous = original.copy()
    for _ in range(150):
        position = env._grasp_position(0, 0)
        assert np.linalg.norm(position - original) <= 0.008 + 1e-12
        assert (
            np.linalg.norm(position - previous) <= 0.025 * env.config.physics_dt + 1e-12
        )
        assert position[2] == original[2]
        previous = position
    assert position[0] > original[0] + 0.007
    assert env.wrist_used_frames[0] == 1
    np.testing.assert_array_equal(track.position, original)
    np.testing.assert_array_equal(env.wrist_offsets[1], [0, 0])
    env.wrist.refinement = lambda *args: None
    for _ in range(200):
        env._grasp_position(0, 0)
    assert np.linalg.norm(env.wrist_offsets[0]) < 0.00001


def test_fresh_overhead_keeps_its_command_without_reading_local_correction():
    env, track = local_controller()
    env.tracks[0] = replace(track, last_seen=0.9)

    def unexpected(*args):
        raise AssertionError("Fresh overhead does not need local correction")

    env.wrist = SimpleNamespace(refinement=unexpected)
    np.testing.assert_array_equal(env._grasp_position(0, 0), track.position)
    assert env.wrist_used_frames[0] == 0


def test_new_overhead_frame_smoothly_retires_an_active_local_correction():
    env, track = local_controller()
    observation = replace(track, position=track.position + [0.008, 0, 0])
    env.wrist = SimpleNamespace(refinement=lambda *args: observation)
    for _ in range(150):
        env._grasp_position(0, 0)
    before = env.wrist_offsets[0].copy()
    assert np.linalg.norm(before) > 0.007
    env.tracks[0] = replace(track, last_seen=0.9)
    env._grasp_position(0, 0)
    assert (
        np.linalg.norm(env.wrist_offsets[0] - before)
        <= 0.025 * env.config.physics_dt + 1e-12
    )
    for _ in range(200):
        env._grasp_position(0, 0)
    assert np.linalg.norm(env.wrist_offsets[0]) < 0.00001


@pytest.mark.parametrize("mode", ["off", "observe"])
def test_non_refining_modes_preserve_overhead_command_exactly(mode):
    env, track = local_controller(mode)

    def unexpected(*args):
        raise AssertionError("Non-refining mode must never request a correction")

    env.wrist = SimpleNamespace(refinement=unexpected)
    np.testing.assert_array_equal(env._grasp_position(0, 0), track.position)


def test_large_disagreement_after_new_global_frame_is_not_applied():
    env, track = local_controller()
    observation = replace(track, position=track.position + [0.04, 0, 0])
    env.wrist = SimpleNamespace(refinement=lambda *args: observation)
    np.testing.assert_array_equal(env._grasp_position(0, 0), track.position)
    assert env.wrist_used_frames[0] == 0


def test_wrist_goals_still_pass_through_collision_planner():
    # Execute the real descent branch with a deliberately refused route.
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1))
    try:
        track = Track(
            7, np.array([-0.3, 0, 0.23]), np.array([0.1, 0, 0]), 0.0, 0, 0.9, 0.0, 0.0
        )
        env.tracks[0] = track
        env.target[0] = 0
        env.phase[0] = "descend"
        corrected = track.position + [0.006, -0.003, 0]
        env._grasp_position = lambda arm, index: corrected.copy()
        checked = []

        def reject(arm, goal, held, **kwargs):
            checked.append(goal.copy())
            env.hold_requested[arm] = True
            return env.data.site_xpos[env.ids.ee_sites[arm]].copy(), False

        env._route_goal = reject
        before = env.data.ctrl[env.ids.arm_actuators[0]].copy()
        env._control()
        np.testing.assert_allclose(checked[0][:2], corrected[:2])
        np.testing.assert_array_equal(env.data.ctrl[env.ids.arm_actuators[0]], before)
        assert env.phase[0] == "descend"
    finally:
        env.close()
