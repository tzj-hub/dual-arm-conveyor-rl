"""Equivalent wrist branches and visible neighbour clearance for finger grasps."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.ik import downward_quaternion
from dual_arm_conveyor.vision import Track


@pytest.fixture
def env():
    instance = ConveyorEnv(
        Config(
            num_arms=4,
            strategy="adaptive",
            sensor_mode="oracle",
            gripper_mode="fingers",
            supply_lanes=2,
            belt_speed=0.15,
            control_speed=0.65,
            max_time=2,
        ),
        seed=111000,
    )
    yield instance
    instance.close()


def target(env, slot=0, y=-0.065):
    track = Track(
        9000 + slot,
        np.array([-0.65, y, 0.23]),
        np.array([0.15, 0.0, 0.0]),
        0.0,
        0,
        1.0,
        float(env.data.time),
        float(env.data.time),
        10,
        True,
    )
    env.tracks[slot] = track
    env.belief[slot] = 1
    return track


@pytest.mark.parametrize("period", [np.pi, np.pi / 2])
def test_grasp_angles_remove_duplicate_turns_and_fit_opening(env, period):
    track = replace(target(env), yaw=0.27, yaw_period=period)
    angles = env._grasp_angles(0, track)
    assert len(angles) == len(set(np.round(angles, 7)))
    assert all(-np.pi <= yaw < np.pi for yaw in angles)
    assert all(env._grasp_size_at_yaw(track, yaw)[1] <= 0.072 for yaw in angles)
    assert np.pi / 2 in angles


@pytest.mark.parametrize("accept_alternate", [False, True])
def test_approach_switches_equivalent_branch_only_after_verification(
    env, monkeypatch, accept_alternate
):
    target(env)
    env.target[0] = 0
    env._set_phase(0, "approach")
    before = 0.27
    env.grasp_yaw[0] = before
    goal = env.data.site_xpos[env.ids.ee_sites[0]].copy()
    calls = []

    def plan(data, arm, requested, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            verified=accept_alternate and len(calls) == 2,
            reason="fixture branch rejected",
            waypoints=(np.array(requested),),
            orientation_at=lambda index, position: kwargs["quaternion"],
        )

    monkeypatch.setattr(env.planner, "plan", plan)
    _, ready = env._route_goal(0, goal, False)
    assert len(calls) == 2 and ready == accept_alternate
    for call in calls:
        assert not call["loaded"]
        assert call["grasp_offset"] == env.grasp_offsets[0]
    expected = (before + 2 * np.pi) % (2 * np.pi) - np.pi
    assert env.grasp_yaw[0] == pytest.approx(expected if accept_alternate else before)
    if accept_alternate:
        assert env.routes[0].verified
        assert env.events[-1]["event"] == "approach_wrist_branch"
        np.testing.assert_allclose(env.quat_override[0], downward_quaternion(expected))
    else:
        assert env.hold_requested[0]


def test_side_by_side_clearance_uses_visual_tracks_only(env, monkeypatch):
    target(env)
    neighbour = target(env, 1, 0.065)
    assert env._finger_neighborhood_clear(0, 0, 0)
    # A hand rotated across the belt remains feasible at ordinary lane spacing.
    assert env._finger_neighborhood_clear(0, 0, np.pi / 2)
    neighbour.position[1] = 0.020
    assert not env._finger_neighborhood_clear(0, 0, 0)
    assert env._finger_neighborhood_clear(0, 0, np.pi / 2)

    def no_truth(*args, **kwargs):
        raise AssertionError("Neighbour admission must not query object truth")

    monkeypatch.setattr(env, "cube_position", no_truth)
    monkeypatch.setattr(env, "cube_velocity", no_truth)
    for address in env.ids.cube_qpos:
        env.data.qpos[address : address + 7] = np.nan
    assert not env._finger_neighborhood_clear(0, 0, 0)
    assert env._finger_neighborhood_clear(0, 0, np.pi / 2)


def test_neighbour_relative_motion_is_checked_over_closure(env):
    target(env)
    neighbour = target(env, 1, 0.065)
    assert env._finger_neighborhood_clear(0, 0, 0)
    neighbour.velocity[1] = -0.2
    assert not env._finger_neighborhood_clear(0, 0, 0)


@pytest.mark.parametrize("phase", ["approach", "descend", "close"])
def test_blocked_neighbour_recovers_without_closing_or_stalling(
    env, monkeypatch, phase
):
    track = target(env)
    target(env, 1, -0.025)
    env.target[0] = 0
    env.belief[0] = 2
    env.grasp_yaw[0] = np.pi / 2
    env._set_phase(0, phase)
    env.drop_kind[0] = 0
    if phase == "approach":
        above = track.position.copy()
        above[2] = (
            env.config.belt_top
            + track.half_height
            + env.grasp_offsets[0]
            + env.approach_clearance
        )
        env.data.qpos[env.ids.arm_qpos[0]] = env.ik.solve_pose(
            env.data, 0, above, downward_quaternion(env.grasp_yaw[0])
        )
        mujoco.mj_forward(env.model, env.data)
        assert np.linalg.norm(env.data.site_xpos[env.ids.ee_sites[0]] - above) < 0.001
    monkeypatch.setattr(
        env, "_route_goal", lambda arm, goal, loaded, **kwargs: (goal, True)
    )
    monkeypatch.setattr(env, "_descent_feasible", lambda *args: True)
    commands = []
    monkeypatch.setattr(
        env.ik,
        "command",
        lambda data, arm, goal, **kwargs: commands.append((arm, kwargs)),
    )
    env._control()
    assert env.phase[0] == "recover"
    assert any(event["event"] == "neighbor_grasp_blocked" for event in env.events)
    command = next(kwargs for arm, kwargs in commands if arm == 0)
    assert command["hold"] and command["gripper"] == 0.04
    assert env.active_motion_plans[0] is None
