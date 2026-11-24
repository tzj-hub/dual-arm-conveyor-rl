"""Sensor-derived grasp depth and real escape from a low recovery pose."""

from dataclasses import replace
import math

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.ik import downward_quaternion
from dual_arm_conveyor.motion import MotionPlan
from dual_arm_conveyor.vision import Track


@pytest.mark.parametrize("height,expected_offset", [(0.060, 0.0), (0.040, -0.009)])
def test_adaptive_grasp_offset_uses_track_height_without_truth_catalogue(
    monkeypatch, height, expected_offset
):
    env = ConveyorEnv(
        Config(sensor_mode="oracle", total_objects=1, grasp_offset=-0.009)
    )
    try:
        track = Track(
            track_id=9_000_001,  # deliberately not a physical pool index
            position=np.array([-0.3, 0, env.config.belt_top + height / 2]),
            velocity=np.array([0.1, 0, 0]),
            yaw=math.pi / 2,
            kind=0,
            confidence=1.0,
            timestamp=0.0,
            last_seen=0.0,
            size=np.array([0.05, 0.04, height]),
            shape="box",
            yaw_period=math.pi,
        )
        env.tracks = [None] * 10
        env.belief[:] = 0
        env.track_slots.clear()
        env._ingest_tracks([track])

        class ForbiddenCatalogue:
            def __getitem__(self, key):
                raise AssertionError(
                    "Grasp controller read the hidden object catalogue"
                )

            def __iter__(self):
                raise AssertionError(
                    "Grasp controller iterated the hidden object catalogue"
                )

            def __len__(self):
                raise AssertionError(
                    "Grasp controller inspected the hidden object catalogue"
                )

        def forbidden_truth(*args):
            raise AssertionError("Grasp controller read an evaluator pose/velocity")

        hidden = replace(env.ids, object_specs=ForbiddenCatalogue())
        for owner in (env, env.ik, env.planner):
            monkeypatch.setattr(owner, "ids", hidden)
        monkeypatch.setattr(env, "cube_position", forbidden_truth)
        monkeypatch.setattr(env, "cube_velocity", forbidden_truth)
        env._assign(0, 0)
        assert env.grasp_offsets[0] == pytest.approx(expected_offset)
        np.testing.assert_allclose(
            env.grasp_sizes[0], [np.linalg.norm(track.size[:2])] * 2 + [height]
        )
        assert env.grasp_min_widths[0] == min(track.size[:2])
        requested_offsets = []
        original_plan = env.planner.plan

        def record_plan(*args, **kwargs):
            requested_offsets.append(kwargs["grasp_offset"])
            return original_plan(*args, **kwargs)

        monkeypatch.setattr(env.planner, "plan", record_plan)
        _, ready = env._route_goal(0, env.ids.home_ee[0] + [0, 0.02, 0], True)
        assert ready
        assert requested_offsets == [pytest.approx(expected_offset)]
    finally:
        env.close()


def _failed_route():
    return MotionPlan((), False, "test blocked route", 0.0, 0.0, 1, 0.0, 0.0)


@pytest.mark.parametrize("aperture,touches", [(0.06, [0.0, 0.0]), (0.08, [0.2, 0.0])])
def test_recovery_waits_for_actual_release_before_either_planner(
    monkeypatch, aperture, touches
):
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1))
    try:
        env._recover(0)
        env.data.qpos[env.ids.finger_qpos[0]] = aperture / 2
        env.data.sensordata[env.ids.finger_touch_adr[0]] = touches

        def forbidden_plan(*args, **kwargs):
            raise AssertionError("Unloaded recovery planned before physical release")

        monkeypatch.setattr(env.planner, "plan", forbidden_plan)
        monkeypatch.setattr(env.planner, "plan_home", forbidden_plan)
        monkeypatch.setattr(env.planner, "plan_escape", forbidden_plan)
        env._control()
        assert env.phase[0] == "recover"
        assert env.hold_requested[0] and env.ik._holding[0]
        assert env.routes[0] is None and env.escape_attempts == 0
        assert env.ik.gripper_targets[0] == pytest.approx(0.04)
        assert env.metrics()["recovery_waiting_for_release"] == [True, False]
    finally:
        env.close()


def test_failed_route_retries_keep_one_joint_hold_in_the_environment(monkeypatch):
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1))
    try:
        env._recover(0)
        env.data.qvel[env.ids.arm_dofs[0]] = [0.02, 0, 0, 0, 0, 0, 0]
        initial = env.data.qpos[env.ids.arm_qpos[0]].copy()
        calls = []

        def reject(*args, **kwargs):
            calls.append(float(env.data.time))
            return _failed_route()

        original_solve = env.ik._solve

        def no_hold_ik(data, arm, *args, **kwargs):
            assert arm != 0, "Failed-route hold must not re-run the null-space IK"
            return original_solve(data, arm, *args, **kwargs)

        monkeypatch.setattr(env.planner, "plan", reject)
        monkeypatch.setattr(env.planner, "plan_home", reject)
        monkeypatch.setattr(env.planner, "plan_escape", reject)
        monkeypatch.setattr(env.ik, "_solve", no_hold_ik)
        for _ in range(160):
            env._control()
            mujoco.mj_step(env.model, env.data)
            assert env.phase[0] == "recover" and env.hold_requested[0]
            np.testing.assert_array_equal(env.ik._hold_q[0], initial)
        assert max(calls) > 0.6
        np.testing.assert_allclose(
            env.data.qpos[env.ids.arm_qpos[0]], initial, atol=1e-5
        )
    finally:
        env.close()


@pytest.mark.parametrize("kind", ["partial", "full"])
def test_escape_finishes_its_own_endpoint_before_normal_recovery(monkeypatch, kind):
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1))
    try:
        start = env.ids.home_ee[0] + [0.08, 0.0, 0.0]
        joints = env.ik.solve_pose(env.data, 0, start)
        env.data.qpos[env.ids.arm_qpos[0]] = joints
        env.data.ctrl[env.ids.arm_actuators[0]] = joints
        mujoco.mj_forward(env.model, env.data)
        env.ik.reset(env.data)
        endpoint = start + [0.0, 0.0, 0.03] if kind == "partial" else env.ids.home_ee[0]
        original_plan = env.planner.plan
        original_home = env.planner.plan_home
        checked = original_plan(env.data, 0, endpoint)
        assert checked.verified
        escape = replace(checked, reason="escape " + kind)
        normal_calls = []

        def reject(*args, **kwargs):
            normal_calls.append(float(env.data.time))
            return _failed_route()

        monkeypatch.setattr(env.planner, "plan", reject)
        monkeypatch.setattr(
            env.planner, "plan_home", lambda *args, **kwargs: _failed_route()
        )
        monkeypatch.setattr(env.planner, "plan_escape", lambda *args, **kwargs: escape)
        env._recover(0)
        for _ in range(500):
            env._control()
            if env.escape_successes:
                # Completing even a full escape cannot bypass the final normal
                # orientation check by transitioning straight to idle.
                assert env.phase[0] == "recover"
                assert env.hold_requested[0] and env.routes[0] is None
                break
            assert env.escape_routes[0]
            np.testing.assert_allclose(env.last_goals[0], endpoint)
            mujoco.mj_step(env.model, env.data)
        else:
            pytest.fail("The injected checked escape did not reach its own endpoint")
        assert len(normal_calls) == 1
        assert (
            np.linalg.norm(env.data.site_xpos[env.ids.ee_sites[0]] - endpoint) < 0.004
        )
        monkeypatch.setattr(env.planner, "plan", original_plan)
        monkeypatch.setattr(env.planner, "plan_home", original_home)
        for _ in range(500):
            env._control()
            mujoco.mj_step(env.model, env.data)
            if env.phase[0] == "idle":
                break
        assert env.phase[0] == "idle"
        assert env.recovery_home_returns[0] == 1
        assert (
            np.max(np.abs(env.data.qpos[env.ids.arm_qpos[0]] - env.ids.home_qpos[0]))
            < 0.02
        )
        assert (
            np.linalg.norm(env.data.site_xpos[env.ids.ee_sites[0]] - env.ids.home_ee[0])
            < 0.004
        )
    finally:
        env.close()


def test_recovery_at_home_waits_for_checked_nominal_orientation():
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1))
    try:
        tilted = downward_quaternion(math.pi / 2 + 0.3)
        joints = env.ik.solve_pose(env.data, 0, env.ids.home_ee[0], tilted)
        env.data.qpos[env.ids.arm_qpos[0]] = joints
        env.data.ctrl[env.ids.arm_actuators[0]] = joints
        mujoco.mj_forward(env.model, env.data)
        env.ik.reset(env.data)
        env._recover(0)
        env._control()
        assert env.phase[0] == "recover"
        for _ in range(500):
            mujoco.mj_step(env.model, env.data)
            env._control()
            if env.phase[0] == "idle":
                break
        actual = np.empty(4)
        mujoco.mju_mat2Quat(actual, env.data.site_xmat[env.ids.ee_sites[0]])
        assert env.phase[0] == "idle"
        assert (
            2
            * math.acos(
                np.clip(abs(np.dot(actual, downward_quaternion(math.pi / 2))), 0, 1)
            )
            < 0.04
        )
    finally:
        env.close()


def test_stuck_release_times_out_to_visible_safe_recovery():
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1))
    try:
        env.target[0] = 0
        env.belief[0] = 3
        env.holding[0] = True
        env._set_phase(0, "release")
        env.data.time = 3.1
        env.data.qpos[env.ids.finger_qpos[0]] = 0.025
        env.data.sensordata[env.ids.finger_touch_adr[0]] = [0.3, 0.3]
        env._control()
        assert env.phase[0] == "recover" and env.release_timeouts == 1
        assert env.arm_releases[0] == 0
        assert env.retries == 1
        assert any(event["event"] == "release_timeout" for event in env.events)
        env._control()
        assert env.hold_requested[0] and env.ik._holding[0]
        assert env.metrics()["recovery_waiting_for_release"][0]
    finally:
        env.close()


def test_idle_hold_stays_fixed_for_ten_seconds_and_resumes_a_new_pick():
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1))
    try:
        parked = np.array([0.0, -0.33, 0.325])
        joints = env.ik.solve_pose(env.data, 0, parked)
        env.data.qpos[env.ids.arm_qpos[0]] = joints
        env.data.ctrl[env.ids.arm_actuators[0]] = joints
        mujoco.mj_forward(env.model, env.data)
        env.ik.reset(env.data)
        env.idle_anchors[0] = env.data.site_xpos[env.ids.ee_sites[0]]
        initial = env.data.qpos[env.ids.arm_qpos[0]].copy()
        for _ in range(2000):
            env._control()
            mujoco.mj_step(env.model, env.data)
        assert env.phase[0] == "idle" and env.ik._holding[0]
        np.testing.assert_array_equal(env.ik._hold_q[0], initial)
        np.testing.assert_allclose(
            env.data.qpos[env.ids.arm_qpos[0]], initial, atol=1e-5
        )
        track = Track(
            track_id=12345,
            position=np.array([-0.3, 0.0, env.config.belt_top + 0.025]),
            velocity=np.array([0.1, 0.0, 0.0]),
            yaw=math.pi / 2,
            kind=0,
            confidence=1.0,
            timestamp=float(env.data.time),
            last_seen=float(env.data.time),
            size=np.array([0.05, 0.04, 0.05]),
            shape="box",
            yaw_period=math.pi,
        )
        env._ingest_tracks([track])
        slot = env.track_slots[track.track_id]
        env._assign(0, slot)
        for _ in range(40):
            env._control()
            mujoco.mj_step(env.model, env.data)
        assert env.phase[0] == "approach" and not env.ik._holding[0]
        assert np.linalg.norm(env.data.qpos[env.ids.arm_qpos[0]] - initial) > 0.01
    finally:
        env.close()


def test_real_low_recovery_escapes_positive_finger_belt_gap_without_deadlock(
    monkeypatch,
):
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1))
    try:
        low_goal = np.array([-0.45, 0.0, 0.215])
        quaternion = downward_quaternion(math.pi / 2)
        # A controlled initial robot pose with a real positive finger/belt gap.
        # Every subsequent movement uses the ordinary actuators and env.step.
        joints = env.ik.solve_pose(env.data, 0, low_goal, quaternion)
        env.data.qpos[env.ids.arm_qpos[0]] = joints
        env.data.ctrl[env.ids.arm_actuators[0]] = joints
        mujoco.mj_forward(env.model, env.data)
        env.ik.reset(env.data)
        strict = env.planner.plan(
            env.data, 0, env.ids.home_ee[0], quaternion=quaternion
        )
        assert not strict.verified and "belt" in strict.reason
        assert 0.001 < strict.minimum_clearance < env.planner.margin
        requests = []
        original_plan = env.planner.plan
        original_home = env.planner.plan_home

        def record_plan(*args, **kwargs):
            requests.append((env.phase[0], kwargs.get("near_belt", False)))
            return original_plan(*args, **kwargs)

        monkeypatch.setattr(env.planner, "plan", record_plan)

        def record_home(*args, **kwargs):
            requests.append((env.phase[0], kwargs.get("near_belt", False)))
            return original_home(*args, **kwargs)

        monkeypatch.setattr(env.planner, "plan_home", record_home)
        env._recover(0)
        for _ in range(80):
            _, _, terminated, truncated, metrics = env.step(120)
            assert not terminated and not truncated
            assert metrics["robot_collision_steps"] == 0
            assert metrics["inter_arm_collision_steps"] == 0
            if env.phase[0] == "idle":
                break
        assert env.phase[0] == "idle"
        assert env.target[0] == -1
        assert (
            np.linalg.norm(env.data.site_xpos[env.ids.ee_sites[0]] - env.ids.home_ee[0])
            < 0.03
        )
        assert requests[0] == ("recover", True)
        # Once recovered, normal planning again enforces the full margin.
        _, ready = env._route_goal(0, low_goal, False)
        assert not ready and "belt" in env.routes[0].reason
        assert requests[-1] == ("idle", False)
    finally:
        env.close()
