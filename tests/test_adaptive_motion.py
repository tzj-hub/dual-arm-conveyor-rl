"""Physical regressions for predictive, collision-checked conveyor following."""

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.ik import PandaIK
from dual_arm_conveyor.model import build_scene, reset_robot_pose
from dual_arm_conveyor.motion import MotionPlanner


def tracking_scene():
    model, ids = build_scene(Config(sensor_mode="oracle"))
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    solver = PandaIK(model, ids, adaptive=True, max_cartesian_speed=0.65)
    start = np.array([-0.45, 0.0, 0.35])
    # Initialize an empty-hand tracking experiment; execution below uses only
    # physical actuators. No state is overwritten during the trajectory.
    data.qpos[ids.arm_qpos[0]] = solver.solve_pose(data, 0, start)
    data.ctrl[ids.arm_actuators[0]] = data.qpos[ids.arm_qpos[0]]
    mujoco.mj_forward(model, data)
    solver.reset(data)
    return model, data, ids, solver, MotionPlanner(model, ids), start


@pytest.mark.parametrize("speed", [0.2, 0.3, 0.4, "variable"])
def test_predictive_checked_path_tracks_fast_and_changing_belt(speed):
    model, data, ids, solver, planner, position = tracking_scene()
    plan = None
    errors = []
    reference_positions = []
    previous_command = solver.q_targets[0].copy()
    dt = model.opt.timestep
    for _ in range(300):
        velocity = (
            0.3 + 0.1 * np.sin(2 * np.pi * data.time / 1.8)
            if speed == "variable"
            else speed
        )
        feedforward = np.array([velocity, 0.0, 0.0])
        planning_endpoint = position + 0.12 * feedforward
        if (
            plan is None
            or np.linalg.norm(planning_endpoint - plan.waypoints[-1]) > 0.01
        ):
            plan = planner.plan(data, 0, planning_endpoint)
            assert plan.verified, plan.reason
        solver.command(data, 0, position, feedforward=feedforward, motion_plan=plan)
        solver.command(data, 1, ids.home_ee[1], hold=True)
        np.testing.assert_allclose(
            solver.last_joint_reference[0],
            plan.joint_reference_at(0, solver.cart_targets[0]),
            atol=1e-12,
        )
        assert (
            np.max(np.abs(solver.q_targets[0] - previous_command))
            <= solver.max_joint_speed * dt + 1e-12
        )
        previous_command = solver.q_targets[0].copy()
        reference_positions.append(solver.cart_targets[0].copy())
        if data.time >= 0.9:
            errors.append(
                np.linalg.norm(data.site_xpos[ids.ee_sites[0], :2] - position[:2])
            )
        mujoco.mj_step(model, data)
        assert not any(
            contact.dist < -0.0005
            and (
                ids.robot_geom_arm[contact.geom1] >= 0
                or ids.robot_geom_arm[contact.geom2] >= 0
            )
            for contact in data.contact
        )
        position[0] += velocity * dt
    # The former .2 m/s regression had 7.8 mm median steady following error,
    # preventing the unchanged 6 mm descent-entry criterion from ever passing.
    assert np.median(errors) < 0.003
    assert max(errors) < 0.006
    velocity = np.diff(reference_positions, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    assert np.max(np.linalg.norm(velocity, axis=1)) <= solver.max_cartesian_speed + 1e-9
    assert (
        np.max(np.linalg.norm(acceleration, axis=1))
        <= solver.max_cartesian_acceleration + 1e-9
    )


def test_adaptive_servo_never_extrapolates_past_checked_joint_endpoint():
    model, data, ids, solver, planner, position = tracking_scene()
    plan = planner.plan(data, 0, position + [0.02, 0, 0])
    assert plan.verified
    for _ in range(100):
        solver.command(
            data,
            0,
            plan.waypoints[0] + [0.04, 0, 0],
            feedforward=[0.4, 0, 0],
            motion_plan=plan,
        )
        solver.command(data, 1, ids.home_ee[1], hold=True)
        mujoco.mj_step(model, data)
    np.testing.assert_array_equal(
        solver.last_joint_reference[0], plan.joint_paths[0][-1]
    )
    np.testing.assert_array_equal(solver.q_targets[0], plan.joint_paths[0][-1])
    assert data.site_xpos[ids.ee_sites[0], 0] <= plan.waypoints[0][0] + 0.001
