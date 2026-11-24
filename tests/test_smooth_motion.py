"""Dynamic checks for continuous execution of checked joint-space paths."""

import mujoco
import numpy as np

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.ik import PandaIK
from dual_arm_conveyor.model import build_scene, reset_robot_pose
from dual_arm_conveyor.motion import MotionPlanner


# Recorded encoders from the v3 vision stream 50000 at 145.1 simulated seconds.
# The tool was at home, but the redundant joint branch differed by 3.56 rad.
BRANCH = np.array(
    [
        [
            2.666177610399279,
            -1.4370342279918296,
            0.024433754802081566,
            -2.308689866503388,
            0.035184226660598256,
            0.8713155400000024,
            -2.776702429943918,
        ],
        [
            -0.4444149649678504,
            -1.742057830738788,
            -0.7666106443525141,
            -2.8292367744920264,
            -0.8331903566389359,
            1.176256282728209,
            -0.3498865529280606,
        ],
    ]
)


def make_branch_scene():
    model, ids = build_scene(Config(sensor_mode="oracle"))
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    data.qpos[ids.arm_qpos] = BRANCH
    data.ctrl[ids.arm_actuators] = BRANCH
    mujoco.mj_forward(model, data)
    solver = PandaIK(model, ids)
    solver.reset(data)
    return model, data, ids, solver, MotionPlanner(model, ids)


def test_joint_home_executes_checked_line_without_stopping_at_sampling_knots():
    model, data, ids, solver, planner = make_branch_scene()
    plan = planner.plan_home(data, 0)
    assert plan.verified and plan.execution_kind == "joint"
    assert len(plan.waypoints) == 1
    assert plan.checked_samples >= 300
    assert plan.minimum_clearance >= 0.008
    # The arm can already have the same tool pose on the wrong joint branch.
    assert not plan.can_advance(0, plan.waypoints[0], np.zeros(3))
    first, last = plan.joint_paths[0][0], plan.joint_paths[0][-1]
    delta = last - first
    axis = int(np.argmax(np.abs(delta)))
    fractions = (plan.joint_paths[0][:, axis] - first[axis]) / delta[axis]
    np.testing.assert_allclose(
        plan.joint_paths[0], first + fractions[:, None] * delta, atol=1e-14
    )
    commands, cartesian, actual = [], [], []
    reached_at = None
    for _ in range(850):
        solver.command(data, 0, plan.waypoints[0], motion_plan=plan)
        solver.command(data, 1, data.site_xpos[ids.ee_sites[1]], hold=True)
        commands.append(solver.q_targets[0].copy())
        cartesian.append(solver.cart_targets[0].copy())
        mujoco.mj_step(model, data)
        actual.append(data.qpos[ids.arm_qpos[0]].copy())
        assert not any(
            contact.dist < -0.0005
            and (
                ids.robot_geom_arm[contact.geom1] >= 0
                or ids.robot_geom_arm[contact.geom2] >= 0
            )
            for contact in data.contact
        )
        if reached_at is None and plan.can_advance(
            0,
            data.site_xpos[ids.ee_sites[0]],
            solver.cart_velocities[0],
            joint_positions=data.qpos[ids.arm_qpos[0]],
        ):
            reached_at = data.time
    assert reached_at is not None and reached_at < 4.0  # v3: 10.89 s
    dt = model.opt.timestep
    velocity = np.diff(commands, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    assert np.max(np.abs(velocity)) <= solver.max_joint_speed + 1e-8
    assert np.max(np.abs(acceleration)) <= solver.max_joint_acceleration + 1e-8
    cartesian_velocity = np.diff(cartesian, axis=0) / dt
    cartesian_acceleration = np.diff(cartesian_velocity, axis=0) / dt
    assert (
        np.max(np.linalg.norm(cartesian_velocity, axis=1)) < solver.max_cartesian_speed
    )
    assert (
        np.max(np.linalg.norm(cartesian_acceleration, axis=1))
        < solver.max_cartesian_acceleration
    )
    # One speed peak instead of sixty stop/start events, with a zero-speed tail.
    moving_velocity = np.abs(velocity[:, axis])
    peak = int(np.argmax(moving_velocity))
    assert np.all(np.diff(moving_velocity[: peak + 1]) >= -1e-9)
    assert np.all(np.diff(moving_velocity[peak:]) <= 1e-9)
    np.testing.assert_allclose(velocity[-10:], 0, atol=1e-10)
    actual = np.asarray(actual)
    projected = np.clip((actual - first) @ delta / (delta @ delta), 0, 1)
    cross_track = actual - (first + projected[:, None] * delta)
    assert np.max(np.abs(cross_track)) < 0.005
    assert np.max(np.abs(actual[-1] - ids.home_qpos[0])) < 0.0002


def test_joint_path_hold_cancels_clock_without_disturbing_the_other_arm():
    model, data, ids, solver, planner = make_branch_scene()
    plan = planner.plan_home(data, 0)
    for _ in range(100):
        solver.command(data, 0, plan.waypoints[0], motion_plan=plan)
        solver.command(data, 1, data.site_xpos[ids.ee_sites[1]], hold=True)
        mujoco.mj_step(model, data)
    other = (solver.q_targets[1].copy(), data.ctrl[ids.arm_actuators[1]].copy())
    for _ in range(100):
        solver.command(data, 0, data.site_xpos[ids.ee_sites[0]], hold=True)
        mujoco.mj_step(model, data)
    assert solver._joint_plan[0] is None
    np.testing.assert_array_equal(solver.q_targets[1], other[0])
    np.testing.assert_array_equal(data.ctrl[ids.arm_actuators[1]], other[1])
    fresh = planner.plan_home(data, 0)
    assert fresh.verified
    before = solver.q_targets[0].copy()
    solver.command(data, 0, fresh.waypoints[0], motion_plan=fresh)
    assert solver._joint_elapsed[0] == model.opt.timestep
    assert (
        np.max(np.abs(solver.q_targets[0] - before))
        <= solver.max_joint_speed * model.opt.timestep
    )


def test_mesh_inertial_frame_is_not_inherited_by_runtime_box_proxies():
    model, data, ids, _, planner = make_branch_scene()
    original_frame = model.geom_sameframe.copy()
    original_position = model.geom_pos.copy()
    original_rotation = model.geom_quat.copy()
    # At least one source slot is a compiled polygon mesh whose frame differs
    # from the free body's geometric frame, exercising MuJoCo's optimization.
    source = [planner.proxy_geom, *planner.bin_proxy_geoms]
    assert any(
        model.geom_sameframe[g] == mujoco.mjtSameFrame.mjSAMEFRAME_INERTIA
        for g in source
    )
    plan = planner.plan(
        data, 0, ids.home_ee[0], loaded=True, object_size=[0.04, 0.04, 0.04]
    )
    assert plan.verified
    expected_payload = planner.scratch.site_xpos[ids.ee_sites[0]] - [
        0,
        0,
        planner._active_grasp_offset,
    ]
    np.testing.assert_allclose(
        planner.scratch.geom_xpos[planner.proxy_geom], expected_payload, atol=1e-12
    )
    np.testing.assert_allclose(
        planner.scratch.geom_xpos[planner.bin_proxy_geoms],
        planner.bin_proxy_centers,
        atol=1e-12,
    )
    for qpos in ids.cube_qpos:
        data.qpos[qpos : qpos + 7] = np.nan
    repeated = planner.plan(
        data, 0, ids.home_ee[0], loaded=True, object_size=[0.04] * 3
    )
    assert repeated.verified == plan.verified
    np.testing.assert_allclose(
        planner.scratch.geom_xpos[planner.proxy_geom], expected_payload, atol=1e-12
    )
    np.testing.assert_array_equal(model.geom_sameframe, original_frame)
    np.testing.assert_array_equal(model.geom_pos, original_position)
    np.testing.assert_array_equal(model.geom_quat, original_rotation)
