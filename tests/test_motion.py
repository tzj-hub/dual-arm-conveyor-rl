"""Numerical route checking with the actual Panda and scene collision meshes."""

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.ik import PandaIK, downward_quaternion
from dual_arm_conveyor.model import build_scene, reset_robot_pose
from dual_arm_conveyor.motion import MotionPlanner, path_length


@pytest.fixture
def scene():
    model, ids = build_scene(Config(sensor_mode="oracle"))
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    return model, data, ids


def test_direct_shortcut_removes_home_detour_without_changing_live_state(scene):
    model, data, ids = scene
    solver = PandaIK(model, ids)
    above_bin = np.array([0.0, -0.33, 0.355])
    data.qpos[ids.arm_qpos[0]] = solver.solve_pose(data, 0, above_bin)
    mujoco.mj_forward(model, data)
    before = [
        data.qpos.copy(),
        data.ctrl.copy(),
        model.geom_type.copy(),
        model.geom_size.copy(),
    ]
    planner = MotionPlanner(model, ids)
    target = np.array([-0.2, 0.0, 0.38])
    route = planner.plan(data, 0, target, via=[ids.home_ee[0]])
    assert route.verified and route.reason == "direct"
    assert len(route.waypoints) == 1 and route.checked_samples > 5
    assert route.length < path_length(above_bin, [ids.home_ee[0], target]) * 0.65
    for expected, actual in zip(
        before, (data.qpos, data.ctrl, model.geom_type, model.geom_size)
    ):
        np.testing.assert_array_equal(expected, actual)


def test_held_envelope_rejects_own_link_collision_that_empty_gripper_clears(scene):
    model, data, ids = scene
    solver = PandaIK(model, ids)
    data.qpos[ids.arm_qpos[0]] = solver.solve_pose(data, 0, [-0.28, 0.0, 0.42])
    mujoco.mj_forward(model, data)
    planner = MotionPlanner(model, ids)
    target = [-0.28, -0.28, 0.40]
    empty = planner.plan(data, 0, target)
    loaded = planner.plan(data, 0, target, loaded=True, object_size=[0.06] * 3)
    assert empty.verified
    assert not loaded.verified and not loaded.waypoints
    assert "cube_geom_0" in loaded.reason
    # Raising the transit goal clears the same proximal link with the payload.
    raised = planner.lowest_safe_goal(
        data, 0, target[:2], [0.40, 0.48, 0.53], loaded=True, object_size=[0.06] * 3
    )
    assert raised.verified
    assert raised.waypoints[-1][2] == pytest.approx(0.48)


def test_rejected_direct_path_uses_checked_corridor_instead(scene):
    model, data, ids = scene
    # Move one known fixture into a direct route, leaving the endpoint clear.
    obstacle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "binwall_0_0")
    model.geom_pos[obstacle] = [-0.08, -0.33, 0.43]
    model.geom_size[obstacle] = [0.02, 0.035, 0.025]
    model.geom_rbound[obstacle] = 0.05
    mujoco.mj_forward(model, data)
    planner = MotionPlanner(model, ids)
    goal = [0.065, -0.33, 0.355]
    direct = planner.plan(data, 0, goal)
    assert not direct.verified and "binwall_0_0" in direct.reason
    near_belt = planner.plan(data, 0, goal, near_belt=True)
    assert not near_belt.verified and "binwall_0_0" in near_belt.reason
    route = planner.plan(data, 0, goal, via=[[-0.3, -0.1, 0.55], [0.065, -0.1, 0.55]])
    assert route.verified and route.reason.startswith("corridor:")
    assert len(route.waypoints) == 2
    assert route.length > route.direct_length
    assert route.minimum_clearance >= planner.margin


def test_hidden_object_poses_cannot_change_a_plan(scene):
    model, data, ids = scene
    planner = MotionPlanner(model, ids)
    goal = [0.0, -0.33, 0.355]
    original = planner.plan(data, 0, goal, loaded=True, object_size=[0.05] * 3)
    for q in ids.cube_qpos:
        data.qpos[q : q + 7] = np.nan
    altered = planner.plan(data, 0, goal, loaded=True, object_size=[0.05] * 3)
    assert original.verified and altered.verified
    assert original.length == altered.length
    assert original.checked_samples == altered.checked_samples
    np.testing.assert_array_equal(original.waypoints, altered.waypoints)


def test_unreachable_goal_returns_no_executable_route(scene):
    model, data, ids = scene
    planner = MotionPlanner(model, ids)
    route = planner.plan(data, 0, [3.0, 0.0, 0.5])
    assert not route.verified and route.waypoints == ()
    assert "IK residual" in route.reason


def test_checked_orientation_schedule_avoids_low_bin_self_collision(scene):
    model, data, ids = scene
    solver = PandaIK(model, ids)
    start = [-0.09, -0.375, 0.317]
    start_quaternion = downward_quaternion(0.8606)
    data.qpos[ids.arm_qpos[0]] = solver.solve_pose(data, 0, start, start_quaternion)
    data.ctrl[ids.arm_actuators[0]] = data.qpos[ids.arm_qpos[0]]
    model.geom_contype[ids.cube_geoms] = 0
    model.geom_conaffinity[ids.cube_geoms] = 0
    mujoco.mj_forward(model, data)
    solver.reset(data)
    planner = MotionPlanner(model, ids)
    final_quaternion = downward_quaternion(np.pi / 2)
    route = planner.plan(data, 0, [-0.5, 0.053, 0.35], quaternion=final_quaternion)
    assert route.verified and len(route.waypoints) == 1
    np.testing.assert_allclose(
        route.orientation_at(0, route.start_position),
        route.start_quaternion,
        atol=1e-12,
    )
    midpoint = (route.start_position + route.waypoints[0]) / 2
    assert abs(np.dot(route.orientation_at(0, midpoint), final_quaternion)) < 0.999
    # Regression: immediately sending the final yaw here made the hand sweep
    # through link1 for 12 physics steps, despite the Cartesian route passing.
    for _ in range(600):
        ee = data.site_xpos[ids.ee_sites[0]]
        solver.command(
            data,
            0,
            route.waypoints[0],
            target_quat=route.orientation_at(0, ee),
            motion_plan=route,
            waypoint_index=0,
        )
        solver.command(data, 1, ids.home_ee[1])
        mujoco.mj_step(model, data)
        assert not any(
            contact.dist < -0.0005
            and (
                ids.robot_geom_arm[contact.geom1] >= 0
                or ids.robot_geom_arm[contact.geom2] >= 0
            )
            for contact in data.contact
        )
    assert np.linalg.norm(data.site_xpos[ids.ee_sites[0]] - route.waypoints[0]) < 0.003
    np.testing.assert_allclose(
        solver.last_joint_reference[0], route.joint_paths[0][-1], atol=0.002
    )


def test_corner_advance_requires_braking_distance_inside_clearance_margin(scene):
    model, data, ids = scene
    planner = MotionPlanner(model, ids)
    route = planner.plan(data, 0, [0.0, -0.33, 0.355])
    assert route.verified
    end = route.waypoints[0]
    assert not route.can_advance(0, end, [0.2, 0.0, 0.0])
    assert not route.can_advance(0, end + [0.007, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert route.can_advance(0, end + [0.002, 0.0, 0.0], [0.02, 0.0, 0.0])


def test_grasp_palm_clearance_allows_calibrated_gap_but_rejects_penetration(scene):
    model, data, ids = scene
    planner = MotionPlanner(model, ids, grasp_offset=-0.009)
    goal = ids.home_ee[0] + [0.0, 0.02, 0.0]
    normal = planner.plan(data, 0, goal, loaded=True, object_size=[0.06] * 3)
    oversized = planner.plan(data, 0, goal, loaded=True, object_size=[0.08] * 3)
    assert normal.verified
    assert 0.002 <= normal.minimum_clearance < 0.008
    assert not oversized.verified and oversized.minimum_clearance < 0.0


def test_joint_path_preserves_the_checked_redundancy_branch(scene):
    model, data, ids = scene
    planner = MotionPlanner(model, ids)
    route = planner.plan(data, 0, [0.0, -0.33, 0.355])
    assert route.verified
    joints = route.joint_paths[0]
    assert joints.ndim == 2 and joints.shape[1] == 7 and len(joints) > 5
    np.testing.assert_allclose(
        route.joint_reference_at(0, route.start_position), data.qpos[ids.arm_qpos[0]]
    )
    np.testing.assert_array_equal(
        route.joint_reference_at(0, route.waypoints[0]), joints[-1]
    )
    for index in range(len(joints)):
        fraction = index / (len(joints) - 1)
        position = route.start_position + fraction * (
            route.waypoints[0] - route.start_position
        )
        np.testing.assert_allclose(
            route.joint_reference_at(0, position), joints[index], atol=1e-12
        )


@pytest.mark.parametrize("execution_z", [0.210, 0.2135])
def test_near_belt_allows_only_positive_finger_approach_clearance(scene, execution_z):
    model, data, ids = scene
    solver = PandaIK(model, ids)
    data.qpos[ids.arm_qpos[0]] = solver.solve_pose(data, 0, [-0.45, 0.0, 0.35])
    mujoco.mj_forward(model, data)
    planner = MotionPlanner(model, ids)
    goal = [-0.45, 0.0, 0.210]
    strict = planner.plan(data, 0, goal)
    approach = planner.plan(data, 0, goal, near_belt=True)
    too_close = planner.plan(data, 0, [-0.45, 0.0, 0.208], near_belt=True)
    penetrating = planner.plan(data, 0, [-0.45, 0.0, 0.205], near_belt=True)
    assert not strict.verified and "belt" in strict.reason
    assert approach.verified and 0.001 <= approach.minimum_clearance < 0.008
    assert not too_close.verified and 0 < too_close.minimum_clearance < 0.001
    assert not penetrating.verified and penetrating.minimum_clearance < 0.0
    # The exception belongs to this plan, not subsequent unrestricted motion.
    assert not planner.plan(data, 0, goal).verified
    # Cover both the lowest task grasp and a lower positive-clearance goal;
    # braking must keep the live servo from overrunning the verified endpoint.
    execution_goal = [-0.45, 0.0, execution_z]
    execution_plan = planner.plan(data, 0, execution_goal, near_belt=True)
    assert execution_plan.verified
    model.geom_contype[ids.cube_geoms] = 0
    model.geom_conaffinity[ids.cube_geoms] = 0
    data.ctrl[ids.arm_actuators[0]] = data.qpos[ids.arm_qpos[0]]
    mujoco.mj_forward(model, data)
    solver.reset(data)
    for _ in range(400):
        solver.command(
            data, 0, execution_goal, motion_plan=execution_plan, waypoint_index=0
        )
        solver.command(data, 1, ids.home_ee[1])
        mujoco.mj_step(model, data)
        assert not any(
            contact.dist < -0.0001
            and (
                ids.robot_geom_arm[contact.geom1] >= 0
                or ids.robot_geom_arm[contact.geom2] >= 0
            )
            for contact in data.contact
        )
    assert np.linalg.norm(data.site_xpos[ids.ee_sites[0]] - execution_goal) < 0.003


def test_near_belt_does_not_relax_payload_or_proximal_link_clearance(scene):
    model, data, ids = scene
    solver = PandaIK(model, ids)
    data.qpos[ids.arm_qpos[0]] = solver.solve_pose(data, 0, [-0.45, 0.0, 0.35])
    mujoco.mj_forward(model, data)
    planner = MotionPlanner(model, ids)
    payload = planner.plan(
        data,
        0,
        [-0.45, 0.0, 0.2175],
        loaded=True,
        object_size=[0.035] * 3,
        near_belt=True,
    )
    assert not payload.verified
    assert "cube_geom_0" in payload.reason and "belt" in payload.reason
    assert 0.001 < payload.minimum_clearance < 0.008
    data.qpos[ids.arm_qpos[0]] = solver.solve_pose(data, 0, [-0.28, 0.0, 0.42])
    mujoco.mj_forward(model, data)
    proximal = planner.plan(
        data,
        0,
        [-0.28, -0.28, 0.40],
        loaded=True,
        object_size=[0.06] * 3,
        near_belt=True,
    )
    assert not proximal.verified and "cube_geom_0" in proximal.reason


def test_proxy_preserves_estimated_height_and_per_plan_offset(scene):
    model, data, ids = scene
    planner = MotionPlanner(model, ids, grasp_offset=-0.009)
    goal = ids.home_ee[0] + [0.0, 0.02, 0.0]
    wide_short = planner.plan(
        data, 0, goal, loaded=True, object_size=[0.060, 0.050, 0.035]
    )
    assert wide_short.verified and wide_short.minimum_clearance >= 0.008
    np.testing.assert_allclose(
        planner.model.geom_size[planner.proxy_geom], [0.032, 0.027, 0.0185]
    )
    assert planner.model.geom_rbound[planner.proxy_geom] == pytest.approx(
        np.linalg.norm([0.032, 0.027, 0.0185])
    )
    tall = planner.plan(data, 0, goal, loaded=True, object_size=[0.060, 0.050, 0.060])
    assert tall.verified and 0.002 <= tall.minimum_clearance < 0.008
    adjusted = planner.plan(
        data, 0, goal, loaded=True, object_size=[0.060, 0.050, 0.060], grasp_offset=0.0
    )
    assert adjusted.verified and adjusted.minimum_clearance >= 0.008
    np.testing.assert_allclose(
        planner.scratch.qpos[planner.proxy_qpos : planner.proxy_qpos + 3],
        planner.scratch.site_xpos[ids.ee_sites[0]],
        atol=1e-12,
    )
    assert planner.grasp_offset == -0.009
    repeated = planner.plan(
        data, 0, goal, loaded=True, object_size=[0.060, 0.050, 0.060]
    )
    assert repeated.minimum_clearance == pytest.approx(tall.minimum_clearance)
    assert planner._active_grasp_offset == -0.009
    with pytest.raises(ValueError, match="grasp_offset"):
        planner.plan(data, 0, goal, grasp_offset=float("nan"))
    with pytest.raises(ValueError, match="object_size"):
        planner.plan(data, 0, goal, object_size=[0.05, 0.05, -0.01])


def test_bin_interior_proxies_are_fixed_and_independent_of_hidden_objects(scene):
    model, data, ids = scene
    original_size = model.geom_size.copy()
    original_types = model.geom_type.copy()
    original_contacts = model.geom_contype.copy()
    planner = MotionPlanner(model, ids)
    goal = [0.0, -0.33, 0.355]
    first = planner.plan(data, 0, goal, loaded=True, object_size=[0.04] * 3)
    assert first.verified
    assert planner.proxy_geom not in planner.bin_proxy_geoms
    expected_centers = np.column_stack(
        (ids.bin_centers[:, :2], np.full(2, (ids.bin_floor_z + ids.bin_rim_z) / 2))
    )
    expected_size = np.r_[ids.bin_half_sizes, (ids.bin_rim_z - ids.bin_floor_z) / 2]
    np.testing.assert_allclose(
        planner.scratch.geom_xpos[planner.bin_proxy_geoms], expected_centers
    )
    np.testing.assert_allclose(
        planner.model.geom_size[planner.bin_proxy_geoms], np.tile(expected_size, (2, 1))
    )
    for qpos in ids.cube_qpos:
        data.qpos[qpos : qpos + 7] = np.nan
    repeated = planner.plan(data, 0, goal, loaded=True, object_size=[0.04] * 3)
    assert repeated.verified and repeated.checked_samples == first.checked_samples
    np.testing.assert_array_equal(repeated.waypoints, first.waypoints)
    np.testing.assert_allclose(
        planner.scratch.geom_xpos[planner.bin_proxy_geoms], expected_centers
    )
    np.testing.assert_array_equal(model.geom_size, original_size)
    np.testing.assert_array_equal(model.geom_type, original_types)
    np.testing.assert_array_equal(model.geom_contype, original_contacts)


@pytest.mark.parametrize("final_x", [-0.233, -0.19])
def test_filled_bin_elbow_regression_rejects_original_branch(scene, final_x):
    model, data, ids = scene
    # Encoder snapshot before the former elbow/sorted-object collision at
    # t=134.865 s (catalog 0 / stream 0). No object state is needed to reproduce
    # the unsafe path into the known bin interior.
    data.qpos[ids.arm_qpos] = [
        [
            1.5601346920593886,
            -1.4630273948336336,
            -0.9812474486058028,
            -2.421723411902887,
            -1.0383715052436182,
            1.2828709158364913,
            -2.2564959231343926,
        ],
        [
            -0.4280414525787363,
            -1.2743584633772715,
            -0.9427289004340811,
            -2.6186097164187783,
            -0.8853659675069909,
            1.5431078216011662,
            1.780789113342792,
        ],
    ]
    data.qpos[ids.finger_qpos] = (
        np.array([0.07999667282055781, 0.07999999990367065])[:, None] / 2
    )
    mujoco.mj_forward(model, data)
    planner = MotionPlanner(model, ids)
    goal = [final_x, 0.042, 0.21468541359126153]
    yaw = -6.97473169930752
    original = planner.plan(
        data, 0, goal, quaternion=downward_quaternion(yaw), near_belt=True
    )
    alternate = planner.plan(
        data, 0, goal, quaternion=downward_quaternion(yaw + np.pi), near_belt=True
    )
    assert not original.verified and "cube_geom_1" in original.reason
    assert original.minimum_clearance < 0
    assert alternate.verified


def test_held_payload_keeps_full_margin_from_bin_interior(scene):
    model, data, ids = scene
    planner = MotionPlanner(model, ids)
    goal = [0.0, -0.33, 0.29]
    assert planner.plan(data, 0, goal, near_belt=True).verified
    loaded = planner.plan(
        data,
        0,
        goal,
        loaded=True,
        object_size=[0.06] * 3,
        grasp_offset=0.0,
        near_belt=True,
    )
    assert not loaded.verified
    assert "cube_geom_0 / cube_geom_1" in loaded.reason
    assert 0 < loaded.minimum_clearance < 0.008


# Robot encoders and controller references captured from the real oracle
# catalog 0 / stream 50000 run, during the persistent recovery stall.
# Source env e41047947f2e, IK 4b035b61e2ab, model 7e50f965a190.
# Objects are deliberately excluded: neither an escape nor this regression
# needs object ground-truth poses to reproduce the robot/fixture deadlock.
_ESCAPE_STATES = {
    "46.3": {
        "arm_qpos": [
            [
                0.76641437264407,
                -1.7259839826743013,
                -0.8980339358609989,
                -2.54777991866161,
                -0.7008947789865864,
                0.2608121571974881,
                -0.7473273771456322,
            ],
            [
                -0.23789643312827535,
                -1.7477999999999998,
                -0.8763262757450271,
                -2.701951187866993,
                -0.9925143164922732,
                1.126039816680225,
                0.17678628818389744,
            ],
        ],
        "arm_qvel": [
            [
                -0.0018658064080153873,
                0.07027289787617946,
                -0.31253670682157053,
                0.2711348441333793,
                0.679502177444466,
                -0.34198364571596157,
                1.6123063090897032,
            ],
            [
                -9.032177327552236e-14,
                -2.6094254668189916e-14,
                -6.109015664950165e-14,
                -1.195233291790576e-13,
                1.6043637498223844e-13,
                -6.591281821675126e-14,
                -3.578169747765576e-14,
            ],
        ],
        "finger_qpos": [
            [
                0.04000671630378589,
                0.03999322360346091,
            ],
            [
                0.0400000010072682,
                0.039999990319125316,
            ],
        ],
        "finger_qvel": [
            [
                -0.0001565750847871549,
                7.437986019343651e-05,
            ],
            [
                1.4543643984302838e-12,
                -1.1975177763632606e-11,
            ],
        ],
        "gate_qpos": [
            8.108806039943615e-15,
            -4.3407659635719927e-08,
        ],
        "gate_qvel": [
            -5.610163125562231e-13,
            -1.6798459758523295e-06,
        ],
        "gate_ctrl": [
            0.0,
            0.0,
        ],
        "q_targets": [
            [
                0.8117457961255052,
                -1.7478,
                -0.9131166327154526,
                -2.5338332735367093,
                -0.6650891197262244,
                0.24294429066705664,
                -0.6630094413559573,
            ],
            [
                -0.23789643312827796,
                -1.7478,
                -0.8763262757450303,
                -2.7019511878669955,
                -0.9925143164922752,
                1.1260398166802288,
                0.17678628818389586,
            ],
        ],
        "cart_targets": [
            [
                -0.044269192865011145,
                -0.30398441688229333,
                0.34399064520613704,
            ],
            [
                -0.00020211137663174385,
                0.36507946736320773,
                0.3171926822256503,
            ],
        ],
        "cart_velocities": [
            [
                0.0071222942152857766,
                -0.060539176266012906,
                0.03089830996591214,
            ],
            [
                -7.356311740314148e-16,
                4.9960036108132044e-15,
                -1.554312234475219e-14,
            ],
        ],
        "gripper_targets": [
            0.04,
            0.04,
        ],
    },
    "50.0": {
        "arm_qpos": [
            [
                0.8277977237216556,
                -1.4981114795073096,
                -1.6218014549355182,
                -2.8636626833023793,
                1.1283817099989977,
                0.5964949881649706,
                0.36067548895367973,
            ],
            [
                -0.23789643312828154,
                -1.7478000000000005,
                -0.8763262757450332,
                -2.7019511878669995,
                -0.9925143164922755,
                1.1260398166802295,
                0.1767862881838941,
            ],
        ],
        "arm_qvel": [
            [
                -0.0007583491333470812,
                -3.136497678975487e-05,
                0.0015519888187543233,
                -0.0017532150051344402,
                0.006391006178840813,
                0.04566839636866459,
                0.11800116241776769,
            ],
            [
                6.302162219277213e-14,
                6.247095110614272e-14,
                -8.103588393134452e-14,
                1.59162041996074e-13,
                -7.499254190652588e-13,
                4.868436093870195e-13,
                -1.5871136181660477e-14,
            ],
        ],
        "finger_qpos": [
            [
                0.04000008896047588,
                0.0399991056832577,
            ],
            [
                0.040000001010024766,
                0.03999999032020071,
            ],
        ],
        "finger_qvel": [
            [
                5.590806040796499e-06,
                -3.596445176632039e-06,
            ],
            [
                3.5896763431289205e-11,
                4.9324137540200944e-11,
            ],
        ],
        "gate_qpos": [
            1.9061847585814206e-14,
            9.049882933276488e-08,
        ],
        "gate_qvel": [
            2.0602285322622855e-13,
            1.6094390264406605e-05,
        ],
        "gate_ctrl": [
            0.0,
            0.0,
        ],
        "q_targets": [
            [
                0.823555719605767,
                -1.5030576029040128,
                -1.6253703910320871,
                -2.8653492759270582,
                1.1286990597605777,
                0.5988276316542472,
                0.3667084761985181,
            ],
            [
                -0.23789643312828115,
                -1.7478,
                -0.8763262757450334,
                -2.701951187866998,
                -0.9925143164922775,
                1.126039816680231,
                0.17678628818389377,
            ],
        ],
        "cart_targets": [
            [
                -0.1273549856573078,
                -0.2628016815463294,
                0.5051812290445461,
            ],
            [
                -0.00020211137663181345,
                0.36507946736320773,
                0.31719268222564934,
            ],
        ],
        "cart_velocities": [
            [
                -0.007266730418936107,
                0.006593079505636812,
                0.0058892551977363095,
            ],
            [
                -2.710505431213761e-18,
                4.9960036108132044e-15,
                -5.551115123125783e-15,
            ],
        ],
        "gripper_targets": [
            0.04,
            0.04,
        ],
    },
}


def _restore_escape_robot(scene, timestamp):
    model, data, ids = scene
    state = _ESCAPE_STATES[timestamp]
    data.qpos[ids.arm_qpos] = state["arm_qpos"]
    data.qvel[ids.arm_dofs] = state["arm_qvel"]
    data.qpos[ids.finger_qpos] = state["finger_qpos"]
    data.qvel[ids.finger_dofs] = state["finger_qvel"]
    data.qpos[ids.bin_gate_qpos] = state["gate_qpos"]
    data.qvel[ids.bin_gate_dofs] = state["gate_qvel"]
    data.ctrl[ids.arm_actuators] = state["q_targets"]
    data.ctrl[ids.gripper_actuators] = state["gripper_targets"]
    data.ctrl[ids.bin_gate_actuators] = state["gate_ctrl"]
    mujoco.mj_forward(model, data)
    solver = PandaIK(model, ids)
    solver.reset(data)
    for name in ("q_targets", "cart_targets", "cart_velocities", "gripper_targets"):
        getattr(solver, name)[:] = state[name]
    return solver


@pytest.mark.parametrize("timestamp", ["46.3", "50.0"])
def test_real_recovery_escape_restores_clearance_during_live_execution(
    scene, timestamp
):
    model, data, ids = scene
    solver = _restore_escape_robot(scene, timestamp)
    planner = MotionPlanner(model, ids)
    via = [[data.site_xpos[ids.ee_sites[0], 0], -0.18, 0.52]]
    blocked = planner.plan(data, 0, ids.home_ee[0], via=via)
    assert not blocked.verified and blocked.reason.startswith("initial clearance")
    before = [
        data.qpos.copy(),
        data.qvel.copy(),
        data.ctrl.copy(),
        model.geom_size.copy(),
        model.geom_contype.copy(),
    ]
    route = planner.plan_escape(data, 0, ids.home_ee[0], via=via)
    assert route.verified and route.reason.startswith("escape")
    if timestamp == "50.0":
        # The failed Cartesian trials must roll back their separation progress
        # before this alternative redundant-joint path starts at real encoders.
        assert route.reason == "escape joint home"
    assert route.escape_initial_clearances
    np.testing.assert_array_equal(route.start_joint_positions, data.qpos[ids.arm_qpos])
    for expected, actual in zip(
        before,
        (data.qpos, data.qvel, data.ctrl, model.geom_size, model.geom_contype),
    ):
        np.testing.assert_array_equal(actual, expected)
    # Escape is confined to this call: ordinary planning keeps its full margin.
    assert not planner.plan(data, 0, ids.home_ee[0], via=via).verified

    # Independently inspect every accepted joint interpolation. Existing
    # deficiencies can only improve; restored pairs never lose their margin.
    planner._initialize(data)
    progress = {
        (a, b): distance for a, b, distance, _ in route.escape_initial_clearances
    }
    requirements = {
        (a, b): margin for a, b, _, margin in route.escape_initial_clearances
    }
    for path in route.joint_paths:
        for previous, following in zip(path[:-1], path[1:]):
            steps = max(1, int(np.ceil(np.max(np.abs(following - previous)) / 0.01)))
            for alpha in np.linspace(0, 1, steps + 1)[1:]:
                planner.scratch.qpos[ids.arm_qpos[0]] = previous + alpha * (
                    following - previous
                )
                mujoco.mj_kinematics(planner.model, planner.scratch)
                for first, second in planner._pairs[0]:
                    pair = (int(first), int(second))
                    distance = mujoco.mj_geomDistance(
                        planner.model, planner.scratch, *pair, planner.margin, None
                    )
                    if pair in progress:
                        margin = requirements[pair]
                        assert distance >= progress[pair] - (
                            1e-6 if progress[pair] < margin else 0
                        )
                        progress[pair] = min(margin, max(progress[pair], distance))
                    else:
                        assert distance >= planner.margin
    assert all(progress[pair] >= margin for pair, margin in requirements.items())

    index = 0
    for _ in range(1600):
        ee = data.site_xpos[ids.ee_sites[0]]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[ids.ee_sites[0]])
        while index < len(route.waypoints) - 1 and route.can_advance(
            index,
            ee,
            solver.cart_velocities[0],
            quaternion=quaternion,
            joint_positions=data.qpos[ids.arm_qpos[0]],
        ):
            index += 1
        solver.command(
            data, 0, route.waypoints[index], motion_plan=route, waypoint_index=index
        )
        # The other arm holds its measured, planned-against joint position.
        data.qfrc_applied[ids.arm_dofs[1]] = data.qfrc_bias[ids.arm_dofs[1]]
        mujoco.mj_step(model, data)
        assert not any(
            contact.dist < -0.0005
            and (
                ids.robot_geom_arm[contact.geom1] >= 0
                or ids.robot_geom_arm[contact.geom2] >= 0
            )
            for contact in data.contact
        )
        if index == len(route.waypoints) - 1 and route.can_advance(
            index,
            data.site_xpos[ids.ee_sites[0]],
            solver.cart_velocities[0],
            joint_positions=data.qpos[ids.arm_qpos[0]],
        ):
            break
    else:
        pytest.fail("The real recovery state did not escape within eight seconds")
    planner._initialize(data)
    assert planner._clearance(0, False)[0]
    assert np.linalg.norm(data.site_xpos[ids.ee_sites[0]] - ids.home_ee[0]) < 0.004


def test_escape_rejects_deep_penetration_and_keeps_object_truth_out(scene):
    model, data, ids = scene
    _restore_escape_robot(scene, "50.0")
    planner = MotionPlanner(model, ids)
    original = planner.plan_escape(data, 0, ids.home_ee[0])
    for qpos in ids.cube_qpos:
        data.qpos[qpos : qpos + 7] = np.nan
    repeated = planner.plan_escape(data, 0, ids.home_ee[0])
    assert original.verified and repeated.verified
    assert original.reason == repeated.reason
    assert original.checked_samples == repeated.checked_samples
    np.testing.assert_array_equal(original.waypoints, repeated.waypoints)
    assert original.escape_initial_clearances == repeated.escape_initial_clearances
    # A one-centiradian measured shoulder change makes this penetration deeper
    # than the narrowly bounded contact permitted for an escape start.
    data.qpos[ids.arm_qpos[0, 0]] += 0.01
    rejected = planner.plan_escape(data, 0, ids.home_ee[0])
    assert not rejected.verified and rejected.waypoints == ()
    assert rejected.reason.startswith("escape initial penetration")
    assert rejected.minimum_clearance < -0.0005


def test_partial_escape_finishes_at_full_margin_and_checks_joint_arrival(scene):
    model, data, ids = scene
    _restore_escape_robot(scene, "50.0")
    planner = MotionPlanner(model, ids)
    route = planner.plan_escape(data, 0, [3.0, 0.0, 0.5])
    assert route.verified and route.reason == "escape partial"
    assert 0.029 <= route.length <= 0.081
    assert planner._clearance(0, False)[0]
    end = route.waypoints[-1]
    joints = route.joint_paths[-1][-1]
    assert route.can_advance(0, end, np.zeros(3), joint_positions=joints)
    assert not route.can_advance(
        0, end, np.zeros(3), joint_positions=joints + [0.03, 0, 0, 0, 0, 0, 0]
    )


def test_joint_home_recovers_calibrated_branch_when_tool_pose_is_already_home(scene):
    model, data, ids = scene
    solver = PandaIK(model, ids)
    # Returning from a bin with Cartesian IK reaches the same tool pose on a
    # different redundant branch. Its tiny Cartesian error cannot certify home.
    data.qpos[ids.arm_qpos[0]] = solver.solve_pose(data, 0, [0.0, -0.33, 0.355])
    mujoco.mj_forward(model, data)
    data.qpos[ids.arm_qpos[0]] = solver.solve_pose(data, 0, ids.home_ee[0])
    data.ctrl[ids.arm_actuators[0]] = data.qpos[ids.arm_qpos[0]]
    mujoco.mj_forward(model, data)
    solver.reset(data)
    assert np.linalg.norm(data.site_xpos[ids.ee_sites[0]] - ids.home_ee[0]) < 1e-5
    assert np.max(np.abs(data.qpos[ids.arm_qpos[0]] - ids.home_qpos[0])) > 0.2
    planner = MotionPlanner(model, ids)
    route = planner.plan_home(data, 0)
    assert route.verified and route.reason == "joint home"
    assert route.minimum_clearance >= planner.margin
    np.testing.assert_allclose(route.joint_paths[-1][-1], ids.home_qpos[0], atol=1e-14)
    np.testing.assert_allclose(route.waypoints[-1], ids.home_ee[0], atol=1e-12)
    assert not route.can_advance(
        0, route.waypoints[0], np.zeros(3), joint_positions=data.qpos[ids.arm_qpos[0]]
    )
    index = 0
    for _ in range(600):
        ee = data.site_xpos[ids.ee_sites[0]]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[ids.ee_sites[0]])
        if index < len(route.waypoints) - 1 and route.can_advance(
            index,
            ee,
            solver.cart_velocities[0],
            quaternion=quaternion,
            joint_positions=data.qpos[ids.arm_qpos[0]],
        ):
            index += 1
        solver.command(
            data, 0, route.waypoints[index], motion_plan=route, waypoint_index=index
        )
        solver.command(data, 1, ids.home_ee[1], hold=True)
        mujoco.mj_step(model, data)
        assert not any(
            contact.dist < -0.0005
            and (
                ids.robot_geom_arm[contact.geom1] >= 0
                or ids.robot_geom_arm[contact.geom2] >= 0
            )
            for contact in data.contact
        )
        if (
            index == len(route.waypoints) - 1
            and np.max(np.abs(data.qpos[ids.arm_qpos[0]] - ids.home_qpos[0])) < 0.01
        ):
            break
    else:
        pytest.fail("Cartesian home pose prevented recovery of the calibrated elbow")


def test_joint_home_is_strict_and_independent_of_hidden_object_poses(scene):
    model, data, ids = scene
    solver = PandaIK(model, ids)
    data.qpos[ids.arm_qpos[0]] = solver.solve_pose(data, 0, [0.0, -0.33, 0.355])
    mujoco.mj_forward(model, data)
    planner = MotionPlanner(model, ids)
    before = (
        data.qpos.copy(),
        data.qvel.copy(),
        data.ctrl.copy(),
        model.geom_size.copy(),
    )
    original = planner.plan_home(data, 0)
    assert original.verified
    for actual, expected in zip(
        (data.qpos, data.qvel, data.ctrl, model.geom_size), before
    ):
        np.testing.assert_array_equal(actual, expected)
    for qpos in ids.cube_qpos:
        data.qpos[qpos : qpos + 7] = np.nan
    repeated = planner.plan_home(data, 0)
    assert repeated.verified and original.checked_samples == repeated.checked_samples
    np.testing.assert_array_equal(original.waypoints, repeated.waypoints)
    # The older real near-wall failure needs an explicit escape; plan_home
    # cannot silently reuse that relaxation when asked for an ordinary path.
    data.qpos[ids.arm_qpos] = _ESCAPE_STATES["50.0"]["arm_qpos"]
    rejected = planner.plan_home(data, 0)
    assert not rejected.verified and rejected.waypoints == ()
    assert rejected.reason.startswith("joint home initial clearance")
    assert rejected.minimum_clearance < 0


def test_real_vision_home_branch_recovers_previously_unreachable_pick_routes(scene):
    model, data, ids = scene
    # Real quiescent encoder state at t=145.1 s, vision stream 50000/catalog 0.
    # The former controller had accepted home at t=142.1 s, despite ending
    # 3.56 rad away from its calibrated redundant configuration.
    data.qpos[ids.arm_qpos] = [
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
    data.qpos[ids.finger_qpos] = [
        [0.03999996594249961, 0.04000000355056592],
        [0.04000000446244682, 0.03999995714210194],
    ]
    data.ctrl[ids.arm_actuators] = data.qpos[ids.arm_qpos]
    mujoco.mj_forward(model, data)
    solver = PandaIK(model, ids)
    solver.reset(data)
    planner = MotionPlanner(model, ids)
    actual = np.empty(4)
    mujoco.mju_mat2Quat(actual, data.site_xmat[ids.ee_sites[0]])
    nominal = downward_quaternion(np.pi / 2)
    assert np.linalg.norm(data.site_xpos[ids.ee_sites[0]] - ids.home_ee[0]) < 0.001
    assert 2 * np.arccos(np.clip(abs(actual @ nominal), 0, 1)) < 0.04
    assert np.max(np.abs(data.qpos[ids.arm_qpos[0]] - ids.home_qpos[0])) > 3.5
    goals = [
        [-0.540405067538722, -0.029693575178599243, 0.3587068799224179],
        [-0.3840915162457661, -0.030143583139079753, 0.35864254808899193],
        [-0.2642112711821901, -0.028348392029147282, 0.35867643650551295],
    ]

    def pick_route(goal):
        return planner.plan(
            data,
            0,
            goal,
            quaternion=nominal,
            via=[ids.home_ee[0], [goal[0], -0.18, 0.48]],
        )

    assert all(not pick_route(goal).verified for goal in goals)
    home = planner.plan_home(data, 0)
    assert home.verified and home.minimum_clearance >= 0.008
    np.testing.assert_allclose(home.joint_paths[-1][-1], ids.home_qpos[0], atol=1e-14)

    def execute(route, maximum_steps):
        index = 0
        for _ in range(maximum_steps):
            ee = data.site_xpos[ids.ee_sites[0]]
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, data.site_xmat[ids.ee_sites[0]])
            while index < len(route.waypoints) - 1 and route.can_advance(
                index,
                ee,
                solver.cart_velocities[0],
                quaternion=quaternion,
                joint_positions=data.qpos[ids.arm_qpos[0]],
            ):
                index += 1
            solver.command(
                data, 0, route.waypoints[index], motion_plan=route, waypoint_index=index
            )
            solver.command(data, 1, data.site_xpos[ids.ee_sites[1]], hold=True)
            mujoco.mj_step(model, data)
            assert not any(
                contact.dist < -0.0005
                and (
                    ids.robot_geom_arm[contact.geom1] >= 0
                    or ids.robot_geom_arm[contact.geom2] >= 0
                )
                for contact in data.contact
            )
            if index == len(route.waypoints) - 1 and route.can_advance(
                index,
                data.site_xpos[ids.ee_sites[0]],
                solver.cart_velocities[0],
                joint_positions=data.qpos[ids.arm_qpos[0]],
            ):
                return
        pytest.fail("The measured redundant-branch regression failed to finish")

    execute(home, 2600)
    assert np.max(np.abs(data.qpos[ids.arm_qpos[0]] - ids.home_qpos[0])) < 0.02
    assert all(pick_route(goal).verified for goal in goals)
    execute(pick_route(goals[-1]), 400)
    assert np.linalg.norm(data.site_xpos[ids.ee_sites[0]] - goals[-1]) < 0.004
