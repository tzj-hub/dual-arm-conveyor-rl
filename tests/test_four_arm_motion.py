"""All-arm servo independence and non-partner collision checking for four Pandas."""

from itertools import combinations
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.ik import PandaIK, downward_quaternion
from dual_arm_conveyor.model import build_scene, reset_robot_pose
from dual_arm_conveyor.motion import MotionPlanner


@pytest.fixture
def scene():
    model, ids = build_scene(
        Config(strategy="adaptive", num_arms=4, sensor_mode="oracle")
    )
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    return model, data, ids


def test_each_arm_servo_owns_only_its_actuators_and_does_not_edit_physics(scene):
    model, data, ids = scene
    servo = PandaIK(model, ids, adaptive=True)
    np.testing.assert_array_equal(servo.q_targets, ids.home_qpos)
    servo.reset(data)
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
    assert servo.cart_targets.shape == servo.cart_velocities.shape == (4, 3)
    assert servo.q_targets.shape == (4, 7)
    for arm in range(4):
        np.testing.assert_allclose(
            servo.default_quats[arm],
            downward_quaternion(np.pi / 2 if arm % 2 == 0 else -np.pi / 2),
        )
        previous = data.ctrl.copy()
        goal = ids.home_ee[arm] + [0.0, 0.0, 0.02]
        servo.command(data, arm, goal, feedforward=[0.0, 0.0, 0.01], gripper=0.03)
        others = [other for other in range(4) if other != arm]
        np.testing.assert_array_equal(
            data.ctrl[ids.arm_actuators[others]], previous[ids.arm_actuators[others]]
        )
        np.testing.assert_array_equal(
            data.ctrl[ids.gripper_actuators[others]],
            previous[ids.gripper_actuators[others]],
        )
        assert np.isfinite(data.ctrl[ids.arm_actuators[arm]]).all()
        assert (
            data.ctrl[ids.gripper_actuators[arm]] < previous[ids.gripper_actuators[arm]]
        )
        servo.command(data, arm, goal, hold=True)
    np.testing.assert_array_equal(data.qpos, initial_qpos)
    np.testing.assert_array_equal(data.qvel, initial_qvel)
    assert servo._holding.tolist() == [True] * 4
    servo.reset(data)
    assert not servo._holding.any()
    assert servo._joint_plan == servo.last_joint_reference == [None] * 4


def test_all_four_arms_support_direct_home_and_escape_plans_without_live_mutation(
    scene,
):
    model, data, ids = scene
    planner = MotionPlanner(model, ids)
    before = (
        data.qpos.copy(),
        data.qvel.copy(),
        data.ctrl.copy(),
        model.geom_size.copy(),
    )
    for arm in range(4):
        goal = ids.home_ee[arm] + [0.0, 0.0, 0.04]
        routes = (
            planner.plan(data, arm, goal),
            planner.plan_home(data, arm),
            planner.plan_escape(data, arm, goal),
        )
        for route in routes:
            assert route.verified, (arm, route.reason)
            assert route.checked_samples > 0
        for route in routes[1:]:
            assert route.start_joint_positions.shape == (4, 7)
        # The stored static scene includes every other robot, not only arm^1.
        encountered = {
            int(ids.robot_geom_arm[geom])
            for pair in planner._pairs[arm]
            for geom in pair
            if ids.robot_geom_arm[geom] >= 0
        }
        assert encountered == {0, 1, 2, 3}
    for expected, actual in zip(
        before, (data.qpos, data.qvel, data.ctrl, model.geom_size)
    ):
        np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize("arm,other", [(0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (3, 1)])
def test_planner_rejects_real_collision_with_nonpartner_arm(scene, arm, other):
    model, data, ids = scene
    own_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"r{arm}/link0")
    other_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"r{other}/link0")
    # A deliberately invalid fixture co-locates a nonpartner robot. A planner
    # checking only the local north/south partner would incorrectly permit it.
    model.body_pos[other_base] = model.body_pos[own_base]
    model.body_quat[other_base] = model.body_quat[own_base]
    data.qpos[ids.arm_qpos[other]] = data.qpos[ids.arm_qpos[arm]]
    mujoco.mj_forward(model, data)
    planner = MotionPlanner(model, ids)
    route = planner.plan(data, arm, data.site_xpos[ids.ee_sites[arm]] + [0, 0, 0.02])
    assert not route.verified and not route.waypoints
    collision_geoms = [
        int(name.removeprefix("geom#"))
        if name.startswith("geom#")
        else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in route.reason.split(": ", 1)[1].split(" / ")
    ]
    assert {int(ids.robot_geom_arm[geom]) for geom in collision_geoms} == {arm, other}


@pytest.mark.parametrize("near_pair", list(combinations(range(4), 2)))
def test_interarm_clearance_finds_the_closest_of_all_six_arm_pairs(near_pair):
    # Exact simple geometry isolates all-arm measurement from Panda IK/fixture
    # layout. There is precisely one colliding pair, including same-side pairs.
    positions = np.arange(4, dtype=float) * 2.0
    positions[near_pair[1]] = positions[near_pair[0]] + 0.09
    bodies = "".join(
        f'<body pos="{position} 0 0"><geom type="sphere" size=".05"/><site name="ee{arm}"/></body>'
        for arm, position in enumerate(positions)
    )
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{bodies}</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = SimpleNamespace(
        ee_sites=np.arange(4),
        home_ee=data.site_xpos.copy(),
        home_qpos=np.zeros((4, 7)),
        robot_geom_arm=np.arange(4),
    )
    servo = PandaIK(model, ids)
    assert servo.interarm_clearance(data) == pytest.approx(-0.01, abs=1e-8)
    assert set(servo.last_clearance_pair) == set(near_pair)


@pytest.mark.parametrize("invalid", [-1, 4, 1.5])
def test_all_planning_entrypoints_reject_invalid_arm_indices(scene, invalid):
    model, data, ids = scene
    planner = MotionPlanner(model, ids)
    with pytest.raises(ValueError, match="arm"):
        planner.plan(data, invalid, ids.home_ee[0])
    with pytest.raises(ValueError, match="arm"):
        planner.plan_home(data, invalid)
    with pytest.raises(ValueError, match="arm"):
        planner.plan_escape(data, invalid, ids.home_ee[0])
