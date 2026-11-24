"""Physical guards around contact withdrawal and off-belt preparation."""
import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.ik import downward_quaternion
from dual_arm_conveyor.motion import MotionPlan


@pytest.fixture
def env():
    e = ConveyorEnv(Config(sensor_mode="oracle", total_objects=5, pool_size=16,
                           num_arms=4, strategy="adaptive", motion_mode="optimized",
                           deep_bins=True, auto_unload=False, bin_capacity=24,
                           active_recovery=True, future_planning=True, safe_preposition=True))
    yield e
    e.close()


def low_pose(e, arm=2):
    point = np.array([0.55, 0.04, 0.23])
    joints = e.ik.solve_pose(e.data, arm, point, downward_quaternion(np.pi / 2))
    e.data.qpos[e.ids.arm_qpos[arm]] = joints
    e.data.ctrl[e.ids.arm_actuators[arm]] = joints
    mujoco.mj_forward(e.model, e.data)
    e.ik.reset(e.data)
    e._recover(arm)
    e.data.time = 0.7


@pytest.mark.parametrize("aperture,touch", [(0.06,[0.7,0]), (0.08,[0.7,0.8]), (0.08,[9,0])])
def test_withdrawal_never_moves_closed_bilaterally_loaded_or_overloaded_hand(env, monkeypatch, aperture, touch):
    low_pose(env)
    env.data.qpos[env.ids.finger_qpos[2]] = aperture / 2
    env.data.sensordata[env.ids.finger_touch_adr[2]] = touch

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsafe release condition attempted a motion plan")

    monkeypatch.setattr(env.planner, "plan", forbidden)
    monkeypatch.setattr(env.planner, "plan_home", forbidden)
    monkeypatch.setattr(env.planner, "plan_escape", forbidden)
    env._control()
    assert env.phase[2] == "recover" and env.hold_requested[2]
    assert not env.recovery_withdrawal_attempts.any()


def test_rejected_withdrawal_keeps_recovery_resources_and_never_uses_home_plan(env, monkeypatch):
    low_pose(env)
    env.data.qpos[env.ids.finger_qpos[2]] = 0.04
    env.data.sensordata[env.ids.finger_touch_adr[2]] = [0.7,0]
    env.bin_zone_owner[1,0] = 2
    failures = []

    def reject(*args, **kwargs):
        failures.append(kwargs)
        return MotionPlan((), False, "test obstacle", 0, 0, 1, 0, 0)

    def no_home(*args, **kwargs):
        raise AssertionError("Contact withdrawal attempted an unrestricted return")

    monkeypatch.setattr(env.planner, "plan", reject)
    monkeypatch.setattr(env.planner, "plan_home", no_home)
    env._control()
    assert len(failures) == 1
    assert env.phase[2] == "recover" and env.hold_requested[2]
    assert env.bin_zone_owner[1,0] == 2
    assert env.recovery_withdrawal_fault[2] == "route_not_certified"
    env.data.time = 2.3
    env._control()
    assert env.recovery_withdrawal_fault[2] == "withdrawal_time_limit"
    assert env.recovery_withdrawal_goal[2] is None
    assert env.bin_zone_owner[1,0] == 2
    env._control()
    assert env.recovery_withdrawal_attempts[2] == 1


def set_preposition(e, arm=0):
    e.future_preposition_goals = np.full((4,3), np.nan)
    e.future_preposition_yaws = np.full(4, np.nan)
    e.future_preposition_goals[arm] = [e.ids.home_ee[arm,0] + 0.03, -0.26, 0.40]
    e.future_preposition_yaws[arm] = np.pi / 2


def test_preposition_uses_real_checked_motion_and_counts_as_busy(env):
    set_preposition(env)
    target = env.future_preposition_goals[0].copy()
    initial = env.data.qpos[env.ids.arm_qpos[0]].copy()
    for _ in range(20):
        _, _, _, _, metrics = env.step(env.wait_action)
    assert np.linalg.norm(env.data.site_xpos[env.ids.ee_sites[0]] - target) < 0.004
    assert np.linalg.norm(env.data.qpos[env.ids.arm_qpos[0]] - initial) > 0.01
    assert env.target[0] == -1 and env.phase[0] == "idle"
    assert metrics["preposition"]["motion_seconds"][0] > 0.1
    processing = metrics["arm_utilization"]["cumulative"]["per_arm_seconds"][0]["processing"]
    assert processing == pytest.approx(metrics["preposition"]["motion_seconds"][0])
    assert metrics["robot_collision_steps"] == metrics["inter_arm_collision_steps"] == 0


def test_preposition_stops_when_partner_starts_moving_and_rejects_belt_goal(env):
    set_preposition(env)
    env.data.time = 0.1
    env._control()
    assert env.preposition_active[0]
    env.phase[1] = "return"
    assert env._safe_preposition_step(0) is None
    assert env._preposition_live_goal[0] is None
    env.phase[1] = "idle"
    env.future_preposition_goals[0,1] = 0.0
    assert env._safe_preposition_step(0) is None
    assert env.target[0] == -1


def test_preposition_completion_uses_measured_not_requested_orientation(env, monkeypatch):
    set_preposition(env)
    goal = env.future_preposition_goals[0].copy()
    actual_yaw = np.pi / 2 - 0.15
    joints = env.ik.solve_pose(env.data, 0, goal, downward_quaternion(actual_yaw))
    env.data.qpos[env.ids.arm_qpos[0]] = joints
    env.data.ctrl[env.ids.arm_actuators[0]] = joints
    mujoco.mj_forward(env.model, env.data)
    env.ik.reset(env.data)
    env.data.time = 0.1
    actual = np.empty(4)
    mujoco.mju_mat2Quat(actual, env.data.site_xmat[env.ids.ee_sites[0]])
    plan = env.planner.plan(env.data, 0, goal, quaternion=downward_quaternion(np.pi / 2))
    assert plan.verified
    checked = []
    original = MotionPlan.can_advance

    def record_check(self, index, position, velocity, **kwargs):
        checked.append(np.asarray(kwargs['quaternion']).copy())
        return original(self, index, position, velocity, **kwargs)

    monkeypatch.setattr(MotionPlan, 'can_advance', record_check)
    env._control()
    assert checked
    np.testing.assert_allclose(checked[-1], actual, atol=1e-12)
    assert env.preposition_active[0] and not env.hold_requested[0]


def test_release_withdrawal_keyword_cannot_bypass_ordinary_recovery_gate(env):
    low_pose(env)
    env.data.qpos[env.ids.finger_qpos[2]] = 0.02
    env.data.sensordata[env.ids.finger_touch_adr[2]] = [0.7, 0.8]
    _, ready = env._route_goal(2, env.ids.home_ee[2], False, release_withdrawal=True)
    assert not ready and env.hold_requested[2]
    assert env.routes[2] is None


def test_preposition_switch_waits_for_real_partner_braking_and_still_dwell(env):
    set_preposition(env)
    env.step(env.wait_action)
    assert env.preposition_active[0]
    env.future_preposition_goals[0] = np.nan
    env.future_preposition_yaws[0] = np.nan
    env.future_preposition_goals[1] = [env.ids.home_ee[1,0] + 0.03, 0.26, 0.40]
    env.future_preposition_yaws[1] = -np.pi / 2
    env._control()
    assert not env.preposition_active[1]
    assert env.hold_requested[1]
    assert env.phase[0] == env.phase[1] == "idle"
    # Phase labels stay idle throughout; actual braking, reference speed and
    # a measured still interval must all pass before switching preparation.
    env.data.qvel[env.ids.arm_dofs[0]] = 0.1
    env._control()
    assert not env.preposition_active[1]
    env.data.qvel[env.ids.arm_dofs[0]] = 0
    env.ik.cart_velocities[0] = [0.02, 0, 0]
    env._control()
    assert not env.preposition_active[1]
    env.ik.cart_velocities[0] = 0
    env._control()
    assert not env.preposition_active[1]
    env.data.time += 0.06
    env._control()
    assert env.preposition_active[1]


@pytest.mark.parametrize('has_planner', [False, True])
def test_raw_reset_then_native_wait_cannot_execute_previous_episode_advisory(env, has_planner):
    planner = None
    if has_planner:
        from dual_arm_conveyor.future_plan import SoftPlan, planner_for
        planner = planner_for(env)
        planner.plans[0] = SoftPlan(0, 123456, 0.0)
        planner.counts['created'] = 4
    set_preposition(env)
    env.step(env.wait_action)
    assert env.preposition_seconds[0] > 0
    env.reset(seed=456)
    assert np.isnan(env.future_preposition_goals).all()
    assert np.isnan(env.future_preposition_yaws).all()
    assert not env.preposition_active.any()
    assert not env._previous_preposition_active.any()
    assert not env.preposition_seconds.any()
    if has_planner:
        # Inspect the same object directly: calling planner_for here would
        # hide a broken native reset by performing its lazy reset instead.
        assert planner.plans == [None, None]
        assert planner.counts['created'] == 0
        assert planner._tracks_container is env.tracks
    before = env.data.qpos[env.ids.arm_qpos].copy()
    for _ in range(5):
        env.step(env.wait_action)
    np.testing.assert_allclose(env.data.qpos[env.ids.arm_qpos], before, atol=1e-5)
    assert not env.preposition_seconds.any()
    assert np.all(env.target == -1)
