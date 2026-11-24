"""Four physical arms share two bins and cannot double-book a station/track."""

import numpy as np
import mujoco
import pytest
from types import SimpleNamespace
from dataclasses import replace

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv, WAIT
from dual_arm_conveyor.ik import downward_quaternion
from dual_arm_conveyor.vision import Track


def make_env():
    return ConveyorEnv(
        Config(
            num_arms=4,
            strategy="adaptive",
            sensor_mode="oracle",
            belt_speed=0.3,
            control_speed=0.65,
            max_time=2,
        ),
        seed=55000,
    )


def test_joint_dispatch_reserves_station_and_objects():
    env = make_env()
    try:
        tracks = [
            Track(
                9000 + i,
                np.array([x, 0.0, 0.23]),
                np.array([0.3, 0, 0]),
                0.0,
                color,
                1.0,
                0.0,
                0.0,
                10,
                True,
            )
            for i, (x, color) in enumerate(
                [(-0.75, 0), (-0.70, 1), (-0.10, 0), (-0.05, 1)]
            )
        ]
        env._ingest_tracks(tracks)
        obs = env.observe()
        actions = env.action_choices[obs["mask"]]
        assert len(actions) > 1
        for action in actions:
            assigned = [int(i) for i in action if i < WAIT]
            assert len(assigned) == len(set(assigned))
            assert sum(action[:2] < WAIT) <= 1
            assert sum(action[2:] < WAIT) <= 1
        assert obs["mask"][env.heuristic_action()]
    finally:
        env.close()


def test_shared_bin_does_not_unload_while_downstream_arm_has_reservation():
    env = make_env()
    try:
        env.bin_occupied[0] = True
        env.bin_slots[0] = env.config.bin_capacity
        env.target[2] = 0
        env.drop_kind[0] = 0
        env._service_bins()
        assert env.bin_service[0] == "closed"
        env.target[2] = -1
        env._service_bins()
        assert env.bin_service[0] == "opening"
    finally:
        env.close()


def test_four_arm_physics_steps_and_reset_keep_two_bins():
    env = make_env()
    try:
        for _ in range(5):
            env.step(env.heuristic_action())
        metrics = env.metrics()
        assert metrics["arm_count"] == 4
        assert metrics["bin_count"] == 2
        assert len(metrics["arm_sorted"]) == 4
        assert metrics["robot_collision_steps"] == 0
        env.reset(55001)
        assert env.phase == ["idle"] * 4
        assert env.arm_assignments.sum() == 0
        assert env.bin_slots.shape == (2,)
    finally:
        env.close()


def _set_target(env, arm=0, slot=0, x=-0.65):
    env.tracks[slot] = Track(
        9900 + slot,
        np.array([x, 0.0, 0.23]),
        np.array([0.3, 0, 0]),
        np.pi / 2 if arm % 2 == 0 else -np.pi / 2,
        arm % 2,
        1.0,
        float(env.data.time),
        float(env.data.time),
        10,
        True,
    )
    env.target[arm] = slot
    env.belief[slot] = 2
    env.drop_kind[slot] = arm % 2
    env.drop_targets[slot] = env.ids.bin_centers[arm % 2].copy()
    env.drop_targets[slot, 0] += env.ids.arm_base_positions[arm, 0]


def _put_tool(env, arm, position):
    q = env.ik.solve_pose(
        env.data, arm, np.asarray(position), downward_quaternion(env.grasp_yaw[arm])
    )
    env.data.qpos[env.ids.arm_qpos[arm]] = q
    mujoco.mj_forward(env.model, env.data)
    assert np.linalg.norm(env.data.site_xpos[env.ids.ee_sites[arm]] - position) < 0.001


def test_local_bin_zones_remain_owned_when_other_station_releases(monkeypatch):
    env = make_env()
    try:
        monkeypatch.setattr(env.ik, "command", lambda *args, **kwargs: None)
        monkeypatch.setattr(env, "_gripper_has_object", lambda arm: True)
        for arm, slot in ((0, 0), (2, 1)):
            _set_target(env, arm, slot)
            env.phase[arm] = "transfer_wait"
            env.idle_anchors[arm] = env.data.site_xpos[env.ids.ee_sites[arm]].copy()
        env._control()
        assert env.phase[0] == env.phase[2] == "transport"
        np.testing.assert_array_equal(env.bin_zone_owner[:, 0], [0, 2])
        env._release_resources(0)
        assert env.bin_zone_owner[0, 0] == -1
        assert env.bin_zone_owner[1, 0] == env.bin_owner[0] == 2
        # Even with no target bookkeeping, the surviving physical reservation
        # must prevent opening a full bin underneath the other station.
        env.target[:] = -1
        env.bin_occupied[0] = True
        env.bin_slots[0] = env.config.bin_capacity
        env._service_bins()
        assert env.bin_service[0] == "closed"
        env._release_resources(2)
        assert env.bin_owner[0] == -1
        env._service_bins()
        assert env.bin_service[0] == "opening"
    finally:
        env.close()


def test_return_keeps_bin_reserved_until_checked_robot_settles_at_ready_pose(
    monkeypatch,
):
    env = make_env()
    try:
        _set_target(env)
        env.bin_owner[0] = env.bin_zone_owner[0, 0] = 0
        env.phase[0] = "retreat"
        end = env.drop_targets[0].copy()
        end[2] = env.ids.bin_rim_z + 0.07
        _put_tool(env, 0, end)
        monkeypatch.setattr(env.ik, "command", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            env,
            "_route_goal",
            lambda arm, goal, loaded, **kwargs: (np.asarray(goal), True),
        )
        env._control()
        assert env.phase[0] == "return"
        assert env.target[0] == 0 and env.bin_zone_owner[0, 0] == env.bin_owner[0] == 0
        env._control()  # Still physically over the bin; cannot release ownership.
        assert env.phase[0] == "return" and env.bin_zone_owner[0, 0] == 0
        env.data.qpos[env.ids.arm_qpos[0]] = env.ids.home_qpos[0]
        mujoco.mj_forward(env.model, env.data)
        accepted = [False]
        env.routes[0] = SimpleNamespace(
            verified=True,
            waypoints=(env.ids.home_ee[0],),
            can_advance=lambda *args, **kwargs: accepted[0],
        )
        env._control()
        assert env.phase[0] == "return" and env.bin_zone_owner[0, 0] == 0
        accepted[0] = True
        env._control()
        assert env.phase[0] == "idle" and env.target[0] == -1
        assert env.bin_zone_owner[0, 0] == env.bin_owner[0] == -1
    finally:
        env.close()


@pytest.mark.parametrize("already_reoriented", [False, True])
@pytest.mark.parametrize("extended_deadline", [False, True])
def test_exit_preview_rejection_prevents_descent_without_reading_object_truth(
    monkeypatch, already_reoriented, extended_deadline
):
    env = make_env()
    env.config = replace(env.config, extended_pick_deadline=extended_deadline)
    try:
        _set_target(env)
        env.phase[0] = "approach"
        env.reoriented[0] = already_reoriented
        position = env._position(0)
        above = position.copy()
        above[2] = (
            env.config.belt_top
            + env.tracks[0].half_height
            + env.grasp_offsets[0]
            + 0.13
        )
        _put_tool(env, 0, above)
        assert env._descent_feasible(0, env.tracks[0], position)
        calls = []
        monkeypatch.setattr(env.ik, "command", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            env,
            "_route_goal",
            lambda arm, goal, loaded, **kwargs: (np.asarray(goal), True),
        )
        monkeypatch.setattr(
            env.planner, "plan", lambda *args, **kwargs: SimpleNamespace(verified=True)
        )

        def reject_exit(data, arm, closure, lift, **kwargs):
            calls.append(
                (arm, closure.copy(), lift.copy(), kwargs["grasp_position"].copy())
            )
            return SimpleNamespace(verified=False)

        def no_truth(*args):
            raise AssertionError(
                "Operational admission must not query object ground truth"
            )

        monkeypatch.setattr(env.planner, "grasp_exit_feasible", reject_exit)
        monkeypatch.setattr(env, "cube_position", no_truth)
        monkeypatch.setattr(env, "cube_velocity", no_truth)
        env._control()
        assert len(calls) == 1 and calls[0][0] == 0
        _, closure, lift, grasp = calls[0]
        assert grasp[0] < closure[0] < lift[0] and lift[2] > closure[2]
        assert env.phase[0] == ("recover" if already_reoriented else "reorient")
        assert env.phase[0] != "descend"
    finally:
        env.close()


@pytest.mark.parametrize("phase", ["descend", "close", "lift"])
def test_failed_low_route_immediately_opens_and_holds_before_recovery(
    monkeypatch, phase
):
    env = make_env()
    try:
        _set_target(env)
        env.phase[0] = phase
        commands = []
        monkeypatch.setattr(
            env,
            "_route_goal",
            lambda arm, goal, loaded, **kwargs: (
                env.data.site_xpos[env.ids.ee_sites[arm]].copy(),
                False,
            ),
        )
        monkeypatch.setattr(
            env.ik,
            "command",
            lambda data, arm, target, **kwargs: commands.append((arm, kwargs)),
        )
        env._control()
        assert env.phase[0] == "recover" and env.retries == 1
        first = next(kwargs for arm, kwargs in commands if arm == 0)
        assert first["hold"] and first["gripper"] == 0.04
        assert env.active_motion_plans[0] is None
    finally:
        env.close()
