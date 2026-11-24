"""Task invariants exercised against the real MuJoCo environment."""

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv, State


@pytest.fixture
def env():
    instance = ConveyorEnv(Config(sensor_mode="oracle", total_objects=2))
    yield instance
    instance.close()


def test_grasp_model_has_no_weld_or_mocap_attachment(env):
    assert not np.any(env.model.eq_type == int(mujoco.mjtEq.mjEQ_WELD))
    assert not np.any(env.model.body_mocapid[env.ids.cube_bodies] >= 0)
    assert np.all(
        env.model.jnt_type[env.ids.cube_joints] == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    assert env.ids.native_surface_velocity


def test_seeded_reset_reproduces_initial_state_and_arrivals(env):
    first = env.reset(seed=49)
    qpos = env.data.qpos.copy()
    arrivals = env.arrivals.copy()
    lateral = env.lateral.copy()
    for _ in range(3):
        env.step(120)
    second = env.reset(seed=49)
    np.testing.assert_array_equal(qpos, env.data.qpos)
    np.testing.assert_array_equal(arrivals, env.arrivals)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    env.reset(seed=50)
    assert not np.array_equal(lateral, env.lateral)


def _place_on_belt(env, index, x):
    # Controlled initial conditions for scheduling invariants, not a grasp.
    while env.serial[index] < 0:
        env.next_arrival = env.data.time
        env._spawn()
    q, v = env.ids.cube_qpos[index], env.ids.cube_dofs[index]
    env.data.qpos[q : q + 3] = [
        x,
        0,
        env.config.belt_top + env.ids.object_specs[index].half_height,
    ]
    env.data.qvel[v : v + 3] = [env.config.belt_speed, 0, 0]
    env.states[index] = State.BELT
    mujoco.mj_forward(env.model, env.data)
    env._sense()


def test_joint_mask_prevents_duplicate_targets_and_expired_deadlines(env):
    _place_on_belt(env, 0, -0.3)
    _place_on_belt(env, 1, 0.2)
    obs = env.observe()
    assert obs["mask"][1]  # arm 0 gets object 0; arm 1 gets object 1
    for index in range(10):
        assert not obs["mask"][index * 11 + index]
    _place_on_belt(env, 0, env.windows[0, 1] - 0.005)
    assert not env._candidate(0, 0)[0]
    assert not env.observe()["mask"][0 * 11 + 10]


def test_dispatch_reserves_one_distinct_object_per_arm(env):
    _place_on_belt(env, 0, -0.3)
    _place_on_belt(env, 1, 0.2)
    env.step(1)
    assert list(env.target) == [0, 1]
    assert list(env.belief[:2]) == [2, 2]
    assert list(env.arm_assignments) == [1, 1]
    # Busy arms must not be reassigned; waiting preserves current tasks.
    assert np.flatnonzero(env.observe()["mask"]).tolist() == [120]


def test_uncollected_body_moves_with_belt_and_counts_as_missed():
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1))
    try:
        start = env.cube_position(0).copy()
        for _ in range(10):
            env.step(120)
        travelled = env.cube_position(0)[0] - start[0]
        assert travelled == pytest.approx(0.1, abs=0.005)
        assert env.cube_position(0)[2] == pytest.approx(
            env.config.belt_top + env.ids.object_specs[0].half_height, abs=0.003
        )
        for _ in range(200):
            _, _, terminated, truncated, _ = env.step(120)
            if terminated or truncated:
                break
        metrics = env.metrics()
        assert terminated and not truncated
        assert metrics["missed"] == 1
        assert metrics["sorted"] == 0
        assert metrics["unfinished"] == 0
        assert not metrics["episode_success"]
    finally:
        env.close()


def test_invalid_joint_action_cannot_dispatch_an_object(env):
    with pytest.raises(ValueError):
        env.step(121)
    # At reset, the cube has not yet reached either arm's pick window.
    assert not env.observe()["mask"][0]
    env.step(0)
    assert env.invalid_actions == 1
    assert list(env.target) == [-1, -1]


def test_later_displacement_revokes_previously_counted_success(env):
    env.owner[0] = 0
    env.states[0] = State.SORTED
    env.arm_completions[0] = 1
    center = env.ids.bin_centers[env.types[0]]
    q = env.ids.cube_qpos[0]
    env.data.qpos[q : q + 3] = center + np.array([0.2, 0, env.config.cube_size / 2])
    mujoco.mj_forward(env.model, env.data)
    env.step(120)
    assert env.metrics()["sorted"] == 0
    assert env.metrics()["dropped"] == 1
    assert list(env.arm_completions) == [0, 0]


def test_resting_in_wrong_color_bin_cannot_count_as_sorted():
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1))
    try:
        env.owner[0] = 0
        env.states[0] = State.RELEASED
        wrong_center = env.ids.bin_centers[1 - env.types[0]]
        q, v = env.ids.cube_qpos[0], env.ids.cube_dofs[0]
        env.data.qpos[q : q + 3] = wrong_center + np.array(
            [0, 0, env.ids.object_specs[0].half_height]
        )
        env.data.qvel[v : v + 6] = 0
        mujoco.mj_forward(env.model, env.data)
        for _ in range(60):
            _, _, done, truncated, _ = env.step(120)
            if done or truncated:
                break
        assert env.metrics()["sorted"] == 0
        assert env.metrics()["dropped"] == 1
        assert not env.metrics()["episode_success"]
    finally:
        env.close()


def test_dense_transport_leaves_belt_before_crossing_other_gripper():
    # Regression: this exact stream previously produced 23 hand/hand collision
    # steps because diagonal loaded transport crossed another arm's lift.
    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=10, spacing=0.40))
    try:
        env.reset(seed=4000)
        while True:
            _, _, done, truncated, _ = env.step(env.heuristic_action())
            if done or truncated:
                break
        metrics = env.metrics()
        assert metrics["inter_arm_collision_steps"] == 0
        assert metrics["robot_collision_steps"] == 0
        assert metrics["sorted"] >= 8
        assert metrics["unfinished"] == 0
        assert min(metrics["arm_sorted"]) > 0
    finally:
        env.close()


def test_operational_state_does_not_read_truth(env, monkeypatch):
    _place_on_belt(env, 0, -0.30)
    _place_on_belt(env, 1, 0.20)
    before = env.observe()
    action = env.heuristic_action()

    def forbidden(*args):
        raise AssertionError("Controller accessed evaluator truth")

    monkeypatch.setattr(env, "cube_position", forbidden)
    monkeypatch.setattr(env, "cube_velocity", forbidden)
    env.types[:] = 99
    env.states[:] = State.DROPPED
    env.yaws[:] = 999.0
    env.arrivals.clear()
    env.arrivals.append(-100.0)
    env.next_arrival = -100.0
    env.serial[:] = -777
    env.arm_completions[:] = 99
    after = env.observe()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert env.heuristic_action() == action
    env._assign(0, 0)
    env._control()
    env._recover(0)


def test_random_stream_and_orientation_are_not_periodic():
    env = ConveyorEnv(Config(sensor_mode="oracle"), seed=91)
    try:
        # Inspect the supply generator independently of robot motion. Future
        # arrivals are drawn only when an object enters the physical scene.
        assert env.spawned == 1
        for _ in range(9):
            env.data.time = env.next_arrival
            env._spawn()
        gaps = np.diff(env.arrivals) * env.config.belt_speed
        assert np.std(gaps) > 0.03
        assert np.min(gaps) >= 0.12
        assert np.ptp(env.spawn_yaw_log) > 4.0
        assert env.ids.bin_centers.shape == (2, 3)
    finally:
        env.close()


def test_reserved_slot_is_not_reused_before_return(env):
    from dual_arm_conveyor.vision import Track

    _place_on_belt(env, 0, -0.3)
    env._assign(0, 0)
    env.belief[0] = 4
    env._ingest_tracks(
        [
            Track(
                50,
                np.array([-0.8, 0, 0.23]),
                np.array([0.1, 0, 0]),
                0.0,
                0,
                1.0,
                0.0,
                0.0,
            )
        ]
    )
    assert env.tracks[0].track_id == 0
    assert env.target[0] == 0


def test_failed_grasp_releases_deposit_reservation(env):
    _place_on_belt(env, 0, -0.3)
    color = env.tracks[0].kind
    env._assign(0, 0)
    assert np.count_nonzero(env.bin_occupied[color]) == 1
    env._recover(0)
    assert not np.any(env.bin_occupied[color])
    assert env.belief[0] == 1
    assert env.phase[0] == "recover"


def test_mask_rejects_crossed_simultaneous_pick_order(env):
    _place_on_belt(env, 0, -0.20)
    _place_on_belt(env, 1, 0.20)
    assert not env.observe()["mask"][11]  # arm0 downstream, arm1 upstream


def test_joint_action_cannot_overbook_last_slot_in_shared_bin():
    instance = ConveyorEnv(
        Config(sensor_mode="oracle", total_objects=2, bin_capacity=1)
    )
    try:
        _place_on_belt(instance, 0, -0.3)
        _place_on_belt(instance, 1, 0.2)
        instance.types[:2] = 0
        instance._sense()
        observation = instance.observe()
        assert observation["mask"][10]  # arm 0 alone can reserve the last slot
        assert observation["mask"][111]  # arm 1 alone can reserve the last slot
        assert not observation["mask"][1]  # both cannot reserve that same slot
    finally:
        instance.close()


def _move_arm_away_from_home(env, arm=0):
    from dual_arm_conveyor.ik import downward_quaternion

    goal = env.ids.home_ee[arm] + np.array([0.07, 0, 0])
    env.data.qpos[env.ids.arm_qpos[arm]] = env.ik.solve_pose(
        env.data, arm, goal, downward_quaternion(env.grasp_yaw[arm])
    )
    mujoco.mj_forward(env.model, env.data)
    assert (
        np.linalg.norm(env.data.site_xpos[env.ids.ee_sites[arm]] - env.ids.home_ee[arm])
        > 0.03
    )


def test_old_recovery_phase_does_not_discard_new_verified_route(env, monkeypatch):
    _place_on_belt(env, 0, -0.3)
    env._assign(0, 0)
    _move_arm_away_from_home(env)
    env._set_phase(0, "recover")
    env.data.time = 8.0
    verified_route = object()

    def verified_route_result(arm, goal, loaded, via=()):
        # Isolate phase/retry scheduling from route geometry. The planner has
        # just accepted a route; its own collision checks are tested separately.
        env.routes[arm] = verified_route
        return np.asarray(goal), True

    monkeypatch.setattr(env, "_route_goal", verified_route_result)
    env._control()
    assert env.phase[0] == "recover"
    assert env.routes[0] is verified_route


@pytest.mark.parametrize("verified", [False, True])
def test_crossing_wait_requires_final_staging_pose_not_current_waypoint(
    env, monkeypatch, verified
):
    _place_on_belt(env, 0, -0.3)
    env.types[0] = 1  # arm 0 must cross to the blue bin
    env._sense()
    env._assign(0, 0)
    _move_arm_away_from_home(env)
    env.holding[0] = True
    env.belief[0] = 3
    env._set_phase(0, "transfer_wait")

    def staging_route_result(arm, goal, loaded, via=()):
        # A failed plan holds here; a verified multi-segment plan can also have
        # a reached intermediate waypoint here. Neither means home was reached.
        return env.data.site_xpos[env.ids.ee_sites[arm]].copy(), verified

    monkeypatch.setattr(env, "_route_goal", staging_route_result)
    monkeypatch.setattr(env, "_gripper_has_object", lambda arm: True)
    env._control()
    assert env.phase[0] == "transfer_wait"
    assert env.transfer_owner == -1
    assert env.bin_owner[1] == -1


def test_unavailable_transfer_resource_times_out_to_explicit_recovery(env, monkeypatch):
    _place_on_belt(env, 0, -0.3)
    env.types[0] = 0
    env._sense()
    env._assign(0, 0)
    env.holding[0] = True
    env.belief[0] = 3
    env.bin_owner[0] = 1  # a conflicting owner prevents the transfer
    env._set_phase(0, "transfer_wait")
    env.data.time = 12.1
    monkeypatch.setattr(env, "_gripper_has_object", lambda arm: True)
    env._control()
    assert env.phase[0] == "recover"
    assert env.retries == 1
    assert not env.holding[0]
    assert env.arm_releases[0] == env.arm_completions[0] == 0
    assert any(event[0] == "recover" for event in env.eval_queue)


def _begin_release(env):
    _place_on_belt(env, 0, -0.3)
    env._assign(0, 0)
    env.holding[0] = True
    env.belief[0] = 3
    env._set_phase(0, "release")
    env.data.time = 0.5


def _release_readings(env, aperture, touches):
    # Sensor-injection unit fixture for release confirmation. Physical contact
    # and grasp success remain covered by the unmodified rollout regression.
    env.data.qpos[env.ids.finger_qpos[0]] = aperture / 2
    env.data.sensordata[env.ids.finger_touch_adr[0]] = touches


@pytest.mark.parametrize(
    "aperture,touches",
    [
        (0.08, [0.2, 0.0]),
        (0.08, [0.0, 0.2]),
        (0.08, [0.1, 0.0]),
        (0.075, [0.0, 0.0]),
        (0.06, [0.0, 0.0]),
    ],
)
def test_release_keeps_task_reserved_if_either_finger_loaded_or_not_open(
    env, aperture, touches
):
    _begin_release(env)
    _release_readings(env, aperture, touches)
    env._control()
    env.data.time = 2.0
    _release_readings(env, aperture, touches)
    env._control()
    assert env.phase[0] == "release"
    assert env.target[0] == 0 and env.holding[0]
    assert env.arm_releases[0] == 0
    assert np.isnan(env.release_clear_since[0])
    # Even a single finger retaining the old object prevents a new dispatch.
    assert not env.observe()["mask"].reshape(11, 11)[:10].any()


def test_release_requires_continuous_clear_contact_interval(env):
    _begin_release(env)
    for timestamp in (0.5, 0.595):
        env.data.time = timestamp
        _release_readings(env, 0.08, [0.099, 0.099])
        env._control()
        assert env.phase[0] == "release" and env.arm_releases[0] == 0
    env.data.time = 0.61
    _release_readings(env, 0.08, [0.099, 0.099])
    env._control()
    assert env.phase[0] == "retreat"
    assert env.arm_releases[0] == 1
    assert not env.holding[0] and env.belief[0] == 4
    assert env.bin_last_release[env.drop_kind[0]] == pytest.approx(0.61)


def test_contact_reappearance_restarts_release_confirmation_and_reset_clears_it(env):
    _begin_release(env)
    _release_readings(env, 0.08, [0.0, 0.0])
    env._control()
    assert env.release_clear_since[0] == pytest.approx(0.5)
    env.data.time = 0.58
    _release_readings(env, 0.08, [0.0, 0.2])
    env._control()
    assert np.isnan(env.release_clear_since[0])
    for timestamp in (0.60, 0.68):
        env.data.time = timestamp
        _release_readings(env, 0.08, [0.0, 0.0])
        env._control()
        assert env.phase[0] == "release"
    assert env.release_clear_since[0] == pytest.approx(0.60)
    env.data.time = 0.705
    _release_readings(env, 0.08, [0.0, 0.0])
    env._control()
    assert env.phase[0] == "retreat" and env.arm_releases[0] == 1
    env._set_phase(0, "release")
    assert np.isnan(env.release_clear_since[0])
    env.release_clear_since[:] = [4.0, 5.0]
    env.reset(seed=0)
    assert np.isnan(env.release_clear_since).all()


def test_phase_transition_preserves_verified_segment_for_current_command(
    env, monkeypatch
):
    from types import SimpleNamespace
    from dual_arm_conveyor.ik import downward_quaternion

    _place_on_belt(env, 0, -0.3)
    env._assign(0, 0)
    env._set_phase(0, "transport")
    goal = env.drop_targets[0].copy()
    goal[2] = env.ids.bin_rim_z + 0.07
    quaternion = downward_quaternion(env.grasp_yaw[0])
    env.data.qpos[env.ids.arm_qpos[0]] = env.ik.solve_pose(
        env.data, 0, goal, quaternion
    )
    mujoco.mj_forward(env.model, env.data)
    # A two-segment accepted plan has reached its final segment. The test spies
    # on servo handoff; geometric route checking is covered by test_motion.py.
    plan = SimpleNamespace(
        verified=True,
        waypoints=(goal + [0, 0, 0.05], goal),
        orientation_at=lambda index, position: quaternion,
    )
    env.routes[0] = plan
    env.route_index[0] = 1
    commands = []

    def capture_command(data, arm, target, **kwargs):
        commands.append((arm, kwargs["motion_plan"], kwargs["waypoint_index"]))

    monkeypatch.setattr(env.ik, "command", capture_command)
    monkeypatch.setattr(env, "_gripper_has_object", lambda arm: True)
    env._control()
    assert env.phase[0] == "place"
    assert env.route_index[0] == 0 and env.routes[0] is None
    assert commands[0] == (0, plan, 1)
    assert env.active_motion_indices[0] == 1
