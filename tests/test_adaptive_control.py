"""Variable-speed integration and sensor/controller boundary regression tests."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from dataclasses import replace

import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.vision import Track


@pytest.fixture
def make_env():
    instances = []

    def factory(**kwargs):
        values = dict(
            strategy="adaptive",
            sensor_mode="vision",
            belt_speed=0.3,
            control_speed=0.65,
            max_time=10.0,
            total_objects=1,
        )
        values.update(kwargs)
        instance = ConveyorEnv(Config(**values), seed=53001)
        instances.append(instance)
        return instance

    yield factory
    for instance in instances:
        instance.close()


class DenyDrive:
    def __getattr__(self, name):
        raise AssertionError(f"Visual controller accessed physical drive: {name}")


def deny_object_truth(*args, **kwargs):
    raise AssertionError("Visual controller accessed evaluator object truth")


@pytest.mark.parametrize("profile", ["sine", "random"])
def test_variable_drive_physically_moves_objects_and_stripes_then_resets(
    make_env, profile
):
    # Oracle avoids image cost here; this test checks the physical scene, not policy.
    env = make_env(sensor_mode="oracle", belt_profile=profile)
    initial_position = env.cube_position(0)
    marker_ids = env.belt_markers.mocap_ids
    initial_markers = env.data.mocap_pos[marker_ids].copy()
    initial_speed = float(env.model.geom_surfacevel[env.ids.belt_geom, 0])
    for _ in range(15):
        env.step(120)
    speed = float(env.model.geom_surfacevel[env.ids.belt_geom, 0])
    assert abs(speed - initial_speed) > 0.001
    assert env.cube_velocity(0)[0] == pytest.approx(speed, abs=0.01)
    # Actual displacement must follow integrated speed, not final speed times time.
    travel = env.cube_position(0)[0] - initial_position[0]
    assert travel == pytest.approx(env.belt_sample.distance, abs=0.006)
    marker_travel = np.remainder(
        env.data.mocap_pos[marker_ids, 0] - initial_markers[:, 0], env.belt_markers.span
    )
    np.testing.assert_allclose(
        marker_travel, env.belt_sample.distance % env.belt_markers.span, atol=1e-10
    )
    assert abs(env.belt_sample.distance - speed * env.data.time) > 0.001
    env.reset(53001)
    assert env.data.time == 0
    assert env.belt_sample.distance == 0
    assert env.model.geom_surfacevel[env.ids.belt_geom, 0] == initial_speed
    np.testing.assert_array_equal(env.data.mocap_pos[marker_ids], initial_markers)
    np.testing.assert_array_equal(env.cube_position(0), initial_position)


def test_active_visual_control_cannot_read_generator_or_object_truth(
    make_env, monkeypatch
):
    env = make_env(belt_profile="sine")
    # Obtain the target from actual delayed camera observations.
    for _ in range(50):
        env.step(env.heuristic_action())
        if "approach" in env.phase:
            break
    assert "approach" in env.phase
    before = env.observe()
    action = env.heuristic_action()
    speed = env._belt_velocity()
    monkeypatch.setattr(env, "belt_sample", DenyDrive())
    monkeypatch.setattr(env, "belt_motion", DenyDrive())
    monkeypatch.setattr(env, "cube_position", deny_object_truth)
    monkeypatch.setattr(env, "cube_velocity", deny_object_truth)
    original_config = env.config

    class SensorOnlyConfig:
        def __getattr__(self, name):
            if name in {
                "belt_profile",
                "belt_speed_period",
                "belt_speed_amplitude",
                "belt_max_acceleration",
            }:
                raise AssertionError(
                    f"Controller accessed physical speed profile: {name}"
                )
            return getattr(original_config, name)

    monkeypatch.setattr(env, "config", SensorOnlyConfig())
    after = env.observe()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert env._belt_velocity() == speed
    assert env.heuristic_action() == action
    env._control()  # No truth access while approaching a real camera target.


def set_target(env, phase, x=-0.4):
    env.tracks[0] = Track(
        41,
        np.array([x, 0.0, 0.23]),
        np.array([0.3, 0.0, 0.0]),
        0.0,
        0,
        0.95,
        float(env.data.time),
        float(env.data.time),
        hits=8,
    )
    env.track_slots = {41: 0}
    env.belief[0] = 2
    env.target[0] = 0
    env.close_position[0] = [x, 0.0, 0.23]
    env.lift_xy[0] = [x, 0.0]
    env.close_displacement[0] = 0.0
    env.lift_displacement[0] = 0.0
    env._set_phase(0, phase)


def capture_commands(env, monkeypatch):
    commands = []
    monkeypatch.setattr(
        env, "_route_goal", lambda arm, goal, *args, **kwargs: (np.array(goal), True)
    )
    monkeypatch.setattr(env, "_gripper_has_object", lambda arm: False)

    def command(data, arm, goal, **kwargs):
        if arm == 0:
            commands.append((np.array(goal), dict(kwargs)))

    monkeypatch.setattr(env.ik, "command", command)
    return commands


def test_close_follows_changing_observed_speed_instead_of_configured_mean(
    make_env, monkeypatch
):
    env = make_env()
    set_target(env, "close")
    commands = capture_commands(env, monkeypatch)
    monkeypatch.setattr(env, "belt_sample", DenyDrive())
    monkeypatch.setattr(env, "belt_motion", DenyDrive())
    speeds = [0.22] * 5 + [0.39] * 5
    expected_x = env.close_position[0, 0]
    for speed in speeds:
        monkeypatch.setattr(
            env.sensor,
            "belt_velocity",
            lambda now, value=speed: np.array([value, 0.0, 0.0]),
        )
        env._control()
        expected_x += speed * env.config.physics_dt
        goal, command = commands[-1]
        assert goal[0] == pytest.approx(expected_x, abs=1e-10)
        assert command["feedforward"][0] == speed
        assert command["gripper"] == 0
        env.data.time += env.config.physics_dt
    assert env.phase[0] == "close"


def test_lift_uses_observed_speed_then_stops_following_after_departure(
    make_env, monkeypatch
):
    env = make_env()
    commands = capture_commands(env, monkeypatch)
    goals, feeds = [], []
    for observed_speed in (0.2, 0.4):
        env.data.time = 0.0
        set_target(env, "lift")
        monkeypatch.setattr(
            env.sensor,
            "belt_velocity",
            lambda now, speed=observed_speed: np.array([speed, 0.0, 0.0]),
        )
        for _ in range(20):
            env._control()
            env.data.time += env.config.physics_dt
        goals.append(commands[-1][0][0] - env.lift_xy[0, 0])
        feeds.append(commands[-1][1]["feedforward"][0])
    assert goals[0] > 0 and goals[1] == pytest.approx(2 * goals[0])
    assert feeds[0] > 0 and feeds[1] == pytest.approx(2 * feeds[0])
    env.data.time = 0.21
    env._control()
    finished_x = commands[-1][0][0]
    assert commands[-1][1]["feedforward"][0] == 0.0
    env.data.time = 0.3
    env._control()
    assert commands[-1][0][0] == finished_x


def test_too_late_fast_target_recovers_before_descent_without_closing(
    make_env, monkeypatch
):
    env = make_env()
    x = float(env.windows[0, 1] - 0.01)
    set_target(env, "approach", x=x)
    env.tracks[0] = replace(env.tracks[0], velocity=np.array([0.4, 0.0, 0.0]))
    commands = capture_commands(env, monkeypatch)
    assert not env._descent_feasible(0, env.tracks[0], env._position(0))
    upstream = env._position(0).copy()
    upstream[0] = env.windows[0, 0] + 0.03
    assert env._descent_feasible(0, env.tracks[0], upstream)
    env._control()
    assert env.phase[0] == "recover"
    assert env.retries == 1
    assert all(command["gripper"] > 0 for _, command in commands)
    assert not env.holding[0]


def test_existing_default_strategy_and_constant_speed_remain_baseline(make_env):
    defaults = Config()
    assert defaults.strategy == "baseline"
    assert defaults.belt_profile == "constant"
    assert defaults.belt_speed == 0.1
    assert defaults.control_speed == 0.45
    env = make_env(strategy="baseline", belt_speed=0.1, control_speed=0.45)
    assert not env.ik.adaptive and not env.sensor.tracker.adaptive
    for _ in range(3):
        env.step(120)
        assert env._belt_velocity() == 0.1
        assert env.model.geom_surfacevel[env.ids.belt_geom, 0] == 0.1
