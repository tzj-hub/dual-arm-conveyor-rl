"""A near-lane deadline extension never grants extra far-side or station reach."""

from dataclasses import replace

import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.vision import Track


@pytest.fixture
def env():
    simulation = ConveyorEnv(
        Config(
            num_arms=4,
            strategy="adaptive",
            sensor_mode="oracle",
            gripper_mode="fingers",
            deep_bins=True,
            auto_unload=False,
            total_objects=8,
            pool_size=8,
            bin_capacity=180,
            physics_dt=0.0025,
            belt_speed=0.15,
        )
    )
    yield simulation
    simulation.close()


@pytest.mark.parametrize(
    "arm,y,expected",
    [
        (0, -0.065, 0),
        (1, 0.065, 0),
        (0, 0.065, -0.1),
        (1, -0.065, -0.1),
        (0, -0.04, -0.1),
        (0, -0.085, -0.1),
        (1, 0.04, -0.1),
        (1, 0.085, -0.1),
        (2, -0.065, 0.82),
        (3, 0.065, 0.82),
        (2, 0.065, 0.82),
        (3, -0.065, 0.82),
    ],
)
def test_extension_is_only_for_upstream_near_band(env, arm, y, expected):
    original = env.windows.copy()
    boundary = expected - 0.02 - 0.15 * 1.05
    track = Track(
        1,
        np.array([boundary, y, 0.23]),
        np.array([0.15, 0, 0]),
        np.pi / 2,
        arm % 2,
        0.99,
        0,
        0,
        size=np.array([0.05, 0.04, 0.05]),
    )
    assert env._descent_feasible(arm, track, np.array([boundary - 1e-6, y, 0.23]))
    assert not env._descent_feasible(arm, track, np.array([boundary + 1e-6, y, 0.23]))
    np.testing.assert_array_equal(env.windows, original)


def test_additional_execution_room_does_not_admit_a_late_new_task(env):
    track = Track(
        1,
        np.array([-0.2, -0.065, 0.23]),
        np.array([0.15, 0, 0]),
        np.pi / 2,
        0,
        0.99,
        0,
        0,
        size=np.array([0.05, 0.04, 0.05]),
    )
    env.tracks[0] = track
    env.belief[0] = 1
    accepted, _, deadline = env._candidate(0, 0)
    assert not accepted
    assert deadline == pytest.approx((-0.1 - track.position[0]) / 0.15)
    assert env._descent_feasible(0, track, track.position)


@pytest.mark.parametrize("arm,y", [(0, -0.065), (1, 0.065)])
def test_descent_keeps_full_travel_and_twenty_mm_reserve(env, arm, y):
    # 0.45s descent + 0.40s closure + 0.20s lift reserve at 0.15m/s.
    boundary = -0.02 - 0.15 * 1.05
    track = Track(
        1,
        np.array([boundary, y, 0.23]),
        np.array([0.15, 0, 0]),
        0,
        arm % 2,
        0.95,
        0,
        0,
        size=np.array([0.05, 0.04, 0.05]),
    )
    assert env._descent_feasible(arm, track, np.array([boundary - 1e-6, y, 0.23]))
    assert not env._descent_feasible(arm, track, np.array([boundary + 1e-6, y, 0.23]))
    assert not env._descent_feasible(arm, track, np.array([boundary - 1e-6, -y, 0.23]))
    faster = replace(track, velocity=np.array([0.25, 0, 0]))
    assert not env._descent_feasible(arm, faster, np.array([boundary - 1e-6, y, 0.23]))


def test_existing_cross_station_reservation_is_unchanged(env, monkeypatch):
    for slot, x in [(0, -0.25), (1, 0.02)]:
        env.tracks[slot] = Track(
            slot + 1,
            np.array([x, -0.065, 0.23]),
            np.array([0.15, 0, 0]),
            np.pi / 2,
            0,
            0.99,
            0,
            0,
            size=np.array([0.05, 0.04, 0.05]),
        )
        env.belief[slot] = 1
    # Isolate the joint action guard from deadline/availability decisions.
    monkeypatch.setattr(
        env,
        "_candidate",
        lambda arm, slot: (slot in (0, 1) and arm in (0, 2), 0.1, 2.0),
    )
    observation = env._observe_multi()
    choices = env.action_choices
    combined = np.flatnonzero(
        (choices[:, 0] == 0)
        & (choices[:, 2] == 1)
        & (choices[:, 1] == 10)
        & (choices[:, 3] == 10)
    )
    assert len(combined) == 1 and not observation["mask"][combined[0]]
    env.tracks[1] = replace(env.tracks[1], position=np.array([0.10, -0.065, 0.23]))
    assert env._observe_multi()["mask"][combined[0]]


def test_extended_deadline_is_opt_in_and_rejects_incompatible_arms():
    assert not Config().extended_pick_deadline
    with pytest.raises(ValueError, match="extended_pick_deadline must be bool"):
        Config(extended_pick_deadline=1)
    with pytest.raises(ValueError, match="four adaptive finger arms"):
        Config(extended_pick_deadline=True)
    with pytest.raises(ValueError, match="four adaptive finger arms"):
        Config(num_arms=4, strategy="adaptive", gripper_mode="vacuum", extended_pick_deadline=True)


@pytest.mark.parametrize(
    "arm,y,expected",
    [(0, -0.045, -0.02), (0, -0.080, -0.02), (1, 0.045, -0.02),
     (1, 0.080, -0.02), (0, -0.0449, -0.10), (0, -0.0801, -0.10),
     (0, 0.065, -0.10), (1, -0.065, -0.10), (2, -0.065, 0.82),
     (3, 0.065, 0.82)],
)
def test_completion_bound_only_extends_upstream_calibrated_near_band(env, arm, y, expected):
    env.config = replace(env.config, extended_pick_deadline=True)
    windows = env.windows.copy()
    assert env.completion_window_high(arm, np.array([-0.44, y, 0.23])) == pytest.approx(expected)
    np.testing.assert_array_equal(env.windows, windows)


def candidate_track(x=-0.44, y=-0.065, vy=0.0):
    return Track(
        1, np.array([x, y, 0.23]), np.array([0.2, vy, 0]), np.pi / 2,
        0, 0.99, 0, 0, size=np.array([0.05, 0.04, 0.05]),
    )


def test_completion_deadline_admits_useful_near_task_without_changing_eta(env):
    env.config = replace(env.config, control_speed=0.65, use_dispatch_preferences=False)
    env.tracks[0] = candidate_track()
    env.belief[0] = 1
    before = env._candidate(0, 0)
    assert not before[0]
    env.config = replace(env.config, extended_pick_deadline=True)
    after = env._candidate(0, 0)
    assert after[0]
    assert after[1] == before[1]
    assert after[2] - before[2] == pytest.approx(0.40)
    assert after[2] > after[1] + 0.25
    # The original raw window still closes at x=-.10.
    env.tracks[0] = candidate_track(x=-0.099)
    assert not env._candidate(0, 0)[0]


def test_prediction_leaving_lateral_calibration_does_not_get_extra_deadline(env):
    env.config = replace(env.config, control_speed=0.65, use_dispatch_preferences=False)
    env.tracks[0] = candidate_track(vy=0.02)
    env.belief[0] = 1
    old = env._candidate(0, 0)
    env.config = replace(env.config, extended_pick_deadline=True)
    assert env.completion_window_high(0, env.tracks[0].position) == pytest.approx(-0.02)
    new = env._candidate(0, 0)
    assert new == old


def test_deadline_flag_does_not_extend_descent_envelope_or_double_subtract_margin(env):
    track = candidate_track()
    positions = [np.array([x, -0.065, 0.23]) for x in (-0.231, -0.229)]
    before = [env._descent_feasible(0, track, p) for p in positions]
    assert before == [True, False]
    env.config = replace(env.config, extended_pick_deadline=True)
    assert [env._descent_feasible(0, track, p) for p in positions] == before


def test_extended_deadline_keeps_existing_approach_hard_stop(env, monkeypatch):
    env.config = replace(env.config, extended_pick_deadline=True)
    env.tracks[0] = candidate_track(x=-0.019)
    env.belief[0] = 2
    env.target[0] = 0
    env._set_phase(0, "approach")
    monkeypatch.setattr(env, "_descent_feasible", lambda *args: True)
    monkeypatch.setattr(env, "_route_goal", lambda arm, goal, loaded, **kwargs: (np.asarray(goal), True))
    monkeypatch.setattr(env.ik, "command", lambda *args, **kwargs: None)
    env._control()
    assert env.phase[0] == "recover" and env.retries == 1
