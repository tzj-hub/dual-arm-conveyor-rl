"""Completion room must reach the learned scheduler's final action mask.

Use real controller/forecast/planner code with initial robot kinematics. No
physics rollout or policy training is required to test the dispatch boundary.
"""
from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.future_plan import KEEP, OPTIONS, planner_for
from dual_arm_conveyor.ik import downward_quaternion
from dual_arm_conveyor.scheduler import encode_observation
from dual_arm_conveyor.vision import Track


@pytest.fixture
def env():
    root = Path(__file__).resolve().parents[1]
    scene = json.loads((root / 'tests/fixtures/baseline_scene.json').read_text())
    simulation = ConveyorEnv(replace(Config(**scene), total_objects=1, pool_size=8,
                                    bin_capacity=20, downstream_completion=True))
    yield simulation
    simulation.close()


def install_track(env, arm, *, x=.55, vy=0., put_above=True):
    side = -1 if arm % 2 == 0 else 1
    track = Track(991, np.array([x, side * .065, .23]), np.array([.2, vy, 0.]),
                  np.pi / 2, arm % 2, .99, 0., 0., size=np.array([.05, .04, .05]))
    env.tracks[0], env.belief[0] = track, 1
    if put_above:
        point = track.position.copy()
        point[2] = env.config.belt_top + track.half_height + env._grasp_offset(track) + env.approach_clearance
        put_tool(env, arm, point)
    return track


def put_tool(env, arm, point):
    yaw = np.pi / 2 if arm % 2 == 0 else -np.pi / 2
    q = env.ik.solve_pose(env.data, arm, point, downward_quaternion(yaw))
    env.data.qpos[env.ids.arm_qpos[arm]] = q
    env.data.ctrl[env.ids.arm_actuators[arm]] = q
    env.data.qvel[env.ids.arm_dofs[arm]] = 0.
    mujoco.mj_forward(env.model, env.data)
    env.ik.reset(env.data)
    actual = env.data.site_xpos[env.ids.ee_sites[arm]].copy()
    assert np.linalg.norm(actual - point) < .001
    env.idle_anchors[arm] = actual


def action_for(arm):
    return KEEP * OPTIONS + (arm % 2) * 10


@pytest.mark.parametrize('arm', [2, 3])
def test_downstream_extension_reaches_proposal_actor_mask_and_native_commit(env, arm):
    install_track(env, arm)
    planner = planner_for(env)
    selected = action_for(arm)

    env.config = replace(env.config, downstream_completion=False)
    baseline = env._candidate(arm, 0)
    _, proposals = planner.snapshot()
    assert not baseline[0] and (arm, 0) not in proposals
    assert not planner.choices_and_mask(proposals)[1][selected]
    assert not encode_observation(env)['mask'][selected]

    env.config = replace(env.config, downstream_completion=True)
    native = env.observe()
    extended = env._candidate(arm, 0)
    _, proposals = planner.snapshot()
    assert extended[0]
    proposal = proposals[arm, 0]
    assert proposal.start == 0.
    assert proposal.deadline == pytest.approx(extended[2])
    assert proposal.deadline - baseline[2] == pytest.approx(.30)
    assert proposal.capture + .25 <= proposal.deadline
    choices, mask = planner.choices_and_mask(proposals)
    assert mask[selected] and choices[selected, arm] == 0
    assert encode_observation(env)['mask'][selected]

    command = planner.commit(selected, native_observation=native)
    assert command != env.wait_action and native['mask'][command]
    expected = [10] * 4
    expected[arm] = 0
    assert env.action_choices[command].tolist() == expected
    assert planner.metrics()['dispatch_commands'] == 1
    # Committing an intent emits a native command; it does not mutate the
    # physical task reservation before the environment executes that command.
    assert env.target[arm] == -1 and env.belief[0] == 1


@pytest.mark.parametrize('arm', [2, 3])
def test_lateral_exit_removes_extension_in_final_future_mask(env, arm):
    side = -1 if arm % 2 == 0 else 1
    install_track(env, arm, vy=side * .03)
    planner = planner_for(env)
    _, proposals = planner.snapshot()
    assert not env._candidate(arm, 0)[0]
    assert (arm, 0) not in proposals
    assert not planner.choices_and_mask(proposals)[1][action_for(arm)]


@pytest.mark.parametrize('arm', [2, 3])
def test_completion_room_does_not_extend_raw_assignment_window(env, arm):
    install_track(env, arm, x=.821)
    planner = planner_for(env)
    _, proposals = planner.snapshot()
    assert env.completion_window_high(arm, env.tracks[0].position) == .88
    assert env.windows[arm, 1] == .82
    assert not env._candidate(arm, 0)[0] and (arm, 0) not in proposals
    assert not planner.choices_and_mask(proposals)[1][action_for(arm)]


@pytest.mark.parametrize('arm', [2, 3])
def test_valid_extended_proposal_still_obeys_final_native_safety_mask(env, arm):
    install_track(env, arm)
    planner = planner_for(env)
    _, proposals = planner.snapshot()
    assert planner.choices_and_mask(proposals)[1][action_for(arm)]
    native = env.observe()
    native['mask'][:] = False
    native['mask'][env.wait_action] = True
    assert planner.commit(action_for(arm), native_observation=native) == env.wait_action
    assert planner.metrics()['dispatch_commands'] == 0
    assert planner.metrics()['safety_deferred_ticks'] == 1


def test_cross_station_reservation_still_defers_native_dispatch(env):
    install_track(env, 2, x=-.14, put_above=False)
    # An upstream hand has lifted its own object at the overlapping edge of
    # the stations. Only robot encoders and controller reservations are used.
    put_tool(env, 0, np.array([-.02, -.065, env.lift_height]))
    env.phase[0] = 'lift'
    env.phase_start[0] = env.data.time
    env.target[0], env.belief[1], env.holding[0], env.drop_kind[1] = 1, 3, True, 0
    assert env._candidate(2, 0)[0]
    native = env.observe()
    command = int(np.ravel_multi_index((10, 10, 0, 10), (11,) * 4))
    assert not native['mask'][command]
    planner = planner_for(env)
    state, proposals = planner.snapshot()
    assert any(r.station == 0 for r in state.reservations)
    if (2, 0) in proposals:
        assert proposals[2, 0].start > 0.
        assert planner.commit(action_for(2), native_observation=native) == env.wait_action
        assert planner.metrics()['dispatch_commands'] == 0
    else:
        assert not planner.choices_and_mask(proposals)[1][action_for(2)]
