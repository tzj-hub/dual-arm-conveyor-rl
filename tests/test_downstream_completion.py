"""Limited downstream completion room preserves actual grasp/path checks."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv, State
from dual_arm_conveyor.forecast import visual_forecast_state
from dual_arm_conveyor.ik import downward_quaternion
from dual_arm_conveyor.vision import Track


@pytest.fixture
def env():
    root = Path(__file__).resolve().parents[1]
    scene = json.loads((root / 'tests/fixtures/baseline_scene.json').read_text())
    e = ConveyorEnv(replace(Config(**scene), total_objects=1, pool_size=8, bin_capacity=20,
                            downstream_completion=True))
    yield e
    e.close()


def track_at(arm=2, x=.55, y=None, vy=0.0, height=.05):
    if y is None:
        y = (-1 if arm % 2 == 0 else 1) * .065
    return Track(991, np.array([x, y, .205+height/2]), np.array([.2,vy,0]),
                 np.pi/2, arm%2, .99, 0, 0, size=np.array([.05,.04,height]))


def put_above(e, arm, track):
    point = track.position.copy()
    point[2] = e.config.belt_top + track.half_height + e._grasp_offset(track) + e.approach_clearance
    yaw = np.pi/2 if arm%2 == 0 else -np.pi/2
    q = e.ik.solve_pose(e.data, arm, point, downward_quaternion(yaw))
    e.data.qpos[e.ids.arm_qpos[arm]] = q
    e.data.ctrl[e.ids.arm_actuators[arm]] = q
    e.data.qvel[e.ids.arm_dofs[arm]] = 0
    mujoco.mj_forward(e.model, e.data)
    e.ik.reset(e.data)
    e.idle_anchors[arm] = e.data.site_xpos[e.ids.ee_sites[arm]]
    assert np.linalg.norm(e.data.site_xpos[e.ids.ee_sites[arm]]-point) < .001


@pytest.mark.parametrize('arm,lateral,expected', [
    (2,.045,.88),(2,.080,.88),(3,.045,.88),(3,.080,.88),
    (2,.0449,.82),(3,.0801,.82),(2,-.065,.82),(3,-.065,.82),
    (0,.065,-.02),(1,.065,-.02),
])
def test_completion_extension_preserves_station_band_and_raw_windows(env, arm, lateral, expected):
    side = -1 if arm%2 == 0 else 1
    point = np.array([.55,side*lateral,.23])
    windows = env.windows.copy()
    assert env.completion_window_high(arm, point) == pytest.approx(expected)
    np.testing.assert_array_equal(env.windows, windows)
    env.config = replace(env.config, downstream_completion=False)
    assert env.completion_window_high(arm, point) == pytest.approx(
        -.02 if arm < 2 else .82)


@pytest.mark.parametrize('arm', [2,3])
def test_descent_extension_keeps_full_travel_and_twenty_mm_reserve(env, arm):
    track = track_at(arm)
    horizon = env.descent_time + env.closure_timeout + .20
    boundary = .88 - .2*horizon
    assert env._descent_feasible(arm, track, np.array([boundary-1e-6,track.position[1],.23]))
    assert not env._descent_feasible(arm, track, np.array([boundary+1e-6,track.position[1],.23]))
    env.config = replace(env.config, downstream_completion=False)
    assert not env._descent_feasible(arm, track, np.array([boundary-1e-6,track.position[1],.23]))


def test_lateral_forecast_leaving_band_does_not_borrow_execution_room(env):
    stationary = track_at(x=.65)
    leaving = replace(stationary, velocity=np.array([.2,-.03,0]))
    assert env._descent_feasible(2, stationary, stationary.position)
    assert not env._descent_feasible(2, leaving, leaving.position)
    env.tracks[0],env.belief[0] = leaving,1
    put_above(env,2,leaving)
    extended = env._candidate(2,0)
    env.config = replace(env.config, downstream_completion=False)
    assert env._candidate(2,0) == extended


@pytest.mark.parametrize('arm', [2,3])
def test_native_and_forecast_use_same_new_deadline_without_earlier_dispatch_window(env, arm):
    track = track_at(arm)
    env.tracks[0],env.belief[0] = track,1
    put_above(env,arm,track)
    extended = env._candidate(arm,0)
    state = visual_forecast_state(env)
    assert state.jobs[0].completion_high[1] == .88
    assert extended[0]
    assert (arm,0) in state.current_eta
    env.config = replace(env.config, downstream_completion=False)
    baseline = env._candidate(arm,0)
    assert not baseline[0]
    assert baseline[1] == extended[1]
    assert extended[2]-baseline[2] == pytest.approx(.30)
    env.config = replace(env.config, downstream_completion=True)
    env.tracks[0] = replace(track, position=np.array([.821,track.position[1],track.position[2]]))
    assert not env._candidate(arm,0)[0]


def test_known_short_block_bin_wall_counterexample_is_still_rejected(env):
    arm,y,height,closure_x = 2,-.080,.035,.86
    offset = max(env.config.grasp_offset,height/2-.030)
    low_z = env.config.belt_top + height/2 + offset
    initial_x = closure_x - .2*(.5+env.closure_timeout)
    quaternion = downward_quaternion(0)
    approach = env.planner.plan(env.data,arm,[initial_x,y,low_z+env.approach_clearance],
                                quaternion=quaternion,grasp_offset=offset)
    assert approach.verified
    env.data.qpos[env.ids.arm_qpos[arm]] = approach.joint_paths[-1][-1]
    mujoco.mj_forward(env.model,env.data)
    rejected = env.planner.grasp_exit_feasible(env.data,arm,[closure_x,y,low_z],
        [closure_x+.020,y,env.lift_height],quaternion=quaternion,grasp_offset=offset,
        grasp_position=[closure_x-.2*env.closure_timeout,y,low_z])
    assert not rejected.verified
    assert 'clearance' in rejected.reason and 'binwall' in rejected.reason


@pytest.mark.parametrize('already_reoriented', [False,True])
def test_preview_rejection_cannot_be_overridden_by_extended_deadline(env,monkeypatch,already_reoriented):
    arm=2
    track=track_at(arm,x=.63)
    env.tracks[0],env.belief[0],env.target[arm] = track,2,0
    env.grasp_yaw[arm] = np.pi/2
    env.grasp_offsets[arm] = env._grasp_offset(track)
    env.reoriented[arm] = already_reoriented
    put_above(env,arm,track)
    env._set_phase(arm,'approach')
    assert env._descent_feasible(arm,track,track.position)
    called=[]
    monkeypatch.setattr(env,'_route_goal',lambda a,g,loaded,**kw:(np.asarray(g),True))
    monkeypatch.setattr(env.planner,'plan',lambda *args,**kw:SimpleNamespace(verified=True))
    def reject(*args,**kw):
        called.append(True)
        return SimpleNamespace(verified=False,reason='calibrated bin-wall counterexample')
    monkeypatch.setattr(env.planner,'grasp_exit_feasible',reject)
    monkeypatch.setattr(env.ik,'command',lambda *args,**kw:None)
    env._control()
    assert called
    assert env.phase[arm] == ('recover' if already_reoriented else 'reorient')


@pytest.mark.parametrize('arm',[2,3])
def test_physical_late_start_uses_real_finger_grasp_and_finishes_in_bin(env,arm):
    # Predeclared late-start fixture: a physical box at x=.55, a correctly
    # oriented open hand above it. Only fixture initialization changes qpos;
    # after dispatch, ordinary actuator/contacts and env.step drive everything.
    body_slot = next(i for i,spec in enumerate(env.ids.object_specs) if spec.shape == 'box')
    side = -1 if arm%2 == 0 else 1
    env._admit_object(body_slot,0.0,side*.065,arm%2)
    env.data.qpos[env.ids.cube_qpos[body_slot]] = .55
    mujoco.mj_forward(env.model,env.data)
    env._sense()
    slot = env.track_slots[int(env.serial[body_slot])]
    put_above(env,arm,env.tracks[slot])
    env.config = replace(env.config,downstream_completion=False)
    baseline = env._candidate(arm,slot)
    env.config = replace(env.config,downstream_completion=True)
    extended = env._candidate(arm,slot)
    assert not baseline[0] and extended[0], (baseline,extended)
    choices=[10]*4
    choices[arm]=slot
    action=int(np.ravel_multi_index(tuple(choices),(11,)*4))
    assert env.observe()['mask'][action]
    phases=set()
    physical_hold=False
    def inspect(e):
        nonlocal physical_hold
        phases.add(e.phase[arm])
        physical_hold |= bool(e._gripper_has_object(arm) and e.phase[arm] in ('close','lift','transport'))
    _,_,terminated,truncated,info=env.step(action,physics_callback=inspect)
    for _ in range(119):
        if terminated or truncated:
            break
        _,_,terminated,truncated,info=env.step(env.wait_action,physics_callback=inspect)
    print(json.dumps(dict(event='downstream_late_fixture',arm=arm,baseline=baseline,
        extended=extended,phases=sorted(phases),physical_contact_hold=physical_hold,
        sorted=info['sorted'],missed=info['missed'],dropped=info['dropped'],
        robot_collision_steps=info['robot_collision_steps'],
        inter_arm_collision_steps=info['inter_arm_collision_steps'])),flush=True)
    assert env.vacuum is None and env.config.gripper_mode == 'fingers'
    # The fixture starts already aligned above the box, so approach can end
    # inside the first control call before the post-physics callback observes
    # it. Dispatch is checked by its real event, rather than requiring a
    # sampled dwell in a legitimately instantaneous phase.
    assert any(event['event'] == 'dispatch' and event['arm'] == arm for event in env.events)
    assert physical_hold and {'descend','close','lift','release'} <= phases
    assert info['sorted'] == 1 and info['missed'] == info['dropped'] == 0
    assert env.states[body_slot] == State.SORTED
    assert info['robot_collision_steps'] == info['inter_arm_collision_steps'] == 0
