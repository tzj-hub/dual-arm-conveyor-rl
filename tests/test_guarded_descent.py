"""Moving wide-part entry: actual pickup and sensor-only disturbance handling."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from descent_case import run_fixture
from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.vision import Track


@pytest.fixture(scope='module')
def recorded_entry():
    return json.loads((Path(__file__).parent/'fixtures/descent_810001_entry.json').read_text())


def test_known_open_pad_sweep_is_physically_reproduced_then_grasped(recorded_entry):
    baseline = run_fixture(recorded_entry, False, seconds=.9)
    assert baseline['robot_collision_steps'] > 0
    assert baseline['max_descent_tilt_deg'] > 45
    assert not baseline['gripper_has_object']
    guarded = run_fixture(recorded_entry, True)
    assert guarded['robot_collision_steps'] == guarded['inter_arm_collision_steps'] == 0
    assert guarded['max_open_touch_n'] == 0
    assert guarded['max_descent_tilt_deg'] < .1
    assert guarded['holding'] and guarded['gripper_has_object']
    assert [p['phase'] for p in guarded['phases']][:3] == ['descend','close','lift']
    assert not guarded['guard_stops']


@pytest.mark.parametrize('speed', [.15,.2,.3])
@pytest.mark.parametrize('yaw', [-math.pi/2,-math.pi/2+.4])
def test_tracking_pickup_preserves_low_jaw_clearance_across_speed_and_yaw(recorded_entry, speed, yaw):
    result = run_fixture(recorded_entry, True, belt_speed=speed, grasp_yaw=yaw)
    assert result['robot_collision_steps'] == result['inter_arm_collision_steps'] == 0
    assert result['max_open_touch_n'] == 0
    assert result['min_low_descent_jaw_clearance_m'] > .0015
    assert result['holding'] and result['gripper_has_object']
    assert not result['guard_stops']


@pytest.mark.parametrize('fault', ['touch','rise','timeout'])
def test_native_guard_holds_open_and_recovers_from_observable_disturbance(monkeypatch, fault):
    env = ConveyorEnv(Config(sensor_mode='oracle', total_objects=1, strategy='adaptive', guarded_descent=True))
    try:
        now = 2.0
        env.data.time = now
        p = np.array([-.3,0,env.config.belt_top+.025])
        if fault == 'rise':
            p[2] += .012
        track = Track(9_000_001, p, np.array([.2,0,0]),0,0,1,now,now,
                      size=np.array([.05,.04,.05]))
        env.tracks[0] = track
        env.track_slots = {track.track_id:0}
        env.target[0] = 0
        env.belief[0] = 2
        env.phase[0] = 'descend'
        env.phase_start[0] = now - (1.3 if fault == 'timeout' else .1)
        if fault == 'touch':
            env.data.sensordata[env.ids.finger_touch_adr[0]] = [.7,0]

        def forbidden_truth(*args, **kwargs):
            raise AssertionError('Descent guard accessed hidden workpiece truth')

        monkeypatch.setattr(env,'cube_position',forbidden_truth)
        monkeypatch.setattr(env,'cube_velocity',forbidden_truth)
        env._control()
        assert env.phase[0] == 'recover'
        assert env.ik._holding[0] and env.hold_requested[0]
        assert env.ik.gripper_targets[0] == pytest.approx(.04)
        assert sum(env.descent_guard_stops.values()) == 1
        assert env.routes[0] is None
    finally:
        env.close()
