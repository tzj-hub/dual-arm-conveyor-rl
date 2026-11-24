"""Short physical regression constructed from measured descent-entry data.

This is not the exact 100-object replay: unrelated objects are absent and the
command state is reconstructed from its recorded reference. The frozen full
episode remains necessary to validate final metrics. No live-state mutation
occurs after the one initial fixture setup.
"""
from pathlib import Path
import argparse
import json
import math
import sys

sys.dont_write_bytecode = True
import mujoco
import numpy as np
from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv, State
from dual_arm_conveyor.ik import downward_quaternion


def run_fixture(fixture, guarded, seconds=1.5, lead_scale=None, lead_feedback=0.0,
                belt_speed=None, grasp_yaw=None):
    options = dict(fixture['config'], guarded_descent=guarded)
    if belt_speed is not None:
        options['belt_speed'] = belt_speed
    env = ConveyorEnv(Config(**options))
    entry = fixture['entry']
    d, ids, m = env.data, env.ids, env.model
    if lead_scale is not None:
        def exploratory_lead(arm, track, velocity, remaining, position=None):
            error = d.site_xpos[ids.ee_sites[arm], :2] - track.predict(d.time)[:2]
            return np.asarray(velocity)[:2]*remaining*lead_scale - lead_feedback*error
        env._descent_lead = exploratory_lead
    try:
        env.states[:] = State.PENDING
        env.serial[:] = -1
        env.owner[:] = -1
        for i, q in enumerate(ids.cube_qpos):
            d.qpos[q:q+7] = [3+i*.1, 0, -1, 1, 0, 0, 0]
            d.qvel[ids.cube_dofs[i]:ids.cube_dofs[i]+6] = 0
            m.geom_contype[ids.cube_geoms[i]] = m.geom_conaffinity[ids.cube_geoms[i]] = 0
            m.body_gravcomp[ids.cube_bodies[i]] = 1
        for a, row in enumerate(entry['arms']):
            d.qpos[ids.arm_qpos[a]] = row['qpos']
            d.qvel[ids.arm_dofs[a]] = row['qvel']
            d.qpos[ids.finger_qpos[a]] = row['finger_qpos']
            d.ctrl[ids.arm_actuators[a]] = row['qpos']
            env.idle_anchors[a] = row['ee']
        d.time = entry['time']
        i = 95
        obj = entry['object95']
        d.qpos[ids.cube_qpos[i]:ids.cube_qpos[i]+7] = [*obj['position'], *obj['quaternion']]
        d.qvel[ids.cube_dofs[i]:ids.cube_dofs[i]+3] = obj['velocity']
        if belt_speed is not None:
            d.qvel[ids.cube_dofs[i]:ids.cube_dofs[i]+3] = [belt_speed, 0, 0]
        m.geom_contype[ids.cube_geoms[i]] = m.geom_conaffinity[ids.cube_geoms[i]] = 1
        m.body_gravcomp[ids.cube_bodies[i]] = 0
        env.states[i] = State.RESERVED
        env.serial[i] = 95
        env.types[i] = 1
        env.owner[i] = 1
        env.truth_target[1] = i
        env.tracks = [None]*10
        env.track_slots = {}
        env.belief[:] = 0
        mujoco.mj_forward(m, d)
        row = dict(entry['arms'][1])
        if grasp_yaw is not None:
            row['grasp_yaw'] = grasp_yaw
            q = env.ik.solve_pose(d, 1, row['ee'], downward_quaternion(grasp_yaw))
            ep, er = env.ik.pose_error(d, 1, q, row['ee'], downward_quaternion(grasp_yaw))
            assert ep < .003 and er < .03, (ep, er)
            d.qpos[ids.arm_qpos[1]] = q
            mujoco.mj_forward(m, d)
        if belt_speed is not None or grasp_yaw is not None:
            speed = options['belt_speed']
            jacp, jacr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
            mujoco.mj_jacSite(m,d,jacp,jacr,ids.ee_sites[1])
            jac = np.r_[jacp[:,ids.arm_dofs[1]],jacr[:,ids.arm_dofs[1]]]
            d.qvel[ids.arm_dofs[1]] = np.linalg.pinv(jac) @ np.array([speed,0,0,0,0,0])
            row['cartesian_reference'] = (d.site_xpos[ids.ee_sites[1]] + [.045*speed,0,0]).tolist()
            row['reference_velocity'] = [speed,0,0]
            row['joint_reference'] = d.qpos[ids.arm_qpos[1]].tolist()
        env.ik.reset(d)
        env._sense()
        slot = env.track_slots[95]
        env._assign(1, slot)
        env.grasp_yaw[1] = row['grasp_yaw']
        env.grasp_sizes[1] = row['grasp_size']
        env.grasp_offsets[1] = row['grasp_offset']
        env.phase[1] = 'descend'
        env.phase_start[1] = d.time - row['phase_elapsed']
        env.ik.cart_targets[1] = row['cartesian_reference']
        env.ik.cart_velocities[1] = row['reference_velocity']
        env.ik.q_targets[1] = row['joint_reference']
        d.ctrl[ids.arm_actuators[1]] = row['joint_reference']
        env.eval_queue.clear()
        env.events.clear()
        initial = float(d.time)
        samples, phases = [], []
        previous = None
        max_tilt = max_open_touch = 0.0
        low_clearances = []
        for step in range(round(seconds / env.config.physics_dt)):
            env._sense()
            env._control()
            mujoco.mj_step(m, d)
            env._outcomes()
            env._collisions()
            phase = env.phase[1]
            if phase != previous:
                phases.append({'time':float(d.time-initial), 'phase':phase})
                previous = phase
            q = d.qpos[ids.cube_qpos[i]+3:ids.cube_qpos[i]+7]
            tilt = math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
            if phase == 'descend':
                max_tilt = max(max_tilt, tilt)
                max_open_touch = max(max_open_touch, float(np.max(d.sensordata[ids.finger_touch_adr[1]])))
                track = env.tracks[slot]
                if d.site_xpos[ids.ee_sites[1],2] <= env.config.belt_top + track.size[2] + .005:
                    axis = np.array([-math.sin(env.grasp_yaw[1]),math.cos(env.grasp_yaw[1])])
                    error = d.site_xpos[ids.ee_sites[1],:2] - env.cube_position(i)[:2]
                    width = env._grasp_size_at_yaw(track,env.grasp_yaw[1])[1]
                    opening = np.sum(d.qpos[ids.finger_qpos[1]])
                    low_clearances.append(float(.5*(opening-width) - abs(np.dot(error,axis))))
            if step % 8 == 0:
                samples.append({'time':float(d.time-initial), 'phase':phase,
                                'ee':d.site_xpos[ids.ee_sites[1]].tolist(), 'object':env.cube_position(i).tolist(),
                                'tilt_deg':float(tilt), 'touch':d.sensordata[ids.finger_touch_adr[1]].tolist()})
        return {'guarded':guarded,'phases':phases,'robot_collision_steps':env.robot_collision_steps,
                'inter_arm_collision_steps':env.collision_steps,'max_descent_tilt_deg':max_tilt,
                'max_open_touch_n':max_open_touch,'guard_stops':dict(env.descent_guard_stops),
                'min_low_descent_jaw_clearance_m':min(low_clearances) if low_clearances else None,
                'holding':bool(env.holding[1]),'gripper_has_object':bool(env._gripper_has_object(1)),
                'events':list(env.events),'samples':samples}
    finally:
        env.close()
