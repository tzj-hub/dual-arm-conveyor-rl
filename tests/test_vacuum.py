"""Bounded suction physics, contact-only sealing, release and overload tests."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.vacuum import VacuumGripper, VacuumParameters


def scene(
    *,
    mass=0.05,
    free_cup=False,
    gravity=True,
    lateral=0.0,
    dimensions=(0.05, 0.05, 0.05),
):
    attachment = '<freejoint name="cup_joint"/>' if free_cup else ""
    mocap = "" if free_cup else 'mocap="true"'
    gravity_text = "0 0 -9.81" if gravity else "0 0 0"
    half_sizes = " ".join(str(value / 2) for value in dimensions)
    xml = f'''<mujoco>
      <option timestep=".005" gravity="{gravity_text}" integrator="implicitfast"/>
      <worldbody>
        <geom name="table" type="plane" size="1 1 .1"/>
        <body name="workpiece" pos="0 0 {dimensions[2] / 2}"><freejoint name="object_joint"/>
          <geom name="workpiece_geom" type="box" size="{half_sizes}" mass="{mass}" friction=".7 .01 .001"/>
        </body>
        <body name="cup" pos="{lateral} 0 {dimensions[2] + 0.0019}" quat="0 1 0 0" {mocap}>
          {attachment}
          <geom name="cup_geom" type="cylinder" size=".012 .002" mass=".03" friction=".7 .01 .001"/>
          <site name="cup_site" pos="0 0 .002"/>
        </body>
      </worldbody>
    </mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def identifier(kind, name):
        return mujoco.mj_name2id(model, kind, name)

    ids = SimpleNamespace(
        vacuum_sites=np.array([identifier(mujoco.mjtObj.mjOBJ_SITE, "cup_site")]),
        vacuum_geoms=np.array([identifier(mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")]),
        vacuum_bodies=np.array([identifier(mujoco.mjtObj.mjOBJ_BODY, "cup")]),
        cube_geoms=np.array([identifier(mujoco.mjtObj.mjOBJ_GEOM, "workpiece_geom")]),
    )
    return model, data, ids


def warm_seal(device, data, ticks=20):
    device.set_enabled(0, True)
    # A controlled stationary-contact fixture isolates pressure/wrench tests.
    for _ in range(ticks):
        device.step(data)
        data.time += device.model.opt.timestep
    assert device.sensor(0)["sealed"]


def test_seal_requires_enabled_contact_and_all_rim_samples_on_one_workpiece():
    model, data, ids = scene(gravity=False)
    device = VacuumGripper(model, ids)
    assert any(ids.vacuum_geoms[0] in contact.geom for contact in data.contact)
    assert np.count_nonzero(device.step(data)) == 0
    assert not device.sensor(0)["sealed"]
    data.time += model.opt.timestep
    warm_seal(device, data)
    assert device.sensor(0)["pressure_pa"] > 5000
    assert set(device.sensor(0)) == {"sealed", "pressure_pa", "enabled"}
    # The cup still touches the box, but its rim overhangs the edge.
    model, data, ids = scene(gravity=False, lateral=0.022)
    device = VacuumGripper(model, ids)
    device.set_enabled(0, True)
    for _ in range(20):
        device.step(data)
        data.time += model.opt.timestep
    assert not device.sensor(0)["sealed"]
    assert device.sensor(0)["pressure_pa"] == 0.0


def test_enabled_cup_cannot_seal_to_a_workpiece_outside_one_mm_sensor_range():
    model, data, ids = scene(gravity=False)
    data.mocap_pos[0, 2] += 0.003
    mujoco.mj_forward(model, data)
    device = VacuumGripper(model, ids)
    device.set_enabled(0, True)
    for _ in range(10):
        assert not np.any(device.step(data))
        data.time += model.opt.timestep
    assert not device.sensor(0)["sealed"]


def test_device_returns_fresh_bounded_forces_without_editing_state_or_controls():
    model, data, ids = scene(gravity=False)
    device = VacuumGripper(model, ids)
    warm_seal(device, data)
    data.mocap_pos[0, :3] += [0.004, 0, 0.004]
    data.qvel[:3] = [-1.0, 0.0, -1.0]
    mujoco.mj_forward(model, data)
    before = [
        data.qpos.copy(),
        data.qvel.copy(),
        data.ctrl.copy(),
        data.qfrc_applied.copy(),
    ]
    forces = device.step(data)
    assert np.isfinite(forces).all() and np.linalg.norm(forces) > 0
    force, torque = device._last_wrenches[0, :3], device._last_wrenches[0, 3:]
    normal = data.site_xmat[ids.vacuum_sites[0]].reshape(3, 3)[:, 2]
    capacity = min(5.0, device.sensor(0)["pressure_pa"] * device.areas[0])
    assert abs(force @ normal) <= capacity + 1e-12
    assert np.linalg.norm(force - (force @ normal) * normal) <= 0.7 * capacity + 1e-12
    assert np.linalg.norm(torque) <= 0.02 + 1e-12
    for previous, actual in zip(
        before, (data.qpos, data.qvel, data.ctrl, data.qfrc_applied)
    ):
        np.testing.assert_array_equal(previous, actual)
    assert not np.any(model.eq_type == int(mujoco.mjtEq.mjEQ_WELD))


@pytest.mark.parametrize(
    "offset,reason", [([0, 0, 0.010], "gap"), ([0.010, 0, 0], "shear")]
)
def test_excess_gap_or_shear_breaks_seal_instead_of_teleporting_object(offset, reason):
    model, data, ids = scene(gravity=False)
    device = VacuumGripper(model, ids)
    warm_seal(device, data)
    data.mocap_pos[0] += offset
    mujoco.mj_forward(model, data)
    original = data.qpos.copy()
    assert not np.any(device.step(data))
    assert not device.sensor(0)["sealed"]
    assert device.diagnostics()["last_break"] == [reason]
    np.testing.assert_array_equal(data.qpos, original)


def test_free_cup_receives_equal_opposite_reaction_force_and_total_moment():
    model, data, ids = scene(free_cup=True, gravity=False)
    device = VacuumGripper(model, ids)
    warm_seal(device, data)
    # Lift the independent cup initial condition, leaving no physical contact.
    cup_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cup_joint")
    qpos = model.jnt_qposadr[cup_joint]
    dof = model.jnt_dofadr[cup_joint]
    data.qpos[qpos : qpos + 3] += [0.003, 0, 0.003]
    mujoco.mj_forward(model, data)
    forces = device.step(data)
    np.testing.assert_allclose(forces[:3] + forces[dof : dof + 3], 0, atol=1e-12)
    world_torque = np.zeros(3)
    for joint_name, body_name in [("object_joint", "workpiece"), ("cup_joint", "cup")]:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        index = model.jnt_dofadr[joint]
        rotation = data.xmat[body].reshape(3, 3)
        world_torque += rotation @ forces[index + 3 : index + 6] + np.cross(
            data.xpos[body], forces[index : index + 3]
        )
    np.testing.assert_allclose(world_torque, 0, atol=1e-10)


def test_excess_tilt_breaks_a_seal_even_when_its_centre_has_not_moved():
    model, data, ids = scene(gravity=False)
    device = VacuumGripper(model, ids)
    warm_seal(device, data)
    face = data.site_xpos[ids.vacuum_sites[0]].copy()
    turn = np.array([np.cos(0.25), np.sin(0.25), 0.0, 0.0])
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, turn, data.mocap_quat[0].copy())
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    data.mocap_quat[0] = quat
    data.mocap_pos[0] = face - rotation.reshape(3, 3) @ [0, 0, 0.002]
    mujoco.mj_forward(model, data)
    assert not np.any(device.step(data))
    assert device.diagnostics()["last_break"] == ["tilt"]
    assert not device.sensor(0)["sealed"]


@pytest.mark.parametrize("shape", ["box", "pentagonal_prism", "hexagonal_prism"])
def test_real_panda_cup_geometry_seals_to_supported_flat_prism_faces(shape):
    from dual_arm_conveyor.config import Config
    from dual_arm_conveyor.model import build_scene, reset_robot_pose

    model, ids = build_scene(
        Config(strategy="adaptive", num_arms=4, gripper_mode="vacuum")
    )
    data = mujoco.MjData(model)
    reset_robot_pose(model, data, ids)
    index = next(i for i, spec in enumerate(ids.object_specs) if spec.shape == shape)
    spec = ids.object_specs[index]
    face = data.site_xpos[ids.vacuum_sites[0]].copy()
    joint = ids.cube_qpos[index]
    data.qpos[joint : joint + 7] = [
        face[0],
        face[1],
        face[2] + 0.0002 - spec.half_height,
        np.cos(0.35),
        0,
        0,
        np.sin(0.35),
    ]
    mujoco.mj_forward(model, data)
    device = VacuumGripper(model, ids)
    warm_seal(device, data)
    assert device.sensor(0)["sealed"]
    assert not any(device.sensor(arm)["sealed"] for arm in (1, 2, 3))


@pytest.mark.parametrize("mass", [0.02, 0.05, 0.10])
def test_physical_lift_and_release_of_dynamic_workpiece_at_production_timestep(mass):
    model, data, ids = scene(mass=mass)
    device = VacuumGripper(model, ids)
    device.set_enabled(0, True)
    initial_height = data.mocap_pos[0, 2]
    for step in range(200):
        time = step * model.opt.timestep
        progress = np.clip((time - 0.10) / 0.5, 0, 1)
        data.mocap_pos[0, 2] = initial_height + 0.05 * (
            3 * progress**2 - 2 * progress**3
        )
        mujoco.mj_forward(model, data)
        data.qfrc_applied[:] = device.step(data)
        mujoco.mj_step(model, data)
    assert device.sensor(0)["sealed"], device.diagnostics()
    assert data.qpos[2] > 0.065
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    device.set_enabled(0, False)
    for _ in range(120):
        data.qfrc_applied[:] = device.step(data)
        mujoco.mj_step(model, data)
    assert not device.sensor(0)["sealed"]
    assert data.qpos[2] == pytest.approx(0.025, abs=0.003)
    assert device.sensor(0)["pressure_pa"] < 1.0


def test_pressure_limit_cannot_lift_an_overweight_body_and_it_detaches():
    model, data, ids = scene(mass=1.0)
    device = VacuumGripper(model, ids)
    device.set_enabled(0, True)
    initial_height = data.mocap_pos[0, 2]
    for step in range(180):
        progress = np.clip((step * model.opt.timestep - 0.1) / 0.5, 0, 1)
        data.mocap_pos[0, 2] = initial_height + 0.05 * (
            3 * progress**2 - 2 * progress**3
        )
        mujoco.mj_forward(model, data)
        data.qfrc_applied[:] = device.step(data)
        mujoco.mj_step(model, data)
    assert not device.sensor(0)["sealed"]
    assert device.diagnostics()["break_count"][0] >= 1
    assert data.qpos[2] < 0.03


def test_low_pressure_and_reset_never_preserve_a_fake_hold():
    model, data, ids = scene(gravity=False)
    device = VacuumGripper(
        model, ids, replace(VacuumParameters(), max_pressure_pa=100.0)
    )
    device.set_enabled(0, True)
    for _ in range(30):
        device.step(data)
        data.time += model.opt.timestep
    assert not device.sensor(0)["sealed"]
    assert device.diagnostics()["last_break"] == ["pressure"]
    device.reset()
    assert device.sensor(0) == dict(sealed=False, pressure_pa=0.0, enabled=False)
    assert device.diagnostics()["break_count"] == [0]


@pytest.mark.parametrize("mass", [0.02, 0.06, 0.10])
@pytest.mark.parametrize("offset", [0.0, 0.0035, 0.007])
@pytest.mark.parametrize(
    "axis,acceleration", [(0, -2.5), (0, 0.0), (0, 2.5), (1, -2.5), (1, 2.5)]
)
def test_implicit_small_inertia_offcentre_cup_is_stable_in_both_lateral_axes(
    mass, offset, axis, acceleration
):
    # This minimum-size 20 g body with a 7 mm offset made the explicit model
    # reach >50 rad/s and detach. A centred vertical lift did not expose it.
    model, data, ids = scene(mass=mass, dimensions=(0.04, 0.035, 0.035))
    data.mocap_pos[0, axis] = offset
    mujoco.mj_forward(model, data)
    device = VacuumGripper(
        model,
        ids,
        replace(VacuumParameters(), angular_stiffness=0.10, angular_damping=0.001),
    )
    device.set_enabled(0, True)
    start = data.mocap_pos[0].copy()
    maximum_angular_speed = 0.0
    maximum_tilt = 0.0

    def smooth(now, begin):
        progress = np.clip((now - begin) / 0.5, 0, 1)
        return 3 * progress**2 - 2 * progress**3

    for step in range(400):
        now = step * model.opt.timestep
        data.mocap_pos[0] = start
        data.mocap_pos[0, 2] += 0.05 * smooth(now, 0.1)
        # Cubic displacement has peak acceleration 6*distance/duration².
        data.mocap_pos[0, axis] += (acceleration * 0.5**2 / 6) * (
            smooth(now, 0.7) - smooth(now, 1.3)
        )
        mujoco.mj_forward(model, data)
        data.qfrc_applied[:] = device.step(data)
        mujoco.mj_step(model, data)
        maximum_angular_speed = max(
            maximum_angular_speed, float(np.linalg.norm(data.qvel[3:6]))
        )
        rotation = data.xmat[model.geom_bodyid[ids.cube_geoms[0]]].reshape(3, 3)
        maximum_tilt = max(
            maximum_tilt, float(np.arccos(np.clip(rotation[2, 2], -1, 1)))
        )
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert np.linalg.norm(device._last_wrenches[0, 3:]) <= 0.02 + 1e-12
    assert device.sensor(0)["sealed"], device.diagnostics()
    assert device.diagnostics()["break_count"] == [0]
    assert data.qpos[2] > 0.06
    assert maximum_angular_speed < 8.0
    assert maximum_tilt < np.radians(20)
