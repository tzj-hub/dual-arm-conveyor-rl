"""Physical model checks: varied objects, contact-only grasps and hopper gates."""

import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import mujoco
import numpy as np

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.ik import PandaIK, downward_quaternion
from dual_arm_conveyor.model import BASE_POSITIONS, build_scene, reset_robot_pose
from dual_arm_conveyor.objects import make_object_catalog


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.model, self.ids = build_scene(self.config)
        self.data = mujoco.MjData(self.model)
        self.model.geom_contype[self.ids.cube_geoms] = 0
        self.model.geom_conaffinity[self.ids.cube_geoms] = 0
        self.model.geom_surfacevel[self.ids.belt_geom] = 0
        reset_robot_pose(self.model, self.data, self.ids)
        self.ik = PandaIK(self.model, self.ids)

    def activate(self, index, position, yaw=0.0):
        spec = self.ids.object_specs[index]
        geom, body, q = (
            self.ids.cube_geoms[index],
            self.ids.cube_bodies[index],
            self.ids.cube_qpos[index],
        )
        self.model.geom_contype[geom] = self.model.geom_conaffinity[geom] = 1
        self.model.body_gravcomp[body] = 0
        self.data.qpos[q : q + 7] = [*position, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[self.ids.cube_dofs[index] : self.ids.cube_dofs[index] + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        return spec, body

    def unexpected_contacts(self):
        result = []
        for contact in self.data.contact:
            body1, body2 = (
                self.model.geom_bodyid[contact.geom1],
                self.model.geom_bodyid[contact.geom2],
            )
            robot = (
                self.ids.robot_geom_arm[contact.geom1] >= 0
                or self.ids.robot_geom_arm[contact.geom2] >= 0
            )
            finger_object = (
                body1 in self.ids.finger_bodies and body2 in self.ids.cube_bodies
            ) or (body2 in self.ids.finger_bodies and body1 in self.ids.cube_bodies)
            adjacent = (
                body1 == body2
                or self.model.body_parentid[body1] == body2
                or self.model.body_parentid[body2] == body1
            )
            if robot and not finger_object and not adjacent and contact.dist < -0.0005:
                result.append((int(contact.geom1), int(contact.geom2)))
        return result

    def run_servo(self, arm, target, duration=2.5, gripper=0.04, yaw=0.0):
        contacts = []
        for _ in range(round(duration / self.model.opt.timestep)):
            for a in range(2):
                tool_yaw = (
                    (yaw + (np.pi if a == 1 else 0.0))
                    if a == arm
                    else (np.pi / 2 if a == 0 else -np.pi / 2)
                )
                self.ik.command(
                    self.data,
                    a,
                    target if a == arm else self.ids.home_ee[a],
                    target_quat=downward_quaternion(tool_yaw),
                    gripper=gripper if a == arm else 0.04,
                )
            mujoco.mj_step(self.model, self.data)
            contacts.extend(self.unexpected_contacts())
        return contacts

    def test_catalog_is_seeded_diverse_and_immutable(self):
        specs = make_object_catalog(64, 7)
        self.assertEqual(specs, make_object_catalog(64, 7))
        self.assertNotEqual(specs, make_object_catalog(64, 8))
        self.assertEqual(
            {s.shape for s in specs}, {"box", "pentagonal_prism", "hexagonal_prism"}
        )
        self.assertEqual({s.solref[0] for s in specs}, {0.006, 0.012, 0.025})
        for spec in specs:
            self.assertTrue(0.02 <= spec.mass <= 0.10)
            self.assertTrue(0.5 <= spec.friction[0] <= 1.2)
            self.assertTrue(0.035 <= spec.grasp_width <= 0.060)
            if spec.prism_sides is not None:
                vertices, faces = spec.prism_mesh()
                np.testing.assert_allclose(np.ptp(vertices, axis=0), spec.size)
                self.assertEqual(len(vertices), 2 * spec.prism_sides)
                self.assertTrue(np.all(faces >= 0) and np.all(faces < len(vertices)))
        with self.assertRaises(FrozenInstanceError):
            specs[0].mass = 1.0
        for index, spec in enumerate(self.ids.object_specs):
            geom, body = self.ids.cube_geoms[index], self.ids.cube_bodies[index]
            self.assertEqual(
                self.model.geom_type[geom],
                mujoco.mjtGeom.mjGEOM_BOX
                if spec.shape == "box"
                else mujoco.mjtGeom.mjGEOM_MESH,
            )
            self.assertAlmostEqual(self.model.body_mass[body], spec.mass)
            self.assertTrue(np.all(self.model.body_inertia[body] > 0))

    def test_home_ids_and_parked_gravity_compensation(self):
        np.testing.assert_allclose(
            self.data.site_xpos[self.ids.ee_sites], self.ids.home_ee, atol=1e-6
        )
        self.assertEqual(self.ids.arm_actuators.shape, (2, 7))
        self.assertEqual(self.ids.cube_bodies.size, self.config.n_objects)
        self.assertEqual(self.data.ncon, 0)
        self.assertTrue(np.all(self.model.body_gravcomp[self.ids.cube_bodies] == 1))
        before = self.data.qpos[self.ids.cube_qpos].copy()
        self.run_servo(0, self.ids.home_ee[0], 0.5)
        np.testing.assert_allclose(
            self.data.qpos[self.ids.cube_qpos], before, atol=1e-12
        )
        for arm in range(2):
            for part in [
                *(f"link{k}" for k in range(8)),
                "hand",
                "left_finger",
                "right_finger",
            ]:
                body = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"r{arm}/{part}"
                )
                self.assertGreaterEqual(body, 0)
                self.assertTrue(
                    np.any(
                        (self.model.geom_bodyid == body)
                        & (self.model.geom_contype != 0)
                    )
                )

    def test_full_workspace_ik(self):
        for arm in range(2):
            for x in (-0.65, -0.3, 0.25) if arm == 0 else (-0.25, 0.3, 0.65):
                for y in (-0.06, 0.06):
                    position = np.array([x, y, 0.225])
                    q = self.ik.solve_pose(self.data, arm, position)
                    self.assertLess(
                        self.ik.pose_error(self.data, arm, q, position)[0], 1e-3
                    )
            for center in self.ids.bin_centers:
                for dx in (-0.065, 0.0, 0.065):
                    for dy in (-0.025, 0.025):
                        for z in (0.305, 0.325):
                            for yaw in (-1.55, -0.78, 0.0, 0.78, 1.55):
                                position = np.array([center[0] + dx, center[1] + dy, z])
                                quat = downward_quaternion(
                                    (np.pi / 2 if arm == 0 else -np.pi / 2) + yaw
                                )
                                q = self.ik.solve_pose(
                                    self.data, arm, position, quat, max_iterations=400
                                )
                                self.assertLess(
                                    self.ik.pose_error(
                                        self.data, arm, q, position, quat
                                    )[0],
                                    0.0015,
                                )

    def test_shared_hoppers_and_calibrated_camera(self):
        self.assertEqual(self.ids.bin_centers.shape, (2, 3))
        self.assertEqual(self.ids.bin_geoms.shape, (2,))
        np.testing.assert_allclose(self.ids.bin_half_sizes, [0.15, 0.10])
        self.assertAlmostEqual(self.ids.bin_rim_z - self.ids.bin_floor_z, 0.12)
        camera = self.ids.overhead_camera
        self.assertEqual(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera),
            "overhead",
        )
        np.testing.assert_allclose(self.model.cam_pos[camera], [0, 0, 1.8])
        np.testing.assert_allclose(self.model.cam_quat[camera], [1, 0, 0, 0])
        self.assertEqual(self.model.cam_fovy[camera], 60.0)
        self.assertEqual(self.ids.finger_touch_adr.shape, (2, 2))

    def test_physical_gates_drain_into_receivers_without_state_edits(self):
        indices = [
            next(i for i, s in enumerate(self.ids.object_specs) if s.shape == shape)
            for shape in ("box", "pentagonal_prism", "hexagonal_prism")
        ]
        # One object of each shape in each gate, sequentially; no recycling is
        # performed by this test. Gravity and gate contact produce the descent.
        for color in range(2):
            if color:
                mujoco.mj_resetData(self.model, self.data)
                reset_robot_pose(self.model, self.data, self.ids)
                self.ik.reset(self.data)
            for k, i in enumerate(indices):
                spec = self.ids.object_specs[i]
                self.activate(
                    i,
                    [
                        (-0.075, 0.0, 0.075)[k],
                        self.ids.bin_centers[color, 1],
                        self.ids.bin_floor_z + spec.half_height + 0.003,
                    ],
                )
            self.run_servo(0, self.ids.home_ee[0], 1.0)
            self.assertTrue(
                np.all(
                    self.data.xpos[self.ids.cube_bodies[indices], 2]
                    > self.ids.bin_floor_z
                )
            )
            self.data.ctrl[self.ids.bin_gate_actuators[color]] = (
                self.ids.bin_gate_open_position
            )
            self.run_servo(0, self.ids.home_ee[0], 2.0)
            self.assertAlmostEqual(
                self.data.qpos[self.ids.bin_gate_qpos[color]], 0.23, delta=0.003
            )
            positions = self.data.xpos[self.ids.cube_bodies[indices]]
            self.assertTrue(np.all(positions[:, 2] < self.ids.bin_outlet_z))
            self.assertTrue(
                np.all(
                    np.abs(positions[:, :2] - self.ids.bin_centers[color, :2])
                    < self.ids.bin_half_sizes
                )
            )

    def test_polygon_prism_physically_settles_after_hopper_release(self):
        # A tilted polygonal prism must physically settle before it is counted;
        # the evaluator must not mistake a moving in-bin object for success.
        index = next(
            i
            for i, s in enumerate(self.ids.object_specs)
            if s.shape == "pentagonal_prism"
        )
        tilt = 0.15
        for color in range(2):
            mujoco.mj_resetData(self.model, self.data)
            reset_robot_pose(self.model, self.data, self.ids)
            self.ik.reset(self.data)
            _, body = self.activate(
                index, [-0.055, self.ids.bin_centers[color, 1], 0.318]
            )
            q = self.ids.cube_qpos[index]
            self.data.qpos[q + 3 : q + 7] = np.array(
                [np.cos(tilt / 2), np.sin(tilt / 2), np.sin(tilt / 2), np.cos(tilt / 2)]
            ) / np.sqrt(2)
            mujoco.mj_forward(self.model, self.data)
            settled = 0.0
            for _ in range(round(1.5 / self.model.opt.timestep)):
                for arm in range(2):
                    self.ik.command(self.data, arm, self.ids.home_ee[arm])
                mujoco.mj_step(self.model, self.data)
                position = self.data.xpos[body]
                dof = self.ids.cube_dofs[index]
                speed = np.linalg.norm(self.data.qvel[dof : dof + 3])
                inside = (
                    np.all(
                        np.abs(position[:2] - self.ids.bin_centers[color, :2])
                        < self.ids.bin_half_sizes
                    )
                    and self.ids.bin_floor_z - 0.01
                    < position[2]
                    < self.ids.bin_rim_z + 0.04
                    and speed < 0.15
                )
                settled = settled + self.model.opt.timestep if inside else 0.0
            self.assertGreaterEqual(settled, self.config.settle_time)
            self.assertLess(speed, 0.01)
            self.assertFalse(self.unexpected_contacts())

    def test_empty_gripper_closes_without_finger_self_contact(self):
        # A sudden full-close command used to overshoot the compliant joint
        # limits and drive opposing finger collision meshes through each other.
        for _ in range(round(2.0 / self.model.opt.timestep)):
            for arm in range(2):
                self.ik.command(self.data, arm, self.ids.home_ee[arm], gripper=0.0)
            mujoco.mj_step(self.model, self.data)
            self.assertFalse(self.unexpected_contacts())
            for contact in self.data.contact:
                bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
                self.assertFalse(all(body in self.ids.finger_bodies for body in bodies))
        self.assertTrue(np.all(self.data.qpos[self.ids.finger_qpos] > 0.001))
        self.assertTrue(np.all(self.data.qpos[self.ids.finger_qpos] < 0.003))
        np.testing.assert_allclose(
            self.data.sensordata[self.ids.finger_touch_adr], 0, atol=1e-8
        )

    def test_every_shape_lifts_with_contact_only(self):
        for shape in ("box", "pentagonal_prism", "hexagonal_prism"):
            index = next(
                i for i, s in enumerate(self.ids.object_specs) if s.shape == shape
            )
            mujoco.mj_resetData(self.model, self.data)
            self.model.geom_contype[self.ids.cube_geoms] = self.model.geom_conaffinity[
                self.ids.cube_geoms
            ] = 0
            reset_robot_pose(self.model, self.data, self.ids)
            self.ik.reset(self.data)
            spec = self.ids.object_specs[index]
            _, body = self.activate(
                index, [-0.3, 0, self.config.belt_top + spec.half_height + 0.001]
            )
            grasp_z = (
                self.config.belt_top
                + spec.half_height
                + max(-0.006, spec.half_height - 0.030)
            )
            self.run_servo(0, [-0.3, 0, 0.4], 1.8)
            self.run_servo(0, [-0.3, 0, grasp_z], 1.2)
            self.run_servo(0, [-0.3, 0, grasp_z], 0.5, gripper=0)
            self.run_servo(0, [-0.3, 0, 0.43], 1.5, gripper=0)
            self.assertGreater(self.data.xpos[body, 2], 0.39, shape)
            self.assertTrue(
                np.all(self.data.sensordata[self.ids.finger_touch_adr[0]] > 0.3), shape
            )
            self.assertLess(
                np.linalg.norm(
                    self.data.xpos[body] - self.data.site_xpos[self.ids.ee_sites[0]]
                ),
                0.03,
            )
        self.assertTrue(np.all(self.model.eq_type == mujoco.mjtEq.mjEQ_JOINT))

    def test_heavy_hexagonal_prism_releases_from_each_finger_orientation(self):
        # Verify actual gravity-driven release of a heavy polygonal prism
        # from both arms and multiple hand orientations.
        self.config = Config(seed=1)
        self.model, self.ids = build_scene(self.config)
        self.data = mujoco.MjData(self.model)
        self.model.geom_surfacevel[self.ids.belt_geom] = 0
        index = next(
            i
            for i, spec in enumerate(self.ids.object_specs)
            if spec.shape == "hexagonal_prism"
            and spec.mass > 0.09
            and spec.grasp_width < 0.048
        )
        spec = self.ids.object_specs[index]
        for arm in range(2):
            for yaw in (0.0, 0.78):
                mujoco.mj_resetData(self.model, self.data)
                self.model.geom_contype[self.ids.cube_geoms] = 0
                self.model.geom_conaffinity[self.ids.cube_geoms] = 0
                self.model.body_gravcomp[self.ids.cube_bodies] = 1
                reset_robot_pose(self.model, self.data, self.ids)
                self.ik = PandaIK(self.model, self.ids)
                x = (-0.3, 0.3)[arm]
                _, body = self.activate(
                    index, [x, 0, self.config.belt_top + spec.half_height + 0.001]
                )
                grasp_z = (
                    self.config.belt_top
                    + spec.half_height
                    + max(-0.009, spec.half_height - 0.030)
                )
                contacts = []
                for target, duration, grip in (
                    ([x, 0, 0.4], 1.8, 0.04),
                    ([x, 0, grasp_z], 1.2, 0.04),
                    ([x, 0, grasp_z], 0.4, 0.0),
                    ([x, 0, 0.4], 1.5, 0.0),
                    ([x, 0, 0.4], 1.0, 0.0),
                ):
                    contacts.extend(
                        self.run_servo(arm, target, duration, grip, yaw=yaw)
                    )
                self.assertGreater(self.data.xpos[body, 2], 0.38)
                self.assertTrue(
                    np.all(self.data.sensordata[self.ids.finger_touch_adr[arm]] > 0.1)
                )
                contacts.extend(self.run_servo(arm, [x, 0, 0.4], 1.5, 0.04, yaw=yaw))
                self.assertFalse(contacts, contacts[:3])
                self.assertGreater(
                    np.sum(self.data.qpos[self.ids.finger_qpos[arm]]), 0.075
                )
                self.assertTrue(
                    np.all(self.data.sensordata[self.ids.finger_touch_adr[arm]] < 0.1)
                )
                self.assertLess(self.data.xpos[body, 2], 0.30)

    def test_dynamic_soft_pentagonal_prism_grasp_holds_for_twenty_seconds(self):
        # Exercise a heavy compliant-contact pentagonal prism in a moving grasp
        # and twenty-second hold on both arms, including finger/belt and
        # self-contact checks. No attachment constraint is used.
        self.config = Config(seed=1)
        self.model, self.ids = build_scene(self.config)
        self.data = mujoco.MjData(self.model)
        index = next(
            i
            for i, spec in enumerate(self.ids.object_specs)
            if spec.shape == "pentagonal_prism"
            and spec.mass > 0.085
            and spec.solref[0] == 0.025
        )
        spec = self.ids.object_specs[index]
        for arm in range(2):
            mujoco.mj_resetData(self.model, self.data)
            self.model.geom_contype[self.ids.cube_geoms] = 0
            self.model.geom_conaffinity[self.ids.cube_geoms] = 0
            self.model.body_gravcomp[self.ids.cube_bodies] = 1
            reset_robot_pose(self.model, self.data, self.ids)
            self.ik = PandaIK(self.model, self.ids)
            x0 = BASE_POSITIONS[arm, 0] - 0.25
            y = -0.065 if arm == 0 else 0.065
            _, body = self.activate(
                index, [x0, y, self.config.belt_top + spec.half_height + 0.001]
            )
            self.data.qvel[self.ids.cube_dofs[index]] = 0.1
            grasp_z = (
                self.config.belt_top
                + spec.half_height
                + max(-0.009, spec.half_height - 0.030)
            )
            contacts = []
            initial_hold_offset = None
            for phase, duration, z, grip in (
                ("approach", 1.5, 0.4, 0.04),
                ("descend", 1.1, grasp_z, 0.04),
                ("close", 0.4, grasp_z, 0),
                ("lift", 1.2, 0.38, 0),
                ("hold", 20.0, 0.38, 0),
            ):
                phase_start = self.data.time
                if phase == "lift":
                    lift_x = x0 + 0.1 * self.data.time
                if phase == "hold":
                    initial_hold_offset = (
                        self.data.xpos[body, 2]
                        - self.data.site_xpos[self.ids.ee_sites[arm], 2]
                    )
                for _ in range(round(duration / self.model.opt.timestep)):
                    if phase in ("approach", "descend", "close"):
                        x, feedforward = x0 + 0.1 * self.data.time, [0.1, 0, 0]
                    elif phase == "lift":
                        x, feedforward = (
                            lift_x + 0.1 * min(self.data.time - phase_start, 0.2),
                            None,
                        )
                    else:
                        x, feedforward = lift_x + 0.02, None
                    for a in range(2):
                        self.ik.command(
                            self.data,
                            a,
                            [x, y, z] if a == arm else self.ids.home_ee[a],
                            target_quat=downward_quaternion(np.pi if arm == 1 else 0.0)
                            if a == arm
                            else None,
                            feedforward=feedforward if a == arm else None,
                            gripper=grip if a == arm else 0.04,
                        )
                    mujoco.mj_step(self.model, self.data)
                    contacts.extend(self.unexpected_contacts())
            self.assertFalse(contacts, contacts[:3])
            self.assertTrue(
                np.all(self.data.sensordata[self.ids.finger_touch_adr[arm]] > 0.3)
            )
            self.assertLess(
                np.linalg.norm(
                    self.data.xpos[body] - self.data.site_xpos[self.ids.ee_sites[arm]]
                ),
                0.02,
            )
            final_offset = (
                self.data.xpos[body, 2] - self.data.site_xpos[self.ids.ee_sites[arm], 2]
            )
            self.assertLess(abs(final_offset - initial_hold_offset), 0.003)

    def test_both_arms_physically_place_into_both_hoppers(self):
        index = next(i for i, s in enumerate(self.ids.object_specs) if s.shape == "box")
        spec = self.ids.object_specs[index]
        for arm in range(2):
            for color in range(2):
                mujoco.mj_resetData(self.model, self.data)
                reset_robot_pose(self.model, self.data, self.ids)
                x, side = BASE_POSITIONS[arm, 0], (-1 if arm == 0 else 1)
                yaw = (
                    0.68
                    if (arm, color) == (0, 1)
                    else (-0.65 if (arm, color) == (1, 0) else 0.0)
                )
                _, body = self.activate(
                    index, [x, 0, self.config.belt_top + spec.half_height + 0.001], yaw
                )
                self.ik.reset(self.data)
                grasp_z = (
                    self.config.belt_top
                    + spec.half_height
                    + max(-0.006, spec.half_height - 0.030)
                )
                self.run_servo(arm, [x, 0, 0.4], yaw=yaw)
                self.run_servo(arm, [x, 0, grasp_z], 1.5, yaw=yaw)
                self.run_servo(arm, [x, 0, grasp_z], 0.5, gripper=0, yaw=yaw)
                center = self.ids.bin_centers[color]
                for target in (
                    [x, 0, 0.53],
                    [x, side * 0.26, 0.53],
                    [0, side * 0.18, 0.52],
                    [center[0], center[1], 0.325],
                    [center[0], center[1], 0.305],
                ):
                    contacts = self.run_servo(arm, target, 3.0, gripper=0, yaw=yaw)
                    self.assertFalse(contacts, (arm, color, target, contacts[:3]))
                    self.assertLess(
                        np.linalg.norm(
                            self.data.xpos[body]
                            - self.data.site_xpos[self.ids.ee_sites[arm]]
                        ),
                        0.03,
                    )
                self.run_servo(arm, [center[0], center[1], 0.305], 1.0, yaw=yaw)
                self.assertLess(
                    np.linalg.norm(self.data.xpos[body, :2] - center[:2]), 0.04
                )
                self.assertAlmostEqual(
                    self.data.xpos[body, 2],
                    self.ids.bin_floor_z + spec.half_height,
                    delta=0.008,
                )

    def test_exact_interarm_clearance(self):
        self.assertEqual(self.ik.interarm_clearance(self.data), 0.08)
        for arm, x in ((0, -0.12), (1, 0.12)):
            self.data.qpos[self.ids.arm_qpos[arm]] = self.ik.solve_pose(
                self.data, arm, [x, 0, 0.236]
            )
        mujoco.mj_forward(self.model, self.data)
        self.assertGreater(self.ik.interarm_clearance(self.data), 0.02)
        self.assertLess(self.ik.interarm_clearance(self.data), 0.06)

    def test_ik_does_not_teleport_live_state(self):
        positions, velocities = self.data.qpos.copy(), self.data.qvel.copy()
        self.ik.solve_pose(self.data, 0, [-0.3, 0, 0.225])
        self.ik.command(self.data, 0, [-0.3, 0, 0.225])
        np.testing.assert_array_equal(self.data.qpos, positions)
        np.testing.assert_array_equal(self.data.qvel, velocities)

    def test_cartesian_reference_and_joint_command_limits(self):
        old_velocity = self.ik.cart_velocities[0].copy()
        old_q = self.data.qpos[self.ids.arm_qpos[0]].copy()
        for step in range(800):
            target = [-0.3, 0, 0.43] if step < 400 else [0, -0.33, 0.325]
            self.ik.command(self.data, 0, target)
            self.ik.command(self.data, 1, self.ids.home_ee[1])
            dt = self.model.opt.timestep
            self.assertLessEqual(np.linalg.norm(self.ik.cart_velocities[0]), 0.450001)
            self.assertLessEqual(
                np.linalg.norm(self.ik.cart_velocities[0] - old_velocity) / dt, 2.00001
            )
            self.assertLessEqual(
                np.max(np.abs(self.ik.q_targets[0] - old_q)) / dt, 1.800001
            )
            old_velocity, old_q = (
                self.ik.cart_velocities[0].copy(),
                self.ik.q_targets[0].copy(),
            )
            mujoco.mj_step(self.model, self.data)

    def test_latched_joint_hold_resists_drift_and_keeps_gripper_active(self):
        # Actual seed-50000 pose at the first rejected approach. Repeated
        # Cartesian "hold current pose" IK used to move the elbow by 0.47 rad.
        held_q = np.array(
            [
                -0.0351379149,
                -1.7481428426,
                -0.8575738569,
                -2.6903014175,
                -0.9945039738,
                1.0920530371,
                -2.1284183007,
            ]
        )
        self.data.qpos[self.ids.arm_qpos[0]] = held_q
        self.data.qvel[self.ids.arm_dofs[0]] = [
            0.3467724,
            0.0058012,
            0.5444398,
            -0.6802064,
            0.8219329,
            0.1624403,
            -1.0515539,
        ]
        mujoco.mj_forward(self.model, self.data)
        self.ik.reset(self.data)
        hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "r0/hand")
        original_solve = self.ik._solve

        def solve_other_arm(data, arm, *args, **kwargs):
            self.assertEqual(arm, 1, "A held arm must not run IK")
            return original_solve(data, arm, *args, **kwargs)

        previous = self.ik.q_targets[0].copy()
        with patch.object(self.ik, "_solve", side_effect=solve_other_arm):
            for step in range(round(12.0 / self.model.opt.timestep)):
                elapsed = step * self.model.opt.timestep
                # A real external force tests return to the latched pose,
                # rather than repeatedly accepting the displaced encoder pose.
                self.data.xfrc_applied[hand, 0] = 8.0 if 4.0 < elapsed < 4.3 else 0.0
                self.ik.command(
                    self.data,
                    0,
                    [0.3, 0.0, 0.2],
                    target_quat=downward_quaternion(elapsed),
                    feedforward=[0.1, 0, 0],
                    gripper=0.002 if elapsed < 2.0 else 0.04,
                    hold=True,
                )
                self.assertLessEqual(
                    np.max(np.abs(self.ik.q_targets[0] - previous)),
                    self.ik.max_joint_speed * self.model.opt.timestep + 1e-12,
                )
                previous = self.ik.q_targets[0].copy()
                self.ik.command(self.data, 1, self.ids.home_ee[1])
                mujoco.mj_step(self.model, self.data)
                self.assertFalse(self.unexpected_contacts())
                if elapsed > 6.0:
                    self.assertLess(
                        np.max(np.abs(self.data.qpos[self.ids.arm_qpos[0]] - held_q)),
                        0.001,
                    )
        np.testing.assert_allclose(self.ik.q_targets[0], held_q, atol=1e-12)
        self.assertGreater(np.sum(self.data.qpos[self.ids.finger_qpos[0]]), 0.079)

    def test_hold_resume_preserves_other_arm_and_joint_slew_limit(self):
        self.ik.reset(self.data)
        # A pending actuator target can lead its encoder when a plan fails.
        self.ik.q_targets[0, 0] += 0.06
        self.data.ctrl[self.ids.arm_actuators[0]] = self.ik.q_targets[0]
        previous = self.ik.q_targets[0].copy()
        self.ik.command(self.data, 0, self.ids.home_ee[0], hold=True)
        self.assertLessEqual(
            np.max(np.abs(self.ik.q_targets[0] - previous)), 0.009 + 1e-12
        )
        self.ik.command(self.data, 1, self.ids.home_ee[1] + [0.02, 0, -0.02])
        other = [
            value[1].copy()
            for value in (
                self.ik.q_targets,
                self.ik.cart_targets,
                self.ik.cart_velocities,
                self.ik.gripper_targets,
            )
        ]
        other_ctrl = self.data.ctrl[self.ids.arm_actuators[1]].copy()
        previous = self.ik.q_targets[0].copy()
        target = self.ids.home_ee[0] + [0.03, 0.0, -0.02]
        self.ik.command(self.data, 0, target, hold=False)
        self.assertLessEqual(
            np.max(np.abs(self.ik.q_targets[0] - previous)), 0.009 + 1e-12
        )
        for expected, value in zip(
            other,
            (
                self.ik.q_targets,
                self.ik.cart_targets,
                self.ik.cart_velocities,
                self.ik.gripper_targets,
            ),
        ):
            np.testing.assert_array_equal(value[1], expected)
        np.testing.assert_array_equal(
            self.data.ctrl[self.ids.arm_actuators[1]], other_ctrl
        )
        for _ in range(round(2.0 / self.model.opt.timestep)):
            previous = self.ik.q_targets[0].copy()
            self.ik.command(self.data, 0, target)
            self.ik.command(self.data, 1, self.ids.home_ee[1])
            self.assertLessEqual(
                np.max(np.abs(self.ik.q_targets[0] - previous)), 0.009 + 1e-12
            )
            mujoco.mj_step(self.model, self.data)
        self.assertLess(
            np.linalg.norm(self.data.site_xpos[self.ids.ee_sites[0]] - target), 0.005
        )

    def test_native_belt_following_contact_grasp(self):
        index = next(i for i, s in enumerate(self.ids.object_specs) if s.shape == "box")
        spec = self.ids.object_specs[index]
        self.model.geom_surfacevel[self.ids.belt_geom, 0] = 0.1
        _, body = self.activate(
            index, [-0.55, 0, self.config.belt_top + spec.half_height]
        )
        self.data.qvel[self.ids.cube_dofs[index]] = 0.1
        grasp_z = (
            self.config.belt_top
            + spec.half_height
            + max(-0.006, spec.half_height - 0.030)
        )
        for duration, z, grip in (
            (1.5, 0.4, 0.04),
            (1.3, grasp_z, 0.04),
            (0.45, grasp_z, 0),
            (0.9, 0.43, 0),
        ):
            for _ in range(round(duration / self.model.opt.timestep)):
                goal = [-0.55 + 0.1 * self.data.time, 0, z]
                self.ik.command(
                    self.data,
                    0,
                    goal,
                    target_quat=downward_quaternion(0),
                    feedforward=[0.1, 0, 0],
                    gripper=grip,
                )
                self.ik.command(self.data, 1, self.ids.home_ee[1])
                mujoco.mj_step(self.model, self.data)
        self.assertGreater(self.data.xpos[body, 2], 0.39)
        self.assertLess(abs(self.data.xpos[body, 1]), 0.01)
        self.assertLess(
            np.linalg.norm(
                self.data.xpos[body] - self.data.site_xpos[self.ids.ee_sites[0]]
            ),
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
