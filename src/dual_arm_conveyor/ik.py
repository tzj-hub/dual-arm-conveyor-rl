"""Damped least-squares Cartesian servo for independent Panda arms.

IK operates on scratch MjData. Commands affect only actuators and gravity
compensation forces; live generalized positions and velocities are never edited.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .model import ModelIds


def downward_quaternion(yaw: float = 0.0) -> np.ndarray:
    """Tool +z points down, tool +x has the requested world yaw (wxyz)."""
    return np.array([0.0, np.cos(yaw / 2), np.sin(yaw / 2), 0.0])


def _orientation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    # Quaternion difference in the world frame, matching mj_jacSite rotational
    # Jacobians. Unlike the cross-product approximation, this handles 180 deg.
    desired_q = np.empty(4)
    current_q = np.empty(4)
    mujoco.mju_mat2Quat(desired_q, target.reshape(9))
    mujoco.mju_mat2Quat(current_q, current.reshape(9))
    inv = current_q.copy()
    inv[1:] *= -1
    delta = np.empty(4)
    mujoco.mju_mulQuat(delta, desired_q, inv)
    if delta[0] < 0:
        delta *= -1
    angle = 2 * np.arctan2(np.linalg.norm(delta[1:]), max(delta[0], 1e-12))
    return delta[1:] * angle / max(np.linalg.norm(delta[1:]), 1e-12)


class PandaIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        ids: ModelIds,
        max_joint_speed: float = 1.8,
        max_cartesian_speed: float = 0.45,
        max_cartesian_acceleration: float = 2.0,
        adaptive: bool = False,
    ):
        self.model, self.ids = model, ids
        self.num_arms = len(ids.ee_sites)
        self.scratch = mujoco.MjData(model)
        self.max_joint_speed = max_joint_speed
        self.max_cartesian_speed = max_cartesian_speed
        self.max_cartesian_acceleration = max_cartesian_acceleration
        self.adaptive = bool(adaptive)
        self.cart_targets = ids.home_ee.copy()
        self.cart_velocities = np.zeros((self.num_arms, 3))
        self.gripper_targets = np.full(self.num_arms, 0.04)
        self.q_targets = ids.home_qpos.copy()
        self._holding = np.zeros(self.num_arms, dtype=bool)
        self._hold_q = self.q_targets.copy()
        self._joint_plan = [None] * self.num_arms
        self._joint_elapsed = np.zeros(self.num_arms)
        self._joint_duration = np.zeros(self.num_arms)
        self.max_joint_acceleration = 3.6
        self.default_quats = tuple(
            downward_quaternion(np.pi / 2 if arm % 2 == 0 else -np.pi / 2)
            for arm in range(self.num_arms)
        )
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self.last_position_error = np.zeros(self.num_arms)
        self.last_orientation_error = np.zeros(self.num_arms)
        self.last_joint_reference = [None] * self.num_arms
        self._initialized = False
        self._collision_geoms = tuple(
            np.flatnonzero((ids.robot_geom_arm == arm) & (model.geom_contype != 0))
            for arm in range(self.num_arms)
        )
        self.last_clearance_pair = None

    def reset(self, data: mujoco.MjData):
        self.q_targets[:] = data.qpos[self.ids.arm_qpos]
        self.cart_targets[:] = data.site_xpos[self.ids.ee_sites]
        self.cart_velocities[:] = 0
        self.gripper_targets[:] = np.clip(
            data.qpos[self.ids.finger_qpos].mean(axis=1), 0.002, 0.04
        )
        self.last_joint_reference = [None] * self.num_arms
        self._holding[:] = False
        self._hold_q[:] = self.q_targets
        self._joint_plan = [None] * self.num_arms
        self._joint_elapsed[:] = 0
        self._initialized = True

    def _solve(self, data, arm, position, quaternion, iterations, initial=None):
        scratch = self.scratch
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0
        qadr = self.ids.arm_qpos[arm]
        dofs = self.ids.arm_dofs[arm]
        q = data.qpos[qadr].copy() if initial is None else np.asarray(initial).copy()
        limits = self.model.jnt_range[self.ids.arm_joints[arm]]
        target_rot = np.empty(9)
        mujoco.mju_quat2Mat(target_rot, quaternion)
        target_rot = target_rot.reshape(3, 3)
        site = self.ids.ee_sites[arm]
        for _ in range(iterations):
            scratch.qpos[qadr] = q
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)
            ep = np.asarray(position) - scratch.site_xpos[site]
            er = _orientation_error(target_rot, scratch.site_xmat[site].reshape(3, 3))
            if np.linalg.norm(ep) < 2e-5 and np.linalg.norm(er) < 2e-4:
                break
            mujoco.mj_jacSite(self.model, scratch, self._jacp, self._jacr, site)
            # Metres and radians weighted to favour accurate finger positioning.
            jac = np.vstack((self._jacp[:, dofs], 0.35 * self._jacr[:, dofs]))
            error = np.concatenate((ep, 0.35 * er))
            damping = 0.006
            inverse = jac.T @ np.linalg.solve(
                jac @ jac.T + damping**2 * np.eye(6), np.eye(6)
            )
            step = inverse @ error
            # Small null-space home bias keeps the redundant elbow away from
            # joint limits while allowing accurate Cartesian tracking.
            step += (np.eye(7) - inverse @ jac) @ (
                0.006 * (self.ids.home_qpos[arm] - q)
            )
            scale = min(1.0, 0.18 / max(np.max(np.abs(step)), 1e-9))
            q = np.clip(q + scale * step, limits[:, 0] + 0.015, limits[:, 1] - 0.015)
        return q

    def solve_pose(
        self, data, arm: int, position, quaternion=None, *, max_iterations: int = 120
    ) -> np.ndarray:
        """Return a kinematic solution without changing simulation state."""
        quat = (
            self.default_quats[arm]
            if quaternion is None
            else np.asarray(quaternion, dtype=float)
        )
        return self._solve(data, arm, np.asarray(position), quat, max_iterations)

    def pose_error(
        self, data, arm: int, q, position, quaternion=None
    ) -> tuple[float, float]:
        """Measure an offline solution's position and orientation error."""
        self.scratch.qpos[:] = data.qpos
        self.scratch.qpos[self.ids.arm_qpos[arm]] = q
        mujoco.mj_kinematics(self.model, self.scratch)
        quat = self.default_quats[arm] if quaternion is None else quaternion
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        site = self.ids.ee_sites[arm]
        ep = np.linalg.norm(np.asarray(position) - self.scratch.site_xpos[site])
        er = np.linalg.norm(
            _orientation_error(
                rotation.reshape(3, 3), self.scratch.site_xmat[site].reshape(3, 3)
            )
        )
        return float(ep), float(er)

    def interarm_clearance(self, data: mujoco.MjData, margin: float = 0.08) -> float:
        """Exact collision-mesh distance, capped at ``margin`` metres.

        Bounding spheres only prune pairs that provably cannot improve the
        current minimum. Near pairs use MuJoCo's mesh/convex narrow phase. This
        is a measurement, not a collision-avoidance controller. Call after
        mj_forward/mj_step so the supplied world geometry poses are current.
        """
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError("margin must be finite and positive")
        best = float(margin)
        self.last_clearance_pair = None
        # Every pair of arms matters: checking only each station's partner
        # would miss neighboring-station elbows and same-side robots.
        for first_arm in range(self.num_arms):
            first = self._collision_geoms[first_arm]
            for second_arm in range(first_arm + 1, self.num_arms):
                second = self._collision_geoms[second_arm]
                delta = data.geom_xpos[first, None, :] - data.geom_xpos[None, second, :]
                lower_bound = (
                    np.linalg.norm(delta, axis=-1)
                    - self.model.geom_rbound[first, None]
                    - self.model.geom_rbound[None, second]
                )
                pairs = np.argwhere(lower_bound < best)
                order = np.argsort(lower_bound[pairs[:, 0], pairs[:, 1]])
                for index in order:
                    i, j = pairs[index]
                    if lower_bound[i, j] >= best:
                        continue
                    geom1, geom2 = int(first[i]), int(second[j])
                    distance = mujoco.mj_geomDistance(
                        self.model, data, geom1, geom2, margin, None
                    )
                    if distance < best:
                        best = float(distance)
                        self.last_clearance_pair = (geom1, geom2)
        return best

    def command(
        self,
        data,
        arm: int,
        target_pos,
        target_quat=None,
        feedforward=None,
        gripper: float = 0.04,
        motion_plan=None,
        waypoint_index: int = 0,
        hold: bool = False,
    ):
        """Send one servo command at ``model.opt.timestep`` seconds.

        feedforward is a world-frame Cartesian velocity in m/s, used to
        compensate the finite position-servo lag while following the belt.
        Gripper targets are half-apertures in metres, 0 closed to .04 open.
        hold latches this arm's measured joint pose once, without running IK.
        Gripper actuation and gravity compensation remain active during a hold.
        """
        if not self._initialized:
            self.reset(data)
        if hold:
            self._joint_plan[arm] = None
            if not self._holding[arm]:
                self._hold_q[arm] = data.qpos[self.ids.arm_qpos[arm]]
                self.cart_targets[arm] = data.site_xpos[self.ids.ee_sites[arm]]
                self.cart_velocities[arm] = 0
                self._holding[arm] = True
            self.last_joint_reference[arm] = None
            # Re-solving a moving Cartesian "hold" target would retain the
            # null-space home bias and slowly sweep redundant links into walls.
            self._apply_actuator_targets(data, arm, self._hold_q[arm], gripper)
            self.last_position_error[arm] = np.linalg.norm(
                self.cart_targets[arm] - data.site_xpos[self.ids.ee_sites[arm]]
            )
            return
        resumed = bool(self._holding[arm])
        if resumed:
            self._holding[arm] = False
            self.cart_targets[arm] = data.site_xpos[self.ids.ee_sites[arm]]
            self.cart_velocities[arm] = 0
            self.last_joint_reference[arm] = None
            # Keep q_targets as the previous actuator command so the shared
            # joint slew limit also applies across the hold/resume boundary.
        if motion_plan is not None and motion_plan.execution_kind == "joint":
            self._command_joint_path(data, arm, motion_plan, gripper)
            return
        self._joint_plan[arm] = None
        quaternion = (
            self.default_quats[arm] if target_quat is None else np.asarray(target_quat)
        )
        position = np.asarray(target_pos, dtype=float).copy()
        if feedforward is not None:
            position += 0.045 * np.asarray(feedforward)
        dt = self.model.opt.timestep
        delta = position - self.cart_targets[arm]
        # A critically damped reference with explicit velocity feedforward.
        # This avoids snapping velocity to zero at a waypoint, which otherwise
        # defeats the acceleration bound in the last few millimetres.
        # The explicit acceleration limiter below bounds transients. Keeping
        # the position response independent avoids a slow tail at every FSM
        # waypoint without increasing velocity or acceleration limits.
        response = 10.0
        desired_velocity = response * delta
        # Brake before reaching a checked endpoint. A pure proportional speed
        # target only begins braking after its stopping distance is exhausted
        # and can overshoot a low grasp by several millimetres.
        braking_speed = 0.8 * np.sqrt(
            2 * self.max_cartesian_acceleration * np.linalg.norm(delta)
        )
        desired_velocity *= min(
            1.0, braking_speed / max(np.linalg.norm(desired_velocity), 1e-12)
        )
        if feedforward is not None:
            desired_velocity += np.asarray(feedforward)
        desired_velocity *= min(
            1.0, self.max_cartesian_speed / max(np.linalg.norm(desired_velocity), 1e-12)
        )
        velocity_change = desired_velocity - self.cart_velocities[arm]
        acceleration_step = self.max_cartesian_acceleration * dt
        velocity_change *= min(
            1.0, acceleration_step / max(np.linalg.norm(velocity_change), 1e-12)
        )
        self.cart_velocities[arm] += velocity_change
        step = self.cart_velocities[arm] * dt
        self.cart_targets[arm] += step
        # A short leash prevents integral windup against unmodelled contacts.
        actual = data.site_xpos[self.ids.ee_sites[arm]]
        lead = self.cart_targets[arm] - actual
        if np.linalg.norm(lead) > 0.06:
            self.cart_targets[arm] = actual + 0.06 * lead / np.linalg.norm(lead)
        if motion_plan is not None:
            # The same sampled joint path that passed mesh checks selects the
            # redundant elbow branch. Re-solving each control tick from the
            # previous command accumulated a different null-space home bias
            # than the coarser planner and could sweep a hand into link1.
            reference = motion_plan.joint_reference_at(
                waypoint_index, self.cart_targets[arm]
            )
            quaternion = motion_plan.orientation_at(
                waypoint_index, self.cart_targets[arm]
            )
            if self.adaptive:
                # A moving target is certified ahead of the observation by the
                # caller. Tracking can then use the checked joint branch itself
                # instead of extrapolating a bounded but unchecked IK correction
                # beyond a stale endpoint. Interpolation stays on the sampled
                # joint segments that were checked by the planner.
                solution = reference
            else:
                corrected = self._solve(
                    data, arm, self.cart_targets[arm], quaternion, 4, initial=reference
                )
                correction = corrected - reference
                correction *= min(1.0, 0.015 / max(np.max(np.abs(correction)), 1e-12))
                solution = reference + correction
            self.last_joint_reference[arm] = reference.copy()
        else:
            solution = self._solve(
                data,
                arm,
                self.cart_targets[arm],
                quaternion,
                4,
                initial=data.qpos[self.ids.arm_qpos[arm]]
                if resumed
                else self.q_targets[arm],
            )
            self.last_joint_reference[arm] = None
        self._apply_actuator_targets(data, arm, solution, gripper)
        self.last_position_error[arm] = np.linalg.norm(
            np.asarray(target_pos) - data.site_xpos[self.ids.ee_sites[arm]]
        )

    def _command_joint_path(self, data, arm, plan, gripper):
        """Execute a checked straight joint path with one smooth time law.

        The quintic scale has zero velocity and acceleration at both ends.
        Duration bounds joint speed/acceleration and the sampled FK Cartesian
        derivatives; the ordinary actuator slew limit is still applied. The
        interpolation remains on the exact checked joint line, without IK
        corrections or stopping at numerical collision-sampling knots.
        """
        dt = self.model.opt.timestep
        first, last = plan.joint_paths[0][0], plan.joint_paths[0][-1]
        delta = last - first
        if self._joint_plan[arm] is not plan:
            self._joint_plan[arm] = plan
            self._joint_elapsed[arm] = 0.0
            self.cart_targets[arm] = data.site_xpos[self.ids.ee_sites[arm]]
            self.cart_velocities[arm] = 0.0
            joint_distance = float(np.max(np.abs(delta)))
            duration = max(
                0.15,
                1.875 * joint_distance / self.max_joint_speed,
                np.sqrt(5.774 * joint_distance / self.max_joint_acceleration),
            )
            positions = plan.joint_cartesian_samples
            if positions is not None and len(positions) >= 3 and joint_distance > 1e-8:
                axis = int(np.argmax(np.abs(delta)))
                spacing = (plan.joint_paths[0][:, axis] - first[axis]) / delta[axis]
                first_derivative = np.gradient(positions, spacing, axis=0, edge_order=2)
                second_derivative = np.gradient(
                    first_derivative, spacing, axis=0, edge_order=2
                )
                # Five percent allowance for FK derivative discretization.
                speed = 1.05 * np.max(np.linalg.norm(first_derivative, axis=1))
                curvature = 1.05 * np.max(np.linalg.norm(second_derivative, axis=1))
                duration = max(
                    duration,
                    1.875 * speed / self.max_cartesian_speed,
                    np.sqrt(
                        (1.875**2 * curvature + 5.774 * speed)
                        / self.max_cartesian_acceleration
                    ),
                )
            self._joint_duration[arm] = duration
        self._joint_elapsed[arm] += dt
        tau = min(1.0, self._joint_elapsed[arm] / self._joint_duration[arm])
        progress = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
        reference = first + progress * delta
        self.last_joint_reference[arm] = reference.copy()
        self._apply_actuator_targets(data, arm, reference, gripper)
        # Keep Cartesian diagnostics and the next controller's reference in
        # sync with the FK of this same checked joint reference.
        self.scratch.qpos[self.ids.arm_qpos[arm]] = reference
        mujoco.mj_kinematics(self.model, self.scratch)
        current_target = self.scratch.site_xpos[self.ids.ee_sites[arm]].copy()
        self.cart_velocities[arm] = (current_target - self.cart_targets[arm]) / dt
        self.cart_targets[arm] = current_target
        self.last_position_error[arm] = np.linalg.norm(
            current_target - data.site_xpos[self.ids.ee_sites[arm]]
        )

    def _apply_actuator_targets(self, data, arm, solution, gripper):
        """Apply finite joint/gripper targets without editing physical state."""
        dt = self.model.opt.timestep
        joint_step = solution - self.q_targets[arm]
        max_step = self.max_joint_speed * dt
        # One scale factor preserves joint-space direction, unlike independently
        # clipping seven joint increments (which distorts Cartesian trajectories).
        joint_step *= min(1.0, max_step / max(np.max(np.abs(joint_step)), 1e-12))
        self.q_targets[arm] += joint_step
        data.ctrl[self.ids.arm_actuators[arm]] = self.q_targets[arm]
        # Rate-limit half aperture to 0.10 m/s and retain a 4 mm empty-jaw
        # gap. An instantaneous closure command can overshoot the finger
        # stops with soft contact and a 5 ms physics step.
        desired_gripper = np.clip(gripper, 0.002, 0.04)
        self.gripper_targets[arm] += np.clip(
            desired_gripper - self.gripper_targets[arm], -0.10 * dt, 0.10 * dt
        )
        data.ctrl[self.ids.gripper_actuators[arm]] = self.gripper_targets[arm]
        dofs = self.ids.arm_dofs[arm]
        data.qfrc_applied[dofs] = data.qfrc_bias[dofs]
