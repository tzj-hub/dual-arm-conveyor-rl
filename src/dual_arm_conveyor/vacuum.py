"""Compliant, force-limited vacuum cups driven by physical contact sensors.

This physics-device model may inspect contact geometry to apply equal and
opposite wrenches. Its public sensor exposes only seal/pressure state, never an
object identity or pose. It adds no equality constraint and never edits qpos,
qvel or actuator controls. ``step`` returns a fresh generalized-force vector;
the simulation adds it after its ordinary gravity/actuator control each tick.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import mujoco
import numpy as np


@dataclass(frozen=True)
class VacuumParameters:
    max_pressure_pa: float = 15000.0
    seal_pressure_pa: float = 5000.0
    max_normal_force: float = 5.0
    friction: float = 0.7
    normal_stiffness: float = 600.0
    normal_damping: float = 3.0
    tangent_stiffness: float = 300.0
    tangent_damping: float = 2.5
    angular_stiffness: float = 0.10
    angular_damping: float = 0.001
    max_torque: float = 0.02
    seal_gap: float = 0.001
    break_gap: float = 0.006
    break_compression: float = 0.004
    break_shear: float = 0.006
    seal_tilt: float = math.radians(15.0)
    break_tilt: float = math.radians(20.0)
    pressure_rise_time: float = 0.025
    pressure_fall_time: float = 0.01
    pressure_grace: float = 0.12
    rearm_delay: float = 0.15

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.seal_gap >= self.break_gap or self.seal_tilt >= self.break_tilt:
            raise ValueError("Seal tolerances must be tighter than break tolerances")


@dataclass
class _Seal:
    body: int
    geom: int
    local_anchor: np.ndarray
    local_normal: np.ndarray
    relative_rotation: np.ndarray
    created: float


def _rotation_error(desired, actual):
    first, second = np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(first, desired.reshape(9))
    mujoco.mju_mat2Quat(second, actual.reshape(9))
    second[1:] *= -1
    delta = np.empty(4)
    mujoco.mju_mulQuat(delta, first, second)
    if delta[0] < 0:
        delta *= -1
    sine = np.linalg.norm(delta[1:])
    angle = 2 * np.arctan2(sine, max(delta[0], 1e-12))
    return delta[1:] * angle / max(sine, 1e-12)


class VacuumGripper:
    """One contact-established, pressure-limited adhesive device per robot.

    Eight rim rays plus a centre ray must all see the contacted workpiece
    within 1 mm. This prevents an overhanging cup or side contact from creating
    a seal. After sealing, an elastic attachment transmits bounded forces and
    torques; excessive separation, shear, tilt or insufficient pressure breaks
    it. A suspended workpiece remains a fully dynamic free body.
    """

    def __init__(self, model, ids, parameters=None):
        self.model = model
        self.parameters = parameters or VacuumParameters()
        if any(
            getattr(ids, name, None) is None
            for name in ("vacuum_sites", "vacuum_geoms", "vacuum_bodies")
        ):
            raise ValueError("VacuumGripper requires a scene with physical vacuum cups")
        self.sites = np.asarray(ids.vacuum_sites, dtype=int).copy()
        self.geoms = np.asarray(ids.vacuum_geoms, dtype=int).copy()
        self.bodies = np.asarray(ids.vacuum_bodies, dtype=int).copy()
        self.num_arms = len(self.sites)
        if (
            not self.num_arms
            or len(self.geoms) != self.num_arms
            or len(self.bodies) != self.num_arms
        ):
            raise ValueError("Vacuum site, geometry and body arrays must match")
        if np.any(model.geom_type[self.geoms] != int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            raise ValueError("Vacuum cup geometry must be a physical cylinder")
        self.radii = model.geom_size[self.geoms, 0].copy()
        self.areas = math.pi * self.radii**2
        self._objects = set(map(int, ids.cube_geoms))
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._ray_geom = np.full(1, -1, dtype=np.int32)
        self.reset()

    def reset(self):
        self._enabled = np.zeros(self.num_arms, dtype=bool)
        self._pressure = np.zeros(self.num_arms)
        self._seals = [None] * self.num_arms
        self._cooldown = np.zeros(self.num_arms)
        self._last_time = None
        self._last_wrenches = np.zeros((self.num_arms, 6))
        self._seal_count = np.zeros(self.num_arms, dtype=int)
        self._break_count = np.zeros(self.num_arms, dtype=int)
        self._last_break = [None] * self.num_arms
        self._max_force = np.zeros(self.num_arms)
        self._max_torque = np.zeros(self.num_arms)

    def set_enabled(self, arm, enabled):
        if not isinstance(arm, (int, np.integer)) or not 0 <= arm < self.num_arms:
            raise ValueError("Invalid vacuum arm index")
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("Vacuum command must be boolean")
        self._enabled[arm] = bool(enabled)
        if not enabled and self._seals[arm] is not None:
            self._seals[arm] = None
            self._last_break[arm] = "release"

    def sensor(self, arm):
        return {
            "sealed": bool(
                self._seals[arm] is not None
                and self._pressure[arm] >= self.parameters.seal_pressure_pa
            ),
            "pressure_pa": float(self._pressure[arm]),
            "enabled": bool(self._enabled[arm]),
        }

    def _break(self, arm, now, reason):
        self._seals[arm] = None
        self._pressure[arm] = 0.0
        self._cooldown[arm] = now + self.parameters.rearm_delay
        self._break_count[arm] += 1
        self._last_break[arm] = reason

    def _contact_candidate(self, data, arm, normal):
        cup = int(self.geoms[arm])
        candidates = []
        for contact in data.contact:
            first, second = int(contact.geom1), int(contact.geom2)
            if first == cup and second in self._objects:
                other, direction = second, np.asarray(contact.frame[:3])
            elif second == cup and first in self._objects:
                other, direction = first, -np.asarray(contact.frame[:3])
            else:
                continue
            alignment = float(direction @ normal)
            if contact.dist <= self.parameters.seal_gap and alignment >= math.cos(
                self.parameters.seal_tilt
            ):
                candidates.append((float(contact.dist), other))
        return None if not candidates else min(candidates)[1]

    def _supported_surface(self, data, arm, geom, centre, rotation):
        normal = rotation[:, 2]
        # Begin just above the mouth, including a 1 mm physical compression.
        offset = 1.2 * self.parameters.seal_gap
        angles = np.arange(8) * math.pi / 4
        local = np.vstack(
            (
                np.zeros((1, 2)),
                np.column_stack((np.cos(angles), np.sin(angles)))
                * self.radii[arm]
                * 0.85,
            )
        )
        centre_surface = None
        for index, xy in enumerate(local):
            ray_start = centre + rotation[:, :2] @ xy - normal * offset
            self._ray_geom[0] = -1
            distance = mujoco.mj_ray(
                self.model,
                data,
                ray_start,
                normal,
                None,
                1,
                int(self.bodies[arm]),
                self._ray_geom,
            )
            if (
                int(self._ray_geom[0]) != geom
                or not 0 <= distance <= offset + self.parameters.seal_gap
            ):
                return None
            if index == 0:
                centre_surface = ray_start + normal * distance
        return centre_surface

    def _try_seal(self, data, arm, now, centre, rotation):
        normal = rotation[:, 2]
        geom = self._contact_candidate(data, arm, normal)
        if geom is None:
            return
        point = self._supported_surface(data, arm, geom, centre, rotation)
        if point is None:
            return
        body = int(self.model.geom_bodyid[geom])
        object_rotation = data.xmat[body].reshape(3, 3)
        self._seals[arm] = _Seal(
            body,
            geom,
            object_rotation.T @ (point - data.xpos[body]),
            object_rotation.T @ (-normal),
            rotation.T @ object_rotation,
            now,
        )
        self._seal_count[arm] += 1
        self._last_break[arm] = None

    def _implicit_wrench(self, data, arm, seal, anchor, rotation, delta, dt):
        """Backward-Euler compliant wrench in the cup's co-moving frame.

        The centre-minus-anchor derivative in this frame is exactly the
        relative point velocity, including the rotating-frame term. Both
        Jacobians therefore use the same force point. Six solves with MuJoCo's
        existing sparse mass factorization include payload off-centre inertia
        and articulated robot reaction without forming a dense mass inverse.

        Angular log-error velocity uses a small-angle approximation inside the
        unchanged 20-degree break limit. External forces/contacts remain in the
        physics engine: this regularized discrete compliance is slightly softer
        than the nominal continuous spring under steady load.
        """
        mujoco.mj_jac(self.model, data, self._jacp, self._jacr, anchor, seal.body)
        relative = np.vstack((self._jacp.copy(), self._jacr.copy()))
        mujoco.mj_jac(
            self.model, data, self._jacp, self._jacr, anchor, int(self.bodies[arm])
        )
        relative -= np.vstack((self._jacp, self._jacr))
        relative[:3] = rotation.T @ relative[:3]
        relative[3:] = rotation.T @ relative[3:]
        inverse_product = np.empty_like(relative)
        mujoco.mj_solveM(self.model, data, inverse_product, relative)
        mobility = relative @ inverse_product.T
        desired_rotation = rotation @ seal.relative_rotation
        object_rotation = data.xmat[seal.body].reshape(3, 3)
        error = np.r_[
            rotation.T @ (-delta),
            rotation.T @ _rotation_error(desired_rotation, object_rotation),
        ]
        velocity = -(relative @ data.qvel)
        p = self.parameters
        stiffness = np.array(
            [p.tangent_stiffness] * 2 + [p.normal_stiffness] + [p.angular_stiffness] * 3
        )
        damping = np.array(
            [p.tangent_damping] * 2 + [p.normal_damping] + [p.angular_damping] * 3
        )
        implicit_scale = dt * damping + dt**2 * stiffness
        return np.linalg.solve(
            np.eye(6) + implicit_scale[:, None] * mobility,
            stiffness * error + (damping + dt * stiffness) * velocity,
        )

    def step(self, data):
        """Return this tick's force contribution, without changing live state.

        Call once per physics tick after forward/contact computation. The
        caller clears/composes qfrc_applied each tick, then adds this returned
        vector. Repeated calls at one time do not advance pressure or forces.
        """
        now = float(data.time)
        if self._last_time is not None and now < self._last_time - 1e-9:
            raise ValueError("Vacuum time went backwards; call reset")
        dt = (
            float(self.model.opt.timestep)
            if self._last_time is None
            else now - self._last_time
        )
        if dt <= 1e-12:
            raise ValueError("Call vacuum.step exactly once per physics tick")
        self._last_time = now
        forces = np.zeros(self.model.nv)
        self._last_wrenches.fill(0)
        p = self.parameters
        for arm, site in enumerate(self.sites):
            centre = data.site_xpos[site].copy()
            rotation = data.site_xmat[site].reshape(3, 3)
            normal = rotation[:, 2]
            if (
                self._enabled[arm]
                and self._seals[arm] is None
                and now >= self._cooldown[arm]
            ):
                self._try_seal(data, arm, now, centre, rotation)
            seal = self._seals[arm]
            pressure_target = (
                p.max_pressure_pa if self._enabled[arm] and seal is not None else 0.0
            )
            time_constant = (
                p.pressure_rise_time if pressure_target else p.pressure_fall_time
            )
            self._pressure[arm] += (pressure_target - self._pressure[arm]) * (
                1 - math.exp(-dt / time_constant)
            )
            if seal is None:
                continue
            object_rotation = data.xmat[seal.body].reshape(3, 3)
            anchor = data.xpos[seal.body] + object_rotation @ seal.local_anchor
            delta = anchor - centre
            gap = float(delta @ normal)
            shear = delta - gap * normal
            alignment = float(
                np.clip((object_rotation @ seal.local_normal) @ (-normal), -1.0, 1.0)
            )
            reason = (
                "gap"
                if gap > p.break_gap or gap < -p.break_compression
                else "shear"
                if np.linalg.norm(shear) > p.break_shear
                else "tilt"
                if alignment < math.cos(p.break_tilt)
                else "pressure"
                if now - seal.created > p.pressure_grace
                and self._pressure[arm] < p.seal_pressure_pa
                else None
            )
            if reason is not None:
                self._break(arm, now, reason)
                continue
            normal_capacity = min(
                p.max_normal_force, self._pressure[arm] * self.areas[arm]
            )
            wrench = self._implicit_wrench(data, arm, seal, anchor, rotation, delta, dt)
            # Adhesion only pulls normally; physical collision handles compression.
            wrench[2] = np.clip(wrench[2], -normal_capacity, 0.0)
            tangent_capacity = p.friction * normal_capacity
            wrench[:2] *= min(
                1.0, tangent_capacity / max(np.linalg.norm(wrench[:2]), 1e-12)
            )
            torque_capacity = min(
                p.max_torque, 0.5 * self.radii[arm] * tangent_capacity
            )
            wrench[3:] *= min(
                1.0, torque_capacity / max(np.linalg.norm(wrench[3:]), 1e-12)
            )
            force = rotation @ wrench[:3]
            torque = rotation @ wrench[3:]
            # Both wrenches act at the same world point, so their total force
            # and moment cancel exactly, even with compliant anchor separation.
            mujoco.mj_applyFT(
                self.model, data, force, torque, anchor, seal.body, forces
            )
            mujoco.mj_applyFT(
                self.model, data, -force, -torque, anchor, int(self.bodies[arm]), forces
            )
            self._last_wrenches[arm, :3] = force
            self._last_wrenches[arm, 3:] = torque
            self._max_force[arm] = max(
                self._max_force[arm], float(np.linalg.norm(force))
            )
            self._max_torque[arm] = max(
                self._max_torque[arm], float(np.linalg.norm(torque))
            )
        return forces

    def diagnostics(self):
        return {
            "parameters": asdict(self.parameters),
            "model": "contact_established_bounded_compliant_vacuum",
            "integration": "effective_mass_backward_euler_6d",
            "sensors": [self.sensor(arm) for arm in range(self.num_arms)],
            "seal_count": self._seal_count.tolist(),
            "break_count": self._break_count.tolist(),
            "last_break": self._last_break.copy(),
            "max_applied_force_n": self._max_force.tolist(),
            "max_applied_torque_nm": self._max_torque.tolist(),
            "max_normal_force_n": self.parameters.max_normal_force,
            "no_weld_or_pose_edit": True,
        }
