"""Sampled, collision-checked Cartesian shortcuts for the existing servo.

Planning reads robot/gripper/gate encoders and known scene geometry. It never
reads an object's simulator pose. A held object's conservative box lives only
in a copied model and follows the estimated grasp transform. Plans certify the
sampled, static configurations of the other arms, not their future motion: the
environment must retain corridor reservations and monitor execution.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
import math

import mujoco
import numpy as np

from .ik import PandaIK


@dataclass(frozen=True)
class MotionPlan:
    waypoints: tuple[np.ndarray, ...]
    verified: bool
    reason: str
    length: float
    direct_length: float
    checked_samples: int
    minimum_clearance: float
    joint_travel: float
    start_position: np.ndarray | None = None
    start_quaternion: np.ndarray | None = None
    waypoint_quaternions: tuple[np.ndarray, ...] = ()
    clearance_margin: float = 0.008
    joint_paths: tuple[np.ndarray, ...] = ()
    start_joint_positions: np.ndarray | None = None
    escape_initial_clearances: tuple[tuple[int, int, float, float], ...] = ()
    execution_kind: str = "cartesian"
    joint_cartesian_samples: np.ndarray | None = None

    def _progress(self, index, position):
        start = self.start_position if index == 0 else self.waypoints[index - 1]
        delta = self.waypoints[index] - start
        denominator = float(np.dot(delta, delta))
        return (
            1.0
            if denominator < 1e-12
            else float(
                np.clip(
                    np.dot(np.asarray(position) - start, delta) / denominator, 0.0, 1.0
                )
            )
        )

    def joint_reference_at(self, index, position):
        """Interpolate the exact redundancy branch checked during planning."""
        if self.execution_kind == "preview":
            raise ValueError("A grasp exit preview is not an executable motion plan")
        if not self.verified or not 0 <= index < len(self.joint_paths):
            raise ValueError("A verified joint path and valid segment are required")
        samples = self.joint_paths[index]
        progress = self._progress(index, position) * (len(samples) - 1)
        if progress <= 0:
            return samples[0].copy()
        if progress >= len(samples) - 1:
            return samples[-1].copy()
        lower = min(int(progress), len(samples) - 2)
        fraction = progress - lower
        return samples[lower] + fraction * (samples[lower + 1] - samples[lower])

    def orientation_at(self, index, position):
        """Orientation on the checked segment at measured Cartesian progress.

        Sending the final orientation immediately can sweep the hand through a
        shoulder before translation clears it. Failed plans must instead hold
        the measured orientation and must never call this method.
        """
        if self.execution_kind == "preview":
            raise ValueError("A grasp exit preview is not an executable motion plan")
        if not self.verified or not 0 <= index < len(self.waypoints):
            raise ValueError("A verified plan and valid waypoint index are required")
        first = (
            self.start_quaternion
            if index == 0
            else self.waypoint_quaternions[index - 1]
        )
        last = self.waypoint_quaternions[index]
        return _slerp(first, last, self._progress(index, position))

    def can_advance(
        self,
        index,
        position,
        velocity,
        *,
        quaternion=None,
        joint_positions=None,
        acceleration=2.0,
    ):
        """Require settling inside the clearance budget before a corner.

        Position error plus stopping distance consumes at most half the margin;
        the other half remains available for tracking and numerical error.
        Use the Cartesian reference velocity (or a larger measured velocity).
        Optional joint encoders also verify arrival of a redundant elbow branch
        when the end effector alone cannot distinguish two configurations.
        This deliberately does not alter the existing acceleration limit.
        """
        if (
            not self.verified
            or self.execution_kind == "preview"
            or not 0 <= index < len(self.waypoints)
            or acceleration <= 0
            or (self.execution_kind == "joint" and joint_positions is None)
        ):
            return False
        distance = np.linalg.norm(np.asarray(position) - self.waypoints[index])
        stopping = np.dot(velocity, velocity) / (2 * acceleration)
        if distance + stopping > self.clearance_margin / 2:
            return False
        if quaternion is not None:
            actual = np.asarray(quaternion) / np.linalg.norm(quaternion)
            angle = 2 * math.acos(
                np.clip(abs(np.dot(actual, self.waypoint_quaternions[index])), 0.0, 1.0)
            )
            if angle > 0.04:
                return False
        if joint_positions is not None:
            if not 0 <= index < len(self.joint_paths):
                return False
            if np.max(
                np.abs(np.asarray(joint_positions) - self.joint_paths[index][-1])
            ) > (0.002 if self.execution_kind == "joint" else 0.02):
                return False
        return True


def path_length(start, waypoints) -> float:
    points = np.asarray([np.asarray(start), *waypoints], dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _slerp(first, last, fraction):
    first, last = np.asarray(first), np.asarray(last)
    dot = float(np.dot(first, last))
    if dot < 0:
        last, dot = -last, -dot
    angle = math.acos(np.clip(dot, -1.0, 1.0))
    if angle < 1e-6:
        return first.copy()
    return (
        math.sin((1 - fraction) * angle) * first + math.sin(fraction * angle) * last
    ) / math.sin(angle)


class MotionPlanner:
    """Check direct motion, then shortcut a supplied conservative corridor.

    ``via`` is ordered and excludes the final goal. An unsuccessful plan has
    no executable waypoints; callers must hold or request another corridor.
    Cartesian and joint sampling bounds are explicit, so this is a numerical
    preflight rather than a continuous collision-free proof.
    """

    def __init__(
        self,
        model,
        ids,
        ik=None,
        *,
        cube_size=0.05,
        margin=0.008,
        cartesian_step=0.025,
        joint_step=0.06,
        grasp_offset=-0.009,
    ):
        if min(cube_size, margin, cartesian_step, joint_step) <= 0:
            raise ValueError("Planning dimensions and sampling steps must be positive")
        self.ids = ids
        self.num_arms = len(ids.ee_sites)
        # Only this copy's proxy geometry is changed. Live physics is untouched.
        self.model = copy(model)
        self.scratch = mujoco.MjData(self.model)
        self.solver = PandaIK(self.model, ids)
        self.margin = float(margin)
        self.cartesian_step = float(cartesian_step)
        self.joint_step = float(joint_step)
        self.cube_size = float(cube_size)
        if not np.isfinite(grasp_offset):
            raise ValueError("grasp_offset must be finite")
        self.grasp_offset = float(grasp_offset)
        self._active_grasp_offset = self.grasp_offset
        self._near_belt = False
        self.proxy_geom = int(ids.cube_geoms[0])
        self.proxy_qpos = int(ids.cube_qpos[0])
        self.model.geom_type[self.proxy_geom] = int(mujoco.mjtGeom.mjGEOM_BOX)
        self.model.geom_dataid[self.proxy_geom] = -1
        self.model.geom_pos[self.proxy_geom] = 0
        self.model.geom_quat[self.proxy_geom] = [1, 0, 0, 0]
        self.model.geom_sameframe[self.proxy_geom] = int(
            mujoco.mjtSameFrame.mjSAMEFRAME_BODY
        )
        if len(ids.cube_geoms) < 3:
            raise ValueError(
                "Planning needs three object slots for independent proxies"
            )
        self.bin_proxy_geoms = np.asarray(ids.cube_geoms[1:3], dtype=int).copy()
        self.bin_proxy_qpos = np.asarray(ids.cube_qpos[1:3], dtype=int).copy()
        center_z = (ids.bin_floor_z + ids.bin_rim_z) / 2
        self.bin_proxy_centers = np.column_stack(
            (ids.bin_centers[:, :2], np.full(2, center_z))
        )
        self.bin_proxy_half_size = np.r_[
            ids.bin_half_sizes, (ids.bin_rim_z - ids.bin_floor_z) / 2
        ]
        # Treat the known bin interiors as occupied, even before a bin fills.
        # These two boxes exist only in the copied planning model. Their pose
        # and extent never depend on hidden object positions or evaluator state.
        for geom in self.bin_proxy_geoms:
            self.model.geom_type[geom] = int(mujoco.mjtGeom.mjGEOM_BOX)
            self.model.geom_dataid[geom] = -1
            self.model.geom_pos[geom] = 0
            self.model.geom_quat[geom] = [1, 0, 0, 0]
            self.model.geom_sameframe[geom] = int(mujoco.mjtSameFrame.mjSAMEFRAME_BODY)
            self.model.geom_size[geom] = self.bin_proxy_half_size
            self.model.geom_rbound[geom] = np.linalg.norm(self.bin_proxy_half_size)
            self.model.geom_contype[geom] = 1
            self.model.geom_conaffinity[geom] = 1
        self._encoder_indices = np.ones(model.nq, dtype=bool)
        for q in ids.cube_qpos:
            self._encoder_indices[q : q + 7] = False
        self._encoder_indices = np.flatnonzero(self._encoder_indices)
        objects = set(map(int, ids.cube_geoms))
        objects.difference_update(map(int, self.bin_proxy_geoms))
        self._scene_geoms = np.array(
            [
                g
                for g in range(model.ngeom)
                if g not in objects
                and (model.geom_contype[g] or model.geom_conaffinity[g])
            ],
            dtype=int,
        )
        self._pairs = tuple(self._collision_pairs(a) for a in range(self.num_arms))
        self._proxy_obstacles = tuple(
            np.array(
                [
                    g
                    for g in self._scene_geoms
                    if model.geom_bodyid[g] not in set(ids.finger_bodies[a])
                ],
                dtype=int,
            )
            for a in range(self.num_arms)
        )
        self._hand_bodies = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"r{a}/hand")
            for a in range(self.num_arms)
        )
        cups = getattr(ids, "vacuum_geoms", None)
        self._vacuum_cup_geoms = (
            np.full(self.num_arms, -1, dtype=int)
            if cups is None
            else np.asarray(cups, dtype=int).copy()
        )
        self._vacuum_stem_geoms = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"r{a}/vacuum_stem")
            if self._vacuum_cup_geoms[a] >= 0
            else -1
            for a in range(self.num_arms)
        )
        self._checked = 0
        self._minimum = math.inf
        self._escape_progress = None
        self._initialize_aabb_bounds()

    def _initialize_aabb_bounds(self):
        # Compiled local mesh boxes enclose their convex collision geometry.
        # Primitive boxes are refreshed from geom_size below because this
        # scratch model changes the payload and bin proxy type/size at runtime;
        # their compiled geom_aabb values still describe the original objects.
        supported = (
            mujoco.mjtGeom.mjGEOM_SPHERE,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_BOX,
            mujoco.mjtGeom.mjGEOM_MESH,
        )
        self._aabb_supported = np.isin(self.model.geom_type, supported)
        self._aabb_center = self.model.geom_aabb[:, :3].copy()
        self._aabb_half = self.model.geom_aabb[:, 3:].copy()
        self._primitive_geoms = {
            int(kind): np.flatnonzero(self.model.geom_type == int(kind))
            for kind in supported
            if kind != mujoco.mjtGeom.mjGEOM_MESH
        }

    def _aabb_lower_bounds(self, first, second):
        """Conservative world-axis-box distance for supported collision geoms.

        A positive box separation is a lower bound on surface distance.
        Overlapping boxes are never used to reject a geometry pair. Planes,
        heightfields and unknown types retain the original sphere/plane path.
        """
        model, scratch = self.model, self.scratch
        centre = self._aabb_center.copy()
        half = self._aabb_half.copy()
        for kind, ids in self._primitive_geoms.items():
            centre[ids] = 0
            size = model.geom_size[ids]
            if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                half[ids] = size[:, :1]
            elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                half[ids] = size[:, :1]
                half[ids, 2] += size[:, 1]
            elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                half[ids, :2] = size[:, :1]
                half[ids, 2] = size[:, 1]
            else:
                half[ids] = size
        rotation = scratch.geom_xmat.reshape(-1, 3, 3)
        world_center = scratch.geom_xpos + np.einsum("gij,gj->gi", rotation, centre)
        world_half = np.einsum("gij,gj->gi", np.abs(rotation), half)
        separation = np.maximum(
            np.abs(world_center[first] - world_center[second])
            - world_half[first]
            - world_half[second],
            0,
        )
        lower = np.linalg.norm(separation, axis=1) - 1e-10
        lower[~(self._aabb_supported[first] & self._aabb_supported[second])] = -np.inf
        return lower

    def _adjacent(self, first, second):
        model = self.model
        a, b = int(model.geom_bodyid[first]), int(model.geom_bodyid[second])
        if model.body_parentid[a] == b or model.body_parentid[b] == a:
            # Only exempt direct robot links, not the world body's fixtures.
            if a != 0 and b != 0:
                return True
        wa, wb = int(model.body_weldid[a]), int(model.body_weldid[b])
        if wa == wb:
            return True
        pa = int(model.body_weldid[model.body_parentid[wa]])
        pb = int(model.body_weldid[model.body_parentid[wb]])
        # World-attached fixtures are not all "adjacent" to a first link.
        return (wa != 0 and wb != 0) and (pa == wb or pb == wa)

    def _collision_pairs(self, arm):
        model, ids = self.model, self.ids
        moving = [
            g
            for g in self._scene_geoms
            if ids.robot_geom_arm[g] == arm
            and model.body_weldid[model.geom_bodyid[g]] != 0
        ]
        pairs = set()
        for first in moving:
            for second in self._scene_geoms:
                if first == second or self._adjacent(first, second):
                    continue
                if not (
                    (model.geom_contype[first] & model.geom_conaffinity[second])
                    or (model.geom_contype[second] & model.geom_conaffinity[first])
                ):
                    continue
                pairs.add(tuple(sorted((int(first), int(second)))))
        return np.asarray(sorted(pairs), dtype=int).reshape(-1, 2)

    def _initialize(self, data):
        # Avoid even copying hidden object state into the planner. All remaining
        # DOFs are robot/gripper encoders and the controlled bin doors.
        self.scratch.qpos[:] = self.model.qpos0
        self.scratch.qpos[self._encoder_indices] = data.qpos[self._encoder_indices]
        for center, qpos in zip(self.bin_proxy_centers, self.bin_proxy_qpos):
            self.scratch.qpos[qpos : qpos + 7] = [*center, 1, 0, 0, 0]
        self.scratch.qvel[:] = 0
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)

    def _pair_margin(self, first, second, arm, loaded):
        """Keep contact-approach exceptions local to their intended pair."""
        model = self.model
        required = self.margin
        if self._near_belt:
            other = (
                second
                if first == self.ids.belt_geom
                else first
                if second == self.ids.belt_geom
                else None
            )
            if (
                other is not None
                and model.geom_bodyid[other] in self.ids.finger_bodies[arm]
            ):
                required = min(required, 0.001)
        other = (
            second
            if first == self.proxy_geom
            else first
            if second == self.proxy_geom
            else None
        )
        if (
            loaded
            and other is not None
            and model.geom_bodyid[other] == self._hand_bodies[arm]
        ):
            # A calibrated grasp has a small positive fixed gap below its own
            # palm; this does not relax moving-obstacle or proximal-link checks.
            required = min(required, 0.002)
        if loaded and other is not None:
            if other == self._vacuum_cup_geoms[arm]:
                # The real top face touches its own cup face. The conservative
                # proxy expands 1 mm vertically and the calibrated target adds
                # 1 mm seal compression. Permit only those two contributions
                # plus 0.1 mm numerical tolerance.
                # This exception is specific to this arm's cup, not its body,
                # stem, other robot links or another arm's vacuum tool.
                required = min(required, -0.0021)
            elif other == self._vacuum_stem_geoms[arm]:
                # The stem ends 4 mm above the uncompressed top face. Inflation
                # and seal compression leave 2 mm; allow 0.2 mm FK tolerance
                # while retaining a positive clearance around the rigid stem.
                required = min(required, 0.0018)
        return required

    def _clearance(self, arm, loaded):
        model, scratch = self.model, self.scratch
        self._checked += 1
        pairs = self._pairs[arm]
        if loaded:
            site = self.ids.ee_sites[arm]
            q = self.proxy_qpos
            scratch.qpos[q : q + 3] = scratch.site_xpos[site] - [
                0,
                0,
                self._active_grasp_offset,
            ]
            mujoco.mju_mat2Quat(scratch.qpos[q + 3 : q + 7], scratch.site_xmat[site])
            mujoco.mj_kinematics(model, scratch)
            pairs = np.concatenate(
                (
                    pairs,
                    np.column_stack(
                        (
                            np.full(len(self._proxy_obstacles[arm]), self.proxy_geom),
                            self._proxy_obstacles[arm],
                        )
                    ),
                )
            )
        first, second = pairs.T
        lower = (
            np.linalg.norm(scratch.geom_xpos[first] - scratch.geom_xpos[second], axis=1)
            - model.geom_rbound[first]
            - model.geom_rbound[second]
        )
        plane = (model.geom_type[first] == int(mujoco.mjtGeom.mjGEOM_PLANE)) | (
            model.geom_type[second] == int(mujoco.mjtGeom.mjGEOM_PLANE)
        )
        sphere_lower = lower
        lower = np.maximum(lower, self._aabb_lower_bounds(first, second))
        # Preserve the existing capped minimum-distance diagnostic: an exact
        # query pruned here would return its distmax (the unchanged margin).
        if np.any((sphere_lower < self.margin) & (lower >= self.margin) & ~plane):
            self._minimum = min(self._minimum, self.margin)
        # Progress belongs to the whole accepted configuration. Do not retain
        # improvements from an earlier pair when a later pair rejects a sample.
        progress = (
            None if self._escape_progress is None else self._escape_progress.copy()
        )
        if progress is not None:
            for index in np.flatnonzero((lower >= self.margin) & ~plane):
                pair = tuple(map(int, pairs[index]))
                if pair in progress:
                    progress[pair] = self._pair_margin(*pair, arm, loaded)
        for index in np.flatnonzero((lower < self.margin) | plane):
            distance = float(
                mujoco.mj_geomDistance(
                    model,
                    scratch,
                    int(first[index]),
                    int(second[index]),
                    self.margin,
                    None,
                )
            )
            self._minimum = min(self._minimum, distance)
            required = self._pair_margin(first[index], second[index], arm, loaded)
            pair = tuple(map(int, pairs[index]))
            limit = required
            if progress is not None and pair in progress:
                previous = progress[pair]
                # Initially deficient pairs may only separate. Once their full
                # margin is restored, ordinary strict checking is permanent.
                if previous < required:
                    limit = previous - 1e-6
                progress[pair] = min(required, max(previous, distance))
            if distance < limit:
                names = [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                    or f"geom#{g}"
                    for g in pairs[index]
                ]
                return False, f"clearance {distance:.4f} m: {' / '.join(names)}"
        if progress is not None:
            self._escape_progress = progress
        return True, "clear"

    def _segment(
        self, arm, goal, quaternion, loaded, *, cartesian_step=None, joint_step=None
    ):
        scratch, ids = self.scratch, self.ids
        start = scratch.site_xpos[ids.ee_sites[arm]].copy()
        initial_quat = np.empty(4)
        mujoco.mju_mat2Quat(initial_quat, scratch.site_xmat[ids.ee_sites[arm]])
        angle = 2 * math.acos(np.clip(abs(np.dot(initial_quat, quaternion)), 0, 1))
        samples = max(
            1,
            math.ceil(
                np.linalg.norm(goal - start)
                / (self.cartesian_step if cartesian_step is None else cartesian_step)
            ),
            math.ceil(angle / 0.12),
        )
        travel = 0.0
        self._segment_joints = [scratch.qpos[ids.arm_qpos[arm]].copy()]
        for fraction in np.linspace(0, 1, samples + 1)[1:]:
            position = start + fraction * (goal - start)
            rotation = _slerp(initial_quat, quaternion, fraction)
            previous = scratch.qpos[ids.arm_qpos[arm]].copy()
            solution = self.solver.solve_pose(
                scratch, arm, position, rotation, max_iterations=45
            )
            ep, er = self.solver.pose_error(scratch, arm, solution, position, rotation)
            if ep > 0.004 or er > 0.04:
                return False, f"IK residual {ep:.4f} m / {er:.4f} rad", travel
            steps = max(
                1,
                math.ceil(
                    np.max(np.abs(solution - previous))
                    / (self.joint_step if joint_step is None else joint_step)
                ),
            )
            for alpha in np.linspace(0, 1, steps + 1)[1:]:
                scratch.qpos[ids.arm_qpos[arm]] = previous + alpha * (
                    solution - previous
                )
                mujoco.mj_kinematics(self.model, scratch)
                ok, reason = self._clearance(arm, loaded)
                if not ok:
                    return False, reason, travel
            travel += float(np.linalg.norm(solution - previous))
            self._segment_joints.append(solution.copy())
        return True, "clear", travel

    def _joint_home_path(self, arm):
        """Check calibrated home joints and retain their FK execution segments."""
        first = self.scratch.qpos[self.ids.arm_qpos[arm]].copy()
        last = self.ids.home_qpos[arm]
        count = max(1, math.ceil(np.max(np.abs(last - first)) / 0.06))
        points, rotations, paths = [], [], []
        cartesian_samples = [self.scratch.site_xpos[self.ids.ee_sites[arm]].copy()]
        for index in range(count):
            previous = first + (last - first) * index / count
            following = first + (last - first) * (index + 1) / count
            steps = max(
                1,
                math.ceil(
                    np.max(np.abs(following - previous)) / min(self.joint_step, 0.01)
                ),
            )
            path = np.linspace(previous, following, steps + 1)
            for sample in path[1:]:
                self.scratch.qpos[self.ids.arm_qpos[arm]] = sample
                mujoco.mj_kinematics(self.model, self.scratch)
                ok, reason = self._clearance(arm, False)
                if not ok:
                    return False, reason, (), (), (), 0.0
                cartesian_samples.append(
                    self.scratch.site_xpos[self.ids.ee_sites[arm]].copy()
                )
            points.append(self.scratch.site_xpos[self.ids.ee_sites[arm]].copy())
            rotation = np.empty(4)
            mujoco.mju_mat2Quat(
                rotation, self.scratch.site_xmat[self.ids.ee_sites[arm]]
            )
            rotations.append(rotation)
            paths.append(path)
        self._joint_cartesian_samples = np.asarray(cartesian_samples)
        return (
            True,
            "clear",
            points,
            rotations,
            paths,
            float(np.linalg.norm(last - first)),
        )

    def _continuous_joint_home(self, plan):
        """Sampling knots are collision certificates, not stopping points.

        This path is one straight line in joint space. Concatenating the
        checked samples changes only its execution parameterization; there is
        no corner cutting or new interpolation between unconnected segments.
        """
        samples = np.concatenate(
            [plan.joint_paths[0], *(path[1:] for path in plan.joint_paths[1:])]
        )
        return replace(
            plan,
            waypoints=(plan.waypoints[-1],),
            waypoint_quaternions=(plan.waypoint_quaternions[-1],),
            joint_paths=(samples,),
            execution_kind="joint",
            joint_cartesian_samples=self._joint_cartesian_samples.copy(),
            length=path_length(
                self._joint_cartesian_samples[0], self._joint_cartesian_samples[1:]
            ),
        )

    def plan_home(self, data, arm, *, near_belt=False, grasp_offset=None) -> MotionPlan:
        """Return an unloaded path to the calibrated seven-joint home pose.

        Matching only the home tool pose leaves the redundant elbow branch
        unspecified and can make every subsequent pick unreachable. This path
        ends at ``ids.home_qpos`` and enforces ordinary margins at its initial
        state and every sampled configuration. A clearance deficiency must be
        resolved separately by an explicitly checked escape.
        """
        if not isinstance(arm, (int, np.integer)) or not 0 <= arm < self.num_arms:
            raise ValueError(f"arm must be an integer in [0, {self.num_arms - 1}]")
        if not isinstance(near_belt, (bool, np.bool_)):
            raise ValueError("near_belt must be a boolean")
        active_offset = (
            self.grasp_offset if grasp_offset is None else float(grasp_offset)
        )
        if not np.isfinite(active_offset):
            raise ValueError("grasp_offset must be finite")
        self._near_belt = bool(near_belt)
        self._active_grasp_offset = active_offset
        self._escape_progress = None
        self._checked, self._minimum = 0, self.margin
        self._initialize(data)
        start = self.scratch.site_xpos[self.ids.ee_sites[arm]].copy()
        start_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(
            start_quaternion, self.scratch.site_xmat[self.ids.ee_sites[arm]]
        )
        start_joints = data.qpos[self.ids.arm_qpos].copy()
        direct = float(np.linalg.norm(self.ids.home_ee[arm] - start))
        ok, reason = self._clearance(arm, False)
        points, rotations, paths, travel = (), (), (), 0.0
        if ok:
            ok, reason, points, rotations, paths, travel = self._joint_home_path(arm)
        else:
            reason = "initial " + reason
        plan = MotionPlan(
            tuple(points),
            ok,
            "joint home" if ok else "joint home " + reason,
            path_length(start, points) if ok else 0.0,
            direct,
            self._checked,
            self._minimum,
            travel,
            start,
            start_quaternion,
            tuple(rotations),
            self.margin,
            tuple(paths),
            start_joints,
        )
        return self._continuous_joint_home(plan) if ok else plan

    def grasp_exit_feasible(
        self,
        data,
        arm,
        closure_position,
        lift_position,
        *,
        quaternion=None,
        grasp_offset=None,
        grasp_position=None,
    ) -> MotionPlan:
        """Preview a moving grasp through closure and the entire robot exit.

        The optional ``grasp_position`` is the predicted low pose at descent
        completion; ``closure_position`` includes the subsequent belt travel.
        Each segment starts on the exact joint branch reached by the preceding
        segment, rather than solving the exit independently from the high pose.
        Only robot/gripper/gate encoders are copied into a separate MjData;
        hidden workpiece poses, velocities and live actuator commands are never
        read or changed.

        This is an admission check, not an executable or time-parameterized
        trajectory. Like the ordinary initial lift, it checks the robot and
        fixed scene with ``near_belt=True`` and no carried-object proxy: the
        payload initially rests on the belt. It does not certify finger
        closure, future motion of another arm, or payload/contact dynamics.
        Execution must retain reservations, sensing and loaded transit checks.
        """
        if not isinstance(arm, (int, np.integer)) or not 0 <= arm < self.num_arms:
            raise ValueError(f"arm must be an integer in [0, {self.num_arms - 1}]")
        closure = np.asarray(closure_position, dtype=float)
        lift = np.asarray(lift_position, dtype=float)
        stages = []
        if grasp_position is not None:
            stages.append(("descent", np.asarray(grasp_position, dtype=float)))
        stages.extend((("closure", closure), ("lift", lift)))
        if any(p.shape != (3,) or not np.isfinite(p).all() for _, p in stages):
            raise ValueError("Every preview position must be a finite 3D point")
        if lift[2] <= closure[2]:
            raise ValueError("The lift endpoint must be above the closure endpoint")

        preview = mujoco.MjData(self.model)
        preview.qpos[self._encoder_indices] = data.qpos[self._encoder_indices]
        mujoco.mj_kinematics(self.model, preview)
        start = preview.site_xpos[self.ids.ee_sites[arm]].copy()
        start_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(start_quaternion, preview.site_xmat[self.ids.ee_sites[arm]])
        start_joints = preview.qpos[self.ids.arm_qpos].copy()
        waypoints, rotations, paths = [], [], []
        checked, minimum, travel, length = 0, self.margin, 0.0, 0.0

        def result(verified, reason):
            return MotionPlan(
                tuple(waypoints) if verified else (),
                verified,
                reason,
                length if verified else 0.0,
                float(np.linalg.norm(lift - start)),
                checked,
                minimum,
                travel,
                start,
                start_quaternion,
                tuple(rotations) if verified else (),
                self.margin,
                tuple(paths) if verified else (),
                start_joints,
                execution_kind="preview",
            )

        for name, position in stages:
            segment = self.plan(
                preview,
                arm,
                position,
                quaternion=quaternion,
                loaded=False,
                near_belt=True,
                grasp_offset=grasp_offset,
            )
            checked += segment.checked_samples
            minimum = min(minimum, segment.minimum_clearance)
            travel += segment.joint_travel
            if not segment.verified:
                return result(False, f"grasp exit {name}: {segment.reason}")
            length += segment.length
            waypoints.extend(segment.waypoints)
            rotations.extend(segment.waypoint_quaternions)
            paths.extend(segment.joint_paths)
            preview.qpos[self.ids.arm_qpos[arm]] = segment.joint_paths[-1][-1]
            mujoco.mj_kinematics(self.model, preview)
        return result(True, "grasp exit verified (robot geometry preview only)")

    def plan(
        self,
        data,
        arm,
        goal,
        *,
        quaternion=None,
        loaded=False,
        object_size=None,
        via=(),
        near_belt=False,
        grasp_offset=None,
    ) -> MotionPlan:
        if not isinstance(arm, (int, np.integer)) or not 0 <= arm < self.num_arms:
            raise ValueError(f"arm must be an integer in [0, {self.num_arms - 1}]")
        goal = np.asarray(goal, dtype=float)
        points = [np.asarray(p, dtype=float) for p in via] + [goal]
        if any(p.shape != (3,) or not np.isfinite(p).all() for p in points):
            raise ValueError("Every waypoint must be a finite 3D point")
        quaternion = np.asarray(
            self.solver.default_quats[arm] if quaternion is None else quaternion,
            dtype=float,
        )
        if (
            quaternion.shape != (4,)
            or not np.isfinite(quaternion).all()
            or np.linalg.norm(quaternion) < 1e-9
        ):
            raise ValueError("quaternion must be finite and nonzero")
        quaternion = quaternion / np.linalg.norm(quaternion)
        size = np.asarray(
            [self.cube_size] * 3 if object_size is None else object_size, dtype=float
        )
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("object_size must contain three positive full dimensions")
        if not isinstance(near_belt, (bool, np.bool_)):
            raise ValueError("near_belt must be a boolean")
        active_offset = (
            self.grasp_offset if grasp_offset is None else float(grasp_offset)
        )
        if not np.isfinite(active_offset):
            raise ValueError("grasp_offset must be finite")
        self._active_grasp_offset = active_offset
        self._near_belt = bool(near_belt)
        # Shape dimensions are expressed in the estimated object/grasp axes.
        # The box follows tool orientation, with 2 mm horizontal and 1 mm
        # vertical uncertainty per face. Preserving measured height avoids
        # treating a wide, short ellipsoid as a tall cube. This assumes the
        # calibrated upright grasp, not stability under arbitrary object roll.
        half_size = size / 2 + [0.002, 0.002, 0.001]
        self.model.geom_size[self.proxy_geom] = half_size
        self.model.geom_rbound[self.proxy_geom] = np.linalg.norm(half_size)
        self._checked, self._minimum = 0, self.margin
        self._initialize(data)
        start = self.scratch.site_xpos[self.ids.ee_sites[arm]].copy()
        start_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(
            start_quaternion, self.scratch.site_xmat[self.ids.ee_sites[arm]]
        )

        accepted_joint_paths = []

        def result(
            waypoints,
            verified,
            reason,
            length,
            direct_length,
            checked_samples,
            minimum_clearance,
            joint_travel,
        ):
            return MotionPlan(
                waypoints,
                verified,
                reason,
                length,
                direct_length,
                checked_samples,
                minimum_clearance,
                joint_travel,
                start.copy(),
                start_quaternion.copy(),
                tuple(quaternion.copy() for _ in waypoints),
                self.margin,
                tuple(accepted_joint_paths) if verified else (),
            )

        direct = float(np.linalg.norm(goal - start))
        initial_ok, reason = self._clearance(arm, loaded)
        if not initial_ok:
            return result(
                (),
                False,
                "initial " + reason,
                0.0,
                direct,
                self._checked,
                self._minimum,
                0.0,
            )
        start_qpos = self.scratch.qpos.copy()
        initial_minimum = self._minimum
        ok, reason, travel = self._segment(arm, goal, quaternion, loaded)
        if ok:
            accepted_joint_paths.append(np.array(self._segment_joints))
            return result(
                (goal.copy(),),
                True,
                "direct",
                direct,
                direct,
                self._checked,
                self._minimum,
                travel,
            )
        direct_reason = reason
        if len(via) == 0:
            return result(
                (), False, direct_reason, 0.0, direct, self._checked, self._minimum, 0.0
            )
        self.scratch.qpos[:] = start_qpos
        self._minimum = initial_minimum
        mujoco.mj_kinematics(self.model, self.scratch)
        selected, total_travel, index = [], 0.0, 0
        while index < len(points):
            segment_start = self.scratch.qpos.copy()
            segment_minimum = self._minimum
            # Farthest reachable waypoint first; restore the joint branch after
            # every failed shortcut rather than starting it inside an obstacle.
            for candidate in range(len(points) - 1, index - 1, -1):
                self.scratch.qpos[:] = segment_start
                self._minimum = segment_minimum
                mujoco.mj_kinematics(self.model, self.scratch)
                ok, reason, travel = self._segment(
                    arm, points[candidate], quaternion, loaded
                )
                if ok:
                    selected.append(points[candidate].copy())
                    accepted_joint_paths.append(np.array(self._segment_joints))
                    total_travel += travel
                    index = candidate + 1
                    break
            else:
                return result(
                    (), False, reason, 0.0, direct, self._checked, self._minimum, 0.0
                )
        return result(
            tuple(selected),
            True,
            "corridor: " + direct_reason,
            path_length(start, selected),
            direct,
            self._checked,
            self._minimum,
            total_travel,
        )

    def plan_escape(
        self,
        data,
        arm,
        goal,
        *,
        quaternion=None,
        via=(),
        near_belt=False,
        grasp_offset=None,
    ) -> MotionPlan:
        """Find an unloaded exit from a small initial clearance deficiency.

        Each initially deficient pair must separate monotonically at every
        sampled joint configuration. Existing penetration deeper than 0.5 mm
        is rejected; all other pairs retain ordinary margins. An endpoint is
        executable only after *every* pair has regained its full margin.
        For the calibrated home goal, a checked joint-space exit can restore
        its calibrated orientation without a Cartesian IK branch change. A
        short partial exit can otherwise precede ordinary recovery planning.
        This operates on encoder-derived scratch state, never live physics.
        """
        if not isinstance(arm, (int, np.integer)) or not 0 <= arm < self.num_arms:
            raise ValueError(f"arm must be an integer in [0, {self.num_arms - 1}]")
        goal = np.asarray(goal, dtype=float)
        via = tuple(np.asarray(point, dtype=float) for point in via)
        if any(
            point.shape != (3,) or not np.isfinite(point).all()
            for point in (*via, goal)
        ):
            raise ValueError("Every waypoint must be a finite 3D point")
        if not isinstance(near_belt, (bool, np.bool_)):
            raise ValueError("near_belt must be a boolean")
        active_offset = (
            self.grasp_offset if grasp_offset is None else float(grasp_offset)
        )
        if not np.isfinite(active_offset):
            raise ValueError("grasp_offset must be finite")
        self._near_belt = bool(near_belt)
        self._active_grasp_offset = active_offset
        self._checked, self._minimum = 0, self.margin
        self._initialize(data)
        start = self.scratch.site_xpos[self.ids.ee_sites[arm]].copy()
        start_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(
            start_quaternion, self.scratch.site_xmat[self.ids.ee_sites[arm]]
        )
        quaternion = np.asarray(
            start_quaternion if quaternion is None else quaternion, dtype=float
        )
        if (
            quaternion.shape != (4,)
            or not np.isfinite(quaternion).all()
            or np.linalg.norm(quaternion) < 1e-9
        ):
            raise ValueError("quaternion must be finite and nonzero")
        quaternion = quaternion / np.linalg.norm(quaternion)
        start_qpos = self.scratch.qpos.copy()
        # Record all arms' actual encoder readings, including the stationary
        # configurations on which this check depends.
        start_joints = data.qpos[self.ids.arm_qpos].copy()
        direct = float(np.linalg.norm(goal - start))
        deficient = {}
        initial_clearances = []
        for first, second in self._pairs[arm]:
            pair = (int(first), int(second))
            required = self._pair_margin(*pair, arm, False)
            distance = float(
                mujoco.mj_geomDistance(
                    self.model, self.scratch, *pair, self.margin, None
                )
            )
            self._minimum = min(self._minimum, distance)
            if distance < required:
                deficient[pair] = distance
                initial_clearances.append((*pair, distance, required))
        initial_minimum = self._minimum

        def result(points, rotations, paths, verified, reason, travel=0.0):
            return MotionPlan(
                tuple(point.copy() for point in points),
                verified,
                "escape " + reason,
                path_length(start, points) if verified else 0.0,
                direct,
                self._checked,
                self._minimum,
                travel,
                start.copy(),
                start_quaternion.copy(),
                tuple(rotation.copy() for rotation in rotations),
                self.margin,
                tuple(paths),
                start_joints.copy(),
                tuple(initial_clearances),
            )

        if initial_minimum < -0.0005:
            return result((), (), (), False, "initial penetration exceeds 0.5 mm")

        def restore_start():
            # A failed segment may have made substantial separation progress.
            # Both that progress and the joint branch must roll back before
            # trying a different path from the measured starting state.
            self.scratch.qpos[:] = start_qpos
            self._minimum = initial_minimum
            self._escape_progress = deficient.copy()
            mujoco.mj_kinematics(self.model, self.scratch)

        def attempt(points, rotations):
            restore_start()
            paths, travel = [], 0.0
            for point, rotation in zip(points, rotations):
                ok, reason, segment_travel = self._segment(
                    arm,
                    point,
                    rotation,
                    False,
                    cartesian_step=min(self.cartesian_step, 0.005),
                    joint_step=min(self.joint_step, 0.01),
                )
                if not ok:
                    return False, reason, (), 0.0
                paths.append(np.array(self._segment_joints))
                travel += segment_travel
            self._escape_progress = None
            ok, reason = self._clearance(arm, False)
            return ok, reason, paths if ok else (), travel if ok else 0.0

        try:
            full_paths = [(goal,)]
            if via:
                full_paths.append((*via, goal))
            for points in full_paths:
                rotations = (quaternion,) * len(points)
                ok, reason, paths, travel = attempt(points, rotations)
                if ok:
                    return result(points, rotations, paths, True, "full", travel)
            if np.linalg.norm(goal - self.ids.home_ee[arm]) < 0.005:
                # Cartesian IK can be trapped in a redundant elbow branch even
                # after a local upward exit restores clearance. For an empty
                # hand, the calibrated home joints provide another concrete
                # path to check. FK supplies every segment's position and
                # orientation; execution follows these exact joint samples.
                restore_start()
                ok, reason, points, rotations, paths, travel = self._joint_home_path(
                    arm
                )
                if ok:
                    self._escape_progress = None
                    ok, reason = self._clearance(arm, False)
                    if ok:
                        return self._continuous_joint_home(
                            result(
                                points,
                                rotations,
                                paths,
                                True,
                                "joint home",
                                travel,
                            )
                        )
            if not deficient:
                return result((), (), (), False, "no deficient pairs; " + reason)
            # Keep the measured wrist orientation while releasing a proximal
            # link. Short vertical/side exits avoid forcing a full home path
            # while the starting branch is still close to a bin wall.
            directions = [
                [0, 0, 1],
                [1, 0, 1],
                [-1, 0, 1],
                [0, 1, 1],
                [0, -1, 1],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
            ]
            for length in (0.03, 0.05, 0.08):
                for direction in directions:
                    direction = np.asarray(direction, dtype=float)
                    endpoint = start + length * direction / np.linalg.norm(direction)
                    ok, reason, paths, travel = attempt(
                        (endpoint,), (start_quaternion,)
                    )
                    if ok:
                        return result(
                            (endpoint,),
                            (start_quaternion,),
                            paths,
                            True,
                            "partial",
                            travel,
                        )
            return result((), (), (), False, "no monotone exit; " + reason)
        finally:
            self._escape_progress = None

    def lowest_safe_goal(self, data, arm, xy, heights, **kwargs):
        """Return the first verified height from a caller's ordered candidates."""
        last = None
        for height in heights:
            last = self.plan(data, arm, [*xy, height], **kwargs)
            if last.verified:
                return last
        if last is None:
            raise ValueError("At least one height is required")
        return last
