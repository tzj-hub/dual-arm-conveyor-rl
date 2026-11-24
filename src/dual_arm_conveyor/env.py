"""Contact simulation with an explicit sensor/controller/evaluator boundary.

Vision control consumes image-derived tracks and robot joint encoders only.
Ground truth is used by spawning, the oracle ablation and the evaluator; the
latter never changes operational beliefs. Objects are never attached/teleported.
"""

from __future__ import annotations
from dataclasses import replace
from enum import IntEnum
from collections import Counter, deque
import math
import numpy as np
import mujoco
from .config import Config
from .model import build_scene
from .ik import PandaIK, downward_quaternion
from .vision import SensorPipeline, Track, fit_prism_outline, symmetric_angle
from .motion import MotionPlanner
from .wrist import WristSensorPipeline
from .live_view import CameraViewGroup
from .belt_visual import BeltMarkers
from .conveyor_speed import BeltMotion
from .supply import LaneSupply
from .utilization import ArmUtilization
from .cooperation import OpportunityMonitor


class State(IntEnum):
    PENDING = 0
    BELT = 1
    RESERVED = 2
    HELD = 3
    RELEASED = 4
    SORTED = 5
    MISSED = 6
    DROPPED = 7


WAIT = 10
ACTION_COUNT = 121
PHASES = (
    "idle",
    "approach",
    "reorient",
    "descend",
    "close",
    "lift",
    "depart",
    "transfer_wait",
    "align",
    "transport",
    "place",
    "release",
    "retreat",
    "return",
    "recover",
    "yield",
)


class ConveyorEnv:
    def __init__(self, config: Config | None = None, seed: int | None = None):
        self.config = config or Config()
        self.model, self.ids = build_scene(self.config)
        self.num_arms = len(self.ids.ee_sites)
        self.suction = self.config.gripper_mode == "vacuum"
        self.vacuum = None
        if self.suction:
            from .vacuum import VacuumGripper

            self.vacuum = VacuumGripper(self.model, self.ids)
        self.action_count = 11**self.num_arms
        self.wait_action = self.action_count - 1
        self.action_choices = np.array(
            np.unravel_index(np.arange(self.action_count), (11,) * self.num_arms)
        ).T
        if not hasattr(self.model, "geom_surfacevel"):
            raise RuntimeError("Physical conveyor requires MuJoCo >=3.11")
        self.data = mujoco.MjData(self.model)
        self.belt_markers = BeltMarkers(self.model, self.config)
        self.ik = PandaIK(
            self.model,
            self.ids,
            max_cartesian_speed=self.config.control_speed,
            adaptive=self.config.strategy == "adaptive",
        )
        self.planner = MotionPlanner(
            self.model, self.ids, grasp_offset=self.config.grasp_offset
        )
        self.sensor = (
            SensorPipeline(self.model, self.config)
            if self.config.sensor_mode == "vision"
            else None
        )
        self.wrist = (
            WristSensorPipeline(self.model, self.config)
            if self.config.wrist_camera_mode != "off"
            else None
        )
        self.camera_display = (
            CameraViewGroup(self.sensor, self.wrist)
            if self.wrist is not None
            else self.sensor
        )
        self._oracle_outline_candidates = None
        if self.sensor is None:
            self._oracle_outline_candidates = []
            for spec in self.ids.object_specs:
                if spec.prism_sides is None:
                    xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.asarray(
                        spec.half_sizes[:2]
                    )
                else:
                    xy = spec.prism_mesh()[0][: spec.prism_sides, :2]
                self._oracle_outline_candidates.append(fit_prism_outline(xy)[3])
        self.windows = (
            np.array([[-0.65, 0.25], [-0.25, 0.65]])
            if self.num_arms == 2
            else np.array(
                [[-0.95, -0.10], [-0.95, -0.10], [-0.15, 0.82], [-0.15, 0.82]]
            )
        )
        self.seed = self.config.seed if seed is None else seed
        self.reset(self.seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        c, d, ids = self.config, self.data, self.ids
        # Observational only, including in the legacy-controller comparison.
        # Sampled at scheduling frequency; never changes masks or dispatch.
        self.cooperation_monitor = OpportunityMonitor(enabled=self.num_arms == 4)
        mujoco.mj_resetData(self.model, d)
        self.belt_motion = BeltMotion(c, self.seed)
        self.belt_sample = self.belt_motion.sample(0.0)
        self.belt_speed_min = self.belt_speed_max = self.belt_sample.speed
        self.model.geom_surfacevel[ids.belt_geom, 0] = self.belt_sample.speed
        self.belt_markers.update(d, distance=self.belt_sample.distance)
        n = c.n_objects
        # Truth belongs to spawning/evaluation only. IDs are lifetime serials,
        # while physical body slots are reused after leaving the active cell.
        self.states = np.full(n, State.PENDING, dtype=np.int8)
        self.owner = np.full(n, -1, dtype=int)
        self.types = np.zeros(n, dtype=int)
        self.serial = np.full(n, -1, dtype=int)
        self.generations = np.zeros(n, dtype=int)
        self.yaws = np.zeros(n)
        self.lateral = np.zeros(n)
        self.spawned = 0
        self.next_arrival = 0.0
        self.next_arrival_distance = 0.0
        self.arrival_distances = deque(maxlen=512)
        self.next_pool_slot = 0
        self.arrivals = deque(maxlen=512)
        self.spawn_yaw_log = deque(maxlen=512)
        self.spawn_position_log = deque(maxlen=512)
        self.spawn_lane_log = deque(maxlen=512)
        self.supply_source = LaneSupply(c, self.seed) if c.supply_lanes == 2 else None
        self.supply_inlet_blocked_steps = Counter()
        self.supply_wait_distance_sum = 0.0
        self.supply_max_wait_distance = 0.0
        self.supply_blocked_steps = 0
        self.recycled = 0
        self.max_active_objects = 0
        self.archived = Counter()
        self.archived_arm_color = np.zeros((self.num_arms, 2), dtype=int)
        self.archived_cycles_sum = 0.0
        self.object_records = deque(maxlen=512)
        self.shape_spawned = Counter()
        self.shape_sorted = Counter()
        self.material_spawned = Counter()
        self.truth_target = np.full(self.num_arms, -1, dtype=int)
        self.object_dispatch_time = np.zeros(n)
        self.release_time = np.zeros(n)
        self.cube_settle = np.zeros(n)
        self.cycle_times = np.full(n, np.nan)
        # Reward bookkeeping is separate from revocable physical outcomes.
        # Slot-local serials keep continuous runs bounded when bodies recycle.
        self.rewarded_completion_serial = np.full(n, -1, dtype=int)
        self.revoked_completion_serial = np.full(n, -1, dtype=int)
        self.first_completion_rewards = 0
        self.revoked_completion_rewards = 0
        self.arm_completions = np.zeros(self.num_arms, dtype=int)
        self.eval_queue = []
        self.vision_sq_error = 0.0
        self.vision_yaw_error = 0.0
        self.vision_match_count = 0
        self.vision_yaw_count = 0
        self.vision_id_matches = {}
        self.vision_id_switches = 0
        self._eval_last_frame = -1.0
        # Operational beliefs contain only sensor tracks and robot encoders.
        self.tracks = [None] * 10
        self.track_slots = {}
        self.retired_tracks = {}
        self.belief = np.zeros(10, dtype=np.int8)
        self.phase = ["idle"] * self.num_arms
        self.phase_start = np.zeros(self.num_arms)
        self.target = np.full(self.num_arms, -1, dtype=int)
        self.idle_anchors = ids.home_ee.copy()
        self.preposition_active = np.zeros(self.num_arms, dtype=bool)
        self._previous_preposition_active = np.zeros(self.num_arms, dtype=bool)
        self._preposition_stationary_since = np.zeros(self.num_arms)
        self.preposition_seconds = np.zeros(self.num_arms)
        self.preposition_starts = np.zeros(self.num_arms, dtype=int)
        self._preposition_live_goal = [None] * self.num_arms
        self._preposition_live_yaw = np.full(self.num_arms, np.nan)
        self.return_goals = ids.home_ee.copy()
        self.planned_return_active = np.zeros(self.num_arms, dtype=bool)
        self.planned_return_attempts = np.zeros(self.num_arms, dtype=int)
        self.planned_return_completed = np.zeros(self.num_arms, dtype=int)
        self.planned_return_fallbacks = np.zeros(self.num_arms, dtype=int)
        self.direct_retask_returns = np.zeros(self.num_arms, dtype=int)
        self.direct_retask_fallbacks = np.zeros(self.num_arms, dtype=int)
        self.last_goals = ids.home_ee.copy()
        self.lift_xy = np.zeros((self.num_arms, 2))
        self.close_position = np.zeros((self.num_arms, 3))
        self.close_displacement = np.zeros(self.num_arms)
        self.lift_displacement = np.zeros(self.num_arms)
        self.lift_start_z = np.zeros(self.num_arms)
        self.lift_deceleration_start = np.full(self.num_arms, np.nan)
        self.grasp_contact_since = np.full(self.num_arms, np.nan)
        self.descent_guard_stops = Counter()
        self.grasp_yaw = np.array(
            [np.pi / 2 if a % 2 == 0 else -np.pi / 2 for a in range(self.num_arms)]
        )
        self.tool_yaw = self.grasp_yaw.copy()
        self.grasp_sizes = np.full((self.num_arms, 3), 0.05)
        self.grasp_min_widths = np.full(self.num_arms, 0.05)
        self.grasp_offsets = np.full(self.num_arms, c.grasp_offset)
        self.wrist_offsets = np.zeros((self.num_arms, 2))
        self.wrist_control_seconds = np.zeros(self.num_arms)
        self.wrist_max_applied = np.zeros(self.num_arms)
        self.wrist_used_frames = np.zeros(self.num_arms, dtype=int)
        self.wrist_last_used_timestamp = np.full(self.num_arms, -np.inf)
        self.reoriented = np.zeros(self.num_arms, dtype=bool)
        self.reorient_goals = ids.home_ee.copy()
        self.drop_targets = np.zeros((10, 3))
        self.drop_kind = np.zeros(10, dtype=int)
        self.bin_slots = np.zeros(2, dtype=int)
        self.bin_occupied = np.zeros((2, c.bin_capacity), dtype=bool)
        self.drop_slot = np.full(10, -1, dtype=int)
        self.bin_owner = np.full(2, -1, dtype=int)
        self.bin_zone_owner = np.full((self.num_arms // 2, 2), -1, dtype=int)
        self.bin_service = ["closed", "closed"]
        self.bin_service_start = np.zeros(2)
        self.bin_last_release = np.full(2, -100.0)
        self.bin_release_counts = np.zeros(2, dtype=int)
        self.bin_unloads = np.zeros(2, dtype=int)
        self.bin_drain_requested = np.zeros(2, dtype=bool)
        self.transfer_owner = -1
        self.retry_after = np.zeros((self.num_arms, 10))
        self.dispatch_time = np.zeros(self.num_arms)
        self.arm_assignments = np.zeros(self.num_arms, dtype=int)
        self.assignments_at_last_sort = np.zeros(self.num_arms, dtype=int)
        self.max_assignments_without_sort = np.zeros(self.num_arms, dtype=int)
        self.last_sort_time = np.zeros(self.num_arms)
        self.arm_releases = np.zeros(self.num_arms, dtype=int)
        self.holding = np.zeros(self.num_arms, dtype=bool)
        self.empty_since = np.full(self.num_arms, np.nan)
        self.release_clear_since = np.full(self.num_arms, np.nan)
        self.release_start_aperture = np.full(self.num_arms, np.nan)
        self.responsive_release_count = 0
        self.quat_override = [None] * self.num_arms
        self.hold_requested = np.zeros(self.num_arms, dtype=bool)
        self.active_motion_plans = [None] * self.num_arms
        self.active_motion_indices = np.zeros(self.num_arms, dtype=int)
        self.routes = [None] * self.num_arms
        self.route_index = np.zeros(self.num_arms, dtype=int)
        self.route_retry_time = np.zeros(self.num_arms)
        self.route_failures = 0
        self.route_shortcuts = 0
        self.escape_routes = np.zeros(self.num_arms, dtype=bool)
        self.escape_attempts = 0
        self.escape_successes = 0
        self.release_timeouts = 0
        self.recovery_withdrawal_goal = [None] * self.num_arms
        self.recovery_withdrawal_quaternion = [None] * self.num_arms
        self.recovery_withdrawal_start = np.zeros(self.num_arms)
        self.recovery_withdrawal_used = np.zeros(self.num_arms, dtype=bool)
        self.recovery_withdrawal_attempts = np.zeros(self.num_arms, dtype=int)
        self.recovery_withdrawal_successes = np.zeros(self.num_arms, dtype=int)
        self.recovery_withdrawal_fault = [None] * self.num_arms
        self.recovery_home_returns = np.zeros(self.num_arms, dtype=int)
        self.home_returns = np.zeros(self.num_arms, dtype=int)
        self.path_distance = np.zeros(self.num_arms)
        self.joint_distance = np.zeros(self.num_arms)
        self.phase_seconds = Counter()
        self.phase_duration_samples = [
            {phase: deque(maxlen=128) for phase in PHASES}
            for _ in range(self.num_arms)
        ]
        self.utilization = ArmUtilization(self.num_arms, window_seconds=30.0)
        self.max_phase_duration = Counter()
        self.events = deque(maxlen=2000)
        self.collision_steps = self.robot_collision_steps = self.invalid_actions = (
            self.retries
        ) = 0
        self.robot_contact_pair_steps = Counter()
        self.robot_contact_pair_penetration = Counter()
        self.robot_contact_pair_phase_steps = Counter()
        self.shared_wait_time = self.overlap_time = self.active_time = (
            self.max_penetration
        ) = 0.0
        self._done = False
        self.termination_reason = "running"
        self._last_mask = None
        for a in range(self.num_arms):
            d.qpos[ids.arm_qpos[a]] = ids.home_qpos[a]
            d.qpos[ids.finger_qpos[a]] = 0.04
            d.ctrl[ids.arm_actuators[a]] = ids.home_qpos[a]
            d.ctrl[ids.gripper_actuators[a]] = 0.04
        d.ctrl[ids.bin_gate_actuators] = 0
        for i, q in enumerate(ids.cube_qpos):
            d.qpos[q : q + 7] = (3.0 + i * 0.1, 0, -1, 1, 0, 0, 0)
            self.model.body_gravcomp[ids.cube_bodies[i]] = 1
            self.model.geom_contype[ids.cube_geoms[i]] = 0
            self.model.geom_conaffinity[ids.cube_geoms[i]] = 0
        mujoco.mj_forward(self.model, d)
        self.ik.reset(d)
        if self.vacuum is not None:
            self.vacuum.reset()
        if self.sensor is not None:
            self.sensor.reset(self.seed + 1729)
        if self.wrist is not None:
            self.wrist.reset(self.seed + 7919)
        self._spawn()
        self._sense()
        # Native users may reset then execute WAIT/heuristic without ever
        # touching the scheduler adapter. Never retain prior-episode advisory
        # motion or soft job plans until a later lazy planner access.
        self.future_preposition_goals = np.full((self.num_arms, 3), np.nan)
        self.future_preposition_yaws = np.full(self.num_arms, np.nan)
        future_planner = getattr(self, "_future_planner", None)
        if future_planner is not None:
            future_planner.reset()
        return self.observe()

    def _spawn(self):
        c, d = self.config, self.data
        if c.total_objects is not None and self.spawned >= c.total_objects:
            return
        if self.supply_source is not None:
            self._spawn_multilane()
            return
        spatial_supply = c.spatial_supply
        if spatial_supply:
            if self.belt_sample.distance + 1e-9 < self.next_arrival_distance:
                return
        elif d.time + 1e-9 < self.next_arrival:
            return
        free = self._free_source_slots()
        if not free:
            self.supply_blocked_steps += 1
            return
        yaw = float(self.rng.uniform(-c.yaw_range, c.yaw_range))
        lateral = float(self.rng.uniform(-c.lateral_range, c.lateral_range))
        kind = int(self.rng.integers(2))
        self._admit_object(free[0], yaw, lateral, kind)
        if spatial_supply:
            self.next_arrival_distance = self.belt_sample.distance + max(
                0.12, c.spacing + self.rng.uniform(-c.spacing_jitter, c.spacing_jitter)
            )
        else:
            interval = max(
                0.12 / c.belt_speed,
                c.spacing / c.belt_speed
                + self.rng.uniform(-c.arrival_jitter, c.arrival_jitter),
            )
            self.next_arrival = d.time + interval

    def _free_source_slots(self):
        count = self.config.n_objects
        return [
            (self.next_pool_slot + j) % count
            for j in range(count)
            if self.states[(self.next_pool_slot + j) % count] == State.PENDING
        ]

    def _inlet_obstruction(self, slot, yaw, lateral):
        """Conservative birth-only clearance; never used by the controller.

        Existing objects use their current free-joint pose, including roll and
        pitch. Robot collision-geometry bounding spheres are conservative, so
        nearby hands can delay admission but cannot be intersected at birth.
        """
        c, d = self.config, self.data
        spec = self.ids.object_specs[slot]
        centre = np.array(
            [
                -0.98 if self.num_arms == 4 else -0.88,
                lateral,
                c.belt_top + spec.half_height + 0.001,
            ]
        )
        cosine, sine = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
        half = np.abs(rotation) @ spec.half_sizes
        margin = 0.003
        if (
            abs(centre[0]) + half[0] + margin >= c.belt_half_length
            or abs(centre[1]) + half[1] + margin >= c.belt_half_width
        ):
            return "edge"
        for other in np.flatnonzero(self.states != State.PENDING):
            q = self.ids.cube_qpos[other]
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, d.qpos[q + 3 : q + 7])
            other_half = (
                np.abs(matrix.reshape(3, 3)) @ self.ids.object_specs[other].half_sizes
            )
            if np.all(np.abs(centre - d.qpos[q : q + 3]) < half + other_half + margin):
                return "object"
            # Enforce planar *edge* clearance across both source lanes. Only
            # workpieces still at the belt surface constrain feed spacing;
            # carried pieces above it and pieces inside bins do not. Existing
            # 3D birth-collision checks above still cover every active body.
            other_centre = d.qpos[q : q + 3]
            on_belt = (
                abs(other_centre[2] - other_half[2] - c.belt_top) <= 0.015
                and abs(other_centre[0]) <= c.belt_half_length
                and abs(other_centre[1]) <= c.belt_half_width
            )
            if c.min_object_gap > 0 and on_belt:
                separation = np.maximum(
                    np.abs(centre[:2] - other_centre[:2]) - half[:2] - other_half[:2],
                    0.0,
                )
                # Rotated world AABBs provide a conservative lower bound on
                # actual footprint clearance, including diagonal neighbours.
                if np.linalg.norm(separation) + 1e-9 < c.min_object_gap + margin:
                    return "gap"
        robot = np.flatnonzero(
            (self.ids.robot_geom_arm >= 0)
            & ((self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0))
        )
        # mj_step exposes post-integration qpos while geom_xpos can still be
        # from the previous forward pass. Refresh FK only, without stepping or
        # changing any free-joint pose, before checking a moving hand.
        mujoco.mj_kinematics(self.model, d)
        distances = np.linalg.norm(d.geom_xpos[robot] - centre, axis=1)
        if np.any(
            distances < self.model.geom_rbound[robot] + spec.bounding_radius + margin
        ):
            return "robot"
        return None

    def _spawn_multilane(self):
        source, c = self.supply_source, self.config
        distance = float(self.belt_sample.distance)
        source.advance(distance)
        order = sorted(
            (lane for lane in range(2) if source.pending[lane]),
            key=lambda lane: (source.pending[lane][0].distance, lane),
        )
        blocked = False
        for lane in order:
            if c.total_objects is not None and self.spawned >= c.total_objects:
                break
            free = self._free_source_slots()
            if not free:
                self.supply_inlet_blocked_steps["pool"] += 1
                blocked = True
                continue
            arrival = source.pending[lane][0]
            reason = self._inlet_obstruction(free[0], arrival.yaw, arrival.lateral)
            if reason is not None:
                self.supply_inlet_blocked_steps[reason] += 1
                blocked = True
                continue
            self._admit_object(
                free[0], arrival.yaw, arrival.lateral, arrival.kind, lane=lane
            )
            source.accepted(lane)
            wait = max(0.0, distance - arrival.distance)
            self.supply_wait_distance_sum += wait
            self.supply_max_wait_distance = max(self.supply_max_wait_distance, wait)
        self.supply_blocked_steps += int(blocked)
        self.next_arrival_distance = min(
            queue[0].distance if queue else source.next_distance[lane]
            for lane, queue in enumerate(source.pending)
        )

    def _admit_object(self, i, yaw, lateral, kind, *, lane=0):
        c, d = self.config, self.data
        self.next_pool_slot = (i + 1) % c.n_objects
        q, v = self.ids.cube_qpos[i], self.ids.cube_dofs[i]
        spec = self.ids.object_specs[i]
        self.yaws[i] = yaw
        self.lateral[i] = lateral
        self.types[i] = kind
        self.serial[i] = self.spawned
        self.spawned += 1
        self.generations[i] += 1
        self.owner[i] = -1
        self.cube_settle[i] = 0.0
        self.cycle_times[i] = np.nan
        d.qpos[q : q + 7] = (
            -0.98 if self.num_arms == 4 else -0.88,
            lateral,
            c.belt_top + spec.half_height + 0.001,
            math.cos(yaw / 2),
            0,
            0,
            math.sin(yaw / 2),
        )
        d.qvel[v : v + 6] = (self.belt_sample.speed, 0, 0, 0, 0, 0)
        self.model.body_gravcomp[self.ids.cube_bodies[i]] = 0
        self.model.geom_rgba[self.ids.cube_geoms[i]] = (
            (0.87, 0.13, 0.08, 1) if self.types[i] == 0 else (0.08, 0.28, 0.92, 1)
        )
        self.model.geom_contype[self.ids.cube_geoms[i]] = 1
        self.model.geom_conaffinity[self.ids.cube_geoms[i]] = 1
        self.states[i] = State.BELT
        self.arrivals.append(float(d.time))
        self.arrival_distances.append(float(self.belt_sample.distance))
        self.spawn_yaw_log.append(yaw)
        self.spawn_position_log.append(d.qpos[q : q + 3].tolist())
        self.spawn_lane_log.append(lane)
        self.shape_spawned[spec.shape] += 1
        self.material_spawned[spec.stiffness_label] += 1
        self.max_active_objects = max(
            self.max_active_objects, int(np.count_nonzero(self.states != State.PENDING))
        )
        self.events.append(
            {
                "time": float(d.time),
                "event": "spawn",
                "object": int(self.serial[i]),
                "pool_slot": i,
                "shape": spec.shape,
                "lane": lane,
                "position": d.qpos[q : q + 3].tolist(),
            }
        )
        mujoco.mj_forward(self.model, d)

    def _recycle(self):
        # This external collection service only retires resolved objects below
        # a hopper outlet or outside the manipulation volume; never a live pick.
        changed = False
        candidates = self.states >= State.SORTED
        if self.config.deep_bins:
            candidates &= self.states != State.SORTED
        for i in np.flatnonzero(candidates).tolist():
            state = self.states[i]
            if state < State.SORTED or i in self.truth_target:
                continue
            if self.config.deep_bins and state == State.SORTED:
                # A deep fixed bin stores every completed workpiece physically.
                # Capacity is never manufactured by retiring stored bodies.
                continue
            p = self.cube_position(i)
            color = self.types[i]
            center = self.ids.bin_centers[color]
            in_receiver = (
                abs(p[0] - center[0]) < self.ids.bin_half_sizes[0] + 0.015
                and abs(p[1] - center[1]) < self.ids.bin_half_sizes[1] + 0.015
                and p[2] < self.ids.bin_outlet_z
            )
            if in_receiver:
                # A tilted tall object can have its centre past the outlet
                # while still intersecting the sliding floor. The external
                # collector waits until the whole conservative body bound has
                # cleared the gate underside, independently of controller state.
                rotation = self.data.xmat[self.ids.cube_bodies[i]].reshape(3, 3)
                top = p[2] + np.dot(
                    np.abs(rotation[2]), self.ids.object_specs[i].half_sizes
                )
                in_receiver = bool(top < self.ids.bin_floor_z - 0.013)
            outside = (
                p[0] > self.config.belt_half_length + 0.02 or p[2] < 0.06
            ) and state in (State.MISSED, State.DROPPED)
            if self.config.deep_bins:
                inside_any_bin = any(
                    np.all(np.abs(p[:2] - centre[:2]) < self.ids.bin_half_sizes)
                    and p[2] >= self.ids.bin_floor_z - 0.03
                    for centre in self.ids.bin_centers
                )
                # A misplaced piece remains in the wrong bin as a physical
                # obstruction and a failed result; it is not recycled away.
                in_receiver = False
                outside = outside and not inside_any_bin
            if not (in_receiver or outside):
                continue
            if state == State.SORTED and not in_receiver:
                continue
            self.archived[int(state)] += 1
            if state == State.SORTED:
                a = self.owner[i]
                if a >= 0:
                    self.archived_arm_color[a, color] += 1
                self.archived_cycles_sum += (
                    float(self.cycle_times[i])
                    if np.isfinite(self.cycle_times[i])
                    else 0.0
                )
            self.object_records.append(
                {
                    "id": int(self.serial[i]),
                    "state": State(int(state)).name,
                    "shape": self.ids.object_specs[i].shape,
                    "mass": self.ids.object_specs[i].mass,
                    "time": float(self.data.time),
                }
            )
            self.vision_id_matches.pop(int(self.serial[i]), None)
            q, dof = self.ids.cube_qpos[i], self.ids.cube_dofs[i]
            self.data.qpos[q : q + 7] = (3.0 + i * 0.1, 0, -1, 1, 0, 0, 0)
            self.data.qvel[dof : dof + 6] = 0
            self.model.geom_contype[self.ids.cube_geoms[i]] = 0
            self.model.geom_conaffinity[self.ids.cube_geoms[i]] = 0
            self.model.body_gravcomp[self.ids.cube_bodies[i]] = 1
            self.states[i] = State.PENDING
            self.serial[i] = -1
            self.recycled += 1
            changed = True
        if changed:
            mujoco.mj_forward(self.model, self.data)

    def cube_position(self, i):
        """Evaluator/oracle access only; never used by visual control."""
        return self.data.xpos[self.ids.cube_bodies[i]].copy()

    def cube_velocity(self, i):
        return self.data.qvel[self.ids.cube_dofs[i] : self.ids.cube_dofs[i] + 3].copy()

    def _sense(self):
        if self.sensor is not None:
            incoming = self.sensor.update(self.data, self.data.time)
        else:
            # Explicit privileged-state ablation, with the identical control API.
            incoming = []
            for i in np.flatnonzero(
                (self.states == State.BELT) | (self.states == State.RESERVED)
            ).tolist():
                p = self.cube_position(i)
                if (
                    self.states[i] not in (State.BELT, State.RESERVED)
                    or abs(p[1]) > self.config.belt_half_width
                    or p[2] > self.config.belt_top + 0.10
                ):
                    continue
                rotation = self.data.xmat[self.ids.cube_bodies[i]].reshape(3, 3)
                yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                slot = self.track_slots.get(int(self.serial[i]))
                previous = self.tracks[slot] if slot is not None else None
                reference = previous.yaw if previous is not None else 0.0
                candidate = min(
                    self._oracle_outline_candidates[i],
                    key=lambda item: abs(
                        symmetric_angle(yaw + item[0] - reference, np.pi)
                    ),
                )
                p[:2] += rotation[:2, :2] @ np.asarray(candidate[1:3])
                yaw = symmetric_angle(yaw + candidate[0], np.pi)
                incoming.append(
                    Track(
                        int(self.serial[i]),
                        p,
                        self.cube_velocity(i),
                        yaw,
                        int(self.types[i]),
                        1.0,
                        float(self.data.time),
                        float(self.data.time),
                        10,
                        True,
                        size=np.r_[candidate[3:5], self.ids.object_specs[i].height],
                        shape=self.ids.object_specs[i].shape,
                        yaw_period=np.pi,
                    )
                )
        self._ingest_tracks(incoming)
        if self.wrist is not None:
            # Local cameras never create tasks or modify the overhead tracker.
            # Stop acquiring once fingers close: the grasp point is frozen then.
            self.wrist.set_targets(
                [
                    replace(
                        self.tracks[i],
                        yaw=float(self.grasp_yaw[a]),
                        size=self.grasp_sizes[a].copy(),
                    )
                    if i >= 0 and self.phase[a] in ("approach", "descend")
                    else None
                    for a, i in enumerate(self.target)
                ]
            )
            self.wrist.update(self.data, self.data.time)

    def _ingest_tracks(self, incoming):
        self.retired_tracks = {
            k: v for k, v in self.retired_tracks.items() if self.data.time - v < 3.0
        }
        for track in incoming:
            if track.track_id in self.retired_tracks:
                continue
            slot = self.track_slots.get(track.track_id)
            if slot is None:
                free = [
                    i
                    for i in range(10)
                    if i not in self.target
                    and (
                        self.belief[i] in (0, 4)
                        or (
                            self.belief[i] == 1
                            and self.data.time - self.tracks[i].last_seen > 2.0
                        )
                    )
                ]
                if not free:
                    continue
                slot = free[0]
                previous = self.tracks[slot]
                if previous is not None:
                    self.track_slots.pop(previous.track_id, None)
                    self.retired_tracks[previous.track_id] = self.data.time
                self.track_slots[track.track_id] = slot
                self.belief[slot] = 1
                self.retry_after[:, slot] = 0
            self.tracks[slot] = track

    def _position(self, i):
        return (
            self.tracks[i].predict(self.data.time)
            if self.tracks[i] is not None
            else np.zeros(3)
        )

    def _belt_velocity(self):
        """Control-facing belt estimate; never expose the simulated drive profile.

        Vision obtains this from visible RGB-D tracks. The separate oracle
        ablation may use the physical drive as an explicitly privileged sensor.
        """
        if self.config.strategy != "adaptive":
            return self.config.belt_speed
        if self.sensor is not None:
            return float(self.sensor.belt_velocity(self.data.time)[0])
        return self.belt_sample.speed

    def _grasp_offset(self, track):
        return (
            track.half_height + 0.039
            if self.suction
            else max(self.config.grasp_offset, track.half_height - 0.030)
        )

    @property
    def approach_clearance(self):
        return 0.08 if self.suction else 0.13

    @property
    def descent_time(self):
        return 0.30 if self.suction else 0.45

    @property
    def closure_timeout(self):
        return 0.18 if self.suction else self.config.close_time

    @property
    def lift_height(self):
        return 0.40 if self.suction else 0.38

    @property
    def transport_height(self):
        return self.ids.bin_rim_z + (0.16 if self.suction else 0.07)

    @property
    def deposit_height(self):
        if self.config.direct_bin_release:
            # Deep fixed bins do not need a separate 20 mm down/up stroke.
            # Keep the original settled place, sensed release and certified
            # retreat states; this only changes their shared height target.
            return self.transport_height
        return self.ids.bin_rim_z + (0.12 if self.suction else 0.05)

    def _grasp_position(self, arm, i):
        """Smooth a bounded, target-specific visual correction before planning.

        Neither global association, vertical clearance nor grasp yaw is changed.
        Stale/ambiguous local observations smoothly hand control back to overhead.
        Every resulting goal still goes through the existing collision planner.
        """
        position = self._position(i)
        if self.wrist is None or self.config.wrist_camera_mode != "refine":
            return position
        desired = np.zeros(2)
        track = self.tracks[i]
        # A clear overhead frame is already an accurate, stable global estimate.
        # Local views help bridge occlusion, not replace every fresh estimate.
        # Allow normal delivery latency and a capture interval plus jitter.
        freshness_budget = self.config.camera_latency + 1.5 / self.config.camera_hz
        observation = (
            self.wrist.refinement(arm, track.track_id, self.data.time)
            if self.data.time - track.last_seen > freshness_budget
            else None
        )
        limit = self.config.wrist_max_correction
        if observation is not None:
            delta = observation.predict(self.data.time)[:2] - position[:2]
            distance = np.linalg.norm(delta)
            # A changed overhead estimate can invalidate an earlier match.
            if distance <= 2 * limit:
                desired = delta * min(1.0, limit / max(distance, 1e-12))
                if observation.timestamp != self.wrist_last_used_timestamp[arm]:
                    self.wrist_used_frames[arm] += 1
                    self.wrist_last_used_timestamp[arm] = observation.timestamp
        step = (desired - self.wrist_offsets[arm]) * (
            1 - math.exp(-self.config.physics_dt / 0.12)
        )
        step *= min(
            1.0,
            0.025 * self.config.physics_dt / max(np.linalg.norm(step), 1e-12),
        )
        self.wrist_offsets[arm] += step
        magnitude = np.linalg.norm(self.wrist_offsets[arm])
        if magnitude > 0.0001:
            self.wrist_control_seconds[arm] += self.config.physics_dt
        self.wrist_max_applied[arm] = max(self.wrist_max_applied[arm], magnitude)
        position[:2] += self.wrist_offsets[arm]
        return position

    def _set_phase(self, arm, phase):
        previous = self.phase[arm]
        duration = float(self.data.time - self.phase_start[arm])
        if duration > 0:
            self.phase_duration_samples[arm].setdefault(previous, deque(maxlen=128)).append(duration)
        self.max_phase_duration[previous] = max(
            self.max_phase_duration[previous],
            duration,
        )
        self.phase[arm] = phase
        if phase != "idle":
            self._preposition_live_goal[arm] = None
            self.preposition_active[arm] = False
        if phase != "return":
            self.planned_return_active[arm] = False
        self.phase_start[arm] = self.data.time
        self.routes[arm] = None
        self.escape_routes[arm] = False
        self.route_index[arm] = 0
        self.route_retry_time[arm] = self.data.time
        if phase == "release":
            self.release_clear_since[arm] = np.nan
            if self.config.responsive_release:
                # Capture encoders only. Keep the actuator target and its
                # original slew limiter untouched throughout opening.
                aperture = float(np.sum(self.data.qpos[self.ids.finger_qpos[arm]]))
                self.release_start_aperture[arm] = (
                    aperture if math.isfinite(aperture) and 0.004 <= aperture <= 0.081 else np.nan
                )
        elif phase in ("approach", "recover", "idle"):
            self.release_start_aperture[arm] = np.nan
        if phase == "close":
            self.grasp_contact_since[arm] = np.nan

    def _grasp_angles(self, arm, track):
        """Reachable wrist preferences whose measured footprint fits the hand."""
        nominal = np.pi / 2 if arm % 2 == 0 else -np.pi / 2
        raw = (
            [nominal]
            if track.yaw_period == 0 or self.suction
            else [track.yaw + k * track.yaw_period for k in range(-3, 4)]
        )
        if not self.suction:
            # Finger contact may rotate a prism into a stable antipodal grasp.
            # It need not reproduce the noisy top-face minimum-width angle if
            # the complete observed rectangle fits inside the 80 mm opening.
            offsets = (0, 10, -10, 20, -20, 30, -30, 45, -45)
            raw = [nominal + np.deg2rad(offset) for offset in offsets] + [
                nominal + np.pi,
                *raw,
            ]
        angles = []
        for yaw in raw:
            yaw = float((yaw + np.pi) % (2 * np.pi) - np.pi)
            if not self.suction and self._grasp_size_at_yaw(track, yaw)[1] > 0.072:
                continue
            if not any(
                abs((yaw - old + np.pi) % (2 * np.pi) - np.pi) < 1e-7 for old in angles
            ):
                angles.append(yaw)
        return sorted(
            angles,
            key=lambda yaw: abs(
                (yaw - self.tool_yaw[arm] + np.pi) % (2 * np.pi) - np.pi
            ),
        )

    @staticmethod
    def _grasp_size_at_yaw(track, yaw):
        """Observed rectangle projected into the selected jaw frame."""
        delta = float(track.yaw - yaw)
        cs, sn = abs(math.cos(delta)), abs(math.sin(delta))
        size = np.asarray(track.size).copy()
        size[:2] = np.array([[cs, sn], [sn, cs]]) @ size[:2]
        return size

    def _finger_neighborhood_clear(self, arm, i, yaw):
        """Check observed neighbours around the open fingers before a low grasp.

        Only visual tracks are used. The 60 x 130 mm rectangle bounds the open
        finger pair in the grasp plane; neighbouring footprints use conservative
        circumscribed radii. This supplements, rather than replaces, the robot
        and fixed-scene path checks. It is not an unseen-object guarantee.
        """
        if self.suction or getattr(self.config, "supply_lanes", 1) == 1:
            return True
        target = self.tracks[i]
        if target is None:
            return False
        now = float(self.data.time)
        horizon = self.descent_time + self.closure_timeout
        times = np.linspace(now, now + horizon, 5)
        target_positions = np.array([target.predict(t)[:2] for t in times])
        cs, sn = math.cos(yaw), math.sin(yaw)
        axes = np.array([[cs, -sn], [sn, cs]])
        for other, track in enumerate(self.tracks):
            if (
                other == i
                or track is None
                or self.belief[other] not in (1, 2)
                or now - track.last_seen > self.config.grasp_coast_time
            ):
                continue
            positions = np.array([track.predict(t)[:2] for t in times])
            relative = (positions - target_positions) @ axes
            radius = 0.5 * float(np.linalg.norm(track.size[:2]))
            # Cover linear motion between samples plus observed position error.
            margin = 0.008 + min(0.012, max(0.0, float(track.uncertainty)))
            margin += 0.5 * float(
                np.max(np.linalg.norm(np.diff(relative, axis=0), axis=1))
            )
            outside = np.maximum(np.abs(relative) - [0.030, 0.065], 0.0)
            if np.min(np.linalg.norm(outside, axis=1)) <= radius + margin:
                return False
        return True

    def _intercept_motion_time(self, start, goal, velocity):
        """Conservative acceleration-limited travel estimate for dispatch.

        A constant-speed interception omits the start/brake transients. The
        scalar triangular/trapezoidal profile models both while retaining the
        existing speed reserve for joint and orientation tracking.
        """
        speed = 0.85 * self.config.control_speed
        acceleration = self.ik.max_cartesian_acceleration
        delta = np.asarray(goal) - np.asarray(start)
        velocity = np.asarray(velocity)
        quadratic = float(np.dot(velocity, velocity) - speed**2)
        linear = float(2 * np.dot(delta, velocity))
        constant = float(np.dot(delta, delta))
        discriminant = linear**2 - 4 * quadratic * constant
        if discriminant < 0 or quadratic >= -1e-8:
            return math.inf
        duration = max(0.0, (-linear - math.sqrt(discriminant)) / (2 * quadratic))
        for _ in range(8):
            distance = float(np.linalg.norm(delta + velocity * duration))
            updated = (
                2 * math.sqrt(distance / acceleration)
                if distance <= speed**2 / acceleration
                else distance / speed + speed / acceleration
            )
            if abs(updated - duration) < 1e-5:
                return updated
            duration = updated
        return duration

    def _returning_partner_has_urgent_work(self, arm, own_eta, own_deadline):
        """Leave a short opening for the returning hand's earlier deadline.

        An idle hand must not lock a station around a fresh object just before
        its partner returns to pick an older one. Use only visual tracks and
        measured robot poses; physical station exclusion remains unchanged.
        """
        # The upstream station has a downstream recovery opportunity; preserve
        # its throughput instead of waiting for an older opposite-color part.
        if self.num_arms != 4 or arm < 2:
            return False
        partner = arm ^ 1
        if self.phase[partner] not in ("retreat", "return"):
            return False
        color = partner % 2
        if self.bin_service[color] != "closed" or self.bin_drain_requested[color]:
            return False
        per_zone = self.config.bin_capacity // 2
        start = (partner // 2) * per_zone
        if np.all(self.bin_occupied[color, start : start + per_zone]):
            return False
        now = float(self.data.time)
        home = (
            self.return_goals[partner]
            if self.config.direct_retask and self.phase[partner] == "return"
            else self._normal_return_goal(partner)
            if self.config.direct_retask
            else self.ids.home_ee[partner]
        )
        ee = self.data.site_xpos[self.ids.ee_sites[partner]]
        speed = 0.85 * self.config.control_speed
        ready_delay = np.linalg.norm(home - ee) / speed + 0.25
        if self.phase[partner] == "retreat":
            ready_delay += 0.15
        low, high = self.windows[partner]
        for other, track in enumerate(self.tracks):
            if (
                track is None
                or self.belief[other] != 1
                or track.kind != color
                or track.confidence < 0.35
                or now - track.last_seen > self.config.track_max_age
                or now + ready_delay < self.retry_after[partner, other]
            ):
                continue
            p = track.predict(now)
            if not low <= p[0] <= high or abs(p[1]) > 0.11:
                continue
            velocity = np.r_[np.clip(track.predict_velocity(now)[:2], -0.6, 0.6), 0.0]
            deadline = (self.completion_window_high(partner, p) - p[0]) / max(velocity[0], 0.04)
            above = track.predict(now + ready_delay).copy()
            above[2] = (
                self.config.belt_top
                + track.half_height
                + self._grasp_offset(track)
                + self.approach_clearance
            )
            intercept = self._intercept_motion_time(home, above, velocity)
            eta = ready_delay + intercept + self.descent_time + self.closure_timeout
            slack = deadline - eta
            if not 0.25 < slack < own_deadline - own_eta - 0.35:
                continue
            angles = self._grasp_angles(partner, track)
            if any(
                self._finger_neighborhood_clear(partner, other, yaw) for yaw in angles
            ):
                return True
        return False

    def completion_window_high(self, arm, position):
        """Completion deadline bound, distinct from where a task may start.

        The upstream near-side extension uses its existing x=0 envelope.
        The optional downstream extension uses x=.90 minus a 20 mm reserve;
        runtime descent/lift previews remain authoritative for each wrist
        branch. Raw admission windows and physical hard stops stay unchanged.
        This is a pure visual/calibration helper for scheduling features too.
        """
        side = -1 if arm % 2 == 0 else 1
        if (
            self.config.extended_pick_deadline
            and self.num_arms == 4
            and not self.suction
            and arm < 2
            and 0.045 <= side * float(position[1]) <= 0.080
        ):
            return -0.02
        if (
            getattr(self.config, "downstream_completion", False)
            and self.num_arms == 4
            and not self.suction
            and arm >= 2
            and 0.045 <= side * float(position[1]) <= 0.080
        ):
            return 0.88
        return float(self.windows[arm, 1])

    def _candidate(self, arm, i):
        track = self.tracks[i]
        if (
            self.phase[arm] != "idle"
            or self.belief[i] != 1
            or track is None
            or self.data.time < self.retry_after[arm, i]
            or self.data.time - track.last_seen > self.config.track_max_age
        ):
            return False, 0.0, 0.0
        if (
            track.confidence < 0.35
            or self.transfer_owner >= 0
            or np.all(self.bin_occupied[track.kind])
            or self.bin_service[track.kind] != "closed"
            or self.bin_drain_requested[track.kind]
            or any(
                self.phase[b] in ("close", "lift", "depart", "transfer_wait")
                and self.target[b] >= 0
                and self.drop_kind[self.target[b]] != b % 2
                for b in range(self.num_arms)
            )
        ):
            return False, 0.0, 0.0
        p = self._position(i)
        low, high = self.windows[arm]
        if not low <= p[0] <= high or abs(p[1]) > 0.11:
            return False, 0.0, 0.0
        if not self.suction and not self._grasp_angles(arm, track):
            return False, 0.0, 0.0
        if not self._finger_neighborhood_clear(arm, i, track.yaw):
            if not any(
                self._finger_neighborhood_clear(arm, i, yaw)
                for yaw in self._grasp_angles(arm, track)
            ):
                return False, 0.0, 0.0
        if self.num_arms == 4:
            # A station has two opposing hands. Reserve its belt approach
            # until the current hand has lifted and moved out of the lane.
            # The downstream station can still work concurrently.
            if track.kind != arm % 2:
                return False, 0.0, 0.0
            per_zone = self.config.bin_capacity // 2
            zone = slice((arm // 2) * per_zone, (arm // 2 + 1) * per_zone)
            if np.all(self.bin_occupied[track.kind, zone]):
                return False, 0.0, 0.0
            partner = arm ^ 1
            partner_ee = self.data.site_xpos[self.ids.ee_sites[partner]]
            if self.phase[partner] in (
                "approach",
                "descend",
                "close",
                "lift",
                "reorient",
            ) or (self.phase[partner] != "idle" and abs(partner_ee[1]) < 0.23):
                return False, 0.0, 0.0
        ee = self.data.site_xpos[self.ids.ee_sites[arm]]
        eta = np.linalg.norm(p[:2] - ee[:2]) / 0.32 + 0.65 + self.config.close_time
        if self.config.strategy == "adaptive":
            # Solve a moving rendezvous rather than time a trip to where the
            # object used to be. Reserve speed for joint/servo transients and
            # time for descent and finger closure; a missed deadline is never
            # fixed by issuing an infeasible faster arm command.
            above = p.copy()
            above[2] = (
                self.config.belt_top
                + track.half_height
                + self._grasp_offset(track)
                + self.approach_clearance
            )
            velocity = np.r_[
                np.clip(track.predict_velocity(self.data.time)[:2], -0.6, 0.6), 0.0
            ]
            intercept = self._intercept_motion_time(ee, above, velocity)
            if not math.isfinite(intercept):
                return False, 0.0, 0.0
            eta = max(0.0, intercept) + self.descent_time + self.closure_timeout
        speed = (
            track.predict_velocity(self.data.time)[0]
            if self.config.strategy == "adaptive"
            else track.velocity[0]
        )
        completion_high = self.completion_window_high(arm, p)
        if completion_high > high:
            # Do not borrow near-side reach for a track predicted to leave its
            # calibrated lateral band before grasp and initial lift complete.
            completion_high = min(
                completion_high,
                *(self.completion_window_high(arm, track.predict(self.data.time + offset))
                  for offset in np.linspace(0.0, eta + 0.20, 5)),
            )
        deadline = (completion_high - p[0]) / max(speed, 0.04)
        # This is a scheduling preference, not a physical feasibility bound.
        # A learned high-level policy may choose when to yield for itself.
        if self.config.use_dispatch_preferences and self._returning_partner_has_urgent_work(
            arm, eta, deadline
        ):
            return False, float(eta), float(deadline)
        return (
            bool(deadline > eta + (0.30 if self.suction else 0.25)),
            float(eta),
            float(deadline),
        )

    def _observe_multi(self):
        """Joint action mask for two coordinated, sequential picking stations.

        The pretrained two-arm neural scheduler has a different observation
        contract; four-arm mode deliberately uses the model-based dispatcher.
        """
        valid = np.zeros((self.num_arms, 11), dtype=bool)
        valid[:, WAIT] = True
        scores = np.zeros((self.num_arms, 11))
        positions = np.array([self._position(i) for i in range(10)])
        for arm in range(self.num_arms):
            for i in range(10):
                ok, eta, deadline = self._candidate(arm, i)
                if not ok:
                    continue
                # Reservations protect both stations at their overlap. Do
                # not assume the other arm will have left before rendezvous.
                future_x = (
                    positions[i, 0]
                    + self.tracks[i].predict_velocity(self.data.time)[0] * eta
                )
                for other in range(self.num_arms):
                    if other == arm or other // 2 == arm // 2 or self.target[other] < 0:
                        continue
                    other_ee = self.data.site_xpos[self.ids.ee_sites[other]]
                    if (
                        self.phase[other]
                        in ("approach", "descend", "close", "lift", "reorient")
                        or abs(other_ee[1]) < 0.23
                    ):
                        if abs(future_x - other_ee[0]) < 0.32:
                            ok = False
                valid[arm, i] = ok
                scores[arm, i] = 10 + 2 / max(deadline - eta, 0.2) - 0.35 * eta
        choices = self.action_choices
        mask = np.all(valid[np.arange(self.num_arms)[None, :], choices], axis=1)
        for first in range(self.num_arms):
            for second in range(first + 1, self.num_arms):
                both = (choices[:, first] < WAIT) & (choices[:, second] < WAIT)
                if first // 2 == second // 2:
                    mask[both] = False
                    continue
                rows = np.flatnonzero(mask & both)
                one, two = choices[rows, first], choices[rows, second]
                mask[
                    rows[
                        (one == two)
                        | (np.abs(positions[one, 0] - positions[two, 0]) < 0.32)
                    ]
                ] = False
                # Both arms of one color share a finite bin, including slots
                # that have been reserved but not yet physically filled.
                if (
                    first % 2 == second % 2
                    and np.count_nonzero(~self.bin_occupied[first % 2]) < 2
                ):
                    mask[both] = False
        mask[self.wait_action] = True
        self._last_mask = mask
        self._multi_scores = scores
        return {"per_arm_scores": scores, "mask": mask}

    def _heuristic_multi(self):
        observation = self._observe_multi()
        totals = self._multi_scores[
            np.arange(self.num_arms)[None, :], self.action_choices
        ].sum(axis=1)
        totals[~observation["mask"]] = -np.inf
        return int(np.argmax(totals))

    def _descent_lead(self, arm, track, velocity, remaining, position=None):
        """Correct measured tracking error before certifying the descent path.

        Executing the checked joint path couples horizontal and vertical
        progress. A fixed remaining-height/vertical-speed intercept can then
        place an open pad ahead of the part while the opposite extreme (zero
        correction) lags it. Use the observed relative position to correct the
        endpoint in either direction; retain the servo's existing velocity
        feedforward, including its 45 ms position compensation. This bounded
        correction still goes through the ordinary complete robot path check.
        """
        lead = np.asarray(velocity, dtype=float)[:2] * remaining
        if not getattr(self.config, "guarded_descent", False):
            return lead
        observed = track.predict(self.data.time) if position is None else np.asarray(position)
        error = self.data.site_xpos[self.ids.ee_sites[arm], :2] - observed[:2]
        correction = -2.0 * error
        return correction * min(1.0, 0.04 / max(np.linalg.norm(correction), 1e-12))

    def _descent_alignment_ready(self, arm, track, position):
        if not getattr(self.config, "guarded_descent", False):
            return True
        yaw = float(self.grasp_yaw[arm])
        closing_axis = np.array([-math.sin(yaw), math.cos(yaw)])
        error = self.data.site_xpos[self.ids.ee_sites[arm], :2] - np.asarray(position)[:2]
        opening = float(np.sum(self.data.qpos[self.ids.finger_qpos[arm]]))
        width = float(self._grasp_size_at_yaw(track, yaw)[1])
        return bool(abs(float(np.dot(error, closing_axis))) + 0.002 < 0.5 * (opening - width))

    def _descent_guard_reason(self, arm, track, position, elapsed):
        """Reject disturbed open-hand entries using observable signals only."""
        if not getattr(self.config, "guarded_descent", False) or self.suction:
            return None
        if np.max(self.data.sensordata[self.ids.finger_touch_adr[arm]]) > 0.5:
            return "unexpected_open_finger_contact"
        if position[2] - (self.config.belt_top + track.half_height) > max(0.006, 2 * track.uncertainty):
            return "target_height_changed"
        if elapsed > max(1.0, 2.5 * self.descent_time):
            return "descent_time_limit"
        return None

    def _descent_feasible(self, arm, track, position):
        """Reserve downstream travel through descent, closure and lift-off.

        This uses only the observed track and a calibrated workspace envelope.
        A collision-free pose now does not imply a moving grasp can finish in
        reach. Recheck at descent entry after approach/IK delays have elapsed.
        """
        if self.config.strategy != "adaptive":
            return True
        # At most .20 s following plus .20 s symmetric velocity decay (.10 s
        # effective travel). Reserve the entire .30*v displacement.
        horizon = (
            self.descent_time + self.closure_timeout + (0.30 if self.suction else 0.20)
        )
        velocity = max(0.0, float(track.predict_velocity(self.data.time)[0]))
        acceleration = max(0.0, float(track.acceleration[0]))
        remaining_acceleration_horizon = min(
            horizon, max(0.0, 0.25 - (self.data.time - track.timestamp))
        )
        finish_x = (
            position[0]
            + velocity * horizon
            + acceleration
            * (
                remaining_acceleration_horizon * horizon
                - 0.5 * remaining_acceleration_horizon**2
            )
        )
        high = self.windows[arm, 1]
        side = -1 if arm % 2 == 0 else 1
        if (
            self.num_arms == 4
            and not self.suction
            and arm < 2
            and 0.045 <= side * float(position[1]) <= 0.080
        ):
            # Near-side FK/IK and the complete robot grasp-exit preview remain
            # feasible through x=0 across this calibrated lane band. Keep the
            # 20 mm reserve below, and leave admission/scoring/hard-stop windows
            # unchanged: this only helps an already assigned task finish after
            # an approach delay. Runtime scene/exit checks and cross-station
            # reservations still apply; far-side reach is not extended.
            high = 0.0
        if (
            getattr(self.config, "downstream_completion", False)
            and self.num_arms == 4
            and not self.suction
            and arm >= 2
            and 0.045 <= side * float(position[1]) <= 0.080
        ):
            # Only borrow completion room while the observed lateral forecast
            # stays in the calibrated near band. This coarse envelope is not
            # grasp permission: the full robot descent/closure/lift preview
            # below must still pass, including short-block/bin-wall cases.
            now = float(self.data.time)
            observed_y = float(track.predict(now)[1])
            stays_near = all(
                0.045 <= side * (
                    float(position[1]) + float(track.predict(now + offset)[1]) - observed_y
                ) <= 0.080
                for offset in np.linspace(0.0, horizon, 5)
            )
            if stays_near:
                high = 0.90
        return bool(finish_x <= high - 0.02)

    def observe(self):
        if self.num_arms == 4:
            return self._observe_multi()
        features = np.zeros((121, 22), dtype=np.float32)
        per_arm = np.zeros((2, 11, 10), dtype=np.float32)
        valid = np.zeros((2, 11), dtype=bool)
        valid[:, WAIT] = True
        for a in range(self.num_arms):
            per_arm[a, WAIT, 0] = 1
            per_arm[a, :, 1] = self.phase[a] == "idle"
            for i in range(10):
                ok, eta, deadline = self._candidate(a, i)
                valid[a, i] = ok
                track = self.tracks[i]
                if track is None:
                    continue
                p = self._position(i)
                per_arm[a, i, 2:] = (
                    np.clip(p[0], -2, 2),
                    np.clip(p[1], -1, 1),
                    min(deadline, 15) / 10,
                    min(eta, 10) / 5,
                    float(track.kind),
                    float(ok),
                    min(self.data.time - track.last_seen, 2) / 2,
                    track.confidence,
                )
        mask = (valid[0, :, None] & valid[1, None, :]).reshape(-1)
        for i in range(11):
            for j in range(11):
                k = i * 11 + j
                features[k, :10], features[k, 10:20] = per_arm[0, i], per_arm[1, j]
                if i < 10 and j < 10:
                    gap = abs(self._position(i)[0] - self._position(j)[0])
                    features[k, 20] = min(gap, 1.0)
                    if i == j or self._position(j)[0] - self._position(i)[0] < 0.28:
                        mask[k] = False
                    if self.tracks[i] is not None and self.tracks[j] is not None:
                        kind = self.tracks[i].kind
                        if (
                            kind == self.tracks[j].kind
                            and np.count_nonzero(~self.bin_occupied[kind]) < 2
                        ):
                            mask[k] = False
                for a, obj in ((0, i), (1, j)):
                    other = 1 - a
                    other_near = (
                        abs(self.data.site_xpos[self.ids.ee_sites[other], 1]) < 0.23
                    )
                    if (
                        obj < 10
                        and self.target[other] >= 0
                        and (
                            self.phase[other]
                            in ("approach", "descend", "close", "lift", "depart")
                            or other_near
                        )
                    ):
                        eta = self._candidate(a, obj)[1]
                        other_x = self.data.site_xpos[self.ids.ee_sites[other], 0]
                        own_speed = self.config.belt_speed
                        if self.config.strategy == "adaptive":
                            if self.tracks[obj] is not None:
                                own_speed = self.tracks[obj].predict_velocity(
                                    self.data.time
                                )[0]
                        gap = (other_x - (self._position(obj)[0] + own_speed * eta)) * (
                            1 if a == 0 else -1
                        )
                        if gap < 0.28:
                            mask[k] = False
                features[k, 21] = mask[k]
        mask[-1] = True
        glob = np.zeros(20, dtype=np.float32)
        glob[:2] = [PHASES.index(p) / (len(PHASES) - 1) for p in self.phase]
        glob[2:8] = self.data.site_xpos[self.ids.ee_sites].reshape(-1)
        glob[8:10] = self.arm_releases / max(int(self.arm_releases.sum()), 1)
        glob[10] = np.count_nonzero(self.belief == 1) / 10
        glob[11:13] = self.bin_slots / self.config.bin_capacity
        glob[13] = self._belt_velocity()
        glob[14:16] = self.bin_owner >= 0
        glob[16] = self.transfer_owner >= 0
        glob[17:19] = self.holding
        glob[19] = self.retries / max(int(self.arm_assignments.sum()), 1)
        self._last_mask = mask
        return {"features": features, "global": glob, "mask": mask}

    def heuristic_action(self):
        if self.num_arms == 4:
            return self._heuristic_multi()
        obs = self.observe()
        best, score = 120, -float("inf")
        for k in np.flatnonzero(obs["mask"]):
            s = 0.0
            for a, i in enumerate(divmod(int(k), 11)):
                if i == WAIT:
                    continue
                _, eta, deadline = self._candidate(a, i)
                # For this sparse stream, local-bin roles avoid unnecessary
                # crossings and corridor waits. The action API still permits
                # both colors per arm for learned or alternative dispatchers.
                if self.tracks[i].kind != a:
                    s -= 100.0
                s += (
                    10
                    + 2 / max(deadline - eta, 0.2)
                    - 0.35 * eta
                    - 0.15
                    * min(self.arm_assignments[a] - min(self.arm_assignments), 10)
                )
            if s > score:
                best, score = int(k), s
        return best

    def _deposit_position(self, arm, color, rank):
        """The existing calibrated deposit grid, independent of slot ordering."""
        result = self.ids.bin_centers[color] + np.array(
            [(-0.055, 0.0, 0.055)[rank % 3], (-0.035, 0.035)[(rank // 3) % 2], 0.0]
        )
        if self.num_arms == 4:
            local_rank = rank % (self.config.bin_capacity // 2)
            offsets = (-0.18, -0.09, 0.0, 0.09, 0.18) if self.config.deep_bins else (-0.09, 0.0, 0.09)
            result[:2] = self.ids.bin_centers[color, :2] + [
                self.ids.arm_base_positions[arm, 0] + offsets[local_rank % len(offsets)],
                (-0.045, 0.045)[(local_rank // len(offsets)) % 2],
            ]
        return result

    def _choose_drop_slot(self, arm, track_slot, free):
        """Minimize estimated travel within the current ten-position layer.

        Restricting selection to the earliest unfinished layer preserves the
        existing deposit spread: no location is reused until its entire layer
        has been reserved. With station contention, also prefer clearing the
        belt before the final position-servo settling tail. This reads visual
        motion and robot state only; execution still checks the entire route.
        """
        if not self.config.nearest_bin_slot:
            return int(free[0])
        per_zone = self.config.bin_capacity // 2
        local = free - (arm // 2) * per_zone
        layer = int(local.min()) // 10
        eligible = free[(local // 10) == layer]
        track = self.tracks[track_slot]
        p = self._position(track_slot).copy()
        velocity = np.r_[np.clip(track.predict_velocity(self.data.time)[:2], -0.6, 0.6), 0.0]
        above = p.copy()
        above[2] = self.config.belt_top + track.half_height + self._grasp_offset(track) + self.approach_clearance
        eta = self._intercept_motion_time(self.data.site_xpos[self.ids.ee_sites[arm]], above, velocity)
        if not math.isfinite(eta):
            eta = 0.0
        lifted = p + velocity * (eta + self.descent_time + self.closure_timeout + 0.20)
        lifted[0] = np.clip(lifted[0], *self.windows[arm])
        lifted[2] = self.lift_height
        partner_demand = False
        if getattr(self.config, "cooperative_drop_cost", False) and self.num_arms == 4:
            partner = arm ^ 1
            color = partner % 2
            zone = slice((partner // 2) * per_zone, (partner // 2 + 1) * per_zone)
            if (
                self.phase[partner] in ("idle", "retreat", "return")
                and self.bin_service[color] == "closed"
                and not self.bin_drain_requested[color]
                and not np.all(self.bin_occupied[color, zone])
            ):
                now = float(self.data.time)
                low, high = self.windows[partner]
                for i, pending in enumerate(self.tracks):
                    if (
                        pending is None
                        or self.belief[i] != 1
                        or pending.kind != color
                        or pending.confidence < 0.35
                        or now - pending.last_seen > self.config.track_max_age
                        or now < self.retry_after[partner, i]
                    ):
                        continue
                    observed = pending.predict(now)
                    if low <= observed[0] <= high and abs(observed[1]) <= 0.11:
                        partner_demand = True
                        break
        costs = []
        for rank in eligible:
            drop = self._deposit_position(arm, track.kind, int(rank))
            drop[2] = self.transport_height
            ready = self.ids.home_ee[arm].copy()
            if self.config.direct_retask:
                ready[0] = drop[0]
                if getattr(self.config, "cooperative_motion", False):
                    ready[1] = (-1 if arm % 2 == 0 else 1) * 0.26
                ready[2] = max(self.lift_height, self.ids.bin_rim_z + 0.12)
            cost = np.linalg.norm(lifted - drop) + np.linalg.norm(drop - ready)
            if partner_demand:
                side = -1 if arm % 2 == 0 else 1
                if side * lifted[1] < 0.23:
                    # Integrate dy/dt = min(speed, 10 * remaining) until
                    # |y|=.23. An inner slot at .235 leaves only 5 mm of
                    # remaining error there, inside the slow settling tail;
                    # an existing outer slot at .325 clears at cruising speed.
                    # This is a lateral scheduling estimate, not a collision
                    # certificate or a promise about actual completion time.
                    speed = max(float(self.config.control_speed), 0.01)
                    cruise_error = speed / 10.0
                    start_error = max(side * (drop[1] - lifted[1]), 1e-6)
                    end_error = max(side * drop[1] - 0.23, 1e-6)
                    cruise_time = (
                        max(start_error - cruise_error, 0.0)
                        - max(end_error - cruise_error, 0.0)
                    ) / speed
                    settle_time = math.log(
                        min(start_error, cruise_error) / min(end_error, cruise_error)
                    ) / 10.0
                    # Give a contended station twice the weight of one arm's
                    # travel, expressed in the existing distance-cost units.
                    cost += 2.0 * speed * max(cruise_time + settle_time, 0.0)
            costs.append(cost)
        return int(eligible[int(np.argmin(costs))])

    def _baseline_return_goal(self, arm):
        """Choose an empty-hand ready pose; execution must still settle there.

        A short return preserves the deposit's longitudinal coordinate instead
        of going back to the arm base x. It clears the bin vertically before
        another assignment, and never changes a moving path in the mask. The
        usual motion certificate and braking checks remain authoritative.
        """
        goal = self.ids.home_ee[arm].copy()
        target = int(self.target[arm])
        if (
            not self.config.direct_retask
            or self.holding[arm]
            or target < 0
            or self.belief[target] < 4
            or self.drop_kind[target] != arm % 2
            or not self._gripper_released(arm)
        ):
            return goal
        goal[0] = np.clip(
            self.drop_targets[target, 0],
            self.ids.arm_base_positions[arm, 0] - 0.18,
            self.ids.arm_base_positions[arm, 0] + 0.18,
        )
        if getattr(self.config, "cooperative_motion", False):
            # Remain outside the station strip after a confirmed empty-hand
            # release. This is a checked physical endpoint, not early idle or
            # permission to bypass the partner's measured exclusion state.
            goal[1] = (-1 if arm % 2 == 0 else 1) * 0.26
        goal[2] = max(self.lift_height, self.ids.bin_rim_z + 0.12)
        return goal

    def _planned_ready_goal(self, arm):
        """Pure advisory task-end pose; this does not authorize any motion.

        Forecasting can use the same calibrated endpoint before release.
        Execution locks its return goal after measured release, then checks
        the complete robot path and keeps its reservations until settled.
        """
        base_x = float(self.ids.arm_base_positions[arm, 0])
        target = int(self.target[arm])
        x = float(self.drop_targets[target, 0]) if target >= 0 else base_x
        suggestions = getattr(self, "future_preposition_goals", None)
        if suggestions is not None and np.shape(suggestions) == (self.num_arms, 3):
            proposed = np.asarray(suggestions[arm], dtype=float)
            if np.isfinite(proposed).all():
                x = float(proposed[0])
        return np.array([
            np.clip(x, base_x - 0.18, base_x + 0.18),
            (-1 if arm % 2 == 0 else 1) * 0.26,
            max(self.lift_height, self.ids.bin_rim_z + 0.12),
        ])

    def _planned_return_available(self, arm):
        target = int(self.target[arm])
        return bool(
            getattr(self.config, "planned_return", False)
            and self.num_arms == 4 and not self.suction
            and not self.holding[arm] and target >= 0
            and self.belief[target] >= 4 and self.drop_kind[target] == arm % 2
            and self._gripper_released(arm)
        )

    def _normal_return_goal(self, arm):
        if self._planned_return_available(arm):
            return self._planned_ready_goal(arm)
        return self._baseline_return_goal(arm)

    def _assign(self, arm, i):
        self.wrist_offsets[arm] = 0
        self.wrist_last_used_timestamp[arm] = -np.inf
        self.target[arm] = i
        self.belief[i] = 2
        self.dispatch_time[arm] = self.data.time
        self.arm_assignments[arm] += 1
        self.max_assignments_without_sort[arm] = max(
            self.max_assignments_without_sort[arm],
            self.arm_assignments[arm] - self.assignments_at_last_sort[arm],
        )
        track = self.tracks[i]
        self.grasp_sizes[arm] = np.asarray(track.size)
        self.grasp_min_widths[arm] = min(track.size[:2])
        if self.suction:
            # Suction does not align the object's yaw to the tool frame. Use a
            # yaw-independent envelope for every loaded collision check.
            self.grasp_sizes[arm, :2] = np.linalg.norm(track.size[:2])
        self.grasp_offsets[arm] = self._grasp_offset(track)
        self.reoriented[arm] = False
        angles = self._grasp_angles(arm, track)
        clear_angles = [
            yaw for yaw in angles if self._finger_neighborhood_clear(arm, i, yaw)
        ]
        if clear_angles:
            angles = clear_angles
        if not angles:
            raise RuntimeError("Assigned visual footprint does not fit finger opening")
        self.grasp_yaw[arm] = angles[0]
        # Test symmetry-equivalent wrist branches using robot FK only.
        p = self._position(i).copy()
        p[2] += self.grasp_offsets[arm] + 0.10
        best = None
        for yaw in angles[:4]:
            q = self.ik.solve_pose(self.data, arm, p, downward_quaternion(yaw))
            ep, er = self.ik.pose_error(self.data, arm, q, p, downward_quaternion(yaw))
            cost = (
                100 * ep
                + er
                + 0.02 * np.linalg.norm(q - self.data.qpos[self.ids.arm_qpos[arm]])
                + 0.30
                * abs(
                    (yaw - (np.pi / 2 if arm % 2 == 0 else -np.pi / 2) + np.pi)
                    % (2 * np.pi)
                    - np.pi
                )
            )
            if best is None or cost < best[0]:
                best = (cost, yaw)
        self.grasp_yaw[arm] = best[1]
        if not self.suction:
            self.grasp_sizes[arm] = self._grasp_size_at_yaw(track, best[1])
            # A diagonally clamped prism may rotate while closing. Retain a
            # yaw-independent conservative payload envelope for path checks.
            self.grasp_sizes[arm, :2] = np.linalg.norm(track.size[:2])
        color = self.tracks[i].kind
        free = np.flatnonzero(~self.bin_occupied[color])
        if self.num_arms == 4:
            per_zone = self.config.bin_capacity // 2
            first = (arm // 2) * per_zone
            free = free[(free >= first) & (free < first + per_zone)]
        if not len(free):
            raise RuntimeError("No free calibrated deposit slot")
        rank = self._choose_drop_slot(arm, i, free)
        self.bin_occupied[color, rank] = True
        self.drop_slot[i] = rank
        self.bin_slots[color] = np.count_nonzero(self.bin_occupied[color])
        self.drop_kind[i] = color
        self.drop_targets[i] = self._deposit_position(arm, color, rank)
        self.holding[arm] = False
        self._set_phase(arm, "approach")
        self.eval_queue.append(("dispatch", arm, i, self._position(i).copy()))
        self.events.append(
            {
                "time": float(self.data.time),
                "event": "dispatch",
                "track": int(self.tracks[i].track_id),
                "slot": i,
                "arm": arm,
                "class": int(color),
            }
        )

    def _release_resources(self, arm):
        self.bin_owner[self.bin_owner == arm] = -1
        self.bin_zone_owner[self.bin_zone_owner == arm] = -1
        if self.num_arms == 4:
            for color in range(2):
                owners = self.bin_zone_owner[:, color]
                active = owners[owners >= 0]
                self.bin_owner[color] = int(active[0]) if len(active) else -1
        if self.transfer_owner == arm:
            self.transfer_owner = -1

    def _recover(self, arm):
        i = self.target[arm]
        if self.phase[arm] == "recover":
            return
        if i >= 0:
            if self.belief[i] < 4 and not self.holding[arm] and self.drop_slot[i] >= 0:
                color = self.drop_kind[i]
                self.bin_occupied[color, self.drop_slot[i]] = False
                self.bin_slots[color] = np.count_nonzero(self.bin_occupied[color])
            if self.belief[i] < 4:
                self.belief[i] = 4 if self.holding[arm] else 1
            self.retry_after[arm, i] = self.data.time + 1.2
        self.eval_queue.append(("recover", arm, i, None))
        self.retries += 1
        self.holding[arm] = False
        self.grasp_yaw[arm] = np.pi / 2 if arm % 2 == 0 else -np.pi / 2
        self.tool_yaw[arm] = self.grasp_yaw[arm]
        self.recovery_withdrawal_goal[arm] = None
        self.recovery_withdrawal_quaternion[arm] = None
        self.recovery_withdrawal_used[arm] = False
        self.recovery_withdrawal_fault[arm] = None
        self._set_phase(arm, "recover")

    def _gripper_has_object(self, arm):
        if self.vacuum is not None:
            return bool(self.vacuum.sensor(arm)["sealed"])
        aperture = float(np.sum(self.data.qpos[self.ids.finger_qpos[arm]]))
        width = self.grasp_sizes[arm, 1]
        return max(0.008, self.grasp_min_widths[arm] - 0.025) < aperture < min(
            0.078, width + 0.020
        ) and bool(np.all(self.data.sensordata[self.ids.finger_touch_adr[arm]] > 0.10))

    def _responsive_release_active(self, arm):
        return bool(
            self.config.responsive_release
            and not self.suction
            and self.phase[arm] in ("release", "retreat", "return")
            and self.target[arm] >= 0
            and np.isfinite(self.release_start_aperture[arm])
        )

    def _release_aperture_threshold(self, arm):
        if self._responsive_release_active(arm):
            # Ten millimetres of total free jaw clearance is sufficient once
            # both pads remain unloaded. Wide parts retain the original
            # 75 mm release criterion instead of an unreachable >80 mm goal.
            return min(0.075, float(self.release_start_aperture[arm]) + 0.010)
        return 0.075

    def _gripper_released(self, arm):
        if self.vacuum is not None:
            sensor = self.vacuum.sensor(arm)
            return not sensor["sealed"] and sensor["pressure_pa"] < 100.0
        return bool(
            np.sum(self.data.qpos[self.ids.finger_qpos[arm]])
            > self._release_aperture_threshold(arm)
            and np.all(self.data.sensordata[self.ids.finger_touch_adr[arm]] < 0.10)
        )

    def _release_diagnostics(self, arm):
        """Encoder/tactile evidence; never substitutes for release confirmation."""
        return {
            "aperture_m": float(np.sum(self.data.qpos[self.ids.finger_qpos[arm]])),
            "required_aperture_m": float(self._release_aperture_threshold(arm)),
            "finger_positions_m": self.data.qpos[self.ids.finger_qpos[arm]].tolist(),
            "finger_touch": self.data.sensordata[self.ids.finger_touch_adr[arm]].tolist(),
            "commanded_half_aperture_m": float(self.ik.gripper_targets[arm]),
            "released": bool(self._gripper_released(arm)),
        }

    def _robot_contact_diagnostics(self):
        """Evaluator-only contact pairs, excluded from scheduling observations."""
        return [
            {
                "geom1_id": pair[0],
                "geom2_id": pair[1],
                "geom1": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, pair[0]),
                "geom2": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, pair[1]),
                "body1_id": int(self.model.geom_bodyid[pair[0]]),
                "body2_id": int(self.model.geom_bodyid[pair[1]]),
                "body1": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[pair[0]])),
                "body2": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[pair[1]])),
                "arm1": int(self.ids.robot_geom_arm[pair[0]]),
                "arm2": int(self.ids.robot_geom_arm[pair[1]]),
                "physics_steps": count,
                "max_penetration_m": self.robot_contact_pair_penetration[pair],
                "phase_pair_steps": [
                    {"phase1": key[2], "phase2": key[3], "physics_steps": phase_count}
                    for key, phase_count in self.robot_contact_pair_phase_steps.items()
                    if key[:2] == pair
                ],
            }
            for pair, count in self.robot_contact_pair_steps.most_common()
        ]

    def _release_withdrawal_step(self, arm):
        """One checked, open-hand vertical separation from unilateral contact.

        Fully open fingers can rest against a workpiece without holding it.
        Keeping the joints fixed then prevents the tactile signal clearing.
        This is a contact-release primitive, not an empty transit certificate:
        it retains the recovery reservation, opens the hand, preserves its
        measured orientation and only retreats vertically above the belt.
        The planner still checks all robot links, fixtures and other arms;
        as for the initial grasp lift, it does not certify workpiece dynamics.
        Bilateral load, insufficient aperture, a rejected route or a time
        limit stops motion. Ordinary recovery still requires actual release.
        No hidden workpiece state is read by this control primitive.
        """
        if not getattr(self.config, "active_recovery", False) or self.suction:
            return None
        now = float(self.data.time)
        goal = self.recovery_withdrawal_goal[arm]
        aperture = float(np.sum(self.data.qpos[self.ids.finger_qpos[arm]]))
        touch = self.data.sensordata[self.ids.finger_touch_adr[arm]]
        ee = self.data.site_xpos[self.ids.ee_sites[arm]]
        if goal is None:
            if self.recovery_withdrawal_used[arm] or now - self.phase_start[arm] < 0.6:
                return None
            if self._gripper_released(arm):
                return None
            if not (
                aperture > 0.075
                and np.count_nonzero(touch >= 0.10) == 1
                and np.max(touch) < 8.0
                and abs(ee[1]) < self.config.belt_half_width + 0.05
                and self.config.belt_top < ee[2] < self.config.belt_top + 0.12
            ):
                return None
            goal = ee.copy() + [0.0, 0.0, 0.08]
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ids.ee_sites[arm]])
            self.recovery_withdrawal_goal[arm] = goal
            self.recovery_withdrawal_quaternion[arm] = quat
            self.recovery_withdrawal_start[arm] = now
            self.recovery_withdrawal_used[arm] = True
            self.recovery_withdrawal_attempts[arm] += 1
            self.routes[arm] = None
            self.route_retry_time[arm] = now
            self.events.append({"time": now, "event": "release_withdrawal_start", "arm": arm})
        fault = None
        if aperture <= 0.075 or np.count_nonzero(touch >= 0.10) > 1 or np.max(touch) >= 8.0:
            fault = "opening_or_unilateral_contact_lost"
        elif now - self.recovery_withdrawal_start[arm] > 1.5:
            fault = "withdrawal_time_limit"
        if fault is not None:
            self.recovery_withdrawal_fault[arm] = fault
            self.recovery_withdrawal_goal[arm] = None
            self.routes[arm] = None
            self.hold_requested[arm] = True
            self.events.append({"time": now, "event": "release_withdrawal_stopped", "arm": arm, "reason": fault})
            return ee.copy(), False
        point, valid = self._route_goal(arm, goal, False, release_withdrawal=True)
        plan = self.routes[arm]
        actual_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(actual_quaternion, self.data.site_xmat[self.ids.ee_sites[arm]])
        if not valid:
            self.recovery_withdrawal_fault[arm] = "route_not_certified"
        elif plan.can_advance(
            len(plan.waypoints) - 1, ee, self.ik.cart_velocities[arm],
            quaternion=actual_quaternion,
            joint_positions=self.data.qpos[self.ids.arm_qpos[arm]],
            acceleration=self.ik.max_cartesian_acceleration,
        ):
            self.recovery_withdrawal_goal[arm] = None
            self.routes[arm] = None
            self.hold_requested[arm] = True
            if self._gripper_released(arm):
                self.recovery_withdrawal_successes[arm] += 1
                self.recovery_withdrawal_fault[arm] = None
            else:
                self.recovery_withdrawal_fault[arm] = "contact_remains_after_withdrawal"
            self.events.append({"time": now, "event": "release_withdrawal_complete", "arm": arm,
                                "released": bool(self._gripper_released(arm))})
        # Never report the home-return motion complete from a release primitive.
        return point, False

    def _safe_preposition_step(self, arm):
        """Prepare off the belt while the opposing arm is stationary.

        Soft plans never claim grasp/bin resources. A moving opposing arm
        cancels preparation; this first implementation does not pretend a
        static collision certificate covers simultaneous future arm motion.
        """
        ee = self.data.site_xpos[self.ids.ee_sites[arm]]
        goals = getattr(self, "future_preposition_goals", None)
        yaws = getattr(self, "future_preposition_yaws", None)
        permitted = (
            getattr(self.config, "safe_preposition", False)
            and self.num_arms == 4 and self.phase[arm] == "idle"
            and self.target[arm] < 0 and self._gripper_released(arm)
            and self.phase[arm ^ 1] in ("idle", "release")
            and not self.preposition_active[arm ^ 1]
            and not self._previous_preposition_active[arm ^ 1]
            and np.max(np.abs(self.data.qvel[self.ids.arm_dofs[arm ^ 1]])) <= 0.02
            and np.linalg.norm(self.ik.cart_velocities[arm ^ 1]) <= 0.005
            and np.isfinite(self._preposition_stationary_since[arm ^ 1])
            and self.data.time - self._preposition_stationary_since[arm ^ 1] >= 0.05
            and goals is not None and yaws is not None
            and np.shape(goals) == (self.num_arms, 3)
            and np.shape(yaws) == (self.num_arms,)
        )
        if permitted:
            goal = np.asarray(goals[arm], dtype=float)
            yaw = float(yaws[arm])
            side = -1 if arm % 2 == 0 else 1
            permitted = bool(
                np.isfinite(goal).all() and math.isfinite(yaw)
                and self.windows[arm, 0] <= goal[0] <= self.windows[arm, 1]
                and 0.25 <= side * goal[1] <= 0.40
                and max(self.config.belt_top + 0.12, self.ids.bin_rim_z + 0.08) <= goal[2] <= 0.55
            )
        if not permitted:
            if self._preposition_live_goal[arm] is not None:
                self.idle_anchors[arm] = ee.copy()
                self._preposition_live_goal[arm] = None
                self.routes[arm] = None
            return None
        previous = self._preposition_live_goal[arm]
        if previous is None:
            self.preposition_starts[arm] += 1
            self.routes[arm] = None
            self.route_retry_time[arm] = self.data.time
        elif abs(symmetric_angle(yaw - self._preposition_live_yaw[arm], 2 * np.pi)) > 0.05:
            self.routes[arm] = None
            self.route_retry_time[arm] = self.data.time
        self._preposition_live_goal[arm] = goal.copy()
        self._preposition_live_yaw[arm] = yaw
        self.tool_yaw[arm] = yaw
        self.hold_requested[arm] = False
        point, valid = self._route_goal(arm, goal, False)
        plan = self.routes[arm]
        actual_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(actual_quaternion, self.data.site_xmat[self.ids.ee_sites[arm]])
        if valid and plan.can_advance(
            len(plan.waypoints) - 1, ee, self.ik.cart_velocities[arm],
            quaternion=actual_quaternion,
            joint_positions=self.data.qpos[self.ids.arm_qpos[arm]],
            acceleration=self.ik.max_cartesian_acceleration,
        ):
            self.idle_anchors[arm] = ee.copy()
            self.hold_requested[arm] = True
        self.preposition_active[arm] = bool(valid and not self.hold_requested[arm])
        return point, yaw

    def _route_goal(self, arm, goal, loaded, via=(), velocity=None, *, release_withdrawal=False):
        if release_withdrawal and not (
            getattr(self.config, "active_recovery", False)
            and self.phase[arm] == "recover" and not loaded
            and self.recovery_withdrawal_goal[arm] is not None
            and np.linalg.norm(np.asarray(goal) - self.recovery_withdrawal_goal[arm]) < 1e-9
        ):
            self.hold_requested[arm] = True
            return self.data.site_xpos[self.ids.ee_sites[arm]].copy(), False
        if self.phase[arm] == "recover" and not self._gripper_released(arm) and not release_withdrawal:
            # Recovery discards the estimated grasp only after the actual hand
            # is open and unloaded. Until then, opening the fingers and holding
            # the measured joints is safer than an unloaded motion certificate.
            self.routes[arm] = None
            self.escape_routes[arm] = False
            self.route_retry_time[arm] = self.data.time
            self.hold_requested[arm] = True
            return self.data.site_xpos[self.ids.ee_sites[arm]].copy(), False
        if self.config.motion_mode == "legacy":
            return np.asarray(goal), True
        current_quat = np.empty(4)
        mujoco.mju_mat2Quat(current_quat, self.data.site_xmat[self.ids.ee_sites[arm]])
        self.quat_override[arm] = current_quat
        existing = self.routes[arm]
        plan_goal = np.asarray(goal).copy()
        if self.config.strategy == "adaptive" and velocity is not None:
            # Certify the servo's short forward horizon on the same joint
            # branch. Ending a certificate at the instantaneous moving target
            # clips the servo lead and causes persistent tracking lag.
            lead = np.asarray(velocity) * 0.12
            lead *= min(1.0, 0.06 / max(np.linalg.norm(lead), 1e-12))
            plan_goal += lead
        # Near the object, a stale 25 mm endpoint is large enough to tip a
        # cylinder before closure. Refresh belt tracking at millimetre scale.
        tracking_tolerance = (
            0.003
            if self.phase[arm] in ("descend", "close", "lift")
            else 0.010
            if self.phase[arm] == "approach"
            else 0.025
        )
        if (
            existing is not None
            and existing.verified
            and not self.escape_routes[arm]
            and np.linalg.norm(plan_goal - existing.waypoints[-1]) > tracking_tolerance
        ):
            self.routes[arm] = None
            self.route_retry_time[arm] = self.data.time
        if self.routes[arm] is None and self.data.time >= self.route_retry_time[arm]:
            near_belt = self.phase[arm] in ("descend", "close", "lift") or (
                self.phase[arm] == "recover"
                and self.data.site_xpos[self.ids.ee_sites[arm], 2]
                < self.config.belt_top + 0.13
            )
            plan = None
            if self.phase[arm] == "recover" and not release_withdrawal:
                # Equal tool poses can have different redundant elbow branches.
                # Restore the calibrated joints before declaring recovery done.
                plan = self.planner.plan_home(
                    self.data,
                    arm,
                    near_belt=near_belt,
                    grasp_offset=self.grasp_offsets[arm],
                )
            if plan is None or not plan.verified:
                plan = self.planner.plan(
                    self.data,
                    arm,
                    plan_goal,
                    quaternion=self.recovery_withdrawal_quaternion[arm] if release_withdrawal else downward_quaternion(
                        self.grasp_yaw[arm]
                        if self.target[arm] >= 0
                        else self.tool_yaw[arm]
                    ),
                    loaded=loaded,
                    object_size=self.grasp_sizes[arm],
                    grasp_offset=self.grasp_offsets[arm],
                    near_belt=near_belt,
                    via=via,
                )
            if (
                not plan.verified
                and self.phase[arm] == "approach"
                and not self.suction
                and self.config.strategy == "adaptive"
            ):
                # Opposing fingers admit a pi-rotated equivalent grasp. The
                # first wrist branch can be unreachable even though its twin
                # is feasible. Switch only after the full path is certified.
                alternate = float(
                    (self.grasp_yaw[arm] + 2 * np.pi) % (2 * np.pi) - np.pi
                )
                candidate = self.planner.plan(
                    self.data,
                    arm,
                    plan_goal,
                    quaternion=downward_quaternion(alternate),
                    loaded=False,
                    object_size=self.grasp_sizes[arm],
                    grasp_offset=self.grasp_offsets[arm],
                    near_belt=near_belt,
                    via=via,
                )
                if candidate.verified:
                    plan = candidate
                    self.grasp_yaw[arm] = alternate
                    self.events.append(
                        {
                            "time": float(self.data.time),
                            "event": "approach_wrist_branch",
                            "arm": arm,
                        }
                    )
            if not plan.verified and self.phase[arm] == "transport":
                # A symmetric hand orientation can have a different feasible
                # joint branch. Retain the grasp orientation whenever feasible;
                # classification into a bin does not require an extra rotation.
                nominal = np.pi / 2 if arm % 2 == 0 else -np.pi / 2
                for alternate in (
                    nominal,
                    nominal + np.pi,
                    self.grasp_yaw[arm] + np.pi,
                ):
                    candidate = self.planner.plan(
                        self.data,
                        arm,
                        goal,
                        quaternion=downward_quaternion(alternate),
                        loaded=loaded,
                        object_size=self.grasp_sizes[arm],
                        grasp_offset=self.grasp_offsets[arm],
                        via=via,
                    )
                    if candidate.verified:
                        plan = candidate
                        self.grasp_yaw[arm] = alternate
                        break
            if not plan.verified and self.phase[arm] == "recover" and not loaded and not release_withdrawal:
                self.escape_attempts += 1
                escape = self.planner.plan_escape(
                    self.data,
                    arm,
                    goal,
                    quaternion=current_quat,
                    near_belt=near_belt,
                    grasp_offset=self.grasp_offsets[arm],
                    via=via,
                )
                if escape.verified:
                    plan = escape
                    self.escape_routes[arm] = True
                    self.events.append(
                        {
                            "time": float(self.data.time),
                            "event": "escape_start",
                            "arm": arm,
                            "reason": plan.reason,
                        }
                    )
            self.routes[arm] = plan
            self.route_index[arm] = 0
            if not plan.verified:
                self.route_failures += 1
                self.route_retry_time[arm] = self.data.time + 0.3
                self.events.append(
                    {
                        "time": float(self.data.time),
                        "event": "route_rejected",
                        "arm": arm,
                        "phase": self.phase[arm],
                        "reason": plan.reason,
                        "goal": np.asarray(goal).tolist(),
                    }
                )
            elif len(plan.waypoints) < len(via) + 1:
                self.route_shortcuts += 1
        plan = self.routes[arm]
        ee = self.data.site_xpos[self.ids.ee_sites[arm]]
        if plan is None or not plan.verified:
            if self.data.time >= self.route_retry_time[arm]:
                self.routes[arm] = None
            self.hold_requested[arm] = True
            return ee.copy(), False
        index = int(self.route_index[arm])
        while index < len(plan.waypoints) - 1 and plan.can_advance(
            index,
            ee,
            self.ik.cart_velocities[arm],
            quaternion=current_quat,
            acceleration=self.ik.max_cartesian_acceleration,
            joint_positions=self.data.qpos[self.ids.arm_qpos[arm]],
        ):
            index += 1
        self.route_index[arm] = index
        if (
            self.escape_routes[arm]
            and index == len(plan.waypoints) - 1
            and plan.can_advance(
                index,
                ee,
                self.ik.cart_velocities[arm],
                quaternion=current_quat,
                acceleration=self.ik.max_cartesian_acceleration,
                joint_positions=self.data.qpos[self.ids.arm_qpos[arm]],
            )
        ):
            self.routes[arm] = None
            self.escape_routes[arm] = False
            self.escape_successes += 1
            self.route_retry_time[arm] = self.data.time
            self.hold_requested[arm] = True
            # Even a full escape can finish at home with the original wrist
            # orientation. Let a fresh normal plan certify the remaining turn.
            return ee.copy(), False
        self.quat_override[arm] = plan.orientation_at(index, ee)
        self.active_motion_plans[arm] = plan
        self.active_motion_indices[arm] = index
        # The last point may be a moving visual interception target.
        return (
            np.asarray(goal)
            if index == len(plan.waypoints) - 1 and not self.escape_routes[arm]
            else plan.waypoints[index]
        ), True

    def _service_bins(self):
        for color in range(2):
            stage = self.bin_service[color]
            elapsed = self.data.time - self.bin_service_start[color]
            if stage == "closed":
                if self.num_arms == 4 and self.config.auto_unload:
                    per_zone = self.config.bin_capacity // 2
                    full_zone = any(
                        np.all(
                            self.bin_occupied[
                                color, zone * per_zone : (zone + 1) * per_zone
                            ]
                        )
                        for zone in range(2)
                    )
                    if (
                        full_zone
                        and self.bin_slots[color]
                        >= math.ceil(0.75 * self.config.bin_capacity)
                        and not self.bin_drain_requested[color]
                    ):
                        self.bin_drain_requested[color] = True
                        self.events.append(
                            {
                                "time": float(self.data.time),
                                "event": "unload_requested",
                                "bin": color,
                                "reserved_slots": int(self.bin_slots[color]),
                                "reason": "station_full_and_at_least_75_percent",
                            }
                        )
                pending = any(
                    self.target[a] >= 0 and self.drop_kind[self.target[a]] == color
                    for a in range(self.num_arms)
                )
                if (
                    self.config.auto_unload
                    and (
                        self.bin_slots[color] >= self.config.bin_capacity
                        or self.bin_drain_requested[color]
                    )
                    and not pending
                    and self.bin_owner[color] < 0
                    and np.all(self.bin_zone_owner[:, color] < 0)
                    and self.data.time - self.bin_last_release[color] > 1.5
                ):
                    self.bin_service[color] = "opening"
                    self.bin_service_start[color] = self.data.time
                    self.events.append(
                        {
                            "time": float(self.data.time),
                            "event": "unload_start",
                            "bin": color,
                        }
                    )
            elif stage == "opening":
                actuator = self.ids.bin_gate_actuators[color]
                self.data.ctrl[actuator] = (
                    min(
                        self.ids.bin_gate_open_position,
                        self.data.ctrl[actuator] + 0.6 * self.config.physics_dt,
                    )
                    if self.num_arms == 4
                    else self.ids.bin_gate_open_position
                )
                if (
                    self.data.qpos[self.ids.bin_gate_qpos[color]]
                    > self.ids.bin_gate_open_position - 0.02
                ):
                    self.bin_service[color] = "draining"
                    self.bin_service_start[color] = self.data.time
            elif stage == "draining":
                if elapsed >= 2.0:
                    self.bin_service[color] = "closing"
                    self.bin_service_start[color] = self.data.time
            elif stage == "closing":
                actuator = self.ids.bin_gate_actuators[color]
                self.data.ctrl[actuator] = (
                    max(0.0, self.data.ctrl[actuator] - 0.6 * self.config.physics_dt)
                    if self.num_arms == 4
                    else 0.0
                )
                if self.data.qpos[self.ids.bin_gate_qpos[color]] < 0.005:
                    self.bin_service[color] = "closed"
                    self.bin_occupied[color] = False
                    self.bin_slots[color] = 0
                    self.bin_release_counts[color] = 0
                    self.bin_unloads[color] += 1
                    self.bin_drain_requested[color] = False
                    self.events.append(
                        {
                            "time": float(self.data.time),
                            "event": "unload_complete",
                            "bin": color,
                        }
                    )

    def _control(self):
        c, d = self.config, self.data
        self._service_bins()
        self._previous_preposition_active[:] = self.preposition_active
        self.preposition_active[:] = False
        for a in range(self.num_arms):
            stationary = (
                self.phase[a] in ("idle", "release")
                and not self._previous_preposition_active[a]
                and np.max(np.abs(d.qvel[self.ids.arm_dofs[a]])) <= 0.02
                and np.linalg.norm(self.ik.cart_velocities[a]) <= 0.005
            )
            if not stationary:
                self._preposition_stationary_since[a] = np.nan
            elif not np.isfinite(self._preposition_stationary_since[a]):
                self._preposition_stationary_since[a] = d.time
        for a in range(self.num_arms):
            phase, i = self.phase[a], self.target[a]
            elapsed = d.time - self.phase_start[a]
            ee = d.site_xpos[self.ids.ee_sites[a]]
            goal = self.idle_anchors[a].copy()
            grip = 0.04
            ff = np.zeros(3)
            yaw = self.grasp_yaw[a] if i >= 0 else self.tool_yaw[a]
            side = -1 if a % 2 == 0 else 1
            self.quat_override[a] = None
            # Waiting at an idle anchor or opening a released grasp should not
            # run the redundant IK's null-space bias on every control tick.
            self.hold_requested[a] = phase in ("idle", "release")
            self.active_motion_plans[a] = None
            if phase == "idle":
                prepared = self._safe_preposition_step(a)
                if prepared is not None:
                    goal, yaw = prepared
            elif phase in ("approach", "descend", "close"):
                track = self.tracks[i]
                if phase in (
                    "descend",
                    "close",
                ) and not self._finger_neighborhood_clear(a, i, yaw):
                    self.events.append(
                        {
                            "time": float(d.time),
                            "event": "neighbor_grasp_blocked",
                            "arm": a,
                            "track": int(track.track_id),
                        }
                    )
                    self._recover(a)
                    self.hold_requested[a] = True
                    self.active_motion_plans[a] = None
                    self.ik.command(d, a, ee, gripper=0.04, hold=True)
                    continue
                p = (
                    self._grasp_position(a, i)
                    if phase != "close"
                    else self._position(i)
                )
                goal = p.copy()
                goal[2] = c.belt_top + track.half_height + self.grasp_offsets[a]
                velocity_limit = 0.6 if c.strategy == "adaptive" else 0.2
                velocity = (
                    track.predict_velocity(d.time)
                    if c.strategy == "adaptive"
                    else track.velocity
                )
                ff[:2] = np.clip(velocity[:2], -velocity_limit, velocity_limit)
                if phase == "approach":
                    goal[2] += self.approach_clearance
                    direct = goal.copy()
                    if (
                        c.strategy == "adaptive"
                        and not self.suction
                        and self.num_arms == 4
                        and a >= 2
                    ):
                        # Aim at the moving rendezvous, not an earlier pose
                        # that can still lie outside this station's reach.
                        # The horizon falls to zero as the actual hand meets
                        # the observed object. Descent admission below still
                        # compares against its current measured position.
                        delta = goal - ee
                        speed = 0.85 * c.control_speed
                        quadratic = float(np.dot(ff, ff) - speed**2)
                        linear = float(2 * np.dot(delta, ff))
                        discriminant = linear**2 - 4 * quadratic * float(
                            np.dot(delta, delta)
                        )
                        if quadratic < -1e-8 and discriminant >= 0:
                            rendezvous = (-linear - math.sqrt(discriminant)) / (
                                2 * quadratic
                            )
                            lead = ff * min(1.2, max(0.0, rendezvous))
                            lead *= min(1.0, 0.15 / max(np.linalg.norm(lead), 1e-12))
                            goal += lead
                    goal, ready = self._route_goal(
                        a,
                        goal,
                        False,
                        via=[self.ids.home_ee[a], np.array([p[0], side * 0.18, 0.48])],
                        velocity=ff,
                    )
                    # A verified equivalent wrist branch can be selected by
                    # the approach planner. Use its orientation immediately
                    # for alignment and the descent preview in this tick.
                    yaw = self.grasp_yaw[a]
                    if np.linalg.norm(goal - direct) > 0.02:
                        ff[:] = 0
                    if c.strategy == "adaptive" and not self._descent_feasible(
                        a, track, p
                    ):
                        self.events.append(
                            {
                                "time": float(d.time),
                                "event": "intercept_expired",
                                "arm": a,
                                "track": int(track.track_id),
                            }
                        )
                        self._recover(a)
                        self.hold_requested[a] = True
                        ff[:] = 0
                        self.active_motion_plans[a] = None
                        self.ik.command(d, a, ee, gripper=0.04, hold=True)
                        continue
                    if (
                        ready
                        and np.linalg.norm(ee - direct) < 0.018
                        and np.linalg.norm(ee[:2] - direct[:2]) < 0.006
                    ):
                        actual = np.empty(4)
                        mujoco.mju_mat2Quat(actual, d.site_xmat[self.ids.ee_sites[a]])
                        if (
                            2
                            * math.acos(
                                np.clip(
                                    abs(np.dot(actual, downward_quaternion(yaw))), 0, 1
                                )
                            )
                            < 0.05
                        ):
                            if not self._finger_neighborhood_clear(a, i, yaw):
                                self.events.append(
                                    {
                                        "time": float(d.time),
                                        "event": "neighbor_grasp_blocked",
                                        "arm": a,
                                        "track": int(track.track_id),
                                    }
                                )
                                self._recover(a)
                                self.hold_requested[a] = True
                                self.active_motion_plans[a] = None
                                self.ik.command(d, a, ee, gripper=0.04, hold=True)
                                continue
                            low = p.copy()
                            low[2] = (
                                c.belt_top + track.half_height + self.grasp_offsets[a]
                            )
                            future = low.copy()
                            future[:2] += (
                                self.descent_time if self.suction else 0.5
                            ) * np.clip(velocity[:2], -velocity_limit, velocity_limit)
                            descent = self.planner.plan(
                                d,
                                a,
                                future,
                                quaternion=downward_quaternion(yaw),
                                near_belt=True,
                                via=[low],
                            )
                            if descent.verified and c.strategy == "adaptive":
                                closure = low.copy()
                                closure[:2] += (
                                    (self.descent_time if self.suction else 0.5)
                                    + self.closure_timeout
                                ) * velocity[:2]
                                lift = closure.copy()
                                lift[0] += (0.30 if self.suction else 0.10) * max(
                                    0.0, float(velocity[0])
                                )
                                lift[2] = self.lift_height
                                descent = self.planner.grasp_exit_feasible(
                                    d,
                                    a,
                                    closure,
                                    lift,
                                    quaternion=downward_quaternion(yaw),
                                    grasp_offset=self.grasp_offsets[a],
                                    grasp_position=future,
                                )
                            if descent.verified:
                                self._set_phase(a, "descend")
                            elif not self.reoriented[a]:
                                self.reoriented[a] = True
                                self.grasp_yaw[a] += np.pi
                                self.reorient_goals[a] = ee + [0, 0, 0.06]
                                self._set_phase(a, "reorient")
                            else:
                                self._recover(a)
                elif phase == "descend":
                    guard_reason = self._descent_guard_reason(a, track, p, elapsed)
                    if guard_reason is not None:
                        self.descent_guard_stops[guard_reason] += 1
                        self.events.append({"time": float(d.time), "event": "descent_guard_stop",
                                            "arm": a, "track": int(track.track_id), "reason": guard_reason})
                        self._recover(a)
                        self.hold_requested[a] = True
                        self.active_motion_plans[a] = None
                        self.ik.command(d, a, ee, gripper=0.04, hold=True)
                        continue
                    end = goal.copy()
                    if c.strategy == "adaptive" and not self.suction:
                        # Reach the predicted object position when the
                        # remaining vertical descent completes, instead of
                        # sweeping a lagging open finger into the workpiece.
                        remaining = min(0.35, max(0.0, float(ee[2] - end[2])) / 0.40)
                        end[:2] += self._descent_lead(a, track, ff, remaining, position=p)
                    goal, ready = self._route_goal(a, end, False, velocity=ff)
                    if not ready:
                        ff[:] = 0
                        if c.strategy == "adaptive":
                            self._recover(a)
                            self.hold_requested[a] = True
                            self.active_motion_plans[a] = None
                            self.ik.command(d, a, ee, gripper=0.04, hold=True)
                            continue
                    if ready and self._descent_alignment_ready(a, track, p) and np.linalg.norm(ee - end) < (
                        0.007 if self.suction else 0.009
                    ):
                        self.close_position[a] = p
                        self.close_displacement[a] = 0.0
                        self._set_phase(a, "close")
                else:
                    speed = self._belt_velocity()
                    if c.strategy == "adaptive":
                        self.close_displacement[a] += speed * c.physics_dt
                        displacement = self.close_displacement[a]
                    else:
                        displacement = c.belt_speed * elapsed
                    goal[:2] = self.close_position[a, :2] + [displacement, 0]
                    ff[:] = (speed, 0, 0)
                    grip = 0
                    goal, ready = self._route_goal(a, goal, False, velocity=ff)
                    if not ready:
                        ff[:] = 0
                        if c.strategy == "adaptive":
                            self._recover(a)
                            self.hold_requested[a] = True
                            self.active_motion_plans[a] = None
                            self.ik.command(d, a, ee, gripper=0.04, hold=True)
                            continue
                    tactile_complete = False
                    if c.strategy == "adaptive":
                        if self._gripper_has_object(a):
                            if np.isnan(self.grasp_contact_since[a]):
                                self.grasp_contact_since[a] = d.time
                        else:
                            self.grasp_contact_since[a] = np.nan
                        tactile_complete = elapsed >= (
                            0.04 if self.suction else 0.12
                        ) and d.time - self.grasp_contact_since[a] >= (
                            0.02 if self.suction else 0.04
                        )
                    if (
                        c.strategy == "adaptive"
                        and elapsed >= self.closure_timeout
                        and not self._gripper_has_object(a)
                    ):
                        # A timed-out closure is not evidence of a grasp.
                        # Reopening here also prevents the empty finger pads
                        # being driven into one another during a failed lift.
                        self._recover(a)
                        grip = 0.04
                        self.hold_requested[a] = True
                        self.active_motion_plans[a] = None
                    elif elapsed >= self.closure_timeout or tactile_complete:
                        self.lift_xy[a] = ee[:2]
                        self.lift_displacement[a] = 0.0
                        self.lift_start_z[a] = (
                            d.site_xpos[self.ids.vacuum_sites[a], 2]
                            if self.suction
                            else ee[2]
                        )
                        self.lift_deceleration_start[a] = np.nan
                        self._set_phase(a, "lift")
                if (
                    p[0] > self.windows[a, 1] + 0.08
                    or elapsed > 4.0
                    or (
                        phase != "close"
                        and d.time - track.last_seen > c.grasp_coast_time
                    )
                ):
                    self._recover(a)
            elif phase == "reorient":
                end = self.reorient_goals[a]
                goal, ready = self._route_goal(a, end, False)
                actual = np.empty(4)
                mujoco.mju_mat2Quat(actual, d.site_xmat[self.ids.ee_sites[a]])
                if (
                    ready
                    and np.linalg.norm(ee - end) < 0.008
                    and 2
                    * math.acos(
                        np.clip(abs(np.dot(actual, downward_quaternion(yaw))), 0, 1)
                    )
                    < 0.04
                ):
                    self._set_phase(a, "approach")
                elif elapsed > 3.0:
                    self._recover(a)
            elif phase == "lift":
                grip = 0
                goal = np.array(
                    [
                        *self.lift_xy[a],
                        0.53 if c.motion_mode == "legacy" else self.lift_height,
                    ]
                )
                if c.strategy == "adaptive":
                    if self.suction:
                        # The object can still touch the moving belt during the
                        # first lift centimetres. Keep following until encoder
                        # FK shows 25 mm of cup rise, then decelerate smoothly.
                        # Do not spend unbounded reach if lift-off is blocked.
                        if np.isnan(self.lift_deceleration_start[a]):
                            cup_rise = (
                                d.site_xpos[self.ids.vacuum_sites[a], 2]
                                - self.lift_start_z[a]
                            )
                            if cup_rise >= 0.025:
                                self.lift_deceleration_start[a] = d.time
                            elif elapsed >= 0.20:
                                self._recover(a)
                                self.hold_requested[a] = True
                                self.active_motion_plans[a] = None
                                self.ik.command(d, a, ee, gripper=0.04, hold=True)
                                continue
                        started = self.lift_deceleration_start[a]
                        u = (
                            0.0
                            if np.isnan(started)
                            else np.clip((d.time - started) / 0.20, 0.0, 1.0)
                        )
                        ff[0] = self._belt_velocity() * (
                            1.0 - 3 * u * u + 2 * u * u * u
                        )
                        self.lift_displacement[a] += ff[0] * c.physics_dt
                    elif elapsed <= 0.20:
                        ff[0] = self._belt_velocity() * max(0.0, 1.0 - elapsed / 0.20)
                        self.lift_displacement[a] += ff[0] * c.physics_dt
                    goal[0] += self.lift_displacement[a]
                else:
                    goal[0] += c.belt_speed * min(elapsed, 0.20)
                end = goal.copy()
                # The payload initially touches the belt; during this vertical
                # departure check robot geometry and the known bin volumes.
                goal, ready = self._route_goal(a, end, False, velocity=ff)
                if self.suction and elapsed > 0.03 and not self._gripper_has_object(a):
                    self._recover(a)
                    self.hold_requested[a] = True
                    self.active_motion_plans[a] = None
                    self.ik.command(d, a, ee, gripper=0.04, hold=True)
                    continue
                if c.strategy == "adaptive" and not ready:
                    self._recover(a)
                    self.hold_requested[a] = True
                    self.active_motion_plans[a] = None
                    self.ik.command(d, a, ee, gripper=0.04, hold=True)
                    continue
                empty_aperture = float(np.sum(d.qpos[self.ids.finger_qpos[a]])) < max(
                    0.008, self.grasp_min_widths[a] - 0.025
                )
                if c.strategy == "adaptive" and elapsed > 0.12 and empty_aperture:
                    self._recover(a)
                    grip = 0.04
                    self.hold_requested[a] = True
                    self.active_motion_plans[a] = None
                if elapsed > 0.20 and self._gripper_has_object(a):
                    self.holding[a] = True
                    self.belief[i] = 3
                    if ready and np.linalg.norm(ee - end) < 0.020:
                        self.lift_xy[a] = ee[:2]
                        self.idle_anchors[a] = ee.copy()
                        self._set_phase(
                            a,
                            "depart" if c.motion_mode == "legacy" else "transfer_wait",
                        )
                elif elapsed > 1.5:
                    self._recover(a)
                if elapsed > 3.0:
                    self._recover(a)
            elif phase == "depart":
                grip = 0
                goal = np.array([self.lift_xy[a, 0], side * 0.26, 0.53])
                if np.linalg.norm(ee - goal) < 0.02:
                    self._set_phase(a, "transfer_wait")
                elif elapsed > 3.0:
                    self._recover(a)
            elif phase == "transfer_wait":
                grip = 0
                other = a ^ 1
                cross = self.drop_kind[i] != a % 2
                staging_ready = True
                if c.motion_mode == "legacy":
                    goal = (
                        self.ids.home_ee[a].copy()
                        if cross
                        else np.array([self.lift_xy[a, 0], side * 0.26, 0.53])
                    )
                else:
                    goal = self.idle_anchors[a].copy()
                    if cross:
                        goal = self.ids.home_ee[a].copy()
                        goal, staging_ready = self._route_goal(
                            a, goal, True, via=[np.array([ee[0], side * 0.18, 0.53])]
                        )
                other_safe = np.linalg.norm(
                    d.site_xpos[self.ids.ee_sites[other]] - self.ids.home_ee[other]
                ) < 0.03 and self.phase[other] in ("idle", "transfer_wait")
                if cross and not other_safe and self.phase[other] == "idle":
                    self.home_returns[other] += 1
                    self._set_phase(other, "yield")
                # A waiting loaded partner parks only when needed for a crossing.
                if (
                    cross
                    and self.phase[other] == "transfer_wait"
                    and not other_safe
                    and self.transfer_owner < 0
                ):
                    goal = self.ids.home_ee[a].copy()
                    if c.motion_mode == "optimized":
                        goal, staging_ready = self._route_goal(
                            a, goal, True, via=[np.array([ee[0], side * 0.18, 0.53])]
                        )
                own_ready = (
                    staging_ready
                    and np.linalg.norm(ee - (self.ids.home_ee[a] if cross else goal))
                    < 0.025
                )
                if (
                    self.transfer_owner < 0
                    and (
                        self.bin_zone_owner[a // 2, self.drop_kind[i]] < 0
                        if self.num_arms == 4
                        else self.bin_owner[self.drop_kind[i]] < 0
                    )
                    and self.bin_service[self.drop_kind[i]] == "closed"
                    and own_ready
                    and (not cross or other_safe)
                ):
                    if cross:
                        self.transfer_owner = a
                    self.bin_owner[self.drop_kind[i]] = a
                    self.bin_zone_owner[a // 2, self.drop_kind[i]] = a
                    self._set_phase(
                        a, "align" if c.motion_mode == "legacy" else "transport"
                    )
                else:
                    self.shared_wait_time += c.physics_dt
                if self.phase[a] == "transfer_wait" and elapsed > 12.0:
                    self._recover(a)
            elif phase == "align":
                grip = 0
                goal = np.array([self.drop_targets[i, 0], side * 0.18, 0.52])
                if np.linalg.norm(ee - goal) < 0.018:
                    self._set_phase(a, "transport")
                elif elapsed > 5.0:
                    self._recover(a)
            elif phase == "transport":
                grip = 0
                end = self.drop_targets[i].copy()
                end[2] = self.transport_height
                goal, ready = self._route_goal(
                    a,
                    end,
                    True,
                    via=[
                        np.array([self.lift_xy[a, 0], self.lift_xy[a, 1], 0.53]),
                        np.array([self.lift_xy[a, 0], side * 0.26, 0.53]),
                        np.array([end[0], side * 0.18, 0.52]),
                    ],
                )
                actual = np.empty(4)
                mujoco.mju_mat2Quat(actual, d.site_xmat[self.ids.ee_sites[a]])
                settled = (
                    np.linalg.norm(ee - end) < 0.008
                    and 2
                    * math.acos(
                        np.clip(
                            abs(np.dot(actual, downward_quaternion(self.grasp_yaw[a]))),
                            0,
                            1,
                        )
                    )
                    < 0.04
                )
                if ready and settled:
                    self._set_phase(a, "place")
                elif elapsed > 7.0:
                    self._recover(a)
            elif phase == "place":
                grip = 0
                goal = self.drop_targets[i].copy()
                goal[2] = self.deposit_height
                end = goal.copy()
                goal, ready = self._route_goal(a, end, True)
                settled = (
                    ready
                    and np.linalg.norm(ee - end) < 0.012
                    and np.linalg.norm(d.qvel[self.ids.arm_dofs[a]]) < 0.3
                )
                if settled and c.direct_bin_release:
                    # An inner-bin target has |y|=.235, only 5 mm beyond the
                    # station exclusion strip. With no downward place stroke,
                    # a 12 mm pose tolerance could freeze the hand at |y|=.229
                    # for all of release, blocking its partner. Finish the
                    # checked route and physically clear the unchanged strip
                    # before holding the robot and opening the fingers.
                    actual = np.empty(4)
                    mujoco.mju_mat2Quat(actual, d.site_xmat[self.ids.ee_sites[a]])
                    plan = self.routes[a]
                    settled = bool(
                        side * ee[1] >= 0.232
                        and abs(ee[1] - end[1]) <= 0.0025
                        and plan is not None
                        and plan.verified
                        and self.route_index[a] == len(plan.waypoints) - 1
                        and plan.can_advance(
                            int(self.route_index[a]),
                            ee,
                            self.ik.cart_velocities[a],
                            quaternion=actual,
                            joint_positions=d.qpos[self.ids.arm_qpos[a]],
                            acceleration=self.ik.max_cartesian_acceleration,
                        )
                    )
                if settled:
                    self._set_phase(a, "release")
                elif elapsed > 4.0:
                    self._recover(a)
            elif phase == "release":
                goal = self.drop_targets[i].copy()
                goal[2] = self.deposit_height
                unloaded = self._gripper_released(a)
                if unloaded:
                    if np.isnan(self.release_clear_since[a]):
                        self.release_clear_since[a] = d.time
                else:
                    self.release_clear_since[a] = np.nan
                responsive = self._responsive_release_active(a)
                if (
                    elapsed > (0.08 if self.suction or responsive else 0.4)
                    and unloaded
                    and d.time - self.release_clear_since[a]
                    > (0.05 if self.suction else 0.1)
                ):
                    self.belief[i] = 4
                    self.retired_tracks[self.tracks[i].track_id] = d.time
                    self.holding[a] = False
                    self.arm_releases[a] += 1
                    if responsive:
                        self.responsive_release_count += 1
                    self.bin_release_counts[self.drop_kind[i]] += 1
                    self.bin_last_release[self.drop_kind[i]] = d.time
                    self.eval_queue.append(("release", a, i, None))
                    self._set_phase(a, "retreat")
                elif elapsed > 3.0:
                    self.release_timeouts += 1
                    self.events.append(
                        {"time": float(d.time), "event": "release_timeout", "arm": a}
                    )
                    self._recover(a)
            elif phase == "retreat":
                goal = self.drop_targets[i].copy()
                goal[2] = self.transport_height
                end = goal.copy()
                goal, ready = self._route_goal(a, end, False)
                if ready and np.linalg.norm(ee - end) < 0.008:
                    if (
                        c.motion_mode == "legacy"
                        or self.transfer_owner == a
                        or self.num_arms == 4
                    ):
                        # Two same-color arms now share each bin. Parking over
                        # its mouth leaves the idle hand in the other station's
                        # grasp/transport corridor. Retain bin ownership until a
                        # checked return clears this shared volume.
                        self.planned_return_active[a] = self._planned_return_available(a)
                        self.return_goals[a] = self._normal_return_goal(a)
                        if self.planned_return_active[a]:
                            self.planned_return_attempts[a] += 1
                            self.events.append({"time": float(d.time), "event": "planned_return_start",
                                                "arm": a, "goal": self.return_goals[a].tolist()})
                        if np.linalg.norm(self.return_goals[a] - self.ids.home_ee[a]) <= 0.01:
                            self.home_returns[a] += 1
                        self._set_phase(a, "return")
                    else:
                        self.idle_anchors[a] = ee.copy()
                        self.tool_yaw[a] = yaw
                        self._release_resources(a)
                        self.target[a] = -1
                        self._set_phase(a, "idle")
                elif elapsed > 3.0:
                    self._recover(a)
            elif phase in ("return", "recover", "yield"):
                if (
                    phase == "return"
                    and (self.planned_return_active[a] or getattr(c, "cooperative_motion", False))
                    and not self._gripper_released(a)
                ):
                    # New tactile load or inadequate opening invalidates an
                    # empty-hand return; keep the physical scene and recover.
                    self.events.append({"time": float(d.time),
                                        "event": "planned_return_release_lost" if self.planned_return_active[a]
                                        else "cooperative_return_release_lost", "arm": a})
                    self._recover(a)
                    self.hold_requested[a] = True
                    self.active_motion_plans[a] = None
                    self.ik.command(d, a, ee, gripper=0.04, hold=True)
                    continue
                if phase == "recover":
                    yaw = np.pi / 2 if a % 2 == 0 else -np.pi / 2
                    self.grasp_yaw[a] = self.tool_yaw[a] = yaw
                end = (
                    self.return_goals[a].copy()
                    if phase == "return"
                    else self.ids.home_ee[a].copy()
                )
                # Release grasps before recovery; returning empty still checks
                # the changed shoulder geometry and the other robot's encoders.
                withdrawal = self._release_withdrawal_step(a) if phase == "recover" else None
                if withdrawal is not None:
                    goal, ready = withdrawal
                else:
                    goal, ready = self._route_goal(
                        a, end, False, via=[np.array([ee[0], side * 0.18, 0.52])]
                    )
                if phase == "return" and self.planned_return_active[a] and not ready:
                    # The advisory endpoint is optional. Fall back first to
                    # exactly the pre-existing return, with its own full check.
                    self.planned_return_active[a] = False
                    self.planned_return_fallbacks[a] += 1
                    self.return_goals[a] = self._baseline_return_goal(a)
                    self.routes[a] = None
                    self.route_retry_time[a] = d.time
                    self.events.append({"time": float(d.time), "event": "planned_return_fallback",
                                        "arm": a, "goal": self.return_goals[a].tolist()})
                    end = self.return_goals[a].copy()
                    goal, ready = self._route_goal(
                        a, end, False, via=[np.array([ee[0], side * 0.18, 0.52])]
                    )
                short_return = phase == "return" and np.linalg.norm(end - self.ids.home_ee[a]) > 0.01
                if short_return and not ready:
                    # A new ready branch is optional. Revert to the calibrated
                    # home route if its full robot path cannot be certified.
                    self.return_goals[a] = self.ids.home_ee[a].copy()
                    self.routes[a] = None
                    self.route_retry_time[a] = d.time
                    self.direct_retask_fallbacks[a] += 1
                    self.home_returns[a] += 1
                    self.events.append({"time": float(d.time), "event": "direct_retask_fallback", "arm": a})
                    end = self.return_goals[a].copy()
                    goal, ready = self._route_goal(
                        a, end, False, via=[np.array([ee[0], side * 0.18, 0.52])]
                    )
                    short_return = False
                settled = (
                    ready
                    and not self.escape_routes[a]
                    and np.linalg.norm(ee - end) < 0.03
                )
                if settled and c.motion_mode != "legacy":
                    plan = self.routes[a]
                    actual = np.empty(4)
                    mujoco.mju_mat2Quat(actual, d.site_xmat[self.ids.ee_sites[a]])
                    settled = (
                        plan is not None
                        and plan.verified
                        and self.route_index[a] == len(plan.waypoints) - 1
                        and plan.can_advance(
                            int(self.route_index[a]),
                            ee,
                            self.ik.cart_velocities[a],
                            quaternion=actual,
                            joint_positions=d.qpos[self.ids.arm_qpos[a]],
                            acceleration=self.ik.max_cartesian_acceleration,
                        )
                    )
                if settled and phase == "recover":
                    actual = np.empty(4)
                    mujoco.mju_mat2Quat(actual, d.site_xmat[self.ids.ee_sites[a]])
                    settled = (
                        2
                        * math.acos(
                            np.clip(abs(np.dot(actual, downward_quaternion(yaw))), 0, 1)
                        )
                        < 0.04
                    )
                    if (
                        settled
                        and c.motion_mode != "legacy"
                        and np.max(
                            np.abs(d.qpos[self.ids.arm_qpos[a]] - self.ids.home_qpos[a])
                        )
                        > 0.02
                    ):
                        # A Cartesian fallback may have reached the tool pose
                        # while preserving an unusable redundant branch. Finish
                        # that segment, then request a checked calibrated return.
                        settled = False
                        self.routes[a] = None
                        self.route_retry_time[a] = d.time
                        self.active_motion_plans[a] = None
                        self.hold_requested[a] = True
                if settled and short_return:
                    # A path certificate is not evidence that the old payload
                    # was released; preserve its reservation until both hold.
                    settled = bool(not self.holding[a] and self._gripper_released(a))
                if settled:
                    if phase == "recover":
                        self.recovery_home_returns[a] += 1
                    if phase == "return" and self.planned_return_active[a]:
                        self.planned_return_completed[a] += 1
                    self.idle_anchors[a] = ee.copy()
                    if short_return:
                        self.direct_retask_returns[a] += 1
                        self.events.append({"time": float(d.time), "event": "direct_retask_ready", "arm": a})
                    self._release_resources(a)
                    self.target[a] = -1
                    self._set_phase(a, "idle")
                elif phase != "recover" and elapsed > 7.0:
                    self._recover(a)
            if phase in (
                "depart",
                "transfer_wait",
                "align",
                "transport",
                "place",
            ) and not self._gripper_has_object(a):
                if np.isnan(self.empty_since[a]):
                    self.empty_since[a] = d.time
                elif d.time - self.empty_since[a] > 0.15:
                    self._recover(a)
            else:
                self.empty_since[a] = np.nan
            self.tool_yaw[a] = yaw
            self.last_goals[a] = goal
            self.ik.command(
                d,
                a,
                goal,
                target_quat=self.quat_override[a]
                if self.quat_override[a] is not None
                else downward_quaternion(yaw),
                feedforward=ff,
                gripper=0.04 if self.suction else grip,
                motion_plan=self.active_motion_plans[a],
                waypoint_index=int(self.active_motion_indices[a]),
                hold=bool(self.hold_requested[a]),
            )

    def _outcomes(self):
        c, d = self.config, self.data
        for event, a, slot, point in self.eval_queue:
            if event == "dispatch":
                eligible = np.flatnonzero(
                    (self.states == State.BELT) | (self.states == State.RESERVED)
                ).tolist()
                candidates = [
                    (np.linalg.norm(self.cube_position(i) - point), i) for i in eligible
                ]
                self.truth_target[a] = -1
                if candidates:
                    distance, i = min(candidates)
                    if distance < 0.10:
                        self.truth_target[a] = i
                        self.owner[i] = a
                        self.states[i] = State.RESERVED
                        self.object_dispatch_time[i] = d.time
            else:
                i = self.truth_target[a]
                if i < 0:
                    continue
                if event == "release":
                    self.states[i] = State.RELEASED
                    self.release_time[i] = d.time
                elif event == "recover":
                    p = self.cube_position(i)
                    if abs(p[1]) < c.belt_half_width and p[2] < c.belt_top + 0.10:
                        self.states[i] = State.BELT
                        self.owner[i] = -1
                    else:
                        self.states[i] = State.DROPPED
                self.truth_target[a] = -1
        self.eval_queue.clear()
        for i in np.flatnonzero(
            (self.states != State.PENDING)
            & (self.states != State.MISSED)
            & (self.states != State.DROPPED)
        ).tolist():
            state = self.states[i]
            if state in (State.PENDING, State.MISSED, State.DROPPED):
                continue
            p, v = self.cube_position(i), self.cube_velocity(i)
            color = self.types[i]
            center = self.ids.bin_centers[color]
            inside_xy = np.all(np.abs(p[:2] - center[:2]) < self.ids.bin_half_sizes)
            gate_open = (
                self.bin_service[color] != "closed"
                or d.qpos[self.ids.bin_gate_qpos[color]] > 0.005
            )
            if state == State.SORTED:
                # Authorized bottom-gate unloading preserves completed sorting.
                if gate_open and inside_xy:
                    continue
                if not inside_xy or p[2] < self.ids.bin_floor_z - 0.02:
                    self.states[i] = State.DROPPED
                    self.arm_completions[self.owner[i]] -= 1
                    self.shape_sorted[self.ids.object_specs[i].shape] -= 1
                elif p[2] > self.ids.bin_rim_z + 0.04 or np.linalg.norm(v) > 0.30:
                    self.states[i] = State.RELEASED
                    self.arm_completions[self.owner[i]] -= 1
                    self.shape_sorted[self.ids.object_specs[i].shape] -= 1
                    self.cube_settle[i] = 0.0
                    self.release_time[i] = d.time
                continue
            if state in (State.BELT, State.RESERVED) and p[0] > 0.85:
                self.states[i] = State.MISSED
                continue
            minimum_height = (
                self.ids.bin_floor_z - 0.02
                if self.config.deep_bins
                and inside_xy
                and (
                    state == State.RELEASED
                    or (
                        state == State.HELD
                        and self.owner[i] >= 0
                        and self.phase[self.owner[i]] == "release"
                    )
                )
                else 0.08
            )
            if p[2] < minimum_height or (
                state in (State.BELT, State.RESERVED)
                and abs(p[1]) > c.belt_half_width + 0.05
                and p[2] < c.belt_top + 0.01
            ):
                self.states[i] = State.DROPPED
                continue
            if state == State.RESERVED and p[2] > c.belt_top + 0.10:
                self.states[i] = State.HELD
            if state != State.RELEASED:
                continue
            inside = (
                inside_xy
                and self.ids.bin_floor_z - 0.01 < p[2] < self.ids.bin_rim_z + 0.04
                and np.linalg.norm(v) < 0.15
            )
            self.cube_settle[i] = self.cube_settle[i] + c.physics_dt if inside else 0.0
            if self.cube_settle[i] >= c.settle_time:
                self.states[i] = State.SORTED
                self.arm_completions[self.owner[i]] += 1
                self.assignments_at_last_sort[self.owner[i]] = self.arm_assignments[
                    self.owner[i]
                ]
                self.last_sort_time[self.owner[i]] = d.time
                self.shape_sorted[self.ids.object_specs[i].shape] += 1
                self.cycle_times[i] = d.time - self.object_dispatch_time[i]
                self.events.append(
                    {
                        "time": float(d.time),
                        "event": "sorted",
                        "object": int(self.serial[i]),
                        "arm": int(self.owner[i]),
                    }
                )
            elif d.time - self.release_time[i] > 5.0:
                self.states[i] = State.DROPPED
        if getattr(c, "stable_completion_reward", False):
            self._record_completion_rewards()

    def _record_completion_rewards(self):
        """Credit one stable success, revoke only on an actual final loss.

        A stored object may bounce and settle again. Those transient
        SORTED/RELEASED changes remain visible in outcomes but are not fresh
        successes. A later genuine loss cancels its earlier success credit;
        the ordinary lost-object penalty still applies in step().
        """
        good = (self.states == State.SORTED) & (self.serial >= 0)
        new = good & (self.rewarded_completion_serial != self.serial)
        self.first_completion_rewards += int(np.count_nonzero(new))
        self.rewarded_completion_serial[new] = self.serial[new]
        lost = np.isin(self.states, (State.MISSED, State.DROPPED)) & (self.serial >= 0)
        revoke = lost & (self.rewarded_completion_serial == self.serial) & (
            self.revoked_completion_serial != self.serial
        )
        self.revoked_completion_rewards += int(np.count_nonzero(revoke))
        self.revoked_completion_serial[revoke] = self.serial[revoke]

    def _evaluate_vision(self):
        if self.sensor is None:
            return
        for track in self.tracks:
            if (
                track is None
                or not track.visible
                or track.timestamp <= self._eval_last_frame
            ):
                continue
            p = track.predict(self.data.time)
            candidates = [
                (np.linalg.norm(self.cube_position(i) - p), i)
                for i in range(self.config.n_objects)
                if self.states[i] in (State.BELT, State.RESERVED)
                and self.types[i] == track.kind
            ]
            if not candidates:
                continue
            dist, i = min(candidates)
            if dist > 0.08:
                continue
            self.vision_sq_error += float(dist**2)
            self.vision_match_count += 1
            if self.ids.object_specs[i].shape != "cylinder":
                rotation = self.data.xmat[self.ids.cube_bodies[i]].reshape(3, 3)
                yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                period = max(track.yaw_period, np.pi / 2)
                self.vision_yaw_error += float(
                    abs((track.yaw - yaw + period / 2) % period - period / 2)
                )
                self.vision_yaw_count += 1
            serial = int(self.serial[i])
            previous = self.vision_id_matches.get(serial)
            if previous is not None and previous != track.track_id:
                self.vision_id_switches += 1
            self.vision_id_matches[serial] = track.track_id
        timestamps = [t.timestamp for t in self.tracks if t is not None]
        if timestamps:
            self._eval_last_frame = max(timestamps)

    def _collisions(self):
        hit = False
        any_robot_hit = False
        contact_pairs = set()
        for contact in self.data.contact:
            a, b = (
                self.ids.robot_geom_arm[contact.geom1],
                self.ids.robot_geom_arm[contact.geom2],
            )
            if contact.dist >= -0.0005:
                continue
            body_a = self.model.geom_bodyid[contact.geom1]
            body_b = self.model.geom_bodyid[contact.geom2]
            if a >= 0 or b >= 0:
                finger_cube = (
                    body_a in self.ids.finger_bodies and body_b in self.ids.cube_bodies
                ) or (
                    body_b in self.ids.finger_bodies and body_a in self.ids.cube_bodies
                )
                if self.vacuum is not None:
                    finger_cube = (
                        finger_cube
                        or (
                            contact.geom1 in self.ids.vacuum_geoms
                            and body_b in self.ids.cube_bodies
                        )
                        or (
                            contact.geom2 in self.ids.vacuum_geoms
                            and body_a in self.ids.cube_bodies
                        )
                    )
                adjacent = (
                    body_a == body_b
                    or self.model.body_parentid[body_a] == body_b
                    or self.model.body_parentid[body_b] == body_a
                )
                if not finger_cube and not adjacent:
                    any_robot_hit = True
                    pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                    contact_pairs.add(pair)
                    self.robot_contact_pair_penetration[pair] = max(
                        self.robot_contact_pair_penetration[pair], float(-contact.dist)
                    )
            if a >= 0 and b >= 0 and a != b and contact.dist < -0.0005:
                hit = True
                self.max_penetration = max(self.max_penetration, float(-contact.dist))
        self.collision_steps += int(hit)
        self.robot_collision_steps += int(any_robot_hit)
        self.robot_contact_pair_steps.update(contact_pairs)
        for pair in contact_pairs:
            phases = tuple(
                self.phase[int(self.ids.robot_geom_arm[g])]
                if self.ids.robot_geom_arm[g] >= 0 else "environment_or_workpiece"
                for g in pair
            )
            self.robot_contact_pair_phase_steps[(*pair, *phases)] += 1
        return hit

    def step(self, action: int, *, physics_callback=None):
        """Advance one scheduling interval; optionally observe each physics step.

        Rendering callbacks receive the completed physical state without
        changing the decision rate, sensor schedule, or simulation timestep.
        """
        if physics_callback is not None and not callable(physics_callback):
            raise TypeError("physics_callback must be callable")
        if self._done:
            raise RuntimeError("Episode has ended; reset before stepping")
        if (
            not isinstance(action, (int, np.integer))
            or not 0 <= action < self.action_count
        ):
            raise ValueError(f"action must be an integer in [0,{self.wait_action}]")
        obs = self.observe()
        self.cooperation_monitor.sample(self, self.config.decision_dt)
        reward = 0.0
        if not obs["mask"][action]:
            self.invalid_actions += 1
            action = self.wait_action
            reward -= 0.2
        if action == self.wait_action and np.any(obs["mask"][: self.wait_action]):
            reward -= 0.01
        for a, i in enumerate(self.action_choices[int(action)]):
            if i < WAIT:
                self._assign(a, i)
        before_sorted = self._count(State.SORTED)
        before_completion_reward = self.first_completion_rewards - self.revoked_completion_rewards
        before_lost = self._count(State.MISSED) + self._count(State.DROPPED)
        before_retry, before_collisions = self.retries, self.collision_steps
        before_robot_collisions = self.robot_collision_steps
        for _ in range(round(self.config.decision_dt / self.config.physics_dt)):
            self.belt_sample = self.belt_motion.sample(float(self.data.time))
            self.belt_speed_min = min(self.belt_speed_min, self.belt_sample.speed)
            self.belt_speed_max = max(self.belt_speed_max, self.belt_sample.speed)
            self.model.geom_surfacevel[self.ids.belt_geom, 0] = self.belt_sample.speed
            self.belt_markers.update(self.data, distance=self.belt_sample.distance)
            self._recycle()
            self._spawn()
            self._sense()
            old_ee = self.data.site_xpos[self.ids.ee_sites].copy()
            old_q = self.data.qpos[self.ids.arm_qpos].copy()
            if self.vacuum is not None:
                self.data.qfrc_applied[:] = 0.0
            self._control()
            if self.vacuum is not None:
                for arm, phase in enumerate(self.phase):
                    self.vacuum.set_enabled(
                        arm,
                        phase
                        in (
                            "close",
                            "lift",
                            "depart",
                            "transfer_wait",
                            "align",
                            "transport",
                            "place",
                        ),
                    )
                self.data.qfrc_applied[:] += self.vacuum.step(self.data)
            activity_phases = [
                "approach" if self.preposition_active[a] else phase
                for a, phase in enumerate(self.phase)
            ]
            for a, phase in enumerate(self.phase):
                self.phase_seconds["preposition" if self.preposition_active[a] else phase] += self.config.physics_dt
            self.preposition_seconds += self.preposition_active * self.config.physics_dt
            self.utilization.record(
                activity_phases, self.hold_requested, self.config.physics_dt
            )
            mujoco.mj_step(self.model, self.data)
            self.path_distance += np.linalg.norm(
                self.data.site_xpos[self.ids.ee_sites] - old_ee, axis=1
            )
            self.joint_distance += np.abs(
                self.data.qpos[self.ids.arm_qpos] - old_q
            ).sum(axis=1)
            self._outcomes()
            self._collisions()
            self._evaluate_vision()
            active = sum(p not in ("idle", "recover") for p in activity_phases)
            self.overlap_time += self.config.physics_dt * (active >= 2)
            self.active_time += self.config.physics_dt * (active > 0)
            if physics_callback is not None:
                physics_callback(self)
        sorted_count = self._count(State.SORTED)
        lost = self._count(State.MISSED) + self._count(State.DROPPED)
        completion_delta = (
            self.first_completion_rewards - self.revoked_completion_rewards - before_completion_reward
            if getattr(self.config, "stable_completion_reward", False)
            else sorted_count - before_sorted
        )
        reward += (
            5 * completion_delta
            - 3 * (lost - before_lost)
            - 0.2 * (self.retries - before_retry)
        )
        reward -= 0.001 + 0.1 * (self.collision_steps > before_collisions)
        # Environment/self collisions are measured separately from inter-arm
        # contact; both must provide feedback to the high-level scheduler.
        reward -= 0.25 * (self.robot_collision_steps > before_robot_collisions)
        terminal = bool(
            self.config.total_objects is not None
            and self.spawned >= self.config.total_objects
            and sorted_count + lost >= self.spawned
            and all(p == "idle" for p in self.phase)
        )
        invalid = not (
            np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        )
        truncated = bool(self.data.time >= self.config.horizon or invalid)
        if terminal or truncated:
            self._done = True
            self.termination_reason = (
                "invalid_physics"
                if invalid
                else ("finished" if terminal else "time_limit")
            )
        return self.observe(), float(reward), terminal, truncated, self.metrics()

    def _count(self, state):
        return int(self.archived[int(state)] + np.count_nonzero(self.states == state))

    def utilization_report(self):
        """Return occupancy timers without running perception or outcome logic."""
        report = self.utilization.report()
        if getattr(self.config, "safe_preposition", False):
            report["definition"] = "busy = processing/return/waiting/recovery time / all simulation time; processing includes off-belt preposition motion"
            report["processing_note"] += " Includes physically moving off-belt preparation while the dispatch phase remains idle."
        return report

    def metrics(self):
        max_phase_duration = self.max_phase_duration.copy()
        phase_elapsed = self.data.time - self.phase_start
        for arm, phase in enumerate(self.phase):
            max_phase_duration[phase] = max(
                max_phase_duration[phase], float(phase_elapsed[arm])
            )
        count = self._count(State.SORTED)
        missed = self._count(State.MISSED)
        dropped = self._count(State.DROPPED)
        arms = self.archived_arm_color.copy()
        for a in range(self.num_arms):
            for kind in range(2):
                arms[a, kind] += np.count_nonzero(
                    (self.states == State.SORTED)
                    & (self.owner == a)
                    & (self.types == kind)
                )
        cycles = self.archived_cycles_sum + float(
            np.nansum(self.cycle_times[self.states == State.SORTED])
        )
        complete = (
            self.config.total_objects is not None and count == self.config.total_objects
        )
        result = {
            "seed": self.seed,
            "time": float(self.data.time),
            "total_objects": self.spawned,
            "requested_objects": self.config.total_objects,
            "continuous": self.config.total_objects is None,
            "sorted": count,
            "success_rate": count / max(self.spawned, 1),
            "completed_success_rate": count / max(count + missed + dropped, 1),
            "episode_success": complete,
            "collision_free_success": bool(
                complete and self.robot_collision_steps == 0
            ),
            "missed": missed,
            "dropped": dropped,
            "unfinished": self.spawned - count - missed - dropped,
            "arm_sorted": arms.sum(axis=1).tolist(),
            "arm_color_sorted": arms.tolist(),
            "arm_assignments": self.arm_assignments.tolist(),
            "arm_assignments_since_last_sort": (
                self.arm_assignments - self.assignments_at_last_sort
            ).tolist(),
            "max_arm_assignments_without_sort": self.max_assignments_without_sort.tolist(),
            "arm_seconds_since_last_sort": (
                self.data.time - self.last_sort_time
            ).tolist(),
            "arm_releases": self.arm_releases.tolist(),
            "retries": self.retries,
            "inter_arm_collision_steps": self.collision_steps,
            "robot_collision_steps": self.robot_collision_steps,
            "max_inter_arm_penetration": self.max_penetration,
            "invalid_actions": self.invalid_actions,
            "throughput_per_minute": 60 * count / max(self.data.time, 1e-6),
            "overlap_rate": self.overlap_time / max(self.active_time, 1e-6),
            "shared_transfer_wait_seconds": self.shared_wait_time,
            "mean_pick_place_time": cycles / max(count, 1),
            "termination_reason": self.termination_reason,
            "belt_backend": "native_surfacevel",
            "strategy": self.config.strategy,
            "belt_motion": {
                **self.belt_motion.metadata(),
                "current_speed_m_s": self.belt_sample.speed,
                "realized_min_speed": self.belt_speed_min,
                "realized_max_speed": self.belt_speed_max,
                "distance_m": self.belt_sample.distance,
                "mean_speed_m_s": self.belt_motion.sample(
                    float(self.data.time)
                ).distance
                / max(float(self.data.time), 1e-12),
                "control_speed_estimate": self._belt_velocity(),
            },
            "sensor_mode": self.config.sensor_mode,
            "motion_mode": self.config.motion_mode,
            "bin_count": 2,
            "transport_tool_height_m": float(self.transport_height),
            "deposit_tool_height_m": float(self.deposit_height),
            "deposit_tool_clearance_above_rim_m": float(self.deposit_height - self.ids.bin_rim_z),
            "deep_bins": self.config.deep_bins,
            "cooperative_motion": self.config.cooperative_motion,
            "cooperative_drop_cost": self.config.cooperative_drop_cost,
            "downstream_completion": self.config.downstream_completion,
            **self.cooperation_monitor.metrics(),
            "bin_depth_m": self.ids.bin_rim_z - self.ids.bin_floor_z,
            "arm_count": self.num_arms,
            "gripper_mode": self.config.gripper_mode,
            "bin_unloads": self.bin_unloads.tolist(),
            "bin_service": list(self.bin_service),
            "bin_drain_requested": self.bin_drain_requested.tolist(),
            "bin_floor_z": self.ids.bin_floor_z,
            "bin_rim_z": self.ids.bin_rim_z,
            "pool_size": self.config.pool_size,
            "recycled_objects": self.recycled,
            "max_active_objects": self.max_active_objects,
            "max_slot_generations": int(self.generations.max()),
            "supply_blocked_steps": self.supply_blocked_steps,
            "shape_spawned": dict(self.shape_spawned),
            "shape_sorted": dict(self.shape_sorted),
            "contact_material_spawned": dict(self.material_spawned),
            "arrival_intervals": np.diff(self.arrivals).tolist(),
            "arrival_distances": np.diff(self.arrival_distances).tolist(),
            "spawn_yaws": list(self.spawn_yaw_log),
            "spawn_positions": list(self.spawn_position_log),
            "spawn_lanes": list(self.spawn_lane_log),
            "log_window": 512,
            "ee_path_m": self.path_distance.tolist(),
            "joint_travel_rad": self.joint_distance.tolist(),
            "home_returns": self.home_returns.tolist(),
            "direct_retask_returns": self.direct_retask_returns.tolist(),
            "direct_retask_fallbacks": self.direct_retask_fallbacks.tolist(),
            "planned_return": {
                "attempts": self.planned_return_attempts.tolist(),
                "completed": self.planned_return_completed.tolist(),
                "fallbacks": self.planned_return_fallbacks.tolist(),
                "active": self.planned_return_active.tolist(),
            },
            "route_shortcuts": self.route_shortcuts,
            "route_failures": self.route_failures,
            "escape_attempts": self.escape_attempts,
            "escape_successes": self.escape_successes,
            "release_timeouts": self.release_timeouts,
            "preposition": {
                "starts": self.preposition_starts.tolist(),
                "motion_seconds": self.preposition_seconds.tolist(),
                "active": self.preposition_active.tolist(),
                "utilization_category": "processing",
                "partner_constraint": "opposing arm idle/release, no active preparation, measured joint speed <=0.02rad/s and reference tool speed <=0.005m/s for >=0.05s",
            },
            "recovery_withdrawal": {
                "attempts": self.recovery_withdrawal_attempts.tolist(),
                "successes": self.recovery_withdrawal_successes.tolist(),
                "fault": list(self.recovery_withdrawal_fault),
                "active": [goal is not None for goal in self.recovery_withdrawal_goal],
            },
            "gripper_release_status": [self._release_diagnostics(a) for a in range(self.num_arms)],
            "robot_contact_pairs": self._robot_contact_diagnostics(),
            "guarded_descent": {"stops": dict(self.descent_guard_stops)},
            "completion_reward_events": {
                "first_successes": self.first_completion_rewards,
                "revoked_for_final_loss": self.revoked_completion_rewards,
                "enabled": bool(getattr(self.config, "stable_completion_reward", False)),
            },
            "phase_duration_statistics": [
                {
                    phase: {
                        "samples": len(values),
                        "mean_seconds": float(np.mean(values)),
                        "p90_seconds": float(np.quantile(values, 0.9)),
                    }
                    for phase, values in history.items() if values
                }
                for history in self.phase_duration_samples
            ],
            "responsive_release_count": self.responsive_release_count,
            "recovery_waiting_for_release": [
                self.phase[a] == "recover" and not self._gripper_released(a)
                for a in range(self.num_arms)
            ],
            "recovery_home_returns": self.recovery_home_returns.tolist(),
            "phase_seconds": dict(self.phase_seconds),
            "arm_utilization": self.utilization_report(),
            "max_phase_duration_seconds": dict(max_phase_duration),
            "current_phases": list(self.phase),
            "current_phase_elapsed_seconds": phase_elapsed.tolist(),
            "vision_position_rmse_m": math.sqrt(
                self.vision_sq_error / self.vision_match_count
            )
            if self.vision_match_count
            else None,
            "vision_yaw_mae_deg": math.degrees(
                self.vision_yaw_error / self.vision_yaw_count
            )
            if self.vision_yaw_count
            else None,
            "vision_matches": self.vision_match_count,
            "vision_id_switches": self.vision_id_switches,
        }
        if self.supply_source is not None:
            result["supply"] = {
                **self.supply_source.diagnostics(float(self.data.time)),
                "inlet_blocked_lane_steps": dict(self.supply_inlet_blocked_steps),
                "mean_wait_distance_m": self.supply_wait_distance_sum
                / max(self.spawned, 1),
                "max_wait_distance_m": self.supply_max_wait_distance,
                "offered_success_rate": count
                / max(int(self.supply_source.offered.sum()), 1),
            }
        if self.sensor is not None:
            result["sensor"] = self.sensor.diagnostics()
        future_planner = getattr(self, "_future_planner", None)
        if future_planner is not None:
            result["future_planning"] = future_planner.metrics()
        if self.vacuum is not None:
            result["vacuum"] = self.vacuum.diagnostics()
        if self.wrist is not None:
            result["wrist_sensor"] = self.wrist.diagnostics()
            result["wrist_control"] = {
                "policy": "fill_overhead_occlusion",
                "overhead_freshness_budget_seconds": self.config.camera_latency
                + 1.5 / self.config.camera_hz,
                "corrected_seconds": self.wrist_control_seconds.tolist(),
                "max_applied_xy_m": self.wrist_max_applied.tolist(),
                "used_frames": self.wrist_used_frames.tolist(),
            }
        return result

    def close(self):
        if self.sensor is not None:
            self.sensor.close()
        if self.wrist is not None:
            self.wrist.close()
