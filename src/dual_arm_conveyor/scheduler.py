"""Visual scheduling adapter over the unchanged four-arm grasp controller.

Each station dispatches at most one of its two hands. Its 21 options are ten
track slots for the first hand, ten for the second, and WAIT. The 441 station
pairs cover every native action permitted by that existing station exclusion.
Only track estimates, robot encoders, and controller reservations enter the
policy. Rewards and physical execution remain entirely in ``ConveyorEnv``.
"""

from __future__ import annotations

from numbers import Integral
from typing import Callable

import numpy as np

from .forecast import (
    FORECAST_HORIZON, MAX_ACTIVE_WAIT, controller_readiness,
    nominal_tail_seconds, rank_actions, visual_forecast_state,
)
from . import future_plan

TRACK_SLOTS = 10
STATION_OPTIONS = 21
STATION_WAIT = 20
ACTION_COUNT = STATION_OPTIONS**2
WAIT_ACTION = ACTION_COUNT - 1
NATIVE_ACTION_COUNT = 11**4
ACTION_SCHEMA = "two_station_21x21_v1"
OBSERVATION_SCHEMA = "visual_station_candidates_timing_v2"
PHASE_NAMES = (
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
DISPATCH_GROUP_NAMES = {
    0: "wait",
    1: "station0",
    2: "station1",
    3: "both_stations",
}

OPTION_FEATURE_NAMES = (
    "wait",
    "local_arm0",
    "local_arm1",
    "track_present",
    "candidate_feasible",
    "position_x",
    "position_y",
    "height_above_belt",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "acceleration_x",
    "acceleration_y",
    "relative_ee_x",
    "relative_ee_y",
    "relative_ee_z",
    "size_x",
    "size_y",
    "size_z",
    "yaw_sin2",
    "yaw_cos2",
    "yaw_period",
    "observation_age",
    "confidence",
    "uncertainty",
    "visible",
    "color0",
    "color1",
    "unassigned",
    "reserved",
    "reserved_by_selected_arm",
    "candidate_eta",
    "candidate_deadline",
    "candidate_slack",
    "time_to_window_exit",
    "distance_from_window_entry",
    "distance_to_window_exit",
    "nearest_visible_neighbor",
    "nearby_visible_count",
    "estimated_station_occupancy",
    "estimated_arm_occupancy",
    "partner_ready_in",
    "downstream_arm_ready_in",
)
PAIR_FEATURE_NAMES = (
    "both_dispatch",
    "same_track_slot",
    "target_planar_distance",
    "target_longitudinal_difference",
    "same_color",
    "exit_time_difference",
    "sum_candidate_eta",
    "joint_feasible",
)
ARM_FEATURE_NAMES = (
    *(f"phase_{name}" for name in PHASE_NAMES),
    "phase_elapsed",
    "ee_x",
    "ee_y",
    "ee_z",
    *(f"joint_{j}" for j in range(7)),
    *(f"joint_velocity_{j}" for j in range(7)),
    "finger0",
    "finger1",
    "holding",
    "target_present",
    "target_relative_x",
    "target_relative_y",
    "target_relative_z",
    "target_velocity_x",
    "target_velocity_y",
    "target_velocity_z",
    "target_age",
    "estimated_ready_in",
    "estimated_station_clear_in",
)
QUEUE_FEATURE_NAMES = (
    "upstream_count",
    "window_count",
    "unassigned_in_window",
    "reserved_in_window",
    "feasible_count",
    "minimum_exit_time",
    "minimum_feasible_slack",
    "mean_speed",
    "next_unassigned_entry",
    "second_unassigned_entry",
    "minimum_last_start",
    "minimum_ready_slack",
    "unassigned_ahead_count",
)
FEATURE_NAMES = (
    tuple(
        f"station{station}_{name}"
        for station in range(2)
        for name in OPTION_FEATURE_NAMES
    )
    + PAIR_FEATURE_NAMES
)
GLOBAL_FEATURE_NAMES = (
    tuple(f"arm{arm}_{name}" for arm in range(4) for name in ARM_FEATURE_NAMES)
    + tuple(
        f"station{station}_color{color}_{name}"
        for station in range(2)
        for color in range(2)
        for name in QUEUE_FEATURE_NAMES
    )
    + ("observed_belt_speed", "fresh_track_count", "coasting_track_count")
)
FEATURE_DIM = len(FEATURE_NAMES)
GLOBAL_DIM = len(GLOBAL_FEATURE_NAMES)


def _make_action_ids():
    choices = np.full((ACTION_COUNT, 4), TRACK_SLOTS, dtype=np.int64)
    pairs = np.array(np.unravel_index(np.arange(ACTION_COUNT), (21, 21))).T
    for station in range(2):
        active = pairs[:, station] != STATION_WAIT
        options = pairs[active, station]
        choices[np.flatnonzero(active), 2 * station + options // 10] = options % 10
    return np.ravel_multi_index(choices.T, (11, 11, 11, 11)), pairs


_NATIVE_IDS, _STATION_CHOICES = _make_action_ids()
_REVERSE_IDS = np.full(NATIVE_ACTION_COUNT, -1, dtype=np.int64)
_REVERSE_IDS[_NATIVE_IDS] = np.arange(ACTION_COUNT)
for _table in (_NATIVE_IDS, _STATION_CHOICES, _REVERSE_IDS):
    _table.setflags(write=False)


def joint_action_ids() -> np.ndarray:
    """Return an independent 441-entry station-action -> native-action map."""
    return _NATIVE_IDS.copy()


def _action_integer(action, upper):
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral):
        raise ValueError("Action must be an integer")
    if not 0 <= action < upper:
        raise ValueError(f"Action must be in [0, {upper - 1}]")
    return int(action)


def to_native_action(action: int) -> int:
    return int(_NATIVE_IDS[_action_integer(action, ACTION_COUNT)])


def from_native_action(action: int) -> int:
    result = int(_REVERSE_IDS[_action_integer(action, NATIVE_ACTION_COUNT)])
    if result < 0:
        raise ValueError("Native action dispatches both hands of one station")
    return result


def action_metadata(env=None) -> dict:
    if env is not None and future_plan.enabled(env):
        return {
            "schema": future_plan.ACTION_SCHEMA,
            "action_count": future_plan.ACTION_COUNT,
            "wait_action": future_plan.WAIT_ACTION,
            "station_options": future_plan.OPTIONS,
            "station_wait": future_plan.KEEP,
            "station_cancel": future_plan.CANCEL,
            "num_arms": 4, "track_slots": TRACK_SLOTS,
            "encoding": "action = upstream_option * 22 + downstream_option",
            "station_option_encoding": "0..9 first hand next job; 10..19 second hand next job; 20 KEEP; 21 CANCEL",
            "wait_semantics": "KEEP preserves soft plans and dispatches them only if the current physical mask permits; executing grasps continue",
            "reservation_semantics": "track-identity soft plans, cancellable/reassignable; never alter physical target, belief, bin or station ownership",
            "dispatch_group_names": dict(future_plan.GROUP_NAMES),
        }
    return {
        "schema": ACTION_SCHEMA,
        "action_count": ACTION_COUNT,
        "wait_action": WAIT_ACTION,
        "station_options": STATION_OPTIONS,
        "station_wait": STATION_WAIT,
        "num_arms": 4,
        "track_slots": TRACK_SLOTS,
        "encoding": "action = upstream_option * 21 + downstream_option",
        "station_option_encoding": "0..9 first local hand; 10..19 second local hand; 20 WAIT",
        "native_encoding": "base-11 arm0, arm1, arm2, arm3; per-arm WAIT=10",
        "dispatch_group_names": dict(DISPATCH_GROUP_NAMES),
        "wait_semantics": "Do not dispatch a new task; executing grasps continue",
    }


def observation_metadata(env=None) -> dict:
    if env is not None and future_plan.enabled(env):
        metadata = observation_metadata()
        features = list(FEATURE_NAMES) + [f"station{station}_future_{name}"
            for station in range(2) for name in future_plan.EXTRA_OPTION_NAMES]
        globals_ = list(GLOBAL_FEATURE_NAMES) + [f"station{station}_{name}"
            for station in range(2) for name in future_plan.PLAN_GLOBAL_NAMES]
        metadata.update(schema=future_plan.OBSERVATION_SCHEMA,
            feature_dim=len(features), global_dim=len(globals_),
            feature_names=features, global_feature_names=globals_,
            timing_estimates="task-specific transitions from predicted controller task-end poses; completed-phase 70th percentile after 8 samples, nominal cold start",
            future_plans="observable track identity, cancellable soft intent; no future supply or workpiece truth")
        return metadata
    return {
        "schema": OBSERVATION_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "global_dim": GLOBAL_DIM,
        "feature_names": list(FEATURE_NAMES),
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        "phase_names": list(PHASE_NAMES),
        "source": "visual tracks, robot encoders, controller beliefs/reservations, calibrated windows",
        "scales": {
            "x": 1.2,
            "y": 0.2,
            "height": 0.1,
            "velocity": 0.5,
            "acceleration": 1.0,
            "size": 0.1,
            "time": 5.0,
            "age": 1.5,
            "uncertainty": 0.02,
            "joint": float(np.pi),
            "joint_velocity": 2.0,
            "finger": 0.04,
            "count": 10.0,
        },
        "scalar_clip": [-4.0, 4.0],
        "empty_queue_minimum": 0.0,
        "timing_estimates": "nominal phase budgets, visual window arrival/deadline and measured robot return distance; no teacher scores",
        "window_semantics": {
            "candidate_deadline": "controller completion deadline, including any calibrated near-side extension and lateral sweep check",
            "time_to_window_exit": "raw assignment-window exit, unchanged by completion extensions",
            "queue_minimum_last_start": "conservative nominal estimate using the raw assignment-window exit",
            "forecast_assignment": "future assignments must start inside the raw window; completion extension requires all five projected lateral samples through capture + 0.20 s inside the calibrated band",
        },
    }


def _scaled(value, scale=1.0):
    return np.clip(np.asarray(value, dtype=float) / scale, -4.0, 4.0)


def _check_env(env):
    if env.num_arms != 4 or env.action_count != NATIVE_ACTION_COUNT:
        raise ValueError(
            "The station scheduler requires the native four-arm action space"
        )
    if len(env.tracks) != TRACK_SLOTS:
        raise ValueError("The station scheduler requires ten visual track slots")


def encode_observation(env, observation=None) -> dict[str, np.ndarray]:
    """Encode a native observation without reading workpiece simulator truth.

    The supplied native mask is authoritative. In particular, the policy may
    choose WAIT even when a dispatch exists. No heuristic priority score enters
    the features. Queue summaries and task descriptors contain no track IDs or
    slot numbers, so a consistent track-slot permutation only permutes actions.
    """
    _check_env(env)
    native = env.observe() if observation is None else observation
    native_mask = np.asarray(native["mask"], dtype=bool)
    if native_mask.shape != (NATIVE_ACTION_COUNT,):
        raise ValueError("Expected a native mask with 14641 entries")
    mask = native_mask[_NATIVE_IDS].copy()
    if not mask[WAIT_ACTION]:
        raise ValueError("The native controller must preserve all-WAIT")
    now = float(env.data.time)
    ee = np.asarray(env.data.site_xpos[env.ids.ee_sites], dtype=float)
    arm_ready, station_clear = controller_readiness(env)
    full_cycle_tail = nominal_tail_seconds(env)
    nominal_pick = (0.90 + float(getattr(env, "descent_time", 0.45))
                    + float(getattr(env, "closure_timeout", 0.40)))
    present = np.array([track is not None for track in env.tracks])
    positions, velocities = np.zeros((10, 3)), np.zeros((10, 3))
    ages, kinds = np.zeros(10), np.full(10, -1, dtype=int)
    for slot, track in enumerate(env.tracks):
        if track is not None:
            positions[slot] = track.predict(now)
            velocities[slot] = track.predict_velocity(now)
            ages[slot] = max(0.0, now - track.last_seen)
            kinds[slot] = int(track.kind)
    relevant = (
        present & np.isin(env.belief, (1, 2)) & (ages <= env.config.grasp_coast_time)
    )
    fresh = relevant & (ages <= env.config.track_max_age)
    feasible = np.zeros((4, 10), dtype=bool)
    eta, deadline, exits = np.zeros((4, 10)), np.zeros((4, 10)), np.zeros((4, 10))
    options = np.zeros((2, 21, len(OPTION_FEATURE_NAMES)), dtype=np.float32)
    options[:, STATION_WAIT, 0] = 1.0
    for arm in range(4):
        low, high = env.windows[arm]
        for slot, track in enumerate(env.tracks):
            row = options[arm // 2, (arm % 2) * 10 + slot]
            row[1 + arm % 2] = 1.0
            if track is None:
                continue
            ok, travel, remaining = env._candidate(arm, slot)
            feasible[arm, slot], eta[arm, slot], deadline[arm, slot] = (
                ok,
                travel,
                remaining,
            )
            p, v = positions[slot], velocities[slot]
            exit_time = (high - p[0]) / max(float(v[0]), 0.04)
            exits[arm, slot] = exit_time
            neighbours = np.flatnonzero(relevant & (np.arange(10) != slot))
            distances = np.linalg.norm(positions[neighbours, :2] - p[:2], axis=1)
            reserved_by_arm = int(env.target[arm]) == slot
            row[:] = np.r_[
                0.0,
                float(arm % 2 == 0),
                float(arm % 2 == 1),
                1.0,
                float(ok),
                _scaled(p[0], 1.2),
                _scaled(p[1], 0.2),
                _scaled(p[2] - env.config.belt_top, 0.1),
                _scaled(v, 0.5),
                _scaled(track.acceleration[:2]),
                _scaled(p - ee[arm]),
                _scaled(track.size, 0.1),
                np.sin(2 * track.yaw),
                np.cos(2 * track.yaw),
                _scaled(track.yaw_period, np.pi),
                _scaled(ages[slot], 1.5),
                _scaled(track.confidence),
                _scaled(track.uncertainty, 0.02),
                float(track.visible),
                float(track.kind == 0),
                float(track.kind == 1),
                float(env.belief[slot] == 1),
                float(env.belief[slot] == 2),
                float(reserved_by_arm),
                _scaled(travel, 5),
                _scaled(remaining, 5),
                _scaled(remaining - travel, 5),
                _scaled(exit_time, 5),
                _scaled(p[0] - low),
                _scaled(high - p[0]),
                _scaled(float(distances.min()) if len(distances) else 0.8, 0.2),
                np.count_nonzero(distances < 0.15) / 10.0,
                _scaled((travel if travel > 0 else nominal_pick) + 0.82, 5),
                _scaled((travel if travel > 0 else nominal_pick) + full_cycle_tail, 5),
                _scaled(arm_ready[arm ^ 1], 5),
                _scaled(arm_ready[arm + 2] if arm < 2 else 0.0, 5),
            ]

    first, second = _STATION_CHOICES.T
    pair = np.zeros((ACTION_COUNT, len(PAIR_FEATURE_NAMES)), dtype=np.float32)
    both = (first != STATION_WAIT) & (second != STATION_WAIT)
    rows = np.flatnonzero(both)
    left_slot, right_slot = first[rows] % 10, second[rows] % 10
    left_arm, right_arm = first[rows] // 10, 2 + second[rows] // 10
    known = present[left_slot] & present[right_slot]
    rows = rows[known]
    left_slot, right_slot = left_slot[known], right_slot[known]
    left_arm, right_arm = left_arm[known], right_arm[known]
    delta = positions[right_slot] - positions[left_slot]
    pair[:, 0] = both
    pair[rows, 1] = left_slot == right_slot
    pair[rows, 2] = _scaled(np.linalg.norm(delta[:, :2], axis=1), 1.2)
    pair[rows, 3] = _scaled(delta[:, 0], 1.2)
    pair[rows, 4] = kinds[left_slot] == kinds[right_slot]
    pair[rows, 5] = _scaled(
        exits[right_arm, right_slot] - exits[left_arm, left_slot], 5
    )
    pair[rows, 6] = _scaled(eta[left_arm, left_slot] + eta[right_arm, right_slot], 5)
    pair[:, -1] = mask
    features = np.concatenate((options[0, first], options[1, second], pair), axis=1)

    global_parts = []
    for arm in range(4):
        phase = np.zeros(len(PHASE_NAMES))
        if env.phase[arm] not in PHASE_NAMES:
            raise ValueError(f"Unknown controller phase: {env.phase[arm]}")
        phase[PHASE_NAMES.index(env.phase[arm])] = 1.0
        target_slot = int(env.target[arm])
        has_target = 0 <= target_slot < 10 and present[target_slot]
        global_parts.append(
            np.r_[
                phase,
                _scaled(max(0, now - env.phase_start[arm]), 5),
                _scaled(ee[arm]),
                _scaled(env.data.qpos[env.ids.arm_qpos[arm]], np.pi),
                _scaled(env.data.qvel[env.ids.arm_dofs[arm]], 2),
                _scaled(env.data.qpos[env.ids.finger_qpos[arm]], 0.04),
                float(env.holding[arm]),
                float(has_target),
                _scaled(positions[target_slot] - ee[arm])
                if has_target
                else np.zeros(3),
                _scaled(velocities[target_slot], 0.5) if has_target else np.zeros(3),
                _scaled(ages[target_slot], 1.5) if has_target else 0.0,
                _scaled(arm_ready[arm], 5),
                _scaled(station_clear[arm // 2], 5),
            ]
        )
    for station in range(2):
        for color in range(2):
            arm = 2 * station + color
            low, high = env.windows[arm]
            color_tracks = relevant & (kinds == color)
            window = color_tracks & (positions[:, 0] >= low) & (positions[:, 0] <= high)
            ready = window & feasible[arm]
            pending = (fresh & (kinds == color) & (env.belief == 1)
                       & (positions[:, 0] <= high))
            entry_times = np.maximum(
                0.0, (low - positions[pending, 0]) / np.maximum(velocities[pending, 0], 0.04)
            )
            ordered_entries = np.sort(entry_times)
            last_starts = exits[arm, pending] - nominal_pick - 0.25
            ready_slacks = last_starts - np.maximum(
                entry_times, max(arm_ready[arm], station_clear[station])
            )
            global_parts.append(
                np.r_[
                    np.count_nonzero(color_tracks & (positions[:, 0] < low)) / 10,
                    np.count_nonzero(window) / 10,
                    np.count_nonzero(window & (env.belief == 1)) / 10,
                    np.count_nonzero(window & (env.belief == 2)) / 10,
                    np.count_nonzero(ready) / 10,
                    _scaled(
                        float(exits[arm, window].min()) if np.any(window) else 0, 5
                    ),
                    _scaled(
                        float((deadline[arm, ready] - eta[arm, ready]).min())
                        if np.any(ready)
                        else 0,
                        5,
                    ),
                    _scaled(
                        float(velocities[window, 0].mean()) if np.any(window) else 0,
                        0.5,
                    ),
                    _scaled(float(ordered_entries[0]) if len(ordered_entries) else 0, 5),
                    _scaled(float(ordered_entries[1]) if len(ordered_entries) > 1 else 0, 5),
                    _scaled(float(last_starts.min()) if len(last_starts) else 0, 5),
                    _scaled(float(ready_slacks.min()) if len(ready_slacks) else 0, 5),
                    np.count_nonzero(pending & (positions[:, 0] < low)) / 10.0,
                ]
            )
    sensor = getattr(env, "sensor", None)
    observed_speed = (
        float(sensor.belt_velocity(now)[0])
        if sensor is not None
        else float(np.median(velocities[relevant, 0]))
        if np.any(relevant)
        else 0.0
    )
    global_parts.append(
        np.r_[
            _scaled(observed_speed, 0.5),
            np.count_nonzero(fresh) / 10,
            np.count_nonzero(relevant & ~fresh) / 10,
        ]
    )
    global_features = np.concatenate(global_parts).astype(np.float32)
    if features.shape != (ACTION_COUNT, FEATURE_DIM) or global_features.shape != (
        GLOBAL_DIM,
    ):
        raise RuntimeError(
            "Scheduler observation schema changed without updating metadata"
        )
    if not (np.isfinite(features).all() and np.isfinite(global_features).all()):
        raise ValueError("Nonfinite visual/controller scheduler observation")
    result = {"features": features, "global": global_features, "mask": mask}
    return future_plan.encode_future_observation(env, result) if future_plan.enabled(env) else result


class SchedulerEnv:
    """PPO-compatible scheduling view; every step delegates one physical step."""

    action_count = ACTION_COUNT
    wait_action = WAIT_ACTION
    dispatch_group_names = DISPATCH_GROUP_NAMES

    def __init__(self, env, teacher_mode="returning_partner_soft_preference"):
        _check_env(env)
        if teacher_mode not in ("returning_partner_soft_preference", "predictive", "sequence"):
            raise ValueError("Unknown scheduler teacher mode")
        self.env = env
        self.teacher_mode = teacher_mode
        self._predictive_wait_started = None
        if future_plan.enabled(env):
            self.action_count = future_plan.ACTION_COUNT
            self.wait_action = future_plan.WAIT_ACTION
            self.dispatch_group_names = dict(future_plan.GROUP_NAMES)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        self._predictive_wait_started = None
        observation = self.env.reset(*args, **kwargs)
        if future_plan.enabled(self.env):
            future_plan.planner_for(self.env).reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        return encode_observation(self.env, observation)

    def observe(self):
        return encode_observation(self.env)

    def step(self, action, **kwargs):
        observation, reward, terminated, truncated, info = self.env.step(
            self.action_to_native(action), **kwargs
        )
        return (
            encode_observation(self.env, observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def heuristic_action(self):
        if future_plan.enabled(self.env):
            return future_plan.planner_for(self.env).teacher_action()
        return from_native_action(self.env.heuristic_action())

    def action_to_native(self, action, observation=None):
        """Single stateful mapping for training, evaluation and live playback."""
        if future_plan.enabled(self.env):
            return future_plan.planner_for(self.env).commit(action, observation)
        return to_native_action(action)

    def metrics(self):
        result = dict(self.env.metrics())
        if future_plan.enabled(self.env):
            result['future_planning'] = future_plan.planner_for(self.env).metrics()
        return result

    def expert_action_groups(self, actions):
        """BC balance groups describe plan edits, never physical grasp success."""
        actions = np.asarray(actions, dtype=np.int64)
        side = future_plan.OPTIONS if future_plan.enabled(self.env) else STATION_OPTIONS
        wait = future_plan.KEEP if future_plan.enabled(self.env) else STATION_WAIT
        return (actions // side != wait).astype(int) + 2 * (actions % side != wait).astype(int)

    def teacher_action(self):
        """Apply the original visual dispatch preference to a private mask copy.

        The student retains every physically feasible choice, including the
        option to dispatch while the teacher waits for a returning partner.
        Native-action argmax preserves the original rule's tie breaking.
        """
        if self.teacher_mode == "sequence":
            return self._sequence_teacher_action()
        if future_plan.enabled(self.env):
            return future_plan.planner_for(self.env).teacher_action()
        if self.teacher_mode == "predictive":
            return self._predictive_teacher_action()
        native = self.env.observe()
        physical_mask = np.asarray(native["mask"], dtype=bool)
        teacher_mask = physical_mask.copy()
        choices = self.env.action_choices
        for arm in range(self.env.num_arms):
            for slot in np.unique(choices[physical_mask, arm]):
                if slot >= TRACK_SLOTS:
                    continue
                feasible, eta, deadline = self.env._candidate(arm, int(slot))
                if feasible and self.env._returning_partner_has_urgent_work(
                    arm, eta, deadline
                ):
                    teacher_mask[choices[:, arm] == slot] = False
        if not teacher_mask[self.env.wait_action]:
            raise RuntimeError("The physical scheduler must always permit WAIT")
        totals = native["per_arm_scores"][
            np.arange(self.env.num_arms)[None, :], choices
        ].sum(axis=1)
        totals[~teacher_mask] = -np.inf
        return from_native_action(int(np.argmax(totals)))

    def teacher_metadata(self):
        if self.teacher_mode == "sequence":
            from dataclasses import asdict
            from .sequence_plan import DEFAULT_CONFIG
            return {
                "mode": "sequence", **asdict(DEFAULT_CONFIG),
                "horizon_seconds": DEFAULT_CONFIG.horizon,
                "inputs": "visible tracks, measured robot/controller state, completed phase durations; no future supply or object truth",
                "mask": "unchanged future intent mask and native dispatch mask",
                "forecast": "bounded global beam with greedy continuation; approximate model, no optimality or physical success guarantee",
                "timing": "per-arm observed phase budgets; upward decision-grid rounding; fixed per-arm predicted ready poses for hypothetical returns",
                "search_scope": "root actions keep the existing 6-second intent mask; continuation forecasts extend to 12 seconds; all legal roots retain greedy scores and a bounded beam explores continuations",
                "current_dispatch": "exact current native mask and candidate deadline-minus-ETA subset priority; future feasibility remains approximate",
            }
        if future_plan.enabled(self.env):
            return {
                "mode": "reversible_future_plan",
                "horizon_seconds": FORECAST_HORIZON,
                "inputs": "visual tracks, measured robot state, controller goals and completed phase durations; no future supply or object truth",
                "mask": "predicted future feasibility for intent, unchanged native physical mask at dispatch",
                "forecast": "approximate event scheduling; not a physics or success guarantee",
                "plan_scope": "one cancellable next task per station; no hard target reservation before actual dispatch",
            }
        return {
            "mode": self.teacher_mode,
            "horizon_seconds": FORECAST_HORIZON if self.teacher_mode == "predictive" else None,
            "maximum_active_wait_seconds": MAX_ACTIVE_WAIT if self.teacher_mode == "predictive" else None,
            "inputs": "visual tracks, robot encoders, controller phases/reservations; no future supply or object truth",
            "mask": "unchanged physical action mask",
            "forecast": "approximate event scheduling; not a physics or success guarantee",
            "window_semantics": "raw assignment windows; controller completion bounds with conservative projected lateral-band check",
        }

    def _sequence_teacher_action(self):
        from .sequence_plan import rank_sequence_actions, sequence_snapshot
        deferred = future_plan.enabled(self.env)
        if deferred:
            planner = future_plan.planner_for(self.env)
            base, proposals = planner.snapshot()
            choices, mask = planner.choices_and_mask(proposals)
            state, tails, clear = sequence_snapshot(self.env, base=base)
            wait = future_plan.WAIT_ACTION
            native = self.env.observe()
            current_choices = self.env.action_choices[np.asarray(native["mask"], dtype=bool)]
            current_slack = {}
            for arm, slot in state.current_eta:
                valid, eta, deadline = self.env._candidate(arm, slot)
                if valid:
                    current_slack[arm, slot] = float(deadline - eta)
        else:
            current_choices = current_slack = None
            native = self.env.observe()
            choices = self.env.action_choices[_NATIVE_IDS]
            mask = np.asarray(native["mask"], dtype=bool)[_NATIVE_IDS].copy()
            state, tails, clear = sequence_snapshot(self.env)
            wait = WAIT_ACTION
        scores, diagnostics = rank_sequence_actions(
            state, choices, mask, deferred=deferred,
            tail_by_arm=tails, clear_by_arm=clear,
            native_choices=current_choices, current_slack=current_slack,
        )
        self._sequence_last_search = diagnostics['search']
        if deferred:
            # Preserve the established tiny KEEP tie-break without suppressing
            # a model-predicted gain of even one captured workpiece.
            for station, plan in enumerate(planner.plans):
                if plan is not None:
                    scores[future_plan.PAIRS[:, station] == future_plan.KEEP] += 0.003
        else:
            now = float(self.env.data.time)
            if (self._predictive_wait_started is not None
                    and now - self._predictive_wait_started >= MAX_ACTIVE_WAIT - 1e-9):
                scores[wait] = -np.inf
        if not np.isfinite(scores).any():
            # Retain a valid action even if a forecast rejects a current root.
            return self.heuristic_action()
        action = int(np.argmax(scores))
        if not deferred:
            if action == wait and np.count_nonzero(mask) > 1:
                if self._predictive_wait_started is None:
                    self._predictive_wait_started = float(self.env.data.time)
            else:
                self._predictive_wait_started = None
        return action

    def _predictive_teacher_action(self):
        native = self.env.observe()
        mask = np.asarray(native["mask"], dtype=bool)[_NATIVE_IDS].copy()
        if not mask[WAIT_ACTION]:
            raise RuntimeError("The physical scheduler must always permit WAIT")
        if np.count_nonzero(mask) == 1:
            self._predictive_wait_started = None
            return WAIT_ACTION
        state = visual_forecast_state(self.env)
        choices = self.env.action_choices[_NATIVE_IDS]
        scores, _ = rank_actions(state, choices, mask)
        now = float(self.env.data.time)
        if (self._predictive_wait_started is not None
                and now - self._predictive_wait_started >= MAX_ACTIVE_WAIT - 1e-9):
            scores[WAIT_ACTION] = -np.inf
        if not np.isfinite(scores).any():
            # A track may become stale between independently produced snapshots.
            # Fall back only within the original physical mask, never bypass it.
            action = self.heuristic_action()
        else:
            action = int(np.argmax(scores))
        if action == WAIT_ACTION:
            if self._predictive_wait_started is None:
                self._predictive_wait_started = now
        else:
            self._predictive_wait_started = None
        return action

    def action_metadata(self):
        return action_metadata(self.env)

    def observation_metadata(self):
        return observation_metadata(self.env)

    def model_policy(self, policy: Callable):
        return model_policy(self.env, policy)


def model_policy(env, policy: Callable) -> Callable:
    """Adapt a learned 441-action callable for native playback/evaluation."""
    native_env = env.env if isinstance(env, SchedulerEnv) else env
    _check_env(native_env)
    adapter = env if isinstance(env, SchedulerEnv) else SchedulerEnv(native_env)

    def choose(observation=None):
        encoded = encode_observation(native_env, observation)
        return adapter.action_to_native(policy(encoded), observation)

    return choose
