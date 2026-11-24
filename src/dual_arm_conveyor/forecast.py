"""Short scheduling forecasts from visual tracks and robot/controller state.

This is an approximate scheduling model, not a second physics simulator. It
never reads workpiece bodies, ground-truth identities, or future supply. Its
predictions only rank actions already accepted by the physical action mask.
Times below are nominal controller phase budgets, not learned success claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


FORECAST_HORIZON = 6.0
MAX_ACTIVE_WAIT = 0.6
NOMINAL_PHASE_SECONDS = {
    "approach": 0.90, "reorient": 0.35, "descend": 0.55,
    "close": 0.25, "lift": 0.52, "depart": 0.30,
    "transfer_wait": 0.05, "align": 0.30, "transport": 0.90,
    "place": 0.18, "release": 0.48, "retreat": 0.14,
    "return": 0.70, "recover": 1.40, "yield": 0.70,
}
NORMAL_PHASES = (
    "approach", "descend", "close", "lift", "transfer_wait",
    "transport", "place", "release", "retreat", "return",
)


def phase_budgets(env) -> dict[str, float]:
    """Nominal durations consistent with enabled controller operations.

    The responsive jaw estimate includes command slew from the 2 mm closed
    half-aperture, not merely 10 mm of extra physical jaw clearance. Its width
    estimate uses visible tracks, so it cannot incorporate a hidden catalogue.
    """
    budget = NOMINAL_PHASE_SECONDS.copy()
    if getattr(env.config, "direct_retask", False):
        budget["return"] = 0.40
    if getattr(env.config, "responsive_release", False):
        now = float(env.data.time)
        widths = [float(np.min(track.size[:2])) for track in env.tracks
                  if track is not None
                  and now - float(track.last_seen) <= env.config.track_max_age
                  and np.isfinite(track.size[:2]).all()]
        width = float(np.median(widths)) if widths else 0.045
        threshold = min(0.075, max(0.008, width) + 0.010)
        target_slew = max(0.0, threshold / 2 - 0.002) / 0.10
        budget["release"] = max(0.08, target_slew + 0.10 + 0.04)
    if getattr(env.config, "direct_bin_release", False):
        # Both checks remain in the real state machine, but no 20 mm down/up
        # stroke is requested. Keep a positive allowance for verified settling.
        budget["place"] = 0.04
        budget["retreat"] = 0.04
    return budget


def nominal_tail_seconds(env) -> float:
    """Time from secured grasp through safe readiness for another grasp."""
    budget = phase_budgets(env)
    return sum(budget[p] for p in
               ("lift", "transfer_wait", "transport", "place", "release", "retreat", "return"))


def observed_phase_budgets(env, arm: int) -> dict[str, float]:
    """Use completed controller durations, with a documented cold-start model.

    Observations concern robot execution, never object truth or future supply.
    Eight completed samples are required before replacing a nominal duration.
    The 70th percentile retains an allowance for ordinary execution variation;
    this is an estimate, not a safety or success guarantee.
    """
    budget = phase_budgets(env)
    if not getattr(env.config, "future_planning", False):
        return budget
    samples = getattr(env, "phase_duration_samples", None)
    if samples is None:
        return budget
    for phase, values in samples[arm].items():
        if phase not in budget:
            continue
        values = np.asarray(list(values)[-128:], dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if len(values) >= 8:
            budget[phase] = max(0.02, float(np.quantile(values, 0.70)))
    return budget


def _planned_ready_pose(env, arm: int, fallback: np.ndarray) -> np.ndarray:
    """Read the controller's pure intended endpoint, without re-planning jobs."""
    goal_method = getattr(env, "_planned_ready_goal", None)
    if callable(goal_method):
        return np.asarray(goal_method(arm), dtype=float).copy()
    # Observable-only fallback for lightweight adapters and contract fixtures.
    bases = getattr(env.ids, "arm_base_positions", None)
    base_x = float(bases[arm, 0]) if bases is not None else float(fallback[0])
    slot = int(env.target[arm])
    drops = getattr(env, "drop_targets", None)
    x = float(drops[slot, 0]) if drops is not None and 0 <= slot < len(drops) else base_x
    proposals = getattr(env, "future_preposition_goals", None)
    if proposals is not None and np.shape(proposals) == (4, 3) and np.isfinite(proposals[arm]).all():
        x = float(proposals[arm, 0])
    return np.array([
        np.clip(x, base_x - 0.18, base_x + 0.18),
        -0.26 if arm % 2 == 0 else 0.26,
        max(float(getattr(env, "lift_height", fallback[2])),
            float(getattr(env.ids, "bin_rim_z", 0.0)) + 0.12),
    ])


def predicted_ready_poses(env, ee=None) -> np.ndarray:
    """Predict task-end hand poses from controller-owned drop/return goals."""
    if ee is None:
        ee = np.asarray(env.data.site_xpos[env.ids.ee_sites], dtype=float)
    poses = np.asarray(getattr(env.ids, "home_ee", ee), dtype=float).copy()
    if not getattr(env.config, "direct_retask", False):
        return poses
    returns = np.asarray(getattr(env, "return_goals", poses), dtype=float)
    for arm, phase in enumerate(env.phase):
        if phase == "idle":
            poses[arm] = ee[arm]
        elif phase == "return":
            poses[arm] = returns[arm]
        elif phase in ("recover", "yield"):
            # Both controller branches target calibrated home, even when a
            # previous planned return left a different cached return_goals.
            poses[arm] = (
                np.asarray(getattr(env.ids, "home_ee", ee), dtype=float)[arm]
                if (getattr(env.config, "future_planning", False)
                    or getattr(env.config, "cooperative_motion", False)) else returns[arm]
            )
        elif getattr(env.config, "planned_return", False):
            poses[arm] = _planned_ready_pose(env, arm, poses[arm])
        elif (getattr(env.config, "future_planning", False)
              or getattr(env.config, "cooperative_motion", False)):
            slot = int(env.target[arm])
            targets = getattr(env, "drop_targets", None)
            kinds = getattr(env, "drop_kind", None)
            bases = getattr(env.ids, "arm_base_positions", None)
            if (slot >= 0 and targets is not None and kinds is not None
                    and bases is not None and int(kinds[slot]) == arm % 2):
                poses[arm, 0] = np.clip(targets[slot, 0], bases[arm, 0] - 0.18,
                                       bases[arm, 0] + 0.18)
                if getattr(env.config, "cooperative_motion", False):
                    poses[arm, 1] = (-1 if arm % 2 == 0 else 1) * 0.26
                poses[arm, 2] = max(float(getattr(env, "lift_height", poses[arm, 2])),
                                    float(getattr(env.ids, "bin_rim_z", 0.0)) + 0.12)
        else:
            poses[arm] = returns[arm]
    return poses


def _responsive_release_remaining(env, arm, elapsed):
    starts = getattr(env, "release_start_aperture", None)
    if starts is None or not np.isfinite(starts[arm]):
        return None
    threshold = min(0.075, float(starts[arm]) + 0.010)
    aperture = float(np.sum(env.data.qpos[env.ids.finger_qpos[arm]]))
    targets = getattr(getattr(env, "ik", None), "gripper_targets", None)
    target = float(targets[arm]) if targets is not None else aperture / 2
    remaining_slew = max(0.0, threshold / 2 - target) / 0.10
    clear_since = getattr(env, "release_clear_since", None)
    clear_age = (max(0.0, float(env.data.time) - float(clear_since[arm]))
                 if clear_since is not None and np.isfinite(clear_since[arm]) else 0.0)
    hold = max(0.0, 0.10 - clear_age)
    servo = 0.04 if aperture <= threshold else 0.0
    return max(0.01, 0.08 - elapsed, remaining_slew + servo + hold)


def controller_readiness(env) -> tuple[np.ndarray, np.ndarray]:
    """Estimated arm-idle and station-clear times relative to the current tick.

    Returned arrays do not change controller phases or reservations. An overdue
    phase retains a small positive remainder; it never becomes magically idle.
    Measured distance to the controller's return goal refines return estimates.
    """
    now = float(env.data.time)
    ee = np.asarray(env.data.site_xpos[env.ids.ee_sites], dtype=float)
    speed = max(0.1, 0.85 * float(getattr(env.config, "control_speed", 0.65)))
    arm_ready, station_ready = np.zeros(4), np.zeros(2)
    return_goals = getattr(env, "return_goals", getattr(env.ids, "home_ee", ee))
    planned_poses = predicted_ready_poses(env, ee) if (
        getattr(env.config, "planned_return", False)
        or getattr(env.config, "cooperative_motion", False)
    ) else None
    for arm, phase in enumerate(env.phase):
        if phase == "idle":
            continue
        budget = observed_phase_budgets(env, arm)
        if planned_poses is not None and phase not in ("return", "recover", "yield"):
            slot = int(env.target[arm])
            drops = getattr(env, "drop_targets", None)
            if drops is not None and 0 <= slot < len(drops):
                # The retreat ends above the selected bin location; use that
                # actual controller endpoint rather than a fixed return cost
                # when a next-job hint or an off-belt return changes the
                # intended ready position.
                origin = np.asarray(drops[slot], dtype=float).copy()
                origin[2] = float(getattr(env, "transport_height", origin[2]))
                budget["return"] = max(0.12, float(np.linalg.norm(planned_poses[arm] - origin)) / speed + 0.20)
        elapsed = max(0.0, now - float(env.phase_start[arm]))
        phase_duration = budget.get(phase, 1.0)
        current = max(min(0.08, phase_duration), phase_duration - elapsed)
        if phase == "release" and getattr(env.config, "responsive_release", False):
            measured = _responsive_release_remaining(env, arm, elapsed)
            if measured is not None:
                current = measured
        if phase in ("return", "recover", "yield"):
            motion_goal = np.asarray(return_goals)[arm]
            if (getattr(env.config, "future_planning", False)
                    or getattr(env.config, "cooperative_motion", False)) and phase in ("recover", "yield"):
                motion_goal = np.asarray(getattr(env.ids, "home_ee", ee))[arm]
            current = max(0.12, float(np.linalg.norm(motion_goal - ee[arm])) / speed + 0.20)
            if (getattr(env.config, "future_planning", False) and phase == "recover"
                    and elapsed > 2.0 and callable(getattr(env, "_gripper_released", None))
                    and not env._gripper_released(arm)):
                # An unconfirmed release cannot honestly be forecast as ready
                # in a few tenths of a second throughout a long recovery.
                current = max(current, FORECAST_HORIZON + 1.0)
            arm_ready[arm] = current
        else:
            if phase in NORMAL_PHASES:
                following = NORMAL_PHASES[NORMAL_PHASES.index(phase) + 1:]
            elif phase == "reorient":
                following = NORMAL_PHASES[1:]
            elif phase == "depart":
                following = NORMAL_PHASES[4:]
            elif phase == "align":
                following = NORMAL_PHASES[5:]
            else:
                following = ()
            arm_ready[arm] = current + sum(budget[p] for p in following)
        if phase in ("approach", "reorient", "descend", "close", "lift"):
            clear_tail = {
                "approach": ("descend", "close", "lift"),
                "reorient": ("descend", "close", "lift"),
                "descend": ("close", "lift"), "close": ("lift",), "lift": (),
            }[phase]
            clear = current + sum(budget[p] for p in clear_tail) + 0.30
        elif abs(ee[arm, 1]) < 0.23:
            # A home/fallback return ends inside the exclusion strip and must
            # settle to idle. An off-belt return may physically clear sooner;
            # this remains only a forecast, never native dispatch permission.
            returning_outside = bool(
                phase == "return" and getattr(env.config, "cooperative_motion", False)
                and (-1 if arm % 2 == 0 else 1) * float(np.asarray(return_goals)[arm, 1]) >= 0.26
            )
            clear = arm_ready[arm] if phase in ("return", "recover", "yield") and not returning_outside else min(
                arm_ready[arm], max(0.10, (0.24 - abs(ee[arm, 1])) / speed + 0.15)
            )
        else:
            clear = 0.0
        station_ready[arm // 2] = max(station_ready[arm // 2], clear)
    return arm_ready, station_ready


@dataclass(frozen=True)
class ForecastJob:
    slot: int
    kind: int
    position: np.ndarray
    velocity: np.ndarray
    approach_z: float
    eligible: tuple[bool, bool]
    # Completion bounds reported by the controller for this visual target.
    # Raw state.windows still governs where a future assignment may start.
    completion_high: tuple[float, float] | None = None
    # Observable per-arm retry clocks, relative to this forecast snapshot.
    # Defaults preserve legacy/manual forecast construction. Production capture
    # is enabled for the future-planning controller only.
    available_after: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class BeltReservation:
    station: int
    start: float
    clear: float
    x: float


@dataclass(frozen=True)
class ForecastState:
    jobs: tuple[ForecastJob, ...]
    arm_ready: np.ndarray
    station_ready: np.ndarray
    ee: np.ndarray
    ready_poses: np.ndarray
    windows: np.ndarray
    speed: float
    acceleration: float
    descent_and_close: float
    tail: float
    current_eta: dict[tuple[int, int], float]
    decision_dt: float = 0.1
    horizon: float = FORECAST_HORIZON
    reservations: tuple[BeltReservation, ...] = ()


def visual_forecast_state(env, *, horizon=FORECAST_HORIZON) -> ForecastState:
    """Snapshot only observable tracks, robot encoders and controller resources."""
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Forecast horizon must be finite and positive")
    now = float(env.data.time)
    ee = np.asarray(env.data.site_xpos[env.ids.ee_sites], dtype=float).copy()
    ready, clear = controller_readiness(env)
    if getattr(env.config, "future_planning", False):
        homes = predicted_ready_poses(env, ee)
    else:
        homes = np.asarray(getattr(env.ids, "home_ee", ee), dtype=float).copy()
        if getattr(env.config, "direct_retask", False):
            homes = np.asarray(getattr(env, "return_goals", homes), dtype=float).copy()
    jobs, eta = [], {}
    for slot, track in enumerate(env.tracks):
        if (track is None or int(env.belief[slot]) != 1
                or now - float(track.last_seen) > env.config.track_max_age
                or float(track.confidence) < 0.35 or int(track.kind) not in (0, 1)):
            continue
        p = np.asarray(track.predict(now), dtype=float).copy()
        velocity = np.asarray(track.predict_velocity(now), dtype=float).copy()
        if not np.isfinite(p).all() or not np.isfinite(velocity).all() or abs(p[1]) > 0.11:
            continue
        velocity[:2] = np.clip(velocity[:2], -0.6, 0.6)
        velocity[2] = 0.0
        kind, eligible = int(track.kind), []
        for station in range(2):
            arm = 2 * station + kind
            allowed = True
            if hasattr(env, "bin_service"):
                allowed &= env.bin_service[kind] == "closed"
            if hasattr(env, "bin_drain_requested"):
                allowed &= not bool(env.bin_drain_requested[kind])
            if hasattr(env, "bin_occupied"):
                per_zone = env.bin_occupied.shape[1] // 2
                allowed &= bool(np.any(~env.bin_occupied[kind, station * per_zone:(station + 1) * per_zone]))
            angles = getattr(env, "_grasp_angles", None)
            if callable(angles):
                candidates = angles(arm, track)
                clear_grasp = getattr(env, "_finger_neighborhood_clear", None)
                allowed &= bool(candidates) and (
                    not callable(clear_grasp) or any(clear_grasp(arm, slot, yaw) for yaw in candidates)
                )
            eligible.append(bool(allowed))
            valid, travel, _ = env._candidate(arm, slot)
            if valid:
                eta[arm, slot] = float(travel)
        half_height = float(getattr(track, "half_height", track.size[2] / 2))
        offset = getattr(env, "_grasp_offset", None)
        offset = float(offset(track)) if callable(offset) else -0.009
        approach_z = float(env.config.belt_top + half_height + offset
                           + getattr(env, "approach_clearance", 0.13))
        completion = getattr(env, "completion_window_high", None)
        completion_high = tuple(
            float(completion(2 * station + kind, p)) if callable(completion)
            else float(env.windows[2 * station + kind, 1])
            for station in range(2)
        )
        retry_after = getattr(env, "retry_after", None)
        availability = tuple(
            max(0.0, float(retry_after[arm, slot]) - now)
            if getattr(env.config, "future_planning", False) and retry_after is not None
            else 0.0 for arm in range(4)
        )
        jobs.append(ForecastJob(slot, kind, p, velocity, approach_z,
                                tuple(eligible), completion_high, availability))
    ik = getattr(env, "ik", None)
    reservations = tuple(
        BeltReservation(arm // 2, 0.0, float(clear[arm // 2]), float(ee[arm, 0]))
        for arm in range(4)
        if int(env.target[arm]) >= 0 and clear[arm // 2] > 0
        and (env.phase[arm] in ("approach", "descend", "close", "lift", "reorient")
             or abs(ee[arm, 1]) < 0.23)
    )
    return ForecastState(
        tuple(jobs), ready, clear, ee, homes, np.asarray(env.windows).copy(),
        0.85 * float(getattr(env.config, "control_speed", 0.65)),
        float(getattr(ik, "max_cartesian_acceleration", 2.0)),
        float(getattr(env, "descent_time", 0.45) + getattr(env, "closure_timeout", 0.40)),
        nominal_tail_seconds(env), eta,
        float(getattr(env.config, "decision_dt", 0.1)), float(horizon), reservations,
    )


def pickup_eta(state: ForecastState, job: ForecastJob, arm: int, start: float) -> float:
    """Predict moving rendezvous with acceleration and finger-closure budget."""
    if start <= 1e-8 and (arm, job.slot) in state.current_eta:
        return state.current_eta[arm, job.slot]
    origin = state.ee[arm] if start <= 1e-8 and state.arm_ready[arm] <= 1e-8 else state.ready_poses[arm]
    target = job.position + job.velocity * start
    target = target.copy()
    target[2] = job.approach_z
    velocity = job.velocity.copy()
    delta = target - origin
    speed = max(0.05, state.speed)
    acceleration = max(0.05, state.acceleration)
    quadratic = float(np.dot(velocity, velocity) - speed * speed)
    linear, constant = float(2 * np.dot(delta, velocity)), float(np.dot(delta, delta))
    discriminant = linear * linear - 4 * quadratic * constant
    if quadratic >= -1e-8 or discriminant < 0:
        return math.inf
    duration = max(0.0, (-linear - math.sqrt(discriminant)) / (2 * quadratic))
    for _ in range(6):
        distance = float(np.linalg.norm(delta + velocity * duration))
        updated = (2 * math.sqrt(distance / acceleration)
                   if distance <= speed * speed / acceleration
                   else distance / speed + speed / acceleration)
        if abs(updated - duration) < 1e-4:
            break
        duration = updated
    return duration + state.descent_and_close


def _window_times(state, job, station):
    """Raw assignment entry/exit; never the extended completion envelope."""
    low, high = state.windows[2 * station + job.kind]
    speed = max(0.04, float(job.velocity[0]))
    return max(0.0, (low - job.position[0]) / speed), (high - job.position[0]) / speed


def _completion_exit_time(state, job, arm, start, captured):
    raw_high = float(state.windows[arm, 1])
    high = raw_high if job.completion_high is None else float(job.completion_high[arm // 2])
    if high > raw_high:
        # Extension is granted only while all five projected points remain in
        # the calibrated near-side band, including 0.20 s after grasp capture.
        # Forecast positions use this model's visible constant-velocity motion;
        # they are neither simulator truth nor future camera observations.
        times = np.linspace(start, captured + 0.20, 5)
        side = -1 if arm % 2 == 0 else 1
        near_y = side * (job.position[1] + job.velocity[1] * times)
        # completion_high comes from the controller's opt-in calibrated
        # envelope. Do not silently remove a downstream grant here: native
        # admission and the learned scheduler must use the same deadline.
        if not np.all((near_y >= 0.045) & (near_y <= 0.080)):
            high = raw_high
    return (high - float(job.position[0])) / max(0.04, float(job.velocity[0]))


def _reserved_pickup(state, job, arm, start, reservations):
    """Delay close cross-station forecasts until the reserved belt span clears.

    This uses the controller mask's 0.32 m longitudinal separation as a cheap
    scheduling approximation. It does not replace robot collision checking.
    Reservations use measured x for ongoing tasks and predicted grasp x for
    hypothetical tasks; time only advances, giving a bounded event calculation.
    """
    start = max(float(start), float(job.available_after[arm]))
    for _ in range(len(reservations) + 1):
        eta = pickup_eta(state, job, arm, start)
        if not math.isfinite(eta) or start > state.horizon:
            return start, eta
        captured = start + eta
        x = float(job.position[0] + job.velocity[0] * captured)
        conflicting = [reservation.clear for reservation in reservations
                       if reservation.station != arm // 2
                       and start < reservation.clear
                       and captured + 0.82 > reservation.start
                       and abs(x - reservation.x) < 0.32]
        if not conflicting:
            return start, eta
        start = max(start, max(conflicting))
    # Every iteration passes at least one reservation's end, so this is only a
    # defensive bound against malformed/nonfinite externally constructed data.
    return start, math.inf


def _forecast_action(state: ForecastState, choices: np.ndarray, *, deferred=False) -> dict:
    ready, clear = state.arm_ready.copy(), state.station_ready.copy()
    remaining = {job.slot: job for job in state.jobs}
    captures, root_dispatches = [], 0
    reservations = list(state.reservations)
    # Current dispatches are authoritative: the mask has already checked them.
    for arm, slot in enumerate(choices):
        if slot >= 10:
            continue
        job = remaining.pop(int(slot), None)
        if job is None:
            return {"score": -math.inf, "rescued": 0, "dispatches": 0, "plan": []}
        start = 0.0
        if deferred:
            entry, raw_exit = _window_times(state, job, arm // 2)
            start = max(float(ready[arm]), float(clear[arm // 2]), entry,
                        float(job.available_after[arm]))
            start, eta = _reserved_pickup(state, job, arm, start, reservations)
            captured = start + eta
            if (not job.eligible[arm // 2] or job.kind != arm % 2
                    or start > raw_exit or captured > state.horizon
                    or captured + 0.25 > _completion_exit_time(state, job, arm, start, captured)):
                return {"score": -math.inf, "rescued": 0, "dispatches": 0, "plan": []}
        else:
            if job.available_after[arm] > 1e-9:
                # An immediate action cannot be silently reinterpreted as a
                # future retry. Deferred roots have explicit plan semantics.
                return {"score": -math.inf, "rescued": 0, "dispatches": 0, "plan": []}
            eta = pickup_eta(state, job, arm, start)
            captured = eta
        ready[arm] = captured + state.tail
        clear[arm // 2] = max(clear[arm // 2], captured + 0.82)
        captures.append((arm, job.slot, start, captured))
        reservations.append(BeltReservation(
            arm // 2, start, captured + 0.82,
            float(job.position[0] + job.velocity[0] * captured),
        ))
        root_dispatches += 1

    floor = state.decision_dt
    if root_dispatches == 0:
        # WAIT evaluates the next genuine scheduling event, not a fictitious
        # perpetual 0.1 s postponement of exactly the same greedy assignment.
        events = [float(t) for t in np.r_[ready, clear] if floor <= t <= MAX_ACTIVE_WAIT]
        events.extend(_window_times(state, job, station)[0]
                      for job in remaining.values() for station in range(2)
                      if floor <= _window_times(state, job, station)[0] <= MAX_ACTIVE_WAIT)
        floor = min(events) if events else floor

    for _ in range(len(remaining)):
        options = []
        for job in remaining.values():
            for station in range(2):
                if not job.eligible[station]:
                    continue
                arm = 2 * station + job.kind
                entry, raw_exit_time = _window_times(state, job, station)
                start = max(floor, ready[arm], clear[station], entry,
                            float(job.available_after[arm]))
                if start > min(state.horizon, raw_exit_time):
                    continue
                start, eta = _reserved_pickup(state, job, arm, start, reservations)
                # Reservations may move the start outside raw admission even
                # if completing inside the extended envelope seems possible.
                start_x = float(job.position[0] + job.velocity[0] * start)
                low, raw_high = state.windows[arm]
                if (not low - 1e-9 <= start_x <= raw_high + 1e-9
                        or start > state.horizon or not math.isfinite(eta)):
                    continue
                captured = start + eta
                exit_time = _completion_exit_time(state, job, arm, start, captured)
                if captured > min(exit_time - 0.25, state.horizon):
                    continue
                slack = exit_time - captured
                options.append((start, slack, -station, job.slot, arm, eta))
        if not options:
            break
        # At each predicted event, prioritize the expiring work. Upstream work
        # may instead survive at the second station when its local arm is busy.
        earliest = min(item[0] for item in options)
        selected = min((item for item in options if item[0] <= earliest + 0.025),
                       key=lambda item: (item[1], item[2], item[3], item[4]))
        start, _, _, slot, arm, eta = selected
        job = remaining.pop(slot)
        captured = start + eta
        captures.append((arm, slot, start, captured))
        reservations.append(BeltReservation(
            arm // 2, start, captured + 0.82,
            float(job.position[0] + job.velocity[0] * captured),
        ))
        ready[arm] = captured + state.tail
        clear[arm // 2] = captured + 0.82

    # One additional predicted capture dominates all timing tie breakers. The
    # tiny root-dispatch preference prevents equal-outcome repeated waiting.
    score = len(captures) - 0.002 * sum(c[3] for c in captures) + 0.002 * root_dispatches
    if root_dispatches == 0:
        score -= 0.02
    return {"score": float(score), "rescued": len(captures),
            "dispatches": root_dispatches, "plan": captures}


def rank_actions(state: ForecastState, action_choices: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Rank only accepted actions; return inspectable predicted event sequences."""
    choices, mask = np.asarray(action_choices), np.asarray(mask, dtype=bool)
    if choices.ndim != 2 or choices.shape != (len(mask), 4) or not mask.any():
        raise ValueError("Expected four-arm choices and a nonempty matching mask")
    scores = np.full(len(mask), -np.inf)
    diagnostics = {}
    for action in np.flatnonzero(mask):
        forecast = _forecast_action(state, choices[action])
        scores[action] = forecast["score"]
        diagnostics[int(action)] = forecast
    return scores, diagnostics


def rank_future_actions(state: ForecastState, action_choices: np.ndarray, mask: np.ndarray):
    """Rank cancellable future commitments; execution remains mask-checked."""
    choices, mask = np.asarray(action_choices), np.asarray(mask, dtype=bool)
    if choices.shape != (len(mask), 4) or not mask.any():
        raise ValueError("Expected four-arm choices and a nonempty matching mask")
    scores, diagnostics = np.full(len(mask), -np.inf), {}
    for action in np.flatnonzero(mask):
        forecast = _forecast_action(state, choices[action], deferred=True)
        scores[action] = forecast["score"]
        diagnostics[int(action)] = forecast
    return scores, diagnostics
