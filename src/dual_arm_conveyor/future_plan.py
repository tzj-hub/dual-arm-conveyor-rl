"""Cancellable, observable next-job plans, separate from physical reservations.

Plans never modify ``target``, ``belief``, bin ownership or robot phase. Their
only execution path selects an action from the controller's current native
mask. Optional preparation goals are advisory inputs to checked robot motion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .forecast import (
    FORECAST_HORIZON, _completion_exit_time, _reserved_pickup, _window_times,
    rank_future_actions, visual_forecast_state,
)

OPTIONS = 22
KEEP = 20
CANCEL = 21
ACTION_COUNT = OPTIONS ** 2
WAIT_ACTION = KEEP * OPTIONS + KEEP
ACTION_SCHEMA = "two_station_reversible_plan_22x22_v1"
OBSERVATION_SCHEMA = "observable_future_plans_transition_v1"
GROUP_NAMES = {0: "keep_plans", 1: "change_upstream_plan",
               2: "change_downstream_plan", 3: "change_both_plans"}
PAIRS = np.array(np.unravel_index(np.arange(ACTION_COUNT), (OPTIONS, OPTIONS))).T
EXTRA_OPTION_NAMES = (
    "set_next_job", "keep_next_job", "cancel_next_job", "same_existing_plan",
    "planned_start_in", "planned_capture_in", "transition_eta",
    "completion_slack", "immediately_dispatchable",
)
PLAN_GLOBAL_NAMES = (
    "plan_present", "plan_local_arm0", "plan_local_arm1", "plan_position_x",
    "plan_position_y", "plan_start_in", "plan_capture_in", "plan_age",
)


@dataclass(frozen=True)
class SoftPlan:
    arm: int
    track_id: int
    created_at: float


@dataclass(frozen=True)
class Proposal:
    arm: int
    slot: int
    start: float
    capture: float
    deadline: float

    @property
    def eta(self):
        return self.capture - self.start


def enabled(env):
    return bool(getattr(env.config, "future_planning", False))


def planner_for(env):
    planner = getattr(env, "_future_planner", None)
    if planner is None:
        planner = FuturePlanner(env)
        env._future_planner = planner
    planner.check_reset()
    return planner


class FuturePlanner:
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.plans: list[SoftPlan | None] = [None, None]
        self._tracks_container = self.env.tracks
        self._last_time = float(self.env.data.time)
        self.counts = dict(created=0, replaced=0, cancelled=0, expired=0,
                           dispatch_commands=0, safety_deferred_ticks=0)
        self.env.future_preposition_goals = np.full((4, 3), np.nan)
        self.env.future_preposition_yaws = np.full(4, np.nan)

    def check_reset(self):
        now = float(self.env.data.time)
        if self._tracks_container is not self.env.tracks or now < self._last_time - 1e-9:
            self.reset()
        self._last_time = now

    def slot_for(self, plan):
        if plan is None:
            return None
        matches = [i for i, track in enumerate(self.env.tracks)
                   if track is not None and int(track.track_id) == plan.track_id]
        return matches[0] if len(matches) == 1 else None

    def snapshot(self):
        self.check_reset()
        state = visual_forecast_state(self.env)
        proposals = {}
        for job in state.jobs:
            for station in range(2):
                if not job.eligible[station]:
                    continue
                arm = 2 * station + job.kind
                entry, raw_exit = _window_times(state, job, station)
                start = max(entry, float(state.arm_ready[arm]),
                            float(state.station_ready[station]), float(job.available_after[arm]))
                start, eta = _reserved_pickup(state, job, arm, start, state.reservations)
                capture = start + eta
                deadline = _completion_exit_time(state, job, arm, start, capture)
                if (not math.isfinite(capture) or start > raw_exit
                        or capture > state.horizon or capture + 0.25 > deadline):
                    continue
                proposals[(arm, job.slot)] = Proposal(arm, job.slot, start, capture, deadline)
        return state, proposals

    def prune(self, proposals):
        # Soft intent expires with its observable identity or predicted window.
        # No physical reservation is released by this bookkeeping operation.
        for station, plan in enumerate(self.plans):
            slot = self.slot_for(plan)
            if plan is not None and (slot is None or (plan.arm, slot) not in proposals):
                self.plans[station] = None
                self.counts['expired'] += 1

    def option_proposal(self, station, option, proposals):
        if option < KEEP:
            return proposals.get((2 * station + option // 10, option % 10))
        if option == KEEP:
            plan = self.plans[station]
            slot = self.slot_for(plan)
            return proposals.get((plan.arm, slot)) if plan is not None and slot is not None else None
        return None

    def choices_and_mask(self, proposals):
        local = np.zeros((2, OPTIONS), dtype=bool)
        local[:, KEEP] = True
        for station in range(2):
            existing = self.plans[station]
            local[station, CANCEL] = existing is not None
            for option in range(KEEP):
                proposal = self.option_proposal(station, option, proposals)
                if proposal is None:
                    continue
                track_id = int(self.env.tracks[proposal.slot].track_id)
                # Re-emitting the existing plan is KEEP, not another actor
                # choice or another plan-created statistic.
                local[station, option] = not (
                    existing is not None and existing.arm == proposal.arm
                    and existing.track_id == track_id
                )
        mask = local[0, PAIRS[:, 0]] & local[1, PAIRS[:, 1]]
        choices = np.full((ACTION_COUNT, 4), 10, dtype=np.int64)
        for action in np.flatnonzero(mask):
            used = []
            for station, option in enumerate(PAIRS[action]):
                proposal = self.option_proposal(station, int(option), proposals)
                if proposal is not None:
                    choices[action, proposal.arm] = proposal.slot
                    used.append(proposal.slot)
            if len(used) == 2 and used[0] == used[1]:
                mask[action] = False
        if not mask[WAIT_ACTION]:
            raise RuntimeError("KEEP must remain a valid future-plan action")
        return choices, mask

    def commit(self, action, native_observation=None):
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("Future action must be an integer")
        if not 0 <= action < ACTION_COUNT:
            raise ValueError("Future action is out of range")
        _, proposals = self.snapshot()
        self.prune(proposals)
        _, mask = self.choices_and_mask(proposals)
        if not mask[action]:
            raise ValueError("Future action is not valid in the current observable state")
        now = float(self.env.data.time)
        for station, option in enumerate(PAIRS[action]):
            if option == CANCEL:
                if self.plans[station] is not None:
                    self.counts['cancelled'] += 1
                self.plans[station] = None
            elif option < KEEP:
                proposal = self.option_proposal(station, int(option), proposals)
                self.counts['replaced' if self.plans[station] else 'created'] += 1
                self.plans[station] = SoftPlan(proposal.arm,
                    int(self.env.tracks[proposal.slot].track_id), now)
        self._preposition(proposals)
        return self._dispatch(native_observation)

    def _dispatch(self, native_observation=None):
        native = self.env.observe() if native_observation is None else native_observation
        mask = np.asarray(native['mask'], dtype=bool)
        base = np.full(4, 10, dtype=np.int64)
        candidates = []
        for station, plan in enumerate(self.plans):
            slot = self.slot_for(plan)
            if plan is None or slot is None:
                continue
            feasible, eta, deadline = self.env._candidate(plan.arm, slot)
            if feasible:
                candidates.append((float(deadline - eta), station, plan.arm, slot))
        # Try the requested joint dispatch first. If physically incompatible,
        # execute an urgent requested subset; never substitute another target.
        selections = [candidates] if candidates else []
        selections += [[c] for c in sorted(candidates)] if len(candidates) > 1 else []
        for selected in selections:
            choices = base.copy()
            for _, _, arm, slot in selected:
                choices[arm] = slot
            action = int(np.ravel_multi_index(choices, (11,) * 4))
            if mask[action]:
                for _, station, arm, _ in selected:
                    self.plans[station] = None
                    self.env.future_preposition_goals[arm] = np.nan
                    self.env.future_preposition_yaws[arm] = np.nan
                    self.counts['dispatch_commands'] += 1
                return action
        if any(plan is not None for plan in self.plans):
            self.counts['safety_deferred_ticks'] += 1
        action = int(np.ravel_multi_index(base, (11,) * 4))
        if not mask[action]:
            raise RuntimeError("Physical WAIT is unavailable")
        return action

    def _preposition(self, proposals):
        self.env.future_preposition_goals[:] = np.nan
        self.env.future_preposition_yaws[:] = np.nan
        if not (getattr(self.env.config, 'safe_preposition', False)
                or getattr(self.env.config, 'planned_return', False)):
            return
        now = float(self.env.data.time)
        for station, plan in enumerate(self.plans):
            if plan is None:
                continue
            slot = self.slot_for(plan)
            proposal = proposals.get((plan.arm, slot))
            if proposal is None:
                continue
            track = self.env.tracks[slot]
            position = np.asarray(track.predict(now + proposal.capture), dtype=float).copy()
            position[0] = np.clip(position[0], *self.env.windows[plan.arm])
            position[1] = (-1 if plan.arm % 2 == 0 else 1) * 0.26
            position[2] = max(float(getattr(self.env, 'lift_height', 0.38)),
                              float(getattr(self.env.ids, 'bin_rim_z', 0.255)) + 0.12)
            angles = getattr(self.env, '_grasp_angles', None)
            options = angles(plan.arm, track) if callable(angles) else [float(track.yaw)]
            if options:
                self.env.future_preposition_goals[plan.arm] = position
                self.env.future_preposition_yaws[plan.arm] = float(options[0])

    def teacher_action(self):
        state, proposals = self.snapshot()
        choices, mask = self.choices_and_mask(proposals)
        scores, _ = rank_future_actions(state, choices, mask)
        # Equal predicted outcomes retain their previous soft plan, reducing
        # needless target and preparation motion churn without hard locking.
        for station in range(2):
            if self.plans[station] is not None:
                scores[PAIRS[:, station] == KEEP] += 0.003
        if not np.isfinite(scores).any():
            return WAIT_ACTION
        return int(np.argmax(scores))

    def metrics(self):
        return {**self.counts, 'active_soft_plans': sum(p is not None for p in self.plans),
                'semantics': 'soft intent only; dispatch commands are not successful grasps'}


def encode_future_observation(env, native_encoded):
    """Augment legacy visual features with explicit future intent and timings."""
    from .scheduler import OPTION_FEATURE_NAMES, PAIR_FEATURE_NAMES
    planner = planner_for(env)
    _, proposals = planner.snapshot()
    planner.prune(proposals)
    _, mask = planner.choices_and_mask(proposals)
    legacy_pairs = np.minimum(PAIRS, KEEP)
    for station in range(2):
        proposal = planner.option_proposal(station, KEEP, proposals)
        if proposal is not None:
            # KEEP refers to the actual existing object and hand. Copying the
            # native WAIT row would hide its geometry from the per-action MLP;
            # that MLP cannot inspect another action's masked SET descriptor.
            legacy_pairs[PAIRS[:, station] == KEEP, station] = (
                (proposal.arm % 2) * 10 + proposal.slot
            )
    indices = legacy_pairs[:, 0] * 21 + legacy_pairs[:, 1]
    features = native_encoded['features'][indices].copy()
    option_width = len(OPTION_FEATURE_NAMES)
    extra = np.zeros((ACTION_COUNT, 2 * len(EXTRA_OPTION_NAMES)), dtype=np.float32)
    now = float(env.data.time)
    for station in range(2):
        for option in range(OPTIONS):
            rows = PAIRS[:, station] == option
            offset = station * len(EXTRA_OPTION_NAMES)
            extra[rows, offset + 0] = option < KEEP
            extra[rows, offset + 1] = option == KEEP
            extra[rows, offset + 2] = option == CANCEL
            proposal = planner.option_proposal(station, option, proposals)
            if proposal is None:
                continue
            existing = planner.plans[station]
            same = (existing is not None and existing.arm == proposal.arm
                    and existing.track_id == int(env.tracks[proposal.slot].track_id))
            immediate = bool(env._candidate(proposal.arm, proposal.slot)[0])
            extra[rows, offset + 3:offset + len(EXTRA_OPTION_NAMES)] = np.array([
                same, proposal.start / 5, proposal.capture / 5, proposal.eta / 5,
                (proposal.deadline - proposal.capture) / 5, immediate,
            ], dtype=np.float32)
            # These fields become task-specific even while the arm is busy,
            # where the legacy candidate function intentionally returns zeros.
            for name, value in [('candidate_feasible', 1.0),
                                ('candidate_eta', proposal.eta / 5),
                                ('candidate_deadline', proposal.deadline / 5),
                                ('candidate_slack', (proposal.deadline - proposal.capture) / 5)]:
                features[rows, station * option_width + OPTION_FEATURE_NAMES.index(name)] = value
    # The remapped native pair supplies the resolved plans' actual geometric
    # relationship. Its ETA sum must also use the future transition estimates,
    # since a busy arm's native _candidate intentionally reports zero ETA.
    both = (legacy_pairs[:, 0] < KEEP) & (legacy_pairs[:, 1] < KEEP)
    eta_index = OPTION_FEATURE_NAMES.index('candidate_eta')
    pair_offset = 2 * option_width
    features[both, pair_offset + PAIR_FEATURE_NAMES.index('sum_candidate_eta')] = (
        features[both, eta_index] + features[both, option_width + eta_index]
    )
    features[:, -1] = mask
    globals_extra = []
    for station in range(2):
        plan = planner.plans[station]
        proposal = planner.option_proposal(station, KEEP, proposals)
        if plan is None or proposal is None:
            globals_extra.extend([0.] * len(PLAN_GLOBAL_NAMES))
            continue
        p = env.tracks[proposal.slot].predict(now)
        globals_extra.extend([1., float(plan.arm % 2 == 0), float(plan.arm % 2 == 1),
                              p[0] / 1.2, p[1] / .2, proposal.start / 5,
                              proposal.capture / 5, (now - plan.created_at) / 5])
    encoded = {
        'features': np.clip(np.concatenate((features, extra), axis=1), -4, 4).astype(np.float32),
        'global': np.concatenate((native_encoded['global'], np.clip(globals_extra, -4, 4))).astype(np.float32),
        'mask': mask,
    }
    if not all(np.isfinite(encoded[k]).all() for k in ('features', 'global')):
        raise ValueError('Nonfinite future scheduling observation')
    return encoded
