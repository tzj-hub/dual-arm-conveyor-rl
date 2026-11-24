"""Bounded multi-step scheduling over currently visible workpieces.

This teacher is an approximate event model, not a collision checker or another
physics simulator. It never receives an environment inside the search. Existing
native/future action masks determine which first actions may be considered;
FuturePlanner.commit still checks the native mask before actual dispatch.

Every legal first action gets a greedy continuation. A global beam then explores
alternative continuations for the strongest first actions. Keeping those greedy
scores means truncating the beam cannot reduce the model's best forecast. This
is bounded search, not an optimality claim. Subsequent commands are quantized
upward to the real scheduling period. The ETA cache consequently has at most
4 * 10 * (ceil(horizon / decision_dt) + 1) entries per call.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .forecast import (
    BeltReservation, ForecastState, _completion_exit_time, _window_times,
    observed_phase_budgets, pickup_eta, visual_forecast_state,
)


@dataclass(frozen=True)
class SequenceConfig:
    horizon: float = 12.0
    beam_width: int = 12
    search_depth: int = 4
    max_expansions: int = 960

    def __post_init__(self):
        if not math.isfinite(self.horizon) or not 1 <= self.horizon <= 12:
            raise ValueError("Sequence horizon must be in [1, 12] seconds")
        if not isinstance(self.beam_width, int) or not 1 <= self.beam_width <= 16:
            raise ValueError("Sequence beam width must be in [1, 16]")
        if not isinstance(self.search_depth, int) or not 0 <= self.search_depth <= 4:
            raise ValueError("Sequence depth must be in [0, 4]")
        if not isinstance(self.max_expansions, int) or not 0 <= self.max_expansions <= 1280:
            raise ValueError("Sequence expansion budget must be in [0, 1280]")


DEFAULT_CONFIG = SequenceConfig()
_TAIL_PHASES = ("lift", "transfer_wait", "transport", "place", "release", "retreat", "return")


def sequence_snapshot(env, config=DEFAULT_CONFIG, *, base=None):
    """Read visible tracks and completed controller timing samples only.

    Per-arm 70th-percentile phase estimates come from the existing observable
    controller recorder. Cold starts retain its nominal defaults. Hypothetical
    future returns still use the snapshot's per-arm ready pose; later bin-slot
    choices and unobserved future arrivals are deliberately not invented.
    """
    state = (visual_forecast_state(env, horizon=config.horizon) if base is None
             else replace(base, horizon=config.horizon))
    budgets = [observed_phase_budgets(env, arm) for arm in range(4)]
    tails = tuple(sum(budget[p] for p in _TAIL_PHASES) for budget in budgets)
    # Preserve the existing 0.82 s clear allowance as a floor; measurements may
    # enlarge it but never relax controller exclusion or physical admission.
    clear_tails = tuple(max(0.82, budget['lift'] + 0.30) for budget in budgets)
    return state, tails, clear_tails


@dataclass(frozen=True)
class _Node:
    root: int
    remaining: int
    ready: tuple[float, ...]
    clear: tuple[float, ...]
    reservations: tuple[BeltReservation, ...]
    plan: tuple[tuple[int, int, float, float], ...] = ()
    root_dispatches: int = 0
    floor: float = 0.1


def _score(node):
    return (len(node.plan) - 0.002 * sum(p[3] for p in node.plan)
            + 0.002 * node.root_dispatches
            - (0.02 if not node.root_dispatches else 0.0))


class _Search:
    def __init__(self, state, config, tails, clear_tails, native_choices, current_slack):
        self.state, self.config = state, config
        self.jobs = {j.slot: j for j in state.jobs}
        self.tails, self.clear_tails = tails, clear_tails
        self.native_choices = None if native_choices is None else set(map(tuple, native_choices))
        self.current_slack = {} if current_slack is None else dict(current_slack)
        self.eta_cache = {}
        self.stats = dict(root_actions=0, unique_root_choices=0, root_orderings=0,
                          expansions=0, greedy_rollouts=0, candidate_checks=0,
                          eta_evaluations=0, peak_beam=0, depth_reached=0,
                          budget_exhausted=False)

    def grid_time(self, time):
        return max(0.0, math.ceil((float(time) - 1e-9) / self.state.decision_dt)
                   * self.state.decision_dt)

    def eta(self, arm, slot, start):
        tick = int(round(start / self.state.decision_dt))
        key = (arm, slot, tick)
        if key not in self.eta_cache:
            self.eta_cache[key] = pickup_eta(self.state, self.jobs[slot], arm, start)
            self.stats['eta_evaluations'] += 1
        return self.eta_cache[key]

    def candidate(self, node, arm, slot, *, immediate=False, root=False, not_before=0.0):
        self.stats['candidate_checks'] += 1
        job, station = self.jobs[slot], arm // 2
        if job.kind != arm % 2 or not job.eligible[station]:
            return None
        entry, raw_exit = _window_times(self.state, job, station)
        if immediate:
            # Existing native masks remain authoritative, including their
            # actual current ETA. Never reinterpret a retry as dispatch-now.
            if job.available_after[arm] > 1e-9:
                return None
            start = 0.0
        else:
            start = self.grid_time(max(not_before, 0.0 if root else node.floor,
                                       node.ready[arm], node.clear[station],
                                       entry, job.available_after[arm]))
        for _ in range(len(node.reservations) + 1):
            if start > min(raw_exit, self.state.horizon) + 1e-9:
                return None
            eta = self.eta(arm, slot, start)
            capture = start + eta
            if not math.isfinite(capture) or capture > self.state.horizon:
                return None
            if capture + 0.25 > _completion_exit_time(self.state, job, arm, start, capture):
                return None
            x = float(job.position[0] + job.velocity[0] * capture)
            conflicts = [r.clear for r in node.reservations
                         if r.station != station and start < r.clear
                         and capture + self.clear_tails[arm] > r.start
                         and abs(x - r.x) < 0.32]
            if not conflicts or immediate:
                # For immediate roots, the real joint mask has already checked
                # simultaneity. Approximate reservations may not veto it.
                return arm, slot, start, capture
            start = self.grid_time(max(conflicts))
        return None

    def append(self, node, pickup, *, root=False):
        arm, slot, start, capture = pickup
        ready, clear = list(node.ready), list(node.clear)
        ready[arm] = capture + self.tails[arm]
        clear[arm // 2] = max(clear[arm // 2], capture + self.clear_tails[arm])
        job = self.jobs[slot]
        reservation = BeltReservation(arm // 2, start, clear[arm // 2],
                                      float(job.position[0] + job.velocity[0] * capture))
        return _Node(node.root, node.remaining & ~(1 << slot), tuple(ready), tuple(clear),
                     node.reservations + (reservation,), node.plan + (pickup,),
                     node.root_dispatches + int(root), node.floor)

    def options(self, node):
        result = []
        for slot, job in self.jobs.items():
            if not node.remaining & (1 << slot):
                continue
            for station in range(2):
                pickup = self.candidate(node, 2 * station + job.kind, slot)
                if pickup is not None:
                    result.append(pickup)
        return result

    def greedy(self, node):
        self.stats['greedy_rollouts'] += 1
        while node.remaining:
            options = self.options(node)
            if not options:
                break
            earliest = min(p[2] for p in options)
            close = [p for p in options if p[2] <= earliest + 0.025]
            def key(p):
                arm, slot, start, capture = p
                deadline = _completion_exit_time(self.state, self.jobs[slot], arm, start, capture)
                return deadline - capture, -(arm // 2), slot, arm
            node = self.append(node, min(close, key=key))
        return node

    def roots(self, action, choices, deferred):
        selected = [(arm, int(slot)) for arm, slot in enumerate(choices) if slot < 10]
        if (len({slot for _, slot in selected}) != len(selected)
                or any(slot not in self.jobs for _, slot in selected)
                or len({arm // 2 for arm, _ in selected}) != len(selected)):
            return []
        self.stats['root_orderings'] += 1
        node = _Node(action, sum(1 << slot for slot in self.jobs),
                     tuple(self.state.arm_ready), tuple(self.state.station_ready),
                     self.state.reservations, floor=self.state.decision_dt)
        pending = list(selected)
        if deferred and self.native_choices is not None:
            # Exactly reproduce the current FuturePlanner._dispatch subset:
            # try the joint action, then single candidates by native slack.
            # Future readiness remains approximate, but already known current
            # feasibility and priority must not be replaced by model optimism.
            current = []
            for arm, slot in pending:
                if (arm, slot) not in self.state.current_eta:
                    continue
                eta = self.state.current_eta[arm, slot]
                slack = self.current_slack.get((arm, slot),
                    _completion_exit_time(self.state, self.jobs[slot], arm, 0.0, eta) - eta)
                current.append((slack, arm // 2, arm, slot))
            subsets = [current] if current else []
            if len(current) > 1:
                subsets.extend([[item] for item in sorted(current)])
            for subset in subsets:
                command = [10] * 4
                for _, _, arm, slot in subset:
                    command[arm] = slot
                if tuple(command) not in self.native_choices:
                    continue
                for _, _, arm, slot in subset:
                    pickup = self.candidate(node, arm, slot, immediate=True, root=True)
                    if pickup is None:
                        return []
                    node = self.append(node, pickup, root=True)
                    pending.remove((arm, slot))
                break
        while pending:
            proposals = [self.candidate(node, arm, slot, immediate=not deferred, root=True,
                         not_before=self.state.decision_dt if deferred and self.native_choices is not None else 0.0)
                         for arm, slot in pending]
            if any(pickup is None for pickup in proposals):
                return []
            # Root ordering is NOT a free scheduling action. Actual commit
            # starts whichever station becomes feasible first; if only one of
            # two ready plans can start it tries the tightest deadline first.
            # Mirror that priority in the approximate event model rather than
            # granting the teacher the better of two unrealizable orderings.
            def key(pickup):
                arm, slot, start, capture = pickup
                deadline = _completion_exit_time(self.state, self.jobs[slot], arm, start, capture)
                return start, deadline - capture, arm // 2, arm, slot
            pickup = min(proposals, key=key)
            node = self.append(node, pickup, root=True)
            pending.remove(pickup[:2])
        return [node]


def rank_sequence_actions(state: ForecastState, action_choices, mask, *, deferred=False,
                          config=DEFAULT_CONFIG, tail_by_arm=None, clear_by_arm=None,
                          native_choices=None, current_slack=None):
    """Return bounded beam scores and diagnostics for existing allowed roots.

    All jobs must be distinct current visual slots (at most ten). Search branches
    serialize each station and each arm; the same object can be captured once.
    Scores concern this approximate model only. Physical execution is unchanged.
    Diagnostic root_plans counts initial soft intents, not physical dispatches.
    """
    choices, mask = np.asarray(action_choices), np.asarray(mask, dtype=bool)
    if choices.shape != (len(mask), 4) or not mask.any() or len(mask) > 484:
        raise ValueError("Expected up to 484 four-arm choices and a nonempty matching mask")
    if not np.issubdtype(choices.dtype, np.integer) or np.any((choices < 0) | (choices > 10)):
        raise ValueError("Action choices must be integer track slots or 10 for no job")
    if (len(state.jobs) > 10 or len({j.slot for j in state.jobs}) != len(state.jobs)
            or any(j.slot not in range(10) or j.kind not in (0, 1) for j in state.jobs)):
        raise ValueError("At most ten distinct visible track slots are permitted")
    if not math.isfinite(state.decision_dt) or state.decision_dt < 0.01:
        raise ValueError("Decision period must be finite and at least 0.01 seconds")
    state = replace(state, horizon=min(float(state.horizon), config.horizon))
    tails = tuple([state.tail] * 4 if tail_by_arm is None else tail_by_arm)
    clear = tuple([0.82] * 4 if clear_by_arm is None else clear_by_arm)
    if len(tails) != 4 or len(clear) != 4 or any(not math.isfinite(t) or t < 0 for t in tails + clear):
        raise ValueError("Four finite nonnegative arm tail and clearance times required")
    search = _Search(state, config, tails, clear, native_choices, current_slack)
    scores = np.full(len(mask), -np.inf)
    details, initial, semantic, aliases = {}, [], {}, {}
    for action in np.flatnonzero(mask):
        action = int(action)
        search.stats['root_actions'] += 1
        key = tuple(int(c) for c in choices[action])
        if key not in semantic:
            search.stats['unique_root_choices'] += 1
            semantic[key] = [(node, search.greedy(node))
                             for node in search.roots(action, choices[action], deferred)]
            aliases[action] = []
            initial.extend((_score(completed), node) for node, completed in semantic[key])
        for node, completed in semantic[key]:
            aliases[node.root].append(action)
            value = _score(completed)
            if value > scores[action]:
                scores[action] = value
                details[action] = dict(score=value, greedy_score=value,
                                       rescued=len(completed.plan), root_plans=node.root_dispatches,
                                       plan=list(completed.plan), improved_by_beam=False)

    def best_nodes(items):
        # Distinct search states prevent permutations from consuming beam slots.
        unique = {}
        for value, node in items:
            reservations = tuple(sorted(node.reservations, key=lambda r: (r.station, r.start, r.clear, r.x)))
            key = (node.root, node.remaining, node.ready, node.clear, reservations,
                   node.root_dispatches, sum(p[3] for p in node.plan))
            old = unique.get(key)
            if old is None or value > old[0]:
                unique[key] = value, node
        return sorted(unique.values(), key=lambda item: (-item[0], item[1].root,
                                                         item[1].plan))[:config.beam_width]

    beam = best_nodes(initial)
    search.stats['peak_beam'] = len(beam)
    for depth in range(config.search_depth):
        expanded = []
        for _, node in beam:
            for pickup in search.options(node):
                if search.stats['expansions'] >= config.max_expansions:
                    search.stats['budget_exhausted'] = True
                    break
                child = search.append(node, pickup)
                search.stats['expansions'] += 1
                completed = search.greedy(child)
                value = _score(completed)
                for alias in aliases[child.root]:
                    if value > scores[alias]:
                        scores[alias] = value
                        details[alias].update(score=value, rescued=len(completed.plan),
                                              plan=list(completed.plan), improved_by_beam=True)
                expanded.append((value, child))
            if search.stats['budget_exhausted']:
                break
        if not expanded:
            break
        beam = best_nodes(expanded)
        search.stats['peak_beam'] = max(search.stats['peak_beam'], len(beam))
        search.stats['depth_reached'] = depth + 1
        if search.stats['budget_exhausted']:
            break
    search.stats['eta_cache_bound'] = 4 * len(state.jobs) * (math.ceil(state.horizon / state.decision_dt) + 1)
    search.stats['expansion_bound'] = min(config.max_expansions,
                                        config.beam_width * 2 * len(state.jobs) * config.search_depth)
    return scores, {'actions': details, 'search': search.stats}
