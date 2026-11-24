"""Read-only, decision-rate diagnostics for the four-arm station exclusion.

These are sampled duration estimates, not physics-rate timings or proofs that
a missed object could have been grasped. A coarse visual candidate only passes
visibility, freshness, confidence, reservation, class and window checks. Reach,
grasp geometry, deadlines, bin availability and collision paths are not checked.
"""

from __future__ import annotations

import math

import numpy as np


_EXCLUSIVE_PHASES = frozenset(("approach", "descend", "close", "lift", "reorient"))
_EXIT_PHASES = frozenset(("transport", "place"))
_STRIP_HALF_WIDTH = 0.23


class OpportunityMonitor:
    """Accumulate diagnostics without changing masks, reservations or control.

    Call ``sample(env, decision_dt)`` once before dispatch at each decision and
    ``reset()`` at episode reset. Each snapshot represents one decision interval;
    phase changes within that interval are not integrated. The monitor defaults
    to disabled, in which case ``sample`` does not inspect the environment.

    ``blocked`` arrays belong to the idle, blocked arm. ``exit_tail`` belongs to
    the arm transporting/placing inside the exclusion strip, even if its partner
    is busy. Track counts count fresh *currently visible* perception tracks;
    temporarily occluded predictions are deliberately not called visible.
    """

    def __init__(self, *, enabled: bool = False):
        self.enabled = bool(enabled)
        self.reset()

    def reset(self) -> None:
        self.sample_count = 0
        self.sampled_seconds = 0.0
        self.blocked_seconds = np.zeros(4)
        self.phase_blocked_seconds = np.zeros(4)
        self.strip_blocked_seconds = np.zeros(4)
        self.exit_tail_seconds = np.zeros(4)
        self.return_strip_seconds = np.zeros(4)
        self.blocked_by_return_seconds = np.zeros(4)
        self.blocked_visible_seconds = np.zeros(4)
        self.blocked_visible_ticks = np.zeros(4, dtype=np.int64)
        self.blocked_visible_track_ticks = np.zeros(4, dtype=np.int64)
        self.blocked_visible_track_seconds = np.zeros(4)
        self._blocked_visible_ids = [set() for _ in range(4)]

    def sample(self, env, dt: float) -> None:
        """Read controller state and perception tracks; never run ``_candidate``.

        Positive finite ``dt`` is the duration represented by this snapshot, not
        measured CPU time. Zero is a no-op. Four arms are required when enabled,
        because that is where the diagnosed station exclusion is implemented.
        """
        if not self.enabled:
            return
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("diagnostic sample dt must be finite and nonnegative")
        if dt == 0:
            return
        if env.num_arms != 4:
            raise ValueError("station opportunity diagnostics require four arms")

        phases = tuple(env.phase)
        ee = np.asarray(env.data.site_xpos)[env.ids.ee_sites]
        in_strip = np.abs(ee[:, 1]) < _STRIP_HALF_WIDTH
        blocked = np.zeros(4, dtype=bool)
        self.sample_count += 1
        self.sampled_seconds += dt
        for arm, phase in enumerate(phases):
            if phase in _EXIT_PHASES and in_strip[arm]:
                self.exit_tail_seconds[arm] += dt
            if phase == "return" and in_strip[arm]:
                self.return_strip_seconds[arm] += dt
            if phase != "idle":
                continue
            partner = arm ^ 1
            phase_block = phases[partner] in _EXCLUSIVE_PHASES
            strip_block = phases[partner] != "idle" and in_strip[partner]
            if phase_block:
                self.phase_blocked_seconds[arm] += dt
            if strip_block:
                self.strip_blocked_seconds[arm] += dt
                if phases[partner] == "return":
                    self.blocked_by_return_seconds[arm] += dt
            if phase_block or strip_block:
                blocked[arm] = True
                self.blocked_seconds[arm] += dt

        if not blocked.any():
            return
        now = float(env.data.time)
        max_age = float(env.config.track_max_age)
        candidates = []
        for slot, track in enumerate(env.tracks):
            if (
                track is None
                or env.belief[slot] != 1
                or not track.visible
                or not math.isfinite(float(track.last_seen))
                or now - track.last_seen > max_age
                or not math.isfinite(float(track.confidence))
                or track.confidence < 0.35
            ):
                continue
            position = np.asarray(track.predict(now), dtype=float)
            if not np.all(np.isfinite(position[:2])) or abs(position[1]) > 0.11:
                continue
            candidates.append((int(track.track_id), int(track.kind), float(position[0])))

        for arm in np.flatnonzero(blocked):
            low, high = env.windows[arm]
            ids = {
                track_id for track_id, kind, x in candidates
                if kind == arm % 2 and low <= x <= high
            }
            count = len(ids)
            if count:
                self.blocked_visible_seconds[arm] += dt
                self.blocked_visible_ticks[arm] += 1
                self.blocked_visible_track_ticks[arm] += count
                self.blocked_visible_track_seconds[arm] += count * dt
                self._blocked_visible_ids[arm].update(ids)

    def metrics(self) -> dict:
        """Return independent JSON-compatible metrics for logging.

        Phase and strip reasons overlap, so their durations must not be added.
        ``track_ticks`` counts sightings across sampled decisions; ``unique``
        deduplicates perception IDs per arm across the episode, not object truth.
        """
        return {
            "cooperation_monitor_enabled": self.enabled,
            "cooperation_timing": "decision_snapshot_estimate",
            "cooperation_samples": self.sample_count,
            "cooperation_sampled_seconds": self.sampled_seconds,
            "cooperation_arm_blocked_seconds": self.blocked_seconds.tolist(),
            "cooperation_arm_phase_blocked_seconds": self.phase_blocked_seconds.tolist(),
            "cooperation_arm_strip_blocked_seconds": self.strip_blocked_seconds.tolist(),
            "cooperation_arm_exit_tail_seconds": self.exit_tail_seconds.tolist(),
            "cooperation_arm_return_strip_seconds": self.return_strip_seconds.tolist(),
            "cooperation_arm_blocked_by_return_seconds": self.blocked_by_return_seconds.tolist(),
            "cooperation_arm_blocked_visible_seconds": self.blocked_visible_seconds.tolist(),
            "cooperation_arm_blocked_visible_ticks": self.blocked_visible_ticks.tolist(),
            "cooperation_arm_blocked_visible_track_ticks": self.blocked_visible_track_ticks.tolist(),
            "cooperation_arm_blocked_visible_track_seconds": self.blocked_visible_track_seconds.tolist(),
            "cooperation_arm_blocked_visible_unique_tracks": [len(ids) for ids in self._blocked_visible_ids],
        }
