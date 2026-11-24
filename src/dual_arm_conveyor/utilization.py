"""Account for robot occupancy without changing control or reading object truth.

Occupancy is non-idle time divided by *all* observed simulation time. It includes
waiting and recovery, so a high value alone is not evidence of useful throughput.
The mutually exclusive categories describe activity, not whether a pick succeeds.
"""

from __future__ import annotations

from collections import deque
import math
from numbers import Integral

import numpy as np


CATEGORIES = ("processing", "return", "waiting", "recovery", "idle")
_PROCESSING = {
    "approach", "reorient", "descend", "close", "lift", "depart", "align",
    "transport", "place", "release",
}
_PHASE_CATEGORY = {
    **dict.fromkeys(_PROCESSING, 0),
    "retreat": 1,
    "return": 1,
    "transfer_wait": 2,
    "yield": 2,
    "recover": 3,
    "idle": 4,
}


class ArmUtilization:
    """Integrate per-arm state at each physics step with bounded window history.

    Consecutive identical category tuples share a segment. Window expiration
    removes elapsed portions exactly, including a partially expired segment.
    Reports contain copies; requesting one never changes timers or history.
    """

    def __init__(self, num_arms: int, window_seconds: float = 30.0):
        if isinstance(num_arms, bool) or not isinstance(num_arms, Integral) or num_arms < 1:
            raise ValueError("num_arms must be a positive integer")
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        self.num_arms = int(num_arms)
        self.window_seconds = float(window_seconds)
        self.reset()

    def reset(self):
        self.elapsed = 0.0
        self._total = np.zeros((self.num_arms, len(CATEGORIES)))
        self._recent = np.zeros_like(self._total)
        self._segments = deque()
        self._arms = np.arange(self.num_arms)

    def record(self, phases, hold_requested, dt: float):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        if len(phases) != self.num_arms or len(hold_requested) != self.num_arms:
            raise ValueError("phase/hold vectors must match num_arms")
        codes = []
        for phase, held in zip(phases, hold_requested):
            if phase not in _PHASE_CATEGORY:
                raise ValueError(f"Unclassified controller phase: {phase}")
            category = _PHASE_CATEGORY[phase]
            # Idle holds are idle; recovery remains separately visible. Release
            # deliberately holds position while opening and settling, so that
            # hold is part of processing, not a blocked motion request.
            if held and category in (0, 1) and phase != "release":
                category = 2
            codes.append(category)
        if dt == 0:
            return
        codes = tuple(codes)
        start = self.elapsed
        self.elapsed += dt
        self._total[self._arms, codes] += dt
        self._recent[self._arms, codes] += dt
        if self._segments and self._segments[-1][2] == codes:
            self._segments[-1][1] = self.elapsed
        else:
            self._segments.append([start, self.elapsed, codes])
        cutoff = max(0.0, self.elapsed - self.window_seconds)
        while self._segments and self._segments[0][0] < cutoff:
            first = self._segments[0]
            end = min(first[1], cutoff)
            self._recent[self._arms, first[2]] -= end - first[0]
            if first[1] <= cutoff:
                self._segments.popleft()
            else:
                first[0] = cutoff
                break

    def _summary(self, totals, duration):
        # Floating-point subtraction at a window boundary can leave tiny
        # negative residues; clipping only presentation preserves accounting.
        seconds = np.clip(totals, 0.0, duration)
        ratios = seconds / duration if duration > 0 else np.zeros_like(seconds)
        busy = np.clip(ratios[:, :4].sum(axis=1), 0.0, 1.0)
        return {
            "duration_seconds": float(duration),
            "total_arm_seconds": float(duration * self.num_arms),
            "per_arm_seconds": [dict(zip(CATEGORIES, row.tolist())) for row in seconds],
            "per_arm_ratios": [dict(zip(CATEGORIES, row.tolist())) for row in ratios],
            "per_arm_busy": busy.tolist(),
            "mean_busy": float(np.mean(busy)),
            "mean_ratios": dict(zip(CATEGORIES, np.mean(ratios, axis=0).tolist())),
        }

    def report(self):
        return {
            "schema_version": 1,
            "window_seconds": self.window_seconds,
            "categories": list(CATEGORIES),
            "definition": "busy = non-idle time / all observed simulation time, including unloading and supply gaps",
            "processing_note": "Processing describes attempted task execution, not successful work; assess it alongside sorted, missed, dropped and throughput.",
            "waiting_note": "transfer_wait/yield and motion holds; normal release holds count as processing, recovery holds as recovery",
            "cumulative": self._summary(self._total, self.elapsed),
            "window": self._summary(self._recent, min(self.elapsed, self.window_seconds)),
        }
