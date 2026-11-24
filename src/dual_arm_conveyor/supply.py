"""Independent spatial arrival clocks for two irregular upstream lanes.

This is a physical source model, not a controller input. Requested arrivals
continue while the inlet is occupied. A bounded upstream queue exposes both
backlog and overflow; blocking cannot silently reduce the requested load.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LaneArrival:
    lane: int
    distance: float
    yaw: float
    lateral: float
    kind: int


class LaneSupply:
    def __init__(self, config, seed):
        self.config = config
        self.generators = [
            np.random.default_rng(np.random.SeedSequence([int(seed), 731073, lane]))
            for lane in range(2)
        ]
        # Start with a genuine side-by-side pair. Subsequent clocks advance
        # independently, so rows drift between aligned and staggered arrivals.
        self.next_distance = np.zeros(2)
        self.pending = [deque(), deque()]
        # A finite request must eventually deliver exactly total_objects; do
        # not discard part of that batch merely because the inlet is blocked.
        # Its explicit batch size is already a bound. Continuous operation has
        # no such bound, so its queue remains limited by the physical pool.
        self.capacity_per_lane = max(4, config.pool_size // 2, config.total_objects or 0)
        self.offered = np.zeros(2, dtype=int)
        self.admitted = np.zeros(2, dtype=int)
        self.overflow = np.zeros(2, dtype=int)
        self.max_queued = 0

    def advance(self, distance):
        while min(self.next_distance) <= distance + 1e-9:
            if (
                self.config.total_objects is not None
                and self.offered.sum() >= self.config.total_objects
            ):
                break
            lane = int(np.argmin(self.next_distance))
            rng = self.generators[lane]
            scheduled = float(self.next_distance[lane])
            arrival = LaneArrival(
                lane=lane,
                distance=scheduled,
                yaw=float(rng.uniform(-self.config.yaw_range, self.config.yaw_range)),
                lateral=float(
                    (-1 if lane == 0 else 1) * self.config.lane_offset
                    + rng.uniform(-self.config.lane_jitter, self.config.lane_jitter)
                ),
                kind=int(rng.integers(2)),
            )
            self.offered[lane] += 1
            if len(self.pending[lane]) < self.capacity_per_lane:
                self.pending[lane].append(arrival)
            else:
                self.overflow[lane] += 1
            self.next_distance[lane] += max(
                0.12,
                self.config.spacing
                + rng.uniform(-self.config.spacing_jitter, self.config.spacing_jitter),
            )
        self.max_queued = max(self.max_queued, sum(map(len, self.pending)))

    def accepted(self, lane):
        arrival = self.pending[lane].popleft()
        self.admitted[lane] += 1
        return arrival

    def diagnostics(self, simulation_time):
        offered, admitted = int(self.offered.sum()), int(self.admitted.sum())
        return {
            "lanes": 2,
            "spacing_semantics": "independent distance gap per lane",
            "offered": offered,
            "admitted": admitted,
            "queued_upstream": sum(map(len, self.pending)),
            "overflow": int(self.overflow.sum()),
            "offered_per_lane": self.offered.tolist(),
            "admitted_per_lane": self.admitted.tolist(),
            "overflow_per_lane": self.overflow.tolist(),
            "max_queued_upstream": self.max_queued,
            "queue_capacity": 2 * self.capacity_per_lane,
            "queue_model": "bounded logical upstream buffer; bodies enter physics at admission",
            "overflow_policy": "continuous mode discards newest requests when the upstream buffer is full",
            "offered_per_minute": 60 * offered / simulation_time if simulation_time > 0 else 0.0,
            "admitted_per_minute": 60 * admitted / simulation_time if simulation_time > 0 else 0.0,
            "initial_condition": (
                "two simultaneous upstream requests; admission may be staggered for clearance"
                if self.config.min_object_gap > 0 else "two simultaneous upstream objects"
            ),
            "minimum_object_gap_m": self.config.min_object_gap,
            "gap_definition": "conservative planar edge clearance at admission against belt-level workpieces; 3 mm additional admission margin",
        }
