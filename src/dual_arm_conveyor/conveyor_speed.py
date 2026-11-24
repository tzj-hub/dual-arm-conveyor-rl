"""Reproducible physical belt drive, deliberately separate from perception.

Only the simulator and evaluator may sample this object. The camera controller
must estimate current motion from measurements and must not read future samples.
Distances are analytical integrals, independent of physics/camera frame rates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BeltSample:
    speed: float
    acceleration: float
    distance: float


class BeltMotion:
    """Constant, sinusoidal, or seeded cubic velocity ramps with bounded accel.

    Random knots are generated with a dedicated RNG, so changing camera dropout,
    grasp success, arrival randomness, or sampling frequency cannot alter the
    commanded speed sequence. Ramps have continuous velocity and acceleration.
    The requested period is a minimum: slower ramps enforce max acceleration.
    """

    def __init__(self, config, seed: int):
        self.profile = config.belt_profile
        self.mean = float(config.belt_speed)
        self.amplitude = (
            0.0 if self.profile == "constant" else float(config.belt_speed_amplitude)
        )
        self.max_acceleration = float(config.belt_max_acceleration)
        self.period = max(
            float(config.belt_speed_period),
            2 * math.pi * self.amplitude / self.max_acceleration,
        )
        self.ramp_duration = max(
            float(config.belt_speed_period) / 2,
            3 * self.amplitude / self.max_acceleration,
        )
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 11717]))
        self.knots = [self.mean]
        self.prefix_distance = [0.0]
        self._ensure_knot(1)

    def _ensure_knot(self, index: int):
        while len(self.knots) <= index:
            value = self.mean + float(self.rng.uniform(-self.amplitude, self.amplitude))
            self.prefix_distance.append(
                self.prefix_distance[-1]
                + self.ramp_duration * (self.knots[-1] + value) / 2
            )
            self.knots.append(value)

    def sample(self, time: float) -> BeltSample:
        time = float(time)
        if not math.isfinite(time) or time < 0:
            raise ValueError("belt time must be finite and nonnegative")
        if self.profile == "constant" or self.amplitude == 0:
            return BeltSample(self.mean, 0.0, self.mean * time)
        if self.profile == "sine":
            omega = 2 * math.pi / self.period
            phase = omega * time
            return BeltSample(
                self.mean + self.amplitude * math.sin(phase),
                self.amplitude * omega * math.cos(phase),
                self.mean * time + self.amplitude * (1 - math.cos(phase)) / omega,
            )
        segment = int(time / self.ramp_duration)
        self._ensure_knot(segment + 1)
        u = (time - segment * self.ramp_duration) / self.ramp_duration
        initial, final = self.knots[segment : segment + 2]
        delta = final - initial
        return BeltSample(
            initial + delta * u * u * (3 - 2 * u),
            delta * 6 * u * (1 - u) / self.ramp_duration,
            self.prefix_distance[segment]
            + self.ramp_duration * (initial * u + delta * (u**3 - u**4 / 2)),
        )

    def metadata(self):
        return {
            "profile": self.profile,
            "mean_command_m_s": self.mean,
            "speed_bounds_m_s": [
                self.mean - self.amplitude,
                self.mean + self.amplitude,
            ],
            "max_acceleration_m_s2": self.max_acceleration,
            "effective_sine_period_s": self.period if self.profile == "sine" else None,
            "random_ramp_duration_s": self.ramp_duration
            if self.profile == "random"
            else None,
            "source": "prescribed simulation drive, unavailable to the vision controller",
        }
