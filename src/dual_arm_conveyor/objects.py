"""Reproducible physical object catalogue, unavailable to the vision controller.

The contact labels below vary MuJoCo contact time constants. Objects remain
rigid bodies; these labels do not represent elastic deformation or a measured
Young's modulus. ``size`` always contains full local x/y/z dimensions in metres.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class ObjectSpec:
    shape: str
    size: tuple[float, float, float]
    mass: float
    friction: tuple[float, float, float]
    solref: tuple[float, float]
    stiffness_label: str

    @property
    def half_sizes(self) -> tuple[float, float, float]:
        return tuple(value / 2 for value in self.size)

    @property
    def half_height(self) -> float:
        return self.size[2] / 2

    @property
    def height(self) -> float:
        return self.size[2]

    @property
    def grasp_width(self) -> float:
        return min(self.size[:2])

    @property
    def geom_size(self) -> tuple[float, ...]:
        if self.shape == "cylinder":
            return (self.size[0] / 2, self.size[2] / 2)
        return self.half_sizes

    @property
    def prism_sides(self) -> int | None:
        return {"pentagonal_prism": 5, "hexagonal_prism": 6}.get(self.shape)

    def prism_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Closed convex flat-ended mesh, centered on its bounding box.

        Physical mass/inertia are computed from this geometry by MuJoCo. No
        invisible cylinder or spherical collision approximation is used.
        """
        sides = self.prism_sides
        if sides is None:
            raise ValueError("Only polygonal prisms have a generated mesh")
        angles = np.arange(sides) * (2 * np.pi / sides)
        polygon = np.column_stack((np.cos(angles), np.sin(angles)))
        lo, hi = polygon.min(axis=0), polygon.max(axis=0)
        polygon = (polygon - (hi + lo) / 2) / (hi - lo) * self.size[:2]
        vertices = np.concatenate(
            [
                np.column_stack((polygon, np.full(sides, z)))
                for z in (-self.half_height, self.half_height)
            ]
        )
        faces = []
        for k in range(1, sides - 1):
            faces.extend(((0, k + 1, k), (sides, sides + k, sides + k + 1)))
        for k in range(sides):
            nxt = (k + 1) % sides
            faces.extend(((k, nxt, nxt + sides), (k, nxt + sides, k + sides)))
        return vertices, np.asarray(faces, dtype=int)

    @property
    def bounding_radius(self) -> float:
        return math.sqrt(sum(value * value for value in self.half_sizes))

    def to_dict(self) -> dict:
        return asdict(self)


def make_object_catalog(count: int, seed: int = 0) -> tuple[ObjectSpec, ...]:
    """Build varied cuboids and convex pentagonal/hexagonal prisms."""
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 731003]))
    shapes = np.array(("box", "pentagonal_prism", "hexagonal_prism"))[
        np.arange(count) % 3
    ]
    rng.shuffle(shapes)
    softness = (
        ("firm_contact", 0.006),
        ("medium_contact", 0.012),
        ("soft_contact", 0.025),
    )
    result = []
    for index, shape in enumerate(shapes):
        if shape == "box":
            x = float(rng.uniform(0.040, 0.060))
            y = float(rng.uniform(0.035, min(0.052, x)))
            z = float(rng.uniform(0.035, 0.060))
        else:
            x = float(rng.uniform(0.045, 0.060))
            y = float(rng.uniform(0.040, min(0.055, x)))
            z = float(rng.uniform(0.035, 0.060))
        label, time_constant = softness[index % len(softness)]
        result.append(
            ObjectSpec(
                shape=str(shape),
                size=(x, y, z),
                mass=float(rng.uniform(0.020, 0.100)),
                friction=(float(rng.uniform(0.5, 1.2)), 0.015, 0.001),
                solref=(time_constant, 1.0),
                stiffness_label=label,
            )
        )
    return tuple(result)
