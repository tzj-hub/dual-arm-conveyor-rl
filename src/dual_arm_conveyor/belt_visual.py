"""Non-colliding stripes driven by the belt's integrated travel distance."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


MARKER_SPACING = 0.19
MARKER_HALF_WIDTH = 0.005
MARKER_HALF_HEIGHT = 0.0001
MARKER_PREFIX = "belt_marker_"


def _layout(config):
    half_length = float(config.belt_half_length)
    half_width = min(MARKER_HALF_WIDTH, half_length / 4)
    low = -half_length + half_width
    span = 2 * (half_length - half_width)
    count = max(1, round(span / MARKER_SPACING))
    offsets = np.arange(count, dtype=float) * (span / count)
    height = float(config.belt_top) + MARKER_HALF_HEIGHT + 0.00005
    return low, span, offsets, half_width, height


def append_belt_markers(world, config):
    """Append gray, massless mocap stripes to an XML worldbody element.

    Ten-millimetre strips cross the belt and sit 0.05 mm above its surface to
    avoid z-fighting. Their centers wrap inside the belt by a strip half-width,
    so even the stripe at either endpoint cannot extend beyond the belt.
    """
    low, _, offsets, half_width, height = _layout(config)
    for index, offset in enumerate(offsets):
        body = ET.SubElement(
            world,
            "body",
            name=f"{MARKER_PREFIX}{index}",
            mocap="true",
            pos=f"{low + offset:.12g} 0 {height:.12g}",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{MARKER_PREFIX}geom_{index}",
            type="box",
            size=f"{half_width:.12g} {float(config.belt_half_width):.12g} {MARKER_HALF_HEIGHT:.12g}",
            rgba="0.40 0.44 0.46 1",
            contype="0",
            conaffinity="0",
            density="0",
            group="2",
        )


class BeltMarkers:
    """Update only marker mocap positions; the normal physics pass updates poses.

    Call after assigning/resetting ``data.time`` and before the normal
    ``mj_step``/``mj_forward``. This helper never calls either function and
    never reads object poses, identities, contacts, or camera observations.
    """

    def __init__(self, model, config):
        import mujoco

        self.low, self.span, self.offsets, _, _ = _layout(config)
        self.speed = float(config.belt_speed)
        bodies = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{MARKER_PREFIX}{i}")
            for i in range(len(self.offsets))
        ]
        if any(body < 0 for body in bodies):
            raise ValueError(
                "Belt stripes are missing; build the scene with append_belt_markers"
            )
        self.mocap_ids = np.asarray(model.body_mocapid[bodies], dtype=int)
        if np.any(self.mocap_ids < 0):
            raise ValueError("Belt stripe bodies must be mocap bodies")

    def update(self, data, distance=None):
        """Animate prescribed belt travel; omit distance for legacy constant speed.

        Variable-speed callers supply the integral of belt velocity, rather than
        multiplying current velocity by time (which would make stripes jump).
        """
        travel = float(data.time) * self.speed if distance is None else float(distance)
        data.mocap_pos[self.mocap_ids, 0] = self.low + np.remainder(
            self.offsets + travel, self.span
        )
