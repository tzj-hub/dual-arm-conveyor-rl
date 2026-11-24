"""Read-only occupancy overlays shared by live playback and recordings."""

from __future__ import annotations


def utilization_lines(report):
    """Keep occupancy distinct from successful work; all values use sim time."""
    current, total = report["window"], report["cumulative"]
    span = current["duration_seconds"]
    lines = [
        "ARM OCCUPANCY | busy = non-idle",
        f"Last {span:.1f}s mean {current['mean_busy']:.0%} | total {total['mean_busy']:.0%}",
    ]
    for arm, (recent, overall) in enumerate(
        zip(current["per_arm_busy"], total["per_arm_busy"]), start=1
    ):
        lines.append(f"A{arm}  recent {recent:.0%} | total {overall:.0%}")
    split = current["mean_ratios"]
    lines += [
        f"Recent task {split['processing']:.0%} | return {split['return']:.0%}",
        f"Recent wait {split['waiting']:.0%} | recover {split['recovery']:.0%}",
        "Task time includes unsuccessful attempts.",
    ]
    return tuple(lines)


class UtilizationOverlay:
    """Native viewer text, refreshed at most twice per simulated second."""

    def __init__(self):
        self._tick = None

    def update(self, viewer, env):
        report = getattr(env, "utilization_report", None)
        setter = getattr(viewer, "set_texts", None)
        if not callable(report) or not callable(setter):
            return
        tick = int(float(env.data.time) * 2 + 1e-8)
        if tick == self._tick:
            return
        import mujoco

        setter((
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "\n".join(utilization_lines(report())),
            "",
        ))
        self._tick = tick


def draw_utilization(draw, env, width, height):
    """Draw onto the overview, leaving the delivered camera image untouched."""
    report = getattr(env, "utilization_report", None)
    if not callable(report) or width < 240 or height < 200:
        return
    from .live_view import _fit_text

    lines = utilization_lines(report())
    box_width = min(width - 16, 330)
    draw.rectangle((8, 36, 8 + box_width, 44 + 17 * len(lines)), fill=(18, 24, 32))
    for row, line in enumerate(lines):
        draw.text(
            (16, 42 + row * 17), _fit_text(draw, line, box_width - 16),
            fill=(230, 235, 240) if row < 2 else (190, 210, 225),
        )
