"""Viewer status reports measured occupancy without changing sensor images."""

from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw

from dual_arm_conveyor.utilization import ArmUtilization
from dual_arm_conveyor.utilization_view import (
    UtilizationOverlay, draw_utilization, utilization_lines,
)


def meter():
    value = ArmUtilization(2)
    value.record(["approach", "idle"], [False, True], 10)
    value.record(["return", "recover"], [False, False], 10)
    value.record(["transfer_wait", "idle"], [False, True], 10)
    return value


def test_text_keeps_idle_in_denominator_and_exposes_nonproductive_busy_time():
    lines = utilization_lines(meter().report())
    assert "100%" in lines[2]
    assert "33%" in lines[3]
    assert "67%" in lines[1]
    assert any("wait 17%" in line and "recover 17%" in line for line in lines)
    assert any("unsuccessful attempts" in line for line in lines)


def test_real_viewer_text_adapter_and_rate_limit():
    class Sim:
        def __init__(self):
            self.calls = []

        def set_texts(self, texts):
            self.calls.append(texts)

    sim = Sim()
    viewer = mujoco.viewer.Handle(sim, mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvPerturb(), None)
    counter = meter()
    env = SimpleNamespace(data=SimpleNamespace(time=30.0), utilization_report=counter.report)
    status = UtilizationOverlay()
    status.update(viewer, env)
    env.data.time = 30.4
    status.update(viewer, env)
    assert len(sim.calls) == 1
    assert "A1" in sim.calls[0][0][2] and "A2" in sim.calls[0][0][2]
    env.data.time = 30.5
    status.update(viewer, env)
    assert len(sim.calls) == 2
    env.data.time = 0.0
    status.update(viewer, env)
    assert len(sim.calls) == 3


def test_recording_status_only_changes_overview_area():
    canvas = Image.new("RGB", (1440, 720), (100, 110, 120))
    before = np.asarray(canvas).copy()
    draw_utilization(ImageDraw.Draw(canvas), SimpleNamespace(utilization_report=meter().report), 960, 720)
    after = np.asarray(canvas)
    assert np.any(after[:, :960] != before[:, :960])
    np.testing.assert_array_equal(after[:, 960:], before[:, 960:])
