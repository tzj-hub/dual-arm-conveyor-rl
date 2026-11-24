"""Closing one real sensor must not corrupt another sensor's GL resources."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.vision import SensorPipeline, Track
from dual_arm_conveyor.wrist import WristSensorPipeline


def scene():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <visual><global offwidth="96" offheight="72"/>
        <quality offsamples="0"/></visual>
      <worldbody><light pos="0 -1 3"/>
        <camera name="overhead" pos="0 -3 2" xyaxes="1 0 0 0 .5547 .83205"/>
        <camera name="r0/wrist_rgbd" pos="0 -3 2" xyaxes="1 0 0 0 .5547 .83205"/>
        <camera name="r1/wrist_rgbd" pos=".1 -3 2" xyaxes="1 0 0 0 .5547 .83205"/>
        <geom type="plane" size="2 2 .1" rgba=".4 .4 .4 1"/>
        <geom type="box" pos="0 0 .4" size=".3 .3 .4" rgba=".1 .8 .2 1"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def sensor(kind):
    model, data = scene()
    config = Config(camera_width=96, camera_height=72,
                    wrist_camera_width=96, wrist_camera_height=72,
                    wrist_camera_mode="observe", depth_noise=0, camera_dropout=0)
    if kind == "overhead":
        pipeline = SensorPipeline(model, config)
    else:
        pipeline = WristSensorPipeline(model, config)
        target = Track(1, np.array([0., 0., .4]), np.zeros(3),
                       0., 0, .95, 0., 0., hits=3)
        pipeline.set_targets([target, None])
    return pipeline, data


def capture(pipeline, data):
    if isinstance(pipeline, SensorPipeline):
        pipeline.capture(data)
        frame = pipeline._queue[-1].frames["overhead"]
    else:
        assert pipeline.capture(0, data)
        frame = pipeline._queue[-1]
    return frame.rgb.copy(), frame.depth.copy()


def cleanup(pipeline):
    # Keep cleanup safe even when testing an older, defective close method.
    renderer = pipeline._renderer
    context = getattr(renderer, "_gl_context", None)
    if context is not None:
        context.make_current()
    pipeline.close()


@pytest.mark.parametrize("closing", ["overhead", "wrist"])
@pytest.mark.parametrize("surviving", ["overhead", "wrist"])
def test_closing_inactive_sensor_preserves_other_sensors_rgb_and_depth(closing, surviving):
    first, first_data = sensor(closing)
    second, second_data = sensor(surviving)
    try:
        capture(first, first_data)
        before = capture(second, second_data)  # The second GL context is current.
        assert np.ptp(before[0]) > 0 and np.all(np.isfinite(before[1]))
        first.close()
        assert first._renderer is None and not first._queue
        after = capture(second, second_data)
        for original, current in zip(before, after):
            np.testing.assert_array_equal(original, current)

        # Repeated close must not touch the surviving context either.
        first.close()
        again = capture(second, second_data)
        for original, current in zip(before, again):
            np.testing.assert_array_equal(original, current)
    finally:
        cleanup(first)
        cleanup(second)


@pytest.mark.parametrize("kind", ["overhead", "wrist"])
def test_closing_uncaptured_sensor_is_idempotent(kind):
    pipeline, _ = sensor(kind)
    assert pipeline._renderer is None
    pipeline.close()
    pipeline.close()
    assert pipeline._renderer is None and not pipeline._queue


@pytest.mark.parametrize("kind", ["overhead", "wrist"])
def test_failed_context_activation_does_not_free_another_contexts_resources(kind):
    pipeline, _ = sensor(kind)
    released = []

    def fail_to_activate():
        raise RuntimeError("cannot activate owning context")

    renderer = SimpleNamespace(
        _gl_context=SimpleNamespace(make_current=fail_to_activate),
        close=lambda: released.append(True),
    )
    pipeline._renderer = renderer
    with pytest.raises(RuntimeError, match="cannot activate owning context"):
        pipeline.close()
    assert not released and pipeline._renderer is renderer
    pipeline._renderer = None
    pipeline.close()
