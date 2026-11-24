"""Validate framebuffer reuse against MuJoCo's public reference rendering."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.rgbd import can_reuse_rgb_depth, render_rgbd


def scene(*, samples=0, alpha=1, orthographic=False):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
    <visual><global offwidth="160" offheight="120"/>
    <quality offsamples="{samples}"/></visual>
    <worldbody><light pos="0 0 3"/>
    <camera name="camera" pos="0 0 1.5" fovy="{1.2 if orthographic else 60}"
            projection="{"orthographic" if orthographic else "perspective"}"/>
    <geom type="plane" size="2 2 .1" rgba=".2 .3 .4 1"/>
    <body name="moving" pos="-.1 0 .15"><joint type="slide" axis="1 0 0"/>
    <geom type="box" size=".08 .06 .1" rgba=".9 .2 .1 {alpha}"/></body>
    <geom type="box" pos=".03 .02 .4" size=".025 .12 .03" rgba=".3 .3 .3 1"/>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    return model, data


def prepare(renderer, data):
    mujoco.mj_forward(renderer.model, data)
    renderer.update_scene(data, camera="camera")
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False


def test_rgb_and_depth_match_reference_at_occlusion_edges_and_do_not_alias():
    model, data = scene()
    with mujoco.Renderer(model, height=120, width=160) as renderer:
        old = None
        for offset in (0.0, 0.08, 0.2):
            data.qpos[0] = offset
            prepare(renderer, data)
            if mujoco.__version__.split(".")[:2] == ["3", "17"]:
                assert can_reuse_rgb_depth(renderer)
            flags = renderer.scene.flags.copy()
            rgb, depth = render_rgbd(renderer)
            reference_rgb, reference_depth = render_rgbd(renderer, force_reference=True)
            np.testing.assert_array_equal(rgb, reference_rgb)
            np.testing.assert_array_equal(depth, reference_depth)
            np.testing.assert_array_equal(renderer.scene.flags, flags)
            assert not renderer._depth_rendering
            assert depth.dtype == np.float32 and depth.flags.c_contiguous
            assert rgb.dtype == np.uint8 and not np.shares_memory(rgb, depth)
            if old is not None:
                np.testing.assert_array_equal(old[0], old[2])
                np.testing.assert_array_equal(old[1], old[3])
            old = rgb, depth, rgb.copy(), depth.copy()


@pytest.mark.parametrize(
    "settings", [{"samples": 4}, {"alpha": 0.5}, {"orthographic": True}]
)
def test_special_scenes_use_reference_depth(settings):
    model, data = scene(**settings)
    with mujoco.Renderer(model, height=120, width=160) as renderer:
        prepare(renderer, data)
        assert not can_reuse_rgb_depth(renderer)
        actual = render_rgbd(renderer)
        expected = render_rgbd(renderer, force_reference=True)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)


def test_scene_opacity_is_rechecked_and_unknown_version_falls_back(monkeypatch):
    model, data = scene()
    with mujoco.Renderer(model, height=120, width=160) as renderer:
        prepare(renderer, data)
        renderer.scene.geoms[0].rgba[3] = 0.4
        assert not can_reuse_rgb_depth(renderer)
        renderer.scene.geoms[0].rgba[3] = 1
        monkeypatch.setattr(mujoco, "__version__", "3.18.0")
        assert not can_reuse_rgb_depth(renderer)


def test_unknown_renderer_copies_reused_buffers_and_restores_mode_on_error():
    buffer = np.zeros((2, 2), dtype=np.float32)
    state = {"depth": False, "fail": False}

    def render():
        if state["depth"] and state["fail"]:
            raise RuntimeError("lost context")
        buffer.fill(2 if state["depth"] else 1)
        return buffer

    renderer = SimpleNamespace(
        render=render,
        disable_depth_rendering=lambda: state.update(depth=False),
        enable_depth_rendering=lambda: state.update(depth=True),
    )
    rgb, depth = render_rgbd(renderer)
    assert np.all(rgb == 1) and np.all(depth == 2)
    assert not np.shares_memory(rgb, depth) and not state["depth"]
    state["fail"] = True
    with pytest.raises(RuntimeError, match="lost context"):
        render_rgbd(renderer)
    assert not state["depth"]


@pytest.mark.parametrize(
    "flag",
    [
        mujoco.mjtRndFlag.mjRND_SEGMENT,
        mujoco.mjtRndFlag.mjRND_IDCOLOR,
        mujoco.mjtRndFlag.mjRND_SHADOW,
        mujoco.mjtRndFlag.mjRND_REFLECTION,
        mujoco.mjtRndFlag.mjRND_WIREFRAME,
    ],
)
def test_special_render_flags_do_not_reuse_rgb_depth(flag):
    model, data = scene()
    with mujoco.Renderer(model, height=120, width=160) as renderer:
        prepare(renderer, data)
        renderer.scene.flags[flag] = True
        assert not can_reuse_rgb_depth(renderer)


@pytest.mark.parametrize("mutation", ["hidden", "unsegmented"])
def test_modified_scene_geometries_fall_back_to_reference(mutation):
    model, data = scene()
    with mujoco.Renderer(model, height=120, width=160) as renderer:
        prepare(renderer, data)
        for geom in renderer.scene.geoms[: renderer.scene.ngeom]:
            if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if mutation == "hidden":
                geom.rgba[3] = 0.0
                geom.transparent = False
            else:
                geom.segid = -1
        assert not can_reuse_rgb_depth(renderer)
        actual = render_rgbd(renderer)
        expected = render_rgbd(renderer, force_reference=True)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
