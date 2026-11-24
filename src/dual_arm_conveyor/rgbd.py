# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for this project: guarded RGB framebuffer reuse, public-API fallback,
# and independent RGB-D outputs. Reverse-Z conversion adapted from MuJoCo 3.17.
"""Read an opaque RGB-D camera from one classic MuJoCo rasterization.

The fast path is deliberately limited to the verified MuJoCo 3.17 renderer,
non-MSAA perspective views, and opaque geometry without shadows/reflections.
All other configurations use the public two-pass Renderer API. In particular,
MSAA RGB-buffer depth is *not* equivalent to MuJoCo's segmentation depth pass.

Reverse-Z conversion follows MuJoCo's Apache-2.0 Renderer implementation
(Copyright 2022 DeepMind Technologies Limited); no simulation state is read.
"""

from __future__ import annotations

import mujoco
import numpy as np


def can_reuse_rgb_depth(renderer) -> bool:
    """Check the current scene, not just the scene at renderer construction."""
    if (
        mujoco.__version__.split(".")[:2] != ["3", "17"]
        or type(renderer).__module__ != "mujoco.rendering.classic.renderer"
    ):
        return False
    try:
        context = renderer._mjr_context
        scene = renderer.scene
        if (
            context is None
            or renderer._gl_context is None
            or renderer._depth_rendering
            or renderer._segmentation_rendering
            or context.currentBuffer != mujoco.mjtFramebuffer.mjFB_OFFSCREEN
            or context.offSamples != 0
            or renderer.model.vis.quality.offsamples != 0
            or context.readDepthMap != mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
            or scene.stereo != mujoco.mjtStereo.mjSTEREO_NONE
            or any(camera.orthographic for camera in scene.camera)
        ):
            return False
        if any(
            scene.flags[flag]
            for flag in (
                mujoco.mjtRndFlag.mjRND_SEGMENT,
                mujoco.mjtRndFlag.mjRND_IDCOLOR,
                mujoco.mjtRndFlag.mjRND_SHADOW,
                mujoco.mjtRndFlag.mjRND_REFLECTION,
                mujoco.mjtRndFlag.mjRND_WIREFRAME,
            )
        ):
            return False
        # Alpha/materials can change after creation; inspect the resolved scene.
        return not any(
            geom.transparent or geom.rgba[3] != 1.0 or geom.segid < 0
            for geom in scene.geoms[: scene.ngeom]
        )
    except AttributeError:
        # Different native bindings/backends retain the supported public API.
        return False


def render_rgbd(renderer, *, force_reference=False):
    """Return independent RGB/depth images, leaving depth rendering disabled.

    The RGB render makes its GL context current. Read its depth immediately,
    before another renderer/viewer can replace the active context. Context and
    GL errors propagate; they are never converted into stale or synthetic data.
    """
    renderer.disable_depth_rendering()
    reusable = not force_reference and can_reuse_rgb_depth(renderer)
    rgb = renderer.render()
    if not reusable:
        # Unknown renderers may return reusable internal buffers.
        rgb = rgb.copy()
        renderer.enable_depth_rendering()
        try:
            depth = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
        return rgb, depth

    depth = np.empty((renderer.height, renderer.width), dtype=np.float32)
    mujoco.mjr_readPixels(None, depth, renderer._rect, renderer._mjr_context)
    extent = renderer.model.stat.extent
    near = np.float32(renderer.model.vis.map.znear * extent)
    far = np.float32(renderer.model.vis.map.zfar * extent)
    c_coef = -(far + near) / (far - near)
    d_coef = -(np.float32(2) * far * near) / (far - near)
    c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
    d_coef = np.float32(-0.5) * d_coef
    # Match the upstream float32 projection, float64 inversion and final cast.
    metric = d_coef / (depth.astype(np.float64) + c_coef)
    depth[:] = metric.astype(np.float32)[::-1]
    return rgb, depth
