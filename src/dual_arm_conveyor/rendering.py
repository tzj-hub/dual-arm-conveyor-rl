"""Select an available OpenGL backend before MuJoCo is imported.

Driver/context failures are isolated in short-lived subprocesses. CUDA is not
used as a proxy for OpenGL availability: the actual GL renderer identifies
whether the selected graphics path is hardware accelerated.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time


_PROBE = r"""
import json, time
import mujoco
from OpenGL import GL
model = mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="160" offheight="120"/></visual><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
created = time.monotonic()
with mujoco.Renderer(model, height=120, width=160) as renderer:
    renderer.update_scene(data)
    renderer.render()
    info = {key: GL.glGetString(value).decode('utf-8', 'replace') for key, value in (
        ('renderer', GL.GL_RENDERER), ('vendor', GL.GL_VENDOR), ('version', GL.GL_VERSION))}
    info['context_and_first_render_seconds'] = time.monotonic() - created
    started = time.monotonic()
    for _ in range(3): renderer.render()
    info['probe_render_seconds'] = (time.monotonic() - started) / 3
    print(json.dumps(info), flush=True)
"""

_selection = None


def _probe(backend: str) -> dict:
    environment = os.environ.copy()
    environment.update(MUJOCO_GL=backend, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    try:
        process = subprocess.run(
            [sys.executable, "-c", _PROBE],
            env=environment,
            text=True,
            capture_output=True,
            timeout=8,
            check=False,
        )
        if process.returncode:
            return {
                "backend": backend,
                "available": False,
                "error": (process.stderr or process.stdout)[-1500:].strip(),
            }
        info = json.loads(process.stdout.strip().splitlines()[-1])
        renderer = info["renderer"].lower()
        software = any(
            name in renderer
            for name in ("llvmpipe", "softpipe", "swrast", "software", "swiftshader")
        )
        return {
            "backend": backend,
            "available": True,
            "hardware_accelerated": not software,
            **info,
        }
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError) as exc:
        return {"backend": backend, "available": False, "error": str(exc)}


def select_backend(requested: str = "auto") -> dict:
    """Respect explicit choices; otherwise prefer a verified hardware context."""
    global _selection
    if requested not in ("auto", "egl", "glfw", "osmesa"):
        raise ValueError("render backend must be auto, egl, glfw or osmesa")
    explicit = requested if requested != "auto" else os.environ.get("MUJOCO_GL")
    if "mujoco" in sys.modules:
        loaded = os.environ.get("MUJOCO_GL", "glfw")
        if explicit and explicit != loaded:
            raise ValueError("Select the render backend before importing MuJoCo")
        return _selection or {
            "backend": loaded,
            "selection": "already imported",
            "hardware_accelerated": None,
        }
    started = time.monotonic()
    attempts = []
    candidates = [explicit] if explicit else ["egl", "glfw", "osmesa"]
    for backend in candidates:
        if (
            backend == "glfw"
            and not os.environ.get("DISPLAY")
            and sys.platform.startswith("linux")
        ):
            attempts.append(
                {"backend": backend, "available": False, "error": "DISPLAY is not set"}
            )
            continue
        result = _probe(backend)
        attempts.append(result)
        if result["available"] and (explicit or result["hardware_accelerated"]):
            break
    available = [item for item in attempts if item["available"]]
    if not available:
        detail = "; ".join(
            item["backend"] + ": " + item["error"].splitlines()[-1] for item in attempts
        )
        raise ValueError("No working OpenGL backend. " + detail)
    hardware = [item for item in available if item["hardware_accelerated"]]
    # OSMesa is the predictable software fallback, irrespective of tiny probe
    # timing differences. A software EGL context does not imply GPU rendering.
    chosen = (
        hardware
        or [item for item in available if item["backend"] == "osmesa"]
        or available
    )[0]
    os.environ["MUJOCO_GL"] = chosen["backend"]
    _selection = {
        **chosen,
        "selection": "explicit" if explicit else "auto",
        "attempts": attempts,
        "selection_seconds": time.monotonic() - started,
    }
    print(
        json.dumps(
            {
                "event": "renderer_selected",
                **{
                    key: _selection[key]
                    for key in (
                        "backend",
                        "renderer",
                        "hardware_accelerated",
                        "selection",
                    )
                    if key in _selection
                },
            }
        ),
        flush=True,
    )
    return _selection


def metadata() -> dict:
    return dict(_selection or {"backend": os.environ.get("MUJOCO_GL", "unselected")})
