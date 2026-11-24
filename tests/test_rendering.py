"""Backend failures stay outside the physics process; EGL may still be software."""

from types import SimpleNamespace
import sys
import subprocess

import pytest

from dual_arm_conveyor import rendering


@pytest.fixture
def unloaded(monkeypatch):
    monkeypatch.delitem(sys.modules, "mujoco", raising=False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(rendering, "_selection", None)


def test_auto_selects_actual_hardware_renderer_before_software(unloaded, monkeypatch):
    calls = []

    def probe(backend):
        calls.append(backend)
        return {
            "backend": backend,
            "available": True,
            "hardware_accelerated": backend == "glfw",
            "renderer": "NVIDIA" if backend == "glfw" else "llvmpipe",
        }

    monkeypatch.setattr(rendering, "_probe", probe)
    selected = rendering.select_backend()
    assert selected["backend"] == "glfw" and selected["hardware_accelerated"]
    assert calls == ["egl", "glfw"]


def test_egl_software_is_not_reported_as_gpu(unloaded, monkeypatch):
    def probe(backend):
        return {
            "backend": backend,
            "available": backend != "glfw",
            "hardware_accelerated": False,
            "renderer": "llvmpipe",
            "error": "display unavailable",
        }

    monkeypatch.setattr(rendering, "_probe", probe)
    selected = rendering.select_backend()
    assert selected["backend"] == "osmesa"
    assert selected["hardware_accelerated"] is False


def test_explicit_backend_is_respected(unloaded, monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    calls = []

    def probe(backend):
        calls.append(backend)
        return {"backend": backend, "available": True, "hardware_accelerated": False}

    monkeypatch.setattr(rendering, "_probe", probe)
    assert rendering.select_backend()["selection"] == "explicit"
    assert calls == ["osmesa"]


def test_probe_failure_and_timeout_are_reported_without_parent_failure(monkeypatch):
    monkeypatch.setattr(
        rendering.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=-11, stderr="driver crashed", stdout=""
        ),
    )
    assert rendering._probe("egl") == {
        "backend": "egl",
        "available": False,
        "error": "driver crashed",
    }

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("probe", 8)

    monkeypatch.setattr(rendering.subprocess, "run", timeout)
    assert rendering._probe("egl")["available"] is False


def test_backend_change_after_mujoco_import_is_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace())
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    with pytest.raises(ValueError, match="before importing"):
        rendering.select_backend("egl")
