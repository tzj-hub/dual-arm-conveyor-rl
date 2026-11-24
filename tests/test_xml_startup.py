"""Test native XML/graphics load order in a fresh interpreter, not pytest's cache."""

import ctypes.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux graphics library collision")
@pytest.mark.parametrize(
    "backend,library", [("egl", "EGL"), ("glfw", "GL"), ("osmesa", "OSMesa")]
)
def test_xml_parser_is_ready_before_graphics_and_panda_scene_loading(backend, library):
    if not ctypes.util.find_library(library):
        pytest.skip(f"{library} is unavailable")
    source = Path(__file__).resolve().parents[1] / "src"
    script = """
import os
import sys
settings = {key: os.environ.get(key) for key in
            ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'MUJOCO_GL')}
import dual_arm_conveyor
assert 'pyexpat' in sys.modules, 'XML native runtime was not initialized early'
import mujoco
import xml.etree.ElementTree as ET
assert ET.fromstring('<scene><arm/></scene>').find('arm') is not None
from dual_arm_conveyor.config import Config
from dual_arm_conveyor.model import build_scene
model, ids = build_scene(Config(strategy='adaptive', num_arms=4,
                               gripper_mode='vacuum', bin_capacity=24))
assert len(ids.ee_sites) == 4 and model.nq > 0
assert settings == {key: os.environ.get(key) for key in settings}
print('XML and four-arm scene ready')
"""
    environment = {**os.environ, "MUJOCO_GL": backend}
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source), environment.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "XML and four-arm scene ready" in result.stdout
