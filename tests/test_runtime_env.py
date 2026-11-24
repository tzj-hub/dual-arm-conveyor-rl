"""The optional native math workaround must preserve user runtime settings."""

import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize(
    ("enabled", "original", "expected"),
    [
        ("0", "glibc.malloc.trim_threshold=1024", "glibc.malloc.trim_threshold=1024"),
        ("1", "", "glibc.cpu.hwcaps=-FMA"),
        (
            "1",
            "glibc.malloc.trim_threshold=1024",
            "glibc.malloc.trim_threshold=1024:glibc.cpu.hwcaps=-FMA",
        ),
        (
            "1",
            "glibc.cpu.hwcaps=-AVX2:glibc.malloc.trim_threshold=1024",
            "glibc.cpu.hwcaps=-AVX2,-FMA:glibc.malloc.trim_threshold=1024",
        ),
    ],
)
def test_native_math_compatibility_is_optional_and_preserves_settings(
    enabled, original, expected
):
    script = Path(__file__).resolve().parents[1] / "scripts" / "runtime_env.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; printf "%s" "${GLIBC_TUNABLES:-}"',
            "bash",
            str(script),
        ],
        env={**os.environ, "CONVEYOR_LIBM_COMPAT": enabled, "GLIBC_TUNABLES": original},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == expected
