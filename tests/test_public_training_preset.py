"""The public training entry must work after relocation, without old runs."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_fresh_preview_in_relocated_directory_without_site_packages(tmp_path):
    relocated = tmp_path / "new clone with spaces"
    (relocated / "scripts").mkdir(parents=True)
    (relocated / "configs").mkdir()
    (relocated / "src/dual_arm_conveyor").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/train_sequence.py", relocated / "scripts")
    shutil.copy2(ROOT / "configs/sequence_r3_fresh.json", relocated / "configs")
    output = tmp_path / "never-created"
    result = subprocess.run([sys.executable, "-S", "-B", str(relocated / "scripts/train_sequence.py"),
        "--output", str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.JSONDecoder().raw_decode(result.stdout)[0]
    assert plan["status"] == "preview" and not output.exists()
    assert plan["arguments"]["initialize_from"] is None
    assert plan["arguments"]["resume"] is None
    assert plan["arguments"]["warmstart_steps"] > 0
    assert str(relocated / "scripts/runtime_env.sh") in plan["command"]


def test_default_preset_is_accepted_by_real_training_parser(tmp_path):
    from dual_arm_conveyor import hierarchical_train as training
    spec = importlib.util.spec_from_file_location("public_training", ROOT / "scripts/train_sequence.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    from argparse import Namespace
    plan = launch.prepare(Namespace(config=ROOT / "configs/sequence_r3_fresh.json", output=tmp_path / "run",
        steps=None, device=None, warmstart_steps=None, warmstart_max_examples=None,
        expert_balance=None, initialize_from=None))
    command = plan["command"]
    parsed = training.parser().parse_args(command[command.index("dual_arm_conveyor.hierarchical_train") + 1:])
    training.validate_splits(parsed)
    scene = training.scene_config(parsed)
    assert scene.future_planning and scene.downstream_completion
    assert scene.sensor_mode == "oracle" and scene.num_arms == 4
    assert not (tmp_path / "run").exists()


def test_preview_refuses_existing_training_directory(tmp_path):
    run = tmp_path / "existing"
    run.mkdir()
    marker = run / "keep.txt"
    marker.write_text("original")
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/train_sequence.py"),
        "--output", str(run)], capture_output=True, text=True)
    assert result.returncode != 0 and "Output already exists" in result.stderr
    assert marker.read_text() == "original"
