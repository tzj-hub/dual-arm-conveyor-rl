"""Portable trained-run entry point: previews and stubs never construct physics."""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dual_arm_conveyor.config import Config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_trained_script", ROOT / "scripts/run_trained.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def trained(tmp_path):
    run = tmp_path / "a training directory"
    run.mkdir()
    scene = asdict(Config(total_objects=3, pool_size=8, bin_capacity=20,
        num_arms=4, strategy="adaptive", deep_bins=True, auto_unload=False,
        sensor_mode="oracle", seed=123, physics_dt=.0025,
        future_planning=True, direct_retask=True, nearest_bin_slot=True,
        downstream_completion=True))
    metadata = dict(status="complete", env_config=scene,
        scheduler_action_metadata={"schema": "test_action", "groups": {"0": "keep"}},
        scheduler_observation_metadata={"schema": "test_observation"}, source_sha256={"env.py": "example"})
    (run / "training.json").write_text(json.dumps(metadata))
    (run / "best.pt").write_bytes(b"Preview deliberately does not deserialize this file")
    return run, metadata


def options(run, *arguments):
    return runner.parser().parse_args([*arguments, "--run", str(run)])


def test_preview_works_without_any_site_packages_or_physics(trained, tmp_path):
    run, metadata = trained
    output = tmp_path / "not-created" / "evaluation.json"
    result = subprocess.run([sys.executable, "-S", "-B", str(ROOT / "scripts/run_trained.py"),
        "evaluate", "--run", str(run), "--output", str(output), "--seed", "1700000"],
        capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["config"] == metadata["env_config"]
    assert plan["arrival_seed"] == 1700000
    assert plan["inference_device"] == "cpu" and plan["episodes"] == 1
    assert not plan["execute"] and not plan["checkpoint_interface_verified"]
    assert not plan["training_started"] and not plan["physics_created"]
    assert not output.parent.exists()


@pytest.mark.parametrize("mode", ["print-config", "play", "record", "evaluate"])
def test_complete_scene_reaches_cli_and_only_explicit_catalogue_changes(trained, tmp_path, mode):
    run, metadata = trained
    extra = ["--output", str(tmp_path / ("out.mp4" if mode == "record" else "out.json"))] if mode in ("record", "evaluate") else []
    opts = options(run, mode, "--catalog-seed", "876", "--render-backend", "osmesa", *extra)
    saved, plan = runner.prepare(opts)
    assert plan["config"] == dict(metadata["env_config"], seed=876)
    assert plan["scene_changes"] == {"seed": {"training": 123, "selected": 876}}
    args = runner.cli_arguments(opts, plan["config"], plan["checkpoint"])
    assert asdict(runner.cli.config_from_args(args)) == plan["config"]
    assert args.render_backend == "osmesa"
    assert args.policy == str(run / "best.pt")
    if mode == "evaluate":
        assert args.episodes == 1
    else:
        assert args.speed == 1.0 and args.camera_view == "off"


@pytest.mark.parametrize("change", ["missing", "unknown", "unmapped_nondefault"])
def test_incomplete_or_unrestorable_scene_is_rejected(trained, change):
    run, metadata = trained
    if change == "missing":
        metadata["env_config"].pop("direct_retask")
    elif change == "unknown":
        metadata["env_config"]["future_unsupported_option"] = True
    else:
        metadata["env_config"]["grasp_offset"] = -.008
    (run / "training.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="complete.*scene|complete current Config"):
        runner.prepare(options(run))


@pytest.mark.parametrize("which", ["video", "report", "launch"])
def test_record_never_overwrites_video_or_reports(trained, tmp_path, which):
    run, _ = trained
    output = tmp_path / "video.mp4"
    paths = {"video": output, "report": output.with_suffix(".json"),
             "launch": output.with_name("video.mp4.launch.json")}
    paths[which].write_bytes(b"keep")
    with pytest.raises(ValueError, match="already exists"):
        runner.prepare(options(run, "record", "--output", str(output)))
    assert paths[which].read_bytes() == b"keep"


def test_requested_checkpoint_is_not_silently_substituted(trained):
    run, _ = trained
    with pytest.raises(ValueError, match="Checkpoint missing"):
        runner.prepare(options(run, "--checkpoint", "latest"))
    (run / "latest.pt").write_bytes(b"explicit latest")
    _, plan = runner.prepare(options(run, "--checkpoint", "latest"))
    assert plan["checkpoint"] == str(run / "latest.pt")


def test_multiepisode_argument_is_not_accepted(trained):
    run, _ = trained
    with pytest.raises(SystemExit):
        options(run, "evaluate", "--episodes", "2")


def test_preview_requires_explicit_output_and_plausible_frame_rate(trained, tmp_path):
    run, _ = trained
    with pytest.raises(ValueError, match="require --output"):
        runner.prepare(options(run, "record"))
    with pytest.raises(ValueError, match="physics update"):
        runner.prepare(options(run, "record", "--output", str(tmp_path / "x.mp4"), "--fps", "1000"))
    with pytest.raises(ValueError, match="read-only"):
        runner.prepare(options(run, "--execute"))


def test_record_restores_frame_parameters_without_changing_physics(trained, tmp_path):
    run, metadata = trained
    opts = options(run, "record", "--output", str(tmp_path / "x.mp4"),
                   "--fps", "60", "--width", "1280", "--height", "720")
    _, plan = runner.prepare(opts)
    args = runner.cli_arguments(opts, plan["config"], plan["checkpoint"])
    assert (args.fps, args.width, args.height) == (60, 1280, 720)
    assert plan["config"] == metadata["env_config"]


@pytest.fixture
def policy_stub(monkeypatch, trained):
    run, metadata = trained
    actual = json.loads(json.dumps(metadata))
    actual["scheduler_action_metadata"]["groups"] = {0: "keep"}
    scheduler = SimpleNamespace(
        action_metadata=lambda env: metadata["scheduler_action_metadata"],
        observation_metadata=lambda env: metadata["scheduler_observation_metadata"])
    monkeypatch.setitem(sys.modules, "dual_arm_conveyor.scheduler", scheduler)
    policy = SimpleNamespace(metadata=actual)
    return policy, metadata


def test_actual_policy_and_current_interfaces_are_checked_before_physics(policy_stub):
    policy, metadata = policy_stub
    runner.validate_policy(policy, metadata, metadata["env_config"])
    policy.metadata["env_config"]["belt_speed"] = .4
    with pytest.raises(ValueError, match="scene differs"):
        runner.validate_policy(policy, metadata, metadata["env_config"])


@pytest.mark.parametrize("key", ["scheduler_action_metadata", "scheduler_observation_metadata"])
def test_mismatched_actual_checkpoint_interface_is_rejected(policy_stub, key):
    policy, metadata = policy_stub
    policy.metadata[key]["schema"] = "wrong"
    with pytest.raises(ValueError, match="differs from training.json"):
        runner.validate_policy(policy, metadata, metadata["env_config"])


def test_matching_metadata_cannot_override_current_scheduler_schema(policy_stub, monkeypatch):
    policy, metadata = policy_stub
    monkeypatch.setitem(sys.modules, "dual_arm_conveyor.scheduler", SimpleNamespace(
        action_metadata=lambda env: {"schema": "newer_incompatible"},
        observation_metadata=lambda env: metadata["scheduler_observation_metadata"]))
    with pytest.raises(ValueError, match="incompatible with the current scheduler"):
        runner.validate_policy(policy, metadata, metadata["env_config"])


def test_execute_checks_and_uses_cpu_without_training(trained, tmp_path, policy_stub, monkeypatch):
    run, _ = trained
    policy, _ = policy_stub
    opts = options(run, "evaluate", "--output", str(tmp_path / "new" / "eval.json"), "--execute")
    metadata, plan = runner.prepare(opts)
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda count: calls.append(("threads", count))))
    monkeypatch.setitem(sys.modules, "dual_arm_conveyor.learning", SimpleNamespace(
        load_policy=lambda path, device: (calls.append(("load", device)) or policy)))
    fake = SimpleNamespace(func=lambda args: (calls.append(("execute", "single episode")) or {}))
    monkeypatch.setattr(runner, "cli_arguments", lambda *args: fake)
    assert runner.execute(opts, metadata, plan) == 0
    assert calls == [("threads", 1), ("load", "cpu"), ("execute", "single episode")]
    saved = json.loads((tmp_path / "new/eval.json.launch.json").read_text())
    assert saved["status"] == "complete" and saved["checkpoint_interface_verified"]
    assert saved["training_started"] is False


def test_changed_checkpoint_is_rejected_before_importing_runtime(trained):
    run, _ = trained
    opts = options(run, "play", "--execute")
    metadata, plan = runner.prepare(opts)
    (run / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after preview"):
        runner.execute(opts, metadata, plan)


def test_shell_entry_accepts_explicit_interpreter_and_preserves_caller_paths(trained, tmp_path):
    run, _ = trained
    result = subprocess.run(["bash", str(ROOT / "scripts/run_trained.sh"), "--run", run.name],
        cwd=run.parent, env=dict(os.environ, CONVEYOR_PYTHON=sys.executable), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["run"] == str(run)
