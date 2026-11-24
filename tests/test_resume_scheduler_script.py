"""Resume metadata tests; all training calls are replaced with recorders."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "resume_scheduler_script", Path(__file__).parents[1] / "scripts" / "resume_scheduler.py"
)
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


@pytest.fixture
def run_evidence(tmp_path, monkeypatch):
    run = tmp_path / "run with spaces"
    run.mkdir()
    sources = {"learning.py": "frozen", "hierarchical_train.py": "frozen"}
    monkeypatch.setattr(resume.train, "source_hashes", lambda: sources)
    args = resume.train.parser().parse_args([
        "--output", str(run), "--steps", "2048", "--warmstart-epochs", "40",
        "--rollout-steps", "1024", "--minibatch-size", "256", "--learning-rate", "0.0001",
        "--entropy-coef", "0.02", "--gae-lambda", "0.995", "--seed", "140000",
        "--catalog-seeds", "901", "902", "903", "904", "--catalog-seed", "901",
        "--validation-seed", "240000", "--validation-catalog-seed", "1901",
        "--eval-seed", "340000", "--eval-catalog-seed", "2901", "--eval-bc",
        "--reset-optimizer-on-resume", "--max-time", "300",
    ])
    metadata = {"source_sha256": sources, "env_config": asdict(resume.train.scene_config(args)),
                "arguments": vars(args)}
    checkpoint = {"version": resume.CHECKPOINT_VERSION, "steps": 1024,
                  "ppo_optimizer_steps": 16, "env_config": metadata["env_config"],
                  "train_config": {"seed": args.seed},
                  "hierarchical_training": {"source_sha256": sources, "catalog_bank": args.catalog_seeds}}
    (run / "training.json").write_text(json.dumps(metadata, default=str))
    (run / "selection.json").write_text('{"selected": null}\n')
    torch.save(checkpoint, run / "latest.pt")
    return run, metadata, checkpoint


def test_roundtrip_preserves_saved_hyperparameters_and_clears_one_time_reset(run_evidence):
    run, metadata, _ = run_evidence
    plan, snapshots = resume.prepare_resume(run, steps=4096, device="cuda")
    restored = vars(resume.train.parser().parse_args(plan["argv"]))
    for key, value in metadata["arguments"].items():
        if key not in {"output", "resume", "steps", "device", "reset_optimizer_on_resume"}:
            assert restored[key] == value
    assert restored["steps"] == 4096 and restored["device"] == "cuda"
    assert restored["resume"] == run / "latest.pt"
    assert restored["output"] == run and restored["reset_optimizer_on_resume"] is False
    assert plan["checkpoint_sha256"] == resume.sha256(snapshots["resume_checkpoint.pt"])
    assert not (run / "resume_history").exists()


def test_checkpoint_precedence_and_bc_reset(run_evidence):
    run, metadata, checkpoint = run_evidence
    bc = dict(checkpoint, steps=0, ppo_optimizer_steps=0,
              warmstart_metrics={"physical_steps": metadata["arguments"]["warmstart_steps"],
                                 "decision_examples": 100, "optimizer_steps": 20})
    torch.save(bc, run / "bc.pt")
    assert Path(resume.prepare_resume(run)[0]["checkpoint"]).name == "latest.pt"
    (run / "latest.pt").unlink()
    plan, _ = resume.prepare_resume(run)
    assert Path(plan["checkpoint"]).name == "bc.pt"
    assert plan["reset_optimizer"] and "--reset-optimizer-on-resume" in plan["argv"]
    (run / "bc.pt").unlink()
    with pytest.raises(ValueError, match="Neither latest.pt nor bc.pt"):
        resume.prepare_resume(run)


@pytest.mark.parametrize("steps", [0, 1024, -1])
def test_resume_requires_remaining_ppo_work(run_evidence, steps):
    with pytest.raises(ValueError, match="must exceed"):
        resume.prepare_resume(run_evidence[0], steps=steps)


@pytest.mark.parametrize("field", ["metadata_source", "checkpoint_source", "scene", "seed", "bank", "arguments"])
def test_inconsistent_evidence_is_rejected(run_evidence, field):
    run, metadata, checkpoint = run_evidence
    if field == "metadata_source":
        metadata["source_sha256"] = {"other": "different"}
    elif field == "checkpoint_source":
        checkpoint["hierarchical_training"]["source_sha256"] = {"other": "different"}
    elif field == "scene":
        metadata["env_config"] = {**metadata["env_config"], "belt_speed": 0.9}
    elif field == "seed":
        checkpoint["train_config"]["seed"] = 200
    elif field == "bank":
        checkpoint["hierarchical_training"]["catalog_bank"] = [1, 2]
    else:
        metadata["arguments"]["arbitrary_shell"] = "never execute this"
    (run / "training.json").write_text(json.dumps(metadata, default=str))
    torch.save(checkpoint, run / "latest.pt")
    with pytest.raises(ValueError):
        resume.prepare_resume(run)
    assert not (run / "resume_history").exists()


def test_backup_retains_exact_old_evidence_and_never_overwrites(run_evidence):
    run, _, _ = run_evidence
    plan, snapshots = resume.prepare_resume(run)
    first = resume.backup_metadata(plan, snapshots)
    second = resume.backup_metadata(plan, snapshots)
    assert first != second and first.parent == run / "resume_history"
    for name, data in snapshots.items():
        assert (first / name).read_bytes() == data == (second / name).read_bytes()
    assert (run / "training.json").read_bytes() == snapshots["training.json"]
    assert json.loads((first / "resume_plan.json").read_text())["argv"] == plan["argv"]


def test_default_dry_run_does_not_mutate_or_call_training(run_evidence, monkeypatch, capsys):
    run = run_evidence[0]
    monkeypatch.setattr(resume.train, "main", lambda argv: pytest.fail("dry-run started training"))
    assert resume.main(["--run", str(run)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["execute"] is False and output["target_steps"] == 2048
    assert not (run / "resume_history").exists()


def test_execute_backs_up_before_calling_training_recorder(run_evidence, monkeypatch, capsys):
    run = run_evidence[0]
    calls = []
    monkeypatch.setattr(resume, "running_training_pids", lambda path: [])

    def record(argv):
        histories = list((run / "resume_history").iterdir())
        assert len(histories) == 1
        assert (histories[0] / "training.json").exists()
        assert (histories[0] / "resume_checkpoint.pt").exists()
        calls.append(argv)

    monkeypatch.setattr(resume.train, "main", record)
    assert resume.main(["--run", str(run), "--execute"]) == 0
    assert len(calls) == 1
    capsys.readouterr()


def test_active_training_prevents_execute_and_backup(run_evidence, monkeypatch, capsys):
    run = run_evidence[0]
    monkeypatch.setattr(resume, "running_training_pids", lambda path: [12345])
    monkeypatch.setattr(resume.train, "main", lambda argv: pytest.fail("concurrent training"))
    with pytest.raises(RuntimeError, match="already active"):
        resume.main(["--run", str(run), "--execute"])
    assert not (run / "resume_history").exists()
    capsys.readouterr()


def test_proc_detection_resolves_relative_output_and_resume_commands(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    run = tmp_path / "experiment"
    run.mkdir()
    commands = {
        777001: ["python", "-m", "dual_arm_conveyor.hierarchical_train", "--output", "experiment"],
        777002: ["python", "/project/scripts/resume_scheduler.py", "--run=" + str(run), "--execute"],
        777003: ["python", "-m", "dual_arm_conveyor.hierarchical_train", "--output=elsewhere"],
        777004: ["python", "/project/scripts/resume_scheduler.py", "--run=" + str(run)],
        777005: ["python", "unrelated.py", "--output", str(run)],
    }
    for pid, argv in commands.items():
        directory = proc / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(b"\0".join(token.encode() for token in argv) + b"\0")
        (directory / "cwd").symlink_to(tmp_path, target_is_directory=True)
    assert sorted(resume.running_training_pids(run, proc)) == [777001, 777002]


def test_interrupted_bc_collection_cannot_silently_skip_warmstart(run_evidence):
    run, _, checkpoint = run_evidence
    torch.save(dict(checkpoint, steps=0, ppo_optimizer_steps=0, warmstart_metrics={}), run / "latest.pt")
    with pytest.raises(ValueError, match="BC collection/optimization did not finish"):
        resume.prepare_resume(run)
    assert not (run / "resume_history").exists()


def test_active_training_is_detected_even_before_first_checkpoint(run_evidence, monkeypatch):
    run = run_evidence[0]
    (run / "latest.pt").unlink()
    monkeypatch.setattr(resume, "running_training_pids", lambda path: [12345])
    monkeypatch.setattr(resume.train, "main", lambda argv: pytest.fail("concurrent training"))
    with pytest.raises(RuntimeError, match="already active"):
        resume.main(["--run", str(run), "--execute"])
    assert not (run / "resume_history").exists()


def test_resume_helpers_use_an_exclusive_lock(run_evidence, monkeypatch):
    run = run_evidence[0]
    monkeypatch.setattr(resume, "running_training_pids", lambda path: [])
    monkeypatch.setattr(resume.train, "main", lambda argv: pytest.fail("concurrent helper"))
    with resume.resume_lock(run):
        with pytest.raises(RuntimeError, match="holds this run's lock"):
            resume.main(["--run", str(run), "--execute"])
    assert not (run / "resume_history").exists()


def add_bc_evidence(run, metadata, checkpoint, *, validated=True):
    warmstart = {"physical_steps": metadata["arguments"]["warmstart_steps"],
                 "decision_examples": 100, "optimizer_steps": 20}
    bc = dict(checkpoint, steps=0, ppo_optimizer_steps=0, ppo_actor_optimizer_steps=0,
              warmstart_metrics=warmstart)
    torch.save(bc, run / "bc.pt")
    checkpoint["warmstart_metrics"] = warmstart
    torch.save(checkpoint, run / "latest.pt")
    metadata["warmstart"] = warmstart
    metadata["arguments"]["validate_bc"] = True
    metadata["actual_training_episode_seeds"] = [140000, 140001]
    metadata["actual_training_catalogue_visits"] = [
        {"episode_seed": 140000, "catalog_seed": 901},
        {"episode_seed": 140001, "catalog_seed": 902},
    ]
    (run / "training.json").write_text(json.dumps(metadata, default=str))
    summary = {"complete": True, "success_rate": 0.8}
    if validated:
        (run / "bc_validation.json").write_text(json.dumps({
            "source_sha256": metadata["source_sha256"],
            "checkpoint_sha256": resume.sha256((run / "bc.pt").read_bytes()),
            "evaluation": {"summary": summary, "episodes": [{"seed": 240000}]},
        }))
    return warmstart, summary


@pytest.mark.parametrize("validated", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
def test_resume_preserves_bc_and_training_history_without_inventing_updates(
    run_evidence, monkeypatch, validated, interrupted,
):
    run, metadata, checkpoint = run_evidence
    warmstart, summary = add_bc_evidence(run, metadata, checkpoint, validated=validated)
    monkeypatch.setattr(resume, "running_training_pids", lambda path: [])

    def record(argv):
        # Model the trainer creating segment metadata from scratch.
        plan = json.loads(next((run / "resume_history").glob("*/resume_plan.json")).read_text())
        current = {"source_sha256": metadata["source_sha256"],
                   "resume_checkpoint_sha256": plan["checkpoint_sha256"],
                   "steps": 1500, "actor_optimizer_steps": 23, "wall_seconds": 12,
                   "actual_training_episode_seeds": [140000, 140002],
                   "actual_training_catalogue_visits": [
                       {"episode_seed": 140000, "catalog_seed": 901},
                       {"episode_seed": 140002, "catalog_seed": 903}],
                   "status": "interrupted" if interrupted else "complete"}
        (run / "training.json").write_text(json.dumps(current))
        if interrupted:
            raise KeyboardInterrupt()

    monkeypatch.setattr(resume.train, "main", record)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            resume.main(["--run", str(run), "--execute"])
    else:
        assert resume.main(["--run", str(run), "--execute"]) == 0
    restored = json.loads((run / "training.json").read_text())
    assert restored["warmstart"] == warmstart and restored["bc_checkpoint"] == "bc.pt"
    assert restored["actual_training_episode_seeds"] == [140000, 140001, 140002]
    assert len(restored["actual_training_catalogue_visits"]) == 3
    assert restored["steps"] == 1500 and restored["actor_optimizer_steps"] == 23
    assert restored["wall_seconds"] == 12
    if validated:
        assert restored["bc_validation_summary"] == summary
        assert restored["resume_evidence"]["bc_validation_status"] == "recorded"
    else:
        assert "bc_validation_summary" not in restored
        assert "pending" in restored["resume_evidence"]["bc_validation_status"]


def test_mismatched_bc_validation_is_rejected(run_evidence):
    run, metadata, checkpoint = run_evidence
    add_bc_evidence(run, metadata, checkpoint)
    path = run / "bc_validation.json"
    evidence = json.loads(path.read_text())
    evidence["evaluation"]["episodes"] = [{"seed": 340000}]
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="BC validation does not match"):
        resume.prepare_resume(run)
