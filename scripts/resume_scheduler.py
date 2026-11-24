#!/usr/bin/env python3
"""Prepare an audited same-source resume; dry-run unless --execute is given."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import shlex
import sys
import tempfile
from dataclasses import asdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from dual_arm_conveyor import hierarchical_train as train  # noqa: E402
from dual_arm_conveyor.learning import CHECKPOINT_VERSION  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run", type=Path, required=True)
    result.add_argument("--steps", type=int, help="New cumulative PPO step target; defaults to the saved target")
    result.add_argument("--device", help="Override saved device, for example cpu or cuda")
    result.add_argument("--execute", action="store_true", help="Back up evidence and actually resume training")
    return result


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def arguments_to_argv(arguments):
    """Serialize only current parser actions, without invoking a shell."""
    current = train.parser()
    actions = {action.dest: action for action in current._actions if action.dest != "help"}
    if set(arguments) != set(actions):
        raise ValueError("Saved training arguments do not match the current parser")
    argv = []
    for dest, action in actions.items():
        value = arguments[dest]
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                raise ValueError(f"Saved {dest} must be boolean")
            if value:
                argv.append(flag)
        elif isinstance(action, argparse._StoreAction):
            if value is None:
                if action.default is not None:
                    raise ValueError(f"Saved {dest} unexpectedly has no value")
                continue
            if action.nargs in ("+", "*"):
                if not isinstance(value, list) or (action.nargs == "+" and not value):
                    raise ValueError(f"Saved {dest} requires a nonempty list")
                argv.extend([flag, *map(str, value)])
            elif action.nargs is None:
                if isinstance(value, (list, dict, bool)):
                    raise ValueError(f"Saved {dest} requires a scalar")
                # --option=value also preserves a string that starts with '-'.
                argv.append(f"{flag}={value}")
            else:
                raise ValueError(f"Unsupported parser action for {dest}")
        else:
            raise ValueError(f"Unsupported parser action for {dest}")
    current.parse_args(argv)  # Run the original type/choice validation now.
    return argv


def prepare_resume(run, steps=None, device=None):
    run = Path(run).resolve()
    metadata_data = (run / "training.json").read_bytes()
    metadata = json.loads(metadata_data)
    sources = train.source_hashes()
    if not sources or metadata.get("source_sha256") != sources:
        raise ValueError("Training sources differ; use the formal training CLI with a new output directory")
    checkpoint_path = next((run / name for name in ("latest.pt", "bc.pt") if (run / name).is_file()), None)
    if checkpoint_path is None:
        raise ValueError("Neither latest.pt nor bc.pt exists; no saved checkpoint can be resumed")
    checkpoint_data = checkpoint_path.read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location="cpu", weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    if checkpoint.get("hierarchical_training", {}).get("source_sha256") != sources:
        raise ValueError("Checkpoint sources differ; use the formal training CLI with a new output directory")
    saved = metadata.get("arguments")
    if not isinstance(saved, dict):
        raise ValueError("training.json does not contain saved training arguments")
    arguments = dict(saved)
    current_steps = checkpoint.get("steps")
    target = saved.get("steps") if steps is None else steps
    if (isinstance(current_steps, bool) or not isinstance(current_steps, int) or current_steps < 0
            or isinstance(target, bool) or not isinstance(target, int) or target <= current_steps):
        raise ValueError("The target steps must exceed the saved checkpoint's cumulative PPO steps")
    pre_ppo = int(checkpoint.get("ppo_optimizer_steps", 0)) == 0
    if checkpoint_path.name == "bc.pt" and (current_steps or not pre_ppo):
        raise ValueError("bc.pt is not a pre-PPO checkpoint")
    warmstart = checkpoint.get("warmstart_metrics", {})
    if pre_ppo and saved.get("warmstart_steps", 0) > 0 and not (
        isinstance(warmstart, dict)
        and warmstart.get("physical_steps") == saved["warmstart_steps"]
        and warmstart.get("decision_examples", 0) > 0
        and warmstart.get("optimizer_steps", 0) > 0
    ):
        raise ValueError(
            "BC collection/optimization did not finish in this checkpoint; its partial "
            "demonstrations cannot be resumed. Preserve this run and restart the "
            "documented warmstart in a new run rather than skipping BC."
        )
    arguments.update(output=str(run), resume=str(checkpoint_path), steps=target,
                     reset_optimizer_on_resume=pre_ppo)
    if device is not None:
        arguments["device"] = device
    argv = arguments_to_argv(arguments)
    parsed = train.parser().parse_args(argv)
    validation_seeds, _ = train.validate_splits(parsed)
    bank = train.training_catalogs(parsed)
    parsed.catalog_seed = bank[0]
    expected_scene = asdict(train.scene_config(parsed))
    if metadata.get("env_config") != expected_scene or checkpoint.get("env_config") != expected_scene:
        raise ValueError("Saved arguments, metadata and checkpoint describe different scenes")
    if (checkpoint.get("train_config", {}).get("seed") != parsed.seed
            or checkpoint.get("hierarchical_training", {}).get("catalog_bank") != bank):
        raise ValueError("Checkpoint training seed/catalogue bank differs from saved arguments")
    snapshots = {"training.json": metadata_data, "resume_checkpoint.pt": checkpoint_data}
    for name in ("selection.json", "best.pt", "bc.pt", "bc_validation.json", "updates.jsonl"):
        if (run / name).is_file():
            snapshots[name] = (run / name).read_bytes()
    bc_validation = None
    if "bc.pt" in snapshots:
        bc = torch.load(io.BytesIO(snapshots["bc.pt"]), map_location="cpu", weights_only=True)
        if (bc.get("steps") != 0 or bc.get("ppo_optimizer_steps", 0) != 0
                or bc.get("ppo_actor_optimizer_steps", 0) != 0
                or bc.get("hierarchical_training", {}).get("source_sha256") != sources
                or bc.get("env_config") != expected_scene
                or bc.get("warmstart_metrics", {}).get("optimizer_steps", 0) <= 0):
            raise ValueError("Saved bc.pt is not a trained pre-PPO checkpoint for this scene/source")
    if "bc_validation.json" in snapshots:
        bc_validation = json.loads(snapshots["bc_validation.json"])
        if ("bc.pt" not in snapshots
                or bc_validation.get("source_sha256") != sources
                or bc_validation.get("checkpoint_sha256") != sha256(snapshots["bc.pt"])
                or [episode.get("seed") for episode in bc_validation.get("evaluation", {}).get("episodes", [])]
                != validation_seeds):
            raise ValueError("BC validation does not match the saved BC checkpoint/source")
    bc_validation_status = (
        "recorded" if bc_validation is not None else
        "pending; the trainer resume branch does not replay BC validation" if saved.get("validate_bc") else
        "not requested"
    )
    plan = {
        "run": str(run), "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_data), "training_metadata_sha256": sha256(metadata_data),
        "checkpoint_steps": current_steps, "target_steps": target,
        "physics_resume": "Starts a new episode; checkpoint does not serialize physical state.",
        "reset_optimizer": pre_ppo,
        "optimizer_reason": "pre-PPO checkpoint: begin fresh Adam moments" if pre_ppo else "PPO continuation: retain saved Adam state",
        "source_sha256": sources, "argv": argv,
        "checkpoint_counters": {key: checkpoint.get(key, 0) for key in (
            "steps", "ppo_decision_steps", "ppo_optimizer_steps", "ppo_actor_optimizer_steps",
            "expert_label_steps", "expert_optimizer_steps",
        )},
        "warmstart_counts": {key: warmstart.get(key, 0) for key in (
            "physical_steps", "decision_examples", "optimizer_steps",
        )},
        "bc_validation_status": bc_validation_status,
        "bc_validation_summary": (bc_validation or {}).get("evaluation", {}).get("summary"),
        "command_preview": shlex.join([sys.executable, "-m", "dual_arm_conveyor.hierarchical_train", *argv]),
    }
    return plan, snapshots


def _option(argv, name):
    for index, token in enumerate(argv):
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
        if token == name and index + 1 < len(argv):
            return argv[index + 1]
    return None


def running_training_pids(run, proc_root=Path("/proc")):
    """Detect this project's normal training and resume commands on Linux."""
    run, matches = Path(run).resolve(), []
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = [part.decode() for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
            command_files = {Path(token).name for token in argv[:3]}
            regular = ("dual_arm_conveyor.hierarchical_train" in argv
                       or ("dual_arm_conveyor" in argv and "train-scheduler" in argv)
                       or "train_scheduler.sh" in command_files)
            helper = bool(command_files & {"resume_scheduler.py", "resume_scheduler.sh"}) and "--execute" in argv
            output = _option(argv, "--output") if regular else _option(argv, "--run") if helper else None
            if output is None:
                continue
            target = Path(output)
            if not target.is_absolute():
                target = (entry / "cwd").resolve(strict=True) / target
            if target.resolve() == run:
                matches.append(int(entry.name))
        except (OSError, UnicodeError):
            continue  # Process exited or belongs to an unreadable session.
    return matches


def backup_metadata(plan, snapshots):
    run = Path(plan["run"])
    history = run / "resume_history"
    history.mkdir(exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ_")
    destination = Path(tempfile.mkdtemp(prefix=prefix, dir=history))
    for name, data in snapshots.items():
        (destination / name).write_bytes(data)
    (destination / "resume_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return destination


@contextmanager
def resume_lock(run):
    """Serialize helper executions; ordinary trainers are checked via /proc too."""
    path = Path(run).resolve() / ".resume.lock"
    with path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another resume helper holds this run's lock") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def retain_prior_evidence(plan, snapshots, backup):
    """Restore verified history omitted by the trainer's new segment metadata.

    Counters and wall_seconds remain the trainer's own values. This merges only
    pre-existing evidence, and never invents a BC validation or PPO update.
    """
    path = Path(plan["run"]) / "training.json"
    current_data = path.read_bytes()
    if current_data == snapshots["training.json"]:
        return  # The trainer did not write a new segment (e.g. preflight failed).
    current = json.loads(current_data)
    if (current.get("resume_checkpoint_sha256") != plan["checkpoint_sha256"]
            or current.get("source_sha256") != plan["source_sha256"]):
        raise RuntimeError("Resumed metadata does not match the backed-up checkpoint/source")
    previous = json.loads(snapshots["training.json"])
    checkpoint = torch.load(io.BytesIO(snapshots["resume_checkpoint.pt"]), map_location="cpu", weights_only=True)
    if checkpoint.get("warmstart_metrics"):
        current.setdefault("warmstart", checkpoint["warmstart_metrics"])
    if "bc.pt" in snapshots:
        if (Path(plan["run"]) / "bc.pt").read_bytes() != snapshots["bc.pt"]:
            raise RuntimeError("BC checkpoint changed during resumed training")
        current.setdefault("bc_checkpoint", "bc.pt")
    if "bc_validation.json" in snapshots:
        if (Path(plan["run"]) / "bc_validation.json").read_bytes() != snapshots["bc_validation.json"]:
            raise RuntimeError("BC validation changed during resumed training")
        if plan["bc_validation_summary"] is not None:
            current.setdefault("bc_validation_summary", plan["bc_validation_summary"])
    for key in ("actual_training_episode_seeds", "actual_training_catalogue_visits"):
        history = list(previous.get(key, []))
        for item in current.get(key, []):
            if item not in history:
                history.append(item)
        if history:
            current[key] = history
    current["resume_evidence"] = {
        "backup": str(backup), "prior_metadata_sha256": plan["training_metadata_sha256"],
        "prior_checkpoint_counters": plan["checkpoint_counters"],
        "bc_validation_status": plan["bc_validation_status"],
        "note": "BC and seed history restored from verified prior evidence; wall_seconds is this segment only.",
    }
    temporary = path.with_name("training.json.resume.tmp")
    temporary.write_text(json.dumps(current, indent=2) + "\n")
    os.replace(temporary, path)


def execute_resume(args):
    active = running_training_pids(args.run)
    if active:
        raise RuntimeError(f"Training for this run is already active (PIDs {active}); refusing concurrent resume")
    plan, snapshots = prepare_resume(args.run, args.steps, args.device)
    print(json.dumps({"event": "resume_plan", "execute": True, **plan}, indent=2), flush=True)
    if (train.source_hashes() != plan["source_sha256"]
            or Path(plan["checkpoint"]).read_bytes() != snapshots["resume_checkpoint.pt"]
            or any((Path(plan["run"]) / name).read_bytes() != data
                   for name, data in snapshots.items() if name != "resume_checkpoint.pt")):
        raise RuntimeError("Run evidence changed while preparing resume; inspect the active run before retrying")
    backup = backup_metadata(plan, snapshots)
    print(json.dumps({"event": "resume_backup", "directory": str(backup)}), flush=True)
    try:
        train.main(plan["argv"])
    finally:
        retain_prior_evidence(plan, snapshots, backup)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if args.execute:
        with resume_lock(args.run):
            return execute_resume(args)
    plan, snapshots = prepare_resume(args.run, args.steps, args.device)
    print(json.dumps({"event": "resume_plan", "execute": args.execute, **plan}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
