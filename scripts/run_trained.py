#!/usr/bin/env python3
"""Preview or run a user's trained scheduler with its complete saved scene.

Preview uses only the standard library. --execute enables CPU policy inference
and physics; it never trains. Evaluate runs exactly one episode per process.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dual_arm_conveyor import cli  # noqa: E402; standard-library imports only
from dual_arm_conveyor.config import Config  # noqa: E402


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative seed")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", nargs="?", default="print-config",
                        choices=("print-config", "play", "record", "evaluate"))
    result.add_argument("--run", type=Path, required=True,
                        help="Training directory containing training.json and checkpoints")
    result.add_argument("--checkpoint", choices=("best", "latest", "bc"), default="best")
    result.add_argument("--seed", type=nonnegative_int, default=2000000,
                        help="Arrival/pose seed; separate from the physical object catalogue")
    result.add_argument("--catalog-seed", type=nonnegative_int,
                        help="Optional new physical object catalogue; default preserves the saved catalogue")
    result.add_argument("--render-backend", choices=("auto", "egl", "glfw", "osmesa"), default="auto",
                        help="Rendering environment only; does not change the saved physical scene")
    result.add_argument("--output", type=Path, help="New .mp4 for record, or new .json for evaluate")
    result.add_argument("--fps", type=positive_int, default=30)
    result.add_argument("--width", type=positive_int, default=960,
                        help="Overview width; recording adds the existing status/camera panel")
    result.add_argument("--height", type=positive_int, default=720)
    result.add_argument("--execute", action="store_true", help="Actually play, record or evaluate; default is preview")
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def output_paths(options):
    if options.mode not in ("record", "evaluate"):
        require(options.output is None, "--output is only used by record/evaluate")
        return []
    require(options.output is not None, "record/evaluate require --output")
    output = options.output.resolve()
    extension = ".mp4" if options.mode == "record" else ".json"
    require(output.suffix.lower() == extension, f"{options.mode} requires an {extension} output")
    paths = [output, output.with_name(output.name + ".launch.json")]
    if options.mode == "record":
        paths.append(output.with_suffix(".json"))
    require(all(not path.exists() for path in paths), "Output or its report already exists; choose a new filename")
    return paths


def cli_arguments(options, scene, checkpoint):
    """Map every saved field, then fail closed if any field was lost by the CLI."""
    command = "evaluate" if options.mode == "evaluate" else "play"
    args = cli.parser().parse_args([command])
    aliases = {"total_objects": "objects", "max_time": "duration", "seed": "catalog_seed",
               "use_dispatch_preferences": "dispatch_preferences"}
    for key, value in scene.items():
        destination = aliases.get(key, key)
        if hasattr(args, destination):
            setattr(args, destination, value)
    args.no_auto_unload = not scene["auto_unload"]
    args.continuous = scene["total_objects"] is None
    args.policy, args.seed = str(checkpoint), options.seed
    args.render_backend = options.render_backend
    if command == "evaluate":
        args.episodes, args.output = 1, options.output.resolve()
    else:
        args.fps, args.width, args.height = options.fps, options.width, options.height
        args.speed = 1.0
        args.camera_view = "rgbd" if scene["sensor_mode"] == "vision" else "off"
        args.record = options.output.resolve() if options.mode == "record" else None
    resolved = asdict(cli.config_from_args(args))
    differences = {key: {"saved": scene.get(key), "resolved": resolved.get(key)}
                   for key in scene.keys() | resolved.keys() if scene.get(key) != resolved.get(key)}
    require(not differences, "CLI cannot restore the complete saved scene: " + json.dumps(differences))
    return args


def prepare(options):
    require(not (options.mode == "print-config" and options.execute), "print-config is always read-only; select play, record or evaluate")
    run = options.run.resolve()
    metadata_file = run / "training.json"
    metadata = json.loads(metadata_file.read_text())
    original = metadata.get("env_config")
    names = {field.name for field in fields(Config)}
    require(isinstance(original, dict) and set(original) == names,
            "training.json must contain the complete current Config schema; no missing fields or unknown fields")
    Config(**original)
    require(original["num_arms"] == 4, "This launcher requires a four-arm hierarchical scheduler run")
    for key in ("scheduler_action_metadata", "scheduler_observation_metadata"):
        require(isinstance(metadata.get(key), dict) and bool(metadata[key]), f"training.json is missing {key}")
    checkpoint = run / (options.checkpoint + ".pt")
    require(checkpoint.is_file(), f"Checkpoint missing: {checkpoint}; no other checkpoint is substituted")
    paths = output_paths(options)
    scene = dict(original)
    if options.catalog_seed is not None:
        scene["seed"] = options.catalog_seed
    if options.mode in ("evaluate", "record"):
        require(scene["total_objects"] is not None or scene["max_time"] is not None,
                "Recording/evaluation requires a bounded saved scene")
    require(options.fps * scene["physics_dt"] <= 1 + 1e-9,
            "--fps must not exceed the saved physics update frequency")
    cli_arguments(options, scene, checkpoint)
    plan = dict(event="trained_run_preview", mode=options.mode, run=str(run),
        training_status=metadata.get("status"), checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint), training_metadata_sha256=digest(metadata_file),
        config=scene, scene_changes={key: {"training": original[key], "selected": value}
            for key, value in scene.items() if original[key] != value},
        arrival_seed=options.seed, render_backend=options.render_backend,
        rendering_note="Runtime rendering choice only; physical scene and policy input mode are restored from training.json.",
        inference_device="cpu", episodes=1, fps=options.fps, overview_size=[options.width, options.height],
        output=None if not paths else str(paths[0]),
        output_files=[str(path) for path in paths], execute=options.execute,
        source_sha256=cli.source_hashes(), training_source_sha256=metadata.get("source_sha256"),
        checkpoint_interface_verified=False, physics_created=False, training_started=False,
        cohort_note="Arrival/catalogue seeds are explicit; this launcher does not certify that they are held out of training or model selection.")
    return metadata, plan


def validate_policy(policy, metadata, scene):
    """Validate the actual loaded weights before constructing a physical scene."""
    from dual_arm_conveyor.scheduler import action_metadata, observation_metadata

    require(normalized(policy.metadata.get("env_config")) == normalized(metadata["env_config"]),
            "Checkpoint scene differs from training.json")
    probe = SimpleNamespace(config=Config(**scene))
    expected = {"scheduler_action_metadata": action_metadata(probe),
                "scheduler_observation_metadata": observation_metadata(probe)}
    for key, value in expected.items():
        require(normalized(policy.metadata.get(key)) == normalized(metadata[key]),
                f"Checkpoint {key} differs from training.json")
        require(normalized(policy.metadata[key]) == normalized(value),
                f"Checkpoint {key} is incompatible with the current scheduler")


def execute(options, metadata, plan):
    output_paths(options)  # Check again immediately before execution.
    require(digest(plan["checkpoint"]) == plan["checkpoint_sha256"], "Checkpoint changed after preview")
    require(digest(Path(plan["run"]) / "training.json") == plan["training_metadata_sha256"],
            "Training metadata changed after preview")
    import torch
    from dual_arm_conveyor.learning import load_policy

    torch.set_num_threads(1)
    policy = load_policy(plan["checkpoint"], device="cpu")
    validate_policy(policy, metadata, plan["config"])
    del policy  # The CLI loads its own checked CPU policy; retain only one instance.
    args = cli_arguments(options, plan["config"], plan["checkpoint"])
    launch = dict(plan, event="trained_run_execution", checkpoint_interface_verified=True,
                  status="running", physics_created=False)
    manifest = None
    if plan["output"]:
        output = Path(plan["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = output.with_name(output.name + ".launch.json")
        with manifest.open("x") as stream:
            json.dump(launch, stream, indent=2, ensure_ascii=False)
    try:
        launch["physics_created"] = True
        result = args.func(args)
        interrupted = options.mode in ("record", "play") and result.get("unfinished", 0) > 0
        launch["status"] = "interrupted_or_unfinished" if interrupted else "complete"
        return 130 if interrupted else 0
    except BaseException as error:
        launch.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error))
        raise
    finally:
        if manifest:
            manifest.write_text(json.dumps(launch, indent=2, ensure_ascii=False) + "\n")


def main(argv=None):
    options = parser().parse_args(argv)
    metadata, plan = prepare(options)
    print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if not options.execute:
        return 0
    return execute(options, metadata, plan)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, KeyError, OSError, ImportError) as error:
        raise SystemExit(str(error))
