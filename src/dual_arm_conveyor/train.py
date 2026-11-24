"""Train and validate a masked PPO scheduler in the physical MuJoCo env."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .learning import (
    PPOConfig,
    PPOTrainer,
    Policy,
    evaluate_policy,
    load_policy,
    write_json,
)
from .cli import add_sensor_args, sensor_config_kwargs, positive_float, positive_int


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=Path("results/training"))
    result.add_argument(
        "--steps",
        type=int,
        default=4096,
        help="Total environment decision steps, including forced actions and resumed steps; actor updates are logged separately",
    )
    result.add_argument(
        "--warmstart-steps",
        type=int,
        default=4096,
        help="Physical heuristic decisions for supervised initialization",
    )
    result.add_argument("--warmstart-epochs", type=int, default=100)
    result.add_argument("--rollout-steps", type=int, default=256)
    result.add_argument("--epochs", type=int, default=4)
    result.add_argument("--minibatch-size", type=int, default=64)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--entropy-coef", type=float, default=0.005)
    result.add_argument("--hidden-dim", type=int, default=64)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cpu")
    result.add_argument("--resume", type=Path)
    result.add_argument("--eval-episodes", type=int, default=3)
    result.add_argument(
        "--eval-seed",
        type=int,
        default=20000,
        help="Held-out final test seed range; never used for selection",
    )
    result.add_argument("--validation-episodes", type=int, default=1)
    result.add_argument(
        "--validation-seed",
        type=int,
        default=10000,
        help="Fixed checkpoint-selection seed range",
    )
    result.add_argument(
        "--validation-every",
        type=int,
        default=2048,
        help="PPO steps between checkpoint comparisons; 0 means final only",
    )
    result.add_argument(
        "--total-objects",
        type=positive_int,
        default=30,
        help="finite arrivals per training/validation episode (continuous play is separate)",
    )
    result.add_argument("--pool-size", type=positive_int, default=64)
    result.add_argument("--bin-capacity", type=positive_int, default=12)
    result.add_argument("--no-auto-unload", action="store_true")
    result.add_argument(
        "--motion-mode", choices=("optimized", "legacy"), default="optimized"
    )
    result.add_argument("--belt-speed", type=float, default=0.1)
    result.add_argument("--spacing", type=float, default=0.45)
    result.add_argument(
        "--sensor-mode",
        choices=("oracle", "vision"),
        default="oracle",
        help="training and checkpoint-selection observations; final deployment test always uses vision",
    )
    result.add_argument(
        "--arrival-jitter",
        type=float,
        default=1.2,
        help="uniform interval jitter +/- seconds; minimum safe gap still applies",
    )
    result.add_argument(
        "--yaw-range",
        type=float,
        default=math.pi,
        help="initial object yaw range +/- radians",
    )
    result.add_argument("--decision-dt", type=float, default=0.1)
    result.add_argument("--physics-dt", type=positive_float, default=0.005)
    result.add_argument("--max-time", type=positive_float, default=None)
    add_sensor_args(result)
    return result


def training_seed_range(seed: int, reset_count: int, held_out_seeds: list[int]) -> dict:
    """The trainer increments the episode seed at each reset, including BC."""
    last_seed = seed + reset_count
    overlap = [value for value in held_out_seeds if seed <= value <= last_seed]
    if overlap:
        raise ValueError(
            f"Actual training seeds overlap held-out seeds {overlap}; choose a separate validation/test range"
        )
    return {"first": seed, "last": last_seed, "reset_count": reset_count}


def main(argv=None):
    args = parser().parse_args(argv)
    if (
        args.eval_episodes < 0
        or args.validation_episodes < 0
        or args.validation_every < 0
    ):
        raise ValueError("Evaluation counts and interval must be nonnegative")
    validation_seeds = list(
        range(args.validation_seed, args.validation_seed + args.validation_episodes)
    )
    test_seeds = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
    if set(validation_seeds) & set(test_seeds):
        raise ValueError("Validation and final test seed ranges must not overlap")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    # Final deployment testing renders camera images even when training uses
    # oracle state. Select the headless backend before importing MuJoCo.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from .config import Config
    from .env import ConveyorEnv
    import mujoco

    env_config = Config(
        total_objects=args.total_objects,
        pool_size=args.pool_size,
        bin_capacity=args.bin_capacity,
        auto_unload=not args.no_auto_unload,
        motion_mode=args.motion_mode,
        belt_speed=args.belt_speed,
        spacing=args.spacing,
        decision_dt=args.decision_dt,
        physics_dt=args.physics_dt,
        max_time=args.max_time,
        seed=args.seed,
        sensor_mode=args.sensor_mode,
        arrival_jitter=args.arrival_jitter,
        yaw_range=args.yaw_range,
        **sensor_config_kwargs(args),
    )
    deployment_config = replace(env_config, sensor_mode="vision")
    train_config = PPOConfig(
        steps=args.steps,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=args.device,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    env = ConveyorEnv(env_config)
    validation_env = ConveyorEnv(env_config) if validation_seeds else None
    deployment_env = None
    trainer = PPOTrainer(env, train_config, asdict(env_config))
    started = time.monotonic()
    metadata = {
        "schema_version": 3,
        "environment": "physical MuJoCo ConveyorEnv",
        "env_config": asdict(env_config),
        "train_config": asdict(train_config),
        "deployment_config": asdict(deployment_config),
        "training_sensor_mode": args.sensor_mode,
        "deployment_sensor_mode": "vision",
        "deployment_evaluation_completed": False,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "mujoco_version": mujoco.__version__,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "arguments": vars(args),
        "device": args.device,
        "resume": str(args.resume) if args.resume else None,
        "resume_note": "A resumed run starts a new physics episode; optimizer and policy states are restored.",
        "validation_seeds": validation_seeds,
        "test_seeds": test_seeds,
    }
    selection = {
        "seeds": validation_seeds,
        "sensor_mode": args.sensor_mode,
        "purpose": "checkpoint selection, separate from final vision deployment evaluation",
        "candidates": [],
        "selected": None,
    }
    best_reward = -float("inf")
    last_validation = -1

    def progress(record):
        print(
            f"phase={record['phase']} steps={record['steps']} episodes={record['episodes']} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    def select_checkpoint(label):
        nonlocal best_reward, last_validation
        metadata["actual_training_seed_range"] = training_seed_range(
            args.seed, trainer.reset_count, validation_seeds + test_seeds
        )
        if validation_env is None:
            return
        print(
            f"Checkpoint validation ({label}), seeds={validation_seeds}...", flush=True
        )
        result = evaluate_policy(
            validation_env,
            Policy(trainer.model, args.device),
            validation_seeds,
            progress,
        )
        candidate = {"label": label, "steps": trainer.steps, **result}
        selection["candidates"].append(candidate)
        if result["mean_reward"] > best_reward:
            best_reward = result["mean_reward"]
            trainer.save(args.output / "best.pt")
            selection["selected"] = {
                "label": label,
                "steps": trainer.steps,
                "mean_reward": best_reward,
            }
        last_validation = trainer.steps
        write_json(args.output / "selection.json", selection)
        trainer.model.train()

    try:
        if args.resume:
            trainer.resume(args.resume)
        elif args.warmstart_steps:
            print(
                f"Collecting {args.warmstart_steps} physical expert steps...",
                flush=True,
            )
            metadata["warmstart"] = trainer.warmstart(
                args.warmstart_steps, args.warmstart_epochs, progress
            )
            trainer.save(args.output / "warmstart.pt")
            print(
                f"Warmstart: {metadata['warmstart']['decision_examples']} decision examples",
                flush=True,
            )
        if args.resume or args.warmstart_steps:
            select_checkpoint("resumed" if args.resume else "warmstart")

        def on_update(record):
            metadata["actual_training_seed_range"] = training_seed_range(
                args.seed, trainer.reset_count, validation_seeds + test_seeds
            )
            metadata["elapsed_seconds"] = time.monotonic() - started
            metadata["history"] = trainer.history
            trainer.save(args.output / "policy.pt")
            write_json(args.output / "training.json", metadata)
            print(
                f"steps={record['steps']} reward={record['rollout_reward']:.3f} loss={record['loss']:.4f} "
                f"episodes={record['episodes_total']} elapsed={metadata['elapsed_seconds']:.1f}s",
                flush=True,
            )
            if (
                args.validation_every
                and trainer.steps - last_validation >= args.validation_every
            ):
                select_checkpoint("ppo")

        trainer.train(on_update, progress)
        trainer.save(args.output / "policy.pt")
        if trainer.steps != last_validation:
            select_checkpoint("ppo_final")
        metadata["history"] = trainer.history
        metadata["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "training.json", metadata)
        if args.eval_episodes:
            selected_path = args.output / (
                "best.pt" if validation_seeds else "policy.pt"
            )
            deployment_env = ConveyorEnv(deployment_config)
            print(
                f"Final VISION testing of selected policy and heuristic on seeds {test_seeds}...",
                flush=True,
            )
            validation = {
                "purpose": "held-out camera deployment test, never used for checkpoint selection",
                "sensor_mode": "vision",
                "checkpoint": str(selected_path),
                "seeds": test_seeds,
                "env_config": asdict(deployment_config),
                "training_env_config": asdict(env_config),
                "learned": evaluate_policy(
                    deployment_env,
                    load_policy(selected_path, args.device),
                    test_seeds,
                    progress,
                ),
                "heuristic": evaluate_policy(
                    deployment_env, None, test_seeds, progress
                ),
            }
            write_json(args.output / "validation.json", validation)
            metadata["deployment_evaluation_completed"] = True
            print(
                f"Validation mean reward: learned={validation['learned']['mean_reward']:.3f}, "
                f"heuristic={validation['heuristic']['mean_reward']:.3f}",
                flush=True,
            )
        else:
            print(
                "Deployment vision evaluation was skipped (--eval-episodes 0); this run is not a deployment validation.",
                flush=True,
            )
        metadata["total_elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "training.json", metadata)
    finally:
        close = getattr(env, "close", None)
        if close:
            close()
        if validation_env is not None and getattr(validation_env, "close", None):
            validation_env.close()
        if deployment_env is not None and getattr(deployment_env, "close", None):
            deployment_env.close()
    print(f"Saved checkpoint to {args.output / 'policy.pt'}", flush=True)


if __name__ == "__main__":
    main()
