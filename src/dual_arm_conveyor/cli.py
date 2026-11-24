"""Command line entry points; simulation imports stay lazy for headless rendering."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def dropout_probability(value: str) -> float:
    result = nonnegative_float(value)
    if result >= 1:
        raise argparse.ArgumentTypeError("must be less than 1")
    return result


def camera_dimension(value: str) -> int:
    result = positive_int(value)
    if result < 64:
        raise argparse.ArgumentTypeError("must be at least 64 pixels")
    return result


def add_sensor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wrist-camera-mode",
        choices=("off", "observe", "refine"),
        default="off",
        help="overhead plus two task-triggered wrist cameras: observe or bounded XY refinement",
    )
    parser.add_argument("--wrist-camera-hz", type=positive_float, default=6.0)
    parser.add_argument("--wrist-camera-width", type=camera_dimension, default=320)
    parser.add_argument("--wrist-camera-height", type=camera_dimension, default=240)
    parser.add_argument(
        "--camera-rig",
        choices=("overhead", "dual", "triple"),
        default="overhead",
        help="overhead: one top camera; dual: opposite oblique cameras; triple: all three",
    )
    parser.add_argument(
        "--camera-hz",
        type=positive_float,
        default=10.0,
        help="RGB-D capture rate in Hz",
    )
    parser.add_argument(
        "--camera-width",
        type=camera_dimension,
        default=480,
        help="sensor image width in pixels",
    )
    parser.add_argument(
        "--camera-height",
        type=camera_dimension,
        default=360,
        help="sensor image height in pixels",
    )
    parser.add_argument(
        "--camera-latency",
        type=nonnegative_float,
        default=0.08,
        help="image delivery delay in seconds",
    )
    parser.add_argument(
        "--depth-noise",
        type=nonnegative_float,
        default=0.001,
        help="depth noise standard deviation in metres",
    )
    parser.add_argument(
        "--camera-dropout",
        type=dropout_probability,
        default=0.02,
        help="probability of dropping each captured frame",
    )


def sensor_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: getattr(args, name)
        for name in (
            "camera_rig",
            "camera_hz",
            "camera_width",
            "camera_height",
            "camera_latency",
            "depth_noise",
            "camera_dropout",
            "wrist_camera_mode",
            "wrist_camera_hz",
            "wrist_camera_width",
            "wrist_camera_height",
        )
    }


def _add_scene_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--render-backend",
        choices=("auto", "egl", "glfw", "osmesa"),
        default="auto",
        help="probe OpenGL contexts and prefer hardware; MUJOCO_GL overrides auto",
    )
    parser.add_argument(
        "--visual-quality",
        choices=("balanced", "high"),
        default="balanced",
        help="balanced disables costly offscreen anti-aliasing and overview reflections/shadows",
    )
    parser.add_argument(
        "--seed", type=int, default=100, help="first evaluation / playback seed"
    )
    parser.add_argument(
        "--catalog-seed",
        type=int,
        default=0,
        help="physical shape, mass and material catalogue seed, independent of episode seed",
    )
    arrivals = parser.add_mutually_exclusive_group()
    arrivals.add_argument(
        "--objects",
        type=positive_int,
        default=None,
        help="finite arrival count; play otherwise feeds continuously",
    )
    arrivals.add_argument(
        "--continuous",
        action="store_true",
        help="continuous arrivals; evaluation defaults to a 300 s window",
    )
    parser.add_argument(
        "--duration",
        type=positive_float,
        default=None,
        help="stop after this many simulated seconds",
    )
    parser.add_argument(
        "--pool-size",
        type=positive_int,
        default=64,
        help="reusable physical-object pool; independent of the ten action slots",
    )
    parser.add_argument(
        "--bin-capacity",
        type=positive_int,
        default=12,
        help="reserved/completed objects per bin before unloading; four arms allow even capacities up to 24, two arms up to 12",
    )
    parser.add_argument(
        "--no-auto-unload",
        action="store_true",
        help="stop accepting objects into a full bin instead of unloading it",
    )
    parser.add_argument(
        "--motion-mode",
        choices=("optimized", "legacy"),
        default="optimized",
        help="trajectory implementation for controlled comparison",
    )
    parser.add_argument("--direct-retask", action="store_true", help="Use a verified nearby ready pose after placing, instead of always returning to fixed home")
    parser.add_argument("--nearest-bin-slot", action="store_true", help="Choose a nearer free deposit site within the current balanced bin layer")
    parser.add_argument("--responsive-release", action="store_true", help="Confirm release from measured opening clearance and sustained loss of finger contact")
    parser.add_argument("--direct-bin-release", action="store_true", help="Release into a deep bin at the verified transport height, avoiding a 20 mm down/up detour")
    parser.add_argument("--extended-pick-deadline", action="store_true", help="Use the calibrated upstream near-lane completion boundary while retaining the original dispatch-entry window")
    parser.add_argument("--active-recovery", action="store_true", help="Use verified active recovery after a release-confirmation stall")
    parser.add_argument("--future-planning", action="store_true", help="Enable reversible next-job scheduling with a separate policy action schema")
    parser.add_argument("--safe-preposition", action="store_true", help="Prepare for a planned next job outside the shared belt space; requires future planning")
    parser.add_argument("--planned-return", action="store_true", help="Use a checked outside-belt return goal informed by the next plan; requires future planning and direct retask")
    parser.add_argument("--guarded-descent", action="store_true", help="Bound descent tracking lead by measured jaw clearance and recover from premature contact; requires adaptive finger control")
    parser.add_argument("--stable-completion-reward", action="store_true", help="Reward each object's first physical completion once, retaining final physical outcome checks")
    parser.add_argument("--cooperative-motion", action="store_true", help="Return the released empty hand outside the shared belt strip; keep the shared-zone lock and physical collision checks")
    parser.add_argument("--cooperative-drop-cost", action="store_true", help="Experimental independent ablation: include shared-strip clearance time in deposit-slot cost")
    parser.add_argument("--downstream-completion", action="store_true", help="Experimental checked near-side completion extension at the downstream station; retain dispatch window, hard stop and path checks")
    parser.add_argument(
        "--strategy",
        choices=("baseline", "adaptive"),
        default="baseline",
        help="baseline controller or speed-adaptive perception and moving-target control",
    )
    parser.add_argument(
        "--control-speed",
        type=positive_float,
        default=0.45,
        help="Cartesian command speed limit in m/s; joint velocity/acceleration limits still apply",
    )
    parser.add_argument(
        "--num-arms",
        type=int,
        choices=(2, 4),
        default=2,
        help="four uses two sequential picking stations and the same two bins; requires adaptive strategy",
    )
    parser.add_argument(
        "--gripper-mode",
        choices=("fingers", "vacuum"),
        default="fingers",
        help="physical finger grasp or contact-triggered vacuum tool; vacuum requires adaptive strategy and wrist cameras off",
    )
    parser.add_argument(
        "--deep-bins", action="store_true",
        help="fixed deep collection bins; requires --no-auto-unload",
    )
    parser.add_argument(
        "--spacing",
        type=positive_float,
        default=0.45,
        help="mean longitudinal spacing per lane in metres",
    )
    parser.add_argument(
        "--supply-lanes", type=int, choices=(1, 2), default=1,
        help="one scattered stream or two independently staggered streams",
    )
    parser.add_argument(
        "--lane-offset", type=nonnegative_float, default=0.065,
        help="two-lane centres are +/- this distance from the belt centre (m)",
    )
    parser.add_argument(
        "--lane-jitter", type=nonnegative_float, default=0.012,
        help="random lateral displacement about each two-lane centre (m)",
    )
    parser.add_argument(
        "--min-object-gap", type=nonnegative_float, default=0.0,
        help="minimum planar edge clearance at two-lane admission, in metres (0 disables)",
    )
    parser.add_argument(
        "--spacing-jitter",
        type=nonnegative_float,
        default=0.12,
        help="spatial arrival gap variation +/- metres for distance-based supply",
    )
    parser.add_argument(
        "--arrival-mode",
        choices=("auto", "time", "distance"),
        default="auto",
        help="auto uses distance gaps for adaptive strategy or variable speed, legacy time gaps otherwise",
    )
    parser.add_argument(
        "--belt-speed",
        type=positive_float,
        default=0.1,
        help="constant or mean belt speed in m/s",
    )
    parser.add_argument(
        "--belt-profile",
        choices=("constant", "sine", "random"),
        default="constant",
        help="physics speed profile; the controller does not receive its future schedule",
    )
    parser.add_argument(
        "--belt-speed-amplitude",
        type=nonnegative_float,
        default=0.1,
        help="variable-profile deviation from mean speed in m/s",
    )
    parser.add_argument(
        "--belt-speed-period",
        type=positive_float,
        default=12.0,
        help="sine period or twice random-ramp duration; lengthened to satisfy acceleration limit",
    )
    parser.add_argument(
        "--belt-max-acceleration",
        type=positive_float,
        default=0.08,
        help="maximum magnitude of belt acceleration in m/s^2",
    )
    parser.add_argument(
        "--sensor-mode",
        choices=("vision", "oracle"),
        default="vision",
        help="vision uses rendered images; oracle is the privileged-state baseline",
    )
    parser.add_argument(
        "--arrival-jitter",
        type=nonnegative_float,
        default=1.2,
        help="uniform interval jitter +/- this many seconds; safe minimum gap still applies",
    )
    parser.add_argument(
        "--yaw-range",
        type=nonnegative_float,
        default=math.pi,
        help="initial object yaw range +/- radians",
    )
    parser.add_argument(
        "--physics-dt",
        type=positive_float,
        default=0.005,
        help="physics step in seconds; must divide decision_dt",
    )
    parser.add_argument(
        "--dispatch-preferences",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the heuristic returning-partner priority; physical masks always remain active",
    )
    add_sensor_args(parser)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Dual Panda conveyor sorting with camera perception or oracle baseline"
    )
    commands = result.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser(
        "evaluate",
        help="evaluate finite batches or bounded continuous streams and write JSON",
    )
    _add_scene_args(evaluate)
    evaluate.add_argument("--episodes", type=positive_int, default=10)
    evaluate.add_argument(
        "--policy", default="heuristic", help="heuristic, predictive, sequence, or a trained checkpoint path"
    )
    evaluate.add_argument(
        "--output", type=Path, default=Path("results/evaluation.json")
    )
    evaluate.set_defaults(func=evaluate_command)
    play = commands.add_parser(
        "play", help="open interactive MuJoCo viewer, or record headless MP4"
    )
    _add_scene_args(play)
    play.add_argument("--policy", default="heuristic")
    play.add_argument(
        "--record",
        type=Path,
        metavar="FILE.mp4",
        help="record headless; defaults to 120 simulated seconds for continuous arrivals",
    )
    play.add_argument(
        "--fps",
        type=positive_int,
        default=30,
        help="overview updates per simulated second, independent of policy frequency",
    )
    play.add_argument(
        "--width",
        type=positive_int,
        default=960,
        help="overview width; recording adds a camera panel to the right",
    )
    play.add_argument("--height", type=positive_int, default=720)
    play.add_argument(
        "--camera-view",
        choices=("rgbd", "rgb", "off"),
        default="rgbd",
        help="live viewer inset with delivered RGB/tracks and depth; recording keeps its existing panel",
    )
    play.add_argument(
        "--speed", type=positive_float, default=1.0, help="interactive playback speed"
    )
    play.set_defaults(func=play_command)
    check = commands.add_parser(
        "check", help="inspect installed physics, training and rendering dependencies"
    )
    check.add_argument(
        "--probe-renderer",
        action="store_true",
        help="test OpenGL backends in subprocesses and identify hardware/software rendering",
    )
    check.set_defaults(func=check_command)
    commands.add_parser(
        "train", help="train PPO scheduler (use train --help for options)"
    )
    commands.add_parser(
        "train-scheduler", help="train the four-arm hierarchical PPO scheduler"
    )
    return result


def run_limits(args: argparse.Namespace) -> tuple[int | None, float | None]:
    """Resolve finite task/time limits without constructing physics or rendering."""
    objects, duration = args.objects, args.duration
    if args.command == "evaluate":
        if args.continuous:
            duration = duration or 300.0
        elif objects is None:
            objects = 50
    elif (
        args.command == "play" and args.record and objects is None and duration is None
    ):
        duration = 120.0
    return objects, duration


def config_from_args(args: argparse.Namespace) -> Any:
    """Resolve the complete scenario without constructing physics or cameras."""
    from .config import Config

    objects, duration = run_limits(args)
    return Config(
        total_objects=objects,
        max_time=duration,
        pool_size=args.pool_size,
        bin_capacity=args.bin_capacity,
        auto_unload=not args.no_auto_unload,
        motion_mode=args.motion_mode,
        direct_retask=args.direct_retask,
        nearest_bin_slot=args.nearest_bin_slot,
        responsive_release=args.responsive_release,
        direct_bin_release=args.direct_bin_release,
        extended_pick_deadline=args.extended_pick_deadline,
        active_recovery=args.active_recovery,
        future_planning=args.future_planning,
        safe_preposition=args.safe_preposition,
        planned_return=args.planned_return,
        guarded_descent=args.guarded_descent,
        stable_completion_reward=args.stable_completion_reward,
        cooperative_motion=args.cooperative_motion,
        cooperative_drop_cost=args.cooperative_drop_cost,
        downstream_completion=args.downstream_completion,
        strategy=args.strategy,
        use_dispatch_preferences=args.dispatch_preferences,
        num_arms=args.num_arms,
        gripper_mode=args.gripper_mode,
        deep_bins=args.deep_bins,
        seed=args.catalog_seed,
        control_speed=args.control_speed,
        spacing=args.spacing,
        spacing_jitter=args.spacing_jitter,
        supply_lanes=args.supply_lanes,
        lane_offset=args.lane_offset,
        lane_jitter=args.lane_jitter,
        min_object_gap=args.min_object_gap,
        arrival_mode=args.arrival_mode,
        belt_speed=args.belt_speed,
        belt_profile=args.belt_profile,
        belt_speed_amplitude=args.belt_speed_amplitude,
        belt_speed_period=args.belt_speed_period,
        belt_max_acceleration=args.belt_max_acceleration,
        sensor_mode=args.sensor_mode,
        arrival_jitter=args.arrival_jitter,
        yaw_range=args.yaw_range,
        physics_dt=args.physics_dt,
        visual_quality=args.visual_quality,
        **sensor_config_kwargs(args),
    )


def _make_env(args: argparse.Namespace) -> Any:
    from .env import ConveyorEnv

    return ConveyorEnv(config_from_args(args))


def _policy_action(policy: str, env: Any) -> Callable[[Any], int]:
    if policy == "heuristic":
        return lambda observation: int(env.heuristic_action())
    if policy in ("predictive", "sequence"):
        from .scheduler import SchedulerEnv

        scheduler = SchedulerEnv(env, teacher_mode=policy)
        return lambda observation: scheduler.action_to_native(scheduler.teacher_action())
    from .learning import load_policy

    trained = load_policy(Path(policy))
    training_scene = trained.metadata.get("env_config", {})
    if training_scene.get("num_arms", 2) != getattr(env.config, "num_arms", 2):
        raise ValueError("This checkpoint was trained for a different number of arms")
    if getattr(env.config, "num_arms", 2) == 4:
        from .scheduler import action_metadata, model_policy, observation_metadata

        if (
            trained.metadata.get("scheduler_action_metadata") != action_metadata(env)
            or trained.metadata.get("scheduler_observation_metadata")
            != observation_metadata(env)
            or training_scene.get("num_arms") != 4
        ):
            raise ValueError("This checkpoint is not a compatible four-arm scheduler")
        if (
            training_scene.get("use_dispatch_preferences")
            != env.config.use_dispatch_preferences
        ):
            raise ValueError(
                "Checkpoint dispatch preferences differ; "
                "use --no-dispatch-preferences for hierarchical PPO"
            )
        for option in ("direct_retask", "nearest_bin_slot", "responsive_release", "direct_bin_release", "extended_pick_deadline", "active_recovery", "future_planning", "safe_preposition", "planned_return", "guarded_descent", "stable_completion_reward"):
            if training_scene.get(option, False) != getattr(env.config, option, False):
                raise ValueError(f"Checkpoint {option} differs from the selected controller configuration")
        for option in ("cooperative_motion", "cooperative_drop_cost", "downstream_completion"):
            trained_cooperation = training_scene.get(option, False)
            selected_cooperation = getattr(env.config, option, False)
            if trained_cooperation != selected_cooperation:
                if not selected_cooperation:
                    raise ValueError(f"This checkpoint requires --{option.replace('_', '-')}")
                # Explicit opt-in changes execution, not weights or model schemas.
                # Keep the strict checks above and label each independent ablation.
                print(json.dumps({
                    "event": "controller_transfer", "option": option,
                    "training": trained_cooperation, "selected": selected_cooperation,
                    "weights_updated": False, "shared_zone_lock_retained": True,
                }), flush=True)
    if training_scene.get("gripper_mode", "fingers") != getattr(
        env.config, "gripper_mode", "fingers"
    ):
        raise ValueError("This checkpoint was trained for a different gripper mode")
    if training_scene.get(
        "schema_version"
    ) != env.config.schema_version or training_scene.get("sensor_mode") not in (
        "vision",
        "oracle",
    ):
        raise ValueError(
            "This checkpoint is not compatible with the current scene schema; train a policy for this scene"
        )
    if getattr(env.config, "num_arms", 2) == 4:
        return model_policy(env, trained)
    return lambda observation: int(trained(observation))


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only metrics present in all episodes; preserve units and counts."""
    if not episodes:
        raise ValueError("at least one episode is required")
    keys = set.intersection(*(set(episode) for episode in episodes))
    output = {}
    for key in sorted(keys - {"seed"}):
        values = [episode[key] for episode in episodes]
        if all(
            isinstance(value, (int, float, bool)) and math.isfinite(value)
            for value in values
        ):
            output[key] = {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "std": statistics.pstdev(values),
            }
    return output


def catalog_metadata(env: Any) -> dict[str, Any]:
    """Identify the fixed physical catalogue separately from arrival seeds."""
    specs = getattr(getattr(env, "ids", None), "object_specs", None)
    catalogue = None if specs is None else [spec.to_dict() for spec in specs]
    return {
        "catalog_seed": getattr(env.config, "seed", None),
        "catalog_sha256": (
            hashlib.sha256(
                json.dumps(catalogue, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            if catalogue is not None
            else None
        ),
        "catalog_objects": len(catalogue) if catalogue is not None else None,
        "catalog_note": (
            "Shape, size, mass and contact parameters are fixed per physical pool slot "
            "and reused after recycling. Episode seeds randomize arrivals, colors and "
            "orientations; they do not resample this catalogue."
        ),
    }


def source_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def runtime_metadata(started: float) -> dict[str, Any]:
    import mujoco
    from .rendering import metadata

    return {
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "wall_seconds": time.monotonic() - started,
        "rendering": metadata(),
        "glibc_tunables": os.environ.get("GLIBC_TUNABLES"),
        "libm_compat_requested": os.environ.get("CONVEYOR_LIBM_COMPAT") == "1",
    }


def controller_status(env: Any) -> dict[str, Any]:
    """Read-only operational diagnostics, never fed back into control."""
    result = {}
    for name, key in (
        ("target", "target_slots"),
        ("holding", "holding"),
        ("hold_requested", "hold_requested"),
        ("escape_routes", "escape_active"),
    ):
        values = getattr(env, name, None)
        if values is not None:
            result[key] = values.tolist() if hasattr(values, "tolist") else list(values)
    routes = getattr(env, "routes", None)
    if routes is not None:
        result["route_rejection_reasons"] = [
            getattr(plan, "reason", None)
            if plan is not None and not plan.verified
            else None
            for plan in routes
        ]
    now = getattr(getattr(env, "data", None), "time", None)
    retry = getattr(env, "route_retry_time", None)
    if now is not None and retry is not None:
        result["route_retry_in_seconds"] = [
            max(0.0, float(value - now)) for value in retry
        ]
    return result


def progress_counters(env: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    """Use environment-owned duration/counter semantics without new timeouts."""
    keys = (
        "missed",
        "dropped",
        "unfinished",
        "arm_assignments",
        "arm_sorted",
        "arm_assignments_since_last_sort",
        "max_arm_assignments_without_sort",
        "arm_seconds_since_last_sort",
        "retries",
        "robot_collision_steps",
        "inter_arm_collision_steps",
        "bin_unloads",
        "bin_service",
        "recycled_objects",
        "route_failures",
        "home_returns",
        "recovery_home_returns",
        "escape_attempts",
        "escape_successes",
        "release_timeouts",
        "recovery_waiting_for_release",
        "max_phase_duration_seconds",
        "current_phases",
        "current_phase_elapsed_seconds",
        "arm_utilization",
        "supply",
    )
    return {
        **{key: metrics[key] for key in keys if key in metrics},
        "phase": list(metrics.get("current_phases", getattr(env, "phase", []))),
        "controller_status": controller_status(env),
    }


def evaluate_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.sensor_mode == "vision":
        from .rendering import select_backend

        select_backend(args.render_backend)

    started = time.monotonic()
    source_sha256 = source_hashes()
    env = _make_env(args)
    episodes = []
    try:
        action = _policy_action(args.policy, env)
        checkpoint_sha256 = (
            hashlib.sha256(Path(args.policy).read_bytes()).hexdigest()
            if args.policy not in ("heuristic", "predictive", "sequence")
            else None
        )
        for episode_number in range(args.episodes):
            seed = args.seed + episode_number
            episode_started = time.monotonic()
            last_progress = episode_started
            observation = env.reset(seed=seed)
            episode_return, decisions = 0.0, 0
            while True:
                observation, reward, terminated, truncated, step_metrics = env.step(
                    action(observation)
                )
                episode_return += float(reward)
                decisions += 1
                now = time.monotonic()
                if now - last_progress >= 15.0:
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "episode": episode_number + 1,
                                "seed": seed,
                                "decisions": decisions,
                                "wall_seconds": now - episode_started,
                                "simulation_seconds": step_metrics.get("time"),
                                "spawned": step_metrics.get("total_objects"),
                                "sorted": step_metrics.get("sorted"),
                                **progress_counters(env, step_metrics),
                            },
                            default=_json_default,
                        ),
                        flush=True,
                    )
                    last_progress = now
                if terminated or truncated:
                    break
            metrics = dict(env.metrics())
            metrics["controller_status"] = controller_status(env)
            metrics.update(
                seed=seed,
                episode_return=episode_return,
                decisions=decisions,
                terminated=bool(terminated),
                truncated=bool(truncated),
                wall_seconds=time.monotonic() - episode_started,
            )
            episodes.append(metrics)
            print(
                json.dumps(
                    {"episode": episode_number + 1, **metrics}, default=_json_default
                ),
                flush=True,
            )
        report = {
            "schema_version": 3,
            "policy": args.policy,
            "sensor_mode": args.sensor_mode,
            "purpose": "camera deployment evaluation"
            if args.sensor_mode == "vision"
            else "privileged-state baseline evaluation",
            "checkpoint_sha256": checkpoint_sha256,
            **catalog_metadata(env),
            "source_sha256": source_sha256,
            "config": asdict(env.config)
            if is_dataclass(env.config)
            else vars(env.config),
            "runtime": runtime_metadata(started),
            "episodes": episodes,
            "aggregate": aggregate_episodes(episodes),
        }
        report["config_sha256"] = hashlib.sha256(
            json.dumps(report["config"], sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=_json_default)
            + "\n"
        )
        print(f"Saved evaluation: {args.output.resolve()}")
        return report
    finally:
        close = getattr(env, "close", None)
        if close:
            close()


def _camera(mujoco: Any, *, inset: bool = False, num_arms: int = 2) -> Any:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.6, 0.0, 0.25] if num_arms == 4 else [0.0, 0.0, 0.45]
    camera.distance = 4.4 if num_arms == 4 else 2.8
    camera.azimuth = 135
    camera.elevation = -42
    if inset:
        # Frame the workcell to the left of the live camera inset. Users retain
        # the viewer's normal orbit/pan/zoom controls.
        camera.distance *= 1.18
        azimuth = math.radians(camera.azimuth)
        camera.lookat[0] += 0.55 * math.sin(azimuth)
        camera.lookat[1] -= 0.55 * math.cos(azimuth)
    return camera


def _drive_status_lines(metrics):
    drive = metrics.get("belt_motion", {})
    actual, estimate = (
        drive.get("current_speed_m_s"),
        drive.get("control_speed_estimate"),
    )
    lines = []
    if actual is not None and estimate is not None:
        lines.append(f"Belt {actual:.3f} m/s  estimate {estimate:.3f} m/s")
    if "gripper_mode" in metrics:
        lines.append(f"{metrics.get('arm_count', 2)} arms  {metrics['gripper_mode']}")
    return lines


def _compose_camera_panel(overview: Any, env: Any, camera_panel=None) -> Any:
    """Label the overview and append only the camera image actually delivered.

    This function never captures a new perception frame or reads object state
    to draw detections. The detector owns the complete sensor overlay.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    height, width = overview.shape[:2]
    panel_width = max(320, width // 2)
    canvas = Image.new("RGB", (width + panel_width, height), (18, 24, 32))
    canvas.paste(Image.fromarray(overview), (0, 0))
    draw = ImageDraw.Draw(canvas)
    mode = env.config.sensor_mode
    draw.rectangle((0, 0, width, 29), fill=(18, 24, 32))
    draw.text(
        (12, 9),
        f"SIMULATION OVERVIEW | {mode.upper()} | t={env.data.time:.2f}s",
        fill=(240, 240, 240),
    )
    draw.text(
        (width + 12, 12), "DELIVERED CAMERA + DETECTOR TRACKS", fill=(240, 240, 240)
    )
    from .utilization_view import draw_utilization

    draw_utilization(draw, env, width, height)
    sensor = getattr(env, "camera_display", getattr(env, "sensor", None))
    overlay = getattr(sensor, "overlay", None) if sensor is not None else None
    camera_views = getattr(sensor, "camera_views", None)
    if callable(camera_views) and len(camera_views()) > 1:
        from .live_view import CameraPanel

        composer = camera_panel if camera_panel is not None else CameraPanel("rgbd")
        panel = composer.compose(
            sensor, float(env.data.time), panel_width, max(120, height - 130)
        )
        canvas.paste(Image.fromarray(panel), (width, 0))
        details_y = panel.shape[0] + 8
        metrics = env.metrics()
        unloads = metrics.get("bin_unloads", ["?", "?"])
        for line in (
            "RUN STATUS (EVALUATOR COUNTS)",
            f"Spawned {metrics.get('total_objects', '?')}  sorted {metrics.get('sorted', '?')}",
            f"Missed {metrics.get('missed', '?')}  dropped {metrics.get('dropped', '?')}",
            f"Unfinished {metrics.get('unfinished', '?')}  unloads {unloads[0]}/{unloads[1]}",
            "Arms: " + " / ".join(getattr(env, "phase", ["?", "?"])),
            *_drive_status_lines(metrics),
        ):
            draw.text((width + 12, details_y), line, fill=(230, 205, 155))
            details_y += 18
        return np.asarray(canvas)
    details_y = 72
    if overlay is not None:
        image = Image.fromarray(np.asarray(overlay, dtype=np.uint8))
        max_size = (panel_width - 24, max(32, height - 230))
        image.thumbnail(max_size, Image.Resampling.NEAREST)
        canvas.paste(image, (width + (panel_width - image.width) // 2, 48))
        details_y = 62 + image.height
    else:
        reason = (
            "Waiting for delayed camera frame"
            if mode == "vision"
            else "ORACLE baseline: no camera observations"
        )
        draw.text((width + 12, 52), reason, fill=(230, 190, 110))
    if sensor is not None:
        diagnostics = sensor.diagnostics()

        def display(key):
            value = diagnostics.get(key)
            return (
                "?"
                if value is None
                else (f"{value:.3f}" if isinstance(value, float) else str(value))
            )

        for line in (
            f"Frame: {display('delivered_frame_time')}  age: {display('frame_age_seconds')} s",
            f"Detections: {display('detections')}  tracks: {display('tracks')}",
        ):
            draw.text((width + 12, details_y), line, fill=(200, 210, 220))
            details_y += 18
    # Aggregate evaluator counters are display-only. They never add detections,
    # object identities, or state estimates to the camera/controller pipeline.
    metric_fn = getattr(env, "metrics", None)
    if callable(metric_fn):
        metrics = metric_fn()
        unloads = metrics.get("bin_unloads", ["?", "?"])
        status_lines = (
            "RUN STATUS (EVALUATOR COUNTS)",
            f"Spawned: {metrics.get('total_objects', '?')}  sorted: {metrics.get('sorted', '?')}",
            f"Missed: {metrics.get('missed', '?')}  dropped: {metrics.get('dropped', '?')}",
            f"Unfinished: {metrics.get('unfinished', '?')}  unloads: {unloads[0]}/{unloads[1]}",
            "Arms: " + " / ".join(getattr(env, "phase", ["?", "?"])),
            *_drive_status_lines(metrics),
        )
        for line in status_lines:
            draw.text((width + 12, details_y), line, fill=(230, 205, 155))
            details_y += 18
    draw.text(
        (width + 12, height - 20),
        "Camera labels do not use simulator object IDs.",
        fill=(170, 180, 190),
    )
    return np.asarray(canvas)


def _play_progress(env: Any, started: float, last_progress: float) -> float:
    now = time.monotonic()
    if now - last_progress < 15.0:
        return last_progress
    metrics = env.metrics()
    print(
        json.dumps(
            {
                "event": "progress",
                "command": "play",
                "wall_seconds": now - started,
                "simulation_seconds": metrics.get("time"),
                "spawned": metrics.get("total_objects"),
                "sorted": metrics.get("sorted"),
                **progress_counters(env, metrics),
            },
            default=_json_default,
        ),
        flush=True,
    )
    return now


class FrameClock:
    """Schedule distinct physics states without duplicating missed frames."""

    def __init__(self, fps: int, start_simulation: float = 0.0):
        self.fps = fps
        self.start_simulation = start_simulation
        self.next_index = 0
        self.frames = 0
        self.skipped_deadlines = 0
        self.render_seconds = 0.0
        self.first_render_seconds = None
        self.timestamps = deque(maxlen=512)
        self.first_timestamp = None
        self.last_timestamp = None
        self.minimum_interval = math.inf
        self.maximum_interval = 0.0

    def due(self, simulation_time: float) -> bool:
        deadline = self.start_simulation + self.next_index / self.fps
        if simulation_time + 1e-9 < deadline:
            return False
        index = math.floor((simulation_time - self.start_simulation) * self.fps + 1e-8)
        self.skipped_deadlines += max(0, index - self.next_index)
        self.next_index = index + 1
        return True

    def captured(self, simulation_time: float, render_seconds: float):
        if self.last_timestamp is not None:
            interval = simulation_time - self.last_timestamp
            if interval <= 0:
                raise ValueError(
                    "Each overview frame must capture a newer physics state"
                )
            self.minimum_interval = min(self.minimum_interval, interval)
            self.maximum_interval = max(self.maximum_interval, interval)
        else:
            self.first_timestamp = simulation_time
            self.first_render_seconds = render_seconds
        self.last_timestamp = simulation_time
        self.timestamps.append(float(simulation_time))
        self.frames += 1
        self.render_seconds += render_seconds

    def report(self, elapsed_wall: float, elapsed_simulation: float) -> dict:
        span = (self.last_timestamp or 0.0) - (self.first_timestamp or 0.0)
        return {
            "requested_fps": self.fps,
            "physics_states_rendered": self.frames,
            "duplicated_state_frames": 0,
            "skipped_simulation_frame_deadlines": self.skipped_deadlines,
            "captured_frames_per_simulation_second": (self.frames - 1) / span
            if span > 0
            else None,
            "frames_per_wall_second": self.frames / max(elapsed_wall, 1e-9),
            "real_time_factor": elapsed_simulation / max(elapsed_wall, 1e-9),
            "minimum_frame_simulation_interval": self.minimum_interval
            if self.frames > 1
            else None,
            "maximum_frame_simulation_interval": self.maximum_interval
            if self.frames > 1
            else None,
            "frame_render_and_output_seconds": self.render_seconds,
            "first_frame_render_and_output_seconds": self.first_render_seconds,
            "mean_subsequent_frame_render_and_output_seconds": (
                self.render_seconds - (self.first_render_seconds or 0)
            )
            / (self.frames - 1)
            if self.frames > 1
            else None,
            "frame_timestamps_last_512": list(self.timestamps),
            "note": "Overview frames capture new physics states. The camera panel still updates only when actual sensor frames arrive.",
        }


def _finish_play(
    args, env, started, interrupted=False, source_sha256=None, frame_clock=None
):
    metrics = dict(env.metrics())
    metrics["controller_status"] = controller_status(env)
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        runtime = runtime_metadata(started)
        report = {
            "schema_version": 3,
            "policy": args.policy,
            "sensor_mode": args.sensor_mode,
            "seed": args.seed,
            "config": asdict(env.config),
            **catalog_metadata(env),
            "source_sha256": source_hashes()
            if source_sha256 is None
            else source_sha256,
            "runtime": runtime,
            "metrics": metrics,
            "wall_seconds": runtime["wall_seconds"],
            "interrupted": interrupted,
            "camera_panel": "most recent delivered sensor overlay; absent in oracle mode",
        }
        if frame_clock is not None:
            report["playback"] = frame_clock.report(
                runtime["wall_seconds"], metrics["time"]
            )
            report["playback"]["encoded_duration_seconds"] = (
                frame_clock.frames / args.fps
            )
        args.record.with_suffix(".json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=_json_default)
            + "\n"
        )
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default))
    return metrics


def play_command(args: argparse.Namespace) -> dict[str, Any]:
    # This must happen before importing MuJoCo / OpenGL through the environment.
    if args.record or args.sensor_mode == "vision":
        from .rendering import select_backend

        select_backend(args.render_backend)
    import mujoco

    env = _make_env(args)
    started = time.monotonic()
    source_sha256 = source_hashes()
    last_progress = started
    frame_clock = None
    try:
        if args.fps * env.config.physics_dt > 1 + 1e-9:
            raise ValueError("--fps must not exceed the physics update frequency")
        action = _policy_action(args.policy, env)
        observation = env.reset(seed=args.seed)
        frame_clock = FrameClock(args.fps, float(env.data.time))
        if args.record:
            import imageio.v2 as imageio
            from .live_view import CameraPanel

            recording_panel = CameraPanel("rgbd")
            args.record.parent.mkdir(parents=True, exist_ok=True)
            env.model.vis.global_.offwidth = max(
                args.width, env.model.vis.global_.offwidth
            )
            env.model.vis.global_.offheight = max(
                args.height, env.model.vis.global_.offheight
            )
            camera = _camera(mujoco, num_arms=env.num_arms)
            with mujoco.Renderer(
                env.model, height=args.height, width=args.width
            ) as renderer:
                with imageio.get_writer(
                    args.record, fps=args.fps, codec="libx264", macro_block_size=1
                ) as writer:

                    def capture(current):
                        if not frame_clock.due(float(current.data.time)):
                            return
                        before = time.monotonic()
                        renderer.update_scene(current.data, camera=camera)
                        if args.visual_quality == "balanced":
                            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = (
                                False
                            )
                            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                        writer.append_data(
                            _compose_camera_panel(
                                renderer.render(), current, recording_panel
                            )
                        )
                        frame_clock.captured(
                            float(current.data.time), time.monotonic() - before
                        )

                    capture(env)
                    while True:
                        observation, _, terminated, truncated, _ = env.step(
                            action(observation), physics_callback=capture
                        )
                        last_progress = _play_progress(env, started, last_progress)
                        if terminated or truncated:
                            break
            print(f"Saved video: {args.record.resolve()}")
        else:
            import mujoco.viewer
            from .live_view import CameraPanel
            from .utilization_view import UtilizationOverlay

            camera_panel = CameraPanel(args.camera_view)
            utilization_overlay = UtilizationOverlay()
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                camera = _camera(
                    mujoco, inset=args.camera_view != "off", num_arms=env.num_arms
                )
                viewer.cam.lookat[:] = camera.lookat
                viewer.cam.distance = camera.distance
                viewer.cam.azimuth = camera.azimuth
                viewer.cam.elevation = camera.elevation
                if args.visual_quality == "balanced" and viewer.user_scn is not None:
                    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                playback_started = time.monotonic()

                def refresh(current):
                    if not frame_clock.due(float(current.data.time)):
                        return
                    before = time.monotonic()
                    camera_panel.update(
                        viewer,
                        getattr(
                            current, "camera_display", getattr(current, "sensor", None)
                        ),
                        float(current.data.time),
                    )
                    utilization_overlay.update(viewer, current)
                    viewer.sync()
                    frame_clock.captured(
                        float(current.data.time), time.monotonic() - before
                    )
                    # Global simulation-time pacing avoids a fresh 100 ms pause
                    # at every policy decision and never accelerates physics.
                    deadline = (
                        playback_started
                        + (float(current.data.time) - frame_clock.start_simulation)
                        / args.speed
                    )
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)

                refresh(env)
                while viewer.is_running():
                    observation, _, terminated, truncated, _ = env.step(
                        action(observation), physics_callback=refresh
                    )
                    last_progress = _play_progress(env, started, last_progress)
                    if terminated or truncated:
                        break
        return _finish_play(
            args, env, started, source_sha256=source_sha256, frame_clock=frame_clock
        )
    except KeyboardInterrupt:
        print("Playback stopped by user; closing viewer/recording.", flush=True)
        return _finish_play(
            args,
            env,
            started,
            interrupted=True,
            source_sha256=source_sha256,
            frame_clock=frame_clock,
        )
    finally:
        close = getattr(env, "close", None)
        if close:
            close()


def check_command(args: argparse.Namespace) -> dict[str, Any]:
    import importlib
    import importlib.util

    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "mujoco_gl": os.environ.get("MUJOCO_GL", "default"),
    }
    if getattr(args, "probe_renderer", False):
        from .rendering import select_backend

        try:
            report["rendering"] = select_backend()
        except ValueError as exc:
            report["rendering"] = {"available": False, "error": str(exc)}
    for name in (
        "numpy",
        "scipy",
        "mujoco",
        "torch",
        "imageio",
        "imageio_ffmpeg",
        "PIL",
        "pytest",
    ):
        if importlib.util.find_spec(name) is None:
            report[name] = {"available": False}
            continue
        try:
            module = importlib.import_module(name)
            report[name] = {
                "available": True,
                "version": getattr(module, "__version__", "unknown"),
            }
            if name == "torch":
                report[name]["cuda_available"] = module.cuda.is_available()
                report[name]["cuda_build"] = module.version.cuda
        except Exception as exc:
            report[name] = {"available": False, "error": str(exc)}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "train":
        from .train import main as train_main

        train_main(arguments[1:])
        return
    if arguments and arguments[0] == "train-scheduler":
        from .hierarchical_train import main as train_scheduler_main

        train_scheduler_main(arguments[1:])
        return
    args = parser().parse_args(arguments)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
