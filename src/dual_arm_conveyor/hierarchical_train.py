"""Train a four-arm masked PPO scheduler over physical finger-contact rollouts.

The learned policy selects joint assignments. The existing checked motion and
finger-contact controller executes them at the unchanged physics timestep.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from functools import partial

import torch

from .cli import add_sensor_args, positive_float, positive_int, sensor_config_kwargs, source_hashes
from .learning import Policy, PPOConfig, PPOTrainer, evaluate_policy, load_policy, write_json


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=Path("results/hierarchical_rl_v082"))
    result.add_argument("--steps", type=int, default=8192, help="Total PPO physical decision steps, including resumed work")
    result.add_argument("--warmstart-steps", type=int, default=2048, help="Separate physical heuristic steps for BC initialization")
    result.add_argument("--warmstart-epochs", type=positive_int, default=80)
    result.add_argument("--warmstart-learning-rate", type=positive_float, default=3e-4)
    result.add_argument("--warmstart-minibatch-size", type=positive_int, default=64)
    result.add_argument("--warmstart-max-examples", type=positive_int, default=None,
                        help="Optional uniform reservoir cap for BC choice-state examples; physical sampling steps are unchanged")
    result.add_argument("--warmstart-label-smoothing", type=float, default=0.05)
    result.add_argument("--rollout-steps", type=positive_int, default=256)
    result.add_argument("--epochs", type=positive_int, default=4)
    result.add_argument("--minibatch-size", type=positive_int, default=64)
    result.add_argument("--learning-rate", type=positive_float, default=3e-4)
    result.add_argument("--entropy-coef", type=float, default=0.005)
    result.add_argument("--expert-coef", type=float, default=0.2)
    result.add_argument("--expert-final-coef", type=float, default=0.05)
    result.add_argument("--expert-decay-steps", type=int, default=8192)
    result.add_argument("--expert-label-smoothing", type=float, default=0.05)
    result.add_argument("--expert-balance", action="store_true",
                        help="Balance online expert loss by action-group frequency over each rollout")
    result.add_argument("--teacher-mode", choices=("returning_partner_soft_preference", "predictive", "sequence"), default="returning_partner_soft_preference")
    result.add_argument("--direct-retask", action="store_true")
    result.add_argument("--nearest-bin-slot", action="store_true")
    result.add_argument("--responsive-release", action="store_true")
    result.add_argument("--direct-bin-release", action="store_true")
    result.add_argument("--extended-pick-deadline", action="store_true")
    result.add_argument("--active-recovery", action="store_true")
    result.add_argument("--future-planning", action="store_true")
    result.add_argument("--safe-preposition", action="store_true")
    result.add_argument("--planned-return", action="store_true")
    result.add_argument("--guarded-descent", action="store_true")
    result.add_argument("--stable-completion-reward", action="store_true")
    result.add_argument("--cooperative-motion", action="store_true",
                        help="Return the released empty hand outside the shared strip while retaining its lock")
    result.add_argument("--cooperative-drop-cost", action="store_true",
                        help="Independent experimental deposit-slot clearance-time cost")
    result.add_argument("--downstream-completion", action="store_true",
                        help="Experimental near-side downstream completion extension with unchanged path checks")
    result.add_argument("--reward-scale", type=positive_float, default=0.2)
    result.add_argument("--gamma", type=float, default=0.995)
    result.add_argument("--gae-lambda", type=float, default=0.98)
    result.add_argument("--hidden-dim", type=positive_int, default=64)
    result.add_argument("--actor-context", choices=("independent", "set"), default="independent")
    result.add_argument("--device", default="cpu")
    result.add_argument("--resume", type=Path)
    result.add_argument("--initialize-from", type=Path, help="Transfer compatible model weights into a fresh run; reset optimizer/counters, permit oracle-to-vision sensor transfer")
    result.add_argument("--reset-optimizer-on-resume", action="store_true", help="Explicitly start fresh Adam moments after loading a checkpoint; weights and PPO counters are preserved")
    result.add_argument("--seed", type=int, default=1000, help="First training arrival/pose seed; reset increments it")
    result.add_argument("--catalog-seed", type=int, default=801, help="First training catalogue; default bank has this and the next three seeds")
    result.add_argument("--catalog-seeds", type=int, nargs="+", help="Explicit training catalogue bank, rotated deterministically per episode")
    result.add_argument("--validation-seed", type=int, default=200000)
    result.add_argument("--validation-catalog-seed", type=int, default=1801)
    result.add_argument("--validation-catalog-seeds", type=int, nargs="+", help="Held-out catalogue bank rotated by validation episode seed")
    result.add_argument("--validation-episodes", type=int, default=1)
    result.add_argument("--validation-every", type=int, default=2048, help="PPO steps between validation; 0 means final only")
    result.add_argument("--eval-seed", type=int, default=300000)
    result.add_argument("--eval-catalog-seed", type=int, default=2801)
    result.add_argument("--eval-episodes", type=int, default=1)
    result.add_argument("--eval-bc", action="store_true", help="Also test the separate BC checkpoint, never select it as PPO")
    result.add_argument("--validate-bc", action="store_true", help="Evaluate the BC initialization on the validation cohort before PPO; never use it as a PPO candidate")
    result.add_argument("--total-objects", type=positive_int, default=100)
    result.add_argument("--pool-size", type=positive_int, default=384)
    result.add_argument("--bin-capacity", type=positive_int, default=180)
    result.add_argument("--belt-speed", type=positive_float, default=0.2)
    result.add_argument("--spacing", type=positive_float, default=0.60)
    result.add_argument("--spacing-jitter", type=float, default=0.18)
    result.add_argument("--min-object-gap", type=float, default=0.08)
    result.add_argument("--physics-dt", type=positive_float, default=0.0025)
    result.add_argument("--control-speed", type=positive_float, default=0.65)
    result.add_argument("--max-time", type=positive_float)
    result.add_argument("--sensor-mode", choices=("vision", "oracle"), default="vision")
    result.add_argument("--render-backend", choices=("egl", "auto", "osmesa", "glfw"), default="egl")
    add_sensor_args(result)
    return result


def scene_config(args):
    from .config import Config

    return Config(
        num_arms=4, gripper_mode="fingers", strategy="adaptive", motion_mode="optimized",
        use_dispatch_preferences=False, total_objects=args.total_objects,
        direct_retask=args.direct_retask, nearest_bin_slot=args.nearest_bin_slot,
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
        pool_size=args.pool_size, bin_capacity=args.bin_capacity,
        deep_bins=True, auto_unload=False, control_speed=args.control_speed,
        physics_dt=args.physics_dt, decision_dt=0.1, max_time=args.max_time,
        belt_profile="constant", belt_speed=args.belt_speed, belt_speed_amplitude=0.0,
        arrival_mode="distance", supply_lanes=2, spacing=args.spacing,
        spacing_jitter=args.spacing_jitter, min_object_gap=args.min_object_gap,
        lane_offset=0.065, lane_jitter=0.012,
        sensor_mode=args.sensor_mode, seed=args.catalog_seed, **sensor_config_kwargs(args),
    )


class SeedGuard:
    """Reject held-out arrival seeds before resetting the physical training env."""

    def __init__(self, env, forbidden):
        self.env = env
        self.forbidden = frozenset(forbidden)
        self.seeds = []

    def reset(self, seed=None):
        if seed is None or int(seed) in self.forbidden:
            raise ValueError("Training reset would use an unspecified or held-out seed")
        self.seeds.append(int(seed))
        return self.env.reset(seed=seed)

    def __getattr__(self, name):
        return getattr(self.env, name)


class CatalogBankEnv:
    """Rebuild object physics on catalogue changes, without changing control."""

    def __init__(self, config, bank, first_episode_seed, factory):
        self.base_config = config
        self.bank = tuple(bank)
        self.first_episode_seed = int(first_episode_seed)
        self.factory = factory
        self.current_catalog = self.bank[0]
        self.env = factory(replace(config, seed=self.current_catalog))
        self.catalog_visits = []
        self.catalogue_gc_events = []

    def reset(self, seed=None):
        if seed is None:
            raise ValueError("Catalogue rotation requires an explicit episode seed")
        catalog = self.bank[(int(seed) - self.first_episode_seed) % len(self.bank)]
        previous_catalog = self.current_catalog
        if catalog != self.current_catalog:
            self.env.close()
            self.env = None
            self.env = self.factory(replace(self.base_config, seed=catalog))
            self.current_catalog = catalog
        observation = self.env.reset(seed=seed)
        self.catalog_visits.append({"episode_seed": int(seed), "catalog_seed": catalog})
        if self.current_catalog != previous_catalog:
            # Retired env -> future planner -> env cycles retain MuJoCo models
            # until generation-2 GC. Collect only after the new reset succeeds;
            # neither failed resets nor same-catalogue resets trigger this.
            started = time.monotonic()
            collected = gc.collect(2)
            self.catalogue_gc_events.append({
                "previous_catalogue": previous_catalog,
                "current_catalogue": self.current_catalog,
                "generation": 2, "collected": collected,
                "elapsed_seconds": time.monotonic() - started,
            })
        return observation

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __getattr__(self, name):
        if self.env is None:
            raise AttributeError(name)
        return getattr(self.env, name)


def training_catalogs(args):
    bank = list(args.catalog_seeds) if args.catalog_seeds is not None else list(range(args.catalog_seed, args.catalog_seed + 4))
    if not bank or any(seed < 0 for seed in bank) or len(set(bank)) != len(bank):
        raise ValueError("Training catalogue bank must contain distinct nonnegative seeds")
    return bank


def validation_catalogs(args):
    bank = list(args.validation_catalog_seeds) if args.validation_catalog_seeds is not None else [args.validation_catalog_seed]
    if not bank or any(seed < 0 for seed in bank) or len(set(bank)) != len(bank):
        raise ValueError("Validation catalogue bank must contain distinct nonnegative seeds")
    return bank


def validate_splits(args):
    if args.initialize_from and (args.resume or args.warmstart_steps):
        raise ValueError("--initialize-from requires --warmstart-steps 0 and cannot be combined with --resume")
    if args.reset_optimizer_on_resume and not args.resume:
        raise ValueError("--reset-optimizer-on-resume requires --resume")
    for name in ("steps", "warmstart_steps", "validation_episodes", "eval_episodes", "validation_every"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if args.entropy_coef < 0:
        raise ValueError("entropy_coef must be nonnegative")
    if not 0 <= args.warmstart_label_smoothing <= 1:
        raise ValueError("warmstart_label_smoothing must be in [0, 1]")
    seed_names = ("seed", "catalog_seed", "validation_seed", "validation_catalog_seed", "eval_seed", "eval_catalog_seed")
    if any(getattr(args, name) < 0 for name in seed_names):
        raise ValueError("All episode and catalogue seeds must be nonnegative")
    validation = list(range(args.validation_seed, args.validation_seed + args.validation_episodes))
    final = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
    if set(validation) & set(final) or args.seed in validation + final:
        raise ValueError("Training, validation and final episode seeds must be disjoint")
    bank = training_catalogs(args)
    valid_bank = validation_catalogs(args)
    if args.eval_catalog_seed in valid_bank or set(bank) & (set(valid_bank) | {args.eval_catalog_seed}):
        raise ValueError("Training, validation and final catalogue seeds must be disjoint")
    return validation, final


def summarize_evaluation(result, requested_objects):
    episodes = result["episodes"]
    requested = requested_objects * len(episodes)
    totals = {name: sum(int(ep["info"].get(name, 0)) for ep in episodes) for name in (
        "sorted", "missed", "dropped", "unfinished", "robot_collision_steps", "inter_arm_collision_steps"
    )}
    complete = bool(episodes) and all(
        ep["terminated"] and not ep["truncated"]
        and ep["info"].get("termination_reason") == "finished"
        and ep["info"].get("unfinished") == 0
        and ep["info"].get("total_objects") == requested_objects
        and sum(ep["info"].get(key, 0) for key in ("sorted", "missed", "dropped")) == requested_objects
        for ep in episodes
    )
    seconds = sum(float(ep["info"].get("time", 0.0)) for ep in episodes)
    return {
        **totals, "requested": requested, "complete": complete,
        "success_rate": totals["sorted"] / max(requested, 1),
        "simulation_seconds": seconds,
        "throughput_per_minute": 60 * totals["sorted"] / max(seconds, 1e-9),
        "selection_eligible": complete and totals["robot_collision_steps"] == 0 and totals["inter_arm_collision_steps"] == 0,
    }


def selection_key(summary):
    if not summary["selection_eligible"]:
        return None
    return (summary["success_rate"], -summary["dropped"], summary["throughput_per_minute"])


def actor_updates(trainer):
    return int(getattr(trainer, "ppo_actor_optimizer_steps", sum(
        item.get("actor_optimizer_steps", 0) for item in trainer.history
    )))


def has_ppo_actor_updates(trainer):
    return trainer.ppo_optimizer_steps > 0 and trainer.ppo_decision_steps > 0 and actor_updates(trainer) > 0


def _make_env(config, teacher_mode="returning_partner_soft_preference"):
    from .env import ConveyorEnv
    from .scheduler import SchedulerEnv

    return SchedulerEnv(ConveyorEnv(config), teacher_mode=teacher_mode)


def main(argv=None):
    args = parser().parse_args(argv)
    validation_seeds, test_seeds = validate_splits(args)
    bank = training_catalogs(args)
    args.catalog_seed = bank[0]
    validation_bank = validation_catalogs(args)
    args.validation_catalog_seed = validation_bank[0]
    initial_hashes = source_hashes()
    resume_sha256, resume_sources, source_changed = None, None, False
    source_lineage, optimizer_boundaries = [], []
    initialization = None
    if args.initialize_from:
        initialized = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
        initialization = {
            "path": str(args.initialize_from.resolve()),
            "sha256": hashlib.sha256(args.initialize_from.read_bytes()).hexdigest(),
            "source_sha256": initialized.get("hierarchical_training", {}).get("source_sha256"),
            "source_steps": initialized.get("steps", 0),
            "source_actor_optimizer_steps": initialized.get("ppo_actor_optimizer_steps", 0),
            "source_sensor_mode": initialized.get("env_config", {}).get("sensor_mode"),
            "source_first_episode_seed": initialized.get("train_config", {}).get("seed"),
            "source_reset_count": initialized.get("reset_count", 0),
            "source_catalog_bank": initialized.get("hierarchical_training", {}).get("catalog_bank", []),
            "parent_initialization": initialized.get("hierarchical_training", {}).get("initialization"),
            "semantics": "weights only; fresh optimizer and all current-run training counters start at zero",
        }
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    if args.resume:
        resume_sha256 = hashlib.sha256(args.resume.read_bytes()).hexdigest()
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        provenance = checkpoint.get("hierarchical_training", {})
        resume_sources = provenance.get("source_sha256")
        source_changed = resume_sources != initial_hashes
        if source_changed and (
            args.output.resolve() == args.resume.parent.resolve()
            or any((args.output / name).exists() for name in ("training.json", "selection.json", "updates.jsonl", "latest.pt", "best.pt"))
        ):
            raise ValueError("Cross-source resume requires a new output directory to preserve earlier evidence")
        source_lineage = list(provenance.get("source_lineage", []))
        source_lineage.append({
            "checkpoint": str(args.resume.resolve()), "checkpoint_sha256": resume_sha256,
            "source_sha256": resume_sources, "steps": int(checkpoint.get("steps", 0)),
            "source_changed": source_changed,
        })
        optimizer_boundaries = list(provenance.get("optimizer_boundaries", []))
        if checkpoint.get("train_config", {}).get("seed") != args.seed:
            raise ValueError("Resume requires the original training episode seed")
        if checkpoint.get("hierarchical_training", {}).get("catalog_bank") != bank:
            raise ValueError("Resume requires the original documented training catalogue bank")
        old_last = args.seed + int(checkpoint.get("reset_count", 0))
        if any(args.seed <= seed <= old_last for seed in validation_seeds + test_seeds):
            raise ValueError("Resumed training history overlaps held-out episode seeds")
    from .rendering import metadata as rendering_metadata
    from .rendering import select_backend

    if args.sensor_mode == "vision":
        select_backend(args.render_backend)
    config = scene_config(args)
    ppo = PPOConfig(
        steps=args.steps, rollout_steps=args.rollout_steps, epochs=args.epochs,
        minibatch_size=args.minibatch_size, learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef, hidden_dim=args.hidden_dim, seed=args.seed,
        actor_context=args.actor_context,
        device=args.device, gamma=args.gamma, gae_lambda=args.gae_lambda,
        expert_coef=args.expert_coef, expert_final_coef=args.expert_final_coef,
        expert_decay_steps=args.expert_decay_steps,
        expert_label_smoothing=args.expert_label_smoothing, reward_scale=args.reward_scale,
        expert_balance=args.expert_balance,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    metadata = {
        "schema_version": 1, "status": "running",
        "algorithm": "masked PPO with online expert auxiliary" if max(args.expert_coef, args.expert_final_coef) > 0 else "masked PPO hierarchical task scheduling",
        "training_physics": "MuJoCo finger contacts; controller options recorded in env_config",
        "decision_mode": "fixed 0.1 s; no event-step time compression",
        "env_config": asdict(config), "ppo_config": asdict(ppo), "arguments": vars(args),
        "source_sha256": initial_hashes, "rendering": rendering_metadata(),
        "catalogues": {"training": bank, "validation": args.validation_catalog_seed, "test": args.eval_catalog_seed},
        "validation_catalog_bank": validation_bank,
        "catalogue_note": "Training rotates catalogues by (episode_seed-first_training_seed) modulo bank size; validation/test catalogues never train.",
        "validation_seeds": validation_seeds, "test_seeds": test_seeds,
        "resume": str(args.resume) if args.resume else None,
        "resume_checkpoint_sha256": resume_sha256, "resume_source_sha256": resume_sources,
        "resume_source_changed": source_changed, "source_lineage": source_lineage,
        "initialization": initialization,
        "optimizer_boundaries": optimizer_boundaries,
        "gradient_clipping": "separate_actor_critic",
        "reward_units": {"environment_and_report": "raw", "critic_and_GAE_scale": args.reward_scale},
        "expert_supervision": {
            "teacher": "teacher_action when available; heuristic_action fallback",
            "physical_execution": "sampled PPO action; expert labels never replace it",
            "student_mask": "soft-plan feasibility and physically checked dispatch" if args.future_planning else "physical-only",
            "coefficient_start": args.expert_coef, "coefficient_final": args.expert_final_coef,
            "linear_decay_ppo_steps": args.expert_decay_steps,
            "label_smoothing": args.expert_label_smoothing,
            "smoothing_support": "only valid physical actions",
            "online_class_balancing": args.expert_balance,
            "online_balance_scope": "known non-forced labels over each rollout" if args.expert_balance else None,
            "online_inverse_weight_cap_before_normalization": 4.0 if args.expert_balance else None,
        },
        "selection_metric": "complete collision-free cohorts, then sorted/requested, fewer drops, throughput",
        "final_evaluation_completed": False,
    }
    selection = {"candidates": [], "selected": None, "selected_policy_kind": None}
    previous_best_key = None
    if args.resume and not source_changed:
        old_metadata = args.resume.parent / "training.json"
        old_selection = args.resume.parent / "selection.json"
        if old_metadata.exists() and old_selection.exists():
            previous = json.loads(old_metadata.read_text())
            saved_selection = json.loads(old_selection.read_text())
            selected = saved_selection.get("selected")
            matching_validation = (
                previous.get("validation_seeds") == validation_seeds
                and previous.get("catalogues", {}).get("validation") == args.validation_catalog_seed
                and previous.get("validation_catalog_bank", [args.validation_catalog_seed]) == validation_bank
                and previous.get("env_config") == asdict(config)
                and previous.get("source_sha256") == initial_hashes
            )
            old_best = args.resume.parent / "best.pt"
            if matching_validation and selected and old_best.exists():
                checked = torch.load(old_best, map_location="cpu", weights_only=True)
                old_actor_updates = checked.get("ppo_actor_optimizer_steps", sum(
                    item.get("actor_optimizer_steps", 0) for item in checked.get("history", [])
                ))
                if (
                    checked.get("ppo_optimizer_steps", 0) > 0
                    and checked.get("ppo_decision_steps", 0) > 0 and old_actor_updates > 0
                    and checked.get("hierarchical_training", {}).get("source_sha256") == initial_hashes
                ):
                    previous_best_key = selection_key(selected["summary"])
                    if previous_best_key is not None:
                        if old_best.resolve() != (args.output / "best.pt").resolve():
                            shutil.copy2(old_best, args.output / "best.pt")
                        selection = saved_selection
    trainer = None
    env_factory = _make_env if args.teacher_mode == "returning_partner_soft_preference" else partial(_make_env, teacher_mode=args.teacher_mode)
    with ExitStack() as stack:
        train_base = CatalogBankEnv(config, bank, args.seed, env_factory)
        stack.callback(train_base.close)
        train_env = SeedGuard(train_base, validation_seeds + test_seeds)
        trainer = PPOTrainer(train_env, ppo, asdict(config))
        if initialization is not None:
            if hashlib.sha256(args.initialize_from.read_bytes()).hexdigest() != initialization["sha256"]:
                raise RuntimeError("Initialization checkpoint changed during setup")
            if initialized.get("architecture") != trainer.model.architecture:
                raise ValueError("Initialization model dimensions differ")
            for method, key in (("action_metadata", "scheduler_action_metadata"), ("observation_metadata", "scheduler_observation_metadata")):
                if initialized.get(key) != getattr(train_base, method)():
                    raise ValueError("Initialization scheduler schema differs")
            current_scene, old_scene = asdict(config), initialized.get("env_config", {})
            # Older checkpoints omit these flags; each then means False.
            cooperation_options = ("cooperative_motion", "cooperative_drop_cost", "downstream_completion")
            old_scene = dict(old_scene, **{option: old_scene.get(option, False)
                                         for option in cooperation_options})
            differences = {key for key in set(current_scene) | set(old_scene) if current_scene.get(key) != old_scene.get(key)}
            permitted_changes = {"sensor_mode", "seed"}
            permitted_changes.update(option for option in cooperation_options if getattr(args, option))
            if differences - permitted_changes:
                raise ValueError(f"Initialization scene differs beyond explicit sensor/seed/cooperation transfer: {sorted(differences)}")
            trainer.model.load_state_dict(initialized["model"])
            initialization["scene_differences"] = sorted(differences)
            initialization["scene_change_details"] = {
                key: {"source": old_scene.get(key), "current": current_scene.get(key)}
                for key in sorted(differences)
            }
            assert not trainer.optimizer.state and trainer.steps == 0
        metadata["scheduler_action_metadata"] = train_base.action_metadata()
        metadata["scheduler_observation_metadata"] = train_base.observation_metadata()
        metadata["expert_supervision"]["teacher_mode"] = (
            args.teacher_mode
            if callable(getattr(train_base, "teacher_action", None)) else "heuristic_fallback"
        )
        describe_teacher = getattr(train_base, "teacher_metadata", None)
        if callable(describe_teacher):
            metadata["teacher_metadata"] = describe_teacher()
        validation_env = None
        if validation_seeds:
            validation_env = CatalogBankEnv(replace(config, seed=args.validation_catalog_seed), validation_bank, args.validation_seed, env_factory)
            stack.callback(validation_env.close)
        best_key, last_validation = previous_best_key, 0

        def save_checkpoint(path):
            trainer.save(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint["hierarchical_training"] = {
                "catalog_bank": bank, "first_episode_seed": args.seed,
                "catalog_mapping": "(episode_seed-first_episode_seed) % len(catalog_bank)",
                "source_sha256": initial_hashes,
                "source_lineage": source_lineage,
                "optimizer_boundaries": optimizer_boundaries,
                "initialization": initialization,
            }
            temporary = path.with_name(path.name + ".metadata.tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)

        def progress(record):
            print(json.dumps({"event": "training_progress", "wall_seconds": time.monotonic() - started, **record}), flush=True)

        def reset_stage_optimizer(reason):
            def counters():
                return {
                    "steps": trainer.steps, "ppo_decision_steps": trainer.ppo_decision_steps,
                    "ppo_optimizer_steps": trainer.ppo_optimizer_steps,
                    "actor_optimizer_steps": actor_updates(trainer),
                }
            before = counters()
            previous_states = len(trainer.optimizer.state)
            trainer.reset_optimizer()
            optimizer_boundaries.append({
                "reason": reason, "before": before, "after": counters(),
                "adam_state_entries_before": previous_states,
                "adam_state_entries_after": len(trainer.optimizer.state),
                "resume_checkpoint_sha256": resume_sha256,
            })

        def write_metadata():
            current_hashes = source_hashes()
            metadata.update({
                "wall_seconds": time.monotonic() - started,
                "source_at_end": current_hashes, "source_unchanged": current_hashes == initial_hashes,
                "actual_training_episode_seeds": train_env.seeds,
                "actual_training_catalogue_visits": train_base.catalog_visits,
                "catalogue_retirement_gc": train_base.catalogue_gc_events,
                "steps": trainer.steps, "ppo_decision_steps": trainer.ppo_decision_steps,
                "ppo_optimizer_steps": trainer.ppo_optimizer_steps,
                "actor_optimizer_steps": actor_updates(trainer),
                "ppo_episodes": trainer.ppo_episodes,
                "expert_label_steps": trainer.expert_label_steps,
                "expert_optimizer_steps": trainer.expert_optimizer_steps,
                "has_actual_ppo_actor_updates": has_ppo_actor_updates(trainer),
            })
            write_json(args.output / "training.json", metadata)
            write_json(args.output / "selection.json", selection)
            if not metadata["source_unchanged"]:
                raise RuntimeError("Training source changed during this run")

        def evaluate(env, policy, seeds):
            result = evaluate_policy(env, policy, seeds, progress)
            result["summary"] = summarize_evaluation(result, args.total_objects)
            return result

        def select_checkpoint(label):
            nonlocal best_key, last_validation
            if not has_ppo_actor_updates(trainer):
                return  # BC or critic-only updates cannot become an RL best.
            if validation_env is None:
                return
            result = evaluate(validation_env, Policy(trainer.model, args.device), validation_seeds)
            candidate = {"label": label, "steps": trainer.steps, "ppo_optimizer_steps": trainer.ppo_optimizer_steps, **result}
            selection["candidates"].append(candidate)
            key = selection_key(result["summary"])
            if key is not None and (best_key is None or key > best_key):
                best_key = key
                save_checkpoint(args.output / "best.pt")
                selection["selected"] = {"path": "best.pt", "steps": trainer.steps, "summary": result["summary"]}
                selection["selected_policy_kind"] = "PPO"
            last_validation = trainer.steps
            trainer.model.train()
            write_metadata()

        def update(record):
            save_checkpoint(args.output / "latest.pt")
            with (args.output / "updates.jsonl").open("a") as output:
                output.write(json.dumps({**record, "wall_seconds": time.monotonic() - started, "source_sha256": source_hashes()}) + "\n")
            write_metadata()
            progress({"phase": "ppo_update", "steps": trainer.steps, "episodes": trainer.ppo_episodes,
                      "decision_steps": record.get("decision_steps"), "actor_optimizer_steps": record.get("actor_optimizer_steps")})
            if args.validation_every and trainer.steps - last_validation >= args.validation_every:
                select_checkpoint(f"ppo_{trainer.steps}")

        try:
            write_metadata()
            if args.resume:
                if hashlib.sha256(args.resume.read_bytes()).hexdigest() != resume_sha256:
                    raise RuntimeError("Resume checkpoint changed after provenance inspection")
                trainer.resume(args.resume)
                # A new stage should collect a full validation interval before
                # assessing its first update, even when cumulative steps are large.
                last_validation = trainer.steps
                metadata["validation_interval_start_step"] = trainer.steps
                if args.reset_optimizer_on_resume:
                    reset_stage_optimizer("explicit_resume_reset")
                write_metadata()
            elif args.warmstart_steps:
                metadata["warmstart"] = trainer.warmstart(
                    args.warmstart_steps, args.warmstart_epochs, progress,
                    learning_rate=args.warmstart_learning_rate,
                    minibatch_size=args.warmstart_minibatch_size,
                    label_smoothing=args.warmstart_label_smoothing,
                    max_examples=args.warmstart_max_examples,
                )
                save_checkpoint(args.output / "bc.pt")
                metadata["bc_checkpoint"] = "bc.pt"
                reset_stage_optimizer("bc_to_ppo")
                if args.validate_bc and validation_env is not None:
                    bc_validation = {
                        "kind": "BC initialization; not a PPO selection candidate",
                        "checkpoint_sha256": hashlib.sha256((args.output / "bc.pt").read_bytes()).hexdigest(),
                        "source_sha256": initial_hashes,
                        "evaluation": evaluate(validation_env, Policy(trainer.model, args.device), validation_seeds),
                    }
                    write_json(args.output / "bc_validation.json", bc_validation)
                    metadata["bc_validation_summary"] = bc_validation["evaluation"]["summary"]
                    trainer.model.train()
                write_metadata()
            trainer.train(on_update=update, on_progress=progress)
            save_checkpoint(args.output / "latest.pt")
            if trainer.steps != last_validation:
                select_checkpoint("ppo_final")
            actual_rl = has_ppo_actor_updates(trainer)
            selected_path = args.output / "best.pt" if selection["selected"] else args.output / "latest.pt"
            if actual_rl and test_seeds:
                test_env = env_factory(replace(config, seed=args.eval_catalog_seed))
                stack.callback(test_env.close)
                report = {
                    "schema_version": 1, "sensor_mode": config.sensor_mode, "seeds": test_seeds,
                    "catalog_seed": args.eval_catalog_seed,
                    "checkpoint": str(selected_path),
                    "checkpoint_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
                    "checkpoint_selection": "validation best" if selection["selected"] else "unselected final PPO; no eligible validation best",
                    "source_sha256": initial_hashes, "env_config": asdict(test_env.config),
                    "learned": evaluate(test_env, load_policy(selected_path, args.device), test_seeds),
                    "heuristic_same_physical_mask": evaluate(test_env, None, test_seeds),
                }
                if args.eval_bc and (args.output / "bc.pt").exists():
                    report["bc_initialization"] = evaluate(test_env, load_policy(args.output / "bc.pt", args.device), test_seeds)
                write_json(args.output / "evaluation.json", report)
                metadata["final_evaluation_attempted"] = True
                metadata["final_evaluation_completed"] = all(
                    value["summary"]["complete"]
                    for value in report.values()
                    if isinstance(value, dict) and "summary" in value
                )
            metadata["status"] = "complete" if actual_rl else "no_ppo_actor_updates"
            write_metadata()
        except BaseException as exc:
            metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            metadata["error"] = f"{type(exc).__name__}: {exc}"
            save_checkpoint(args.output / "latest.pt")
            # Preserve diagnostics even when the exception was a source change.
            metadata["source_at_end"] = source_hashes()
            metadata["source_unchanged"] = metadata["source_at_end"] == initial_hashes
            metadata["wall_seconds"] = time.monotonic() - started
            write_json(args.output / "training.json", metadata)
            write_json(args.output / "selection.json", selection)
            raise
    return metadata


if __name__ == "__main__":
    main()
