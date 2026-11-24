"""Training provenance/selection tests; fixture scores are not physics claims."""

import json

import numpy as np
import pytest
import torch
from dual_arm_conveyor import hierarchical_train as train


def test_defaults_preserve_current_physical_visual_scene():
    args = train.parser().parse_args([])
    config = train.scene_config(args)
    assert (args.steps, args.warmstart_steps, args.render_backend) == (8192, 2048, "egl")
    assert (args.expert_coef, args.expert_final_coef, args.reward_scale) == (0.2, 0.05, 0.2)
    assert args.expert_balance is False
    assert (args.warmstart_epochs, args.warmstart_minibatch_size, args.warmstart_learning_rate) == (80, 64, 3e-4)
    assert train.training_catalogs(args) == [801, 802, 803, 804]
    assert config.num_arms == 4 and config.gripper_mode == "fingers"
    assert config.sensor_mode == "vision" and not config.use_dispatch_preferences
    assert config.belt_speed == 0.2 and config.supply_lanes == 2
    assert (config.spacing, config.spacing_jitter, config.min_object_gap) == (0.60, 0.18, 0.08)
    assert config.deep_bins and not config.auto_unload
    assert (config.pool_size, config.bin_capacity, config.physics_dt, config.decision_dt) == (384, 180, 0.0025, 0.1)
    oracle = train.scene_config(train.parser().parse_args(["--sensor-mode", "oracle"]))
    assert oracle.sensor_mode == "oracle"
    assert oracle.belt_speed == config.belt_speed


def test_guarded_descent_training_and_deployment_configs_match():
    from dual_arm_conveyor import cli
    trained = train.scene_config(train.parser().parse_args(["--guarded-descent"]))
    deployed = cli.config_from_args(cli.parser().parse_args([
        "play", "--guarded-descent", "--strategy", "adaptive", "--num-arms", "4",
        "--deep-bins", "--no-auto-unload",
    ]))
    assert trained.guarded_descent and deployed.guarded_descent
    assert not train.scene_config(train.parser().parse_args([])).guarded_descent


@pytest.mark.parametrize("arguments", [
    ["--validation-seed", "1000"],
    ["--validation-seed", "300000"],
    ["--validation-catalog-seed", "803"],
    ["--eval-catalog-seed", "1801"],
    ["--catalog-seeds", "801", "801"],
    ["--validation-catalog-seeds", "1801", "803"],
    ["--validation-catalog-seeds", "1801", "1801"],
])
def test_episode_and_catalogue_splits_reject_leakage(arguments):
    with pytest.raises(ValueError):
        train.validate_splits(train.parser().parse_args(arguments))


def test_seed_guard_rejects_before_underlying_reset():
    class ResetRecorder:
        def reset(self, seed=None):
            raise AssertionError("Held-out seed reached the environment")
    guarded = train.SeedGuard(ResetRecorder(), [10])
    with pytest.raises(ValueError):
        guarded.reset(seed=10)
    assert not guarded.seeds


def test_catalogue_rotation_is_reproducible_and_closes_replaced_models():
    instances = []
    class Recorder:
        def __init__(self, config):
            self.config, self.closed = config, False
            instances.append(self)
        def reset(self, seed=None):
            return self.config.seed, seed
        def close(self):
            self.closed = True
    config = train.scene_config(train.parser().parse_args([]))
    bank = train.CatalogBankEnv(config, [801, 802], 1000, Recorder)
    assert bank.reset(1000) == (801, 1000)
    assert bank.reset(1001) == (802, 1001)
    assert instances[0].closed
    assert bank.reset(1004) == (801, 1004)  # Same mapping after resume/gaps.
    assert instances[1].closed
    bank.close()
    assert all(env.closed for env in instances)


def test_selection_uses_requested_cohort_and_rejects_truncation_or_collision():
    def evaluation(sorted_count, missed, *, reward=1000, truncated=False, collisions=0):
        return {"mean_reward": reward, "episodes": [{"terminated": not truncated, "truncated": truncated, "info": {
            "total_objects": 100, "sorted": sorted_count, "missed": missed, "dropped": 0,
            "unfinished": 100 - sorted_count - missed, "robot_collision_steps": collisions,
            "inter_arm_collision_steps": 0, "time": 150.0, "termination_reason": "finished",
        }}]}
    better = train.summarize_evaluation(evaluation(95, 5, reward=0), 100)
    worse = train.summarize_evaluation(evaluation(90, 10, reward=10000), 100)
    assert train.selection_key(better) > train.selection_key(worse)
    partial = train.summarize_evaluation(evaluation(95, 0, truncated=True), 100)
    assert partial["success_rate"] == 0.95 and train.selection_key(partial) is None
    assert train.selection_key(train.summarize_evaluation(evaluation(99, 1, collisions=1), 100)) is None


@pytest.fixture
def fixture_environments(monkeypatch):
    instances = []
    class SchedulerFixture:
        """Small optimizer fixture, never used for physical performance reports."""
        dispatch_group_names = {0: "wait", 1: "station0", 2: "station1", 3: "both_stations"}
        def __init__(self, config):
            self.config, self.closed, self.seeds = config, False, []
            instances.append(self)
        def observation(self):
            features = np.zeros((441, 2), dtype=np.float32)
            features[1, 0] = 1
            mask = np.zeros(441, dtype=bool)
            mask[[0, 1, 440]] = True
            return {"features": features, "global": np.zeros(1, dtype=np.float32), "mask": mask}
        def reset(self, seed=None):
            self.seeds.append(seed)
            self.steps = 0
            return self.observation()
        def heuristic_action(self):
            return 1
        def step(self, action):
            assert self.observation()["mask"][action]
            self.steps += 1
            return self.observation(), 1.0, self.steps >= 2, False, {
                "fixture": True, "sensor_mode": self.config.sensor_mode,
                "time": float(self.steps), "total_objects": 2,
                "sorted": self.steps, "missed": 0, "dropped": 0,
                "unfinished": max(0, 2 - self.steps), "robot_collision_steps": 0,
                "inter_arm_collision_steps": 0, "termination_reason": "finished",
            }
        def action_metadata(self):
            return {"schema": "test_fixture", "action_count": 441}
        def observation_metadata(self):
            return {"schema": "test_fixture", "feature_dim": 2, "global_dim": 1}
        def close(self):
            self.closed = True
    monkeypatch.setattr(train, "_make_env", SchedulerFixture)
    monkeypatch.setattr(train, "source_hashes", lambda: {"fixture": "fixed"})
    from dual_arm_conveyor import rendering
    monkeypatch.setattr(rendering, "select_backend", lambda backend: {})
    monkeypatch.setattr(rendering, "metadata", lambda: {"fixture": True})
    return instances


def small_args(output):
    return ["--output", str(output), "--steps", "8", "--warmstart-steps", "2",
            "--warmstart-epochs", "1", "--rollout-steps", "4", "--epochs", "1",
            "--minibatch-size", "4", "--hidden-dim", "8", "--total-objects", "2",
            "--validation-every", "4", "--eval-bc"]


def test_actual_optimizer_updates_provenance_and_complete_heldout_evaluation(tmp_path, fixture_environments):
    train.main(small_args(tmp_path))
    metadata = json.loads((tmp_path / "training.json").read_text())
    report = json.loads((tmp_path / "evaluation.json").read_text())
    bc = torch.load(tmp_path / "bc.pt", weights_only=True)
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert bc["ppo_optimizer_steps"] == 0
    assert best["ppo_optimizer_steps"] > 0 and best["ppo_decision_steps"] > 0
    assert metadata["has_actual_ppo_actor_updates"] and metadata["actor_optimizer_steps"] > 0
    assert metadata["status"] == "complete" and metadata["source_unchanged"]
    assert metadata["final_evaluation_completed"]
    assert metadata["algorithm"] == "masked PPO with online expert auxiliary"
    assert metadata["expert_optimizer_steps"] > 0
    assert metadata["expert_supervision"]["student_mask"] == "physical-only"
    assert metadata["reward_units"]["critic_and_GAE_scale"] == 0.2
    boundary = metadata["optimizer_boundaries"][0]
    assert boundary["reason"] == "bc_to_ppo"
    assert boundary["before"] == boundary["after"]
    assert boundary["adam_state_entries_before"] > 0
    assert boundary["adam_state_entries_after"] == 0
    assert bc["optimizer"]["state"]  # The BC artifact precedes the stage reset.
    assert best["hierarchical_training"]["optimizer_boundaries"] == metadata["optimizer_boundaries"]
    assert best["hierarchical_training"]["catalog_bank"] == [801, 802, 803, 804]
    visits = metadata["actual_training_catalogue_visits"]
    assert {visit["catalog_seed"] for visit in visits} == {801, 802, 803, 804}
    assert not {visit["episode_seed"] for visit in visits} & {200000, 300000}
    assert report["learned"]["summary"]["complete"]
    assert report["heuristic_same_physical_mask"]["summary"]["complete"]
    assert report["bc_initialization"]["summary"]["complete"]
    assert all(env.closed and env.config.sensor_mode == "vision" for env in fixture_environments)
    assert len((tmp_path / "updates.jsonl").read_text().splitlines()) == 2


def test_balanced_expert_cli_reaches_optimizer_and_metadata(tmp_path, fixture_environments):
    train.main([*small_args(tmp_path), "--expert-balance"])
    metadata = json.loads((tmp_path / "training.json").read_text())
    rows = [json.loads(line) for line in (tmp_path / "updates.jsonl").read_text().splitlines()]
    assert metadata["ppo_config"]["expert_balance"] is True
    assert metadata["expert_supervision"]["online_class_balancing"] is True
    assert all(row["expert_balance"] and row["expert_balance_statistics"]["eligible_labels"] == 4 for row in rows)


@pytest.mark.parametrize("validate_bc", [False, True])
def test_bc_only_never_becomes_an_rl_best(tmp_path, fixture_environments, validate_bc):
    train.main([*small_args(tmp_path), "--steps", "0", *(["--validate-bc"] if validate_bc else [])])
    metadata = json.loads((tmp_path / "training.json").read_text())
    assert (tmp_path / "bc.pt").exists()
    assert not (tmp_path / "best.pt").exists()
    assert not metadata["has_actual_ppo_actor_updates"]
    assert metadata["status"] == "no_ppo_actor_updates"
    assert not (tmp_path / "evaluation.json").exists()
    assert (tmp_path / "bc_validation.json").exists() == validate_bc
    if validate_bc:
        checked = json.loads((tmp_path / "bc_validation.json").read_text())
        assert checked["evaluation"]["summary"]["complete"]
        assert metadata["bc_validation_summary"]["sorted"] == 2


def test_resume_preserves_best_and_rejects_changed_catalogue_bank(tmp_path, fixture_environments):
    train.main(small_args(tmp_path))
    before = torch.load(tmp_path / "best.pt", weights_only=True)["steps"]
    train.main([*small_args(tmp_path), "--steps", "12", "--resume", str(tmp_path / "latest.pt")])
    assert torch.load(tmp_path / "latest.pt", weights_only=True)["steps"] == 12
    assert torch.load(tmp_path / "best.pt", weights_only=True)["steps"] == before
    with pytest.raises(ValueError, match="catalogue bank"):
        train.main([*small_args(tmp_path), "--resume", str(tmp_path / "latest.pt"), "--catalog-seeds", "801", "805"])


def test_explicit_resume_optimizer_reset_preserves_cumulative_counters(tmp_path, fixture_environments):
    previous, resumed = tmp_path / "previous", tmp_path / "resumed"
    train.main(small_args(previous))
    parent = torch.load(previous / "latest.pt", weights_only=True)
    train.main([*small_args(resumed), "--steps", "12", "--resume", str(previous / "latest.pt"), "--reset-optimizer-on-resume"])
    metadata = json.loads((resumed / "training.json").read_text())
    boundary = metadata["optimizer_boundaries"][-1]
    assert boundary["reason"] == "explicit_resume_reset"
    assert boundary["before"] == boundary["after"]
    assert boundary["before"]["steps"] == parent["steps"] == 8
    assert boundary["before"]["actor_optimizer_steps"] == parent["ppo_actor_optimizer_steps"]
    assert boundary["adam_state_entries_before"] > 0 and boundary["adam_state_entries_after"] == 0
    assert boundary["resume_checkpoint_sha256"] == metadata["resume_checkpoint_sha256"]
    assert metadata["steps"] == 12 and not metadata["resume_source_changed"]


def test_cross_source_resume_requires_new_output_and_does_not_inherit_best(tmp_path, fixture_environments, monkeypatch):
    previous, resumed = tmp_path / "previous", tmp_path / "resumed"
    train.main(small_args(previous))
    old_best_step = torch.load(previous / "best.pt", weights_only=True)["steps"]
    monkeypatch.setattr(train, "source_hashes", lambda: {"fixture": "changed"})
    arguments = ["--steps", "12", "--resume", str(previous / "latest.pt")]
    with pytest.raises(ValueError, match="new output"):
        train.main([*small_args(previous), *arguments])
    train.main([*small_args(resumed), *arguments])
    metadata = json.loads((resumed / "training.json").read_text())
    selection = json.loads((resumed / "selection.json").read_text())
    assert metadata["resume_source_changed"]
    assert metadata["source_lineage"][-1]["source_sha256"] == {"fixture": "fixed"}
    assert metadata["source_sha256"] == {"fixture": "changed"}
    assert selection["selected"]["steps"] == 12 > old_best_step
    assert [candidate["steps"] for candidate in selection["candidates"]] == [12]


def test_resume_validation_waits_for_new_interval(tmp_path, fixture_environments):
    train.main(small_args(tmp_path))
    train.main([*small_args(tmp_path), "--steps", "14", "--rollout-steps", "2",
                "--resume", str(tmp_path / "latest.pt")])
    metadata = json.loads((tmp_path / "training.json").read_text())
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert metadata["validation_interval_start_step"] == 8
    # Periodic at 12, then explicit final evaluation at 14; never early at 10.
    assert [item["steps"] for item in selection["candidates"] if item["steps"] > 8] == [12, 14]


def test_optimizer_reset_flag_requires_resume():
    with pytest.raises(ValueError, match="requires --resume"):
        train.validate_splits(train.parser().parse_args(["--reset-optimizer-on-resume"]))


def test_sensor_transfer_starts_fresh_optimizer_and_counters(tmp_path, fixture_environments):
    source, transferred = tmp_path / "source", tmp_path / "vision"
    train.main([*small_args(source), "--sensor-mode", "oracle"])
    train.main([*small_args(transferred), "--warmstart-steps", "0", "--steps", "4",
                "--initialize-from", str(source / "best.pt")])
    metadata = json.loads((transferred / "training.json").read_text())
    final = torch.load(transferred / "latest.pt", weights_only=True)
    assert metadata["initialization"]["source_sensor_mode"] == "oracle"
    assert metadata["initialization"]["scene_differences"] == ["sensor_mode"]
    assert metadata["env_config"]["sensor_mode"] == "vision"
    assert metadata["steps"] == final["steps"] == 4
    assert metadata["actor_optimizer_steps"] > 0
    assert not metadata["resume"]
    assert not (transferred / "bc.pt").exists()


@pytest.mark.parametrize("arguments", [
    ["--initialize-from", "example.pt"],
    ["--initialize-from", "example.pt", "--warmstart-steps", "0", "--resume", "other.pt"],
])
def test_initialization_rejects_ambiguous_training_stages(arguments):
    with pytest.raises(ValueError, match="initialize-from"):
        train.validate_splits(train.parser().parse_args(arguments))


def test_validation_rotates_heldout_catalogues(tmp_path, fixture_environments):
    train.main([*small_args(tmp_path), '--validation-episodes', '2',
                '--validation-catalog-seeds', '1801', '1802'])
    metadata = json.loads((tmp_path / 'training.json').read_text())
    assert metadata['validation_catalog_bank'] == [1801, 1802]
    visits = {(env.config.seed, seed) for env in fixture_environments for seed in env.seeds}
    assert (1801, 200000) in visits and (1802, 200001) in visits
    assert not any(seed < 200000 and catalogue in (1801, 1802) for catalogue, seed in visits)
