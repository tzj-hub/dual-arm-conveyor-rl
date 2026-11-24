"""Algorithm checks use a tiny fixture, not a substitute for physical evaluation."""

import numpy as np
import pytest
import torch

from dual_arm_conveyor.learning import (
    ActorCritic,
    PPOConfig,
    PPOTrainer,
    Policy,
    generalized_advantages,
    decision_advantages,
    load_policy,
    observation_tensors,
    seed_everything,
)


def observation():
    return {
        "features": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
        "global": np.array([0.5], dtype=np.float32),
        "mask": np.array([True, False, True]),
    }


class TinyEnv:
    """Deterministic finite fixture used solely to exercise the optimizer."""

    def reset(self, seed=None):
        self.steps = 0
        return observation()

    def heuristic_action(self):
        return 2

    def step(self, action):
        assert action != 1
        self.steps += 1
        return (
            observation(),
            float(action == 2),
            self.steps == 4,
            False,
            {"fixture": True},
        )


def test_invalid_action_is_never_sampled():
    seed_everything(17)
    model = ActorCritic(2, 1, 16)
    distribution, value = model(observation_tensors(observation()))
    samples = distribution.sample((1000,))
    assert not (samples == 1).any()
    assert distribution.probs[0, 1] == 0
    assert value.shape == (1,)


def test_all_masked_and_nonfinite_observations_fail():
    obs = observation()
    obs["mask"][:] = False
    with pytest.raises(ValueError, match="at least one"):
        observation_tensors(obs)
    obs = observation()
    obs["features"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        observation_tensors(obs)


def test_time_limit_bootstraps_without_cross_episode_leakage():
    advantages, returns = generalized_advantages(
        rewards=[1, 2, 100],
        values=[0, 0, 0],
        next_values=[3, 4, 0],
        terminated=[False, False, True],
        ended=[False, True, True],
        gamma=0.5,
        gae_lambda=1,
    )
    np.testing.assert_allclose(advantages, [4.5, 4, 100])
    np.testing.assert_allclose(returns, advantages)


def test_warmstart_ppo_checkpoint_and_resume(tmp_path):
    trainer = PPOTrainer(
        TinyEnv(),
        PPOConfig(steps=32, rollout_steps=16, epochs=2, hidden_dim=16, seed=3),
    )
    warmstart = trainer.warmstart(16, epochs=10)
    assert warmstart["decision_examples"] == 16
    initial = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    history = trainer.train()
    assert len(history) == 2 and history[-1]["steps"] == 32
    assert any(
        not torch.equal(initial[key], value)
        for key, value in trainer.model.state_dict().items()
    )
    path = tmp_path / "policy.pt"
    trainer.save(path)
    policy = load_policy(path)
    assert policy(observation()) == Policy(trainer.model)(observation())
    resumed = PPOTrainer(
        TinyEnv(),
        PPOConfig(steps=48, rollout_steps=16, epochs=1, hidden_dim=16, seed=3),
    )
    resumed.resume(path)
    assert resumed.steps == 32
    resumed.train()
    assert resumed.steps == 48
    assert resumed.history[-1]["steps"] == 48


def test_cpu_training_is_reproducible():
    models = []
    for _ in range(2):
        trainer = PPOTrainer(
            TinyEnv(),
            PPOConfig(steps=16, rollout_steps=8, epochs=1, hidden_dim=8, seed=44),
        )
        trainer.train()
        models.append(
            {key: value.clone() for key, value in trainer.model.state_dict().items()}
        )
    assert all(torch.equal(models[0][key], models[1][key]) for key in models[0])


@pytest.mark.parametrize("minibatch_size", [1, 3, 32])
def test_bc_diagnostics_match_full_batch_with_unequal_groups(minibatch_size):
    # Eleven examples leave a short last batch for minibatch_size=3. Three
    # imbalanced groups make an unweighted mean of per-batch or group means
    # observably different from the intended sample-level metrics.
    class DiagnosticEnv:
        labels = [0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 2]

        def reset(self, seed=None):
            self.index = 0
            self.recorded = []
            return self.observation()

        def observation(self):
            return {
                "features": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
                "global": np.array([self.index / 11], dtype=np.float32),
                "mask": np.array([True, True, True]),
            }

        def heuristic_action(self):
            return self.labels[self.index]

        def step(self, action):
            assert action == self.heuristic_action()
            self.recorded.append(self.observation())
            self.index += 1
            return self.observation(), 0.0, False, False, {"fixture": True}

    env = DiagnosticEnv()
    trainer = PPOTrainer(env, PPOConfig(hidden_dim=8, seed=3))
    batches = []
    handle = trainer.model.register_forward_pre_hook(
        lambda _model, args: batches.append(len(args[0]["mask"]))
    )
    metrics = trainer.warmstart(11, epochs=1, minibatch_size=minibatch_size)
    handle.remove()
    assert max(batches) <= minibatch_size

    # Build the independently defined full-batch reference only for this tiny
    # fixture. warmstart resets the environment afterward, so reconstruct its
    # observations explicitly instead of using an implementation accumulator.
    observations = []
    for index in range(11):
        env.index = index
        observations.append(env.observation())
    batch = observation_tensors({
        key: np.stack([obs[key] for obs in observations])
        for key in observations[0]
    })
    labels = torch.tensor(env.labels)
    with torch.no_grad():
        distribution, _ = trainer.model(batch)
    correct = (distribution.logits.argmax(-1) == labels).float()
    group_accuracy = {
        str(group): float(correct[labels == group].mean()) for group in (0, 1, 2)
    }
    assert 0 < float(correct.mean()) < 1  # Check mixed outcomes, not an all-correct fixture.
    assert metrics["training_accuracy"] == pytest.approx(float(correct.mean()))
    assert metrics["mean_teacher_probability"] == pytest.approx(
        float(distribution.probs[torch.arange(11), labels].mean()), abs=1e-7
    )
    assert metrics["training_accuracy_by_dispatch"] == pytest.approx(group_accuracy)
    # All inverse-frequency weights lie below the cap; balancing therefore
    # gives each group one third of the weight, regardless of its sample count.
    assert metrics["balanced_training_accuracy"] == pytest.approx(
        sum(group_accuracy.values()) / 3
    )
    assert metrics["decision_group_counts"] == {"0": 2, "1": 3, "2": 6}


def test_resume_rejects_different_environment(tmp_path):
    trainer = PPOTrainer(
        TinyEnv(), PPOConfig(hidden_dim=8), env_config={"belt_speed": 0.1}
    )
    path = tmp_path / "policy.pt"
    trainer.save(path)
    other = PPOTrainer(
        TinyEnv(), PPOConfig(hidden_dim=8), env_config={"belt_speed": 0.2}
    )
    with pytest.raises(ValueError, match="environment config differs"):
        other.resume(path)


def test_v1_checkpoints_are_rejected_for_inference_and_resume(tmp_path):
    trainer = PPOTrainer(TinyEnv(), PPOConfig(hidden_dim=8))
    path = tmp_path / "obsolete.pt"
    trainer.save(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["version"] = 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="expected 2"):
        load_policy(path)
    with pytest.raises(ValueError, match="expected 2"):
        trainer.resume(path)


def test_forced_wait_does_not_change_policy_advantage_normalization():
    # A large value error while the arms execute an assigned pick must not
    # change the relative policy advantages at the two genuine choices.
    advantages = torch.tensor([1.0, 1000.0, 3.0, -1000.0])
    decisions = torch.tensor([True, False, True, False])
    normalized = decision_advantages(advantages, decisions)
    torch.testing.assert_close(normalized, torch.tensor([-1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(
        normalized[decisions],
        decision_advantages(advantages[decisions], torch.ones(2, dtype=torch.bool)),
    )


class ForcedWaitEnv(TinyEnv):
    def reset(self, seed=None):
        obs = super().reset(seed)
        obs["mask"][:] = [False, False, True]
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs["mask"][:] = [False, False, True]
        return obs, reward, terminated, truncated, info


@pytest.mark.parametrize("actor_context", ["independent", "set"])
def test_forced_wait_updates_critic_without_drifting_initialized_actor(actor_context):
    trainer = PPOTrainer(
        TinyEnv(), PPOConfig(steps=16, rollout_steps=8, epochs=2, hidden_dim=8,
                             actor_context=actor_context)
    )
    trainer.warmstart(16, epochs=4)  # Establish nonzero Adam momentum in the actor.
    actor_before = {
        key: value.clone()
        for key, value in trainer.model.state_dict().items()
        if key.startswith(("encoder.", "actor."))
    }
    critic_before = {
        key: value.clone()
        for key, value in trainer.model.state_dict().items()
        if key.startswith(("critic_encoder.", "critic."))
    }
    trainer.env = ForcedWaitEnv()
    trainer.observation = trainer.env.reset()
    history = trainer.train()
    current = trainer.model.state_dict()
    assert all(torch.equal(value, current[key]) for key, value in actor_before.items())
    assert any(
        not torch.equal(value, current[key]) for key, value in critic_before.items()
    )
    assert trainer.ppo_decision_steps == 0
    assert all(record["actor_optimizer_steps"] == 0 for record in history)
    assert all(
        record["decision_steps"] == 0 and record["forced_action_steps"] == 8
        for record in history
    )
    assert trainer.ppo_optimizer_steps > 0


def test_keep_action_logging_uses_scheduler_semantics_without_changing_rates():
    class FutureMetadataEnv(TinyEnv):
        wait_action = 2

        def action_metadata(self):
            return {
                "schema": "fixture_future_plan",
                "wait_semantics": "KEEP preserves plans and may dispatch a planned grasp",
            }

    records = []
    for env in (TinyEnv(), FutureMetadataEnv()):
        trainer = PPOTrainer(
            env, PPOConfig(steps=4, rollout_steps=4, epochs=1, hidden_dim=8, seed=7)
        )
        records.append(trainer.train()[0])
    baseline, future = records
    assert future["wait_action_schema"] == "fixture_future_plan"
    assert future["wait_action_semantics"] == FutureMetadataEnv().action_metadata()["wait_semantics"]
    assert "may dispatch" in future["active_wait_definition"]
    assert baseline["wait_action_schema"] is None
    assert "Do not dispatch" in baseline["active_wait_definition"]
    for key in ("active_wait_rate", "greedy_active_wait_rate", "teacher_active_wait_rate",
                "decision_steps", "rollout_reward"):
        assert future[key] == baseline[key]


def test_new_scene_observation_shape_and_checkpoint_roundtrip(tmp_path):
    seed_everything(5)
    obs = {
        "features": np.zeros((121, 22), dtype=np.float32),
        "global": np.zeros(20, dtype=np.float32),
        "mask": np.zeros(121, dtype=bool),
    }
    obs["mask"][[10, 120]] = True
    model = ActorCritic(22, 20, 16)
    distribution, value = model(observation_tensors(obs))
    assert distribution.probs.shape == (1, 121)
    assert value.shape == (1,)
    path = tmp_path / "v2.pt"
    torch.save(
        {"version": 2, "architecture": model.architecture, "model": model.state_dict()},
        path,
    )
    assert load_policy(path)(obs) in (10, 120)
