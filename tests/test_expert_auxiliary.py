"""Expert guidance tests use finite optimizer fixtures, not physical rollouts."""

import math

import numpy as np
import pytest
import torch
from dual_arm_conveyor import learning
from torch.distributions import Categorical


class GuidedFixture:
    wait_action = 2

    def __init__(self, teacher=0, reward=2.0):
        self.teacher = teacher
        self.reward = reward
        self.actions = []
        self.teacher_calls = 0

    def observation(self):
        return {
            "features": np.array([[1, 0], [0, 0], [0, 1]], dtype=np.float32),
            "global": np.array([self.elapsed / 4], dtype=np.float32),
            "mask": np.array([True, False, True]),
        }

    def reset(self, seed=None):
        self.elapsed = 0
        return self.observation()

    def teacher_action(self):
        self.teacher_calls += 1
        return self.teacher

    def heuristic_action(self):
        return 2  # Deliberately differs, so teacher selection is observable.

    def step(self, action):
        assert action in (0, 2)
        self.actions.append(action)
        self.elapsed += 1
        return self.observation(), self.reward, self.elapsed == 4, False, {"fixture": True}


class ForcedWaitFixture(GuidedFixture):
    def observation(self):
        result = super().observation()
        result["mask"][:] = [False, False, True]
        return result

    def teacher_action(self):
        raise AssertionError("A forced WAIT must not request an expert label")


def trainer_for(env, **overrides):
    arguments = dict(
        steps=8, rollout_steps=8, epochs=2, minibatch_size=4,
        hidden_dim=8, seed=19, expert_coef=0.5, expert_final_coef=0.2,
        expert_decay_steps=64,
    )
    arguments.update(overrides)
    return learning.PPOTrainer(env, learning.PPOConfig(**arguments))


def test_smoothed_expert_loss_uses_only_valid_actions_and_backpropagates():
    mask = torch.tensor([[True, False, True], [False, True, True]])
    labels = torch.tensor([0, 2])
    logits = torch.tensor([[2.0, 999.0, -1.0], [999.0, 1.0, 3.0]], requires_grad=True)
    distribution = Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))
    losses = learning.masked_expert_loss(distribution, labels, mask, label_smoothing=0.2)
    expected = torch.stack([
        -0.8 * distribution.logits[row, labels[row]]
        - 0.2 * distribution.logits[row, mask[row]].mean()
        for row in range(2)
    ])
    torch.testing.assert_close(losses, expected)
    losses.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    assert torch.count_nonzero(logits.grad[mask]) == 4


@pytest.mark.parametrize("teacher", [1, -1, 3])
def test_invalid_teacher_is_rejected_before_physical_step(teacher):
    env = GuidedFixture(teacher=teacher)
    trainer = trainer_for(env)
    with pytest.raises(ValueError):
        trainer.train()
    assert env.actions == []
    assert trainer.ppo_optimizer_steps == 0


def test_ppo_executes_network_samples_and_uses_the_observed_rewards(monkeypatch):
    env = GuidedFixture()
    trainer = trainer_for(env, reward_scale=0.25)
    recorded_rewards = []
    actual_gae = learning.generalized_advantages

    def observe_gae(rewards, *args, **kwargs):
        recorded_rewards.append(np.asarray(rewards).copy())
        return actual_gae(rewards, *args, **kwargs)

    def sample_wait(distribution, sample_shape=()):
        assert not sample_shape
        return torch.full(distribution.batch_shape, env.wait_action, dtype=torch.long,
                          device=distribution.logits.device)

    monkeypatch.setattr(learning, "generalized_advantages", observe_gae)
    monkeypatch.setattr(Categorical, "sample", sample_wait)
    before = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    history = trainer.train()

    assert env.actions == [env.wait_action] * 8
    assert env.teacher_calls == 8 and trainer.ppo_decision_steps == 8
    assert trainer.ppo_actor_optimizer_steps > 0
    np.testing.assert_allclose(recorded_rewards[0], np.full(8, 0.5))
    record = history[0]
    assert record["rollout_reward"] == pytest.approx(16.0)
    assert record["scaled_rollout_reward"] == pytest.approx(4.0)
    assert [episode["reward"] for episode in record["completed_episodes"]] == [8.0, 8.0]
    assert record["sampled_teacher_agreement"] == 0
    assert 0 <= record["teacher_agreement"] <= 1
    assert record["active_wait_rate"] == 1
    assert record["teacher_active_wait_rate"] == 0
    assert 0 <= record["greedy_active_wait_rate"] <= 1
    assert record["imitation_loss"] > 0 and record["expert_coef"] > 0
    assert math.isfinite(record["explained_variance"])
    assert any(not torch.equal(value, trainer.model.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize("actor_context", ["independent", "set"])
@pytest.mark.parametrize("expert_balance", [False, True])
def test_forced_wait_never_trains_actor_even_with_guidance_and_adam_momentum(actor_context, expert_balance):
    trainer = trainer_for(GuidedFixture(), actor_context=actor_context, expert_balance=expert_balance)
    trainer.warmstart(8, epochs=2)
    before = {
        name: value.clone() for name, value in trainer.model.state_dict().items()
        if name.startswith(("encoder.", "actor."))
    }
    trainer.env = ForcedWaitFixture()
    trainer.observation = trainer.env.reset()
    history = trainer.train()
    assert trainer.ppo_decision_steps == 0 and trainer.ppo_actor_optimizer_steps == 0
    assert trainer.ppo_optimizer_steps > 0
    for name, value in before.items():
        torch.testing.assert_close(value, trainer.model.state_dict()[name], rtol=0, atol=0)
    assert history[0]["imitation_loss"] == 0


def test_reward_changes_actor_updates_with_identical_expert_labels(monkeypatch):
    def sample_wait(distribution, sample_shape=()):
        assert not sample_shape
        return torch.full(distribution.batch_shape, 2, dtype=torch.long,
                          device=distribution.logits.device)

    monkeypatch.setattr(Categorical, "sample", sample_wait)
    initial_actors, final_actors, environments = [], [], []
    for reward in (2.0, -2.0):
        env = GuidedFixture(reward=reward)
        trainer = trainer_for(env, expert_coef=0.05, expert_final_coef=0.05)
        initial_actors.append({
            name: value.clone() for name, value in trainer.model.state_dict().items()
            if name.startswith(("encoder.", "actor."))
        })
        trainer.train()
        final_actors.append({
            name: value.clone() for name, value in trainer.model.state_dict().items()
            if name.startswith(("encoder.", "actor."))
        })
        environments.append(env)
    assert environments[0].actions == environments[1].actions == [2] * 8
    assert environments[0].teacher_calls == environments[1].teacher_calls == 8
    for name in initial_actors[0]:
        torch.testing.assert_close(initial_actors[0][name], initial_actors[1][name], rtol=0, atol=0)
    maximum_actor_difference = max(
        float((final_actors[0][name] - final_actors[1][name]).abs().max())
        for name in final_actors[0]
    )
    assert maximum_actor_difference > 1e-5


def test_expert_training_counters_survive_checkpoint_resume(tmp_path):
    trainer = trainer_for(GuidedFixture(), reward_scale=0.25)
    trainer.train()
    path = tmp_path / "guided.pt"
    trainer.save(path)
    restored = trainer_for(GuidedFixture(), reward_scale=0.25, steps=16)
    restored.resume(path)
    for name in ("expert_label_steps", "expert_optimizer_steps", "ppo_actor_optimizer_steps",
                 "ppo_optimizer_steps", "ppo_decision_steps", "steps"):
        assert getattr(restored, name) == getattr(trainer, name) > 0
    policy = learning.load_policy(path)
    assert policy.metadata["expert_label_steps"] == trainer.expert_label_steps
    assert policy.metadata["expert_optimizer_steps"] == trainer.expert_optimizer_steps
    restored.train()
    assert restored.expert_label_steps == 16
    assert restored.expert_optimizer_steps > trainer.expert_optimizer_steps


def test_resume_rejects_changed_critic_reward_units_before_loading_weights(tmp_path):
    trainer = trainer_for(GuidedFixture(), reward_scale=0.25)
    trainer.train()
    path = tmp_path / "guided.pt"
    trainer.save(path)
    restored = trainer_for(GuidedFixture(), reward_scale=1.0)
    initial = {name: value.clone() for name, value in restored.model.state_dict().items()}
    with pytest.raises(ValueError, match="Reward scale"):
        restored.resume(path)
    for name, value in initial.items():
        torch.testing.assert_close(value, restored.model.state_dict()[name], rtol=0, atol=0)


def test_warmstart_uses_teacher_instead_of_fallback_heuristic():
    env = GuidedFixture()
    trainer = trainer_for(env)
    result = trainer.warmstart(8, epochs=2, learning_rate=5e-4, minibatch_size=2,
                              label_smoothing=0.1)
    assert env.actions == [0] * 8 and env.teacher_calls == 8
    assert result["optimizer_steps"] == 8
    assert result["decision_examples"] == 8
    assert result["learning_rate"] == 5e-4 and result["minibatch_size"] == 2
    assert result["label_smoothing"] == 0.1
    assert trainer.optimizer.param_groups[0]["lr"] == trainer.config.learning_rate
    assert trainer.steps == 0 and trainer.ppo_optimizer_steps == 0


def test_warmstart_caps_rare_group_weight_before_normalizing():
    class RareTeacherFixture(GuidedFixture):
        def teacher_action(self):
            self.teacher_calls += 1
            return 2 if self.teacher_calls == 60 else 0

    trainer = trainer_for(RareTeacherFixture())
    result = trainer.warmstart(60, epochs=1, minibatch_size=16)
    assert result["inverse_weight_cap_before_normalization"] == 4.0
    assert result["sample_weight_max"] > 4.0  # Re-normalization may increase the cap.
    fractions = result["decision_group_weight_fractions"]
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert min(fractions.values()) == pytest.approx(4 / 34)


def test_teacher_falls_back_to_heuristic_when_not_provided():
    class HeuristicFixture:
        wait_action = 2

        def __init__(self):
            self.delegate = GuidedFixture()

        def reset(self, seed=None):
            return self.delegate.reset(seed)

        def heuristic_action(self):
            return 0

        def step(self, action):
            return self.delegate.step(action)

    env = HeuristicFixture()
    trainer = trainer_for(env)
    trainer.warmstart(4, epochs=1)
    assert env.delegate.actions == [0] * 4
