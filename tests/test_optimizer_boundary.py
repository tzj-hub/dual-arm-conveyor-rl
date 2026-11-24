"""Optimizer-boundary checks on a finite fixture, not physical performance tests."""

import math
from copy import deepcopy

import numpy as np
import pytest
import torch
from dual_arm_conveyor.learning import PPOConfig, PPOTrainer


class OptimizerFixture:
    def observation(self):
        return {
            "features": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
            "global": np.array([0.5], dtype=np.float32),
            "mask": np.array([True, False, True]),
        }

    def reset(self, seed=None):
        self.steps = 0
        return self.observation()

    def heuristic_action(self):
        return 2

    def step(self, action):
        assert action in (0, 2)
        self.steps += 1
        return self.observation(), float(action == 2), self.steps == 4, False, {"fixture": True}


def make_trainer(actor_context="independent"):
    return PPOTrainer(
        OptimizerFixture(),
        PPOConfig(steps=16, rollout_steps=8, epochs=2, minibatch_size=4, hidden_dim=8,
                  seed=42, actor_context=actor_context),
    )


def test_optimizer_reset_preserves_weights_counts_history_and_rng():
    trainer = make_trainer()
    trainer.warmstart(8, epochs=2)
    trainer.train()
    assert trainer.optimizer.state
    weights = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    fields = ("steps", "episodes", "reset_count", "ppo_decision_steps", "ppo_optimizer_steps",
              "ppo_actor_optimizer_steps", "ppo_episodes", "episode_reward")
    counts = {name: getattr(trainer, name) for name in fields}
    history, warmstart = deepcopy(trainer.history), deepcopy(trainer.warmstart_metrics)
    rng_state = deepcopy(trainer.rng.bit_generator.state)
    torch_state, numpy_state = torch.get_rng_state().clone(), np.random.get_state()
    observation = trainer.observation
    trainer.config.learning_rate = 7e-5

    trainer.reset_optimizer()

    assert not trainer.optimizer.state
    assert trainer.optimizer.param_groups[0]["lr"] == 7e-5
    assert trainer.optimizer.param_groups[0]["eps"] == 1e-5
    assert {name: getattr(trainer, name) for name in fields} == counts
    assert trainer.history == history and trainer.warmstart_metrics == warmstart
    assert trainer.observation is observation
    assert trainer.rng.bit_generator.state == rng_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0] and current_numpy[2:] == numpy_state[2:]
    np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
    assert all(torch.equal(value, trainer.model.state_dict()[name]) for name, value in weights.items())


def test_normal_resume_preserves_adam_moments(tmp_path):
    trainer = make_trainer()
    trainer.train()
    expected = deepcopy(trainer.optimizer.state_dict())
    path = tmp_path / "policy.pt"
    trainer.save(path)
    restored = make_trainer()
    restored.resume(path)
    actual = restored.optimizer.state_dict()
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for parameter_id, fields in expected["state"].items():
        for name, value in fields.items():
            assert torch.equal(actual["state"][parameter_id][name], value)
    assert restored.steps == trainer.steps
    assert restored.ppo_actor_optimizer_steps == trainer.ppo_actor_optimizer_steps
    assert restored.reset_count == trainer.reset_count + 1


@pytest.mark.parametrize("actor_context", ["independent", "set"])
def test_actor_clipping_is_independent_of_critic_gradient_scale(actor_context):
    trainer = make_trainer(actor_context)
    actor = [*trainer.model.encoder.parameters(), *trainer.model.actor.parameters()]
    critic = [*trainer.model.critic_encoder.parameters(), *trainer.model.critic.parameters()]
    outcomes, norms = [], []
    for critic_scale in (1.0, 1e6):
        for parameter in actor:
            parameter.grad = torch.ones_like(parameter)
        for parameter in critic:
            parameter.grad = torch.full_like(parameter, critic_scale)
        norms.append(trainer._clip_gradients())
        outcomes.append([parameter.grad.clone() for parameter in actor])
        for group in (actor, critic):
            norm = torch.linalg.vector_norm(torch.cat([parameter.grad.flatten() for parameter in group]))
            assert float(norm) <= trainer.config.max_grad_norm + 1e-6
    assert norms[0][0] == pytest.approx(norms[1][0])
    assert norms[1][1] == pytest.approx(norms[0][1] * 1e6, rel=1e-6)
    for normal, huge_critic in zip(outcomes[0], outcomes[1], strict=True):
        torch.testing.assert_close(normal, huge_critic, rtol=0, atol=0)


def test_clipping_without_actor_gradients_keeps_actor_disconnected():
    trainer = make_trainer()
    actor = [*trainer.model.encoder.parameters(), *trainer.model.actor.parameters()]
    critic = [*trainer.model.critic_encoder.parameters(), *trainer.model.critic.parameters()]
    for parameter in critic:
        parameter.grad = torch.ones_like(parameter)
    actor_norm, critic_norm = trainer._clip_gradients()
    assert actor_norm == 0 and critic_norm > 0
    assert all(parameter.grad is None for parameter in actor)


def test_ppo_history_records_finite_separate_gradient_norms():
    trainer = make_trainer()
    history = trainer.train()
    assert trainer.ppo_actor_optimizer_steps > 0
    assert all(row["actor_optimizer_steps"] > 0 for row in history)
    for row in history:
        for name in ("actor_grad_norm", "critic_grad_norm"):
            assert math.isfinite(row[name]) and row[name] > 0
