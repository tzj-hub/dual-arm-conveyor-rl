"""Set-conditioned scheduling scores; synthetic tests, not sorting results."""
import numpy as np
import pytest
import torch

from dual_arm_conveyor.learning import (
    ActorCritic, PPOConfig, PPOTrainer, load_policy, observation_tensors,
)


def observation():
    return {
        "features": np.array([[0.2, 0.5], [0.8, -0.3], [-0.6, 0.9], [0, 0]],
                             dtype=np.float32),
        "global": np.array([0.4], dtype=np.float32),
        "mask": np.array([True, True, True, False]),
    }


def model(context="set"):
    torch.manual_seed(42)
    return ActorCritic(2, 1, hidden_dim=16, actor_context=context)


def test_scores_are_equivariant_to_candidate_order():
    net = model()
    obs = observation_tensors(observation())
    order = torch.tensor([2, 0, 3, 1])
    permuted = dict(obs, features=obs["features"][:, order], mask=obs["mask"][:, order])
    before, value_before = net(obs)
    after, value_after = net(permuted)
    torch.testing.assert_close(after.probs, before.probs[:, order])
    torch.testing.assert_close(value_after, value_before)


def test_invalid_candidates_cannot_influence_valid_scores_or_value():
    net = model()
    obs = observation_tensors(observation())
    changed = {key: value.clone() for key, value in obs.items()}
    changed["features"][0, 3] = torch.tensor([1e8, -1e8])
    before, value_before = net(obs)
    after, value_after = net(changed)
    torch.testing.assert_close(after.probs, before.probs, rtol=0, atol=0)
    torch.testing.assert_close(value_after, value_before, rtol=0, atol=0)
    assert after.probs[0, 3] == 0
    assert not (after.sample((1000,)) == 3).any()


@pytest.mark.parametrize("context", ["independent", "set"])
def test_only_context_actor_can_change_relative_preference_from_competitor(context):
    net = model(context)
    obs = observation_tensors(observation())
    obs["features"].requires_grad_()
    distribution, _ = net(obs)
    # Use log odds, so the common categorical normalizer cancels out.
    log_odds = distribution.logits[0, 0] - distribution.logits[0, 1]
    derivative, = torch.autograd.grad(log_odds, obs["features"])
    competitor_effect = derivative[0, 2].abs().max().item()
    assert competitor_effect > 1e-8 if context == "set" else competitor_effect == 0
    assert derivative[0, 3].abs().max().item() == 0


def test_single_legal_action_is_finite_and_forced():
    net = model()
    obs = observation_tensors(observation())
    obs["mask"][:] = False
    obs["mask"][0, 1] = True
    distribution, value = net(obs)
    assert distribution.probs.tolist() == [[0, 1, 0, 0]]
    assert torch.isfinite(value).all()
    assert torch.isfinite(distribution.entropy()).all()


class ChoiceFixture:
    def reset(self, seed=None):
        self.steps = 0
        return observation()

    def heuristic_action(self):
        return 1

    def step(self, action):
        assert observation()["mask"][action]
        self.steps += 1
        return observation(), float(action == 1), self.steps == 4, False, {}


@pytest.mark.parametrize("context", ["independent", "set"])
def test_learning_checkpoint_and_optimizer_resume_preserve_architecture(tmp_path, context):
    config = PPOConfig(steps=16, rollout_steps=8, epochs=1, minibatch_size=4,
                       hidden_dim=16, seed=7, actor_context=context)
    trainer = PPOTrainer(ChoiceFixture(), config)
    before = {k: v.clone() for k, v in trainer.model.actor.state_dict().items()}
    trainer.warmstart(8, epochs=2)
    trainer.train()
    assert trainer.ppo_actor_optimizer_steps > 0
    assert any(not torch.equal(before[k], v) for k, v in trainer.model.actor.state_dict().items())
    path = tmp_path / "policy.pt"
    trainer.save(path)
    policy = load_policy(path)
    assert policy.model.actor_context == context
    if context == "independent":
        assert "actor_context" not in policy.model.architecture
    else:
        assert policy.model.architecture["actor_context"] == "set"
    expected, _ = trainer.model(observation_tensors(observation()))
    actual, _ = policy.model(observation_tensors(observation()))
    torch.testing.assert_close(actual.probs, expected.probs, rtol=0, atol=0)
    restored = PPOTrainer(ChoiceFixture(), config)
    restored.resume(path)
    assert restored.ppo_actor_optimizer_steps == trainer.ppo_actor_optimizer_steps
    assert restored.optimizer.state_dict()["param_groups"] == trainer.optimizer.state_dict()["param_groups"]
    other = "set" if context == "independent" else "independent"
    with pytest.raises(ValueError, match="Checkpoint"):
        PPOTrainer(ChoiceFixture(), PPOConfig(hidden_dim=16, actor_context=other)).resume(path)


def test_unknown_context_rejected():
    with pytest.raises(ValueError, match="actor_context"):
        ActorCritic(2, 1, actor_context="attention_typo")
    with pytest.raises(ValueError, match="actor_context"):
        PPOConfig(actor_context="attention_typo")
