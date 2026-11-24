"""Rollout supervision checks on arrays/tiny fixtures; no physical simulator."""

from dataclasses import asdict

import numpy as np
import pytest
import torch

from dual_arm_conveyor import learning


class GroupedFixture:
    wait_action = 2
    dispatch_group_names = {0: "keep", 1: "change"}

    def __init__(self):
        self.actions, self.group_inputs = [], []

    def reset(self, seed=None):
        self.index = 0
        return self.observation()

    def observation(self):
        return {
            "features": np.array([[1, 0], [0, 0], [0, 1]], dtype=np.float32),
            "global": np.array([self.index / 8], dtype=np.float32),
            "mask": np.array([self.index not in (2, 7), False, True]),
        }

    def teacher_action(self):
        assert self.index not in (2, 7)
        return 2 if self.index == 6 else 0

    def expert_action_groups(self, labels):
        self.group_inputs.append(labels.copy())
        return (labels == 2).astype(np.int64)

    def step(self, action):
        assert self.observation()["mask"][action]
        self.actions.append(action)
        self.index += 1
        return self.observation(), float(action == 0), self.index == 8, False, {"fixture": True}


def make_trainer(env=None, **overrides):
    config = dict(steps=8, rollout_steps=8, epochs=1, minibatch_size=2, hidden_dim=8,
                  seed=17, expert_coef=0.2, expert_final_coef=0.2, target_kl=0)
    config.update(overrides)
    return learning.PPOTrainer(env or GroupedFixture(), learning.PPOConfig(**config))


def test_rare_groups_receive_more_weight_with_global_mean_one():
    weights, groups, names = learning.action_group_weights(object(), [0] * 100 + [2], 3)
    assert weights.mean() == pytest.approx(1)
    assert weights[-1] > weights[0]
    assert weights[-1] > 4  # The cap is deliberately BEFORE mean normalization.
    expected = np.r_[np.full(100, 101 / 200), 4.0]
    expected /= expected.mean()
    np.testing.assert_array_equal(weights, expected)
    assert names == {0: "0", 2: "2"}
    np.testing.assert_array_equal(groups, [0] * 100 + [2])


def test_equal_groups_leave_supervision_unchanged_and_square_fallback_is_semantic():
    weights, groups, names = learning.action_group_weights(object(), [3, 1, 2, 0], 4)
    np.testing.assert_array_equal(weights, np.ones(4))
    np.testing.assert_array_equal(groups, [0, 1, 2, 3])
    assert names == {0: "wait", 1: "arm0", 2: "arm1", 3: "both"}


def test_only_known_nonforced_labels_reach_group_mapper():
    env = GroupedFixture()
    masks = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=bool)
    weights, info = learning.rollout_expert_weights(env, [0, 2, -1, 2], masks)
    np.testing.assert_array_equal(env.group_inputs[0], [0, 2])
    np.testing.assert_array_equal(weights, [1, 0, 0, 1])
    assert info["eligible_labels"] == 2
    assert info["group_counts"] == {"keep": 1, "change": 1}
    assert info["group_weight_fractions"] == {"keep": 0.5, "change": 0.5}
    env.group_inputs.clear()
    empty, summary = learning.rollout_expert_weights(env, [-1, 2], masks[:2])
    assert env.group_inputs == []
    np.testing.assert_array_equal(empty, [0, 0])
    assert summary["eligible_labels"] == 0 and summary["group_counts"] == {}


@pytest.mark.parametrize("label", [1, 3])
def test_invalid_known_teacher_is_not_hidden_by_balancing(label):
    with pytest.raises(ValueError):
        learning.rollout_expert_weights(object(), [label], [[True, False, True]])


@pytest.mark.parametrize("balanced", [False, True])
def test_online_loss_uses_rollout_weights_without_minibatch_renormalization(monkeypatch, balanced):
    actual_loss = learning.masked_expert_loss
    expected = []

    def capture(distribution, labels, mask, smoothing):
        loss = actual_loss(distribution, labels, mask, smoothing)
        # Six eligible rows: five keep labels and one change label. Weights
        # are 0.6 / 3 over the rollout, even for homogeneous minibatches.
        weights = torch.where(labels == 0, 0.6, 3.0) if balanced else torch.ones_like(loss)
        expected.append(float((loss.detach() * weights).mean()))
        return loss

    monkeypatch.setattr(learning, "masked_expert_loss", capture)
    trainer = make_trainer(expert_balance=balanced)
    row = trainer.train()[0]
    assert row["imitation_loss"] == pytest.approx(float(np.mean(expected)), abs=1e-7)
    assert row["expert_balance"] is balanced
    if balanced:
        info = row["expert_balance_statistics"]
        assert info["group_counts"] == {"keep": 5, "change": 1}
        assert info["group_weight_fractions"] == pytest.approx({"keep": 0.5, "change": 0.5})
        assert len(trainer.env.group_inputs) == 1
    else:
        assert row["expert_balance_statistics"] is None
        assert trainer.env.group_inputs == []


def test_balancing_changes_only_actor_supervision_for_the_same_collected_rollout():
    states, rows, actions = [], [], []
    for balanced in (False, True):
        trainer = make_trainer(expert_balance=balanced)
        rows.append(trainer.train()[0])
        states.append({k: v.clone() for k, v in trainer.model.state_dict().items()})
        actions.append(trainer.env.actions)
    assert actions[0] == actions[1]  # No teacher action substitution.
    assert rows[0]["rollout_reward"] == rows[1]["rollout_reward"]
    for key in states[0]:
        if key.startswith("critic"):
            torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)
    assert any(not torch.equal(states[0][k], states[1][k])
               for k in states[0] if k.startswith(("encoder.", "actor.")))


def test_disabled_default_never_calls_balance_mapper_and_matches_explicit_false(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Default PPO must not run group balancing")
    monkeypatch.setattr(learning, "rollout_expert_weights", forbidden)
    states = []
    for overrides in ({}, {"expert_balance": False}):
        trainer = make_trainer(**overrides)
        trainer.train()
        states.append(trainer.model.state_dict())
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)


def test_no_teacher_uses_no_balancing_labels_or_imitation_updates():
    class NoTeacher(GroupedFixture):
        teacher_action = None

        def expert_action_groups(self, labels):
            raise AssertionError("Unknown teacher labels must not reach the mapper")

    trainer = make_trainer(NoTeacher(), expert_balance=True, expert_coef=0, expert_final_coef=0)
    row = trainer.train()[0]
    assert row["expert_balance_statistics"]["eligible_labels"] == 0
    assert row["imitation_loss"] == 0 and row["expert_optimizer_steps"] == 0
    assert row["teacher_agreement"] is None


def test_legacy_checkpoint_without_balance_field_loads_and_resumes(tmp_path):
    trainer = make_trainer()
    trainer.train()
    path = tmp_path / "old.pt"
    trainer.save(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["train_config"].pop("expert_balance")
    assert learning.PPOConfig(**checkpoint["train_config"]).expert_balance is False
    torch.save(checkpoint, path)
    loaded = learning.load_policy(path)
    resumed = make_trainer(steps=16)
    resumed.resume(path)
    assert resumed.steps == 8
    for key, value in loaded.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], atol=0, rtol=0)
    assert asdict(resumed.config)["expert_balance"] is False
