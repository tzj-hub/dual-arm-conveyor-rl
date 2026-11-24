"""Bounded demonstration sampling on finite fixtures; no physical simulation."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from dual_arm_conveyor import hierarchical_train
from dual_arm_conveyor import learning


class SamplingFixture:
    def __init__(self, mixed=False):
        self.mixed = mixed
        self.actions = []
        self.global_arrays = []

    def observation(self):
        global_features = np.asarray([self.elapsed], dtype=np.float32)
        self.global_arrays.append(global_features)
        return {
            "features": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "global": global_features,
            "mask": np.asarray([not self.mixed or self.elapsed % 3 != 0, True]),
        }

    def reset(self, seed=None):
        self.elapsed = 0
        return self.observation()

    def teacher_action(self):
        return 1 if self.mixed and self.elapsed % 3 == 0 else self.elapsed % 2

    def step(self, action):
        self.actions.append(action)
        self.elapsed += 1
        return self.observation(), 0.0, False, False, {"fixture": True}


def make_trainer(env=None):
    return learning.PPOTrainer(env or SamplingFixture(), learning.PPOConfig(
        steps=0, hidden_dim=8, minibatch_size=4, seed=7,
    ))


def capture_retained(monkeypatch):
    captured = []
    original = learning.observation_tensors

    def capture(observation, device="cpu"):
        if observation["features"].ndim == 3:
            captured.append(np.asarray(observation["global"]).copy())
        return original(observation, device)

    monkeypatch.setattr(learning, "observation_tensors", capture)
    return captured


def test_cap_limits_only_retained_choices_and_progress_counts_seen(monkeypatch):
    retained = capture_retained(monkeypatch)
    env = SamplingFixture(mixed=True)
    trainer = make_trainer(env)
    progress = []
    result = trainer.warmstart(60, epochs=1, max_examples=5, on_progress=progress.append)
    assert len(env.actions) == result["physical_steps"] == 60
    assert result["decision_examples_seen"] == 40
    assert result["decision_examples"] == result["decision_examples_retained"] == 5
    assert result["max_examples"] == 5 and result["sampling"] == "uniform_reservoir"
    assert progress[-1]["examples_seen"] == 40
    assert progress[-1]["examples"] == progress[-1]["examples_retained"] == 5
    assert progress[-1]["max_examples"] == 5
    sample_ids = retained[0][:, 0].astype(int)
    assert len(set(sample_ids)) == 5
    assert all(sample_ids % 3 != 0)  # Forced WAIT states never enter the reservoir.
    assert max(sample_ids) > 30  # Sampling covers later states, not just a prefix.


def test_reservoir_is_reproducible_without_consuming_bc_shuffle_rng(monkeypatch):
    retained = capture_retained(monkeypatch)
    models = []
    for _ in range(2):
        trainer = make_trainer()
        expected_rng = np.random.default_rng(7)
        expected_rng.permutation(5)  # One BC epoch; reservoir has a separate RNG.
        global_rng_before = deepcopy(np.random.get_state())
        trainer.warmstart(60, epochs=1, max_examples=5)
        assert trainer.rng.bit_generator.state == expected_rng.bit_generator.state
        global_rng_after = np.random.get_state()
        assert global_rng_before[0] == global_rng_after[0]
        np.testing.assert_array_equal(global_rng_before[1], global_rng_after[1])
        assert global_rng_before[2:] == global_rng_after[2:]
        models.append({k: v.clone() for k, v in trainer.model.state_dict().items()})
    np.testing.assert_array_equal(retained[0], retained[1])
    assert all(torch.equal(models[0][k], models[1][k]) for k in models[0])


def test_unselected_states_are_not_copied(monkeypatch):
    env = SamplingFixture()
    trainer = make_trainer(env)
    copies = []
    original_array = np.array

    def track_array(value, *args, **kwargs):
        if any(value is observed for observed in env.global_arrays):
            copies.append(int(value[0]))
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr(learning.np, "array", track_array)
    result = trainer.warmstart(100, epochs=1, max_examples=2)
    assert result["decision_examples_seen"] == 100
    assert 2 <= len(copies) < 100
    assert len(env.actions) == 100


def test_default_none_and_nonbinding_cap_preserve_numeric_behavior(monkeypatch):
    retained = capture_retained(monkeypatch)
    results, models, actions = [], [], []
    for options in ({}, {"max_examples": None}, {"max_examples": 100}):
        trainer = make_trainer()
        results.append(trainer.warmstart(12, epochs=2, **options))
        models.append({k: v.clone() for k, v in trainer.model.state_dict().items()})
        actions.append(trainer.env.actions)
    assert actions[0] == actions[1] == actions[2]
    for i in (1, 2):
        np.testing.assert_array_equal(retained[0], retained[i])
        assert all(torch.equal(models[0][k], models[i][k]) for k in models[0])
        for key in ("mean_loss", "training_accuracy", "balanced_training_accuracy", "optimizer_steps"):
            assert results[0][key] == results[i][key]
    assert results[0] == results[1]
    assert results[0]["sampling"] == "all_choice_states"


def test_none_does_not_construct_a_sampling_rng(monkeypatch):
    trainer = make_trainer()

    def unexpected(*args, **kwargs):
        raise AssertionError("Default unlimited collection must not create a sampling RNG")

    monkeypatch.setattr(learning.np.random, "default_rng", unexpected)
    result = trainer.warmstart(4, epochs=1)
    assert result["decision_examples"] == 4


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "5"])
def test_invalid_cap_is_rejected_before_sampling(cap):
    trainer = make_trainer()
    with pytest.raises(ValueError, match="max_examples"):
        trainer.warmstart(8, epochs=1, max_examples=cap)
    assert not trainer.env.actions


def test_cli_reservoir_limit_is_optional_and_positive():
    parser = hierarchical_train.parser()
    assert parser.parse_args([]).warmstart_max_examples is None
    assert parser.parse_args(["--warmstart-max-examples", "8192"]).warmstart_max_examples == 8192
    with pytest.raises(SystemExit):
        parser.parse_args(["--warmstart-max-examples", "0"])
