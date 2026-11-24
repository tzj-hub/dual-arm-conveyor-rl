"""Masked PPO for the physical conveyor scheduler.

The network scores each joint assignment with a shared encoder.  Action masks
remain authoritative at inference; low-level grasp control lives in the env.
This module deliberately has no dependency on MuJoCo, making checkpoint and
optimizer tests possible without starting a simulation.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

Observation = Mapping[str, np.ndarray]
# Schema 2 uses separate actor/critic encoders and the revised scene inputs.
# Reject v1 checkpoints even if somebody happens to pad their observations.
CHECKPOINT_VERSION = 2


def seed_everything(seed: int) -> None:
    """Seed the CPU scheduler and avoid thread-pool overhead for tiny MLPs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def observation_tensors(
    observation: Observation, device: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    result = {
        "features": torch.as_tensor(
            observation["features"], dtype=torch.float32, device=device
        ),
        "global": torch.as_tensor(
            observation["global"], dtype=torch.float32, device=device
        ),
        "mask": torch.as_tensor(observation["mask"], dtype=torch.bool, device=device),
    }
    if result["features"].ndim == 2:
        result = {key: value.unsqueeze(0) for key, value in result.items()}
    if result["features"].ndim != 3 or result["global"].ndim != 2:
        raise ValueError(
            "Expected features [actions, features] and global [features], optionally batched"
        )
    if result["mask"].shape != result["features"].shape[:2]:
        raise ValueError("Action mask shape does not match candidate features")
    if result["global"].shape[0] != result["features"].shape[0]:
        raise ValueError("Global and candidate batch sizes differ")
    if not bool(result["mask"].any(dim=-1).all()):
        raise ValueError("Every observation must have at least one valid action")
    if not bool(
        torch.isfinite(result["features"]).all()
        and torch.isfinite(result["global"]).all()
    ):
        raise ValueError("Observation contains nonfinite features")
    return result


class ActorCritic(nn.Module):
    def __init__(self, feature_dim: int, global_dim: int, hidden_dim: int = 64,
                 actor_context: str = "independent"):
        super().__init__()
        if actor_context not in ("independent", "set"):
            raise ValueError("actor_context must be independent or set")
        self.actor_context = actor_context
        self.architecture = {
            "feature_dim": int(feature_dim),
            "global_dim": int(global_dim),
            "hidden_dim": int(hidden_dim),
        }
        # Keep the original architecture dictionary and state keys unchanged
        # for old checkpoints. Context policies must declare their new graph.
        if actor_context != "independent":
            self.architecture["actor_context"] = actor_context
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + global_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = (
            nn.Linear(hidden_dim, 1) if actor_context == "independent" else
            nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        )
        self.critic_encoder = nn.Sequential(
            nn.Linear(feature_dim + global_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        actor_output = self.actor if actor_context == "independent" else self.actor[-1]
        nn.init.orthogonal_(actor_output.weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1)

    def forward(
        self, observations: Mapping[str, torch.Tensor]
    ) -> tuple[Categorical, torch.Tensor]:
        features, global_features, mask = (
            observations["features"],
            observations["global"],
            observations["mask"],
        )
        expanded_global = global_features[:, None, :].expand(-1, features.shape[1], -1)
        inputs = torch.cat((features, expanded_global), dim=-1)
        encoded = self.encoder(inputs)
        if self.actor_context == "set":
            # The relative score of two jobs can now depend on competing
            # jobs, rather than only their own features and queue aggregates.
            # Pool valid alternatives only. The operation is permutation
            # invariant, with O(actions * hidden_dim) memory and compute.
            weights = mask.unsqueeze(-1).to(encoded.dtype)
            mean = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1)
            maximum = encoded.masked_fill(
                ~mask.unsqueeze(-1), torch.finfo(encoded.dtype).min
            ).amax(1)
            context = torch.cat((mean, maximum), dim=-1)
            actor_input = torch.cat((encoded, context[:, None, :].expand(
                -1, encoded.shape[1], -1)), dim=-1)
        else:
            actor_input = encoded
        logits = self.actor(actor_input).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        # Long robot execution phases have only WAIT available. Their value
        # targets still matter, but must not overwrite the actor's encoder.
        value_encoded = self.critic_encoder(inputs)
        weights = mask.unsqueeze(-1).to(value_encoded.dtype)
        pooled = (value_encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        values = self.critic(torch.cat((pooled, global_features), dim=-1)).squeeze(-1)
        return Categorical(logits=logits), values


class Policy:
    """Callable inference wrapper accepted by evaluation and playback tools."""

    def __init__(
        self, model: ActorCritic, device: str | torch.device = "cpu", metadata=None
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.metadata = metadata or {}

    @torch.no_grad()
    def __call__(self, observation: Observation, deterministic: bool = True) -> int:
        tensors = observation_tensors(observation, self.device)
        if tensors["features"].shape[0] != 1:
            raise ValueError("Policy.__call__ accepts one observation")
        distribution, _ = self.model(tensors)
        action = (
            distribution.logits.argmax(-1) if deterministic else distribution.sample()
        )
        return int(action.item())


def load_policy(path: str | Path, device: str | torch.device = "cpu") -> Policy:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported policy checkpoint version; expected {CHECKPOINT_VERSION}"
        )
    model = ActorCritic(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model"])
    metadata = {
        key: checkpoint[key]
        for key in (
            "env_config", "steps", "train_config", "ppo_decision_steps",
            "ppo_optimizer_steps", "ppo_episodes", "scheduler_action_metadata",
            "scheduler_observation_metadata", "ppo_actor_optimizer_steps",
            "expert_label_steps", "expert_optimizer_steps",
        )
        if key in checkpoint
    }
    return Policy(model, device, metadata)


@dataclass
class PPOConfig:
    steps: int = 4096
    rollout_steps: int = 256
    epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.98
    clip_ratio: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    hidden_dim: int = 64
    seed: int = 0
    device: str = "cpu"
    expert_coef: float = 0.0
    expert_final_coef: float = 0.0
    expert_decay_steps: int = 8192
    expert_label_smoothing: float = 0.05
    expert_balance: bool = False
    reward_scale: float = 1.0
    actor_context: str = "independent"

    def __post_init__(self):
        if not isinstance(self.expert_balance, bool):
            raise ValueError("expert_balance must be boolean")
        if self.actor_context not in ("independent", "set"):
            raise ValueError("actor_context must be independent or set")
        for name in ("rollout_steps", "epochs", "minibatch_size", "hidden_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.steps < 0:
            raise ValueError("steps must be nonnegative")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in their probability ranges")
        if (not math.isfinite(self.expert_coef) or not math.isfinite(self.expert_final_coef)
                or self.expert_coef < 0 or self.expert_final_coef < 0 or self.expert_decay_steps < 0):
            raise ValueError("Expert coefficients and decay steps must be nonnegative")
        if not 0 <= self.expert_label_smoothing <= 1:
            raise ValueError("expert_label_smoothing must be in [0, 1]")
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")


def _reset(env, seed: int) -> Observation:
    result = env.reset(seed=seed)
    return result[0] if isinstance(result, tuple) else result


def generalized_advantages(
    rewards, values, next_values, terminated, ended, gamma, gae_lambda
):
    """Bootstrap time limits but never propagate GAE across an episode reset."""
    rewards, values, next_values = map(
        lambda x: np.asarray(x, dtype=np.float32), (rewards, values, next_values)
    )
    terminated, ended = (
        np.asarray(terminated, dtype=bool),
        np.asarray(ended, dtype=bool),
    )
    advantages = np.zeros_like(rewards)
    accumulator = 0.0
    for index in reversed(range(len(rewards))):
        delta = (
            rewards[index]
            + gamma * next_values[index] * (not terminated[index])
            - values[index]
        )
        accumulator = delta + gamma * gae_lambda * (not ended[index]) * accumulator
        advantages[index] = accumulator
    return advantages, advantages + values


def decision_advantages(
    advantages: torch.Tensor, decisions: torch.Tensor
) -> torch.Tensor:
    """Normalize only genuine choices; forced actions have zero policy weight.

    All transitions remain in GAE and the value loss. Removing physical WAIT
    transitions there would incorrectly shorten time and reward delays.
    """
    result = torch.zeros_like(advantages)
    selected = advantages[decisions]
    if selected.numel() > 1:
        selected = (selected - selected.mean()) / selected.std(
            unbiased=False
        ).clamp_min(1e-8)
    result[decisions] = selected
    return result


def masked_expert_loss(distribution, labels, mask, label_smoothing=0.05):
    """Per-state CE, with smoothing mass only on physically legal actions."""
    if not 0 <= label_smoothing <= 1:
        raise ValueError("label_smoothing must be in [0, 1]")
    if mask.ndim != 2 or labels.shape != mask.shape[:1]:
        raise ValueError("Expected labels [batch] and mask [batch, actions]")
    if bool(((labels < 0) | (labels >= mask.shape[1])).any()):
        raise ValueError("Expert label is outside the action space")
    if not bool(mask.gather(1, labels[:, None]).all()):
        raise ValueError("Expert label violates the physical action mask")
    # masked_fill avoids multiplying -inf (or the minimum float) by zero.
    valid_log_probs = distribution.logits.masked_fill(~mask, 0.0)
    uniform_nll = -valid_log_probs.sum(-1) / mask.sum(-1).clamp_min(1)
    return -(1 - label_smoothing) * distribution.log_prob(labels) + label_smoothing * uniform_nll


def action_group_weights(env, labels, action_count):
    """BC-compatible inverse group frequency, capped before mean normalization.

    The cap bounds the *unnormalized* inverse-frequency weights, not the final
    weights. Group balancing changes supervision only, never the action mask.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or np.any((labels < 0) | (labels >= action_count)):
        raise ValueError("Expected valid one-dimensional expert labels")
    if not len(labels):
        return np.empty(0, dtype=np.float64), labels.copy(), {}
    describe_groups = getattr(env, "expert_action_groups", None)
    side = math.isqrt(action_count)
    if callable(describe_groups):
        groups = np.asarray(describe_groups(labels), dtype=np.int64)
        if groups.shape != labels.shape or np.any(groups < 0):
            raise ValueError("Expert action groups must be nonnegative and match the labels")
        group_names = getattr(env, "dispatch_group_names", {})
        group_names = {int(g): group_names.get(int(g), str(g)) for g in np.unique(groups)}
    elif side * side == action_count:
        groups = (labels // side != side - 1).astype(int) + 2 * (
            labels % side != side - 1
        ).astype(int)
        group_names = getattr(env, "dispatch_group_names", {0: "wait", 1: "arm0", 2: "arm1", 3: "both"})
    else:
        groups = labels
        group_names = {int(g): str(g) for g in np.unique(groups)}
    counts = np.bincount(groups)
    weights = np.minimum(len(labels) / (np.count_nonzero(counts) * counts[groups]), 4.0)
    weights /= weights.mean()
    return weights, groups, group_names


def _group_weight_summary(weights, groups, group_names):
    return {
        "group_counts": {group_names[int(g)]: int(np.count_nonzero(groups == g)) for g in np.unique(groups)},
        "group_weight_fractions": {
            group_names[int(g)]: float(weights[groups == g].sum() / weights.sum()) for g in np.unique(groups)
        },
        "weight_max": float(weights.max()) if len(weights) else 0.0,
        "inverse_weight_cap_before_normalization": 4.0,
    }


def rollout_expert_weights(env, labels, masks):
    """Compute one set of weights over known, non-forced labels in a rollout.

    Unknown labels and forced actions have zero weight and never reach the
    group mapper. Minibatches subsequently index these fixed rollout weights.
    """
    labels, masks = np.asarray(labels, dtype=np.int64), np.asarray(masks, dtype=bool)
    if masks.ndim != 2 or labels.shape != masks.shape[:1]:
        raise ValueError("Expected labels [rollout] and masks [rollout, actions]")
    eligible = (labels >= 0) & (masks.sum(axis=1) > 1)
    if np.any(labels[eligible] >= masks.shape[1]):
        raise ValueError("Expert label is outside the action space")
    if not np.all(masks[np.flatnonzero(eligible), labels[eligible]]):
        raise ValueError("Expert label violates the physical action mask")
    selected, groups, names = action_group_weights(env, labels[eligible], masks.shape[1])
    weights = np.zeros(len(labels), dtype=np.float64)
    weights[eligible] = selected
    return weights, {"eligible_labels": int(eligible.sum()), **_group_weight_summary(selected, groups, names)}


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n"
    )
    os.replace(temporary, path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class PPOTrainer:
    def __init__(self, env, config: PPOConfig, env_config: dict | None = None):
        self.env = env
        self.config = config
        self.env_config = env_config or {}
        self.device = torch.device(config.device)
        seed_everything(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.observation = _reset(env, config.seed)
        tensors = observation_tensors(self.observation, self.device)
        self.model = ActorCritic(
            tensors["features"].shape[-1],
            tensors["global"].shape[-1],
            config.hidden_dim,
            actor_context=config.actor_context,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.steps = 0
        self.episodes = 0
        self.reset_count = 0
        self.episode_reward = 0.0
        self.history: list[dict] = []
        self.warmstart_metrics: dict = {}
        self.ppo_decision_steps = 0
        self.ppo_optimizer_steps = 0
        self.ppo_actor_optimizer_steps = 0
        self.ppo_episodes = 0
        self.expert_label_steps = 0
        self.expert_optimizer_steps = 0

    def _teacher_action(self, observation):
        method = getattr(self.env, "teacher_action", None)
        if not callable(method):
            method = getattr(self.env, "heuristic_action", None)
        if not callable(method):
            raise ValueError("Expert supervision requires teacher_action or heuristic_action")
        action = method()
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("Teacher action must be an integer")
        action = int(action)
        mask = np.asarray(observation["mask"], dtype=bool)
        if action < 0 or action >= len(mask) or not mask[action]:
            raise ValueError("Teacher returned an invalid action for the physical mask")
        return action

    def _expert_coefficient(self, steps):
        fraction = min(1.0, steps / self.config.expert_decay_steps) if self.config.expert_decay_steps else 1.0
        return self.config.expert_coef + fraction * (self.config.expert_final_coef - self.config.expert_coef)

    def reset_optimizer(self):
        """Start a new optimization stage without changing learned state.

        BC and PPO have different gradient scales. Carrying Adam's BC moments
        across that boundary can suppress sparse scheduling-policy updates.
        Ordinary checkpoint resume still restores its original Adam state.
        """
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, eps=1e-5
        )

    def _clip_gradients(self):
        """Clip the separate actor/critic graphs independently; return raw norms."""
        actor_parameters = [*self.model.encoder.parameters(), *self.model.actor.parameters()]
        critic_parameters = [*self.model.critic_encoder.parameters(), *self.model.critic.parameters()]
        actor_norm = nn.utils.clip_grad_norm_(actor_parameters, self.config.max_grad_norm)
        critic_norm = nn.utils.clip_grad_norm_(critic_parameters, self.config.max_grad_norm)
        return float(actor_norm), float(critic_norm)

    def _next_episode(self, completed: bool = True):
        self.episodes += int(completed)
        self.reset_count += 1
        self.observation = _reset(self.env, self.config.seed + self.reset_count)
        self.episode_reward = 0.0

    def warmstart(
        self,
        steps: int = 4096,
        epochs: int = 100,
        on_progress: Callable[[dict], None] | None = None,
        *,
        learning_rate: float | None = None,
        minibatch_size: int | None = None,
        label_smoothing: float = 0.05,
        max_examples: int | None = None,
    ) -> dict:
        """Behavior cloning from *actual physical* heuristic trajectories.

        Forced WAIT states do not contribute to the classification dataset.
        This is initialization; reported PPO steps exclude demonstrations.
        A positive max_examples keeps a uniform reservoir of choice states,
        using its own seeded RNG without consuming the BC shuffle RNG.
        """
        if steps < 0 or epochs <= 0:
            raise ValueError("warmstart steps must be nonnegative and epochs positive")
        bc_lr = self.config.learning_rate if learning_rate is None else learning_rate
        bc_minibatch = self.config.minibatch_size if minibatch_size is None else minibatch_size
        if bc_lr <= 0 or bc_minibatch <= 0 or not 0 <= label_smoothing <= 1:
            raise ValueError("Invalid warmstart learning rate, minibatch size or label smoothing")
        if max_examples is not None:
            if (isinstance(max_examples, (bool, np.bool_))
                    or not isinstance(max_examples, (int, np.integer)) or max_examples <= 0):
                raise ValueError("max_examples must be a positive integer or None")
            max_examples = int(max_examples)
        reservoir_rng = (np.random.default_rng(np.random.SeedSequence([self.config.seed, 0xBC]))
                         if max_examples is not None else None)
        examples = []
        examples_seen = 0
        episode_results = []
        last_progress = time.monotonic()
        for step in range(steps):
            action = self._teacher_action(self.observation)
            mask = np.asarray(self.observation["mask"], dtype=bool)
            if action < 0 or action >= len(mask) or not mask[action]:
                raise ValueError("Heuristic returned an invalid action")
            if int(mask.sum()) > 1:
                examples_seen += 1
                # Algorithm R: every choice state seen so far has the same
                # probability of occupying one of the bounded retained slots.
                index = (len(examples) if max_examples is None or len(examples) < max_examples
                         else int(reservoir_rng.integers(examples_seen)))
                if max_examples is None or index < max_examples:
                    sample = ({k: np.array(v, copy=True) for k, v in self.observation.items()}, action)
                    if index == len(examples):
                        examples.append(sample)
                    else:
                        examples[index] = sample
            self.observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            self.episode_reward += float(reward)
            if terminated or truncated:
                episode_results.append({"reward": self.episode_reward, "info": info})
                self._next_episode()
            if on_progress and (
                time.monotonic() - last_progress > 15 or step + 1 == steps
            ):
                on_progress(
                    {
                        "phase": "warmstart",
                        "steps": step + 1,
                        "episodes": len(episode_results),
                        "examples": len(examples),
                        "examples_seen": examples_seen,
                        "examples_retained": len(examples),
                        "max_examples": max_examples,
                    }
                )
                last_progress = time.monotonic()
        metrics = {
            "physical_steps": steps,
            "epochs": epochs,
            "decision_examples": len(examples),
            "decision_examples_seen": examples_seen,
            "decision_examples_retained": len(examples),
            "max_examples": max_examples,
            "sampling": "uniform_reservoir" if max_examples is not None else "all_choice_states",
            "episodes": episode_results,
            "learning_rate": bc_lr, "minibatch_size": bc_minibatch,
            "label_smoothing": label_smoothing,
        }
        if examples:
            batch = observation_tensors(
                {
                    key: np.stack([sample[0][key] for sample in examples])
                    for key in ("features", "global", "mask")
                },
                self.device,
            )
            labels = torch.tensor(
                [sample[1] for sample in examples], device=self.device
            )
            # A useful expert may wait many decisions for the downstream arm.
            # Unweighted cloning then learns "always wait upstream" and can
            # exceed 90% accuracy while losing objects. Balance dispatch
            # patterns, not individual object IDs, to preserve rare picks.
            weights, groups, group_names = action_group_weights(
                self.env, labels.cpu().numpy(), batch["mask"].shape[-1]
            )
            balance = _group_weight_summary(weights, groups, group_names)
            sample_weights = torch.tensor(
                weights,
                dtype=torch.float32,
                device=self.device,
            )
            metrics["decision_group_counts"] = balance["group_counts"]
            metrics["class_balance"] = "inverse action-group frequency capped before mean normalization"
            metrics["inverse_weight_cap_before_normalization"] = 4.0
            metrics["sample_weight_max"] = balance["weight_max"]
            metrics["decision_group_weight_fractions"] = balance["group_weight_fractions"]
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=bc_lr, eps=1e-5)
            self.model.train()
            losses = []
            for _ in range(epochs):
                for start in range(0, len(examples), bc_minibatch):
                    # Shuffle once per epoch below via an independently permuted batch.
                    if start == 0:
                        order = self.rng.permutation(len(examples))
                    indices = torch.as_tensor(
                        order[start : start + bc_minibatch],
                        device=self.device,
                    )
                    distribution, _ = self.model(
                        {key: value[indices] for key, value in batch.items()}
                    )
                    loss = (
                        masked_expert_loss(distribution, labels[indices], batch["mask"][indices], label_smoothing)
                        * sample_weights[indices]
                    ).mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    losses.append(float(loss.detach()))
            with torch.no_grad():
                # Future-plan observations can have a choice on almost every
                # physics decision. Expanding all BC examples across all
                # actions at once can require several GiB just for the model's
                # concatenated inputs. Keep diagnostics at the same bounded
                # batch size as optimization; only scalar outputs are retained.
                correct = torch.empty(len(labels), device=self.device)
                teacher_probability = torch.empty_like(correct)
                for start in range(0, len(labels), bc_minibatch):
                    stop = min(start + bc_minibatch, len(labels))
                    distribution, _ = self.model(
                        {key: value[start:stop] for key, value in batch.items()}
                    )
                    correct[start:stop] = (
                        distribution.logits.argmax(-1) == labels[start:stop]
                    ).float()
                    teacher_probability[start:stop] = distribution.probs.gather(
                        1, labels[start:stop, None]
                    ).squeeze(1)
                metrics["training_accuracy"] = float(correct.mean())
                metrics["mean_teacher_probability"] = float(teacher_probability.mean())
                metrics["balanced_training_accuracy"] = float(
                    (correct * sample_weights).mean()
                )
                metrics["training_accuracy_by_dispatch"] = {
                    group_names[int(group)]: float(
                        correct[
                            torch.as_tensor(groups == group, device=self.device)
                        ].mean()
                    )
                    for group in np.unique(groups)
                }
            metrics["mean_loss"] = float(np.mean(losses))
            metrics["optimizer_steps"] = len(losses)
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate
        self.warmstart_metrics = metrics
        # Start PPO at a documented episode boundary after demonstrations.
        self._next_episode(completed=False)
        return metrics

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "architecture": self.model.architecture,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "episodes": self.episodes,
            "reset_count": self.reset_count,
            "ppo_decision_steps": self.ppo_decision_steps,
            "ppo_optimizer_steps": self.ppo_optimizer_steps,
            "ppo_actor_optimizer_steps": self.ppo_actor_optimizer_steps,
            "expert_label_steps": self.expert_label_steps,
            "expert_optimizer_steps": self.expert_optimizer_steps,
            "ppo_episodes": self.ppo_episodes,
            "train_config": asdict(self.config),
            "env_config": self.env_config,
            "rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "history": self.history,
            "warmstart_metrics": json.loads(
                json.dumps(self.warmstart_metrics, default=_json_default)
            ),
        }
        if self.device.type == "cuda":
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state(self.device)
        for method, key in (
            ("action_metadata", "scheduler_action_metadata"),
            ("observation_metadata", "scheduler_observation_metadata"),
        ):
            describe = getattr(self.env, method, None)
            if callable(describe):
                checkpoint[key] = describe()
        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    def resume(self, path: str | Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported policy checkpoint version; expected {CHECKPOINT_VERSION}"
            )
        if checkpoint["architecture"] != self.model.architecture:
            raise ValueError("Checkpoint observation dimensions or hidden size differ")
        if checkpoint.get("env_config") != self.env_config:
            raise ValueError(
                "Checkpoint environment config differs; use a matching environment to resume"
            )
        if checkpoint.get("train_config", {}).get("reward_scale", 1.0) != self.config.reward_scale:
            raise ValueError("Reward scale differs; start a fresh run to keep critic value units consistent")
        for method, key in (
            ("action_metadata", "scheduler_action_metadata"),
            ("observation_metadata", "scheduler_observation_metadata"),
        ):
            describe = getattr(self.env, method, None)
            expected = describe() if callable(describe) else None
            if checkpoint.get(key) != expected:
                raise ValueError("Checkpoint scheduler observation/action schema differs")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate
        self.steps = int(checkpoint["steps"])
        self.episodes = int(checkpoint["episodes"])
        self.reset_count = int(checkpoint.get("reset_count", self.episodes))
        self.ppo_decision_steps = int(checkpoint.get("ppo_decision_steps", 0))
        self.ppo_optimizer_steps = int(checkpoint.get("ppo_optimizer_steps", 0))
        self.ppo_episodes = int(checkpoint.get("ppo_episodes", 0))
        self.expert_label_steps = int(checkpoint.get("expert_label_steps", 0))
        self.expert_optimizer_steps = int(checkpoint.get("expert_optimizer_steps", 0))
        self.history = checkpoint.get("history", [])
        self.ppo_actor_optimizer_steps = int(
            checkpoint.get(
                "ppo_actor_optimizer_steps",
                sum(item.get("actor_optimizer_steps", 0) for item in self.history),
            )
        )
        self.warmstart_metrics = checkpoint.get("warmstart_metrics", {})
        self.rng.bit_generator.state = checkpoint["rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), self.device)
        # Physics states are not serialized. Resume intentionally begins a new
        # episode rather than claiming bit-for-bit mid-trajectory continuation.
        self._next_episode(completed=False)

    def train(
        self,
        on_update: Callable[[dict], None] | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        """Run until config.steps total PPO transitions (including resumed work)."""
        self.model.train()
        last_progress = time.monotonic()
        while self.steps < self.config.steps:
            expert_coef = self._expert_coefficient(self.steps)
            count = min(self.config.rollout_steps, self.config.steps - self.steps)
            observations, actions, log_probs, values = [], [], [], []
            rewards, next_values, terminated_flags, ended_flags = [], [], [], []
            expert_labels, greedy_actions = [], []
            completed_episodes = []
            has_teacher = callable(getattr(self.env, "teacher_action", None)) or callable(getattr(self.env, "heuristic_action", None))
            for _ in range(count):
                tensors = observation_tensors(self.observation, self.device)
                with torch.no_grad():
                    distribution, value = self.model(tensors)
                    action = distribution.sample()
                    log_prob = distribution.log_prob(action)
                    greedy_actions.append(int(distribution.logits.argmax(-1).item()))
                choice = int(np.count_nonzero(self.observation["mask"])) > 1
                expert_labels.append(
                    self._teacher_action(self.observation)
                    if choice and (has_teacher or expert_coef > 0) else -1
                )
                observations.append(
                    {
                        key: np.array(self.observation[key], copy=True)
                        for key in ("features", "global", "mask")
                    }
                )
                actions.append(int(action.item()))
                log_probs.append(float(log_prob.item()))
                values.append(float(value.item()))
                self.observation, reward, terminated, truncated, info = self.env.step(
                    actions[-1]
                )
                with torch.no_grad():
                    _, next_value = self.model(
                        observation_tensors(self.observation, self.device)
                    )
                rewards.append(float(reward))
                next_values.append(float(next_value.item()))
                terminated_flags.append(bool(terminated))
                ended_flags.append(bool(terminated or truncated))
                self.episode_reward += float(reward)
                self.steps += 1
                if terminated or truncated:
                    completed_episodes.append(
                        {"reward": self.episode_reward, "info": info}
                    )
                    self.ppo_episodes += 1
                    self._next_episode()
                if on_progress and time.monotonic() - last_progress > 15:
                    on_progress(
                        {"phase": "ppo", "steps": self.steps, "episodes": self.episodes}
                    )
                    last_progress = time.monotonic()
            learning_rewards = np.asarray(rewards, dtype=np.float32) * self.config.reward_scale
            advantage_array, return_array = generalized_advantages(
                learning_rewards,
                values,
                next_values,
                terminated_flags,
                ended_flags,
                self.config.gamma,
                self.config.gae_lambda,
            )
            batch = observation_tensors(
                {
                    key: np.stack([item[key] for item in observations])
                    for key in ("features", "global", "mask")
                },
                self.device,
            )
            action_tensor = torch.tensor(actions, device=self.device)
            expert_tensor = torch.tensor(expert_labels, device=self.device)
            old_log_probs = torch.tensor(log_probs, device=self.device)
            advantages = torch.as_tensor(advantage_array, device=self.device)
            returns = torch.as_tensor(return_array, device=self.device)
            decisions = batch["mask"].sum(-1) > 1
            decision_count = int(decisions.sum())
            self.ppo_decision_steps += decision_count
            self.expert_label_steps += int(np.count_nonzero(np.asarray(expert_labels) >= 0))
            advantages = decision_advantages(advantages, decisions)
            expert_weights, expert_balance_statistics = None, None
            if self.config.expert_balance:
                weights, expert_balance_statistics = rollout_expert_weights(
                    self.env, expert_labels, batch["mask"].cpu().numpy()
                )
                expert_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            losses, entropies, kls = [], [], []
            policy_losses, value_losses, imitation_losses = [], [], []
            actor_grad_norms, critic_grad_norms = [], []
            actor_updates = 0
            expert_updates = 0
            early_stop = False
            for _ in range(self.config.epochs):
                order = self.rng.permutation(count)
                for start in range(0, count, self.config.minibatch_size):
                    indices = torch.as_tensor(
                        order[start : start + self.config.minibatch_size],
                        device=self.device,
                    )
                    distribution, predicted_values = self.model(
                        {key: value[indices] for key, value in batch.items()}
                    )
                    new_log_probs = distribution.log_prob(action_tensor[indices])
                    policy_rows = decisions[indices]
                    if bool(policy_rows.any()):
                        log_ratio = (
                            new_log_probs[policy_rows]
                            - old_log_probs[indices][policy_rows]
                        )
                        ratio = log_ratio.exp()
                        clipped = ratio.clamp(
                            1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                        )
                        selected_advantages = advantages[indices][policy_rows]
                        policy_loss = -torch.minimum(
                            ratio * selected_advantages, clipped * selected_advantages
                        ).mean()
                        if expert_coef > 0:
                            expert_rows = (policy_rows & (expert_tensor[indices] >= 0)
                                           if expert_weights is not None else policy_rows)
                            if bool(expert_rows.any()):
                                supervised = masked_expert_loss(
                                    Categorical(logits=distribution.logits[expert_rows]),
                                    expert_tensor[indices][expert_rows],
                                    batch["mask"][indices][expert_rows],
                                    self.config.expert_label_smoothing,
                                )
                                if expert_weights is not None:
                                    supervised = supervised * expert_weights[indices][expert_rows]
                                imitation_loss = supervised.mean()
                                expert_updates += 1
                                imitation_losses.append(float(imitation_loss.detach()))
                            else:
                                imitation_loss = torch.zeros((), device=self.device)
                        else:
                            imitation_loss = torch.zeros((), device=self.device)
                        entropy = distribution.entropy()[policy_rows].mean()
                        approximate_kl = float(
                            ((ratio - 1) - log_ratio).mean().detach()
                        )
                        actor_updates += 1
                        entropies.append(float(entropy.detach()))
                        kls.append(approximate_kl)
                        policy_losses.append(float(policy_loss.detach()))
                    else:
                        # No actor graph: even Adam's previous momentum must
                        # not move actor parameters on forced-action batches.
                        policy_loss = torch.zeros((), device=self.device)
                        imitation_loss = torch.zeros((), device=self.device)
                        entropy = torch.zeros((), device=self.device)
                        approximate_kl = 0.0
                    value_loss = (
                        0.5 * (predicted_values - returns[indices]).square().mean()
                    )
                    loss = (
                        policy_loss
                        + expert_coef * imitation_loss
                        + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy
                    )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite PPO loss")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    actor_norm, critic_norm = self._clip_gradients()
                    actor_grad_norms.append(actor_norm)
                    critic_grad_norms.append(critic_norm)
                    self.optimizer.step()
                    self.ppo_optimizer_steps += 1
                    losses.append(float(loss.detach()))
                    value_losses.append(float(value_loss.detach()))
                    if (
                        self.config.target_kl > 0
                        and approximate_kl > self.config.target_kl
                    ):
                        early_stop = True
                        break
                if early_stop:
                    break
            self.ppo_actor_optimizer_steps += actor_updates
            self.expert_optimizer_steps += expert_updates
            known_experts = np.asarray(expert_labels) >= 0
            decision_rows = decisions.cpu().numpy()
            wait_action = getattr(self.env, "wait_action", batch["mask"].shape[-1] - 1)
            describe_actions = getattr(self.env, "action_metadata", None)
            action_metadata = describe_actions() if callable(describe_actions) else {}
            wait_semantics = action_metadata.get(
                "wait_semantics", "Do not dispatch a new task; executing tasks continue"
            )
            target_variance = float(np.var(return_array))
            record = {
                "steps": self.steps,
                "episodes_total": self.episodes,
                "ppo_episodes_total": self.ppo_episodes,
                "rollout_reward": float(np.sum(rewards)),
                "scaled_rollout_reward": float(np.sum(learning_rewards)),
                "reward_scale": self.config.reward_scale,
                "loss": float(np.mean(losses)),
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
                "imitation_loss": float(np.mean(imitation_losses)) if imitation_losses else 0.0,
                "expert_coef": expert_coef,
                "expert_balance": self.config.expert_balance,
                "expert_balance_statistics": expert_balance_statistics,
                "expert_label_steps_total": self.expert_label_steps,
                "expert_optimizer_steps": expert_updates,
                "expert_optimizer_steps_total": self.expert_optimizer_steps,
                "teacher_agreement": float(np.mean(np.asarray(greedy_actions)[known_experts] == np.asarray(expert_labels)[known_experts])) if known_experts.any() else None,
                "sampled_teacher_agreement": float(np.mean(np.asarray(actions)[known_experts] == np.asarray(expert_labels)[known_experts])) if known_experts.any() else None,
                "active_wait_rate": float(np.mean(np.asarray(actions)[decision_rows] == wait_action)) if decision_count else None,
                "greedy_active_wait_rate": float(np.mean(np.asarray(greedy_actions)[decision_rows] == wait_action)) if decision_count else None,
                "teacher_active_wait_rate": float(np.mean(np.asarray(expert_labels)[known_experts] == wait_action)) if known_experts.any() else None,
                "active_wait_definition": (
                    "selected wait_action among states with multiple valid scheduler actions; "
                    + wait_semantics
                ),
                "wait_action_semantics": wait_semantics,
                "wait_action_schema": action_metadata.get("schema"),
                "explained_variance": 1 - float(np.var(return_array - np.asarray(values))) / target_variance if target_variance > 1e-8 else None,
                "value_loss": float(np.mean(value_losses)),
                "actor_grad_norm": float(np.mean(actor_grad_norms)),
                "critic_grad_norm": float(np.mean(critic_grad_norms)),
                "gradient_clipping": "separate_actor_critic",
                "entropy": float(np.mean(entropies)) if entropies else 0.0,
                "approximate_kl": float(np.mean(kls)) if kls else 0.0,
                "decision_steps": decision_count,
                "forced_action_steps": count - decision_count,
                "ppo_decision_steps_total": self.ppo_decision_steps,
                "optimizer_steps": len(losses),
                "actor_optimizer_steps": actor_updates,
                "ppo_optimizer_steps_total": self.ppo_optimizer_steps,
                "ppo_actor_optimizer_steps_total": self.ppo_actor_optimizer_steps,
                "kl_early_stop": early_stop,
                "valid_choices_mean": float(batch["mask"].sum(-1).float().mean()),
                "completed_episodes": completed_episodes,
            }
            # Keep checkpoint metadata portable under torch's safe loader.
            record = json.loads(json.dumps(record, default=_json_default))
            self.history.append(record)
            if on_update:
                on_update(record)
        return self.history


def evaluate_policy(
    env,
    policy: Callable[[Observation], int] | None,
    seeds: list[int],
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Evaluate an inference policy, or the env heuristic when policy is None."""
    episodes = []
    last_progress = time.monotonic()
    for seed in seeds:
        observation = _reset(env, seed)
        total_reward, steps = 0.0, 0
        while True:
            action = env.heuristic_action() if policy is None else policy(observation)
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            steps += 1
            if on_progress and time.monotonic() - last_progress > 15:
                on_progress(
                    {
                        "phase": "evaluation",
                        "steps": steps,
                        "episodes": len(episodes),
                        "seed": seed,
                    }
                )
                last_progress = time.monotonic()
            if terminated or truncated:
                episodes.append(
                    {
                        "seed": seed,
                        "reward": total_reward,
                        "steps": steps,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": info,
                    }
                )
                break
    return {
        "episodes": episodes,
        "mean_reward": float(np.mean([episode["reward"] for episode in episodes]))
        if episodes
        else None,
    }
