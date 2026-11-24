"""Scheduler contract checks without creating or advancing a MuJoCo scene."""

from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from dual_arm_conveyor.scheduler import (
    ACTION_COUNT,
    FEATURE_DIM,
    GLOBAL_DIM,
    GLOBAL_FEATURE_NAMES,
    WAIT_ACTION,
    SchedulerEnv,
    action_metadata,
    encode_observation,
    from_native_action,
    joint_action_ids,
    model_policy,
    observation_metadata,
    to_native_action,
)


@dataclass
class VisibleTrack:
    position: np.ndarray
    kind: int
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.0, 0.0]))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    size: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.045, 0.04]))
    timestamp: float = 4.0
    last_seen: float = 3.9
    confidence: float = 0.9
    uncertainty: float = 0.002
    visible: bool = True
    yaw: float = 0.4
    yaw_period: float = np.pi

    def predict(self, now):
        return self.position + self.velocity * (now - self.timestamp)

    def predict_velocity(self, now):
        return self.velocity.copy()


class VisualFixture:
    """Only public visual/controller fields exist; truth access raises."""

    num_arms = 4
    action_count = 11**4
    wait_action = action_count - 1

    def __init__(self):
        self.config = SimpleNamespace(
            grasp_coast_time=1.5,
            track_max_age=0.5,
            belt_top=0.205,
            belt_speed=0.9,
        )
        self.ids = SimpleNamespace(
            ee_sites=np.arange(4),
            arm_qpos=np.arange(36).reshape(4, 9)[:, :7],
            arm_dofs=np.arange(36).reshape(4, 9)[:, :7],
            finger_qpos=np.arange(36).reshape(4, 9)[:, 7:],
        )
        qpos, qvel = np.full(80, np.nan), np.full(80, np.nan)
        qpos[:36], qvel[:36] = 0.0, 0.0
        qpos[self.ids.finger_qpos] = 0.04
        self.data = SimpleNamespace(
            time=4.0,
            qpos=qpos,
            qvel=qvel,
            site_xpos=np.array(
                [
                    [-0.5, -0.25, 0.4],
                    [-0.5, 0.25, 0.4],
                    [0.5, -0.25, 0.4],
                    [0.5, 0.25, 0.4],
                ]
            ),
        )
        self.tracks = [
            VisibleTrack(np.array([-0.65, -0.065, 0.23]), 0),
            VisibleTrack(np.array([-0.5, 0.065, 0.235]), 1),
            VisibleTrack(np.array([0.12, -0.065, 0.23]), 0),
            VisibleTrack(np.array([0.6, 0.065, 0.232]), 1),
            VisibleTrack(np.array([-0.3, -0.06, 0.23]), 1),
            VisibleTrack(np.array([0.2, 0.065, 0.23]), 0, last_seen=2.0),
            None,
            None,
            None,
            None,
        ]
        self.belief = np.array([1, 1, 1, 1, 2, 1, 0, 0, 0, 0])
        self.phase = ["idle", "transport", "idle", "idle"]
        self.phase_start = np.array([3.0, 3.5, 3.0, 3.0])
        self.target = np.array([-1, 4, -1, -1])
        self.holding = np.array([False, True, False, False])
        self.windows = np.array(
            [[-0.95, -0.1], [-0.95, -0.1], [-0.15, 0.82], [-0.15, 0.82]]
        )
        self.sensor = SimpleNamespace(
            belt_velocity=lambda now: np.array([0.21, 0.0, 0.0])
        )
        self.action_choices = np.array(np.unravel_index(np.arange(11**4), (11,) * 4)).T
        self.steps = 0
        self.closed = False

    def __getattr__(self, name):
        if name in {
            "states",
            "types",
            "serial",
            "cube_position",
            "cube_velocity",
            "model",
            "belt_sample",
            "spawned",
            "archived",
            "truth_target",
        }:
            raise AssertionError(
                f"Scheduler accessed forbidden simulator truth: {name}"
            )
        raise AttributeError(name)

    def _candidate(self, arm, slot):
        track = self.tracks[slot]
        if track is None:
            return False, 0.0, 0.0
        p = track.predict(self.data.time)
        low, high = self.windows[arm]
        eta = 0.4 + np.linalg.norm(p - self.data.site_xpos[arm])
        deadline = (high - p[0]) / track.velocity[0]
        ok = (
            self.phase[arm] == "idle"
            and self.belief[slot] == 1
            and track.kind == arm % 2
            and low <= p[0] <= high
            and self.data.time - track.last_seen <= self.config.track_max_age
            and deadline > eta
        )
        return bool(ok), float(eta), float(deadline)

    def observe(self):
        valid = np.ones((4, 11), dtype=bool)
        for arm in range(4):
            valid[arm, :10] = [self._candidate(arm, slot)[0] for slot in range(10)]
        choices = self.action_choices
        mask = np.all(valid[np.arange(4)[None], choices], axis=1)
        mask &= ~((choices[:, 0] < 10) & (choices[:, 1] < 10))
        mask &= ~((choices[:, 2] < 10) & (choices[:, 3] < 10))
        for arm in range(2):
            for other in range(2, 4):
                mask &= ~(
                    (choices[:, arm] < 10) & (choices[:, arm] == choices[:, other])
                )
        # NaNs are intentional: the RL encoder must never read heuristic scores.
        return {"mask": mask, "per_arm_scores": np.full((4, 11), np.nan)}

    def reset(self, seed=None):
        self.seed = seed
        return self.observe()

    def step(self, action, *, physics_callback=None):
        self.last_action = action
        self.steps += 1
        self.data.time += 0.1
        if physics_callback is not None:
            physics_callback(self)
        return self.observe(), -1.25, True, False, {"physical_result": "unchanged"}

    def heuristic_action(self):
        return int(np.flatnonzero(self.observe()["mask"])[0])

    def metrics(self):
        return {"steps": self.steps}

    def close(self):
        self.closed = True


def test_action_map_covers_exactly_the_native_station_exclusion():
    ids = joint_action_ids()
    assert ids.shape == (441,) and len(np.unique(ids)) == 441
    choices = np.array(np.unravel_index(np.arange(11**4), (11,) * 4)).T
    representable = ((choices[:, :2] < 10).sum(1) <= 1) & (
        (choices[:, 2:] < 10).sum(1) <= 1
    )
    np.testing.assert_array_equal(np.sort(ids), np.flatnonzero(representable))
    for action, native in enumerate(ids):
        assert to_native_action(action) == native
        assert from_native_action(int(native)) == action
    assert to_native_action(WAIT_ACTION) == 11**4 - 1
    ids[0] = -1
    assert joint_action_ids()[0] >= 0
    with pytest.raises(ValueError, match="both hands"):
        from_native_action(0)


@pytest.mark.parametrize("invalid", [-1, 441, True, 1.5])
def test_invalid_station_actions_are_rejected(invalid):
    with pytest.raises(ValueError):
        to_native_action(invalid)


def test_encoding_uses_only_visual_and_robot_state_and_preserves_wait():
    env = VisualFixture()
    before = (env.data.qpos.copy(), env.data.qvel.copy(), env.belief.copy())
    native = env.observe()
    obs = encode_observation(env, native)
    assert obs["features"].shape == (ACTION_COUNT, FEATURE_DIM)
    assert obs["global"].shape == (GLOBAL_DIM,)
    assert obs["features"].dtype == obs["global"].dtype == np.float32
    assert np.isfinite(obs["features"]).all() and np.isfinite(obs["global"]).all()
    np.testing.assert_array_equal(obs["mask"], native["mask"][joint_action_ids()])
    assert obs["mask"][WAIT_ACTION] and np.count_nonzero(obs["mask"]) > 1
    speed_index = GLOBAL_FEATURE_NAMES.index("observed_belt_speed")
    assert obs["global"][speed_index] == pytest.approx(0.21 / 0.5)
    for old, current in zip(before, (env.data.qpos, env.data.qvel, env.belief)):
        np.testing.assert_array_equal(old, current)


def test_slot_permutation_only_permutates_candidate_rows():
    env = VisualFixture()
    initial = encode_observation(env)
    changed = deepcopy(env)
    permutation = np.array([4, 8, 2, 6, 0, 9, 1, 7, 3, 5])
    inverse = np.argsort(permutation)
    changed.tracks = [env.tracks[i] for i in permutation]
    changed.belief = env.belief[permutation]
    changed.target = np.array([inverse[t] if t >= 0 else -1 for t in env.target])
    permuted = encode_observation(changed)
    option_permutation = np.r_[permutation, permutation + 10, 20]
    rows = (option_permutation[:, None] * 21 + option_permutation[None, :]).reshape(-1)
    np.testing.assert_allclose(
        permuted["features"], initial["features"][rows], atol=1e-7
    )
    np.testing.assert_array_equal(permuted["mask"], initial["mask"][rows])
    np.testing.assert_allclose(permuted["global"], initial["global"], atol=1e-7)


def test_empty_scene_and_stale_tracks_have_finite_neutral_queues():
    env = VisualFixture()
    env.tracks = [None] * 10
    env.belief[:] = 0
    env.target[:] = -1
    env.sensor = None
    obs = encode_observation(env)
    assert np.flatnonzero(obs["mask"]).tolist() == [WAIT_ACTION]
    assert np.isfinite(obs["features"]).all() and np.isfinite(obs["global"]).all()
    for index, name in enumerate(GLOBAL_FEATURE_NAMES):
        if name.startswith("station") or name in (
            "observed_belt_speed",
            "fresh_track_count",
            "coasting_track_count",
        ):
            assert obs["global"][index] == 0


def test_wrapper_and_playback_preserve_native_physics_reward_and_metadata():
    native = VisualFixture()
    wrapper = SchedulerEnv(native)
    obs = wrapper.reset(seed=71)
    assert native.seed == 71
    assert wrapper.action_count == 441 and wrapper.wait_action == 440
    assert wrapper.config is native.config
    assert wrapper.action_metadata() == action_metadata()
    assert wrapper.observation_metadata() == observation_metadata()
    choice = wrapper.heuristic_action()
    assert obs["mask"][choice]
    assert to_native_action(choice) == native.heuristic_action()
    called = []
    result = wrapper.step(choice, physics_callback=lambda env: called.append(env))
    assert native.steps == 1 and native.data.time == pytest.approx(4.1)
    assert native.last_action == to_native_action(choice) and called == [native]
    assert result[1:] == (-1.25, True, False, {"physical_result": "unchanged"})
    assert wrapper.metrics() == {"steps": 1}

    def learned(encoded):
        assert encoded["features"].shape == (441, FEATURE_DIM)
        return WAIT_ACTION

    assert model_policy(native, learned)(native.observe()) == native.wait_action
    assert wrapper.model_policy(learned)(native.observe()) == native.wait_action
    wrapper.close()
    assert native.closed


def test_encoded_observation_is_accepted_by_existing_actor_critic():
    from dual_arm_conveyor.learning import ActorCritic, observation_tensors

    obs = encode_observation(VisualFixture())
    actor = ActorCritic(FEATURE_DIM, GLOBAL_DIM, hidden_dim=16)
    distribution, values = actor(observation_tensors(obs))
    assert values.shape == (1,)
    probabilities = distribution.probs.detach().numpy()[0]
    assert np.all(probabilities[~obs["mask"]] == 0)
    assert probabilities[WAIT_ACTION] > 0
