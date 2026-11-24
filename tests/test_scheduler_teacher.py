"""Teacher preferences use visual/controller fixtures and original native scoring."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.scheduler import (
    WAIT_ACTION,
    SchedulerEnv,
    joint_action_ids,
    to_native_action,
)


class PreferenceFixture:
    """Execute actual native mask/scoring code, with simple visual candidates."""

    num_arms = 4
    action_count = 11**4
    wait_action = action_count - 1
    _observe_multi = ConveyorEnv._observe_multi
    observe = ConveyorEnv._observe_multi
    heuristic_action = ConveyorEnv._heuristic_multi

    def __init__(self, candidates, rejected=(), *, preferences=False, tied=False):
        self.config = SimpleNamespace(use_dispatch_preferences=preferences)
        self.action_choices = np.array(np.unravel_index(np.arange(self.action_count), (11,) * 4)).T
        self.tracks = [SimpleNamespace(predict_velocity=lambda now: np.array([0.2, 0, 0])) for _ in range(10)]
        self.positions = np.c_[np.arange(10) * 0.6, np.zeros((10, 2))]
        self.data = SimpleNamespace(time=1.0, site_xpos=np.zeros((4, 3)))
        self.ids = SimpleNamespace(ee_sites=np.arange(4))
        self.target = np.full(4, -1)
        self.phase = ["idle"] * 4
        self.bin_occupied = np.zeros((2, 180), dtype=bool)
        self.candidates = set(candidates)
        self.rejected = set(rejected)
        self.parameters = {
            pair: (1.0, 5.0) if tied else (1.0 + pair[1] * 0.1, 5.0 + pair[1] * 0.3)
            for pair in self.candidates
        }
        self.urgency_calls = []

    def __getattr__(self, name):
        if name in {"states", "types", "serial", "cube_position", "cube_velocity", "model", "belt_sample"}:
            raise AssertionError(f"Teacher accessed simulator truth: {name}")
        raise AttributeError(name)

    def _position(self, slot):
        return self.positions[slot]

    def _candidate(self, arm, slot):
        if (arm, slot) not in self.candidates:
            return False, 0.0, 0.0
        eta, deadline = self.parameters[arm, slot]
        if self.config.use_dispatch_preferences and self._returning_partner_has_urgent_work(arm, eta, deadline):
            return False, eta, deadline
        return True, eta, deadline

    def _returning_partner_has_urgent_work(self, arm, eta, deadline):
        matching = {pair for pair, parameters in self.parameters.items()
                    if pair[0] == arm and parameters == (eta, deadline)}
        assert matching, "Urgency was queried for an illegal visual candidate"
        self.urgency_calls.extend(matching)
        return bool(matching & self.rejected)


@pytest.mark.parametrize("candidates,rejected", [
    ({(2, 2)}, {(2, 2)}),
    ({(2, 2), (3, 3)}, {(2, 2)}),
    ({(0, 0), (1, 1), (2, 2), (3, 3)}, {(2, 2), (3, 3)}),
    ({(0, 0), (2, 0), (2, 2), (3, 3)}, {(2, 2)}),
    ({(0, 0), (2, 2), (2, 4), (3, 3)}, set()),
    (set(), set()),
])
def test_teacher_matches_original_soft_preference_native_rule(candidates, rejected):
    student = PreferenceFixture(candidates, rejected)
    original = PreferenceFixture(candidates, rejected, preferences=True)
    action = SchedulerEnv(student).teacher_action()
    assert to_native_action(action) == original.heuristic_action()
    assert student.observe()["mask"][to_native_action(action)]


def test_teacher_wait_is_a_learnable_choice_without_mutating_student():
    student = PreferenceFixture({(2, 2)}, {(2, 2)})
    adapter = SchedulerEnv(student)
    before = student.observe()
    mask_copy, scores_copy = before["mask"].copy(), before["per_arm_scores"].copy()
    config, phases, targets = student.config, list(student.phase), student.target.copy()
    positions, robot_positions = student.positions.copy(), student.data.site_xpos.copy()
    assert np.count_nonzero(before["mask"][joint_action_ids()]) > 1
    assert adapter.heuristic_action() != WAIT_ACTION
    assert adapter.teacher_action() == WAIT_ACTION
    np.testing.assert_array_equal(student._last_mask, mask_copy)
    np.testing.assert_array_equal(before["mask"], mask_copy)
    np.testing.assert_array_equal(before["per_arm_scores"], scores_copy)
    after = student.observe()
    np.testing.assert_array_equal(after["mask"], mask_copy)
    np.testing.assert_array_equal(after["per_arm_scores"], scores_copy)
    assert student.config is config and not config.use_dispatch_preferences
    assert student.phase == phases and student.data.time == 1.0
    np.testing.assert_array_equal(student.target, targets)
    np.testing.assert_array_equal(student.positions, positions)
    np.testing.assert_array_equal(student.data.site_xpos, robot_positions)


def test_teacher_queries_urgency_only_for_options_in_a_physical_joint_action():
    student = PreferenceFixture({(2, 2)})
    # Existing upstream reservation makes this otherwise valid candidate
    # physically unavailable in the native mask before teacher preferences.
    student.target[0] = 0
    student.phase[0] = "approach"
    student.data.site_xpos[0, 0] = student.positions[2, 0] + 0.2 * student.parameters[2, 2][0]
    assert SchedulerEnv(student).teacher_action() == WAIT_ACTION
    assert student.urgency_calls == []


def test_teacher_ties_follow_original_native_argmax_and_preserve_mask_restrictions():
    student = PreferenceFixture({(0, 0), (1, 1), (2, 2), (3, 3)}, tied=True)
    # One shared color-bin slot disallows simultaneous same-color picks.
    student.bin_occupied[0, 1:] = True
    original = deepcopy(student)
    original.config.use_dispatch_preferences = True
    native = student.observe()
    totals = native["per_arm_scores"][np.arange(4)[None, :], student.action_choices].sum(1)
    valid_ids = np.flatnonzero(native["mask"])
    tied_maxima = valid_ids[totals[valid_ids] == totals[valid_ids].max()]
    assert len(tied_maxima) > 1
    chosen = to_native_action(SchedulerEnv(student).teacher_action())
    assert chosen == int(tied_maxima.min()) == original.heuristic_action()
    assert native["mask"][chosen]


def test_random_visual_candidate_sets_preserve_original_teacher_decisions():
    random = np.random.default_rng(706)
    for _ in range(25):
        candidates = {(arm, slot) for arm in range(4) for slot in range(10) if random.random() < 0.15}
        rejected = {pair for pair in candidates if pair[0] >= 2 and random.random() < 0.4}
        student = PreferenceFixture(candidates, rejected)
        original = PreferenceFixture(candidates, rejected, preferences=True)
        assert to_native_action(SchedulerEnv(student).teacher_action()) == original.heuristic_action()
