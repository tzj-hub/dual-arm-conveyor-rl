"""Pure event-model tests: these are not physical success-rate evidence."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from dual_arm_conveyor.forecast import BeltReservation
from dual_arm_conveyor.sequence_plan import (
    DEFAULT_CONFIG, SequenceConfig, rank_sequence_actions, sequence_snapshot,
)
from dual_arm_conveyor.scheduler import SchedulerEnv, joint_action_ids
from test_scheduler_forecast import job, state
from test_scheduler_observation import VisualFixture
from test_future_planning import fixture, plan_action


def test_branching_finds_a_two_job_sequence_that_greedy_loses():
    # A fresh color-0 task is available before the color-1 hand returns. Serving
    # color 1 first and a later color-0 object fits two jobs; earliest-start EDF
    # consumes the shared station with one job. No simulator is involved.
    jobs = [job(i, kind, x) for i, (kind, x) in enumerate([
        (1, .0750282778593101), (1, -.03983911568957499),
        (0, -.3959831767368891), (0, .20052744304174652),
        (0, -.044219171096103516), (0, .4376861326170165),
    ])]
    snapshot = replace(state(jobs, arm_ready=[.4011064, .3572916, .5591968, 1.5601257]), horizon=12)
    choices = np.array([[10, 10, 10, 10]])
    greedy, before = rank_sequence_actions(snapshot, choices, [True], deferred=True,
                                           config=SequenceConfig(search_depth=0))
    beam, after = rank_sequence_actions(snapshot, choices, [True], deferred=True)
    assert before['actions'][0]['rescued'] == 1
    assert after['actions'][0]['rescued'] == 2
    assert beam[0] > greedy[0]
    assert after['actions'][0]['improved_by_beam']
    assert [p[:2] for p in after['actions'][0]['plan']] == [(3, 0), (2, 2)]


def test_beam_never_degrades_any_retained_greedy_first_action():
    snapshot = replace(state([job(0, 0, -.8), job(1, 1, -.6), job(2, 0, .1)]), horizon=12)
    choices = np.array([[0, 10, 10, 10], [10, 1, 10, 10], [10, 10, 2, 10], [10] * 4])
    greedy, _ = rank_sequence_actions(snapshot, choices, np.ones(4, bool), deferred=True,
                                      config=SequenceConfig(search_depth=0))
    beam, _ = rank_sequence_actions(snapshot, choices, np.ones(4, bool), deferred=True)
    assert np.all(beam >= greedy)


def test_disallowed_first_action_is_never_resurrected_and_inputs_stay_unchanged():
    snapshot = replace(state([job(0, 0, -.8)]), horizon=12)
    before = deepcopy(snapshot)
    choices = np.array([[0, 10, 10, 10], [10] * 4])
    mask = np.array([False, True])
    scores, diagnostics = rank_sequence_actions(snapshot, choices, mask, deferred=True)
    assert scores[0] == -np.inf and 0 not in diagnostics['actions']
    assert np.argmax(scores) == 1
    np.testing.assert_array_equal(mask, [False, True])
    np.testing.assert_array_equal(snapshot.ee, before.ee)
    np.testing.assert_array_equal(snapshot.arm_ready, before.arm_ready)
    np.testing.assert_array_equal(snapshot.jobs[0].position, before.jobs[0].position)


def test_duplicate_object_and_two_hands_at_same_station_roots_are_invalid():
    snapshot = replace(state([job(0, 0, -.8), job(1, 1, -.7)]), horizon=12)
    choices = np.array([[0, 10, 0, 10], [0, 1, 10, 10], [10] * 4])
    scores, details = rank_sequence_actions(snapshot, choices, [True] * 3, deferred=True)
    assert np.isneginf(scores[:2]).all()
    for report in details['actions'].values():
        slots = [p[1] for p in report['plan']]
        assert len(slots) == len(set(slots))


def test_every_sequence_preserves_arm_tail_and_station_clear_intervals():
    snapshot = replace(state([job(i, i % 2, -1.15 + .15 * i) for i in range(8)]), horizon=12)
    tails = (2.1, 2.3, 2.5, 2.7)
    clear = (0.82, .9, 1.0, 1.1)
    _, diagnostics = rank_sequence_actions(snapshot, np.array([[10] * 4]), [True],
                                           tail_by_arm=tails, clear_by_arm=clear)
    plan = diagnostics['actions'][0]['plan']
    for i, (arm, _, start, capture) in enumerate(plan):
        for prev_arm, _, _, prev_capture in plan[:i]:
            if prev_arm == arm:
                assert start + 1e-9 >= prev_capture + tails[arm]
            if prev_arm // 2 == arm // 2:
                assert start + 1e-9 >= prev_capture + clear[prev_arm]
        assert start / .1 == pytest.approx(round(start / .1))
        assert capture <= 12


def test_cross_station_reservation_delays_or_rejects_conflicting_pickup():
    snapshot = replace(state([job(0, 0, -.2, eligible=(False, True))]), horizon=12,
                       reservations=(BeltReservation(0, 0.0, 1.5, .25),))
    _, diagnostics = rank_sequence_actions(snapshot, np.array([[10] * 4]), [True])
    for _, _, start, _ in diagnostics['actions'][0]['plan']:
        assert start >= 1.5


def test_retry_time_expiry_and_closed_resource_are_not_bypassed():
    retry = replace(job(0, 0, -.5, eligible=(True, False)), available_after=(9., 0, 0, 0))
    closed = job(1, 1, -.8, eligible=(False, False))
    snapshot = replace(state([retry, closed]), horizon=12)
    _, diagnostics = rank_sequence_actions(snapshot, np.array([[10] * 4]), [True])
    assert diagnostics['actions'][0]['rescued'] == 0


def test_longer_horizon_can_see_visible_downstream_work_beyond_six_seconds():
    snapshot = state([job(0, 0, -1.15, eligible=(False, True))], arm_ready=[0, 0, 6.0, 0], horizon=12)
    _, short = rank_sequence_actions(snapshot, np.array([[10] * 4]), [True],
                                     config=SequenceConfig(horizon=6))
    _, long = rank_sequence_actions(snapshot, np.array([[10] * 4]), [True])
    assert short['actions'][0]['rescued'] == 0
    assert long['actions'][0]['rescued'] == 1
    assert long['actions'][0]['plan'][0][3] > 6


def test_search_and_eta_cache_have_explicit_bounds_when_budget_exhausts():
    snapshot = state([job(i, i % 2, -1.1 + .13 * i) for i in range(10)], horizon=12)
    config = SequenceConfig(beam_width=8, search_depth=4, max_expansions=7)
    scores, diagnostics = rank_sequence_actions(snapshot, np.array([[10] * 4]), [True], config=config)
    info = diagnostics['search']
    assert info['budget_exhausted'] and info['expansions'] == 7
    assert info['expansions'] <= info['expansion_bound']
    assert info['peak_beam'] <= 8 and info['depth_reached'] <= 4
    assert info['eta_evaluations'] <= info['eta_cache_bound']
    assert np.isfinite(scores[0])


def test_snapshot_and_teacher_only_access_visible_controller_fixture():
    env = VisualFixture()  # Undefined hidden/supply properties deliberately raise.
    before = deepcopy(env)
    snapshot, tails, clear = sequence_snapshot(env)
    assert snapshot.horizon == DEFAULT_CONFIG.horizon == 12
    assert {j.slot for j in snapshot.jobs} == {0, 1, 2, 3}
    assert len(tails) == len(clear) == 4 and min(clear) >= .82
    adapter = SchedulerEnv(env, teacher_mode='sequence')
    selected = adapter.teacher_action()
    assert env.observe()['mask'][joint_action_ids()[selected]]
    assert 'no future supply' in adapter.teacher_metadata()['inputs']
    np.testing.assert_array_equal(env.data.qpos, before.data.qpos)
    np.testing.assert_array_equal(env.belief, before.belief)
    assert env.phase == before.phase


def test_sequence_future_plan_keeps_the_existing_soft_mask_and_native_commit():
    env = fixture(busy=True)
    adapter = SchedulerEnv(env, teacher_mode='sequence')
    observation = adapter.observe()
    action = adapter.teacher_action()
    assert observation['mask'][action]
    assert adapter.action_to_native(action) == env.wait_action
    # Even a later valid native candidate cannot bypass a supplied joint mask.
    env.phase[0] = 'idle'
    native = env.observe()
    native['mask'][:] = False
    native['mask'][env.wait_action] = True
    next_action = adapter.teacher_action()
    assert adapter.action_to_native(next_action, native) == env.wait_action


def test_sequence_snapshots_use_observed_tail_samples_without_changing_physics():
    env = fixture()
    env.phase_duration_samples = [dict(return_=[])] * 4
    env.phase_duration_samples = [dict(transport=[1.5] * 8, lift=[.8] * 8) for _ in range(4)]
    _, observed, clear = sequence_snapshot(env)
    env.phase_duration_samples = [dict() for _ in range(4)]
    _, nominal, _ = sequence_snapshot(env)
    assert all(a > b for a, b in zip(observed, nominal))
    assert clear == pytest.approx((1.1,) * 4)


@pytest.mark.parametrize('kwargs', [dict(horizon=13), dict(beam_width=17), dict(search_depth=5), dict(max_expansions=1281)])
def test_unbounded_search_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SequenceConfig(**kwargs)


def test_equivalent_roots_share_beam_slots_and_receive_the_same_improvement():
    snapshot = state([job(i, i % 2, -1.1 + .15 * i) for i in range(6)], horizon=12)
    choices = np.array([[10] * 4] * 3)
    scores, diagnostics = rank_sequence_actions(snapshot, choices, [True] * 3,
                                                config=SequenceConfig(search_depth=0))
    assert diagnostics['search']['root_actions'] == 3
    assert diagnostics['search']['unique_root_choices'] == 1
    assert diagnostics['search']['peak_beam'] == 1
    np.testing.assert_array_equal(scores, [scores[0]] * 3)
    scores, diagnostics = rank_sequence_actions(snapshot, choices, [True] * 3)
    np.testing.assert_array_equal(scores, [scores[0]] * 3)
    assert all(diagnostics['actions'][i]['plan'] == diagnostics['actions'][0]['plan'] for i in range(3))


def test_current_root_dispatch_reproduces_native_priority_when_joint_is_blocked():
    jobs = [job(0, 0, -.65), job(1, 1, .1)]
    snapshot = state(jobs, eta={(0, 0): 1.1, (3, 1): 1.1}, horizon=12)
    choices = np.array([[0, 10, 10, 1]])
    _, diagnostics = rank_sequence_actions(
        snapshot, choices, [True], deferred=True,
        native_choices=[(0, 10, 10, 10), (10, 10, 10, 1), (10,) * 4],
        current_slack={(0, 0): 9.0, (3, 1): .1},
    )
    plan = diagnostics['actions'][0]['plan']
    assert plan[0][:3] == (3, 1, 0.0)
    assert next(p for p in plan if p[0] == 0)[2] >= .1


def test_current_native_joint_dispatch_remains_simultaneous():
    snapshot = state([job(0, 0, -.8), job(1, 1, .1)],
                     eta={(0, 0): 1.1, (3, 1): 1.1}, horizon=12)
    choices = np.array([[0, 10, 10, 1]])
    _, diagnostics = rank_sequence_actions(snapshot, choices, [True], deferred=True,
        native_choices=[(0, 10, 10, 1), (10,) * 4], current_slack={(0, 0): 9., (3, 1): .1})
    assert all(p[2] == 0.0 for p in diagnostics['actions'][0]['plan'][:2])


def test_unavailable_current_root_cannot_be_forecast_as_immediate_dispatch():
    snapshot = state([job(0, 0, -.8)], eta={(0, 0): 1.1}, horizon=12)
    _, diagnostics = rank_sequence_actions(snapshot, np.array([[0, 10, 10, 10]]), [True],
                                            deferred=True, native_choices=[(10,) * 4])
    assert diagnostics['actions'][0]['plan'][0][2] >= .1


def test_multiple_legal_roots_change_choice_when_beam_finds_an_extra_job():
    jobs = [job(i, kind, x) for i, (kind, x) in enumerate([
        (1, -1.1102290681154439), (1, -.8822431448605994),
        (0, -.6341748223162104), (0, -.43432583836852734),
        (0, -.19078136577702226), (0, .06920579010237202),
        (0, .31186105510306467), (0, .5695430144500887),
    ])]
    assert min(np.diff([j.position[0] for j in jobs])) > .19
    snapshot = state(jobs, arm_ready=[.8763612825, .0514544962, .8208186396, .1647003726], horizon=12)
    choices = np.array([[10, 0, 10, 10], [10, 1, 10, 10], [10] * 4])
    greedy, gd = rank_sequence_actions(snapshot, choices, [True] * 3, deferred=True,
                                       config=SequenceConfig(search_depth=0))
    beam, bd = rank_sequence_actions(snapshot, choices, [True] * 3, deferred=True)
    assert np.isfinite(greedy).all() and np.isfinite(beam).all()
    assert int(np.argmax(greedy)) == 0 and gd['actions'][0]['rescued'] == 3
    assert int(np.argmax(beam)) == 1 and bd['actions'][1]['rescued'] == 4
