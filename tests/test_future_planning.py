"""Next-job planning contracts: intent cannot bypass physical execution."""
from copy import deepcopy

import numpy as np
import pytest

from dual_arm_conveyor.forecast import (
    controller_readiness, observed_phase_budgets, predicted_ready_poses, rank_future_actions,
    visual_forecast_state,
)
from dual_arm_conveyor.future_plan import (
    ACTION_COUNT, CANCEL, KEEP, OPTIONS, WAIT_ACTION, planner_for,
)
from dual_arm_conveyor.scheduler import (
    OPTION_FEATURE_NAMES, PAIR_FEATURE_NAMES, SchedulerEnv, action_metadata,
    encode_observation, model_policy, observation_metadata,
)
from test_scheduler_observation import VisualFixture


def fixture(*, busy=True, x=-0.85):
    env = VisualFixture()
    env.config.future_planning = True
    env.config.safe_preposition = True
    env.tracks = [env.tracks[0]] + [None] * 9
    env.tracks[0].track_id = 101
    env.tracks[0].position[0] = x
    env.belief[:] = 0
    env.belief[0] = 1
    env.phase = ['return' if busy else 'idle', 'idle', 'idle', 'idle']
    env.target[:] = -1
    env.holding[:] = False
    env.phase_start[:] = env.data.time - .1
    env.ids.home_ee = env.data.site_xpos.copy()
    env.return_goals = env.ids.home_ee.copy()
    return env


def plan_action(upstream=KEEP, downstream=KEEP):
    return upstream * OPTIONS + downstream


def test_busy_arm_can_plan_without_touching_physical_state_then_dispatches_when_ready():
    env = fixture()
    adapter = SchedulerEnv(env)
    before = deepcopy((env.phase, env.target, env.belief, env.data.qpos))
    obs = adapter.observe()
    action = plan_action(0)
    assert obs['mask'][action]
    assert env.observe()['mask'].sum() == 1
    assert adapter.action_to_native(action) == env.wait_action
    assert planner_for(env).plans[0].track_id == 101
    assert env.phase == before[0]
    for actual, original in zip((env.target, env.belief, env.data.qpos), before[1:]):
        np.testing.assert_array_equal(actual, original)
    env.phase[0] = 'idle'
    native = adapter.action_to_native(WAIT_ACTION)
    assert env.observe()['mask'][native]
    assert env.action_choices[native].tolist() == [0, 10, 10, 10]
    assert planner_for(env).plans[0] is None


def test_upstream_object_can_be_planned_before_its_window_and_cancelled():
    env = fixture(busy=False, x=-1.05)
    adapter = SchedulerEnv(env)
    assert not env._candidate(0, 0)[0]
    assert adapter.observe()['mask'][plan_action(0)]
    assert adapter.action_to_native(plan_action(0)) == env.wait_action
    assert np.isfinite(env.future_preposition_goals[0]).all()
    assert abs(env.future_preposition_goals[0, 1]) >= .25
    assert adapter.observe()['mask'][plan_action(CANCEL)]
    assert adapter.action_to_native(plan_action(CANCEL)) == env.wait_action
    assert planner_for(env).plans[0] is None
    assert np.isnan(env.future_preposition_goals).all()
    assert planner_for(env).metrics()['cancelled'] == 1


def test_repeated_same_plan_is_keep_and_does_not_inflate_created_counts():
    env = fixture()
    adapter = SchedulerEnv(env)
    adapter.action_to_native(plan_action(0))
    assert not adapter.observe()['mask'][plan_action(0)]
    adapter.action_to_native(WAIT_ACTION)
    assert planner_for(env).metrics()['created'] == 1
    assert planner_for(env).metrics()['replaced'] == 0
    assert adapter.dispatch_group_names[0] == 'keep_plans'
    np.testing.assert_array_equal(adapter.expert_action_groups([
        WAIT_ACTION, plan_action(CANCEL), plan_action(KEEP, 0), plan_action(CANCEL, 0),
    ]), [0, 1, 2, 3])


def test_soft_plan_follows_identity_across_slot_changes_and_expires_on_reuse():
    env = fixture()
    adapter = SchedulerEnv(env)
    adapter.action_to_native(plan_action(0))
    env.tracks[7], env.tracks[0] = env.tracks[0], None
    env.belief[7], env.belief[0] = 1, 0
    assert planner_for(env).slot_for(planner_for(env).plans[0]) == 7
    env.tracks[7].track_id = 202
    adapter.observe()
    assert planner_for(env).plans[0] is None
    assert planner_for(env).metrics()['expired'] == 1
    assert not np.any(env.target >= 0)


def test_future_plan_never_bypasses_joint_physical_mask():
    env = fixture(busy=False)
    adapter = SchedulerEnv(env)
    native = env.observe()
    native['mask'][:] = False
    native['mask'][env.wait_action] = True
    action = adapter.action_to_native(plan_action(0), native)
    assert action == env.wait_action
    assert planner_for(env).plans[0] is not None
    assert planner_for(env).metrics()['dispatch_commands'] == 0


def test_same_object_cannot_have_two_station_plans():
    env = fixture(x=-.85)
    encoded = encode_observation(env)
    assert encoded['mask'][plan_action(0)]
    assert encoded['mask'][plan_action(KEEP, 0)]
    assert not encoded['mask'][plan_action(0, 0)]


def test_new_contract_is_distinct_and_teacher_has_no_plan_side_effects():
    env = fixture()
    adapter = SchedulerEnv(env, teacher_mode='predictive')
    obs = adapter.observe()
    before = deepcopy(planner_for(env).plans), deepcopy(planner_for(env).counts)
    action = adapter.teacher_action()
    assert obs['mask'][action]
    assert (planner_for(env).plans, planner_for(env).counts) == before
    assert obs['features'].shape == (ACTION_COUNT, adapter.observation_metadata()['feature_dim'])
    assert obs['global'].shape == (adapter.observation_metadata()['global_dim'],)
    assert adapter.action_metadata()['schema'] != action_metadata()['schema']
    assert adapter.observation_metadata()['schema'] != observation_metadata()['schema']
    assert adapter.wait_action == WAIT_ACTION


def test_keep_retains_full_candidate_description_of_its_reserved_object():
    env = fixture()
    adapter = SchedulerEnv(env)
    before = adapter.observe()
    candidate_action = plan_action(0)
    adapter.action_to_native(candidate_action)
    after = adapter.observe()
    assert after['mask'][WAIT_ACTION]
    assert not after['mask'][candidate_action]
    width = len(OPTION_FEATURE_NAMES)
    np.testing.assert_array_equal(after['features'][WAIT_ACTION, :width],
                                  before['features'][candidate_action, :width])
    names = {name: i for i, name in enumerate(OPTION_FEATURE_NAMES)}
    assert after['features'][WAIT_ACTION, names['track_present']] == 1
    assert after['features'][WAIT_ACTION, names['size_x']] > 0
    assert after['features'][WAIT_ACTION, names['yaw_sin2']] != 0
    assert after['features'][WAIT_ACTION, names['relative_ee_x']] != 0


def test_keep_pair_geometry_and_future_eta_match_two_explicit_target_choices():
    env = fixture()
    second = deepcopy(env.tracks[0])
    second.track_id, second.kind = 102, 1
    second.position = np.array([-.10, .065, .235])
    second.yaw = -.6
    env.tracks[1], env.belief[1] = second, 1
    env.phase[3] = 'return'
    adapter = SchedulerEnv(env)
    selected = plan_action(0, 11)
    before = adapter.observe()
    assert before['mask'][selected]
    adapter.action_to_native(selected)
    assert all(p is not None for p in planner_for(env).plans)
    after = adapter.observe()
    # The existing plans have the same descriptors and geometric relation as
    # their explicit SET action; only extra intent-type flags and masks differ.
    pair_offset = 2 * len(OPTION_FEATURE_NAMES)
    np.testing.assert_array_equal(after['features'][WAIT_ACTION, :pair_offset],
                                  before['features'][selected, :pair_offset])
    np.testing.assert_array_equal(after['features'][WAIT_ACTION, pair_offset:pair_offset + len(PAIR_FEATURE_NAMES) - 1],
                                  before['features'][selected, pair_offset:pair_offset + len(PAIR_FEATURE_NAMES) - 1])
    pair = after['features'][WAIT_ACTION, pair_offset:]
    assert pair[PAIR_FEATURE_NAMES.index('both_dispatch')] == 1
    assert pair[PAIR_FEATURE_NAMES.index('target_planar_distance')] > 0
    eta_index = OPTION_FEATURE_NAMES.index('candidate_eta')
    expected = (after['features'][WAIT_ACTION, eta_index]
                + after['features'][WAIT_ACTION, len(OPTION_FEATURE_NAMES) + eta_index])
    assert pair[PAIR_FEATURE_NAMES.index('sum_candidate_eta')] == pytest.approx(expected)


def test_native_inference_and_training_adapter_use_same_stateful_mapping():
    left, right = fixture(), fixture()
    native_policy = model_policy(left, lambda obs: plan_action(0))
    training = SchedulerEnv(right)
    assert native_policy(left.observe()) == training.action_to_native(plan_action(0))
    assert planner_for(left).plans == planner_for(right).plans
    left.phase[0] = right.phase[0] = 'idle'
    native_policy = model_policy(left, lambda obs: WAIT_ACTION)
    assert native_policy(left.observe()) == training.action_to_native(WAIT_ACTION)


def test_episode_reset_removes_old_plans_even_if_track_ids_repeat():
    env = fixture()
    adapter = SchedulerEnv(env)
    adapter.action_to_native(plan_action(0))
    adapter.reset(seed=4)
    assert planner_for(env).plans == [None, None]
    assert planner_for(env).metrics()['created'] == 0


def test_observed_phase_model_requires_samples_and_uses_only_completed_durations():
    env = fixture()
    env.phase_duration_samples = [dict(transport=[1.7] * 7) for _ in range(4)]
    assert observed_phase_budgets(env, 0)['transport'] == .9
    env.phase_duration_samples[0]['transport'].append(1.7)
    assert observed_phase_budgets(env, 0)['transport'] == pytest.approx(1.7)
    env.config.future_planning = False
    assert observed_phase_budgets(env, 0)['transport'] == .9


def test_busy_arm_transition_origin_predicts_its_current_task_end():
    env = fixture()
    env.config.direct_retask = True
    env.phase[0] = 'transport'
    env.target[0] = 4
    env.drop_targets = np.zeros((10, 3))
    env.drop_targets[4] = [-.35, -.4, .325]
    env.drop_kind = np.zeros(10, dtype=int)
    env.ids.arm_base_positions = np.array([[-.5, -.6, 0], [-.5, .6, 0], [.5, -.6, 0], [.5, .6, 0]])
    env.ids.bin_rim_z = .255
    env.lift_height = .38
    assert predicted_ready_poses(env)[0, 0] == pytest.approx(-.35)
    assert predicted_ready_poses(env)[0, 2] == pytest.approx(.38)


def apply_cooldown(env, seconds):
    env.retry_after = np.zeros((4, 10))
    env.retry_after[0, 0] = env.data.time + seconds
    original = env._candidate
    def candidate(arm, slot):
        if env.data.time < env.retry_after[arm, slot]:
            return False, 0., 0.
        return original(arm, slot)
    env._candidate = candidate


def test_retry_cooldown_removes_future_pick_that_would_finish_past_window():
    env = fixture(busy=False, x=-.5)
    apply_cooldown(env, 1.2)
    state, proposals = planner_for(env).snapshot()
    assert state.jobs[0].available_after == pytest.approx((1.2, 0, 0, 0))
    assert (0, 0) not in proposals
    assert (2, 0) in proposals  # Downstream has its own independent retry clock.
    assert not encode_observation(env)['mask'][plan_action(0)]
    scores, details = rank_future_actions(state, np.array([[0, 10, 10, 10]]), [True])
    assert not np.isfinite(scores[0])
    assert details[0]['rescued'] == 0


def test_cooldown_plan_waits_until_expiry_then_native_mask_allows_dispatch():
    env = fixture(busy=False, x=-.85)
    apply_cooldown(env, .3)
    adapter = SchedulerEnv(env)
    state, proposals = planner_for(env).snapshot()
    assert proposals[0, 0].start == pytest.approx(.3)
    assert adapter.observe()['mask'][plan_action(0)]
    assert adapter.action_to_native(plan_action(0)) == env.wait_action
    env.data.time += .3
    env.tracks[0].last_seen = env.data.time
    state, proposals = planner_for(env).snapshot()
    assert state.jobs[0].available_after[0] == pytest.approx(0.)
    assert proposals[0, 0].start == pytest.approx(0.)
    dispatched = adapter.action_to_native(WAIT_ACTION)
    assert env.action_choices[dispatched].tolist() == [0, 10, 10, 10]
    assert env.observe()['mask'][dispatched]


def test_retry_clock_capture_does_not_change_legacy_forecast_mode():
    env = fixture(busy=False)
    apply_cooldown(env, 1.2)
    env.config.future_planning = False
    state = visual_forecast_state(env)
    assert state.jobs[0].available_after == (0., 0., 0., 0.)


def test_planned_return_emits_next_job_hint_without_idle_preposition():
    env = fixture()
    env.config.safe_preposition = False
    env.config.planned_return = True
    adapter = SchedulerEnv(env)
    assert adapter.action_to_native(plan_action(0)) == env.wait_action
    assert np.isfinite(env.future_preposition_goals[0]).all()
    assert env.future_preposition_goals[0, 1] == pytest.approx(-.26)
    adapter.action_to_native(plan_action(CANCEL))
    assert np.isnan(env.future_preposition_goals).all()


def planned_return_fixture():
    env = fixture()
    env.config.direct_retask = True
    env.config.planned_return = True
    env.phase[0] = 'release'
    env.target[0] = 4
    env.drop_targets = np.zeros((10, 3))
    env.drop_targets[4] = [-.35, -.4, .325]
    env.drop_kind = np.zeros(10, dtype=int)
    env.ids.arm_base_positions = np.array([[-.5, -.6, 0], [-.5, .6, 0], [.5, -.6, 0], [.5, .6, 0]])
    env.ids.bin_rim_z = .255
    env.lift_height = .38
    env.transport_height = .325
    env.future_preposition_goals = np.full((4, 3), np.nan)
    return env


def test_planned_return_prediction_matches_bounded_hint_and_no_hint_fallback():
    env = planned_return_fixture()
    np.testing.assert_allclose(predicted_ready_poses(env)[0], [-.35, -.26, .38])
    env.future_preposition_goals[0] = [-.1, -.26, .4]
    np.testing.assert_allclose(predicted_ready_poses(env)[0], [-.32, -.26, .38])
    env.config.planned_return = False
    np.testing.assert_allclose(predicted_ready_poses(env)[0], [-.35, -.25, .38])


def test_return_in_progress_uses_locked_goal_even_if_next_hint_changes():
    env = planned_return_fixture()
    env.phase[0] = 'return'
    env.return_goals[0] = [-.45, -.26, .38]
    env.future_preposition_goals[0] = [-.1, -.26, .4]
    np.testing.assert_allclose(predicted_ready_poses(env)[0], env.return_goals[0])
    before, _ = controller_readiness(env)
    env.future_preposition_goals[0, 0] = -.9
    after, _ = controller_readiness(env)
    np.testing.assert_array_equal(before, after)


def test_planned_return_readiness_uses_distance_to_intended_endpoint():
    env = planned_return_fixture()
    near, _ = controller_readiness(env)
    env.future_preposition_goals[0] = [-.9, -.26, .38]
    far, _ = controller_readiness(env)
    assert far[0] > near[0] + .25


def test_prediction_calls_controller_pure_goal_without_recursive_scheduler():
    env = planned_return_fixture()
    calls = []
    def pure_goal(arm):
        calls.append(arm)
        return np.array([-.4, -.26, .38])
    env._planned_ready_goal = pure_goal
    np.testing.assert_allclose(predicted_ready_poses(env)[0], [-.4, -.26, .38])
    assert calls == [0]


@pytest.mark.parametrize('phase', ['recover', 'yield'])
def test_planned_return_then_recovery_or_yield_predicts_home_not_cached_return(phase):
    env = planned_return_fixture()
    env.phase[0] = 'return'
    env.return_goals[0] = [-.32, -.26, .38]
    env.data.site_xpos[0] = env.return_goals[0]
    np.testing.assert_allclose(predicted_ready_poses(env)[0], env.return_goals[0])
    env.phase[0] = phase
    np.testing.assert_allclose(predicted_ready_poses(env)[0], env.ids.home_ee[0])
    readiness, _ = controller_readiness(env)
    speed = .85 * .65
    expected = np.linalg.norm(env.ids.home_ee[0] - env.data.site_xpos[0]) / speed + .2
    assert readiness[0] == pytest.approx(expected)
    assert readiness[0] > .2
    env.return_goals[0] = [.2, -.4, .5]
    np.testing.assert_allclose(predicted_ready_poses(env)[0], env.ids.home_ee[0])
    changed, _ = controller_readiness(env)
    np.testing.assert_array_equal(changed, readiness)


@pytest.mark.parametrize('phase', ['recover', 'yield'])
def test_legacy_forecast_recovery_endpoint_behavior_is_unchanged(phase):
    env = planned_return_fixture()
    env.config.future_planning = env.config.planned_return = False
    env.phase[0] = phase
    env.return_goals[0] = [-.32, -.26, .38]
    np.testing.assert_allclose(predicted_ready_poses(env)[0], env.return_goals[0])
    readiness, _ = controller_readiness(env)
    expected = np.linalg.norm(env.return_goals[0] - env.data.site_xpos[0]) / (.85 * .65) + .2
    assert readiness[0] == pytest.approx(expected)
