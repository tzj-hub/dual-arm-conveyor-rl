"""Predictive scheduling uses observable time windows, never simulator truth."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dual_arm_conveyor.forecast import (
    BeltReservation, ForecastJob, ForecastState, controller_readiness,
    nominal_tail_seconds, phase_budgets, pickup_eta, rank_actions,
    rank_future_actions, visual_forecast_state,
)
from dual_arm_conveyor.scheduler import (
    FEATURE_NAMES, GLOBAL_FEATURE_NAMES, SchedulerEnv, WAIT_ACTION, encode_observation,
    joint_action_ids, observation_metadata,
)
from test_scheduler_observation import VisualFixture


def job(slot, kind, x, *, eligible=(True, True)):
    return ForecastJob(slot, kind, np.array([x, -0.065 if kind == 0 else 0.065, 0.23]),
                       np.array([0.2, 0.0, 0.0]), 0.35, eligible)


def state(jobs, *, arm_ready=None, station_ready=None, eta=None, horizon=6.0):
    poses = np.array([[-0.50, -0.20, 0.42], [-0.50, 0.20, 0.42],
                      [0.50, -0.20, 0.42], [0.50, 0.20, 0.42]])
    return ForecastState(
        tuple(jobs), np.zeros(4) if arm_ready is None else np.array(arm_ready, dtype=float),
        np.zeros(2) if station_ready is None else np.array(station_ready, dtype=float),
        poses.copy(), poses.copy(),
        np.array([[-0.95, -0.10], [-0.95, -0.10], [-0.15, 0.82], [-0.15, 0.82]]),
        0.5525, 2.0, 0.85, 2.97, {} if eta is None else eta,
        horizon=horizon,
    )


def test_predictive_wait_preserves_urgent_partner_job_then_catches_fresh_job():
    # Picking the fresh color-0 job immediately blocks the last opportunity for
    # color 1. The other hand becomes ready in 0.2 s; wait and serve it first.
    snapshot = state([job(0, 0, -0.10), job(1, 1, 0.38)],
                     arm_ready=[0, 0, 0, 0.20], station_ready=[0, 0.20],
                     eta={(2, 0): 1.0})
    actions = np.array([[10, 10, 0, 10], [10, 10, 10, 10]])
    scores, detail = rank_actions(snapshot, actions, np.ones(2, dtype=bool))
    assert detail[0]["rescued"] == 1
    assert detail[1]["rescued"] == 2
    assert scores[1] > scores[0]
    assert detail[1]["plan"][0][0:2] == (3, 1)


def test_idle_upstream_miss_can_still_be_recovered_downstream():
    snapshot = state([job(0, 0, -0.50)], arm_ready=[5.0, 0, 0, 0])
    scores, detail = rank_actions(snapshot, np.array([[10, 10, 10, 10]]), [True])
    assert np.isfinite(scores[0])
    assert detail[0]["rescued"] == 1
    assert detail[0]["plan"][0][0] == 2
    assert detail[0]["plan"][0][2] == pytest.approx(1.75)


def test_equivalent_outcomes_prefer_dispatch_now_over_repeated_wait():
    snapshot = state([job(0, 0, -0.70)], eta={(0, 0): 1.2})
    actions = np.array([[0, 10, 10, 10], [10, 10, 10, 10]])
    scores, detail = rank_actions(snapshot, actions, [True, True])
    assert detail[0]["rescued"] == detail[1]["rescued"] == 1
    assert scores[0] > scores[1]


def test_expired_closed_bin_and_beyond_horizon_jobs_are_not_counted():
    snapshot = state([job(0, 0, 0.90), job(1, 1, 0.10, eligible=(False, False)),
                      job(2, 0, -3.0)], horizon=4.0)
    _, detail = rank_actions(snapshot, np.array([[10, 10, 10, 10]]), [True])
    assert detail[0]["rescued"] == 0


def test_forecaster_cannot_resurrect_masked_assignment():
    snapshot = state([job(0, 0, -0.50)], eta={(0, 0): 1.0})
    actions = np.array([[0, 10, 10, 10], [10, 10, 10, 10]])
    scores, details = rank_actions(snapshot, actions, [False, True])
    assert scores[0] == -np.inf and 0 not in details
    assert np.argmax(scores) == 1
    with pytest.raises(ValueError, match="matching mask"):
        rank_actions(snapshot, actions, [False, False])


def test_forecast_root_and_followup_jobs_respect_per_arm_retry_availability():
    cooling = replace(job(0, 0, -.8, eligible=(True, False)),
                      available_after=(.4, 0., 0., 0.))
    snapshot = state([cooling])
    root = np.array([[0, 10, 10, 10]])
    _, deferred = rank_future_actions(snapshot, root, [True])
    assert deferred[0]['rescued'] == 1
    assert deferred[0]['plan'][0][2] >= .4
    scores, immediate = rank_actions(snapshot, root, [True])
    assert not np.isfinite(scores[0]) and immediate[0]['rescued'] == 0
    _, followup = rank_actions(snapshot, np.array([[10, 10, 10, 10]]), [True])
    assert followup[0]['rescued'] == 1
    assert followup[0]['plan'][0][2] >= .4


def test_retry_availability_can_expire_entire_followup_window():
    cooling = replace(job(0, 0, -.5, eligible=(True, False)),
                      available_after=(1.2, 0., 0., 0.))
    _, details = rank_actions(state([cooling]), np.array([[10, 10, 10, 10]]), [True])
    assert details[0]['rescued'] == 0


def test_visual_snapshot_never_reads_object_truth_or_future_supply_and_is_pure():
    env = VisualFixture()
    before = deepcopy(env)
    snapshot = visual_forecast_state(env)
    # The stale track, assigned track, and empty slots do not become future jobs.
    assert {j.slot for j in snapshot.jobs} == {0, 1, 2, 3}
    assert snapshot.jobs[0].velocity[0] == pytest.approx(0.2)
    assert snapshot.jobs[0].velocity[0] != env.config.belt_speed
    assert env.data.time == before.data.time and env.phase == before.phase
    np.testing.assert_array_equal(env.belief, before.belief)
    np.testing.assert_array_equal(env.data.qpos, before.data.qpos)
    np.testing.assert_array_equal(env.data.site_xpos, before.data.site_xpos)


def test_return_readiness_uses_measured_new_goal_and_cannot_expire_to_zero():
    env = VisualFixture()
    env.phase = ["return", "idle", "idle", "idle"]
    env.phase_start[:] = -100.0
    env.return_goals = env.data.site_xpos.copy()
    env.return_goals[0, 0] += 0.3
    far, _ = controller_readiness(env)
    env.return_goals[0] = env.data.site_xpos[0]
    close, _ = controller_readiness(env)
    assert far[0] > close[0] > 0
    np.testing.assert_array_equal(close[1:], 0)


def test_queue_entry_time_distinguishes_same_count_with_different_arrival_times():
    env = VisualFixture()
    env.tracks = [env.tracks[0]] + [None] * 9
    env.belief[:] = 0
    env.belief[0] = 1
    env.target[:] = -1
    env.tracks[0].position[0] = -1.05
    first = encode_observation(env)
    env.tracks[0].position[0] -= 0.20
    later = encode_observation(env)
    index = GLOBAL_FEATURE_NAMES.index("station0_color0_next_unassigned_entry")
    count = GLOBAL_FEATURE_NAMES.index("station0_color0_unassigned_ahead_count")
    assert later["global"][count] == first["global"][count]
    assert later["global"][index] - first["global"][index] == pytest.approx(1.0 / 5.0)
    assert observation_metadata()["schema"] == "visual_station_candidates_timing_v2"


def test_predictive_teacher_is_legal_and_reports_its_approximate_model():
    env = VisualFixture()
    adapter = SchedulerEnv(env, teacher_mode="predictive")
    action = adapter.teacher_action()
    assert env.observe()["mask"][joint_action_ids()[action]]
    assert adapter.teacher_metadata()["mode"] == "predictive"
    assert adapter.teacher_metadata()["horizon_seconds"] == 6.0
    assert "no future supply" in adapter.teacher_metadata()["inputs"]
    with pytest.raises(ValueError, match="teacher mode"):
        SchedulerEnv(env, teacher_mode="not-a-teacher")


def test_rolling_forecast_cannot_postpone_active_dispatch_forever(monkeypatch):
    import dual_arm_conveyor.scheduler as scheduler
    env = VisualFixture()
    adapter = SchedulerEnv(env, teacher_mode="predictive")
    def stubborn_wait(state, choices, mask):
        scores = np.full(len(mask), -np.inf)
        scores[mask] = 0.0
        scores[WAIT_ACTION] = 1.0
        return scores, {}
    monkeypatch.setattr(scheduler, "rank_actions", stubborn_wait)
    assert adapter.teacher_action() == WAIT_ACTION
    env.data.time += 0.3
    # Maintain genuinely delivered observations during this controller test.
    for track in env.tracks:
        if track is not None:
            track.last_seen = env.data.time
    assert adapter.teacher_action() == WAIT_ACTION
    env.data.time += 0.3
    for track in env.tracks:
        if track is not None:
            track.last_seen = env.data.time
    action = adapter.teacher_action()
    assert action != WAIT_ACTION
    assert env.observe()["mask"][joint_action_ids()[action]]
    adapter.reset(seed=90)
    assert adapter._predictive_wait_started is None


def test_interception_requires_arm_to_be_faster_than_visible_belt_motion():
    snapshot = state([job(0, 0, -0.70)])
    fast = ForecastJob(0, 0, snapshot.jobs[0].position, np.array([0.6, 0, 0]), 0.35, (True, True))
    assert pickup_eta(snapshot, fast, 0, 0) == np.inf


def test_nominal_budgets_follow_enabled_motion_and_release_operations():
    env = VisualFixture()
    baseline = phase_budgets(env)
    assert nominal_tail_seconds(env) == pytest.approx(2.97)
    env.config.direct_retask = True
    env.config.direct_bin_release = True
    env.config.responsive_release = True
    changed = phase_budgets(env)
    assert changed["return"] < baseline["return"]
    assert 0 < changed["place"] < baseline["place"]
    assert 0 < changed["retreat"] < baseline["retreat"]
    # Typical visible minimum width is 45 mm. Opening begins at a 2 mm
    # command, so 10 mm of clearance alone does NOT determine release time.
    command_slew = ((0.045 + 0.010) / 2 - 0.002) / 0.10
    assert changed["release"] == pytest.approx(command_slew + 0.10 + 0.04)
    assert changed["release"] > 0.010 / (2 * 0.10) + 0.10
    assert nominal_tail_seconds(env) < 2.97
    # The feature functions return independent budgets, never mutate defaults.
    fresh = VisualFixture()
    assert phase_budgets(fresh) == baseline


def test_responsive_release_readiness_tracks_command_and_contact_hold_progress():
    env = VisualFixture()
    env.config.responsive_release = True
    env.phase = ["release", "idle", "idle", "idle"]
    env.phase_start[0] = env.data.time - 1.0
    env.release_start_aperture = np.array([0.045, np.nan, np.nan, np.nan])
    env.release_clear_since = np.full(4, np.nan)
    env.data.qpos[env.ids.finger_qpos[0]] = 0.0225
    env.ik = SimpleNamespace(gripper_targets=np.array([0.002, 0.04, 0.04, 0.04]))
    slow, _ = controller_readiness(env)
    # Elapsed wall/simulation time cannot make stalled closed jaws ready.
    env.phase_start[0] -= 10.0
    stalled, _ = controller_readiness(env)
    assert stalled[0] == slow[0]
    env.ik.gripper_targets[0] = 0.04
    env.data.qpos[env.ids.finger_qpos[0]] = 0.04
    env.release_clear_since[0] = env.data.time - 0.12
    cleared, _ = controller_readiness(env)
    assert 0 < cleared[0] < stalled[0]


def test_coincident_bin_heights_shorten_following_phases_without_erasing_checks():
    env = VisualFixture()
    env.phase = ["place", "idle", "idle", "idle"]
    env.phase_start[0] = env.data.time
    normal, _ = controller_readiness(env)
    env.config.direct_bin_release = True
    direct, _ = controller_readiness(env)
    assert 0 < direct[0] < normal[0]
    assert normal[0] - direct[0] == pytest.approx((0.18 - 0.04) + (0.14 - 0.04))


def test_future_cross_station_pick_waits_for_nearby_existing_reservation(monkeypatch):
    import dual_arm_conveyor.forecast as forecast
    monkeypatch.setattr(forecast, "pickup_eta", lambda *args: 0.4)
    near = replace(state([job(0, 0, 0.05)]),
                   reservations=(BeltReservation(0, 0.0, 1.0, 0.10),))
    _, detail = rank_actions(near, np.array([[10, 10, 10, 10]]), [True])
    assert detail[0]["rescued"] == 1
    assert detail[0]["plan"][0][0] == 2
    assert detail[0]["plan"][0][2] >= 1.0
    far = replace(near, reservations=(BeltReservation(0, 0.0, 1.0, -0.80),))
    _, far_detail = rank_actions(far, np.array([[10, 10, 10, 10]]), [True])
    assert far_detail[0]["plan"][0][2] == pytest.approx(0.1)


def test_root_pick_creates_future_cross_station_reservation(monkeypatch):
    import dual_arm_conveyor.forecast as forecast
    monkeypatch.setattr(forecast, "pickup_eta", lambda *args: 0.4)
    snapshot = state([job(0, 0, -0.20), job(1, 1, 0.05)])
    _, detail = rank_actions(snapshot, np.array([[0, 10, 10, 10]]), [True])
    assert detail[0]["rescued"] == 2
    first, second = detail[0]["plan"]
    assert first[:2] == (0, 0) and second[:2] == (3, 1)
    assert second[2] >= first[3] + 0.82


def test_cross_station_delay_can_remove_an_unrealistic_forecast_success(monkeypatch):
    import dual_arm_conveyor.forecast as forecast
    monkeypatch.setattr(forecast, "pickup_eta", lambda *args: 0.4)
    snapshot = replace(state([job(0, 0, 0.35)]),
                       reservations=(BeltReservation(0, 0.0, 2.0, 0.45),))
    _, detail = rank_actions(snapshot, np.array([[10, 10, 10, 10]]), [True])
    assert detail[0]["rescued"] == 0


def test_snapshot_records_active_cross_station_span_from_robot_encoders():
    env = VisualFixture()
    env.phase[1] = "approach"
    snapshot = visual_forecast_state(env)
    assert len(snapshot.reservations) == 1
    reservation = snapshot.reservations[0]
    assert reservation.station == 0 and reservation.start == 0
    assert reservation.x == env.data.site_xpos[1, 0]
    assert reservation.clear > 0


def test_completion_extension_can_save_a_pick_started_inside_raw_window(monkeypatch):
    import dual_arm_conveyor.forecast as forecast
    monkeypatch.setattr(forecast, "pickup_eta", lambda *args: 0.8)
    target = job(0, 0, -0.28, eligible=(True, False))
    snapshot = state([target])
    _, raw = rank_actions(snapshot, np.array([[10, 10, 10, 10]]), [True])
    extended = replace(snapshot, jobs=(replace(target, completion_high=(-0.02, 0.82)),))
    _, result = rank_actions(extended, np.array([[10, 10, 10, 10]]), [True])
    assert raw[0]["rescued"] == 0
    assert result[0]["rescued"] == 1
    plan = result[0]["plan"][0]
    assert plan[0] == 0
    assert target.position[0] + target.velocity[0] * plan[2] <= -0.10
    assert target.position[0] + target.velocity[0] * plan[3] <= -0.02


def test_extension_never_allows_a_late_assignment_outside_raw_window(monkeypatch):
    import dual_arm_conveyor.forecast as forecast
    monkeypatch.setattr(forecast, "pickup_eta", lambda *args: 0.01)
    target = replace(job(0, 0, -0.12, eligible=(True, False)),
                     completion_high=(-0.02, 0.82))
    snapshot = state([target], arm_ready=[0.20, 0, 0, 0])
    _, result = rank_actions(snapshot, np.array([[10, 10, 10, 10]]), [True])
    assert result[0]["rescued"] == 0


def test_reservation_delay_must_recheck_raw_start_window_with_extension(monkeypatch):
    import dual_arm_conveyor.forecast as forecast
    monkeypatch.setattr(forecast, "pickup_eta", lambda *args: 0.01)
    target = replace(job(0, 0, -0.20, eligible=(True, False)),
                     completion_high=(-0.02, 0.82))
    snapshot = replace(state([target]),
                       reservations=(BeltReservation(1, 0.0, 0.60, -0.10),))
    _, result = rank_actions(snapshot, np.array([[10, 10, 10, 10]]), [True])
    assert result[0]["rescued"] == 0


@pytest.mark.parametrize("lateral_speed", [0.04, -0.03])
def test_lateral_drift_outside_calibrated_band_removes_completion_extension(monkeypatch, lateral_speed):
    import dual_arm_conveyor.forecast as forecast
    monkeypatch.setattr(forecast, "pickup_eta", lambda *args: 0.8)
    target = replace(job(0, 0, -0.28, eligible=(True, False)),
                     completion_high=(-0.02, 0.82), velocity=np.array([0.2, lateral_speed, 0]))
    _, result = rank_actions(state([target]), np.array([[10, 10, 10, 10]]), [True])
    assert result[0]["rescued"] == 0


def test_snapshot_uses_controller_completion_bound_but_exit_feature_stays_raw():
    env = VisualFixture()
    old = encode_observation(env)
    env.config.extended_pick_deadline = True
    def completion(arm, position):
        side = -1 if arm % 2 == 0 else 1
        return -0.02 if arm < 2 and 0.045 <= side * position[1] <= 0.080 else env.windows[arm, 1]
    env.completion_window_high = completion
    original_candidate = env._candidate
    def candidate(arm, slot):
        ok, eta, deadline = original_candidate(arm, slot)
        track = env.tracks[slot]
        if track is not None:
            p = track.predict(env.data.time)
            deadline += (completion(arm, p) - env.windows[arm, 1]) / track.velocity[0]
        return ok, eta, deadline
    env._candidate = candidate
    changed = encode_observation(env)
    row = 20  # Upstream arm 0 / slot 0, downstream WAIT.
    candidate_index = FEATURE_NAMES.index("station0_candidate_deadline")
    raw_index = FEATURE_NAMES.index("station0_time_to_window_exit")
    assert changed["features"][row, candidate_index] - old["features"][row, candidate_index] == pytest.approx(0.4 / 5)
    assert changed["features"][row, raw_index] == old["features"][row, raw_index]
    snapshot = visual_forecast_state(env)
    visible = {job.slot: job for job in snapshot.jobs}
    assert visible[0].completion_high[0] == -0.02
    assert visible[0].completion_high[1] == 0.82
    assert "raw assignment-window" in observation_metadata()["window_semantics"]["time_to_window_exit"]
