"""Occupancy includes downtime and accounts for every arm-second exactly once."""

import json

import numpy as np
import pytest

from dual_arm_conveyor.utilization import ArmUtilization, CATEGORIES


def assert_conservation(summary):
    duration = summary["duration_seconds"]
    for seconds, ratios, busy in zip(
        summary["per_arm_seconds"], summary["per_arm_ratios"], summary["per_arm_busy"]
    ):
        assert set(seconds) == set(ratios) == set(CATEGORIES)
        assert sum(seconds.values()) == pytest.approx(duration, abs=1e-9)
        assert sum(ratios.values()) == pytest.approx(1.0 if duration else 0.0, abs=1e-10)
        if duration:
            assert busy == pytest.approx(1 - ratios["idle"], abs=1e-10)
        assert all(0 <= value <= 1 for value in ratios.values())
    assert summary["mean_busy"] == pytest.approx(np.mean(summary["per_arm_busy"]))


def test_known_trace_distinguishes_execution_return_wait_recovery_and_idle():
    meter = ArmUtilization(2)
    # An initial supply gap still contributes to the denominator.
    meter.record(["idle", "idle"], [True, True], 10)
    meter.record(["approach", "return"], [False, False], 2)
    meter.record(["approach", "recover"], [True, True], 3)
    meter.record(["release", "transfer_wait"], [True, False], 5)
    result = meter.report()
    report = result["cumulative"]
    assert report["duration_seconds"] == 20
    assert report["per_arm_seconds"] == [
        {"processing": 7, "return": 0, "waiting": 3, "recovery": 0, "idle": 10},
        {"processing": 0, "return": 2, "waiting": 5, "recovery": 3, "idle": 10},
    ]
    assert report["per_arm_busy"] == [0.5, 0.5]
    assert report["mean_ratios"]["processing"] == pytest.approx(7 / 40)
    assert result["window"] == report
    assert_conservation(report)
    json.dumps(result, allow_nan=False)


def test_window_cuts_through_an_activity_segment_and_keeps_real_denominator():
    meter = ArmUtilization(2, window_seconds=30)
    meter.record(["idle", "approach"], [True, False], 12)
    meter.record(["transport", "recover"], [False, False], 13)
    meter.record(["return", "idle"], [False, True], 20)
    report = meter.report()
    assert report["cumulative"]["duration_seconds"] == 45
    assert report["window"]["duration_seconds"] == 30
    assert report["window"]["per_arm_seconds"] == [
        {"processing": 10, "return": 20, "waiting": 0, "recovery": 0, "idle": 0},
        {"processing": 0, "return": 0, "waiting": 0, "recovery": 10, "idle": 20},
    ]
    assert_conservation(report["cumulative"])
    assert_conservation(report["window"])
    # One long interval can expire every previous segment and its own prefix.
    meter.record(["yield", "close"], [False, False], 100)
    recent = meter.report()["window"]
    assert recent["per_arm_seconds"][0]["waiting"] == 30
    assert recent["per_arm_seconds"][1]["processing"] == 30
    assert_conservation(recent)


def test_zero_time_reset_and_report_mutation_do_not_corrupt_accounting():
    meter = ArmUtilization(4)
    for _ in range(2):
        report = meter.report()
        for name in ("cumulative", "window"):
            assert_conservation(report[name])
            assert report[name]["mean_busy"] == 0
        meter.record(["idle"] * 4, [True] * 4, 0)
        assert meter.report() == report
    meter.record(["lift"] * 4, [False] * 4, 5)
    report = meter.report()
    report["window"]["per_arm_seconds"][0]["processing"] = 1_000_000
    assert meter.report()["window"]["per_arm_seconds"][0]["processing"] == 5
    meter.reset()
    assert meter.report()["cumulative"]["duration_seconds"] == 0
    assert_conservation(meter.report()["window"])


def test_window_history_stays_bounded_during_a_long_continuous_run():
    meter = ArmUtilization(4)
    # Changing one arm each 100 ms prevents accidental reliance on compression.
    for i in range(2000):
        meter.record(["approach" if i % 2 else "idle"] * 4, [False] * 4, 0.1)
    assert len(meter._segments) <= 302
    assert_conservation(meter.report()["cumulative"])
    assert_conservation(meter.report()["window"])
    # Repeating one category should retain only one segment after expiry.
    for _ in range(400):
        meter.record(["idle"] * 4, [True] * 4, 0.1)
    assert len(meter._segments) == 1
    assert meter.report()["window"]["mean_busy"] == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_invalid_sample_is_rejected_without_changing_existing_time(duration):
    meter = ArmUtilization(2)
    meter.record(["idle", "return"], [True, False], 1)
    previous = meter.report()
    with pytest.raises(ValueError):
        meter.record(["idle", "return"], [True, False], duration)
    assert meter.report() == previous


def test_unclassified_phase_or_wrong_arm_count_cannot_silently_inflate_busy():
    meter = ArmUtilization(2)
    with pytest.raises(ValueError, match="Unclassified"):
        meter.record(["new_phase", "idle"], [False, True], 1)
    with pytest.raises(ValueError, match="match num_arms"):
        meter.record(["idle"], [True], 1)
    assert meter.elapsed == 0


def test_all_controller_phases_have_an_explicit_category():
    from dual_arm_conveyor.env import PHASES

    meter = ArmUtilization(len(PHASES))
    meter.record(PHASES, [False] * len(PHASES), 1)
    assert_conservation(meter.report()["cumulative"])


def test_environment_records_each_physics_step_and_reset_clears_timers(monkeypatch):
    from dual_arm_conveyor.config import Config
    from dual_arm_conveyor.env import ConveyorEnv

    env = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1))
    try:
        # WAIT leaves both robots idle while the actual belt advances.
        _, _, _, _, metrics = env.step(env.wait_action)
        assert metrics["arm_utilization"]["cumulative"]["mean_busy"] == 0

        def controlled_states():
            env.phase[:] = ["return", "recover"]
            env.hold_requested[:] = [False, True]

        monkeypatch.setattr(env, "_control", controlled_states)
        _, _, _, _, metrics = env.step(env.wait_action)
        report = metrics["arm_utilization"]
        assert report == env.utilization_report()
        assert report["cumulative"]["duration_seconds"] == pytest.approx(env.data.time)
        assert report["cumulative"]["per_arm_busy"] == pytest.approx([0.5, 0.5])
        assert report["cumulative"]["per_arm_seconds"][0]["return"] == pytest.approx(0.1)
        assert report["cumulative"]["per_arm_seconds"][1]["recovery"] == pytest.approx(0.1)
        assert_conservation(report["cumulative"])
        env.reset(seed=52)
        assert env.utilization_report()["cumulative"]["duration_seconds"] == 0
        assert env.metrics()["arm_utilization"]["window"]["per_arm_busy"] == [0, 0]
    finally:
        env.close()
