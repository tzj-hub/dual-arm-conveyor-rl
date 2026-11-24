"""Report integrity tests independent of a lengthy physics rollout."""

from dataclasses import dataclass
import json
import itertools
import math
import sys
from types import SimpleNamespace

import numpy as np

import pytest

from dual_arm_conveyor import cli


def test_invalid_scene_arguments_are_rejected():
    for arguments in (
        ["evaluate", "--spacing", "nan"],
        ["evaluate", "--episodes", "0"],
        ["play", "--belt-speed", "-0.1"],
        ["evaluate", "--arrival-jitter", "-1"],
        ["play", "--sensor-mode", "ground_truth"],
        ["evaluate", "--camera-dropout", "1"],
        ["evaluate", "--camera-width", "63"],
        ["play", "--camera-hz", "nan"],
        ["play", "--physics-dt", "0"],
        ["play", "--duration", "nan"],
        ["play", "--pool-size", "0"],
        ["play", "--motion-mode", "fastest"],
        ["evaluate", "--continuous", "--objects", "50"],
    ):
        with pytest.raises(SystemExit):
            cli.parser().parse_args(arguments)


def test_episode_aggregation_does_not_treat_missing_metrics_as_zero():
    result = cli.aggregate_episodes(
        [
            {"seed": 100, "sorted": 8, "terminated": True, "optional": 12},
            {"seed": 101, "sorted": 10, "terminated": False},
        ]
    )
    assert "seed" not in result
    assert "optional" not in result
    assert result["sorted"] == {"mean": 9.0, "min": 8, "max": 10, "std": 1.0}
    assert result["terminated"]["mean"] == 0.5


@pytest.mark.parametrize("policy", ["heuristic", "predictive"])
def test_evaluation_records_separate_episodes_and_timeout(
    tmp_path, monkeypatch, capsys, policy
):
    from dual_arm_conveyor.objects import make_object_catalog

    @dataclass
    class ExampleConfig:
        total_objects: int = 10
        sensor_mode: str = "oracle"
        seed: int = 0

    class ExampleEnv:
        config = ExampleConfig()
        ids = SimpleNamespace(object_specs=make_object_catalog(10, seed=0))

        def __init__(self):
            self.seeds = []
            self.closed = False

        def reset(self, seed):
            self.seeds.append(seed)
            self.steps = 0
            return {}

        def heuristic_action(self):
            return 0

        def step(self, action):
            self.steps += 1
            return {}, 2.0, False, self.steps == 2, {}

        def metrics(self):
            return {"sorted": self.steps, "missed": 8}

        def close(self):
            self.closed = True

    env = ExampleEnv()
    monkeypatch.setattr(cli, "_make_env", lambda args: env)
    monkeypatch.setattr(cli, "_policy_action", lambda name, env: lambda observation: 0)
    fake_wall_clock = itertools.count(0, 8)
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(fake_wall_clock))
    path = tmp_path / "new" / "evaluation.json"
    args = cli.parser().parse_args(
        [
            "evaluate",
            "--policy", policy,
            "--sensor-mode",
            "oracle",
            "--episodes",
            "2",
            "--seed",
            "123",
            "--output",
            str(path),
        ]
    )
    report = cli.evaluate_command(args)
    persisted = json.loads(path.read_text())
    assert env.seeds == [123, 124]
    assert env.closed
    assert persisted == report
    assert [episode["episode_return"] for episode in report["episodes"]] == [4.0, 4.0]
    assert all(
        episode["truncated"] and not episode["terminated"]
        for episode in report["episodes"]
    )
    assert report["aggregate"]["missed"]["mean"] == 8
    assert report["sensor_mode"] == "oracle"
    assert "baseline" in report["purpose"]
    assert report["schema_version"] == 3
    assert report["checkpoint_sha256"] is None
    assert report["catalog_seed"] == 0
    assert report["catalog_objects"] == 10
    assert len(report["catalog_sha256"]) == 64
    progress = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"event": "progress"')
    ]
    assert [item["seed"] for item in progress] == [123, 124]
    assert all(item["wall_seconds"] >= 15 for item in progress)


def test_catalogue_hash_changes_with_physics_parameters_not_episode_seed():
    from dataclasses import replace
    from dual_arm_conveyor.config import Config
    from dual_arm_conveyor.objects import make_object_catalog

    specs = make_object_catalog(4, seed=0)
    env = SimpleNamespace(
        config=Config(), ids=SimpleNamespace(object_specs=specs), seed=100
    )
    first = cli.catalog_metadata(env)
    env.seed = 200
    assert cli.catalog_metadata(env) == first
    env.ids.object_specs = (replace(specs[0], mass=specs[0].mass + 0.001), *specs[1:])
    assert cli.catalog_metadata(env)["catalog_sha256"] != first["catalog_sha256"]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["play"], (None, None)),
        (["play", "--record", "demo.mp4"], (None, 120.0)),
        (["play", "--duration", "40"], (None, 40.0)),
        (["play", "--objects", "150"], (150, None)),
        (["play", "--objects", "150", "--duration", "60"], (150, 60.0)),
        (["evaluate"], (50, None)),
        (["evaluate", "--continuous"], (None, 300.0)),
        (["evaluate", "--continuous", "--duration", "90"], (None, 90.0)),
        (["evaluate", "--objects", "100"], (100, None)),
    ],
)
def test_streaming_and_finite_run_limits(arguments, expected):
    assert cli.run_limits(cli.parser().parse_args(arguments)) == expected


def test_streaming_scene_options_reach_config(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "dual_arm_conveyor.env",
        SimpleNamespace(ConveyorEnv=lambda config: config),
    )
    args = cli.parser().parse_args(
        [
            "play",
            "--pool-size",
            "32",
            "--bin-capacity",
            "8",
            "--no-auto-unload",
            "--motion-mode",
            "legacy",
        ]
    )
    config = cli._make_env(args)
    assert config.total_objects is None and math.isinf(config.horizon)
    assert config.pool_size == 32 and config.bin_capacity == 8
    assert not config.auto_unload and config.motion_mode == "legacy"
    assert (config.camera_width, config.camera_height) == (480, 360)


def test_training_always_uses_a_finite_default_batch():
    from dual_arm_conveyor.train import parser

    args = parser().parse_args([])
    assert args.total_objects == 30
    assert args.pool_size == 64
    with pytest.raises(SystemExit):
        parser().parse_args(["--total-objects", "0"])


def test_deployment_commands_default_to_camera_randomized_arrivals_and_yaw():
    for command in ("evaluate", "play"):
        args = cli.parser().parse_args([command])
        assert args.sensor_mode == "vision"
        assert args.spacing == pytest.approx(0.45)
        assert args.arrival_jitter == pytest.approx(1.2)
        assert args.yaw_range == pytest.approx(math.pi)


def test_sensor_stress_options_reach_environment_config(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "dual_arm_conveyor.env",
        SimpleNamespace(ConveyorEnv=lambda config: config),
    )
    args = cli.parser().parse_args(
        [
            "evaluate",
            "--camera-hz",
            "5",
            "--camera-width",
            "640",
            "--camera-height",
            "480",
            "--camera-latency",
            "0.2",
            "--depth-noise",
            "0.003",
            "--camera-dropout",
            "0.15",
            "--physics-dt",
            "0.0025",
        ]
    )
    config = cli._make_env(args)
    assert config.camera_hz == 5
    assert (config.camera_width, config.camera_height) == (640, 480)
    assert config.camera_latency == 0.2
    assert config.depth_noise == 0.003
    assert config.camera_dropout == 0.15
    assert config.physics_dt == 0.0025


@pytest.mark.parametrize("old_schema", [None, 1, 2, 3])
def test_cli_rejects_old_policy_even_if_torch_architecture_loads(
    monkeypatch, old_schema
):
    from dual_arm_conveyor import learning

    old_policy = SimpleNamespace(
        metadata={"env_config": {"schema_version": old_schema, "sensor_mode": "vision"}}
    )
    monkeypatch.setattr(learning, "load_policy", lambda path: old_policy)
    env = SimpleNamespace(config=SimpleNamespace(schema_version=4))
    with pytest.raises(ValueError, match="current scene schema"):
        cli._policy_action("old_v1.pt", env)


def test_recording_panel_only_uses_delivered_detector_overlay():
    pytest.importorskip("PIL")
    delivered = np.full((24, 32, 3), [13, 99, 201], dtype=np.uint8)
    sensor = SimpleNamespace(
        overlay=delivered, diagnostics=lambda: {"frame_age_seconds": 0.15}
    )
    env = SimpleNamespace(
        sensor=sensor,
        config=SimpleNamespace(sensor_mode="vision"),
        data=SimpleNamespace(time=1.0),
    )
    overview = np.full((240, 320, 3), 45, dtype=np.uint8)
    frame = cli._compose_camera_panel(overview, env)
    assert frame.shape == (240, 640, 3)
    np.testing.assert_array_equal(frame[100, 100], overview[100, 100])
    assert np.any(np.all(frame[:, 320:] == [13, 99, 201], axis=-1))
    np.testing.assert_array_equal(
        delivered, np.full((24, 32, 3), [13, 99, 201], dtype=np.uint8)
    )


def test_recording_panel_labels_aggregate_status_separately_from_camera(monkeypatch):
    pytest.importorskip("PIL")
    from PIL import ImageDraw

    labels = []
    original_text = ImageDraw.ImageDraw.text

    def capture_text(draw, position, text, *args, **kwargs):
        labels.append(text)
        return original_text(draw, position, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    delivered = np.full((24, 32, 3), [13, 99, 201], dtype=np.uint8)
    env = SimpleNamespace(
        config=SimpleNamespace(sensor_mode="vision"),
        data=SimpleNamespace(time=12.0),
        sensor=SimpleNamespace(overlay=delivered, diagnostics=lambda: {}),
        phase=["recover", "transfer_wait"],
        metrics=lambda: {
            "total_objects": 100,
            "sorted": 80,
            "missed": 12,
            "dropped": 3,
            "unfinished": 5,
            "bin_unloads": [3, 4],
        },
    )
    frame = cli._compose_camera_panel(np.zeros((720, 640, 3), dtype=np.uint8), env)
    assert frame.shape == (720, 960, 3)
    assert "RUN STATUS (EVALUATOR COUNTS)" in labels
    assert "Spawned: 100  sorted: 80" in labels
    assert "Missed: 12  dropped: 3" in labels
    assert "Unfinished: 5  unloads: 3/4" in labels
    assert "Arms: recover / transfer_wait" in labels
    assert np.any(np.all(frame[:, 640:] == [13, 99, 201], axis=-1))


def test_play_progress_uses_fifteen_second_wall_intervals(monkeypatch, capsys):
    clock = iter([14.9, 15.0, 29.9, 30.0])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    env = SimpleNamespace(
        phase=["idle", "approach"],
        metrics=lambda: {
            "time": 4.0,
            "total_objects": 2,
            "sorted": 1,
            "missed": 0,
            "dropped": 0,
            "unfinished": 1,
            "bin_unloads": [0, 0],
        },
    )
    last = 0.0
    for _ in range(4):
        last = cli._play_progress(env, 0.0, last)
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["wall_seconds"] for item in messages] == [15.0, 30.0]
    assert all(
        item["command"] == "play" and item["unfinished"] == 1 for item in messages
    )
    assert messages[-1]["phase"] == ["idle", "approach"]


def test_progress_preserves_environment_phase_ages_and_separate_recovery_counts():
    env = SimpleNamespace(
        data=SimpleNamespace(time=18.0),
        phase=["recover", "idle"],
        target=np.array([2, -1]),
        holding=np.array([False, False]),
        hold_requested=np.array([True, False]),
        escape_routes=np.array([False, False]),
        route_retry_time=np.array([18.2, 17.0]),
        routes=[
            SimpleNamespace(verified=False, reason="initial clearance: link / belt"),
            None,
        ],
    )
    metrics = {
        "current_phases": ["recover", "idle"],
        "current_phase_elapsed_seconds": [13.0, 2.0],
        "max_phase_duration_seconds": {"recover": 13.0, "idle": 5.0},
        "home_returns": [0, 0],
        "recovery_home_returns": [1, 0],
        "escape_attempts": 3,
        "escape_successes": 1,
        "release_timeouts": 2,
        "recovery_waiting_for_release": [True, False],
        "arm_assignments": [4, 6],
        "arm_sorted": [2, 5],
        "arm_assignments_since_last_sort": [2, 0],
        "max_arm_assignments_without_sort": [3, 2],
        "arm_seconds_since_last_sort": [10.0, 2.4],
    }
    result = cli.progress_counters(env, metrics)
    assert result["current_phase_elapsed_seconds"] == [13.0, 2.0]
    assert result["max_phase_duration_seconds"]["recover"] == 13.0
    assert result["home_returns"] == [0, 0]
    assert result["recovery_home_returns"] == [1, 0]
    assert result["escape_successes"] == 1
    assert result["release_timeouts"] == 2
    assert result["recovery_waiting_for_release"] == [True, False]
    assert result["arm_assignments_since_last_sort"] == [2, 0]
    assert result["max_arm_assignments_without_sort"] == [3, 2]
    assert result["arm_seconds_since_last_sort"] == [10.0, 2.4]
    assert result["controller_status"]["hold_requested"] == [True, False]
    assert result["controller_status"]["target_slots"] == [2, -1]
    assert result["controller_status"]["route_retry_in_seconds"] == pytest.approx(
        [0.2, 0]
    )
    assert "link / belt" in result["controller_status"]["route_rejection_reasons"][0]
    assert (
        "phase_deadline_seconds" not in result
    )  # reporting does not invent a control timeout


def test_recording_sidecar_preserves_starting_source_hashes_and_runtime(
    tmp_path, monkeypatch
):
    from dual_arm_conveyor.config import Config
    from dual_arm_conveyor.objects import make_object_catalog

    env = SimpleNamespace(
        config=Config(sensor_mode="oracle"),
        ids=SimpleNamespace(object_specs=make_object_catalog(2)),
        metrics=lambda: {"time": 1.0, "total_objects": 1, "sorted": 0, "unfinished": 1},
    )
    args = SimpleNamespace(
        record=tmp_path / "demo.mp4", policy="heuristic", sensor_mode="oracle", seed=100
    )
    monkeypatch.setattr(cli.time, "monotonic", lambda: 15.0)
    loaded_source = {"env.py": "hash-at-playback-start"}
    cli._finish_play(args, env, started=5.0, source_sha256=loaded_source)
    report = json.loads(args.record.with_suffix(".json").read_text())
    assert report["source_sha256"] == loaded_source
    assert report["runtime"]["python"] and report["runtime"]["mujoco"]
    assert report["runtime"]["wall_seconds"] == report["wall_seconds"] == 10.0
    assert report["catalog_seed"] == 0


def test_recording_default_is_thirty_distinct_physics_states_per_simulation_second():
    assert cli.parser().parse_args(["play"]).fps == 30
    clock = cli.FrameClock(30)
    for step in range(201):
        timestamp = step * 0.005
        if clock.due(timestamp):
            clock.captured(timestamp, 0.002)
    report = clock.report(2.0, 1.0)
    assert report["physics_states_rendered"] == 31
    assert report["captured_frames_per_simulation_second"] == pytest.approx(30)
    assert report["duplicated_state_frames"] == 0
    assert report["skipped_simulation_frame_deadlines"] == 0
    assert report["maximum_frame_simulation_interval"] <= 0.035 + 1e-10
    assert report["real_time_factor"] == 0.5
    assert report["first_frame_render_and_output_seconds"] == 0.002
    assert report["mean_subsequent_frame_render_and_output_seconds"] == pytest.approx(
        0.002
    )


def test_late_frame_is_not_duplicated_to_fill_missing_display_deadlines():
    clock = cli.FrameClock(30)
    for timestamp in (0.0, 0.1):
        assert clock.due(timestamp)
        clock.captured(timestamp, 0.001)
    assert not clock.due(0.1)
    report = clock.report(0.1, 0.1)
    assert report["physics_states_rendered"] == 2
    assert report["skipped_simulation_frame_deadlines"] == 2
    with pytest.raises(ValueError, match="newer physics state"):
        clock.captured(0.1, 0.001)


def test_physics_frame_callback_does_not_change_decisions_or_dynamics():
    from dual_arm_conveyor.config import Config
    from dual_arm_conveyor.env import ConveyorEnv

    first = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1), seed=50001)
    second = ConveyorEnv(Config(sensor_mode="oracle", total_objects=1), seed=50001)
    clock = cli.FrameClock(30)
    callbacks = []

    def capture(env):
        callbacks.append(float(env.data.time))
        if clock.due(float(env.data.time)):
            env.metrics()  # The video panel reads counters at the display cadence.
            clock.captured(float(env.data.time), 0.0)

    try:
        clock.due(0.0)
        clock.captured(0.0, 0.0)
        for _ in range(30):
            action = first.heuristic_action()
            assert action == second.heuristic_action()
            expected = first.step(action)
            observed = second.step(action, physics_callback=capture)
            np.testing.assert_array_equal(first.data.qpos, second.data.qpos)
            np.testing.assert_array_equal(first.data.qvel, second.data.qvel)
            assert expected[1:4] == observed[1:4]
            assert expected[4] == observed[4]
        assert sum(first.metrics()["joint_travel_rad"]) > 0.1
        assert len(callbacks) == 600
        assert clock.frames == 91
        assert len(set(clock.timestamps)) == 91
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("rig", ["overhead", "dual", "triple"])
def test_camera_rig_option_reaches_scene_configuration(monkeypatch, rig):
    monkeypatch.setitem(
        sys.modules,
        "dual_arm_conveyor.env",
        SimpleNamespace(ConveyorEnv=lambda config: config),
    )
    args = cli.parser().parse_args(["play", "--camera-rig", rig])
    assert cli._make_env(args).camera_rig == rig
