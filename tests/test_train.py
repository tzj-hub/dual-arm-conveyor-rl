"""Verify that fast oracle training cannot masquerade as camera deployment."""

import json
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from dual_arm_conveyor import train


def test_training_default_is_explicit_oracle():
    args = train.parser().parse_args([])
    assert args.sensor_mode == "oracle"
    assert args.spacing == 0.45
    assert args.eval_seed >= 20000


def test_actual_training_reset_range_cannot_leak_into_final_seeds():
    assert train.training_seed_range(10, 8, [100, 101]) == {
        "first": 10,
        "last": 18,
        "reset_count": 8,
    }
    with pytest.raises(ValueError, match="Actual training seeds overlap"):
        train.training_seed_range(10, 8, [18, 19])


def test_training_selects_on_oracle_but_final_report_runs_camera(tmp_path, monkeypatch):
    instances = []

    class FixtureEnv:
        """Small optimizer fixture; never used to report physical performance."""

        def __init__(self, config):
            self.config = config
            self.closed = False
            self.seeds = []
            instances.append(self)

        def reset(self, seed=None):
            self.seeds.append(seed)
            self.steps = 0
            return {
                "features": np.array([[0, 1], [1, 0]], dtype=np.float32),
                "global": np.array([0], dtype=np.float32),
                "mask": np.array([True, True]),
            }

        def heuristic_action(self):
            return 1

        def step(self, action):
            self.steps += 1
            observation = {
                "features": np.array([[0, 1], [1, 0]], dtype=np.float32),
                "global": np.array([0], dtype=np.float32),
                "mask": np.array([True, True]),
            }
            return (
                observation,
                1.0,
                self.steps == 2,
                False,
                {"fixture": True, "sensor_mode": self.config.sensor_mode},
            )

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules, "dual_arm_conveyor.env", SimpleNamespace(ConveyorEnv=FixtureEnv)
    )
    train.main(
        [
            "--output",
            str(tmp_path),
            "--steps",
            "8",
            "--warmstart-steps",
            "0",
            "--rollout-steps",
            "4",
            "--epochs",
            "1",
            "--minibatch-size",
            "4",
            "--hidden-dim",
            "8",
            "--validation-every",
            "0",
            "--eval-episodes",
            "1",
            "--camera-hz",
            "6",
            "--camera-width",
            "640",
            "--camera-height",
            "480",
            "--camera-latency",
            "0.12",
            "--depth-noise",
            "0.002",
            "--camera-dropout",
            "0.1",
            "--physics-dt",
            "0.0025",
        ]
    )
    assert [env.config.sensor_mode for env in instances] == [
        "oracle",
        "oracle",
        "vision",
    ]
    assert all(env.closed for env in instances)
    assert all(
        env.config.camera_hz == 6
        and env.config.camera_latency == 0.12
        and env.config.camera_width == 640
        and env.config.camera_height == 480
        and env.config.depth_noise == 0.002
        and env.config.camera_dropout == 0.1
        and env.config.physics_dt == 0.0025
        for env in instances
    )
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["sensor_mode"] == report["env_config"]["sensor_mode"] == "vision"
    assert report["training_env_config"]["sensor_mode"] == "oracle"
    for policy in ("learned", "heuristic"):
        assert report[policy]["episodes"][0]["info"]["sensor_mode"] == "vision"
    metadata = json.loads((tmp_path / "training.json").read_text())
    assert metadata["deployment_evaluation_completed"]
    assert not set(instances[0].seeds) & set(metadata["test_seeds"])
    assert not set(metadata["validation_seeds"]) & set(metadata["test_seeds"])
