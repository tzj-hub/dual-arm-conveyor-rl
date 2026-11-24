"""Controller opt-in retains the original physical scene and model interface."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from dual_arm_conveyor.config import Config


def baseline_config():
    root = Path(__file__).resolve().parents[1]
    return Config(**json.loads((root / 'tests/fixtures/baseline_scene.json').read_text()))


def test_cooperation_is_explicit_and_leaves_supply_and_geometry_unchanged():
    baseline = baseline_config()
    assert baseline.cooperative_motion is False
    assert baseline.cooperative_drop_cost is False
    assert baseline.downstream_completion is False
    changed = replace(baseline, cooperative_motion=True)
    assert {key for key, value in asdict(changed).items()
            if value != asdict(baseline)[key]} == {'cooperative_motion'}


@pytest.mark.parametrize('changes', [dict(num_arms=2), dict(strategy='baseline'),
                                    dict(gripper_mode='vacuum'), dict(deep_bins=False),
                                    dict(motion_mode='legacy')])
def test_cooperation_requires_calibrated_four_arm_scene(changes):
    # Isolate this feature's validation from the baseline's other opt-in flags.
    with pytest.raises(ValueError):
        Config(num_arms=changes.get('num_arms', 4), strategy=changes.get('strategy', 'adaptive'),
               gripper_mode=changes.get('gripper_mode', 'fingers'),
               deep_bins=changes.get('deep_bins', True), auto_unload=False,
               motion_mode=changes.get('motion_mode', 'optimized'), cooperative_motion=True)


def test_cooperation_rejects_non_boolean_setting():
    with pytest.raises(ValueError, match='cooperative_motion must be bool'):
        replace(baseline_config(), cooperative_motion=1)


@pytest.mark.parametrize('option', ['nearest_bin_slot', 'direct_retask'])
@pytest.mark.parametrize('flag', ['cooperative_motion', 'cooperative_drop_cost'])
def test_cooperation_rejects_disabled_controller_dependencies(option, flag):
    with pytest.raises(ValueError, match=f'{flag} requires'):
        replace(baseline_config(), **{flag: True, option: False})


def test_downstream_completion_is_independent_of_return_and_drop_experiments():
    baseline = baseline_config()
    changed = replace(baseline, downstream_completion=True)
    assert {key for key, value in asdict(changed).items()
            if value != asdict(baseline)[key]} == {'downstream_completion'}
    assert not changed.cooperative_motion and not changed.cooperative_drop_cost
    with pytest.raises(ValueError, match='downstream_completion must be bool'):
        replace(baseline, downstream_completion=1)
