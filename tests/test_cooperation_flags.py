"""Independent controller switches; stubs never construct physics or train."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from dual_arm_conveyor import cli, hierarchical_train
from dual_arm_conveyor.config import Config


OPTIONS = ('cooperative_motion', 'cooperative_drop_cost', 'downstream_completion')


def base_config():
    root = Path(__file__).resolve().parents[1]
    return Config(**json.loads((root / 'tests/fixtures/baseline_scene.json').read_text()))


@pytest.mark.parametrize('selected', [(), (OPTIONS[0],), (OPTIONS[1],), (OPTIONS[2],), OPTIONS])
def test_flags_reach_playback_and_training_independently(selected):
    flags = ['--' + name.replace('_', '-') for name in selected]
    common = ['--direct-retask', '--nearest-bin-slot']
    trained = hierarchical_train.scene_config(hierarchical_train.parser().parse_args(common + flags))
    played = cli.config_from_args(cli.parser().parse_args([
        'play', '--num-arms', '4', '--strategy', 'adaptive', '--deep-bins', '--no-auto-unload',
        *common, *flags,
    ]))
    for name in OPTIONS:
        assert getattr(trained, name) is (name in selected)
        assert getattr(played, name) is (name in selected)


def policy_fixture(monkeypatch, config, trained_scene):
    action_metadata, observation_metadata = {'schema': 'fixed_action'}, {'schema': 'fixed_observation'}
    metadata = dict(env_config=trained_scene, scheduler_action_metadata=action_metadata,
                    scheduler_observation_metadata=observation_metadata)
    trained = SimpleNamespace(metadata=metadata)
    monkeypatch.setitem(sys.modules, 'dual_arm_conveyor.learning',
                        SimpleNamespace(load_policy=lambda path: trained))
    monkeypatch.setitem(sys.modules, 'dual_arm_conveyor.scheduler', SimpleNamespace(
        action_metadata=lambda env: action_metadata,
        observation_metadata=lambda env: observation_metadata,
        model_policy=lambda env, learned: lambda obs: 123,
    ))
    return SimpleNamespace(config=config), metadata


@pytest.mark.parametrize('selected', [(), (OPTIONS[0],), (OPTIONS[1],), (OPTIONS[2],), OPTIONS])
def test_old_checkpoint_only_transfers_explicit_switches(monkeypatch, capsys, selected):
    old_scene = asdict(base_config())
    for name in OPTIONS:
        old_scene.pop(name)
    config = replace(base_config(), **{name: name in selected for name in OPTIONS})
    env, _ = policy_fixture(monkeypatch, config, old_scene)
    assert cli._policy_action('stub.pt', env)({}) == 123
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [m['option'] for m in messages] == list(selected)
    assert all(m['event'] == 'controller_transfer' and not m['weights_updated']
               and m['shared_zone_lock_retained'] for m in messages)


@pytest.mark.parametrize('option', OPTIONS)
def test_trained_switch_cannot_be_silently_disabled(monkeypatch, option):
    config = base_config()
    env, _ = policy_fixture(monkeypatch, config, dict(asdict(config), **{option: True}))
    with pytest.raises(ValueError, match='requires --' + option.replace('_', '-')):
        cli._policy_action('stub.pt', env)


def test_explicit_switch_does_not_bypass_scheduler_schema(monkeypatch):
    base = base_config()
    env, metadata = policy_fixture(monkeypatch, replace(base, cooperative_motion=True), asdict(base))
    metadata['scheduler_action_metadata'] = {'schema': 'incompatible'}
    with pytest.raises(ValueError, match='compatible four-arm scheduler'):
        cli._policy_action('stub.pt', env)
