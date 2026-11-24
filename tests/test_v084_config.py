"""Keep training and playback controller contracts aligned."""
from dataclasses import replace

import pytest

from dual_arm_conveyor import cli, hierarchical_train


def test_v084_options_reach_training_and_playback_with_same_meaning():
    flags = ['--active-recovery', '--future-planning', '--safe-preposition', '--stable-completion-reward']
    trained = hierarchical_train.scene_config(hierarchical_train.parser().parse_args(flags))
    deployed = cli.config_from_args(cli.parser().parse_args([
        'evaluate', '--num-arms', '4', '--strategy', 'adaptive', '--deep-bins',
        '--no-auto-unload', '--objects', '100', '--pool-size', '384', '--bin-capacity', '180', *flags,
    ]))
    for name in ('active_recovery', 'future_planning', 'safe_preposition', 'stable_completion_reward'):
        assert getattr(trained, name) is True
        assert getattr(deployed, name) is True


def test_preposition_requires_a_planner_and_physical_finger_scene():
    base = hierarchical_train.scene_config(hierarchical_train.parser().parse_args([]))
    with pytest.raises(ValueError, match='requires future_planning'):
        replace(base, safe_preposition=True)
    with pytest.raises(ValueError, match='four adaptive finger arms'):
        replace(base, future_planning=True, gripper_mode='vacuum')
    with pytest.raises(ValueError, match='planned_return requires future_planning'):
        replace(base, planned_return=True)
    with pytest.raises(ValueError, match='planned_return requires direct_retask'):
        replace(base, planned_return=True, future_planning=True)
    planned = replace(base, planned_return=True, future_planning=True, direct_retask=True)
    assert planned.planned_return and not planned.safe_preposition
