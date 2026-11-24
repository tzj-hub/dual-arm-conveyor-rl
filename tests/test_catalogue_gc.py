"""Collect retired model cycles only at successful catalogue boundaries."""
import gc
import weakref

import pytest

from dual_arm_conveyor import hierarchical_train as train


def test_catalogue_gc_occurs_after_successful_changed_reset_only(monkeypatch):
    events = []
    class Env:
        def __init__(self, config):
            self.catalog = config.seed
        def reset(self, seed=None):
            events.append(('reset', self.catalog, seed))
            if seed == 1003:
                raise RuntimeError('reset failed')
            return ('observation', seed)
        def close(self):
            events.append(('close', self.catalog))
    monkeypatch.setattr(train.gc, 'collect', lambda generation: events.append(('gc', generation)) or 7)
    config = train.scene_config(train.parser().parse_args([]))
    bank = train.CatalogBankEnv(config, [801, 802], 1000, Env)
    assert bank.reset(1000) == ('observation', 1000)
    assert ('gc', 2) not in events
    assert bank.reset(1001) == ('observation', 1001)
    assert events[-2:] == [('reset', 802, 1001), ('gc', 2)]
    assert bank.catalogue_gc_events[0]['collected'] == 7
    bank.reset(1005)  # Same catalogue, no retirement.
    assert events.count(('gc', 2)) == 1
    bank.reset(1002)
    with pytest.raises(RuntimeError, match='reset failed'):
        bank.reset(1003)
    assert events.count(('gc', 2)) == 2
    assert len(bank.catalogue_gc_events) == 2


def test_retired_reference_cycle_is_collected_and_live_env_is_preserved():
    references = []
    class Env:
        def __init__(self, config):
            self.cycle = {'env': self}
            references.append(weakref.ref(self))
        def reset(self, seed=None):
            return seed
        def close(self):
            pass
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        config = train.scene_config(train.parser().parse_args([]))
        bank = train.CatalogBankEnv(config, [801, 802], 1000, Env)
        bank.reset(1000)
        bank.reset(1001)
        assert references[0]() is None
        assert references[1]() is bank.env
        assert len(bank.catalogue_gc_events) == 1
    finally:
        if was_enabled:
            gc.enable()


def test_sequence_and_actor_context_are_explicit_cli_options():
    defaults = train.parser().parse_args([])
    assert defaults.teacher_mode == 'returning_partner_soft_preference'
    assert defaults.actor_context == 'independent'
    args = train.parser().parse_args(['--teacher-mode', 'sequence', '--actor-context', 'set'])
    assert args.teacher_mode == 'sequence' and args.actor_context == 'set'
