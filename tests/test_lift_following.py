"""A sealed payload must clear the moving belt before its hand brakes."""

import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv
from dual_arm_conveyor.vision import Track


@pytest.fixture
def controlled_lift(monkeypatch):
    env = ConveyorEnv(
        Config(
            strategy="adaptive",
            num_arms=4,
            gripper_mode="vacuum",
            sensor_mode="oracle",
            belt_speed=0.3,
            control_speed=0.65,
            max_time=10,
            total_objects=1,
        )
    )
    env.target[0] = 0
    env.belief[0] = 2
    env.lift_xy[0] = env.data.site_xpos[env.ids.ee_sites[0], :2]
    env.lift_start_z[0] = env.data.site_xpos[env.ids.vacuum_sites[0], 2]
    env._set_phase(0, "lift")
    commands = []
    monkeypatch.setattr(env, "_belt_velocity", lambda: 0.32)
    monkeypatch.setattr(env, "_gripper_has_object", lambda arm: True)
    # Planner/servo tests cover geometry. Here independently inject a verified
    # route and measured seal so this test isolates the separation guard.
    monkeypatch.setattr(
        env, "_route_goal", lambda arm, goal, *a, **kw: (np.array(goal), True)
    )

    def command(data, arm, goal, **kwargs):
        if arm == 0:
            commands.append((np.array(goal), kwargs))

    monkeypatch.setattr(env.ik, "command", command)
    yield env, commands
    env.close()


def test_braking_waits_for_measured_clearance_and_is_gradual(controlled_lift):
    env, commands = controlled_lift
    original_z = float(env.lift_start_z[0])
    for time, rise in ((0.005, 0), (0.10, 0.024), (0.105, 0.026)):
        # Inject encoder-derived rise independently of elapsed time.
        env.lift_start_z[0] = original_z - rise
        env.data.time = time
        env._control()
        assert commands[-1][1]["feedforward"][0] == pytest.approx(0.32)
        assert env.phase[0] == "lift"
    env.data.time = 0.155
    env._control()
    braking_velocity = commands[-1][1]["feedforward"][0]
    assert 0 < braking_velocity < 0.32
    env.data.time = 0.31
    env._control()
    assert commands[-1][1]["feedforward"][0] == pytest.approx(0)
    assert not commands[-1][1].get("hold", False)


def test_stalled_lift_exits_without_spending_unbounded_downstream_reach(
    controlled_lift,
):
    env, commands = controlled_lift
    for _ in range(41):
        env.data.time += env.config.physics_dt
        env._control()
        if env.phase[0] == "recover":
            break
    assert env.phase[0] == "recover"
    assert env.data.time <= 0.205
    assert env.lift_displacement[0] <= 0.32 * 0.20 + 1e-9
    assert commands[-1][1]["hold"] is True
    assert commands[-1][1]["gripper"] == 0.04


def test_lift_displacement_budget_rejects_pick_too_close_to_workspace_end(
    controlled_lift,
):
    env, _ = controlled_lift
    track = Track(
        1, np.array([-0.4, 0, 0.23]), np.array([0.4, 0, 0]), 0, 0, 0.95, 0, 0, hits=8
    )
    high = env.windows[0, 1]
    # Only 31 cm remain: descending and sealing use up to 19.2 cm, and
    # clearance-gated braking can need another 12 cm before arm margin.
    assert not env._descent_feasible(0, track, np.array([high - 0.31, 0, 0.23]))
    assert env._descent_feasible(0, track, np.array([high - 0.40, 0, 0.23]))
