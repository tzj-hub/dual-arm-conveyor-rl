"""Settling oscillations do not create success reward or hide a final loss."""
import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv, State


@pytest.fixture
def env():
    e = ConveyorEnv(Config(sensor_mode="oracle", total_objects=5, pool_size=16,
                           strategy="adaptive", motion_mode="optimized",
                           num_arms=4, deep_bins=True, auto_unload=False,
                           bin_capacity=24, stable_completion_reward=True))
    yield e
    e.close()


def place(e, state, *, serial=0, speed=0.0, outside=False):
    e.states[0] = state
    e.serial[0] = serial
    e.types[0] = 0
    e.owner[0] = 0
    q = e.ids.cube_qpos[0]
    position = np.r_[e.ids.bin_centers[0, :2], e.ids.bin_floor_z + 0.03]
    if outside:
        position[1] += 0.6
    e.data.qpos[q:q+3] = position
    dof = e.ids.cube_dofs[0]
    e.data.qvel[dof:dof+6] = 0
    e.data.qvel[dof] = speed
    e.cube_settle[0] = e.config.settle_time
    mujoco.mj_forward(e.model, e.data)


def test_settled_bounce_resettle_is_one_credit_and_final_loss_revokes_once(env):
    place(env, State.RELEASED)
    env._outcomes()
    assert env.states[0] == State.SORTED
    assert env.first_completion_rewards == 1
    place(env, State.SORTED, speed=0.4)
    env._outcomes()
    assert env.states[0] == State.RELEASED
    assert env.first_completion_rewards == 1 and env.revoked_completion_rewards == 0
    place(env, State.RELEASED)
    env._outcomes()
    assert env.states[0] == State.SORTED and env.first_completion_rewards == 1
    place(env, State.SORTED, outside=True)
    env._outcomes()
    assert env.states[0] == State.DROPPED
    assert env.metrics()["dropped"] == 1 and env.metrics()["sorted"] == 0
    assert env.revoked_completion_rewards == 1
    env._outcomes()
    assert env.revoked_completion_rewards == 1


def test_recycled_slot_new_serial_gets_its_own_credit(env):
    place(env, State.RELEASED, serial=7)
    env._outcomes()
    place(env, State.SORTED, serial=7, outside=True)
    env._outcomes()
    place(env, State.RELEASED, serial=8)
    env._outcomes()
    assert env.first_completion_rewards == 2
    assert env.revoked_completion_rewards == 1
    assert env.rewarded_completion_serial[0] == 8
    assert len(env.rewarded_completion_serial) == env.config.n_objects


def test_true_loss_penalty_and_credit_revocation_reach_step_reward(env, monkeypatch):
    # Establish an already credited, physically stable outcome.
    place(env, State.RELEASED)
    env._outcomes()
    original_outcomes = env._outcomes
    changed = False

    def force_one_real_final_loss():
        nonlocal changed
        if not changed:
            place(env, State.SORTED, outside=True)
            changed = True
        original_outcomes()

    monkeypatch.setattr(env, "_outcomes", force_one_real_final_loss)
    _, reward, _, _, metrics = env.step(env.wait_action)
    assert metrics["dropped"] == 1
    assert reward == pytest.approx(-8.001)
    _, reward2, _, _, _ = env.step(env.wait_action)
    assert reward2 == pytest.approx(-0.001)


def test_phase_duration_history_is_measured_per_arm_and_bounded(env):
    for n in range(140):
        env._set_phase(0, "approach")
        env.data.time += 0.2 + n * 0.001
        env._set_phase(0, "idle")
    history = env.phase_duration_samples[0]["approach"]
    assert len(history) == 128
    assert list(history)[-1] == pytest.approx(0.339)
    assert not env.phase_duration_samples[1]["approach"]
