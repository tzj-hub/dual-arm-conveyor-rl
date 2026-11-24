"""Deep bins retain stored bodies and judge drops against their real floor."""

import mujoco
import numpy as np
import pytest

from dual_arm_conveyor.config import Config
from dual_arm_conveyor.env import ConveyorEnv, State


@pytest.fixture
def env():
    instance = ConveyorEnv(
        Config(
            deep_bins=True,
            auto_unload=False,
            num_arms=4,
            strategy="adaptive",
            sensor_mode="oracle",
            total_objects=5,
            pool_size=16,
            bin_capacity=24,
        ),
        seed=14,
    )
    yield instance
    instance.close()


def place(env, state, kind=0, actual_bin=0, height=-0.30):
    env.states[0] = state
    env.serial[0] = 0
    env.types[0] = kind
    env.owner[0] = 0
    q = env.ids.cube_qpos[0]
    env.data.qpos[q : q + 3] = [*env.ids.bin_centers[actual_bin, :2], height]
    env.data.qvel[env.ids.cube_dofs[0] : env.ids.cube_dofs[0] + 6] = 0
    mujoco.mj_forward(env.model, env.data)


@pytest.mark.parametrize("state", [State.HELD, State.RELEASED])
def test_intentional_deep_release_does_not_temporarily_count_as_dropped(env, state):
    place(env, state)
    env.phase[0] = "release"
    env._outcomes()
    assert env.states[0] == state
    assert env.metrics()["dropped"] == 0


def test_unintended_low_held_object_is_still_a_failed_grasp(env):
    place(env, State.HELD)
    env.phase[0] = "transport"
    env._outcomes()
    assert env.states[0] == State.DROPPED


@pytest.mark.parametrize("state", [State.SORTED, State.DROPPED])
def test_stored_or_misplaced_body_is_not_recycled_from_deep_bin(env, state):
    place(env, state, kind=0, actual_bin=1 if state == State.DROPPED else 0)
    before = env.data.qpos.copy()
    env._recycle()
    np.testing.assert_array_equal(env.data.qpos, before)
    assert env.states[0] == state and env.serial[0] == 0
    assert env.recycled == 0


def test_wrong_bin_remains_failure_and_is_not_rescored_as_success(env):
    place(env, State.RELEASED, kind=0, actual_bin=1)
    env.phase[0] = "release"
    env._outcomes()
    assert env.states[0] == State.DROPPED
    env._recycle()
    assert env.serial[0] == 0


def test_deep_surface_settling_can_be_counted_as_sorted(env):
    place(env, State.RELEASED, height=env.ids.bin_floor_z + 0.03)
    env.cube_settle[0] = env.config.settle_time
    env._outcomes()
    assert env.states[0] == State.SORTED
