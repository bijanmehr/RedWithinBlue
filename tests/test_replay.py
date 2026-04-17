"""Tests for ReplayPlayer."""

import numpy as np
import pytest

from red_within_blue.replay import ReplayPlayer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _create_trajectory(tmp_path, num_steps=5):
    """Create a small .npz trajectory file and return its path."""
    save_dict = {}
    for step in range(num_steps):
        save_dict[f"step_{step}/obs/agent_0"] = np.random.rand(10).astype(np.float32)
        save_dict[f"step_{step}/obs/agent_1"] = np.random.rand(10).astype(np.float32)
        if step > 0:
            save_dict[f"step_{step}/actions/agent_0"] = np.int32(step % 5)
            save_dict[f"step_{step}/actions/agent_1"] = np.int32((step + 1) % 5)
            save_dict[f"step_{step}/rewards/agent_0"] = np.float32(step * 0.1)
            save_dict[f"step_{step}/rewards/agent_1"] = np.float32(step * 0.2)

    path = tmp_path / "test_traj.npz"
    np.savez_compressed(str(path), **save_dict)
    return str(path)


# ------------------------------------------------------------------
# 51. test_replay_load
# ------------------------------------------------------------------

def test_replay_load(tmp_path):
    """ReplayPlayer loads a trajectory and reports the correct step count."""
    num_steps = 5
    path = _create_trajectory(tmp_path, num_steps=num_steps)
    player = ReplayPlayer(path)

    assert player.num_steps == num_steps
    assert player.current_step == 0

    # All keys should be discoverable
    all_keys = player.get_all_keys()
    assert len(all_keys) > 0
    assert any("step_0" in k for k in all_keys)


# ------------------------------------------------------------------
# 52. test_replay_step_forward
# ------------------------------------------------------------------

def test_replay_step_forward(tmp_path):
    """step_forward advances the cursor and returns step data."""
    path = _create_trajectory(tmp_path, num_steps=4)
    player = ReplayPlayer(path)

    assert player.current_step == 0

    data = player.step_forward()
    assert player.current_step == 1
    assert "obs/agent_0" in data

    data = player.step_forward()
    assert player.current_step == 2

    # Advance to the end
    player.step_forward()
    assert player.current_step == 3

    # Should clamp at last step
    player.step_forward()
    assert player.current_step == 3


# ------------------------------------------------------------------
# 53. test_replay_step_back
# ------------------------------------------------------------------

def test_replay_step_back(tmp_path):
    """step_back moves the cursor backward and clamps at 0."""
    path = _create_trajectory(tmp_path, num_steps=4)
    player = ReplayPlayer(path)

    # Move forward first
    player.step_forward()
    player.step_forward()
    assert player.current_step == 2

    data = player.step_back()
    assert player.current_step == 1
    assert "obs/agent_0" in data

    player.step_back()
    assert player.current_step == 0

    # Should clamp at 0
    player.step_back()
    assert player.current_step == 0


# ------------------------------------------------------------------
# 54. test_replay_jump_to
# ------------------------------------------------------------------

def test_replay_jump_to(tmp_path):
    """jump_to moves to an arbitrary step and clamps out-of-range values."""
    path = _create_trajectory(tmp_path, num_steps=5)
    player = ReplayPlayer(path)

    data = player.jump_to(3)
    assert player.current_step == 3
    assert "obs/agent_0" in data

    # Jump to 0
    player.jump_to(0)
    assert player.current_step == 0

    # Jump beyond range — should clamp
    player.jump_to(100)
    assert player.current_step == 4  # last valid step

    # Jump below range — should clamp
    player.jump_to(-5)
    assert player.current_step == 0
