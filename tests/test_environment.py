import pytest
import numpy as np
import math
import os
from environment import WallEEnv
from super_environment import EnhancedWallEEnv

def test_original_walle_env():
    """Verify original baseline environment is functional and intact."""
    assert os.path.exists("config.json")
    env = WallEEnv("config.json")
    obs, info = env.reset()
    assert len(obs) == 4
    assert info["location"] == [0, 0]
    assert info["battery"] == 100

    # Test move North
    obs, reward, terminated, truncated, info = env.step(1)
    assert info["location"] == [0, 1]
    assert info["battery"] == 98
    assert not terminated

    # Test scan
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["battery"] == 93
    assert 0.0 <= obs[3] <= 1.0

def test_enhanced_environment_bounds_and_obstacles():
    """Test boundary collisions and obstacle clamping in EnhancedWallEEnv."""
    env = EnhancedWallEEnv(config_path="config_advanced.json", randomize_target=False)
    obs, info = env.reset()
    assert len(obs) == 6

    # Test boundary limits: attempt to move West from [0, 0] -> should clamp at [0, 0]
    obs, reward, terminated, truncated, info = env.step(4)
    assert info["location"] == [0, 0]

    # Test boundary limits: attempt to move South from [0, 0] -> should clamp at [0, 0]
    obs, reward, terminated, truncated, info = env.step(3)
    assert info["location"] == [0, 0]

def test_enhanced_environment_recharge():
    """Verify solar recharge station boosts battery without exceeding 100%."""
    env = EnhancedWallEEnv(config_path="config_advanced.json")
    env.reset()
    env.battery = 50
    # Manually place agent adjacent to recharge station [2, 7]
    env.agent_loc = np.array([2, 6], dtype=np.int32)
    obs, reward, terminated, truncated, info = env.step(1)  # Move North to [2, 7]
    assert info["location"] == [2, 7]
    assert info["battery"] > 50  # Recharged!
    assert info["recharged"] is True

def test_shannon_entropy_calculation():
    """Verify Shannon entropy properties: max on uniform, 0 on deterministic."""
    # Uniform distribution over 100 cells: H = log2(100) ~ 6.6438 bits
    uniform = np.ones((10, 10)) / 100.0
    h_uniform = EnhancedWallEEnv.calculate_shannon_entropy(uniform)
    assert math.isclose(h_uniform, math.log2(100), rel_tol=1e-3)

    # Point mass (Dirac delta) distribution: H = 0 bits
    point = np.zeros((10, 10))
    point[5, 5] = 1.0
    h_point = EnhancedWallEEnv.calculate_shannon_entropy(point)
    assert math.isclose(h_point, 0.0, abs_tol=1e-5)
