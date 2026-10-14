import pytest
import numpy as np
from super_environment import EnhancedWallEEnv
from algorithms import (
    BayesianEntropyAgent,
    InformationAStarAgent,
    CognitiveLLMAgent,
    DRLAgent
)

@pytest.mark.parametrize("agent_cls", [
    BayesianEntropyAgent,
    InformationAStarAgent,
    CognitiveLLMAgent,
    DRLAgent
])
def test_full_mission_loop(agent_cls):
    """Run 15 simulation steps for each agent to confirm zero runtime crashes."""
    env = EnhancedWallEEnv(config_path="config_advanced.json", randomize_target=False)
    agent = agent_cls(env.grid_size)
    obs, info = env.reset(seed=123)
    agent.reset()

    for step in range(15):
        curr_loc = info["location"]
        battery = info["battery"]
        action = agent.choose_action(curr_loc, battery, info)
        assert action in [0, 1, 2, 3, 4]

        obs, reward, terminated, truncated, info = env.step(action)
        sensor_val = obs[3] if len(obs) > 3 else info.get("last_reading", 0.0)
        agent.update_observation(curr_loc, sensor_val, info)

        telem = agent.get_telemetry()
        assert not np.isnan(telem["entropy"])
        assert not np.isinf(telem["entropy"])

        if terminated or truncated:
            break
