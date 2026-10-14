import pytest
import numpy as np
from algorithms import (
    BayesianEntropyAgent,
    InformationAStarAgent,
    CognitiveLLMAgent,
    DRLAgent
)

@pytest.fixture
def grid_size():
    return [10, 10]

def test_bayesian_agent_lifecycle(grid_size):
    agent = BayesianEntropyAgent(grid_size)
    agent.reset()
    assert agent.belief_state.shape == (10, 10)
    assert np.isclose(np.sum(agent.belief_state), 1.0)

    # Ingest observation and check belief update
    agent.update_observation([0, 0], 0.85)
    assert np.isclose(np.sum(agent.belief_state), 1.0)
    assert agent.get_entropy() < 6.6438  # Entropy must decrease

    action = agent.choose_action([0, 0], 100, {"obstacles": [], "recharge_stations": []})
    assert action in [0, 1, 2, 3, 4]

def test_astar_agent_lifecycle(grid_size):
    agent = InformationAStarAgent(grid_size)
    agent.reset()
    agent.update_observation([0, 0], 0.75)
    action = agent.choose_action([0, 0], 100, {"obstacles": [[1, 0]], "recharge_stations": []})
    assert action in [0, 1, 2, 3, 4]

def test_cognitive_llm_agent_lifecycle(grid_size):
    agent = CognitiveLLMAgent(grid_size)
    agent.reset()
    # Battery critical should divert
    action = agent.choose_action([1, 1], 15, {"obstacles": [], "recharge_stations": [[2, 2]]})
    assert action in [0, 1, 2, 3, 4]
    telem = agent.get_telemetry()
    assert "CoT" in telem["last_reasoning"]

def test_drl_agent_lifecycle(grid_size):
    agent = DRLAgent(grid_size)
    agent.reset()
    agent.update_observation([0, 0], 0.5)
    action = agent.choose_action([0, 0], 100)
    assert action in [0, 1, 2, 3, 4]
