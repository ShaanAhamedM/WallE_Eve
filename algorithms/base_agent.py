from abc import ABC, abstractmethod
import numpy as np

class BaseSearchAgent(ABC):
    """
    Abstract Base Class for all autonomous search agents in Operation Axiom.
    Standardizes interface for Bayesian, Heuristic, DRL, and LLM controllers.
    """
    def __init__(self, grid_size, name="BaseAgent"):
        self.grid_x, self.grid_y = grid_size
        self.name = name
        self.reset()

    @abstractmethod
    def reset(self):
        """Reset internal memory, belief distributions, and history."""
        pass

    @abstractmethod
    def update_observation(self, current_loc, sensor_reading, info=None):
        """Ingest new sensor telemetry and update internal state/belief map."""
        pass

    @abstractmethod
    def choose_action(self, current_loc, battery, info=None):
        """
        Select next discrete action:
        0: SCAN, 1: NORTH (+Y), 2: EAST (+X), 3: SOUTH (-Y), 4: WEST (-X)
        """
        pass

    @abstractmethod
    def get_telemetry(self):
        """Returns diagnostic telemetry dictionary for HUD rendering."""
        pass
