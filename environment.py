import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import json

class WallEEnv(gym.Env):
    """
    Custom Environment that follows the Gymnasium interface.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config_path):
        super().__init__()
        
        # Load Config
        with open(config_path, 'r') as file:
            self.config = json.load(file)
            
        self.grid_size = self.config['grid_size']
        self.max_x, self.max_y = self.grid_size
        self.costs = self.config['action_costs']
        self.start_battery = self.config['resource_budget']['battery']
        
        # 1. ACTION SPACE: 0: Scan, 1: North, 2: East, 3: South, 4: West
        self.action_space = spaces.Discrete(5)
        
        # 2. OBSERVATION SPACE: The sensor reading (float between 0.0 and 1.0)
        # We use a Box space for continuous numerical data
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Standard RL method: Resets the environment to the starting state.
        """
        super().reset(seed=seed)
        
        self.agent_loc = np.array(self.config['start_location'], dtype=np.int32)
        self.eve_loc = np.array(self.config['eve_location'], dtype=np.int32)
        self.battery = self.start_battery
        self.step_count = 0
        
        # Return initial observation and info dict
        return np.array([-1.0], dtype=np.float32), self._get_info()

    def step(self, action):
        """
        Standard RL method: Executes one time step within the environment.
        """
        self.step_count += 1
        observation = -1.0 # Default empty reading
        reward = 0.0
        terminated = False
        
        # Map integer actions back to logic
        if action == 0:  # SCAN
            self.battery -= self.costs['scan']
            observation = self._generate_noisy_reading()
        else:            # MOVE
            self.battery -= self.costs['move']
            self._move_agent(action)

        # Check win/loss conditions
        if np.array_equal(self.agent_loc, self.eve_loc):
            reward = 100.0  # Massive reward for finding EVE
            terminated = True
        elif self.battery <= 0:
            reward = -50.0  # Penalty for dying
            terminated = True
            
        # RL requires returning: observation, reward, terminated, truncated, info dict
        return np.array([observation], dtype=np.float32), reward, terminated, False, self._get_info()

    def _move_agent(self, action):
        # 1: North, 2: East, 3: South, 4: West
        x, y = self.agent_loc
        if action == 1 and y < self.max_y - 1: y += 1
        elif action == 2 and x < self.max_x - 1: x += 1
        elif action == 3 and y > 0: y -= 1
        elif action == 4 and x > 0: x -= 1
        self.agent_loc = np.array([x, y])

    def _generate_noisy_reading(self):
        """
        Calculates exact distance, then adds Gaussian noise to simulate real hardware.
        """
        dist_to_eve = math.dist(self.agent_loc, self.eve_loc)
        max_dist = math.dist([0,0], self.grid_size)
        
        true_signal = max(0.0, 1.0 - (dist_to_eve / max_dist))
        
        # Inject standard deviation noise (10% variance)
        noise = np.random.normal(0, 0.1)
        noisy_signal = np.clip(true_signal + noise, 0.0, 1.0)
        
        return round(noisy_signal, 3)

    def _get_info(self):
        # Telemetry data for debugging (hidden from the RL agent's standard observation)
        return {
            "location": self.agent_loc.tolist(),
            "battery": self.battery
        }