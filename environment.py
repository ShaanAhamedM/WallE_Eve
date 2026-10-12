import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import json

class WallEEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path):
        super().__init__()
        
        with open(config_path, 'r') as file:
            self.config = json.load(file)
            
        self.grid_size = self.config['grid_size']
        self.max_x, self.max_y = self.grid_size
        self.costs = self.config['action_costs']
        self.start_battery = self.config['resource_budget']['battery']
        
        # Action Space: 0: Scan, 1: North, 2: East, 3: South, 4: West
        self.action_space = spaces.Discrete(5)
        
        # State Vector: [agent_x, agent_y, battery_level, last_sensor_reading]
        low = np.array([0, 0, 0, 0.0], dtype=np.float32)
        high = np.array([self.max_x, self.max_y, self.start_battery, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.last_reading = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_loc = np.array(self.config['start_location'], dtype=np.int32)
        self.eve_loc = np.array(self.config['eve_location'], dtype=np.int32)
        self.battery = self.start_battery
        self.last_reading = 0.0
        
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        return np.array([self.agent_loc[0], self.agent_loc[1], self.battery, self.last_reading], dtype=np.float32)

    def step(self, action):
        reward = -0.5 # Constant penalty forces the model to solve the map FAST
        terminated = False
        
        if action == 0:  
            self.battery -= self.costs['scan']
            self.last_reading = self._generate_noisy_reading()
            reward -= 0.5 # Extra penalty so it doesn't just sit and spam scan
        else:            
            self.battery -= self.costs['move']
            self._move_agent(action)

        if np.array_equal(self.agent_loc, self.eve_loc):
            reward = 100.0  # Massive payout for success
            terminated = True
        elif self.battery <= 0:
            reward = -50.0  # Massive penalty for dying
            terminated = True
            
        return self._get_obs(), reward, terminated, False, self._get_info()

    def _move_agent(self, action):
        x, y = self.agent_loc
        if action == 1 and y < self.max_y - 1: y += 1
        elif action == 2 and x < self.max_x - 1: x += 1
        elif action == 3 and y > 0: y -= 1
        elif action == 4 and x > 0: x -= 1
        self.agent_loc = np.array([x, y])

    def _generate_noisy_reading(self):
        dist = math.dist(self.agent_loc, self.eve_loc)
        max_dist = math.dist([0,0], self.grid_size)
        true_signal = max(0.0, 1.0 - (dist / max_dist))
        noise = np.random.normal(0, 0.1)
        return round(np.clip(true_signal + noise, 0.0, 1.0), 3)

    def _get_info(self):
        return {"location": self.agent_loc.tolist(), "battery": self.battery}