import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import json
import os

class EnhancedWallEEnv(gym.Env):
    """
    Enhanced POMDP Autonomous Search & Rescue Environment for Operation Axiom.
    Features:
    - Standard Gymnasium API compliance.
    - Stochastic Gaussian sensor noise (Active Range Sensing).
    - Dynamic / Randomized Target (EVE) spawning for robust generalized ML policies.
    - Spatial Obstacles / Hazard Debris Fields (with collision penalty).
    - Solar Recharge Stations (energy harvesting).
    - Shannon Entropy and Information Gain metrics.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config_path="config_advanced.json", randomize_target=False, dynamic_target=False):
        super().__init__()
        
        # Fall back to config.json if config_advanced.json does not exist
        if not os.path.exists(config_path):
            config_path = "config.json"

        with open(config_path, 'r') as file:
            self.config = json.load(file)

        self.grid_size = self.config.get('grid_size', [10, 10])
        self.max_x, self.max_y = self.grid_size
        self.costs = self.config.get('action_costs', {"move": 2, "scan": 5, "recharge": -25})
        self.start_battery = self.config.get('resource_budget', {}).get('battery', 100)
        self.max_battery = 100
        self.max_time_steps = self.config.get('resource_budget', {}).get('time_steps', 60)
        self.sensor_noise_std = self.config.get('sensor_noise_std', 0.1)
        
        self.randomize_target = randomize_target or (self.config.get('target_mode') == 'random')
        self.dynamic_target = dynamic_target or self.config.get('target_dynamic', False)

        # Parse static obstacles and solar recharge stations
        self.obstacles = set(tuple(c) for c in self.config.get('obstacles', []))
        self.recharge_stations = set(tuple(c) for c in self.config.get('recharge_stations', []))

        # Action Space:
        # 0: SCAN
        # 1: NORTH (+Y)
        # 2: EAST (+X)
        # 3: SOUTH (-Y)
        # 4: WEST (-X)
        self.action_space = spaces.Discrete(5)

        # Observation Space (Normalized Telemetry Vector):
        # [norm_agent_x, norm_agent_y, norm_battery, last_sensor_reading, min_recharge_dist, norm_step_ratio]
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_map_dist = math.dist([0, 0], [self.max_x, self.max_y])
        self.agent_loc = np.array(self.config.get('start_location', [0, 0]), dtype=np.int32)
        self.eve_loc = np.array(self.config.get('eve_location', [7, 8]), dtype=np.int32)
        self.battery = self.start_battery
        self.last_reading = 0.0
        self.current_step = 0
        self.visited_locations = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.battery = self.start_battery
        self.current_step = 0
        self.last_reading = 0.0

        # Start location
        start_loc = self.config.get('start_location', [0, 0])
        self.agent_loc = np.array(start_loc, dtype=np.int32)

        # Target (EVE) location
        if self.randomize_target or (options and options.get('randomize_target', False)):
            while True:
                candidate = [np.random.randint(0, self.max_x), np.random.randint(0, self.max_y)]
                if tuple(candidate) not in self.obstacles and not np.array_equal(candidate, self.agent_loc):
                    self.eve_loc = np.array(candidate, dtype=np.int32)
                    break
        else:
            self.eve_loc = np.array(self.config.get('eve_location', [7, 8]), dtype=np.int32)

        self.visited_locations = [self.agent_loc.copy()]
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        norm_x = float(self.agent_loc[0]) / max(1, self.max_x - 1)
        norm_y = float(self.agent_loc[1]) / max(1, self.max_y - 1)
        norm_battery = float(self.battery) / float(self.max_battery)
        last_sensor = float(self.last_reading)
        
        # Distance to closest recharge station
        if self.recharge_stations:
            min_dist = min(math.dist(self.agent_loc, r) for r in self.recharge_stations)
            norm_recharge_dist = float(min_dist) / self.max_map_dist
        else:
            norm_recharge_dist = 1.0

        norm_step_ratio = float(self.current_step) / float(self.max_time_steps)

        return np.array([norm_x, norm_y, norm_battery, last_sensor, norm_recharge_dist, norm_step_ratio], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        reward = -0.5  # Base time-penalty to encourage efficient shortest-path navigation
        terminated = False
        truncated = False
        collision = False
        recharged = False

        prev_dist = math.dist(self.agent_loc, self.eve_loc)

        if action == 0:  # SCAN
            self.battery -= self.costs.get('scan', 5)
            self.last_reading = self._generate_noisy_reading()
            reward -= 0.5  # Extra penalty to deter scan spamming
        else:  # MOVE
            self.battery -= self.costs.get('move', 2)
            collision = self._move_agent(action)
            self.last_reading = self._generate_noisy_reading()
            if collision:
                reward -= 5.0  # Harsh penalty for colliding into obstacles
                self.battery -= 3  # Impact damage drain

        # Check recharge station pickup
        if tuple(self.agent_loc) in self.recharge_stations:
            recharge_val = abs(self.costs.get('recharge', -25))
            if self.battery < self.max_battery:
                self.battery = min(self.max_battery, self.battery + recharge_val)
                reward += 4.0
                recharged = True

        # Optional dynamic target drift (EVE moving slowly)
        if self.dynamic_target and self.current_step % 4 == 0 and not np.array_equal(self.agent_loc, self.eve_loc):
            self._drift_target()

        new_dist = math.dist(self.agent_loc, self.eve_loc)
        # Potential-based reward shaping: reward progress towards target
        reward += (prev_dist - new_dist) * 1.5

        # Check terminal conditions
        if np.array_equal(self.agent_loc, self.eve_loc):
            reward = 120.0  # Massive win payout
            terminated = True
        elif self.battery <= 0:
            reward = -60.0  # Catastrophic failure penalty
            terminated = True
        elif self.current_step >= self.max_time_steps:
            truncated = True

        self.visited_locations.append(self.agent_loc.copy())
        info = self._get_info()
        info['collision'] = collision
        info['recharged'] = recharged
        return self._get_obs(), reward, terminated, truncated, info

    def _move_agent(self, action):
        """
        Executes movement with boundary clamping and obstacle collision detection.
        Returns True if an obstacle collision occurred.
        """
        x, y = self.agent_loc
        orig_x, orig_y = x, y

        if action == 1 and y < self.max_y - 1: y += 1       # North
        elif action == 2 and x < self.max_x - 1: x += 1     # East
        elif action == 3 and y > 0: y -= 1                 # South
        elif action == 4 and x > 0: x -= 1                 # West

        target_cell = (x, y)
        if target_cell in self.obstacles:
            # Revert movement on collision
            self.agent_loc = np.array([orig_x, orig_y], dtype=np.int32)
            return True

        self.agent_loc = np.array([x, y], dtype=np.int32)
        return False

    def _drift_target(self):
        """Dynamic EVE stochastic movement by 1 step in a random safe direction."""
        moves = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        np.random.shuffle(moves)
        for dx, dy in moves:
            nx = np.clip(self.eve_loc[0] + dx, 0, self.max_x - 1)
            ny = np.clip(self.eve_loc[1] + dy, 0, self.max_y - 1)
            if (nx, ny) not in self.obstacles:
                self.eve_loc = np.array([nx, ny], dtype=np.int32)
                break

    def _generate_noisy_reading(self):
        """Calculates distance-attenuated radio beacon signal with Gaussian noise."""
        dist = math.dist(self.agent_loc, self.eve_loc)
        true_signal = max(0.0, 1.0 - (dist / self.max_map_dist))
        noise = np.random.normal(0, self.sensor_noise_std)
        noisy_signal = np.clip(true_signal + noise, 0.0, 1.0)
        return round(float(noisy_signal), 3)

    def _get_info(self):
        true_dist = math.dist(self.agent_loc, self.eve_loc)
        return {
            "location": self.agent_loc.tolist(),
            "eve_location": self.eve_loc.tolist(),
            "battery": int(self.battery),
            "step": self.current_step,
            "last_reading": float(self.last_reading),
            "true_distance": round(true_dist, 3),
            "grid_size": list(self.grid_size),
            "obstacles": [list(o) for o in self.obstacles],
            "recharge_stations": [list(r) for r in self.recharge_stations]
        }

    @staticmethod
    def calculate_shannon_entropy(belief_state):
        """Computes the Shannon Entropy H(B) in bits of a 2D probability distribution."""
        flat = belief_state.flatten()
        flat = flat[flat > 1e-12]
        if len(flat) == 0:
            return 0.0
        return float(-np.sum(flat * np.log2(flat)))
