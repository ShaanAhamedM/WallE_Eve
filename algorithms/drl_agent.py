import numpy as np
import os
import math
from algorithms.base_agent import BaseSearchAgent
from stable_baselines3 import PPO

class DRLAgent(BaseSearchAgent):
    """
    Deep Reinforcement Learning Search Agent.
    Deploys a PyTorch Proximal Policy Optimization (PPO) neural network
    for real-time closed-loop decision making in the POMDP environment.
    """
    def __init__(self, grid_size, model_path="ppo_walle_model.zip", name="PPO Deep Neural Network"):
        self.model_path = model_path
        self.model = None
        self._load_model()
        super().__init__(grid_size, name=name)

    def _load_model(self):
        # Look for model in root or parent
        candidates = [
            self.model_path,
            "ppo_walle_advanced.zip",
            "ppo_walle_model.zip",
            os.path.join("..", self.model_path),
            os.path.join("..", "..", self.model_path)
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    self.model = PPO.load(p)
                    self.loaded_from = p
                    return
                except Exception:
                    pass
        self.model = None
        self.loaded_from = None

    def reset(self):
        self.last_reading = 0.0
        self.action_history = []
        self.last_action_desc = "PPO Model Initialized."
        self.steps_taken = 0
        self.pseudo_belief = np.zeros((self.grid_x, self.grid_y))

    def update_observation(self, current_loc, sensor_reading, info=None):
        if sensor_reading is not None and sensor_reading >= 0:
            self.last_reading = float(sensor_reading)
            # Create a localized radar ring around current location
            max_d = math.dist([0, 0], [self.grid_x, self.grid_y])
            for x in range(self.grid_x):
                for y in range(self.grid_y):
                    pred = max(0.0, 1.0 - (math.dist(current_loc, [x, y]) / max_d))
                    diff = abs(pred - self.last_reading)
                    self.pseudo_belief[x, y] = math.exp(- (diff ** 2) / 0.03)
            tot = np.sum(self.pseudo_belief)
            if tot > 0:
                self.pseudo_belief /= tot

    def choose_action(self, current_loc, battery, info=None):
        self.steps_taken += 1
        
        # Adaptive observation shape detection (6-dim normalized vs 4-dim legacy)
        expected_dim = self.model.observation_space.shape[0] if (self.model and self.model.observation_space) else 4

        if expected_dim == 6:
            norm_x = float(current_loc[0]) / max(1, self.grid_x - 1)
            norm_y = float(current_loc[1]) / max(1, self.grid_y - 1)
            norm_batt = float(battery) / 100.0
            last_s = float(self.last_reading)
            rec_dist = 1.0
            if info and "recharge_stations" in info and info["recharge_stations"]:
                min_d = min(math.dist(current_loc, r) for r in info["recharge_stations"])
                rec_dist = min_d / math.dist([0, 0], [self.grid_x, self.grid_y])
            norm_step = float(self.steps_taken) / 60.0
            obs = np.array([norm_x, norm_y, norm_batt, last_s, rec_dist, norm_step], dtype=np.float32)
        else:
            obs = np.array([current_loc[0], current_loc[1], battery, self.last_reading], dtype=np.float32)

        if self.model is not None:
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = 0 if self.steps_taken == 1 else 2

        action_names = {0: "SCAN", 1: "MOVE NORTH", 2: "MOVE EAST", 3: "MOVE SOUTH", 4: "MOVE WEST"}
        self.last_action_desc = f"PPO Neural Policy: Action {action} ({action_names.get(action, 'UNKNOWN')})"
        return action

    def get_telemetry(self):
        p = self.pseudo_belief.flatten()
        p = p[p > 1e-12]
        entropy = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0
        return {
            "name": self.name,
            "belief_state": self.pseudo_belief.copy(),
            "entropy": entropy,
            "max_confidence": float(np.max(self.pseudo_belief)) if np.sum(self.pseudo_belief) > 0 else 0.0,
            "most_likely_target": list(np.unravel_index(np.argmax(self.pseudo_belief), self.pseudo_belief.shape)),
            "scans_taken": self.steps_taken,
            "last_reasoning": self.last_action_desc
        }
