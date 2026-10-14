import numpy as np
import math

class AutonomousAgent:
    def __init__(self, grid_size):
        self.grid_x, self.grid_y = grid_size
        total_cells = self.grid_x * self.grid_y
        # Uniform prior: every cell has equal probability initially
        self.belief_state = np.ones((self.grid_x, self.grid_y)) / total_cells
        
    def update_belief(self, current_loc, actual_reading):
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                expected = self._simulate_reading(current_loc, [x, y])
                diff = abs(expected - actual_reading)
                
                # CRITICAL UPDATE: Our environment now has noise (std_dev = 0.1).
                # The variance in our Bayesian update must account for this (0.1^2 = 0.01).
                # This prevents the agent from panicking if a sensor reading isn't perfect.
                variance = 0.01 
                likelihood = math.exp(- (diff ** 2) / variance)
                
                self.belief_state[x, y] *= likelihood
                
        # Normalize probabilities so they always sum to 1.0
        total = np.sum(self.belief_state)
        if total > 0:
            self.belief_state /= total

    def _simulate_reading(self, agent_loc, target_loc):
        dist = math.dist(agent_loc, target_loc)
        max_dist = math.dist([0,0], [self.grid_x, self.grid_y])
        return max(0.0, 1.0 - (dist / max_dist))
        
    def get_most_likely_location(self):
        flat_index = np.argmax(self.belief_state)
        return list(np.unravel_index(flat_index, self.belief_state.shape))

    def choose_action(self, current_loc):
        best_guess = self.get_most_likely_location()
        max_prob = np.max(self.belief_state)
        
        # ACTION SPACE: 0: Scan, 1: North, 2: East, 3: South, 4: West
        if max_prob > 0.05:
            target_x, target_y = best_guess
            current_x, current_y = current_loc
            
            if current_x < target_x: return 2     # East
            elif current_x > target_x: return 4   # West
            elif current_y < target_y: return 1   # North
            elif current_y > target_y: return 3   # South
                
        return 0 # Scan