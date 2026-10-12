import numpy as np
import math

class AutonomousAgent:
    def __init__(self, grid_size):
        self.grid_x, self.grid_y = grid_size
        
        # Initialize a 2D grid with uniform probabilities.
        # e.g., A 10x10 grid means 100 cells. 1 / 100 = 0.01 per cell.
        total_cells = self.grid_x * self.grid_y
        self.belief_state = np.ones((self.grid_x, self.grid_y)) / total_cells
        
    def update_belief(self, current_loc, actual_reading):
        """
        Adjusts the probability map across the entire grid based on new sensor data.
        """
        # Iterate through every cell to calculate the likelihood of EVE being there
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                # 1. Calculate the theoretical reading IF Eve were at (x, y)
                expected_reading = self._simulate_reading(current_loc, [x, y])
                
                # 2. Compare theoretical reading to the actual hardware reading
                # We use a Gaussian formula here. If the difference is 0, likelihood is 1.0.
                # As the difference grows, the likelihood drops exponentially.
                difference = abs(expected_reading - actual_reading)
                variance = 0.05 # Represents sensor noise/margin of error
                likelihood = math.exp(- (difference ** 2) / variance) 
                
                # 3. Bayes' Update: New Belief = Old Belief * Likelihood
                self.belief_state[x, y] *= likelihood
                
        # 4. Normalize the entire grid so the total probability still equals 1.0
        self.belief_state /= np.sum(self.belief_state)

    def _simulate_reading(self, agent_loc, target_loc):
        """
        An internal physics model the agent uses to predict sensor readings.
        """
        dist = math.dist(agent_loc, target_loc)
        max_dist = math.dist([0,0], [self.grid_x, self.grid_y])
        return max(0, 1 - (dist / max_dist))
        
    def get_most_likely_location(self):
        """
        Finds the coordinates with the highest probability peak.
        """
        # Find the index of the absolute maximum value in the 2D array
        flat_index = np.argmax(self.belief_state)
        # Convert the flat index back into a 2D (x, y) coordinate
        return list(np.unravel_index(flat_index, self.belief_state.shape))

    def choose_action(self, current_loc, battery):
        """
        Decides the next action based on current confidence and location.
        """
        # 1. Look at our map and find the highest probability coordinate
        best_guess = self.get_most_likely_location()
        
        # Get the actual probability percentage of that specific cell
        max_prob = np.max(self.belief_state)
        
        # 2. Decision Logic: Are we confident enough to drive blindly?
        # A 10x10 grid starts at 0.01 per cell. If a cell reaches > 0.15 (15%), 
        # it means a scan heavily pointed us in that direction.
        if max_prob > 0.05:
            # We are confident. Let's calculate which direction to drive.
            # We compare our current X,Y to the target X,Y.
            
            target_x, target_y = best_guess
            current_x, current_y = current_loc
            
            if current_x < target_x:
                return "move", "East"
            elif current_x > target_x:
                return "move", "West"
            elif current_y < target_y:
                return "move", "North"
            elif current_y > target_y:
                return "move", "South"
                
        # 3. If we are NOT confident (probability is spread out and flat), 
        # or if we are already standing on the best guess and haven't found her,
        # we need more data. Fire the scanner.
        return "scan", None