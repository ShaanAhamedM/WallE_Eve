import json
import math

class SimulationEnvironment:
    def __init__(self, config_path):
        # Load the mission parameters[cite: 1]
        with open(config_path, 'r') as file:
            self.config = json.load(file)
            
        self.grid_size = self.config['grid_size']
        self.eve_loc = self.config['eve_location']
        self.agent_loc = self.config['start_location'].copy()
        
        self.battery = self.config['resource_budget']['battery']
        self.costs = self.config['action_costs']
        self.clues = self.config['clues']
        
        self.mission_active = True

    def step(self, action, direction=None):
        """
        Processes the agent's chosen action, deducts resources, and returns the result.
        """
        if not self.mission_active:
            return {"status": "Mission Over", "battery": self.battery}

        observation = None
        message = ""

        # ACTION: MOVE
        if action == "move" and direction:
            if self.battery >= self.costs['move']:
                self.battery -= self.costs['move']
                self._update_position(direction)
                message = f"Moved {direction}. New position: {self.agent_loc}"
            else:
                message = "Not enough battery to move."
                self.mission_active = False

        # ACTION: SCAN
        elif action == "scan":
            if self.battery >= self.costs['scan']:
                self.battery -= self.costs['scan']
                observation = self._generate_scan_reading()
                message = f"Scanned area. Signal strength: {observation}"
            else:
                message = "Not enough battery to scan."
                self.mission_active = False
                
        # Check win/loss conditions
        if self.agent_loc == self.eve_loc:
            message = "EVE FOUND! Mission Accomplished."
            self.mission_active = False
        elif self.battery <= 0:
            message = "CRITICAL ERROR: Battery depleted."
            self.mission_active = False

        # Return the observation dictionary for this step[cite: 1]
        return {
            "current_location": self.agent_loc,
            "battery_remaining": self.battery,
            "observation": observation,
            "message": message,
            "mission_active": self.mission_active
        }

    def _update_position(self, direction):
        # Update coordinates while preventing the agent from driving off the grid boundaries
        x, y = self.agent_loc
        max_x, max_y = self.grid_size
        
        if direction == "North" and y < max_y - 1: y += 1
        elif direction == "South" and y > 0: y -= 1
        elif direction == "East" and x < max_x - 1: x += 1
        elif direction == "West" and x > 0: x -= 1
        
        self.agent_loc = [x, y]

    def _generate_scan_reading(self):
        """
        Calculates signal strength based on Euclidean distance to EVE.
        Closer = higher signal (closer to 1.0).
        """
        # Distance formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
        dist_to_eve = math.dist(self.agent_loc, self.eve_loc)
        
        # Invert the distance so closer distance equals higher signal strength
        max_possible_dist = math.dist([0,0], self.grid_size)
        signal_strength = max(0, 1 - (dist_to_eve / max_possible_dist))
        
        return round(signal_strength, 3)