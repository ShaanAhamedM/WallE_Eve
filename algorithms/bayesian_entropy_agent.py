import numpy as np
import math
from collections import deque
from algorithms.base_agent import BaseSearchAgent

class BayesianEntropyAgent(BaseSearchAgent):
    """
    Bayesian Active Inference Search Agent.
    Combines Recursive Bayesian State Estimation with Shannon Entropy-guided
    Exploration vs Exploitation and obstacle-aware path planning.
    """
    def __init__(self, grid_size, sensor_variance=0.01, confidence_threshold=0.12, name="Bayesian Active Inference"):
        self.sensor_variance = sensor_variance
        self.confidence_threshold = confidence_threshold
        self.max_map_dist = math.dist([0, 0], grid_size)
        super().__init__(grid_size, name=name)

    def reset(self):
        total_cells = self.grid_x * self.grid_y
        # Uniform prior: P(EVE = (x,y)) = 1 / N
        self.belief_state = np.ones((self.grid_x, self.grid_y), dtype=np.float64) / total_cells
        self.entropy_history = []
        self.confidence_history = []
        self.scans_taken = 0
        self.scanned_locations = set()
        self.last_reasoning = "Initialized uniform prior belief state."
        self.target_peak = [0, 0]

    def update_observation(self, current_loc, sensor_reading, info=None):
        """
        Executes recursive Bayesian update across all grid hypotheses:
        Posterior P(S | z) ~ Likelihood P(z | S) * Prior P(S)
        """
        if sensor_reading is None or sensor_reading < 0:
            return

        for x in range(self.grid_x):
            for y in range(self.grid_y):
                expected = self._simulate_sensor(current_loc, [x, y])
                diff = abs(expected - sensor_reading)
                # Gaussian likelihood: exp(- (z - z_pred)^2 / (2 * var))
                likelihood = math.exp(- (diff ** 2) / (2.0 * self.sensor_variance))
                self.belief_state[x, y] *= likelihood

        total = np.sum(self.belief_state)
        if total > 0:
            self.belief_state /= total
        else:
            # Failsafe uniform reset in degenerate numerical case
            self.belief_state = np.ones((self.grid_x, self.grid_y), dtype=np.float64) / (self.grid_x * self.grid_y)

        # Zero out known obstacle cells (EVE cannot be inside an asteroid!)
        if info and "obstacles" in info:
            for ox, oy in info["obstacles"]:
                if 0 <= ox < self.grid_x and 0 <= oy < self.grid_y:
                    self.belief_state[ox, oy] = 0.0
            re_total = np.sum(self.belief_state)
            if re_total > 0:
                self.belief_state /= re_total

        self.scans_taken += 1
        current_entropy = self.get_entropy()
        max_prob = float(np.max(self.belief_state))
        self.entropy_history.append(current_entropy)
        self.confidence_history.append(max_prob)
        self.target_peak = self.get_most_likely_location()
        self.last_reasoning = f"Scan #{self.scans_taken}: Entropy dropped to {current_entropy:.2f} bits | Peak prob at {self.target_peak}: {max_prob*100:.1f}%"

    def _simulate_sensor(self, agent_loc, target_loc):
        dist = math.dist(agent_loc, target_loc)
        return max(0.0, 1.0 - (dist / self.max_map_dist))

    def get_most_likely_location(self):
        flat_idx = np.argmax(self.belief_state)
        return list(np.unravel_index(flat_idx, self.belief_state.shape))

    def get_entropy(self):
        """Computes Shannon Entropy in bits: H(B) = -sum(p * log2(p))."""
        p = self.belief_state.flatten()
        p = p[p > 1e-12]
        if len(p) == 0:
            return 0.0
        return float(-np.sum(p * np.log2(p)))

    def choose_action(self, current_loc, battery, info=None):
        """
        Active Information Gain Decision Logic:
        1. If global entropy is high (above 5.0 bits) and scans < 2: Scan to triangulate.
        2. If max confidence is below threshold and hasn't scanned this spot: Scan.
        3. If confident: Navigate towards target peak using obstacle-avoiding BFS.
        4. If battery is critically low (< 25%) and a recharge station exists: Divert to recharge!
        """
        obstacles = set(tuple(o) for o in info.get("obstacles", [])) if info else set()
        recharge_stations = [tuple(r) for r in info.get("recharge_stations", [])] if info else []
        max_prob = float(np.max(self.belief_state))
        entropy = self.get_entropy()
        self.target_peak = self.get_most_likely_location()

        dist_to_peak = math.dist(current_loc, self.target_peak)

        # Low battery survival protocol: only divert if battery cannot reach target and station is reachable
        if battery <= 18 and (dist_to_peak * 2.5 > battery) and recharge_stations:
            closest_station = min(recharge_stations, key=lambda r: math.dist(current_loc, r))
            if list(current_loc) != list(closest_station):
                self.last_reasoning = f"LOW BATTERY ({battery}%): Diverting to Solar Station at {closest_station}"
                return self._pathfind_action(current_loc, closest_station, obstacles)

        # Initial scan if we have zero scans
        if self.scans_taken < 1:
            self.last_reasoning = "Initial scan to establish distance radius."
            return 0

        # Step 2: 1-step probe to establish triangulation baseline
        if self.scans_taken == 1:
            self.last_reasoning = "Stepping forward along axis to break radial circle symmetry."
            cx, cy = current_loc
            if (cx + 1, cy) not in obstacles and cx + 1 < self.grid_x:
                return 2  # East probe
            elif (cx, cy + 1) not in obstacles and cy + 1 < self.grid_y:
                return 1  # North probe

        # Arrived right at target peak: trigger scan to verify
        if list(current_loc) == list(self.target_peak):
            if tuple(current_loc) not in self.scanned_locations:
                self.scanned_locations.add(tuple(current_loc))
                self.last_reasoning = f"Arrived at estimated target {self.target_peak}. Verifying target."
                return 0
            else:
                flat_order = np.argsort(self.belief_state.flatten())[::-1]
                for idx in flat_order:
                    candidate = list(np.unravel_index(idx, self.belief_state.shape))
                    if candidate != list(current_loc):
                        self.target_peak = candidate
                        break

        # Confident enough to exploit: move towards peak
        self.last_reasoning = f"Navigating to peak belief {self.target_peak} (Confidence: {max_prob*100:.1f}%)"
        return self._pathfind_action(current_loc, self.target_peak, obstacles)

    def _pathfind_action(self, start, goal, obstacles):
        """
        Breadth-First Search (BFS) to find optimal obstacle-avoiding next move.
        Returns discrete action (1: North, 2: East, 3: South, 4: West).
        """
        start = tuple(start)
        goal = tuple(goal)
        if start == goal:
            return 0  # Scan if already standing on target

        dirs = [
            (1, (0, 1)),   # North
            (2, (1, 0)),   # East
            (3, (0, -1)),  # South
            (4, (-1, 0))   # West
        ]

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if current == goal:
                if path:
                    return path[0]
                break

            for act, (dx, dy) in dirs:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < self.grid_x and 0 <= ny < self.grid_y:
                    neighbor = (nx, ny)
                    if neighbor not in visited and neighbor not in obstacles:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [act]))

        # Greedy fallback if no direct path found
        cx, cy = start
        gx, gy = goal
        if cx < gx and (cx + 1, cy) not in obstacles: return 2
        if cx > gx and (cx - 1, cy) not in obstacles: return 4
        if cy < gy and (cx, cy + 1) not in obstacles: return 1
        if cy > gy and (cx, cy - 1) not in obstacles: return 3
        return 0

    def get_telemetry(self):
        return {
            "name": self.name,
            "belief_state": self.belief_state.copy(),
            "entropy": self.get_entropy(),
            "max_confidence": float(np.max(self.belief_state)),
            "most_likely_target": self.get_most_likely_location(),
            "scans_taken": self.scans_taken,
            "last_reasoning": self.last_reasoning
        }
