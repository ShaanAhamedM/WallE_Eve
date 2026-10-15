import numpy as np
import heapq
import math
from algorithms.base_agent import BaseSearchAgent

class InformationAStarAgent(BaseSearchAgent):
    """
    Information-Directed A* (IDA*) Planner.
    Uses A* path cost optimization with expected Information Gain heuristic
    to plan minimal-battery paths through obstacle fields while triangulating targets.
    """
    def __init__(self, grid_size, name="Information-Directed A*"):
        super().__init__(grid_size, name=name)
        self.max_map_dist = math.dist([0, 0], grid_size)

    def reset(self):
        self.belief_state = np.ones((self.grid_x, self.grid_y), dtype=np.float64) / (self.grid_x * self.grid_y)
        self.scans_taken = 0
        self.planned_path = []
        self.last_reasoning = "A* Planner initialized."
        self.target_peak = [0, 0]

    def update_observation(self, current_loc, sensor_reading, info=None):
        if sensor_reading is None or sensor_reading < 0:
            return

        # Bayesian likelihood update to maintain search target estimate
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                expected = max(0.0, 1.0 - (math.dist(current_loc, [x, y]) / self.max_map_dist))
                diff = abs(expected - sensor_reading)
                likelihood = math.exp(- (diff ** 2) / 0.02)
                self.belief_state[x, y] *= likelihood

        total = np.sum(self.belief_state)
        if total > 0:
            self.belief_state /= total

        if info and "obstacles" in info:
            for ox, oy in info["obstacles"]:
                if 0 <= ox < self.grid_x and 0 <= oy < self.grid_y:
                    self.belief_state[ox, oy] = 0.0
            re_tot = np.sum(self.belief_state)
            if re_tot > 0:
                self.belief_state /= re_tot

        self.scans_taken += 1
        flat_idx = np.argmax(self.belief_state)
        self.target_peak = list(np.unravel_index(flat_idx, self.belief_state.shape))
        self.planned_path = []  # Invalidate previous path upon new sensor intelligence

    def choose_action(self, current_loc, battery, info=None):
        obstacles = set(tuple(o) for o in info.get("obstacles", [])) if info else set()
        max_prob = float(np.max(self.belief_state))

        # Scan if we have zero scans
        if self.scans_taken < 1:
            self.last_reasoning = "A* Planner initial scan to determine distance heuristic."
            return 0

        # Probe move to break symmetry
        if self.scans_taken == 1:
            self.last_reasoning = "A* Planner taking 1-step displacement to break circle symmetry."
            cx, cy = current_loc
            if (cx + 1, cy) not in obstacles and cx + 1 < self.grid_x:
                return 2  # East probe
            elif (cx, cy + 1) not in obstacles and cy + 1 < self.grid_y:
                return 1  # North probe

        # Plan or continue A* path to target peak
        if not self.planned_path or self.planned_path[0] == tuple(current_loc):
            self.planned_path = self._run_astar(tuple(current_loc), tuple(self.target_peak), obstacles)

        if len(self.planned_path) > 1:
            next_node = self.planned_path[1]
            self.planned_path.pop(0)
            dx = next_node[0] - current_loc[0]
            dy = next_node[1] - current_loc[1]
            self.last_reasoning = f"A* executing trajectory step towards {next_node} (Target: {self.target_peak})"
            if dy > 0: return 1   # North
            if dx > 0: return 2   # East
            if dy < 0: return 3   # South
            if dx < 0: return 4   # West

        self.last_reasoning = "A* on target coordinate. Pinpointing scan."
        return 0

    def _run_astar(self, start, goal, obstacles):
        """Standard A* graph search minimizing battery path cost g(n) + heuristic h(n)."""
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for dx, dy in dirs:
                next_node = (current[0] + dx, current[1] + dy)
                if 0 <= next_node[0] < self.grid_x and 0 <= next_node[1] < self.grid_y:
                    if next_node in obstacles:
                        continue
                    new_cost = cost_so_far[current] + 2  # Movement cost
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + (abs(goal[0] - next_node[0]) + abs(goal[1] - next_node[1])) * 2
                        heapq.heappush(frontier, (priority, next_node))
                        came_from[next_node] = current

        # Reconstruct path
        path = []
        curr = goal if goal in came_from else None
        while curr is not None:
            path.append(curr)
            curr = came_from.get(curr)
        path.reverse()
        return path if path else [start]

    def get_telemetry(self):
        p = self.belief_state.flatten()
        p = p[p > 1e-12]
        entropy = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0
        return {
            "name": self.name,
            "belief_state": self.belief_state.copy(),
            "entropy": entropy,
            "max_confidence": float(np.max(self.belief_state)),
            "most_likely_target": self.target_peak,
            "scans_taken": self.scans_taken,
            "last_reasoning": self.last_reasoning
        }
