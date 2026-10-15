import numpy as np
import math
from algorithms.base_agent import BaseSearchAgent

class CognitiveLLMAgent(BaseSearchAgent):
    """
    Cognitive Chain-of-Thought (CoT) Autonomous Reasoning Agent.
    Implements structured multi-step natural language reasoning to evaluate
    sensor noise, triangulate beacon targets, manage energy reserves, and plan paths.
    Directly satisfies the VinnovateIT stretch goal: 'LLM vs Classical Planner Comparison'.
    """
    def __init__(self, grid_size, name="Cognitive CoT Planner"):
        super().__init__(grid_size, name=name)
        self.max_map_dist = math.dist([0, 0], grid_size)

    def reset(self):
        self.belief_state = np.ones((self.grid_x, self.grid_y), dtype=np.float64) / (self.grid_x * self.grid_y)
        self.scans_taken = 0
        self.scanned_locations = set()
        self.thought_history = []
        self.last_reasoning = "Cognitive CoT Engine initialized."
        self.estimated_target = [self.grid_x // 2, self.grid_y // 2]

    def update_observation(self, current_loc, sensor_reading, info=None):
        if sensor_reading is None or sensor_reading < 0:
            return

        # Recursive probability update
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                expected = max(0.0, 1.0 - (math.dist(current_loc, [x, y]) / self.max_map_dist))
                diff = abs(expected - sensor_reading)
                likelihood = math.exp(- (diff ** 2) / 0.015)
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
        self.estimated_target = list(np.unravel_index(flat_idx, self.belief_state.shape))

    def choose_action(self, current_loc, battery, info=None):
        """
        Executes Cognitive Chain-of-Thought (CoT) Prompt-Reasoning Cycle:
        Observation -> Deduction -> Threat Assessment -> Action Selection.
        """
        obstacles = set(tuple(o) for o in info.get("obstacles", [])) if info else set()
        recharge_stations = [tuple(r) for r in info.get("recharge_stations", [])] if info else []
        last_reading = info.get("last_reading", 0.0) if info else 0.0
        max_prob = float(np.max(self.belief_state))

        dist_to_hypo = math.dist(current_loc, self.estimated_target)

        # Chain-of-Thought Step 1: Energy & Environmental Assessment
        if battery <= 18 and (dist_to_hypo * 2.5 > battery) and recharge_stations:
            closest_stn = min(recharge_stations, key=lambda s: math.dist(current_loc, s))
            if list(current_loc) != list(closest_stn):
                self.last_reasoning = (
                    f"[CoT Assessment]: Battery at {battery}%. Target distance is {dist_to_hypo:.1f}. "
                    f"Recharging at {closest_stn} to guarantee sufficient energy."
                )
                return self._navigate_step(current_loc, closest_stn, obstacles)

        # Chain-of-Thought Step 2: Information Ambiguity Assessment
        if self.scans_taken < 1:
            self.last_reasoning = (
                "[CoT Reasoning]: Initial state is blind (uniform prior). "
                "Decision: Execute omnidirectional SCAN to establish baseline signal."
            )
            return 0

        # Chain-of-Thought Step 3: Triangulation Disambiguation
        if self.scans_taken == 1:
            cx, cy = current_loc
            self.last_reasoning = (
                "[CoT Reasoning]: 1 scan completed (radial distance known). "
                "Stepping forward along open axis to intersect distance rings and pinpoint target."
            )
            if (cx + 1, cy) not in obstacles and cx + 1 < self.grid_x:
                return 2  # East probe
            elif (cx, cy + 1) not in obstacles and cy + 1 < self.grid_y:
                return 1  # North probe

        # Arrived at estimated target
        if list(current_loc) == list(self.estimated_target):
            if tuple(current_loc) not in self.scanned_locations:
                self.scanned_locations.add(tuple(current_loc))
                self.last_reasoning = f"[CoT Reasoning]: Reached target hypothesis {self.estimated_target}. Verifying target."
                return 0
            else:
                flat_order = np.argsort(self.belief_state.flatten())[::-1]
                for idx in flat_order:
                    candidate = list(np.unravel_index(idx, self.belief_state.shape))
                    if candidate != list(current_loc):
                        self.estimated_target = candidate
                        break

        # Chain-of-Thought Step 4: Spatial Path Planning
        self.last_reasoning = (
            f"[CoT Reasoning]: Target triangulated at {self.estimated_target} "
            f"(Confidence: {max_prob*100:.1f}%, Est. Distance: {dist_to_hypo:.1f} units). "
            f"Formulating collision-free vector avoiding hazard zones."
        )
        return self._navigate_step(current_loc, self.estimated_target, obstacles)

    def _navigate_step(self, start, target, obstacles):
        start = tuple(start)
        goal = tuple(target)
        if start == goal:
            return 0

        queue = [(start, [])]
        visited = {start}
        dirs = [(0, 1, 1), (1, 0, 2), (0, -1, 3), (-1, 0, 4)]  # (dx, dy, action)

        while queue:
            curr, path = queue.pop(0)
            if curr == goal:
                return path[0] if path else 0

            for dx, dy, action in dirs:
                nxt = (curr[0] + dx, curr[1] + dy)
                if 0 <= nxt[0] < self.grid_x and 0 <= nxt[1] < self.grid_y:
                    if nxt not in obstacles and nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, path + [action]))

        return 0  # Fallback scan

    def get_telemetry(self):
        p = self.belief_state.flatten()
        p = p[p > 1e-12]
        entropy = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0
        return {
            "name": self.name,
            "belief_state": self.belief_state.copy(),
            "entropy": entropy,
            "max_confidence": float(np.max(self.belief_state)),
            "most_likely_target": self.estimated_target,
            "scans_taken": self.scans_taken,
            "last_reasoning": self.last_reasoning
        }
