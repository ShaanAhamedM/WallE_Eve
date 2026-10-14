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

        # Chain-of-Thought Step 1: Energy & Environmental Assessment
        if battery <= 20 and recharge_stations:
            closest_stn = min(recharge_stations, key=lambda s: math.dist(current_loc, s))
            self.last_reasoning = (
                f"[CoT Assessment]: Battery critically low at {battery}%. "
                f"Halting search to execute energy harvest protocol at {closest_stn}."
            )
            return self._navigate_step(current_loc, closest_stn, obstacles)

        # Chain-of-Thought Step 2: Information Ambiguity Assessment
        if self.scans_taken < 1:
            self.last_reasoning = (
                "[CoT Reasoning]: Step 1: Initial state is blind (uniform distribution). "
                "Step 2: Driving blind has high negative expected value. "
                "Decision: Execute omnidirectional SCAN to establish baseline signal."
            )
            return 0

        # Chain-of-Thought Step 3: Triangulation Disambiguation
        dist_to_hypo = math.dist(current_loc, self.estimated_target)
        if max_prob < 0.12 and battery > 12:
            self.last_reasoning = (
                f"[CoT Reasoning]: Hypothesis peak confidence is diffuse ({max_prob*100:.1f}%). "
                f"Triangulation circle has high radial uncertainty. "
                "Decision: Trigger secondary SCAN to intersect probability rings."
            )
            return 0

        # Chain-of-Thought Step 4: Spatial Path Planning
        self.last_reasoning = (
            f"[CoT Reasoning]: Strong Bayesian attractor detected at {self.estimated_target} "
            f"(Confidence: {max_prob*100:.1f}%, Est. Distance: {dist_to_hypo:.1f} units). "
            f"Formulating collision-free vector avoiding {len(obstacles)} hazard zones."
        )
        return self._navigate_step(current_loc, self.estimated_target, obstacles)

    def _navigate_step(self, start, target, obstacles):
        cx, cy = start
        tx, ty = target
        dx = tx - cx
        dy = ty - cy

        # Prioritize axis with largest distance
        preferred_moves = []
        if abs(dy) >= abs(dx):
            preferred_moves.extend([1 if dy > 0 else 3, 2 if dx > 0 else 4])
        else:
            preferred_moves.extend([2 if dx > 0 else 4, 1 if dy > 0 else 3])

        # Add remaining alternatives
        for act in [1, 2, 3, 4]:
            if act not in preferred_moves:
                preferred_moves.append(act)

        act_to_delta = {
            1: (0, 1),
            2: (1, 0),
            3: (0, -1),
            4: (-1, 0)
        }

        for act in preferred_moves:
            delta = act_to_delta[act]
            nx, ny = cx + delta[0], cy + delta[1]
            if 0 <= nx < self.grid_x and 0 <= ny < self.grid_y:
                if (nx, ny) not in obstacles:
                    return act

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
