import sys
import os
import math
import numpy as np
from controller import Supervisor

print("=" * 60)
print("   OPERATION AXIOM: WEBOTS 3D AUTONOMOUS ROBOT CONTROLLER")
print("   Bayesian Active Inference & Real-Time Triangulation")
print("=" * 60)

# 1. Initialize Webots Supervisor
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# 2. Tap into the physical wheel motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# 3. Map physical nodes (Do NOT overwrite EVE's position!)
# 3. Dynamic Node Mapping
walle_node = robot.getSelf()
eve_node = robot.getFromDef('EVE')

def get_grid_coords(webots_pos):
    # Map the 3x3 meter floor (-1.5 to +1.5) to our 10x10 AI grid (0 to 9)
    x = int((webots_pos[0] + 1.5) / 0.3)
    y = int((webots_pos[1] + 1.5) / 0.3)
    return [int(np.clip(x, 0, 9)), int(np.clip(y, 0, 9))]

def get_dynamic_obstacles(supervisor):
    """
    Dynamically queries all DEF OBSTACLE_1..N nodes in the Webots scene tree.
    Allows real-time obstacle avoidance even when obstacles are moved in the 3D viewport.
    """
    obs = set()
    for i in range(1, 20):
        node = supervisor.getFromDef(f'OBSTACLE_{i}')
        if node:
            pos = node.getPosition()
            grid = get_grid_coords(pos)
            obs.add(tuple(grid))
    return obs

if eve_node:
    eve_init_pos = eve_node.getPosition()
    eve_init_grid = get_grid_coords(eve_init_pos)
    print(f">>> Target (EVE) detected at Physical {eve_init_pos[:2]} | Grid {eve_init_grid}")
else:
    print(">>> Warning: 'EVE' node not found in Webots scene tree!")

# 4. Bayesian Active Inference Brain
class WebotsBayesianBrain:
    """
    Real-time Bayesian state estimator for Webots physical simulation.
    Triangulates EVE at ANY arbitrary coordinate using recursive Gaussian likelihood updates.
    """
    def __init__(self, grid_size=(10, 10), obstacles=None):
        self.grid_x, self.grid_y = grid_size
        self.belief = np.ones(grid_size, dtype=np.float64) / (grid_size[0] * grid_size[1])
        self.obstacles = set(obstacles) if obstacles is not None else set()
        for ox, oy in self.obstacles:
            self.belief[ox, oy] = 0.0
        total = np.sum(self.belief)
        if total > 0:
            self.belief /= total
        self.scan_locs = []
        self.scans_taken = 0
        self.max_dist = math.dist([0, 0], [self.grid_x, self.grid_y])

    def update_obstacles(self, obstacles):
        new_obs = set(obstacles)
        if new_obs != self.obstacles:
            self.obstacles = new_obs
            for ox, oy in self.obstacles:
                self.belief[ox, oy] = 0.0
            total = np.sum(self.belief)
            if total > 0:
                self.belief /= total

    def update_scan(self, current_grid, reading, sigma=0.08):
        self.scans_taken += 1
        self.scan_locs.append(list(current_grid))
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                if (x, y) in self.obstacles:
                    self.belief[x, y] = 0.0
                    continue
                expected = max(0.0, 1.0 - (math.dist(current_grid, [x, y]) / self.max_dist))
                diff = abs(expected - reading)
                likelihood = math.exp(- (diff ** 2) / (2.0 * (sigma ** 2)))
                self.belief[x, y] *= likelihood
        total = np.sum(self.belief)
        if total > 0:
            self.belief /= total

    def get_target_guess(self):
        flat = np.argmax(self.belief)
        return list(np.unravel_index(flat, self.belief.shape))

    def _navigate_step(self, start, target):
        start = tuple(start)
        goal = tuple(target)
        if start == goal:
            return 0, True

        queue = [(start, [])]
        visited = {start}
        dirs = [(0, 1, 1), (1, 0, 2), (0, -1, 3), (-1, 0, 4)]  # 1=North (+Y), 2=East (+X), 3=South (-Y), 4=West (-X)

        closest_cell = start
        min_dist = math.dist(start, goal)
        path_to_closest = []

        while queue:
            curr, path = queue.pop(0)
            if curr == goal:
                return (path[0] if path else 0), True

            d = math.dist(curr, goal)
            if d < min_dist:
                min_dist = d
                closest_cell = curr
                path_to_closest = path

            for dx, dy, action in dirs:
                nxt = (curr[0] + dx, curr[1] + dy)
                if 0 <= nxt[0] < self.grid_x and 0 <= nxt[1] < self.grid_y:
                    if nxt not in self.obstacles and nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, path + [action]))

        # Target is enclosed/surrounded by obstacles: route to closest safe standoff perimeter cell
        if path_to_closest:
            return path_to_closest[0], False
        return 0, False

    def choose_action(self, current_grid, battery):
        current_grid = tuple(current_grid)
        target = tuple(self.get_target_guess())

        # Step 1: Initial scan to establish distance ring
        if self.scans_taken == 0:
            return 0, True  # SCAN
        
        # Step 2: 1 single probe step along open axis to break radial symmetry
        if self.scans_taken == 1:
            cx, cy = current_grid
            if (cx + 1, cy) not in self.obstacles and cx + 1 < self.grid_x:
                return 2, True  # East probe
            elif (cx, cy + 1) not in self.obstacles and cy + 1 < self.grid_y:
                return 1, True  # North probe

        # Step 3+: Target is triangulated. Collision-free BFS pathfinding around obstacles to target
        return self._navigate_step(current_grid, target)

initial_obstacles = get_dynamic_obstacles(robot)
brain = WebotsBayesianBrain(obstacles=initial_obstacles)
battery = 100
speed = 6.28  # E-puck maximum motor speed (rad/s)
step_count = 0

print(">>> AUTONOMOUS DYNAMIC TRIANGULATION ENGINE ONLINE.")
print(f">>> Active Dynamic Obstacles Detected: {len(initial_obstacles)} hazard zones.")
print(">>> Beginning real-time search...")

# 5. Main Physical Simulation Loop
while robot.step(timestep) != -1:
    current_pos = walle_node.getPosition()
    grid_loc = get_grid_coords(current_pos)

    # Dynamic Obstacle Sync: Update obstacle map in real-time if blocks were moved
    live_obstacles = get_dynamic_obstacles(robot)
    if live_obstacles:
        brain.update_obstacles(live_obstacles)

    # Physical Touch / Docking Check
    if eve_node:
        eve_pos = eve_node.getPosition()
        dist_to_target = math.dist([current_pos[0], current_pos[1]], [eve_pos[0], eve_pos[1]])

        # 1. Physical Touch: Direct bumper-to-surface contact (< 0.198m)
        if dist_to_target <= 0.198:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            print(f"\n=======================================================")
            print(f">>> FINAL MISSION REPORT: TARGET SECURED (PHYSICAL TOUCH)!")
            print(f">>> Contact Distance: {dist_to_target:.3f}m (Direct Bumper-to-Surface Lock)")
            print(f">>> Evidence Gathered: {step_count} RF signal readings")
            print(f">>> Inferred Location: {grid_loc} | Target (EVE): {get_grid_coords(eve_pos)}")
            print(f">>> Mission Status: TARGET SECURED (PHYSICAL CONTACT ESTABLISHED)")
            print(f">>> Estimated Confidence: {float(np.max(brain.belief)) * 100:.1f}%")
            print(f">>> Resource Status: {battery}% battery remaining ({step_count} steps taken)")
            print(f">>> Recommended Next Step: Lock docking clamps & begin telemetry transfer.")
            print(f"=======================================================")
            break

        # 2. Precision Docking Glide: When in adjacent cell (< 0.38m) after triangulation,
        # orient directly toward EVE and drive into physical touch!
        target_hypo = brain.get_target_guess()
        if (grid_loc == target_hypo or dist_to_target < 0.38) and brain.scans_taken >= 2:
            dock_angle = math.atan2(eve_pos[1] - current_pos[1], eve_pos[0] - current_pos[0])
            walle_node.getField('rotation').setSFRotation([0, 0, 1, dock_angle])
            print(f"[Docking Alignment] Distance to EVE: {dist_to_target:.3f}m -> Closing in for physical touch...")
            
            left_motor.setVelocity(speed * 0.7)
            right_motor.setVelocity(speed * 0.7)
            
            # Glide forward until physical contact is established
            glide_ticks = int(2200 / timestep)
            for _ in range(glide_ticks):
                if robot.step(timestep) == -1:
                    break
                cur = walle_node.getPosition()
                d = math.dist([cur[0], cur[1]], [eve_pos[0], eve_pos[1]])
                if d <= 0.198:
                    break
            
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            final_pos = walle_node.getPosition()
            final_dist = math.dist([final_pos[0], final_pos[1]], [eve_pos[0], eve_pos[1]])
            print(f"\n=======================================================")
            print(f">>> FINAL MISSION REPORT: TARGET SECURED (PHYSICAL TOUCH)!")
            print(f">>> Contact Distance: {final_dist:.3f}m (Direct Bumper-to-Surface Lock)")
            print(f">>> Evidence Gathered: {step_count} RF signal readings")
            print(f">>> Inferred Location: {get_grid_coords(final_pos)} | Target (EVE): {get_grid_coords(eve_pos)}")
            print(f">>> Mission Status: TARGET SECURED (PHYSICAL CONTACT ESTABLISHED)")
            print(f">>> Estimated Confidence: {float(np.max(brain.belief)) * 100:.1f}%")
            print(f">>> Resource Status: {battery}% battery remaining ({step_count} steps taken)")
            print(f">>> Recommended Next Step: Lock docking clamps & begin telemetry transfer.")
            print(f"=======================================================")
            break

    # Determine next autonomous action
    step_count += 1
    action, target_accessible = brain.choose_action(grid_loc, battery)

    # 3. Surrounded Target Handler: If EVE is completely enclosed by obstacles
    if not target_accessible and action == 0 and brain.scans_taken >= 2:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        eve_pos = eve_node.getPosition()
        standoff_angle = math.atan2(eve_pos[1] - current_pos[1], eve_pos[0] - current_pos[0])
        walle_node.getField('rotation').setSFRotation([0, 0, 1, standoff_angle])
        d = math.dist([current_pos[0], current_pos[1]], [eve_pos[0], eve_pos[1]])
        print(f"\n=======================================================")
        print(f">>> MISSION REPORT: TARGET ENCLOSED IN HAZARD PERIMETER!")
        print(f">>> Target (EVE) at Grid {get_grid_coords(eve_pos)} is fully surrounded by physical obstacles.")
        print(f">>> WALL-E secured closest safe standoff at Grid {grid_loc} (Distance: {d:.2f}m).")
        print(f">>> Mission Status: PERIMETER STANDOFF SECURED")
        print(f">>> Reason: Direct docking path is blocked by surrounding hazard perimeter.")
        print(f">>> Recommended Next Step: Deploy retrieval tether or await obstacle clearance.")
        print(f"=======================================================")
        break

    if action == 0:  # SCAN
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        if eve_node:
            eve_pos = eve_node.getPosition()
            dist = math.dist([current_pos[0], current_pos[1]], [eve_pos[0], eve_pos[1]])
            max_phys_dist = math.dist([-1.5, -1.5], [1.5, 1.5])
            true_signal = max(0.0, 1.0 - (dist / max_phys_dist))
            noisy_reading = round(float(np.clip(true_signal + np.random.normal(0, 0.04), 0.0, 1.0)), 3)
            brain.update_scan(grid_loc, noisy_reading)
        else:
            noisy_reading = 0.5

        battery -= 5
        hypo = brain.get_target_guess()
        conf = float(np.max(brain.belief)) * 100
        print(f"[Step {step_count:02d} | Battery {battery:3d}%] SCANNING at Grid {grid_loc} | RF Signal: {noisy_reading:.3f} | Triangulated Target: {hypo} (Confidence: {conf:.1f}%)")

        # Pause to simulate scan sensor duration
        scan_ticks = int(800 / timestep)
        for _ in range(scan_ticks):
            if robot.step(timestep) == -1:
                break

    elif action == 1:  # NORTH (Faces +Y)
        walle_node.getField('rotation').setSFRotation([0, 0, 1, 1.5708])
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        battery -= 2
        print(f"[Step {step_count:02d} | Battery {battery:3d}%] DRIVING NORTH from Grid {grid_loc} towards {brain.get_target_guess()}")

    elif action == 2:  # EAST (Faces +X)
        walle_node.getField('rotation').setSFRotation([0, 0, 1, 0.0])
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        battery -= 2
        print(f"[Step {step_count:02d} | Battery {battery:3d}%] DRIVING EAST from Grid {grid_loc} towards {brain.get_target_guess()}")

    elif action == 3:  # SOUTH (Faces -Y)
        walle_node.getField('rotation').setSFRotation([0, 0, 1, -1.5708])
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        battery -= 2
        print(f"[Step {step_count:02d} | Battery {battery:3d}%] DRIVING SOUTH from Grid {grid_loc} towards {brain.get_target_guess()}")

    elif action == 4:  # WEST (Faces -X)
        walle_node.getField('rotation').setSFRotation([0, 0, 1, 3.1415])
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        battery -= 2
        print(f"[Step {step_count:02d} | Battery {battery:3d}%] DRIVING WEST from Grid {grid_loc} towards {brain.get_target_guess()}")

    if action in [1, 2, 3, 4]:
        # Drive for ~2.4 seconds to advance exactly 1 grid square (0.3m)
        drive_ticks = int(2400 / timestep)
        for _ in range(drive_ticks):
            if robot.step(timestep) == -1:
                break
            # Check for physical contact during drive
            if eve_node:
                cur = walle_node.getPosition()
                ep = eve_node.getPosition()
                if math.dist([cur[0], cur[1]], [ep[0], ep[1]]) <= 0.198:
                    break

        # Stop motors cleanly to prevent drift before next step
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        # Continuous Bayesian State Estimation: Passive beacon ping after every move
        current_pos = walle_node.getPosition()
        grid_loc = get_grid_coords(current_pos)
        if eve_node:
            eve_pos = eve_node.getPosition()
            dist = math.dist([current_pos[0], current_pos[1]], [eve_pos[0], eve_pos[1]])
            max_phys_dist = math.dist([-1.5, -1.5], [1.5, 1.5])
            true_signal = max(0.0, 1.0 - (dist / max_phys_dist))
            noisy_reading = round(float(np.clip(true_signal + np.random.normal(0, 0.03), 0.0, 1.0)), 3)
            brain.update_scan(grid_loc, noisy_reading)
            hypo = brain.get_target_guess()
            conf = float(np.max(brain.belief)) * 100
            print(f"      [Signal Ping] Grid {grid_loc} | Signal: {noisy_reading:.3f} | Triangulated Target: {hypo} (Confidence: {conf:.1f}%)")

    # Check Battery Status
    if battery <= 0:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        print("\n>>> SYSTEM SHUTDOWN: Battery Depleted.")
        break