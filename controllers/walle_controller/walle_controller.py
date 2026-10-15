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
walle_node = robot.getSelf()
eve_node = robot.getFromDef('EVE')

def get_grid_coords(webots_pos):
    # Map the 3x3 meter floor (-1.5 to +1.5) to our 10x10 AI grid (0 to 9)
    x = int((webots_pos[0] + 1.5) / 0.3)
    y = int((webots_pos[1] + 1.5) / 0.3)
    return [int(np.clip(x, 0, 9)), int(np.clip(y, 0, 9))]

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
    def __init__(self, grid_size=(10, 10)):
        self.grid_x, self.grid_y = grid_size
        self.belief = np.ones(grid_size, dtype=np.float64) / (grid_size[0] * grid_size[1])
        self.scan_locs = []
        self.scans_taken = 0
        self.max_dist = math.dist([0, 0], [self.grid_x, self.grid_y])

    def update_scan(self, current_grid, reading, sigma=0.08):
        self.scans_taken += 1
        self.scan_locs.append(list(current_grid))
        for x in range(self.grid_x):
            for y in range(self.grid_y):
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

    def choose_action(self, current_grid, battery):
        current_grid = list(current_grid)
        target = self.get_target_guess()

        # Step 1: Initial scan to paint first distance ring
        if self.scans_taken == 0:
            return 0
        
        # Step 2: If 1 scan taken, move along baseline to create triangulation parallax
        if self.scans_taken == 1:
            first_loc = self.scan_locs[0]
            if math.dist(current_grid, first_loc) < 2.5:
                # Drive East or North to build triangulation baseline
                if current_grid[0] < self.grid_x - 1 and current_grid[0] <= first_loc[0] + 1:
                    return 2  # East
                elif current_grid[1] < self.grid_y - 1:
                    return 1  # North
                elif current_grid[0] < self.grid_x - 1:
                    return 2
                else:
                    return 1
            else:
                # Baseline achieved! Take second scan to intersect rings
                return 0

        # Step 3: We have 2+ scans -> circles intersect at true target. Drive straight to target!
        cx, cy = current_grid
        tx, ty = target

        if cx == tx and cy == ty:
            return 0

        dx = tx - cx
        dy = ty - cy
        if abs(dx) >= abs(dy):
            if dx > 0: return 2   # East
            elif dx < 0: return 4 # West
            elif dy > 0: return 1 # North
            elif dy < 0: return 3 # South
        else:
            if dy > 0: return 1   # North
            elif dy < 0: return 3 # South
            elif dx > 0: return 2 # East
            elif dx < 0: return 4 # West
            
        return 0

brain = WebotsBayesianBrain()
battery = 100
speed = 6.28  # E-puck maximum motor speed (rad/s)
step_count = 0

print(">>> AUTONOMOUS TRIANGULATION ENGINE ONLINE.")
print(">>> Beginning real-time search...")

# 5. Main Physical Simulation Loop
while robot.step(timestep) != -1:
    current_pos = walle_node.getPosition()
    grid_loc = get_grid_coords(current_pos)

    # Check mission success before action
    if eve_node:
        eve_pos = eve_node.getPosition()
        dist_to_target = math.dist([current_pos[0], current_pos[1]], [eve_pos[0], eve_pos[1]])
        if dist_to_target < 0.35:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            print(f"\n=======================================================")
            print(f">>> MISSION ACCOMPLISHED: TARGET SECURED!")
            print(f">>> Final Location: {grid_loc} | Target (EVE): {get_grid_coords(eve_pos)}")
            print(f">>> Remaining Battery: {battery}% | Total Steps: {step_count}")
            print(f"=======================================================")
            break

    # Determine next autonomous action
    step_count += 1
    action = brain.choose_action(grid_loc, battery)

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

        # Stop motors cleanly to prevent drift before next step
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

    # Check Battery Status
    if battery <= 0:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        print("\n>>> SYSTEM SHUTDOWN: Battery Depleted.")
        break

    # Re-check Target Proximity after move
    current_pos = walle_node.getPosition()
    if eve_node:
        eve_pos = eve_node.getPosition()
        dist_to_target = math.dist([current_pos[0], current_pos[1]], [eve_pos[0], eve_pos[1]])
        if dist_to_target < 0.35:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            print(f"\n=======================================================")
            print(f">>> MISSION ACCOMPLISHED: TARGET SECURED!")
            print(f">>> Final Location: {get_grid_coords(current_pos)} | Target (EVE): {get_grid_coords(eve_pos)}")
            print(f">>> Remaining Battery: {battery}% | Total Steps: {step_count}")
            print(f"=======================================================")
            break