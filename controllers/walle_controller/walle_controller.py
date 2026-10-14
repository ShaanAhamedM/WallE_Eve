import sys
import os
import math
import numpy as np
from controller import Supervisor
from stable_baselines3 import PPO

print("--- BOOTING WEBOTS 3D NEURAL INTERFACE ---")

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

# 3. Load the PyTorch Brain
model_path = os.path.join('..', '..', 'ppo_walle_model.zip')
try:
    model = PPO.load(model_path)
    print(">>> SUCCESS: PyTorch PPO Brain Connected!")
except Exception as e:
    print(f">>> CRITICAL ERROR: Could not load model from {model_path}. {e}")
    sys.exit(1)

# 4. Map the physical environment
walle_node = robot.getSelf()
eve_node = robot.getFromDef('EVE')

# Auto-configure / verify EVE's position at grid [7, 8] -> (0.75, 1.05, 0.15)
if eve_node:
    eve_pos = eve_node.getPosition()
    if abs(eve_pos[0] - 0.75) > 0.1 or abs(eve_pos[1] - 1.05) > 0.1:
        eve_node.getField('translation').setSFVec3f([0.75, 1.05, 0.15])
        eve_node.getField('size').setSFVec3f([0.3, 0.3, 0.3])
        print(">>> Auto-placed EVE at target coordinates [7, 8] -> (0.75, 1.05, 0.15)")

# Auto-place WALL-E at starting grid [0, 0] -> (-1.35, -1.35, 0.0) if at default center
init_pos = walle_node.getPosition()
if abs(init_pos[0]) < 0.2 and abs(init_pos[1]) < 0.2:
    walle_node.getField('translation').setSFVec3f([-1.35, -1.35, 0.0])
    walle_node.getField('rotation').setSFRotation([0, 0, 1, 1.5708])
    walle_node.resetPhysics()
    print(">>> Auto-placed WALL-E at starting coordinates [0, 0] -> (-1.35, -1.35, 0.0)")

battery = 100
last_reading = 0.0

def get_grid_coords(webots_pos):
    # Map the 3x3 meter floor (-1.5 to +1.5) to our 10x10 AI grid (0 to 9)
    x = int((webots_pos[0] + 1.5) / 0.3)
    y = int((webots_pos[1] + 1.5) / 0.3)
    return [np.clip(x, 0, 9), np.clip(y, 0, 9)]

# 5. Main Physical Simulation Loop
speed = 6.28  # Maximum E-puck speed (0.1256 m/s)

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
            print(f">>> MISSION ACCOMPLISHED: Target Secured with {battery}% battery remaining at {grid_loc}!")
            break

    # Build telemetry vector
    obs = np.array([grid_loc[0], grid_loc[1], battery, last_reading], dtype=np.float32)
    
    # Neural Network Prediction
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    
    if action == 0:  # SCAN
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        
        if eve_node:
            eve_pos = eve_node.getPosition()
            dist = math.dist([current_pos[0], current_pos[1]], [eve_pos[0], eve_pos[1]])
            max_dist = math.dist([-1.5, -1.5], [1.5, 1.5])
            true_signal = max(0.0, 1.0 - (dist / max_dist))
            last_reading = round(np.clip(true_signal + np.random.normal(0, 0.1), 0.0, 1.0), 3)
            
        battery -= 5
        print(f"[{battery}%] SCANNING at Grid {grid_loc}... Sensor Data: {last_reading}")
        
        # Pause briefly to simulate scan duration
        scan_steps = int(1000 / timestep)
        for _ in range(scan_steps):
            if robot.step(timestep) == -1:
                break
                
    elif action == 1:  # NORTH (Faces +Y)
        walle_node.getField('rotation').setSFRotation([0, 0, 1, 1.5708])
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        battery -= 2
        print(f"[{battery}%] DRIVING NORTH from Grid {grid_loc}")
        
    elif action == 2:  # EAST (Faces +X)
        walle_node.getField('rotation').setSFRotation([0, 0, 1, 0.0])
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        battery -= 2
        print(f"[{battery}%] DRIVING EAST from Grid {grid_loc}")
        
    elif action == 3:  # SOUTH (Faces -Y)
        walle_node.getField('rotation').setSFRotation([0, 0, 1, -1.5708])
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        battery -= 2
        print(f"[{battery}%] DRIVING SOUTH from Grid {grid_loc}")
        
    elif action == 4:  # WEST (Faces -X)
        walle_node.getField('rotation').setSFRotation([0, 0, 1, 3.1415])
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        battery -= 2
        print(f"[{battery}%] DRIVING WEST from Grid {grid_loc}")

    if action in [1, 2, 3, 4]:
        # Drive for ~2.4 seconds to advance exactly 1 grid square (0.3m)
        steps_to_wait = int(2400 / timestep)
        for _ in range(steps_to_wait):
            if robot.step(timestep) == -1:
                break
                
        # Stop motors to prevent drift before next action
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

    # Check Mission Status
    if battery <= 0:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        print(">>> SYSTEM SHUTDOWN: Battery Depleted.")
        break
        
    current_pos = walle_node.getPosition()
    if eve_node:
        eve_pos = eve_node.getPosition()
        dist_to_target = math.dist([current_pos[0], current_pos[1]], [eve_pos[0], eve_pos[1]])
        if dist_to_target < 0.35:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            print(f">>> MISSION ACCOMPLISHED: Target Secured with {battery}% battery remaining at {get_grid_coords(current_pos)}!")
            break