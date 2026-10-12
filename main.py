from environment import SimulationEnvironment
from agent import AutonomousAgent
import time

def run_mission():
    print("--- INITIATING OPERATION AXIOM ---")
    
    # 1. Boot up the environment and the agent
    env = SimulationEnvironment('config.json')
    agent = AutonomousAgent(env.grid_size)
    
    step_count = 0
    
    # 2. The Main Control Loop
    while env.mission_active:
        step_count += 1
        print(f"\n[Step {step_count}] Battery: {env.battery}% | Location: {env.agent_loc}")
        
        # A. Agent decides what to do based on its internal map
        action, direction = agent.choose_action(env.agent_loc, env.battery)
        
        # B. Send the command to the physical environment
        if action == "move":
            print(f"Agent Command: MOVE {direction}")
            telemetry = env.step("move", direction)
        elif action == "scan":
            print("Agent Command: SCAN")
            telemetry = env.step("scan")
            
            # C. If we scanned, feed that data back into the agent's brain
            reading = telemetry["observation"]
            agent.update_belief(env.agent_loc, reading)
            
            # Print top guess for debugging
            best_guess = agent.get_most_likely_location()
            confidence = round(agent.belief_state[best_guess[0], best_guess[1]] * 100, 1)
            print(f"Sensors updated. Best guess for EVE: {best_guess} ({confidence}% confidence)")

        # Print the environment's response
        print(f"System: {telemetry['message']}")
        
        # Pause slightly so we can read the terminal output like a real log
        time.sleep(0.5)

if __name__ == "__main__":
    run_mission()