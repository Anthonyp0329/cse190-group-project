import mujoco
import time
from bridge_env import BridgeBuildingEnv
import numpy as np

def main():
    # Create environment
    env = BridgeBuildingEnv()
    
    # Example block placements
    test_positions = np.array([
        # Block 1: Left side
        0, 1, 0,
        # Block 2: Middle
        0, 1, 0,
        # Block 3: Right side
        0, 1, 0, 0
    ])
    
    print("Placing blocks...")
    
    
    # Step environment with the action
    obs, reward, done, _, info = env.step(test_positions)
    
    print(reward)
    print(obs)
    print(info)
    
    print("All blocks placed. Launching viewer...")
    
    # Launch the viewer
    env.render()  # This will open the viewer and block until closed

if __name__ == "__main__":
    main() 