import numpy as np
from bridge_env import BridgeBuildingEnv

def main():
    # Create the environment
    env = BridgeBuildingEnv()
    
    # Reset the environment
    obs, _ = env.reset()
    
    print("Environment loaded! Press 'ESC' to exit.")
    print("The environment shows:")
    print("- Left platform (gray) at x=-2")
    print("- Right platform (gray) at x=2")
    print("- Ball (red) starting on left platform")
    print("- Three blocks (blue) that can be placed")
    
    env.render()
    
if __name__ == "__main__":
    main() 