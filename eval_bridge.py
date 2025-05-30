import gymnasium as gym
from stable_baselines3 import PPO
from bridge_env import BridgeBuildingEnv
import mujoco
from mujoco.viewer import launch
import time
import numpy as np

def main():
    # Create environment
    env = BridgeBuildingEnv()
    
    # Load the trained model
    print("Loading model from ppo_bridge.zip...")
    model = PPO.load("ppo_bridge")
    
    # Create viewer - removed passive parameter
    # viewer = launch(env.model, env.data)
    
    # Evaluation parameters
    num_episodes = 5
    total_rewards = []
    
    print("\nStarting evaluation...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not (done or truncated):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            print(action)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # Update viewer
            # viewer.sync()
            time.sleep(env.model.opt.timestep)  # Small delay for visualization
            
            # Print status
            if not env.simulation_started:
                print(f"Step {env.steps}: Placing blocks...")
            else:
                print(f"Step {env.steps}: Ball pos={info['ball_pos']}, vel={info['ball_vel']}, reward={reward:.2f}")
            
            if done or truncated:
                print(f"Episode finished after {env.steps} steps")
                print(f"Final reward: {episode_reward:.2f}")
                if env.simulation_started:
                    print(f"Ball stopped: {info['ball_stopped']}")
                total_rewards.append(episode_reward)
        
        # Small pause between episodes
        time.sleep(1.0)
    
    # Print evaluation summary
    print("\nEvaluation finished!")
    print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    
    # Keep viewer open until closed
    viewer = launch(env.model, env.data)
    
    viewer.close()

if __name__ == "__main__":
    main() 