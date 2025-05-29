import numpy as np
from bridge_env import BridgeBuildingEnv
from train import make_env

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
    
from mujoco.viewer import launch
import mujoco
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from bridge_env import BridgeBuildingEnv

def visualize_agent(episodes=1):
    venv = DummyVecEnv([make_env()])
    venv = VecNormalize.load("vec_normalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False

    # Load model
    model = PPO.load("final_model", env=venv)

    # Unwrap for raw mujoco access
    env = venv.venv.envs[0]
    for ep in range(episodes):
        obs, _ = env.reset()
        viewer = launch(env.model, env.data)
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            print("running")
            # Apply action controls
            for i in range(3):
                for j, actuator_idx in enumerate(env.block_actuators[i]):
                    env.data.ctrl[actuator_idx] = action[i*3 + j]
            mujoco.mj_step(env.model, env.data)

            # Render the viewer frame
            viewer.render()

            # Get next obs, reward, done info from env.step()
            obs, reward, done, _, info = env.step(action)

        viewer.close()
    
if __name__ == "__main__":
    main()
    # visualize_agent()