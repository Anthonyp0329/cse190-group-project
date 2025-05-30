
import time
import mujoco.viewer as mj_viewer
from stable_baselines3 import PPO

from bridge_env import BridgeBuildingEnv


def main(steps=300, fps=20):
    env = BridgeBuildingEnv()
    obs, info = env.reset()
    model = PPO.load("ppo_bridge.zip")

    dt = 1.0 / fps
    
    done        = False
    total_r     = 0.0
                   # comment out if running headless

    # Create the viewer (blocking call returns a Viewer object)
        # for _ in range(steps):
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.main(action)
        total_r += reward
        time.sleep(1)

    env.close()


if __name__ == "__main__":
    main()