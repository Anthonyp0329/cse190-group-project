
import time
import mujoco.viewer as mj_viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from bridge_env import BridgeBuildingEnv
import cv2


def main(steps=300, fps=20):
    env = BridgeBuildingEnv()
    # Wrap in VecNormalize to apply the saved observation statistics
    vec_env = DummyVecEnv([lambda: env])
    vec_env.training = False        # do not update running stats
    vec_env.norm_reward = False     # keep rewards un‑normalised for display
    with mj_viewer.launch_passive(env.model, env.data) as viewer:
        for i in range(10):
            obs, info = env.reset()
            model = PPO.load("checkpoints/ppo_bridge_4000.zip")
            dt = 1.0 / fps
            
            done        = False
            total_r     = 0.0
                        # comment out if running headless

            # Create the viewer (blocking call returns a Viewer object)
                # for _ in range(steps):
            counter = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)

                obs, reward, complete, _, info = env.main(action, viewer)
                total_r += reward
                done = complete
                print(reward)
                print(f"Type: {type(obs)}, dtype: {getattr(obs, 'dtype', None)}, shape: {getattr(obs, 'shape', None)}")
                # cv2.imwrite(f"./images/{counter}.png", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
                counter += 1

            print(f"Episode reward: {total_r}")
    env.close()


if __name__ == "__main__":
    main()