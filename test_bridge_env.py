
import time
import mujoco.viewer as mj_viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bridge_env import BridgeBuildingEnv


def main(steps=300, fps=20):
    env = BridgeBuildingEnv()
    # Wrap in VecNormalize to apply the saved observation statistics
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load("checkpoints/vecnormalize_1000000.pkl", vec_env)
    vec_env.training = False        # do not update running stats
    vec_env.norm_reward = False     # keep rewards un‑normalised for display
    with mj_viewer.launch_passive(env.model, env.data) as viewer:
        for i in range(10):
            obs_raw, info = env.reset()
            obs = vec_env.normalize_obs(obs_raw)
            model = PPO.load("checkpoints/ppo_bridge_1000000.zip")
            dt = 1.0 / fps
            
            done        = False
            total_r     = 0.0
                        # comment out if running headless

            # Create the viewer (blocking call returns a Viewer object)
                # for _ in range(steps):
            while not done:
                action, _ = model.predict(obs, deterministic=True)

                next_obs_raw, reward, complete, _, info = env.main(action, viewer)
                obs = vec_env.normalize_obs(next_obs_raw)
                total_r += reward
                done = complete
                print(reward)

            print(f"Episode reward: {total_r}")
    env.close()


if __name__ == "__main__":
    main()