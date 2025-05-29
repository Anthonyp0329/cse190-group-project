import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from bridge_env import BridgeBuildingEnv

def make_env():
    def _init():
        env = BridgeBuildingEnv()
        return env
    return _init

def main():
    # Create and wrap the environment
    env = DummyVecEnv([make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Initialize the agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    # Train the agent
    total_timesteps = 10000
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save("final_model")
    env.save("vec_normalize.pkl")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("best_model", exist_ok=True)
    
    main() 