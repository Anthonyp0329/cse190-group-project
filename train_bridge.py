# train_bridge.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
from pathlib import Path

from bridge_env import BridgeBuildingEnv     # ← your updated env

# ───────────────────────────────────────────────────────────── constants            # a couple of spare moves
MAX_STEPS_PER_EPISODE  = 20
TOTAL_TIMESTEPS        = 1000000            # ~20 k episodes
NUM_ENVS             = 1              # run multiple envs in parallel (tweak for your CPU)
CHECKPOINT_FREQ       = 1000          # save every N environment steps
CHECKPOINT_DIR        = "checkpoints"  # directory to store checkpoints

# ─────────────────────────────────────────── checkpoint callback
class VecNormCheckpointCallback(BaseCallback):
    """
    Save the PPO policy *and* VecNormalize statistics every `save_freq` steps.
    """
    def __init__(self, save_freq: int, save_path: str, vec_env: VecNormalize, name_prefix: str = "model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.vec_env = vec_env
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        # `self.num_timesteps` is provided by BaseCallback
        if self.n_calls % self.save_freq == 0:
            step_tag   = f"{self.num_timesteps}"
            model_file = self.save_path / f"{self.name_prefix}_{step_tag}"
            # Save policy
            self.model.save(model_file)
            # Save normalisation statistics alongside the model
            if self.verbose:
                print(f"[checkpoint] saved to {model_file}")
        return True

# ─────────────────────────────────────────────────────── env factory
def make_env():
    env = BridgeBuildingEnv()
    return env

# ─────────────────────────────────────────────────────────── training
if __name__ == "__main__":
    # launch several envs in parallel for higher throughput
    vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])  # Pass the function, don't call it
    # vec_env = VecNormalize.load("vecnormalize_bridge.pkl", vec_env)
    logger   = configure("runs", ["stdout", "tensorboard"])

    # prepare checkpoint callback
    checkpoint_callback = VecNormCheckpointCallback(
        save_freq   = CHECKPOINT_FREQ,
        save_path   = CHECKPOINT_DIR,
        vec_env     = vec_env,
        name_prefix = "ppo_bridge",
        verbose     = 1,
    )

    model = PPO(
        policy            = "CnnPolicy",
        env               = vec_env,
        n_steps           = 32,             # 40 env‑steps × 4 envs per update
        batch_size        = 32,
        learning_rate     = 3e-4,
        gamma             = 0.99,
        gae_lambda        = 0.95,
        clip_range        = 0.2,
        vf_coef           = 0.4,
        verbose           = 1,
    )
    model.set_logger(logger)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    model.save("ppo_bridge")
    # Save normalisation statistics

    # ─────────────────────────────────────────────────────── evaluation
    print("\n=== demo rollout with trained policy ===")
    demo_env = DummyVecEnv([make_env])
    demo_env = VecNormalize.load("vecnormalize_bridge.pkl", demo_env)
    demo_env.training = False          # turn off updating stats
    demo_env.norm_reward = False       # but keep obs normalisation
    obs, _ = demo_env.reset()
    done   = False
    total_r = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = demo_env.step(action)
        reward = reward[0]           # unwrap vector env reward
        done   = done[0]
        total_r += reward
        demo_env.render()                      # comment out if running headless

    print(f"episode finished, reward = {total_r:.2f}")
    demo_env.close()
