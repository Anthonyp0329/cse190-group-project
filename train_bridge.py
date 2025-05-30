# train_bridge.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from bridge_env import BridgeBuildingEnv     # ← your updated env

# ───────────────────────────────────────────────────────────── constants
PLACEMENT_STEPS        = 50                  # see bridge_env.py
EXTRA_SAFE_STEPS       = 2                  # a couple of spare moves
MAX_STEPS_PER_EPISODE  = PLACEMENT_STEPS + EXTRA_SAFE_STEPS
TOTAL_TIMESTEPS        = 5000            # ~20 k episodes

# ─────────────────────────────────────────────────────── env factory
def make_env():
    env = BridgeBuildingEnv()
    # hard‑cap the high‑level decisions so PPO never stalls in infinite loops
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS_PER_EPISODE)
    return env

# ─────────────────────────────────────────────────────────── training
if __name__ == "__main__":
    vec_env  = DummyVecEnv([make_env])      # single process → deterministic
    logger   = configure("runs", ["stdout", "tensorboard"])

    model = PPO(
        policy            = "MlpPolicy",
        env               = vec_env,
        n_steps           = 32,             # shorter rollout because episodes are tiny
        batch_size        = 32,
        learning_rate     = 3e-4,
        gamma             = 0.99,
        gae_lambda        = 0.95,
        clip_range        = 0.2,
        vf_coef           = 0.4,
        verbose           = 1,
    )
    model.set_logger(logger)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("ppo_bridge")

    # ─────────────────────────────────────────────────────── evaluation
    print("\n=== demo rollout with trained policy ===")
    demo_env, _ = make_env().envs[0], None     # unwrap DummyVecEnv -> real env
    obs, info   = demo_env.reset()
    done        = False
    total_r     = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = demo_env.step(action)
        total_r += reward
        demo_env.render()                      # comment out if running headless

    print(f"episode finished, reward = {total_r:.2f}")
    demo_env.close()
