# bridge_env.py – improved version
import re
import time
import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from mujoco.viewer import launch
import os
import mujoco.viewer as mj_viewer


class BridgeBuildingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ init
    def __init__(self):
        super().__init__()

        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("bridge_model.xml")
        self.data = mujoco.MjData(self.model)

        # Action: desired absolute position (x, y, z) for the box
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0, 0.2], dtype=np.float32),
            high=np.array([1.0,  0, 0.5], dtype=np.float32),
            dtype=np.float32,
        )

        # Define observation space (block positions + platform positions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),  # 9 values for block positions + 6 values for platform positions (2 platforms × 3 coordinates)
            dtype=np.float32
        )

        # Simulation control
        self.simulation_started = False
        self.initial_ball_velocity = np.array([1.0, 0.0, 0.0])  # 1 unit/s to the right
        self.velocity_threshold = 0.01  # Threshold for considering ball stopped

        # Track episode steps
        self.steps = 0
        self.max_steps = 50
        
        self.x_before_ground = -10

        # Free‑joint handles for the three blocks
        self.block_joints = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"block{i}_free")
            for i in range(1, 4)
        ]
        # Starting qpos index (x, y, z, qw, qx, qy, qz) for each block
        self.block_qpos_starts = [
            self.model.jnt_qposadr[jid] for jid in self.block_joints
        ]
        # Get joint index for ball
        self.ball_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        # qpos start index for the ball’s free joint (x, y, z, qw, qx, qy, qz)
        self.ball_qpos_start = self.model.jnt_qposadr[self.ball_joint]

    # ---------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)          # <-- add
        self.simulation_started = False
        self.steps = 0
        return self._get_obs(), {}

    # ---- step ----------------------------------------------------------------
    def step(self, action):
        """
        Teleport a block by overwriting the translational (x, y, z) part of
        its free joint according to the action. Control alternates between blocks.
        """
        # Clip action to bounds
        target_pos = np.clip(action, self.action_space.low, self.action_space.high)

        # Which block do we control this step? 0 → block1, 1 → block2, 2 → block3
        block_idx = self.steps % len(self.block_qpos_starts)
        start     = self.block_qpos_starts[block_idx]

        # Teleport the selected block by overwriting its (x, y, z)
        self.data.qpos[start : start + 3] = target_pos

        # Recompute forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        self.data.qvel[:6] = [1, 0, 0, 0, 0, 0]
        
        # Step simulation
        for _ in range(5000):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if self.data.sensordata[2] < 0.2 and self.x_before_ground==-10:
                self.x_before_ground = self.data.sensordata[0]
        
        # Calculate reward
        (reached, reward) = self._compute_reward()
        print(reward, end="\r")

        # Check if episode is done
        done = (self.steps >= self.max_steps or reached)
        # Additional info
        info = {
            'ball_pos': self.data.sensordata[0:3],
            'ball_vel': self.data.sensordata[3:6],
        }
        # Reset ball position using actuators
        self.data.qpos[:7] = [-2, 0, 1.2, 1, 0, 0, 0]  
        self.data.qvel[:] = 0
        
        # Increment step counter
        self.steps += 1
        self.x_before_ground = -10
        mujoco.mj_forward(self.model, self.data)

        # Observation
        obs = self._get_obs()

        return obs, reward, done, False, info

    # ---------------------------------------------------------------- utils

    def _get_obs(self):
        # Return block positions and platform positions as observation
        # Block positions (9 values) + platform positions (6 values)
        # Platform positions are fixed: left platform at origin, right platform at GOAL_X
        platform_positions = np.array([
            -2.0, 0.0, 0.5,  # Left platform (x, y, z)
            2.0, 0.0, 0.5  # Right platform (x, y, z)
        ], dtype=np.float32)
        
        return np.concatenate([
            self.data.sensordata[6:15].astype(np.float32),  # Block positions
            platform_positions  # Platform positions
        ])
    
    def _compute_reward(self):
        reward = 0
        reached = False
        
        # Get ball position
        ball_pos = self.data.sensordata[0:3]
        
        # Reward for ball moving forward (x-coordinate)
        reward += self.x_before_ground
        
        # Large reward for reaching the right platform
        if ball_pos[0]>2:
            reward += 100
            reached = True
        
        # # Penalty for ball falling
        # if ball_pos[2] < 0:
        #     reward -= 10
        
        return (reached, reward)

    def main(self, action, viewer, steps=300, fps=20):
        dt = 1.0 / fps

        # Create the viewer (blocking call returns a Viewer object)
            # for _ in range(steps):
        target_pos = np.clip(action, self.action_space.low, self.action_space.high)

            # Which block do we control this step? 0 → block1, 1 → block2, 2 → block3
        block_idx = self.steps % len(self.block_qpos_starts)
        start     = self.block_qpos_starts[block_idx]

        # Teleport the selected block by overwriting its (x, y, z)
        self.data.qpos[start : start + 3] = target_pos

        # Recompute forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        self.data.qvel[:6] = [1, 0, 0, 0, 0, 0]
        
        # Step simulation
        for _ in range(5000):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            # Push latest physics state to the window
            viewer.sync()
            # Slow down so you can see what’s happening
            time.sleep(0.0005)
            # print(self.data.sensordata[0])
            if self.data.sensordata[2] < 0.2 and self.x_before_ground==-10:
                print(self.data.sensordata[0])
                self.x_before_ground = self.data.sensordata[0]
        # Calculate reward
        (reached, reward) = self._compute_reward()
        # Check if episode is done
        done = (self.steps >= self.max_steps-1 or reached)
        # Additional info
        info = {
            'ball_pos': self.data.sensordata[0:3],
            'ball_vel': self.data.sensordata[3:6],
        }
        print(info)
        # Reset ball position using actuators
        self.data.qpos[:7] = [-2, 0, 1.2, 1, 0, 0, 0]  
        self.data.qvel[:] = 0

        # Increment step counter
        self.steps += 1

        # Observation
        obs = self._get_obs()

        return obs, reward, done, False, info

    # ---------------------------------------------------------------- render
    def render(self):
        # Launch the viewer
        launch(self.model, self.data)

    def close(self):
        # The viewer is automatically closed when the window is closed
        pass
    
    
    
if __name__ == "__main__":
    env = BridgeBuildingEnv()
    _, _ = env.reset()
    action = env.action_space.sample()
    # action = [-1, 0, 1]
    env.main(action)
