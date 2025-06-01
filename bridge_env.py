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

        # Action: (x, y, z) teleport + done‑flag (0.0 = continue, ≥0.5 = terminate)
        self.action_space = spaces.Box(
            low=np.array([-0.25, 0.0, 0.5], dtype=np.float32),
            high=np.array([1.75,  0.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Define observation space (block positions + platform positions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),  # 9 values for block positions + 6 values for platform positions (2 platforms × 3 coordinates)
            dtype=np.float32
        )

        # Simulation control
        self.simulation_started = False
        self.initial_ball_velocity = np.array([1.0, 0.0, 0.0])  # 1 unit/s to the right
        self.velocity_threshold = 0.01  # Threshold for considering ball stopped

        # Track episode steps
        self.max_steps = 10                 # allow more block placements
        # physics parameters
        self.inner_loop_steps = 1000        # shorter inner MuJoCo roll‑out
        self.ball_speed        = 3.0       # faster x velocity for the ball
        
        # track ball x‑position to compute shaped reward
        self.prev_ball_x = 0.0
        
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
        # will store last "settled" world positions of blocks to detect movement
        self.prev_block_pos = np.zeros((3, 3), dtype=np.float32)
        # Get joint index for ball
        self.ball_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        # qpos start index for the ball's free joint (x, y, z, qw, qx, qy, qz)
        self.ball_qpos_start = self.model.jnt_qposadr[self.ball_joint]

        # Body IDs for the two static platforms — we'll use these to read their
        # positions each step so that the agent can observe different platform
        # heights configured in the XML.
        self.left_platform_body  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_platform"
        )
        self.right_platform_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_platform"
        )

    # ---------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # ------------------------------------------------------------
        #  Randomise platform gap and heights
        # ------------------------------------------------------------
        self.left_x   = -1.25                   # keep left platform anchor
        self.gap      = self.np_random.uniform(1.0, 3.0)   # ≤ 3 block‑widths
        # gap = 3.0
        self.right_x  = self.left_x + 1.0 + self.gap      # 1.0 = two half‑widths (0.5 + 0.5)

        self.left_z   = self.np_random.uniform(1.0, 1.5)
        self.right_z  = self.np_random.uniform(0.4, min(0.9, self.left_z - 0.1))  # shorter

        # self.right_z = 0.5
        # left_z = 0.5
        # update body positions
        self.model.body_pos[self.left_platform_body][:3]  = [self.left_x,  0.0, self.left_z]
        self.model.body_pos[self.right_platform_body][:3] = [self.right_x, 0.0, self.right_z]

        # place ball on top of left platform
        self.ball_position_rest = [self.left_x, 0.0, self.left_z + 1.0 + 0.1]
        self.data.qpos[self.ball_qpos_start:self.ball_qpos_start + 3] = self.ball_position_rest
        self.data.qpos[self.ball_qpos_start + 3:self.ball_qpos_start + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

        self.steps            = 0
        self.x_before_ground  = -10
        # store initial block positions so we can detect movement
        for i in range(3):
            self.prev_block_pos[i] = self.data.sensordata[6 + 3 * i : 9 + 3 * i]
        
        # initialise progress tracker for shaped reward
        self.prev_ball_x = self.data.sensordata[0]
        
        return self._get_obs(), {}

    # ---- step ----------------------------------------------------------------
    def step(self, action):
        """
        Teleport a block by overwriting the translational (x, y, z) part of
        its free joint according to the action. Control alternates between blocks.
        """
        # Clip action to bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_pos     = action
        self.data.qpos[self.ball_qpos_start:self.ball_qpos_start + 3] = self.ball_position_rest


        # Which block do we control this step? 0 → block1, 1 → block2, 2 → block3
        block_idx = self.steps % len(self.block_qpos_starts)
        start     = self.block_qpos_starts[block_idx]

        # Teleport the selected block by overwriting its (x, y, z)
        self.data.qpos[start : start + 3] = target_pos

        # Recompute forward kinematics
        mujoco.mj_forward(self.model, self.data)
        # give the ball a forward shove each agent step
        self.data.qvel[:6] = [self.ball_speed, 0, 0, 0, 0, 0]
        
        # Step simulation
        reward = 0
        reached = False
        self.x_before_ground = 0
        for _ in range(self.inner_loop_steps):
            mujoco.mj_step(self.model, self.data)
            if self.data.sensordata[2] >= self.right_z:
                self.x_before_ground = self.data.sensordata[0] + 1.25

            if self.check_reached():
                reward += 100
                reached = True
        
        # ------------------------------------------------------------
        # Movement penalty for blocks that have shifted after placement
        # ------------------------------------------------------------
        movement_penalty = 0.0
        for i in range(3):
            cur = self.data.sensordata[6 + 3 * i : 9 + 3 * i]
            dist = np.linalg.norm(cur - self.prev_block_pos[i])
            if dist > 0.2:                     # moved > 5 cm
                movement_penalty -= 0.1        # constant penalty per moved block
            self.prev_block_pos[i] = cur        # update reference
        
        # ------------------------------------------------------------
        # Agent‑initiated early termination
        # ------------------------------------------------------------
        
        # Calculate reward
        reward += movement_penalty
        reward += self.x_before_ground

        # -------------------------------
        #  Termination handling
        # -------------------------------

        terminated = reached
        truncated = self.steps >= self.max_steps

        # Additional info
        info = {
            'ball_pos': self.data.sensordata[0:3],
            'ball_vel': self.data.sensordata[3:6],
        }
        
        # Increment step counter
        self.steps += 1
        self.x_before_ground = -10
        mujoco.mj_forward(self.model, self.data)

        # Observation
        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------- utils

    def _get_obs(self):
        # Return block positions, platform positions, and current target block as observation
        # Block positions (9 values) + platform positions (6 values) + target block (3 values)
        # Current world‑frame positions of the left and right platforms
        # (x, y, z for each). Reading them from `data.xpos` lets the agent
        # see the actual platform heights, which may change between episodes.
        platform_positions = np.concatenate(
            [
                self.data.xpos[self.left_platform_body],
                self.data.xpos[self.right_platform_body],
            ]
        ).astype(np.float32)
        
        # Create one-hot encoded vector for current target block
        # Assuming self.current_block is an integer (0, 1, or 2) indicating which block to move
        target_block = np.zeros(3, dtype=np.float32)
        target_block[self.steps % len(self.block_qpos_starts)] = 1.0
        
        return np.concatenate([
            self.data.sensordata[6:15].astype(np.float32),  # Block positions
            platform_positions,  # Platform positions
            target_block  # Current target block (one-hot encoded)
        ])
    
    def check_reached(self):
        """
        Shaped reward:
          • +10 × incremental x‑progress of the ball each step.
          • +100 bonus when the ball reaches or passes the right platform.
        """
        # Current ball position
        ball_pos = self.data.sensordata[0:3]
        reached = False
        if abs(ball_pos[1]) < 0.25 and ball_pos[0] >= self.right_x and ball_pos[2] >= self.right_z + 0.5 and ball_pos[2] < self.right_z + 0.7:
            reached = True

        return reached

    def main(self, action, viewer, steps=300, fps=20):
        dt = 1.0 / fps

        self.data.qpos[self.ball_qpos_start:self.ball_qpos_start + 3] = self.ball_position_rest

        # Create the viewer (blocking call returns a Viewer object)
            # for _ in range(steps):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_pos     = action
            # Which block do we control this step? 0 → block1, 1 → block2, 2 → block3
        block_idx = self.steps % len(self.block_qpos_starts)
        start     = self.block_qpos_starts[block_idx]

        # Teleport the selected block by overwriting its (x, y, z)
        self.data.qpos[start : start + 3] = target_pos

        # Recompute forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        self.data.qvel[:6] = [self.ball_speed, 0, 0, 0, 0, 0]
        
        # Step simulation
        reward = 0
        reached = False
        for _ in range(1000):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            # Push latest physics state to the window
            viewer.sync()
            # Slow down so you can see what's happening
            time.sleep(0.001)
            # print(self.data.sensordata[0])
            if self.data.sensordata[2] < 0.2 and self.x_before_ground==-10:
                print(self.data.sensordata[0])
                self.x_before_ground = self.data.sensordata[0]
            
            if self.check_reached():
                reward += 100
                reached = True
                break
        # Calculate reward
        # (reached, reward) = self._compute_reward()
        # Check if episode is done
        done = (self.steps >= self.max_steps-1 or reached)
        # Additional info
        info = {
            'ball_pos': self.data.sensordata[0:3],
            'ball_vel': self.data.sensordata[3:6],
        }
        print(info)

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
