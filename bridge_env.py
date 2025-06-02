import re
import time
import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from mujoco.viewer import launch
import os
import mujoco.viewer as mj_viewer
import cv2


class BridgeBuildingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ init
    def __init__(self, num_blocks: int = 20, block_size: float = 0.25):
        super().__init__()
        self.IMG_H= 128
        self.IMG_W = 256

        # allow caller to specify how many movable bridge blocks exist
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("bridge_model.xml")
        self.data = mujoco.MjData(self.model)

        self.renderer = mujoco.Renderer(self.model, height=self.IMG_H,
                                        width=self.IMG_W)

        # Make sure the camera exists
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA,
                             "ground_cam") == -1:
            raise RuntimeError("Camera 'ground_cam' missing from XML.")

        # Observation space is now an image
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.IMG_H, self.IMG_W, 3),
            dtype=np.uint8
        )

        self.coord_low = np.array([
            -0.75 + self.block_size,  # x (left of left platform)
            self.block_size,          # z at least one half block above ground
        ], dtype=np.float32)
        self.coord_high = np.array([
            2.25 - self.block_size,   # x (right of right platform)
            2.0,                      # z upper bound
        ], dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


        # Simulation control
        self.simulation_started = False
        self.initial_ball_velocity = np.array([1.0, 0.0, 0.0])  # 1 unit/s to the right
        self.velocity_threshold = 0.01  # Threshold for considering ball stopped

        # Track episode steps
        self.max_steps = self.num_blocks               # allow more block placements
        # physics parameters
        self.inner_loop_steps = 1000        # shorter inner MuJoCo roll‑out
        self.ball_speed        = 3.0       # faster x velocity for the ball
        
        # track ball x‑position to compute shaped reward
        self.prev_ball_x = 0.0
        
        self.x_before_ground = -10

        # Free‑joint handles for the blocks
        self.block_joints = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"block{i}_free")
            for i in range(1, self.num_blocks + 1)
        ]
        # Starting qpos index (x, y, z, qw, qx, qy, qz) for each block
        self.block_qpos_starts = [
            self.model.jnt_qposadr[jid] for jid in self.block_joints
        ]
        # will store last "settled" world positions of blocks to detect movement
        self.prev_block_pos = np.zeros((self.num_blocks, 3), dtype=np.float32)
        # store the first "settled" position for each block; used for distance‑scaled penalties
        self.initial_block_pos = np.zeros((self.num_blocks, 3), dtype=np.float32)
        # scaling factor for movement penalty (negative reward per metre moved)
        self.movement_penalty_scale = 0.5   # tune as needed
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
        self.gap = 3.0
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
        for i in range(self.num_blocks):
            self.prev_block_pos[i] = self.data.sensordata[6 + 3 * i : 9 + 3 * i]

        # the episode starts with every block at its initial (spawn) location
        for i in range(self.num_blocks):
            self.initial_block_pos[i] = self.prev_block_pos[i]
        
        # initialise progress tracker for shaped reward
        self.prev_ball_x = self.data.sensordata[0]
        
        return self._get_obs(), {}

    def _unnormalize_action(self, a: np.ndarray) -> np.ndarray:
        """Convert an action in [-1, 1] range to world coordinates."""
        a = np.clip(a, -1.0, 1.0)
        a = self.coord_low + 0.5 * (a + 1.0) * (self.coord_high - self.coord_low)
        return np.array([a[0],0,a[1]])

    # ---- step ----------------------------------------------------------------
    def step(self, action):
        """
        Teleport a block by overwriting the translational (x, y, z) part of
        its free joint according to the action. Control alternates between blocks.
        """
        # Clip action to bounds
        target_pos = self._unnormalize_action(action)
        self.data.qpos[self.ball_qpos_start:self.ball_qpos_start + 3] = self.ball_position_rest


        # Which block do we control this step? 0 → block1, 1 → block2, 2 → block3
        block_idx = self.steps % self.num_blocks
        start     = self.block_qpos_starts[block_idx]

        # Teleport the selected block by overwriting its (x, y, z)
        self.data.qpos[start : start + 3] = target_pos

        # remember where this block was *placed*; later movement will be punished
        for i in range(self.num_blocks):
            start = self.block_qpos_starts[i]
            self.initial_block_pos[i] = self.data.qpos[start : start + 3]

        # Recompute forward kinematics
        mujoco.mj_forward(self.model, self.data)
        # give the ball a forward shove each agent step
        self.data.qvel[:6] = [0, 0, 0, 0, 0, 0]
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:6] = [self.ball_speed, 0, 0, 0, 0, 0]
        
        # Step simulation
        reward = 0
        reached = False
        self.x_before_ground = 0
        self.x_under_block = 0

        for _ in range(self.inner_loop_steps):
            mujoco.mj_step(self.model, self.data)
            if self.data.sensordata[2] >= self.right_z:
                self.x_before_ground = self.data.sensordata[0]
            if self.data.sensordata[2] >= self.block_size:
                self.x_under_block = self.data.sensordata[0]

            if self.check_reached():
                reward += 100
                reached = True
        
        # ------------------------------------------------------------
        # Distance‑scaled penalty for blocks that drift after placement
        # ------------------------------------------------------------
        movement_penalty = 0.0
        # tolerance = 0.02                         # ignore micro‑vibrations (<2 cm)
        # for i in range(self.num_blocks):
        #     cur  = self.data.sensordata[6 + 3 * i : 9 + 3 * i]
        #     dist = np.linalg.norm(cur - self.initial_block_pos[i])
        #     if dist > tolerance:
        #         movement_penalty -= self.movement_penalty_scale * dist
        
        # ------------------------------------------------------------
        # Agent‑initiated early termination
        # ------------------------------------------------------------
        
        # Calculate reward
        reward += movement_penalty
        reward += ((self.x_before_ground - self.left_x) / self.gap) / 4
        reward -= 1.0
        reward += ((self.x_under_block - self.left_x) / self.gap) / 16
        reward += (self.data.qpos[self.ball_qpos_start + 2] / self.left_z) / 8

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
        """
        Render the scene from the ground‑level camera and return an
        (H, W, 3) uint8 RGB array.
        """
        # Tell the renderer which camera to use
        self.renderer.update_scene(self.data, camera="ground_cam")
        img = self.renderer.render()        # (H, W, 3) uint8
        return img

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

        # Clip action to bounds
        target_pos = self._unnormalize_action(action)

        # Which block do we control this step? 0 → block1, 1 → block2, 2 → block3
        block_idx = self.steps % self.num_blocks
        start = self.block_qpos_starts[block_idx]

        # Teleport the selected block by overwriting its (x, y, z)
        self.data.qpos[start : start + 3] = target_pos

        # remember where this block was *placed*; later movement will be punished
        for i in range(self.num_blocks):
            start = self.block_qpos_starts[i]
            self.initial_block_pos[i] = self.data.qpos[start : start + 3]

        # Recompute forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        self.data.qvel[:6] = [0, 0, 0, 0, 0, 0]
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            viewer.sync()
            time.sleep(0.001)
        self.data.qvel[:6] = [self.ball_speed, 0, 0, 0, 0, 0]
        
        # Step simulation
        reward = 0
        reached = False
        self.x_before_ground = -10
        self.x_under_block = 0

        for _ in range(self.inner_loop_steps):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            viewer.sync()
            time.sleep(0.001)
            
            if self.data.sensordata[2] >= self.right_z:
                self.x_before_ground = self.data.sensordata[0]
            if self.data.sensordata[2] >= self.block_size:
                self.x_under_block = self.data.sensordata[0]
            
            if self.check_reached():
                reward += 100
                reached = True
                break

        # Distance‑scaled penalty for blocks that drift after placement
        movement_penalty = 0.0
        # tolerance = 0.02  # ignore micro‑vibrations (<2 cm)
        # for i in range(self.num_blocks):
        #     cur = self.data.sensordata[6 + 3 * i : 9 + 3 * i]
        #     dist = np.linalg.norm(cur - self.initial_block_pos[i])
        #     if dist > tolerance:
        #         movement_penalty -= self.movement_penalty_scale * dist

        # Calculate final reward
        reward += movement_penalty
        reward += ((self.x_before_ground - self.left_x) / self.gap) / 4
        reward -= 1.0
        reward += ((self.x_under_block - self.left_x) / self.gap) / 16
        reward += (self.data.qpos[self.ball_qpos_start + 2] / self.left_z) / 8

        # Check if episode is done
        done = (self.steps >= self.max_steps-1 or reached)
        
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
    obs, _ = env.reset()
    action = env.action_space.sample()
    while True:
        cv2.imshow("ground_cam", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == 27:
            break