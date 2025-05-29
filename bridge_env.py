import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco
from mujoco.viewer import launch
import os

class BridgeBuildingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("bridge_model.xml")
        self.data = mujoco.MjData(self.model)
        
        # Define action and observation spaces
        # Action space: 9 continuous values (x,y,z for each of 3 blocks)
        self.action_space = spaces.Box(
            low=np.array([-2, -5, 0, -2, -5, 0, -2, -5, 0], dtype=np.float32),
            high=np.array([2, 5, 5, 2, 5, 5, 2, 5, 5], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: positions of ball and blocks, ball velocity
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),  # 3 for ball pos, 3 for ball vel, 3 for each block pos
            dtype=np.float32
        )
        
        # Track episode steps
        self.steps = 0
        self.max_steps = 1000
                
        # Get actuator indices
        self.ball_actuators = []
        self.block_actuators = []
        
        # Find actuator indices by name
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name.startswith('ball_'):
                self.ball_actuators.append(i)
            elif name.startswith('block'):
                block_num = int(name[5]) - 1  # Convert block1_x to 0, block2_x to 1, etc.
                if len(self.block_actuators) <= block_num:
                    self.block_actuators.append([])
                self.block_actuators[block_num].append(i)
        
        # Sort actuators to ensure x,y,z order
        self.ball_actuators.sort(key=lambda i: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        for block_acts in self.block_actuators:
            block_acts.sort(key=lambda i: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset ball position using actuators
        ball_pos = [-2, 0, 1.2]  # Initial ball position
        for i, actuator_idx in enumerate(self.ball_actuators):
            self.data.ctrl[actuator_idx] = ball_pos[i]
        
        # Reset blocks using actuators
        for i in range(3):
            block_pos = [0, (i-1)*3, 0.5]
            for j, actuator_idx in enumerate(self.block_actuators[i]):
                self.data.ctrl[actuator_idx] = block_pos[j]
        
        # Reset episode tracking
        self.steps = 0
        
        # Get initial observation
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        self.steps += 1
        
        # Place blocks according to action
        for i in range(3):
            # Convert action to block position
            block_pos = action[i*3:(i+1)*3]
            # Place block using actuators
            for j, actuator_idx in enumerate(self.block_actuators[i]):
                self.data.ctrl[actuator_idx] = block_pos[j]
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_obs()
        
        # Calculate reward
        (reached, reward) = self._compute_reward()
        
        # Check if episode is done
        done = (self.steps >= self.max_steps or reached)
        # Additional info
        info = {
            'ball_pos': self.data.sensordata[0:3],
            'ball_vel': self.data.sensordata[3:6],
        }
        
        return obs, reward, done, False, info
    
    def _get_obs(self):
        # Get ball position and velocity
        ball_pos = self.data.sensordata[0:3]
        ball_vel = self.data.sensordata[3:6]
        
        # Get block positions
        block_positions = []
        for i in range(3):
            block_pos = self.data.sensordata[6+i*3:9+i*3]
            block_positions.extend(block_pos)
        
        # Combine all observations
        obs = np.concatenate([ball_pos, ball_vel, block_positions])
        return obs.astype(np.float32)
    
    def _compute_reward(self):
        reward = 0
        reached = False
        
        # Get ball position
        ball_pos = self.data.sensordata[0:3]
        
        # Reward for ball moving forward (x-coordinate)
        reward += ball_pos[0] * 0.1
        
        # Large reward for reaching the right platform
        if ball_pos[0] > 1.8 and abs(ball_pos[1]) < 0.5 and ball_pos[2] > 0.3:
            reward += 100
            reached = True
        
        # Penalty for ball falling
        if ball_pos[2] < 0:
            reward -= 10
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.01
        
        return (reached, reward)
    
    def render(self):
        # Launch the viewer
        launch(self.model, self.data)
    
    def close(self):
        # The viewer is automatically closed when the window is closed
        pass 