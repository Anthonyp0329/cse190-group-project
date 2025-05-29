# bridge_env.py – improved version
import re
import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from mujoco.viewer import launch
import os


PLACEMENT_STEPS       = 100      # max steps for the build phase
MAX_ROLL_STEPS        = 50     # physics steps before termination
INITIAL_BALL_VEL      = np.array([1.0, 0.0, 0.0])   # m/s +X
VEL_STOP_THRESHOLD    = 0.01   # when is the ball "stopped"?
GOAL_X                = 2.0    # right platform X
GOAL_Y                = 0.0    # right platform Y
GOAL_Z                = 0.5    # right platform Z
GOAL_RADIUS           = 0.2

class BridgeBuildingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ init
    def __init__(self):
        super().__init__()

        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("bridge_model.xml")
        self.data = mujoco.MjData(self.model)

        # Define action space: 9 values for block positions + 1 done flag
        self.action_space = spaces.Box(
            low=np.array([-2, -0.5, 0, -2, -0.5, 0, -2, -0.5, 0, 0], dtype=np.float32),  # Block positions + done flag
            high=np.array([2, 0.5, 1, 2, 0.5, 1, 2, 0.5, 1, 1], dtype=np.float32),
            dtype=np.float32
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
        self.max_steps = 1000

        # Track block placement
        self.blocks_placed = [False, False, False]

        # Get joint and actuator indices for blocks
        self.block_joints = []  # list of (jx, jy, jz) joint ids
        self.block_actuators = []  # list of (ax, ay, az) actuator ids

        for i in range(1, 4):  # For each block
            # Get joint IDs
            joint_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"block{i}_x")
            joint_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"block{i}_y")
            joint_z = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"block{i}_z")
            self.block_joints.append((joint_x, joint_y, joint_z))

            # Get actuator IDs
            act_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"block{i}_x")
            act_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"block{i}_y")
            act_z = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"block{i}_z")
            self.block_actuators.append((act_x, act_y, act_z))

        # Get joint index for ball
        self.ball_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")

    # ---------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)          # <-- add
        self.simulation_started = False
        self.steps = 0
        self.blocks_placed = [False, False, False]
        return self._get_obs(), {}

    # ---- step ----------------------------------------------------------------
    def step(self, action):
        SERVO_TOL, SERVO_MAX_STEPS = 1e-3, 1000
        done_flag   = action[-1] >= 0.5
        targets     = action[:-1].reshape(3, 3)

        # placement phase ------------------------------------------------------
        if not self.simulation_started:
            for bi, (ax, ay, az) in enumerate(self.block_actuators):
                self.data.ctrl[[ax, ay, az]] = targets[bi]
                self.blocks_placed[bi] = True

            # advance physics until blocks settle
            for _ in range(SERVO_MAX_STEPS):
                mujoco.mj_step(self.model, self.data)
                settled = all(
                    max(abs(self.data.ctrl[self.block_actuators[bi][d]]
                            - self.data.qpos[self.block_joints[bi][d]])
                        for d in range(3)) < SERVO_TOL
                    for bi in range(3)
                )
                if settled:
                    break

            # ONE logical “step” has happened
            self.steps += 1

            if done_flag or self.steps >= PLACEMENT_STEPS:
                self.simulation_started = True
                self.data.qvel[0:3] = INITIAL_BALL_VEL.copy()

            reward = 0.0
            done   = False

        # roll phase -----------------------------------------------------------
        if self.simulation_started:
            mujoco.mj_setKeyframe(self.model, self.data, 0)
            while self.steps < MAX_ROLL_STEPS:
                mujoco.mj_step(self.model, self.data)
                self.steps += 1
                ball_pos = self.data.sensordata[0:3]
                dist     = np.linalg.norm(ball_pos - np.array([GOAL_X, GOAL_Y, GOAL_Z]))
                # terminate if stopped, reached goal, or safety cap
                if self._is_ball_stopped() or dist < GOAL_RADIUS or self.steps >= MAX_ROLL_STEPS:
                    reward = 100.0 if dist < GOAL_RADIUS else -dist
                    done   = True
                    break
            

        obs = self._get_obs()
        info = {
            "ball_pos": ball_pos if self.simulation_started else self.data.sensordata[0:3],
            "ball_vel": self.data.sensordata[3:6],
            "simulation_started": self.simulation_started,
        }
        return obs, reward, done, False, info


    # ---------------------------------------------------------------- utils
    def _is_ball_stopped(self):
        # Check if ball's velocity magnitude is below threshold
        ball_vel = self.data.sensordata[3:6]  # Ball velocity from sensor
        return np.linalg.norm(ball_vel) < self.velocity_threshold

    def _get_obs(self):
        # Return block positions and platform positions as observation
        # Block positions (9 values) + platform positions (6 values)
        # Platform positions are fixed: left platform at origin, right platform at GOAL_X
        platform_positions = np.array([
            -2.0, 0.0, 0.5,  # Left platform (x, y, z)
            GOAL_X, GOAL_Y, GOAL_Z  # Right platform (x, y, z)
        ], dtype=np.float32)
        
        return np.concatenate([
            self.data.sensordata[6:15].astype(np.float32),  # Block positions
            platform_positions  # Platform positions
        ])

    # ---------------------------------------------------------------- render
    def render(self):
        # Launch the viewer
        launch(self.model, self.data)

    def close(self):
        # The viewer is automatically closed when the window is closed
        pass
