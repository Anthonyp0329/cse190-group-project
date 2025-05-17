# Bridge Building RL Project

This project implements a reinforcement learning agent that learns to build bridges between two platforms using blocks, with the goal of allowing a ball to travel across the bridge.

## Environment
- The environment is built using MuJoCo physics simulator
- Two platforms are placed at a distance from each other
- The agent can place blocks to create a bridge
- A ball is used to test if the bridge is successful
- Rewards are given based on the ball's ability to cross the bridge

## Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
- `bridge_env.py`: Custom Gymnasium environment for the bridge-building task
- `bridge_model.xml`: MuJoCo model file defining the simulation environment
- `train.py`: Script for training the RL agent
- `utils.py`: Utility functions for the project

## Training
To train the agent:
```bash
python train.py
```

## Requirements
- Python 3.8+
- MuJoCo 2.3.7
- PyTorch
- Stable-Baselines3
- Gymnasium 