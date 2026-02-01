"""
One file to represent the following:
    - Architecture
    - Training
    - Metrics & Visualization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Environment
GRID_MIN = -10
GRID_MAX = 10

RANDOM_SPAWN_MIN_X = -3
RANDOM_SPAWN_MIN_Y = -3

RANDOM_SPAWN_MAX_X = -2
RANDOM_SPAWN_MAX_Y = -2

GOAL = np.array([5, 5])

# Movement
MAX_STEP_COUNT = 40
MAX_STEP_SIZE = 0.5

def step(state: np.array, action: np.array) -> np.array:
    return np.clip(state + action, GRID_MIN, GRID_MAX)


# DATA

def generate_expert_trajectories(n_trajs: int=30):
    """
    Genearate an array of trajectories that serve as expert demonstrations.
    """
    trajectories = []

    for _ in range(n_trajs):
        
        state = np.ndarray([
            np.random.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
            np.random.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y)
        ])
        traj = []
        
        for _ in range(MAX_STEP_COUNT):
        
            direction = GOAL - state
            direction /= (np.linalg.norm(direction) + 1e-8)
            
            action = MAX_STEP_SIZE * direction
            next_state = step(state, action)
            
            traj.append((state.copy(), action.copy(), next_state.copy()))
            
            state = next_state

        trajectories.append(traj)

    return trajectories


# GENERATOR

INPUT_SIZE = 2 # Represents (x, y) position
OUTPUT_SIZE = 2 # Represents (Δx, Δy) change in pose
HIDDEN_SIZE = 32 # Number of hidden nodes in nueral net

class Generator(nn.Module):
    """
    Class that represents the agent that predicts action based on current state.
    """
    def __init__(self):
        
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
            nn.ReLU()
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        )

    def forward(self, state: np.array):
        return self.net(state)

def generate_generator_trajectories(generator: Generator, n_trajs:int = 10) -> list:
    """
    Generate trajectories through the generator.
    """
    trajectories = []

    for _ in range(n_trajs):

        state = np.ndarray([
            np.random.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
            np.random.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y)
        ])

        traj = []

        for _ in range(n_trajs):

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            action = generator(state_tensor).squeeze(0).detach.numpy()
            next_state = step(state, action)

            traj.append((state.copy(), action.copy(), next_state.copy()))

            state = next_state

        trajectories.append(traj)

    return trajectories

# ENCODER & VARIATIONAL DATA BOTTLENECK

I_C = 0.5 # Upper bound on the mutual information between the encoding and the original features I(X, Z)
E_INPUT_SIZE = 4 # Represents an input of (x, y, x', y')
E_HIDDEN_SIZE = 128
Z_SIZE = 6 # The higher the z size the more complex the task 

class Encoder(nn.Module):
    """
    Class that maps states to latent space Z.
    """
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(E_INPUT_SIZE, E_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(E_HIDDEN_SIZE, E_HIDDEN_SIZE),
            nn.ReLU()
        )

        self.mu = nn.Linear(E_HIDDEN_SIZE, Z_SIZE)

        self.logvar == nn.Linear(E_HIDDEN_SIZE, Z_SIZE)

    def forward(self, x:float, y:float, x_next:float, y_next:float):

        h = torch.cat([x, y, x_next, y_next], dim=-1)
        h = self.net(h)

        mu = self.mu(h)
        logvar = self.logvar(h)

        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


# TODO: Write the KL divergence function ot 




