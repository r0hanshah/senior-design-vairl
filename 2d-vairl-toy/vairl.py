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
        
        state = np.array([
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
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
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

        state = np.array([
            np.random.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
            np.random.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y)
        ])

        traj = []

        for _ in range(MAX_STEP_COUNT):

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            action = generator(state_tensor).squeeze(0).detach().numpy()
            next_state = step(state, action)

            traj.append((state.copy(), action.copy(), next_state.copy()))

            state = next_state

        trajectories.append(traj)

    return trajectories

# ENCODER & VARIATIONAL DATA BOTTLENECK

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

        self.logvar = nn.Linear(E_HIDDEN_SIZE, Z_SIZE)

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


# KL divergence function that serves as an upper bound for I(X, Z)
# I(X, Z) <= E[KL[E(z|x)||r(z)]] s.t. r(z) = N(0,I) & E(z|x) = N(u_E(x), SIGMA_E(x))

BETA = torch.tensor(1.0) # Lagrange multiplier
I_C = 0.5 # Upper bound on the mutual information between the encoding and the original features I(X, Z)
BETA_STEP_SIZE = 1e-3 # Step size for updating the dual variable BETA

def kl_divergence(mu, logvar):
    """
    Calculate the KL divergence.
    """

    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=-1
    )
    
    return kl # E[KL(E(Z|x)||r(z)]]

def update_beta(beta, kl):
    """
    Update the lagrange multiplier beta.
    """

    with torch.no_grad():
        beta += BETA_STEP_SIZE * (kl.detach() - I_C)
        beta.clamp_(min=0)


D_HIDDEN_SIZE = 128

class Discriminator(nn.Module):
    """
    CLass that attempts to differentiate vectors from the latent space.
    """
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(Z_SIZE, D_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(D_HIDDEN_SIZE, D_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(D_HIDDEN_SIZE, 1) # logit
        )

    def forward(self, z):
        return self.net(z)


NUM_EPISODES = 200

def main():
    """
    Training loop.
    """

    beta = BETA

    # Instantiate main components

    # TODO: How many optimizers do we need? Do we need one for the encoder?

    generator = Generator()
    encoder = Encoder()
    discriminator = Discriminator()

    # Generate expert data
    exp_data = generate_expert_trajectories()

    for _ in range(NUM_EPISODES):

        # Generate generator data
        gen_data = generate_generator_trajectories(generator)

        # DATA -> ENCODER
        
        # Map expert data to latent space

        z_list = [] # Holds latent space transforms from state, next_state pairs
        labels = [] # Ground truth of whether a step taken is expert or generator
        kl_list = []

        for traj in exp_data:
            
            for step in traj:

                 state, action, next_state = step
                 x, y = state
                 x_n, y_n = next_state

                 z, mu, logvar = encoder(x, y, x_n, y_n)

                 z_list.append(z)
                 labels.append(torch.ones(z.size(0), 1)) # expert = 1

                 kl_list.append(kl_divergence(mu, logvar))

        # Map generator data to latent space and get reward

        gen_rewards = []
        gen_states = []
        gen_actions = []

        for traj in gen_data:
            
            traj_rewards = []

            for step in traj:

                state, action, next_state = step
                x, y = state
                x_n, y_n = next_state

                z, mu, logvar = encoder(x, y, x_n, y_n)

                # Get reward for generator
                logit = discriminator(z)
                reward = -F.logsigmoid(-logit) #= -log(sigmoid(logit))
                reward = reward.detach()
                traj_rewards.append(reward.squeeze())

                gen_states.append(state)
                gen_actions.append(action)

                z_list.append(z)
                labels.append(torch.zeros(z.size(0), 1)) # generator = 0

                kl_list.append(kl_divergence(mu, logvar))

            gen_rewards.append(traj_rewards)

        
        # REINFORCE-STYLE GENERATOR UPDATE
        generator_loss = 0.0

        for traj_states, traj_actions, traj_rewards in zip(gen_states, gen_actions, gen_rewards):

            returns = None # TODO: How to compute returns for PPO update to generator

        
        z_batch = torch.cat(z_list, dim=0)
        label_batch = torch.cat(labels, dim=0)
        kl_batch = torch.cat(kl_list, dim=0)

        # ENCODER -> DISCRIMINATOR
        
        kl = kl_batch.mean()
        
        logits = discriminator(z_batch)

        discriminator_loss = F.binary_cross_entropy_with_logits(logits, label_batch)

        loss = discriminator_loss + beta * (kl - I_c)

        # Update lagrange multiplier (beta)

        update_beta(beta, kl)

        exit()

if __name__ == "__main__":
    main()


