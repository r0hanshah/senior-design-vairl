"""
One file to represent the following:
    - Architecture
    - Training
    - Metrics & Visualization
"""

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from sb3_contrib import TRPO

# Environment
GRID_MIN = -10
GRID_MAX = 10

RANDOM_SPAWN_MIN_X = -3
RANDOM_SPAWN_MIN_Y = -3

RANDOM_SPAWN_MAX_X = -2
RANDOM_SPAWN_MAX_Y = -2

GOAL = np.array([5, 5])

class VDBEnv(gym.Env):
    """
    Gym environment necessary to conduct TRPO to optimize the generator policy.
    """
    def __init__(self, encoder, discriminator):
        super().__init__()

        self.encoder = encoder
        self.discriminator = discriminator

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.state = None

    def reset(self):
        self.state = np.array([
            np.random.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
            np.random.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y)
        ])
        return self.state

    def step(self, action):
        x, y = self.state
        dx, dy = action
        x_n = x + dx
        y_n = y + dy

        # VDB REWARD

        with torch.no_grad():

            z, _mu, _logvar = self.encoder(
                torch.tensor(x),
                torch.tensor(y),
                torch.tensor(x_n),
                torch.tensor(y_n)
            )

            logits = self.discriminator(z)

            reward = torch.log(torch.sigmoid(logits) + 1e-8).item()

        self.state = next_state
        done = False # TODO: Understand why this is okay...

        return next_state, reward, done, {}

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

def generate_generator_trajectories(generator: TRPO, env: VDBEnv, n_trajs:int = 10) -> list:
    """
    Generate trajectories through the generator.
    """
    trajectories = []

    for _ in range(n_trajs):

        state = env.reset()

        traj = []

        for _ in range(MAX_STEP_COUNT):

            action, _ = generator.predict(state, deterministic=False)
            next_state, _reward, _done, _ = env.step(action)

            traj.append((state.copy(), action.copy(), next_state.copy()))

            state = next_state

            if done:
                state = env.reset()

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

BETA = torch.tensor(1.0, requires_grad=True) # Lagrange multiplier
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


# TRPO IS GENERATOR



NUM_EPISODES = 200
ROLLOUT_STEPS = MAX_STEP_COUNT

GENERATOR_LEARNING_RATE = 1e-3
DISCRIMINATOR_LEARNING_RATE= 1e-3
ENCODER_LEARNING_RATE = 1e-3

def main():
    """
    Training loop.
    """

    beta = BETA

    # Instantiate main components

    env = VDBEnv(encoder, discriminator)
    generator = TRPO(
        policy = "MlpPolicy",
        env = env,
        learning_rate = GENERATOR_LEARNING_RATE,
        batch_size = 2048, # Paper suggests 10,000
        gamma = 0.99,
        gae_lambda = 0.95, # Stabalize advantage estimation
        target_kl = 0.01, # TRPO trust region
        verbose = True,
    )
    encoder = Encoder()
    discriminator = Discriminator()

    # Instantiate optimizers
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=ENCODER_LEARNING_RATE)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=ENCODER_LEARNING_RATE)

    # Generate expert data
    exp_data = generate_expert_trajectories()

    for _ in range(NUM_EPISODES):

        # TRPO update a.k.a. updating the Generator Policy
        generator.learn(total_timesteps=ROLLOUT_STEPS)

        # Generate generator data
        gen_data = generate_generator_trajectories(generator, env)

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

        for traj in gen_data:
            
            for step in traj:

                state, action, next_state = step
                x, y = state
                x_n, y_n = next_state

                z, mu, logvar = encoder(x, y, x_n, y_n)

                z_list.append(z)
                labels.append(torch.zeros(z.size(0), 1)) # generator = 0

                kl_list.append(kl_divergence(mu, logvar))

        z_batch = torch.cat(z_list, dim=0)
        label_batch = torch.cat(labels, dim=0)
        kl_batch = torch.cat(kl_list, dim=0)

        # ENCODER -> DISCRIMINATOR
        
        kl = kl_batch.mean()
        
        logits = discriminator(z_batch)

        discriminator_loss = F.binary_cross_entropy_with_logits(logits, label_batch)

        encoder_loss = discriminator_loss + beta * (kl - I_c)

        # Update DISCRIMINATOR
        disc_opt.zero_grad()
        discriminator_loss.backward()
        disc_opt.step()

        # Update ENCODER
        enc_opt.zero_grad()
        encoder_loss.backward()
        enc_opt.step()

        # Update lagrange multiplier (beta)
        update_beta(beta, kl)

        exit()

if __name__ == "__main__":
    main()


