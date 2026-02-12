"""
One file to represent the following:
    - Architecture
    - Training
    - Metrics & Visualization
"""

from pathlib import Path

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sb3_contrib import TRPO

from metrics import MetricsLogger
from display import plot_expert_vs_generator


# Environment
GRID_MIN = -10
GRID_MAX = 10

RANDOM_SPAWN_MIN_X = -3
RANDOM_SPAWN_MIN_Y = -3

RANDOM_SPAWN_MAX_X = -2
RANDOM_SPAWN_MAX_Y = -2

GOAL = np.array([5, 5])

# WALL

WALL_X = 2.5
WALL_Y_MIN = -10
WALL_Y_MAX = 6.0

def hits_wall(state, next_state):

    if (state[0] < WALL_X <= next_state[0]) or (next_state[0] < WALL_X <= state[0]):
    
        t = (WALL_X - state[0]) / (next_state[0] - state[0] + 1e-8)
        y_cross = state[1] + t * (next_state[1] - state[1])

        if WALL_Y_MIN <= y_cross <= WALL_Y_MAX:
            return True
    
    return False


class VDBEnv(gym.Env):
    """
    Gym environment necessary to conduct TRPO to optimize the generator policy.
    """

    def __init__(self, discriminator, enable_wall=False):
        super().__init__()

        self.discriminator = discriminator
        self.enable_wall = enable_wall

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.state = None
        self.t = 0

    def reset(self):
        self.state = np.array(
            [
                np.random.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
                np.random.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y),
            ]
        )
        self.t = 0
        return self.state

    def step(self, action):
        self.t += 1
        x, y = self.state
        dx, dy = action
        next_state = np.array([x + dx, y + dy])

        # VDB REWARD
        if self.enable_wall and hits_wall(self.state, next_state):
            
            next_state = self.state.copy()
            reward = -1.0
        
        else:
            with torch.no_grad():
                reward, _ = self.discriminator(
                    torch.tensor(self.state, dtype=torch.float32).unsqueeze(0), 
                    torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                )

        self.state = next_state

        dist = np.linalg.norm(next_state - GOAL)
        done = dist < 0.5 or self.t >= MAX_STEP_COUNT

        if done and dist <= 0.5:
            reward += 1.0

        return next_state, reward, done, {}


# Movement
MAX_STEP_COUNT = 40
MAX_STEP_SIZE = 0.5


def step(state: np.array, action: np.array) -> np.array:
    return np.clip(state + action, GRID_MIN, GRID_MAX)


# DATA
def generate_s_shaped_expert_trajectories(
    n_steps=40,
    amplitude=1.0,
    frequency=2.0,
    n_trajs = 30,
):
    trajectories = []

    for _ in range(n_trajs):
        
        state = np.array(
            [
                np.random.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
                np.random.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y),
            ]
        )

        traj = []

        direction = GOAL - state
        direction /= np.linalg.norm(direction) + 1e-8

        perp = np.array([-direction[1], direction[0]])

        for t in range(n_steps):
            alpha = t / n_steps

            # S-shape oscillation
            offset = amplitude * np.sin(2 * np.pi * frequency * alpha)
            move_dir = direction + offset * perp
            move_dir /= np.linalg.norm(move_dir) + 1e-8

            action = MAX_STEP_SIZE * move_dir
            next_state = step(state, action)

            traj.append((state.copy(), action.copy(), next_state.copy()))
            state = next_state

        trajectories.append(traj)

    return trajectories


def generate_curved_expert_trajectories(
    n_steps=40,
    curvature=10,
    n_trajs=30
):
    trajectories = []

    for _ in range(n_trajs):
        
        state = np.array(
            [
                np.random.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
                np.random.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y),
            ]
        )

        traj = []


        for t in range(n_steps):
            alpha = t / n_steps

            direction = GOAL - state
            perp = np.array([-direction[1], direction[0]])
            perp /= np.linalg.norm(perp) + 1e-8
        
            # Curved direction
            curve_offset = curvature * np.sin(np.pi * alpha)
            move_dir = direction + curve_offset * perp
            move_dir /= np.linalg.norm(move_dir) + 1e-8

            action = MAX_STEP_SIZE * move_dir
            next_state = step(state, action)

            traj.append((state.copy(), action.copy(), next_state.copy()))
            state = next_state

        trajectories.append(traj)

    return trajectories


def generate_straight_expert_trajectories(n_trajs: int = 30):
    """
    Generate an array of trajectories that serve as expert demonstrations.
    """
    trajectories = []

    for _ in range(n_trajs):
        state = np.array(
            [
                np.random.uniform(RANDOM_SPAWN_MIN_X, RANDOM_SPAWN_MAX_X),
                np.random.uniform(RANDOM_SPAWN_MIN_Y, RANDOM_SPAWN_MAX_Y),
            ]
        )
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
INPUT_SIZE = 2  # Represents (x, y) position
OUTPUT_SIZE = 2  # Represents (Δx, Δy) change in pose
HIDDEN_SIZE = 32  # Number of hidden nodes in neural net

def generate_generator_trajectories(generator: TRPO, env: VDBEnv, n_trajs: int = 20) -> list:
    """
    Generate trajectories through the generator.
    """
    trajectories = []
    rewards = []

    for _ in range(n_trajs):
        state = env.reset()

        traj = []
        reward_accumulated = 0

        for _ in range(MAX_STEP_COUNT):
            action, _ = generator.predict(state, deterministic=False)
            next_state, reward, done, _ = env.step(action)

            reward_accumulated += reward

            traj.append((state.copy(), action.copy(), next_state.copy()))

            state = next_state

            if done:
                state = env.reset()

        rewards.append(reward_accumulated)
        trajectories.append(traj)

    print(f"Reward mean  : {np.mean(rewards):.3f}")
    print(f"Reward std   : {np.std(rewards):.3f}")

    return trajectories, float(np.mean(rewards)), float(np.std(rewards))


# ENCODER & VARIATIONAL DATA BOTTLENECK
E_INPUT_SIZE = 2  # Represents an input of (x, y)
E_HIDDEN_SIZE = 128
Z_SIZE = 6  # The higher the z size the more complex the task

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
            nn.ReLU(),
        )

        self.mu = nn.Linear(E_HIDDEN_SIZE, Z_SIZE)

        self.logvar = nn.Linear(E_HIDDEN_SIZE, Z_SIZE)

    def forward(self, state):
        
        h = self.net(state)

        mu = self.mu(h)
        logvar = self.logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar

BETA = torch.tensor(0.0, requires_grad=True)  # Lagrange multiplier
I_C = 0.5  # Upper bound on the mutual information between the encoding and the original features I(X, Z)
BETA_STEP_SIZE = 1e-3  # Step size for updating the dual variable BETA


def kl_divergence(mu, logvar):
    """
    Calculate the KL divergence.
    """
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1) 
    return kl


def update_beta(beta, kl):
    """
    Update the lagrange multiplier beta.
    """
    with torch.no_grad():
        beta += BETA_STEP_SIZE * (kl.detach().detach() - I_C)
        beta.clamp_(min=0)


D_HIDDEN_SIZE = 128

class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.net(z)

class HNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.net(z)

GAMMA = 0.99

class VAIRLDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_g = Encoder()
        self.enc_h = Encoder()
        self.g = GNet()
        self.h = HNet()
        self.gamma = GAMMA

    def forward(self, s, s_next):
        z_g, mu_g, logvar_g = self.enc_g(s)
        z_h, mu_h, logvar_h = self.enc_h(s)
        z_hn, mu_hn, logvar_hn = self.enc_h(s_next)

        f = self.g(z_g) + self.gamma * self.h(z_hn) - self.h(z_h)

        return f, (mu_g, logvar_g, mu_h, logvar_h, mu_hn, logvar_hn)


def total_kl(stats):
    mu_g, logvar_g, mu_h, logvar_h, mu_hn, logvar_hn = stats
    return (
        kl_divergence(mu_g, logvar_g)
        + kl_divergence(mu_h, logvar_h)
        + kl_divergence(mu_hn, logvar_hn)
    ).mean()


def train_vairl_discriminator(
    discriminator,
    generator,
    disc_opt,
    expert_batch,
    gen_batch,
    beta,
    alpha_beta,
):
    s_e = torch.tensor(expert_batch[:, 0], dtype=torch.float32)
    a_e = torch.tensor(expert_batch[:, 1], dtype=torch.float32)
    s_next_e = torch.tensor(expert_batch[:, 2], dtype=torch.float32)
    
    s_g = torch.tensor(gen_batch[:, 0], dtype=torch.float32)
    a_g = torch.tensor(gen_batch[:, 1], dtype=torch.float32)
    s_next_g = torch.tensor(gen_batch[:, 2], dtype=torch.float32)

    f_e, stats_e = discriminator(s_e, s_next_e)
    f_g, stats_g = discriminator(s_g, s_next_g)

    # AIRL-style loss
    _estimated_value, log_likelihood, _entropy = generator.policy.evaluate_actions(s_g, a_g)

    log_likelihood = log_likelihood.unsqueeze(1)

    loss_disc = (
        -torch.mean(f_e)
        + torch.mean(torch.logsumexp(torch.stack([f_g, log_likelihood]), dim=0))
    )

    kl = 0.5 * (total_kl(stats_e) + total_kl(stats_g))
    bottleneck = kl - I_C

    loss = loss_disc + beta * bottleneck

    disc_opt.zero_grad()
    loss.backward()
    disc_opt.step()

    # Dual ascent on beta
    beta = max(0.0, beta + alpha_beta * bottleneck.item())

    return beta, loss.item(), kl.item()


NUM_EPISODES = 500
ROLLOUT_STEPS = 2048

TARGET_DISCRIMINATOR_ACC = 0.8 # Until the discriminator hits this accuracy for expert and generator then we can stop training it.

GENERATOR_LEARNING_RATE = 1e-4
DISCRIMINATOR_LEARNING_RATE = 1e-4

# How often to save metrics snapshots
SAVE_EVERY = 10


def main():
    """
    Training loop.
    """
    train_discriminator_flag = True

    path_type = input("\nChoose expert path type:\n- s: S-Curve\n- c: C-Curve\n- else: Straight\n\nChoice: ")
    enable_wall = path_type == "c"

    out_dir = Path("metrics_out")
    metrics = MetricsLogger()

    beta = BETA

    discriminator = VAIRLDiscriminator()

    env = VDBEnv(discriminator, enable_wall)
    
    generator = TRPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=GENERATOR_LEARNING_RATE,
        batch_size=2048,  # Paper suggests 10,000
        gamma=0.99,
        gae_lambda=0.95,  # Stabilize advantage estimation
        target_kl=0.01,  # TRPO trust region
        verbose=True,
    )

    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LEARNING_RATE)

    if path_type == "s":
        exp_data = generate_s_shaped_expert_trajectories()
    elif path_type == "c":
        exp_data = generate_curved_expert_trajectories()
    else:
        exp_data = generate_straight_expert_trajectories()

    for episode in range(NUM_EPISODES):
        
        print(f"EPISODE {episode}")
        
        # EVALUATE

        discriminator.eval()

        gen_data, reward_mean, reward_std = generate_generator_trajectories(generator, env)

        exp_batch = np.array([
            [s, a, s_n] 
            for traj in exp_data
            for s, a, s_n in traj
        ])

        gen_batch = np.array([
            [s, a, s_n] 
            for traj in gen_data
            for s, a, s_n in traj
        ])

        # TRAIN
        
        generator.learn(total_timesteps=ROLLOUT_STEPS)

        logger = generator.logger.name_to_value

        # Keep exact naming / print format
        for k in [
            "train/kl_divergence_loss",
            "train/is_line_search_success",
            "train/std",
            "train/policy_objective",
            "train/explained_variance",
        ]:
            print(k, logger.get(k))

        
        if train_discriminator_flag:
           beta, loss, kl = train_vairl_discriminator(
                discriminator,
                generator,
                disc_opt,
                exp_batch,
                gen_batch,
                beta,
                BETA_STEP_SIZE
            ) 

        print(f"D loss       : {loss:.4f}")
        print(f"kl           : {kl:.3f}")
        print(f"beta         : {beta:.3f}")

        # ---- Metrics capture (no output changes) ----
        metrics.log("train/kl_divergence_loss", logger.get("train/kl_divergence_loss"))
        metrics.log("train/is_line_search_success", logger.get("train/is_line_search_success"))
        metrics.log("train/std", logger.get("train/std"))
        metrics.log("train/policy_objective", logger.get("train/policy_objective"))
        metrics.log("train/explained_variance", logger.get("train/explained_variance"))

        metrics.log("Reward mean", reward_mean)
        metrics.log("Reward std", reward_std)

        metrics.log("D loss", loss)
        metrics.log("KL(z|x)", kl)
        metrics.log("KL - I_c", (kl - I_C))
        metrics.log("beta", float(beta))

        if (episode + 1) % SAVE_EVERY == 0:
            plot_expert_vs_generator(exp_data, gen_data, GOAL, [WALL_X, WALL_Y_MIN, WALL_Y_MAX] if enable_wall else None)
            metrics.save_all(out_dir, show=False)

    metrics.save_all(out_dir, show=True)


if __name__ == "__main__":
    main()

