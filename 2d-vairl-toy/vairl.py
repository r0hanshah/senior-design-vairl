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

    def __init__(self, encoder, discriminator, enable_wall=False):
        super().__init__()

        self.encoder = encoder
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
                z, _mu, _logvar = self.encoder(
                    torch.tensor(x),
                    torch.tensor(y),
                    torch.tensor(next_state[0]),
                    torch.tensor(next_state[1]),
                )

                logits = self.discriminator(z)

                reward = torch.log(torch.sigmoid(logits) + 1e-8)
                reward = torch.clamp(reward, -10.0, 10.0).item()

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


class Generator(nn.Module):
    """
    Class that represents the agent that predicts action based on current state.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, state: np.array):
        return self.net(state)


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
E_INPUT_SIZE = 4  # Represents an input of (x, y, x', y')
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

    def forward(self, x: float, y: float, x_next: float, y_next: float):
        h = torch.stack((x, y, x_next, y_next)).unsqueeze(0).to(torch.float32)
        h = self.net(h)

        mu = self.mu(h)
        logvar = self.logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


BETA = torch.tensor(1.0, requires_grad=True)  # Lagrange multiplier
I_C = 0.5  # Upper bound on the mutual information between the encoding and the original features I(X, Z)
BETA_STEP_SIZE = 1e-3  # Step size for updating the dual variable BETA


def kl_divergence(mu, logvar):
    """
    Calculate the KL divergence.
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl


def update_beta(beta, kl):
    """
    Update the lagrange multiplier beta.
    """
    with torch.no_grad():
        beta += BETA_STEP_SIZE * (kl.detach().detach() - I_C)
        beta.clamp_(min=0)


D_HIDDEN_SIZE = 128


class Discriminator(nn.Module):
    """
    Class that attempts to differentiate vectors from the latent space.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(Z_SIZE, D_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(D_HIDDEN_SIZE, D_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(D_HIDDEN_SIZE, D_HIDDEN_SIZE//2),
            nn.ReLU(),
            nn.Linear(D_HIDDEN_SIZE//2, 1),  # logit
        )

    def forward(self, z):
        return self.net(z)


NUM_EPISODES = 200
ROLLOUT_STEPS = 2048
N = 5 # Train discriminator every N episodes

GENERATOR_LEARNING_RATE = 1e-3
DISCRIMINATOR_LEARNING_RATE = 1e-6
ENCODER_LEARNING_RATE = 1e-2

# How often to save metrics snapshots
SAVE_EVERY = 10


def main():
    """
    Training loop.
    """

    path_type = input("\nChoose expert path type:\n- s: S-Curve\n- c: C-Curve\n- else: Straight\n\nChoice: ")
    enable_wall = path_type == "c"

    out_dir = Path("metrics_out")
    metrics = MetricsLogger()

    beta = BETA

    encoder = Encoder()
    discriminator = Discriminator()

    env = VDBEnv(encoder, discriminator, enable_wall)
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

    enc_opt = torch.optim.Adam(encoder.parameters(), lr=ENCODER_LEARNING_RATE)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=ENCODER_LEARNING_RATE)

    if path_type == "s":
        exp_data = generate_s_shaped_expert_trajectories()
    elif path_type == "c":
        exp_data = generate_curved_expert_trajectories()
    else:
        exp_data = generate_straight_expert_trajectories()

    for episode in range(NUM_EPISODES):
        print(f"EPISODE {episode}")
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

        gen_data, reward_mean, reward_std = generate_generator_trajectories(generator, env)

        z_list = []
        labels = []
        kl_list = []

        for traj in exp_data:
            for step_item in traj:
                state, action, next_state = step_item
                x, y = state
                x_n, y_n = next_state

                z, mu, logvar = encoder(
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32),
                    torch.tensor(x_n, dtype=torch.float32),
                    torch.tensor(y_n, dtype=torch.float32),
                )

                z_list.append(z)
                labels.append(torch.ones(z.size(0), 1))
                kl_list.append(kl_divergence(mu, logvar))

        for traj in gen_data:
            for step_item in traj:
                state, action, next_state = step_item
                x, y = state
                x_n, y_n = next_state

                z, mu, logvar = encoder(
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32),
                    torch.tensor(x_n, dtype=torch.float32),
                    torch.tensor(y_n, dtype=torch.float32),
                )

                z_list.append(z)
                labels.append(torch.zeros(z.size(0), 1))
                kl_list.append(kl_divergence(mu, logvar))

        z_batch = torch.cat(z_list, dim=0)
        label_batch = torch.cat(labels, dim=0)
        kl_batch = torch.cat(kl_list, dim=0)

        z_detached = z_batch.detach()
        logits = discriminator(z_detached)

        discriminator_loss = F.binary_cross_entropy_with_logits(logits, label_batch)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds == label_batch).float().mean().item()
            expert_acc = (probs[label_batch == 1] > 0.5).float().mean()
            gen_acc = (probs[label_batch == 0] < 0.5).float().mean()

        print(f"D loss       : {discriminator_loss.item():.4f}")
        print(f"D acc expert : {expert_acc.item():.3f}")
        print(f"D acc        : {acc:.3f}")
        print(f"D acc gen    : {gen_acc.item():.3f}")

        print("Expert mean prob:", probs[label_batch == 1].mean().item())
        print("Gen mean prob:", probs[label_batch == 0].mean().item())

        # DISCRIMINATOR UPDATE
        disc_opt.zero_grad()
        discriminator_loss.backward()
        disc_opt.step()

        # ENCODER UPDATE
        for p in discriminator.parameters():
            p.requires_grad = False

        logits2 = discriminator(z_batch)

        kl = kl_batch.mean()

        encoder_loss = F.binary_cross_entropy_with_logits(logits2, label_batch) + beta * (kl - I_C)

        enc_opt.zero_grad()
        encoder_loss.backward()
        enc_opt.step()

        print(f"z mean       : {z_batch.mean().item():.3f}")
        print(f"z std        : {z_batch.std().item():.3f}")

        for p in discriminator.parameters():
            p.requires_grad = True

        update_beta(beta, kl)

        print(f"KL(z|x)      : {kl.item():.4f}")
        print(f"I_c          : {I_C:.4f}")
        print(f"KL - I_c     : {(kl - I_C).item():.4f}")
        print(f"beta         : {beta:.4f}")

        # ---- Metrics capture (no output changes) ----
        metrics.log("train/kl_divergence_loss", logger.get("train/kl_divergence_loss"))
        metrics.log("train/is_line_search_success", logger.get("train/is_line_search_success"))
        metrics.log("train/std", logger.get("train/std"))
        metrics.log("train/policy_objective", logger.get("train/policy_objective"))
        metrics.log("train/explained_variance", logger.get("train/explained_variance"))

        metrics.log("Reward mean", reward_mean)
        metrics.log("Reward std", reward_std)

        metrics.log("D loss", discriminator_loss.item())
        metrics.log("D acc expert", expert_acc.item())
        metrics.log("D acc", acc)
        metrics.log("D acc gen", gen_acc.item())
        metrics.log("Expert mean prob", probs[label_batch == 1].mean().item())
        metrics.log("Gen mean prob", probs[label_batch == 0].mean().item())

        metrics.log("z mean", z_batch.mean().item())
        metrics.log("z std", z_batch.std().item())
        metrics.log("KL(z|x)", kl.item())
        metrics.log("KL - I_c", (kl - I_C).item())
        metrics.log("beta", float(beta))

        if (episode + 1) % SAVE_EVERY == 0:
            plot_expert_vs_generator(exp_data, gen_data, GOAL, [WALL_X, WALL_Y_MIN, WALL_Y_MAX])
            metrics.save_all(out_dir, show=False)

    metrics.save_all(out_dir, show=True)


if __name__ == "__main__":
    main()

