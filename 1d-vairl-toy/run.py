import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------------
# Environment
# -----------------------------
STATE_MIN = -1.0
STATE_MAX = 1.0
GOAL = 1.0
T = 30
GAMMA = 0.99


def step(s, a):
    s_next = np.clip(s + a, STATE_MIN, STATE_MAX)
    return s_next


# -----------------------------
# Expert demonstrations
# -----------------------------
def generate_expert_trajectories(n_trajs=20):
    trajectories = []
    for _ in range(n_trajs):
        s = np.random.uniform(-1.0, -0.5)
        traj = []
        for _ in range(T):
            a = 0.1
            s_next = step(s, a)
            traj.append((s, a, s_next))
            s = s_next
        trajectories.append(traj)
    return trajectories


# -----------------------------
# Policy (Generator)
# -----------------------------
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, s):
        return 0.1 * self.net(s)


def rollout_policy(policy, n_trajs=20):
    trajectories = []
    for _ in range(n_trajs):
        s = np.random.uniform(-1.0, -0.5)
        traj = []
        for _ in range(T):
            s_t = torch.tensor([[s]], dtype=torch.float32)
            a = policy(s_t).item()
            s_next = step(s, a)
            traj.append((s, a, s_next))
            s = s_next
        trajectories.append(traj)
    return trajectories


# -----------------------------
# AIRL-style Reward Model
# r(s,a,s') = f(s,a) + γ h(s') − h(s)
# -----------------------------
class AIRLReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.h = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s, a, s_next):
        sa = torch.cat([s, a], dim=-1)
        return self.f(sa) + GAMMA * self.h(s_next) - self.h(s)


# -----------------------------
# Losses
# -----------------------------
def reward_loss(reward_model, expert_trajs, policy_trajs):
    exp = torch.tensor(
        [(s, a, sp) for traj in expert_trajs for s, a, sp in traj],
        dtype=torch.float32
    )
    pol = torch.tensor(
        [(s, a, sp) for traj in policy_trajs for s, a, sp in traj],
        dtype=torch.float32
    )

    r_exp = reward_model(exp[:, :1], exp[:, 1:2], exp[:, 2:])
    r_pol = reward_model(pol[:, :1], pol[:, 1:2], pol[:, 2:])

    # AIRL-style discriminator objective
    return -(r_exp.mean() - r_pol.mean())


def policy_loss(policy, reward_model, policy_trajs, entropy_coef=0.01):
    data = torch.tensor(
        [(s, a, sp) for traj in policy_trajs for s, a, sp in traj],
        dtype=torch.float32
    )

    s = data[:, :1]
    a = policy(s)
    sp = torch.clamp(s + a, STATE_MIN, STATE_MAX)

    r = reward_model(s, a, sp)

    # simple entropy proxy
    entropy = -(a ** 2).mean()

    return -(r.mean() + entropy_coef * entropy)


# -----------------------------
# Visualization
# -----------------------------
def plot_trajectories(policy, expert_trajs, iteration):
    plt.figure(figsize=(7, 5))

    for traj in expert_trajs:
        states = [s for s, _, _ in traj]
        plt.plot(states, color="green", alpha=0.3)

    policy_trajs = rollout_policy(policy, n_trajs=10)
    for traj in policy_trajs:
        states = [s for s, _, _ in traj]
        plt.plot(states, color="red", alpha=0.6)

    plt.axhline(GOAL, linestyle="--")
    plt.title(f"Iteration {iteration} | Green=Expert Red=Policy")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.ylim(-1.1, 1.1)
    plt.show()


def plot_reward(reward_model):
    s = torch.linspace(-1, 1, 200).unsqueeze(1)
    a = torch.ones_like(s) * 0.05
    sp = torch.clamp(s + a, STATE_MIN, STATE_MAX)

    r = reward_model(s, a, sp).detach().numpy()
    plt.plot(s.numpy(), r)
    plt.title("Learned AIRL Reward")
    plt.xlabel("State")
    plt.ylabel("Reward")
    plt.show()


# -----------------------------
# Training Loop
# -----------------------------
def train():
    policy = Policy()
    reward_model = AIRLReward()

    policy_opt = optim.Adam(policy.parameters(), lr=3e-4)
    reward_opt = optim.Adam(reward_model.parameters(), lr=1e-4)

    expert_trajs = generate_expert_trajectories()

    for it in range(401):
        policy_trajs = rollout_policy(policy)

        # Train reward (slower)
        reward_opt.zero_grad()
        r_loss = reward_loss(reward_model, expert_trajs, policy_trajs)
        r_loss.backward()
        reward_opt.step()

        # Train policy
        policy_opt.zero_grad()
        p_loss = policy_loss(policy, reward_model, policy_trajs)
        p_loss.backward()
        policy_opt.step()

        if it % 20 == 0:
            print(
                f"Iter {it:03d} | Reward loss {r_loss.item():+.3f} | "
                f"Policy loss {p_loss.item():+.3f}"
            )
            plot_trajectories(policy, expert_trajs, it)

    plot_reward(reward_model)


if __name__ == "__main__":
    train()

